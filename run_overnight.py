"""Overnight experiments for the reviewer-oriented revision (resumable).

  Stage 0  patch   allow use_constraint without coordination for the M1D ablation
  Stage 1  train   M1D = M1 + reliability Lagrangian, training seeds 7-11, 2000 iterations
  Stage 2  eval    M1D, nominal + compound stress (same protocol as the paper)
  Stage 3  eval    within-policy reasoning ablation for M3/M4/M5 (reasoning_zero)
  Stage 4  report  outputs/overnight/overnight_report.md (+ JSON with all numbers)

Usage (project root, macOS/Linux):
    PYTHONPATH=src .venv/bin/python run_overnight.py            # everything
    PYTHONPATH=src .venv/bin/python run_overnight.py --report   # only rebuild the report
Windows (Git Bash):
    PYTHONPATH=src .venv/Scripts/python.exe run_overnight.py

Every stage skips work that is already finished, so the script can simply be
restarted after an interruption.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "outputs" / "pipeline_logs"
OUT = ROOT / "outputs" / "overnight"
RAW = ROOT / "outputs" / "unified_eval" / "raw"
SEEDS = [7, 8, 9, 10, 11]
W, W_PREP = 0.25, 0.2          # reward weights of the paper (w, w_prep)


# ----------------------------------------------------------------- stage 0 ---
def patch_mappo() -> None:
    path = ROOT / "src" / "leo_mappo" / "mappo.py"
    src = path.read_text(encoding="utf-8")
    old = ("        if self.use_constraint and not self.use_coordination:\n"
           "            raise ValueError(\"use_constraint=true requires coordination\")\n")
    new = ("        # The M1D ablation (plain MAPPO + reliability Lagrangian) is allowed;\n"
           "        # recurrent models still need coordination before the constraint.\n"
           "        if self.use_constraint and self.use_memory and not self.use_coordination:\n"
           "            raise ValueError(\"use_constraint=true requires coordination\")\n")
    if new in src:
        print("[patch] mappo.py already patched")
        return
    if old not in src:
        raise RuntimeError("mappo.py: expected constraint check not found; patch manually")
    backup = path.with_suffix(".py.bak_overnight")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8")
    path.write_text(src.replace(old, new), encoding="utf-8")
    print(f"[patch] mappo.py patched (backup: {backup.name})")


# --------------------------------------------------------------- stages 1-3 ---
def run(cmd: list[str], log_name: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"  # Windows consoles: avoid cp949 decode errors
    log = LOGS / log_name
    print(f"\n>>> {' '.join(cmd)}\n    log: {log}", flush=True)
    t0 = time.monotonic()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.ctime()} : {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:  # type: ignore[union-attr]
            fh.write(line)
            fh.flush()
            if any(k in line for k in ("[done]", "[skip]", "FAILED", "jobs total", "min) |", "Error")):
                print("   " + line.rstrip(), flush=True)
        rc = proc.wait()
    print(f"<<< exit {rc} after {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    if rc != 0:
        raise SystemExit(f"stage failed, see {log}")


# ------------------------------------------------------------------ stats ---
def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def welch_p(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx, vy = stdev(x) ** 2 / len(x), stdev(y) ** 2 / len(y)
    if vx + vy == 0:
        return float("nan")
    t = (mean(x) - mean(y)) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return _betai(df / 2, 0.5, df / (df + t * t))  # two-sided


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def ci(v: list[float]) -> tuple[float, float]:
    m = mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan"))


# ----------------------------------------------------------------- report ---
METRICS = [("outage_ratio", "Outage"), ("service_deficit", "Deficit"),
           ("handover_started", "HO exec./epoch"), ("pingpong_rate", "Ping-pong"),
           ("resource_cost", "Resource"), ("common_reward", "Common reward"),
           ("preparation_requested", "Prep. requests/epoch")]


def load():
    recs = defaultdict(lambda: defaultdict(dict))  # (label, cond) -> ts -> es -> metrics
    for f in RAW.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        m = r["metrics"]
        m["common_reward"] = -m["service_deficit"] - W * m["resource_cost"]
        ab = r.get("ablation", "none")
        label = r["model"] if ab == "none" else f"{r['model']}[{ab}]"
        recs[(label, r["condition"])][r["train_seed"]][r["eval_seed"]] = m
    return recs


def per_seed(recs, label, cond, key):
    return {ts: mean(e[key] for e in evs.values()) for ts, evs in recs.get((label, cond), {}).items()}


def report() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load()
    lines, data = ["# Overnight results\n"], {}

    # (1) M1D vs M1 vs M5 -----------------------------------------------------
    lines.append("## 1. Is goal regulation alone enough? (M1D = M1 + reliability Lagrangian)\n")
    for cond in ("nominal", "stress"):
        lines.append(f"\n### {cond}  (mean ± 95% CI across training seeds; Welch p-values)\n")
        lines.append("| metric | M1 | M1D | M5 | p(M1D vs M1) | p(M1D vs M5) |")
        lines.append("|---|---|---|---|---|---|")
        for key, name in METRICS:
            v = {mdl: per_seed(recs, mdl, cond, key) for mdl in ("M1", "M1D", "M5")}
            if not v["M1D"]:
                lines.append(f"| {name} | (M1D not evaluated yet) | | | | |")
                continue
            cells = []
            for mdl in ("M1", "M1D", "M5"):
                m, h = ci(list(v[mdl].values()))
                cells.append(f"{m:.4f} ± {h:.4f}")
            p1 = welch_p(list(v["M1D"].values()), list(v["M1"].values()))
            p5 = welch_p(list(v["M1D"].values()), list(v["M5"].values()))
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {p1:.3f} | {p5:.3f} |")
            data[f"m1d_{cond}_{key}"] = {mdl: v[mdl] for mdl in v}

    # (2) reasoning ablation --------------------------------------------------
    lines.append("\n## 2. Within-policy reasoning ablation (reasoning feature made uninformative)\n")
    lines.append("Paired change = ablated minus normal, per training seed (mean over eval seeds); "
                 "relative change in %; number of policies with an increase.\n")
    for mdl in ("M3", "M4", "M5"):
        for cond in ("nominal", "stress"):
            a = recs.get((f"{mdl}[reasoning_zero]", cond))
            b = recs.get((mdl, cond))
            if not a or not b:
                continue
            lines.append(f"\n### {mdl}, {cond}\n")
            lines.append("| metric | Δ mean ± CI | rel. change | increases | per seed |")
            lines.append("|---|---|---|---|---|")
            for key, name in METRICS:
                diffs, rels = [], []
                for ts in sorted(set(a) & set(b)):
                    es = sorted(set(a[ts]) & set(b[ts]))
                    if not es:
                        continue
                    na = mean(a[ts][e][key] for e in es)
                    nb = mean(b[ts][e][key] for e in es)
                    diffs.append(na - nb)
                    rels.append(100 * (na - nb) / abs(nb) if nb else float("nan"))
                if not diffs:
                    continue
                m, h = ci(diffs)
                inc = sum(d > 0 for d in diffs)
                lines.append(f"| {name} | {m:+.4f} ± {h:.4f} | {mean(rels):+.1f}% | {inc}/{len(diffs)} | "
                             + ", ".join(f"{d:+.4f}" for d in diffs) + " |")
                data[f"reason_{mdl}_{cond}_{key}"] = diffs

    # (3) reward-weight sensitivity -----------------------------------------------
    lines.append("\n## 3. Common reward vs. handover weight w_HO (post hoc, nominal)\n")
    lines.append("C_ctrl = C_dup + w_prep*C_prep + w_HO*C_HO recomputed from logged components.\n")
    lines.append("| w_HO | " + " | ".join(m for m in ("M1", "M1D", "M5") if recs.get((m, "nominal"))) + " |")
    models = [m for m in ("M1", "M1D", "M5") if recs.get((m, "nominal"))]
    lines.append("|---|" + "---|" * len(models))
    for w_ho in (0.05, 0.5, 1, 2, 4, 8):
        cells = []
        for mdl in models:
            vals = []
            for evs in recs[(mdl, "nominal")].values():
                vals.append(mean(-e["service_deficit"] - W * (e["duplication_train_cost"]
                            + W_PREP * e["preparation_train_cost"] + w_ho * e["handover_train_cost"])
                            for e in evs.values()))
            cells.append(f"{mean(vals):.4f}")
        lines.append(f"| {w_ho} | " + " | ".join(cells) + " |")

    (OUT / "overnight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "overnight_numbers.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT / 'overnight_report.md'}")


# ------------------------------------------------------------------- main ---
def main() -> None:
    try:  # Git Bash / cmd on Windows may default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--report", action="store_true", help="only rebuild the report")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-reasoning", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    py = sys.executable
    seeds = [str(s) for s in SEEDS]
    t0 = time.monotonic()
    patch_mappo()
    if not args.skip_train:
        run([py, "-u", "train_multiseed.py", "--models", "M1D", "--seeds", *seeds,
             "--iterations", str(args.iterations), "--jobs", str(min(args.jobs, len(SEEDS)))],
            "overnight_1_train_m1d.log")
        run([py, "-u", "eval_unified.py", "--models", "M1D", "--train-seeds", *seeds,
             "--episodes", str(args.episodes), "--jobs", str(args.jobs)],
            "overnight_2_eval_m1d.log")
    if not args.skip_reasoning:
        run([py, "-u", "eval_unified.py", "--models", "M3", "M4", "M5", "--train-seeds", *seeds,
             "--ablations", "reasoning_zero", "--episodes", str(args.episodes), "--jobs", str(args.jobs)],
            "overnight_3_eval_reasoning_zero.log")
    report()
    print(f"\nALL DONE in {(time.monotonic() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
