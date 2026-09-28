"""Overnight pipeline for the MADQN baseline (resumable).

  Stage 1  tune    learning rate in {1e-4, 3e-4, 1e-3} (seed 7, 800 iterations,
                   CHO validation seeds 20-22, 20 episodes each, common reward)
  Stage 2  train   MADQN with the selected rate, seeds 7-11, 2000 iterations
  Stage 3  eval    nominal + compound stress, same protocol as M1-M5
  Stage 4  report  outputs/overnight/madqn_report.md

Usage (project root):
  Git Bash:  PYTHONPATH=src python run_madqn_overnight.py
  macOS:     PYTHONPATH=src .venv/bin/python run_madqn_overnight.py
  report only:  ... run_madqn_overnight.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import run_overnight as ro   # run(), welch_p(), ci(), load()

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "overnight"
SEEDS = [7, 8, 9, 10, 11]
METRICS = [("outage_ratio", "Outage"), ("service_deficit", "Deficit"),
           ("handover_started", "HO exec./epoch"), ("pingpong_rate", "Ping-pong"),
           ("handover_failure", "Failed HOs/epoch"), ("resource_cost", "Resource"),
           ("common_reward", "Common reward"), ("preparation_requested", "Prep. requests/epoch")]


def report() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = ro.load()
    lines = ["# MADQN results\n"]
    tf = ROOT / "outputs" / "madqn_tuning" / "tuning.json"
    if tf.is_file():
        t = json.loads(tf.read_text(encoding="utf-8"))
        lines.append(f"Tuning (seed {t['tuning_seed']}, {t['iterations']} iterations, validation seeds "
                     f"{t['validation_seeds']}): best lr = {t['best_lr']:g}\n")
        lines.append("| lr | common reward | outage |\n|---|---|---|")
        for r in t["results"]:
            lines.append(f"| {r['lr']:g} | {r['common_reward']:.5f} | {r['outage_ratio']:.4f} |")
    for cond in ("nominal", "stress"):
        lines.append(f"\n## {cond} (mean ± 95% CI across training seeds; Welch p-values)\n")
        lines.append("| metric | CHO | MADQN | M1 (MAPPO) | M5 | p(MADQN vs M1) | p(MADQN vs M5) |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, name in METRICS:
            cells, v = [], {}
            for mdl in ("CHO-A3-TTT", "MADQN", "M1", "M5"):
                per = ro.per_seed(recs, mdl, cond, key)
                v[mdl] = list(per.values())
                if not per:
                    cells.append("-")
                    continue
                if mdl == "CHO-A3-TTT":   # rule: values per evaluation seed
                    evs = recs[(mdl, cond)][0]
                    vals = [e[key] for e in evs.values()]
                    m, h = ro.ci(vals)
                else:
                    m, h = ro.ci(v[mdl])
                cells.append(f"{m:.4f} ± {h:.4f}")
            p1 = ro.welch_p(v["MADQN"], v["M1"]) if v["MADQN"] else float("nan")
            p5 = ro.welch_p(v["MADQN"], v["M5"]) if v["MADQN"] else float("nan")
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {p1:.3f} | {p5:.3f} |")
    # training summary: common reward over the last 100 iterations
    lines.append("\n## Training: common reward over the last 100 iterations\n")
    import csv
    for s in SEEDS:
        f = ROOT / "outputs" / "v3_anneal_2000_multiseed" / f"madqn_seed{s}" / "training_metrics.csv"
        if f.is_file():
            rows = list(csv.DictReader(f.open(encoding="utf-8")))[-100:]
            cr = mean(-float(r["service_deficit"]) - 0.25 * float(r["resource_cost"]) for r in rows)
            lines.append(f"- seed {s}: {cr:.4f}")
    (OUT / "madqn_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT / 'madqn_report.md'}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--skip-tune", action="store_true", help="use lr=3e-4 without tuning")
    args = ap.parse_args()
    if args.report:
        report()
        return
    py, seeds, t0 = sys.executable, [str(s) for s in SEEDS], time.monotonic()
    if not args.skip_tune:
        ro.run([py, "-u", "train_madqn.py", "tune", "--jobs", str(args.jobs)], "madqn_1_tune.log")
    train_cmd = [py, "-u", "train_madqn.py", "train", "--seeds", *seeds, "--jobs", str(args.jobs)]
    if args.skip_tune:
        train_cmd += ["--lr", "3e-4"]
    ro.run(train_cmd, "madqn_2_train.log")
    ro.run([py, "-u", "eval_unified.py", "--models", "MADQN", "--train-seeds", *seeds,
            "--episodes", "40", "--jobs", str(args.jobs)], "madqn_3_eval.log")
    report()
    print(f"\nALL DONE in {(time.monotonic() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
