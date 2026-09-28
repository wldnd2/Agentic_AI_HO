"""Deterministic (argmax) re-evaluation of the MAPPO-based policies.

MADQN is evaluated greedily, whereas M1-M5 are evaluated with sampled actions.
This script re-evaluates M1, M1D, and M5 with argmax actions on the same
episodes, so that the learned baselines can also be compared under identical
action-selection rules.

Usage (project root):
  Git Bash:  PYTHONPATH=src python run_greedy_eval.py
  macOS:     PYTHONPATH=src .venv/bin/python run_greedy_eval.py
  report only: ... run_greedy_eval.py --report
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import run_overnight as ro   # run(), welch_p(), load(), per_seed()

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "overnight"
SEEDS = [7, 8, 9, 10, 11]
METRICS = [("outage_ratio", "Outage"), ("service_deficit", "Deficit"),
           ("handover_started", "HO exec./epoch"), ("pingpong_rate", "Ping-pong"),
           ("handover_failure", "Failed HOs/epoch"), ("resource_cost", "Resource"),
           ("common_reward", "Common reward"), ("preparation_rejection_rate", "Prep. rejection")]


def fmt(v):
    return f"{mean(v):.4f} ± {stdev(v):.4f}" if len(v) > 1 else "-"


def report() -> None:
    recs = ro.load()
    lines = ["# Sampled vs. argmax evaluation (mean ± std across training seeds)\n"]
    for cond in ("nominal", "stress"):
        lines.append(f"\n## {cond}\n")
        lines.append("| metric | M1 sampled | M1 argmax | M1D sampled | M1D argmax | M5 sampled | M5 argmax "
                     "| MADQN (greedy) | p(M5 argmax vs MADQN) | p(M1 argmax vs MADQN) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, name in METRICS:
            v = {lab: list(ro.per_seed(recs, lab, cond, key).values())
                 for lab in ("M1", "M1[greedy]", "M1D", "M1D[greedy]", "M5", "M5[greedy]", "MADQN")}
            p5 = ro.welch_p(v["M5[greedy]"], v["MADQN"]) if v["M5[greedy]"] and v["MADQN"] else float("nan")
            p1 = ro.welch_p(v["M1[greedy]"], v["MADQN"]) if v["M1[greedy]"] and v["MADQN"] else float("nan")
            lines.append(f"| {name} | " + " | ".join(fmt(v[k]) for k in
                         ("M1", "M1[greedy]", "M1D", "M1D[greedy]", "M5", "M5[greedy]", "MADQN"))
                         + f" | {p5:.3f} | {p1:.3f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "greedy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT / 'greedy_report.md'}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        t0 = time.monotonic()
        ro.run([sys.executable, "-u", "eval_unified.py", "--models", "M5", "M1", "M1D",
                "--train-seeds", *[str(s) for s in SEEDS], "--ablations", "greedy",
                "--episodes", "40", "--jobs", str(args.jobs)], "greedy_eval.log")
        print(f"evaluation finished in {(time.monotonic() - t0) / 60:.1f} min")
    report()


if __name__ == "__main__":
    main()
