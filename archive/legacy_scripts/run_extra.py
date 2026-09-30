"""Final round of experiments (one command). After this, no more experiments.

    PYTHONPATH=src .venv/Scripts/python.exe run_extra.py --cores 12

--cores = number of PHYSICAL CPU cores. Steps (each resumable):
  1. latency_bench.py            decision latency M1-M5 (run first, on an idle machine)
  2. price ablation, seeds 7-9   M4/M5 with coordination price forced to 0
                                 -> first answer (about 1-2 h): does the price matter
                                    in every trained policy, and inside M5?
  3. training                    M1/M2/M4/M5 seeds 10, 11  +  M3 seeds 8-11
  4. main evaluation             M1-M5 x seeds 7-11 + heuristics + CHO (new jobs only)
  5. price ablation, seeds 10-11 M4/M5
All results go to outputs/unified_eval (summary_table.md is rebuilt after each step).
Logs: outputs/pipeline_logs/extra_*.log
"""

from __future__ import annotations

import argparse
import sys
import time

from run_all import banner, keep_awake, launch, wait

ALL = ["M1", "M2", "M3", "M4", "M5"]
SEEDS = ["7", "8", "9", "10", "11"]


def step(name: str, args: list[str]) -> None:
    code = wait(name, *launch(name, args, echo=True))
    if code != 0:
        print(f"Stopped at {name}. Fix the problem, then rerun the same command "
              "(finished work is skipped).")
        sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, required=True, help="number of PHYSICAL cores")
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()
    jobs = str(max(1, args.cores - 1))
    ep = str(args.episodes)
    t0 = time.monotonic()
    keep_awake(True)
    try:
        banner("STEP 1: decision latency (idle machine)")
        step("extra_1_latency", ["latency_bench.py", "--episodes", "3"])

        banner("STEP 2: price ablation on existing seeds 7-9 (M4, M5)")
        step("extra_2_ablation_789", [
            "eval_unified.py", "--models", "M4", "M5", "--train-seeds", "7", "8", "9",
            "--ablations", "price_zero", "--episodes", ep, "--jobs", jobs,
            "--out", "outputs/unified_eval"])

        banner("STEP 3: training M1/M2/M4/M5 seeds 10-11 and M3 seeds 8-11")
        step("extra_3_train", [
            "train_multiseed.py", "--models", *ALL, "--seeds", "8", "9", "10", "11",
            "--jobs", str(min(12, max(1, args.cores - 1)))])  # existing runs are skipped

        banner("STEP 4: main evaluation, seeds 7-11 (only new jobs run)")
        step("extra_4_eval_main", [
            "eval_unified.py", "--models", *ALL, "--train-seeds", *SEEDS, "--heuristics",
            "--episodes", ep, "--jobs", jobs, "--out", "outputs/unified_eval"])

        banner("STEP 5: price ablation on seeds 10-11 (M4, M5)")
        step("extra_5_ablation_1011", [
            "eval_unified.py", "--models", "M4", "M5", "--train-seeds", *SEEDS,
            "--ablations", "price_zero", "--episodes", ep, "--jobs", jobs,
            "--out", "outputs/unified_eval"])

        banner(f"ALL DONE in {(time.monotonic() - t0) / 3600:.1f} h")
        print("Main + ablation tables : outputs/unified_eval/summary_table.md")
        print("Latency                : outputs/latency/latency.csv")
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
