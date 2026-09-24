"""One-command pipeline for the JSAC multi-seed experiments.

    PYTHONPATH=src .venv/Scripts/python.exe run_all.py --cores 8

--cores = number of PHYSICAL CPU cores. Steps:
  1+2 (in parallel)
      1. train M1/M2/M4/M5 with training seeds 8, 9   (train_multiseed.py)
      2. tune CHO baselines on validation seeds        (cho_sweep.py)
         -> skipped if cho_params.json already exists (parameters stay frozen)
  3. main evaluation: M1/M2/M4/M5 x train seeds 7,8,9 + heuristics + CHO
  4. diagnostic: M3 vs M4 (seed 7 only, separate output folder)

Every step is resumable: if the PC restarts or you stop it, run the same command
again and finished work is skipped. Logs: outputs/pipeline_logs/.
On Windows the script also keeps the PC from going to sleep while it runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "outputs" / "pipeline_logs"
MAIN_MODELS = ["M1", "M2", "M4", "M5"]
NEW_SEEDS = [8, 9]


def keep_awake(enable: bool) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def launch(name: str, args: list[str], echo: bool):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    log = open(log_path, "a", encoding="utf-8")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
    proc = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, env=child_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)

    def pump() -> None:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if echo:
                print(f"[{name}] {line}", end="", flush=True)
        log.close()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return proc, thread, log_path


def wait(name: str, proc, thread, log_path) -> int:
    code = proc.wait()
    thread.join()
    status = "OK" if code == 0 else f"FAILED (exit {code}) - see {log_path}"
    print(f"\n>>> {name}: {status}\n", flush=True)
    return code


def banner(text: str) -> None:
    print("\n" + "=" * 72 + f"\n{time.strftime('%H:%M:%S')}  {text}\n" + "=" * 72, flush=True)


def missing_checkpoints() -> list[str]:
    missing = []
    for model in MAIN_MODELS:
        for seed in NEW_SEEDS:
            path = ROOT / "outputs" / "v3_anneal_2000_multiseed" / f"{model.lower()}_seed{seed}" / "mappo_weights.npz"
            if not path.is_file():
                missing.append(f"{model} seed{seed}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, required=True, help="number of PHYSICAL cores")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--skip-m3", action="store_true", help="skip the M3-vs-M4 diagnostic")
    args = parser.parse_args()

    cores = max(1, args.cores)
    sweep_jobs = 2 if cores >= 6 else 1
    train_jobs = max(1, min(8, cores - sweep_jobs))
    eval_jobs = max(1, cores - 1)
    need_sweep = not (ROOT / "cho_params.json").is_file()
    t0 = time.monotonic()
    keep_awake(True)
    try:
        banner(f"STEP 1+2: training (jobs={train_jobs})"
               + (f" + CHO sweep (jobs={sweep_jobs})" if need_sweep else " (CHO params already frozen)"))
        train = launch("1_train", ["train_multiseed.py", "--models", *MAIN_MODELS,
                                   "--seeds", *map(str, NEW_SEEDS), "--jobs", str(train_jobs)],
                       echo=True)
        sweep = launch("2_cho_sweep", ["cho_sweep.py", "--jobs", str(sweep_jobs)],
                       echo=False) if need_sweep else None
        if sweep:
            print("[2_cho_sweep] running in background -> outputs/pipeline_logs/2_cho_sweep.log",
                  flush=True)
            sweep_code = wait("2_cho_sweep", *sweep)
        else:
            sweep_code = 0
        train_code = wait("1_train", *train)

        missing = missing_checkpoints()
        if train_code != 0 or missing or sweep_code != 0:
            print("Stopping before evaluation.")
            if missing:
                print("Missing checkpoints: " + ", ".join(missing))
                print("Check outputs/v3_anneal_2000_multiseed/<run>/train.log, then rerun "
                      "the same command (finished runs are skipped).")
            if sweep_code != 0:
                print("CHO sweep failed: see outputs/pipeline_logs/2_cho_sweep.log")
            sys.exit(1)

        banner(f"STEP 3: main evaluation (jobs={eval_jobs}, episodes={args.episodes})")
        code = wait("3_eval_main", *launch("3_eval_main", [
            "eval_unified.py", "--models", *MAIN_MODELS, "--train-seeds", "7", "8", "9",
            "--heuristics", "--episodes", str(args.episodes), "--jobs", str(eval_jobs),
            "--out", "outputs/unified_eval"], echo=True))
        if code != 0:
            sys.exit(code)

        if not args.skip_m3:
            banner("STEP 4: diagnostic M3 vs M4 (seed 7)")
            wait("4_eval_m3_vs_m4", *launch("4_eval_m3_vs_m4", [
                "eval_unified.py", "--models", "M3", "M4", "--train-seeds", "7",
                "--episodes", str(args.episodes), "--jobs", str(min(eval_jobs, 10)),
                "--out", "outputs/diag_m3_vs_m4"], echo=True))

        hours = (time.monotonic() - t0) / 3600
        banner(f"ALL DONE in {hours:.1f} h")
        print("Main table      : outputs/unified_eval/summary_table.md")
        print("Main long CSV   : outputs/unified_eval/summary_long.csv")
        print("M3 vs M4 table  : outputs/diag_m3_vs_m4/summary_table.md")
        print("CHO parameters  : cho_params.json  (sweep: outputs/cho_sweep/cho_sweep_results.csv)")
        print("Logs            : outputs/pipeline_logs/")
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
