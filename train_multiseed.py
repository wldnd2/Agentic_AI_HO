"""Parallel multi-seed training for the final V3 M1-M5 checkpoints.

Reproduces the configuration of outputs/v3_convergence_anneal_2000/*_seed7
(V3 challenge + eta_handover 0.05 + dwell 15 + linear LR annealing, 2000 iters)
for additional training seeds. Each run is an independent process pinned to one
BLAS thread so that N parallel runs use N cores without oversubscription.

Usage (Git Bash / Windows):
    PYTHONPATH=src .venv/Scripts/python.exe train_multiseed.py --seeds 8 9 --jobs 8
    PYTHONPATH=src .venv/Scripts/python.exe train_multiseed.py --dry-run

Finished runs (mappo_weights.npz present) are skipped, so re-running resumes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs"
BASE = CFG / "default.yaml"
COMMON = [
    CFG / "validation_agentic_challenge_v3.yaml",
    CFG / "eta_handover_005.yaml",
    CFG / "validation_dwell_15.yaml",
    CFG / "lr_anneal_linear.yaml",
]
MODEL_OVERRIDES = {
    "M1": [],
    "M2": [CFG / "memory_only.yaml"],
    "M3": [CFG / "memory_reasoning.yaml"],
    "M4": [CFG / "m4_coordination.yaml"],
    "M5": [CFG / "full_agentic.yaml", CFG / "dual_mid_0005.yaml"],
}
# Heavier models first so the pool packs well.
LAUNCH_ORDER = ["M5", "M4", "M3", "M2", "M1"]
OUTPUT_ROOT = ROOT / "outputs" / "v3_anneal_2000_multiseed"
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def run_dir(model: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"{model.lower()}_seed{seed}"


def build_command(model: str, seed: int, iterations: int) -> list[str]:
    cmd = [sys.executable, "-u", "main.py", "train", "--config", str(BASE)]
    for path in [*COMMON, *MODEL_OVERRIDES[model]]:
        cmd += ["--override", str(path)]
    cmd += [
        "--environment-seed", str(seed),
        "--agent-seed", str(seed),
        "--evaluation-action-seed", str(7000 + seed),
        "--iterations", str(iterations),
        # Real evaluation is done afterwards by eval_unified.py.
        "--eval-episodes", "1",
        "--stochastic-eval-episodes", "1",
        "--output", str(run_dir(model, seed)),
    ]
    return cmd


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    for var in THREAD_VARS:
        env[var] = "1"
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["M1", "M2", "M4", "M5"],
                        choices=list(MODEL_OVERRIDES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[8, 9])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = [str(p) for p in [BASE, *COMMON] + sum(MODEL_OVERRIDES.values(), [])
               if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing config files:\n" + "\n".join(missing))

    ordered = [m for m in LAUNCH_ORDER if m in args.models]
    queue: list[tuple[str, int]] = []
    for model in ordered:
        for seed in args.seeds:
            if (run_dir(model, seed) / "mappo_weights.npz").is_file():
                print(f"[skip] {model} seed={seed}: weights already exist")
            else:
                queue.append((model, seed))

    print(f"cpu_count={os.cpu_count()} jobs={args.jobs} runs={len(queue)} "
          f"iterations={args.iterations}")
    for model, seed in queue:
        print(f"  {model} seed={seed} -> {run_dir(model, seed)}")
    if args.dry_run or not queue:
        if queue:
            print("\nExample command:\n  " + " ".join(build_command(*queue[0], args.iterations)))
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = child_env()
    running: dict[tuple[str, int], tuple[subprocess.Popen, float, object]] = {}
    failed: list[tuple[str, int, int]] = []
    start = time.monotonic()

    while queue or running:
        while queue and len(running) < args.jobs:
            model, seed = queue.pop(0)
            out = run_dir(model, seed)
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / "train.log", "w", encoding="utf-8")
            proc = subprocess.Popen(build_command(model, seed, args.iterations),
                                    cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            running[(model, seed)] = (proc, time.monotonic(), log)
            print(f"[start] {model} seed={seed} pid={proc.pid}", flush=True)

        time.sleep(30)
        for key, (proc, t0, log) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[key]
            ok = code == 0 and (run_dir(*key) / "mappo_weights.npz").is_file()
            status = "done" if ok else f"FAILED(code={code})"
            if not ok:
                failed.append((*key, code))
            print(f"[{status}] {key[0]} seed={key[1]} in {fmt(time.monotonic() - t0)} "
                  f"| running={len(running)} queued={len(queue)} "
                  f"| total {fmt(time.monotonic() - start)}", flush=True)

    print(f"\nAll runs finished in {fmt(time.monotonic() - start)}.")
    if failed:
        print("Failed runs (see train.log in each directory):")
        for model, seed, code in failed:
            print(f"  {model} seed={seed} code={code}")
        sys.exit(1)


if __name__ == "__main__":
    main()
