"""Tune the CHO-A3-TTT baselines on VALIDATION seeds, then freeze the choice.

Validation episodes are disjoint from the held-out evaluation streams used by
eval_unified.py (evaluation seeds 7-11 -> reset seeds 10000..10999). Only the
nominal V3 condition is used for tuning; compound stress stays fully held out.

Stage 1: grid over A3 execution offset x TTT for CHO-A3-TTT.
Stage 2: with the chosen A3 parameters, grid over the duplication threshold for
         CHO-A3-TTT+Dup.
Selection metric: common reward = -service_deficit - lambda_resource * resource_cost
(the common reward used for the M1-M5 comparison).

Writes cho_params.json (read automatically by cho_baselines.py / eval_unified.py)
and outputs/cho_sweep/cho_sweep_results.csv.

Usage:
    PYTHONPATH=src .venv/Scripts/python.exe cho_sweep.py --jobs 2
"""

from __future__ import annotations

import eval_unified as eu  # sets BLAS thread limits before numpy is imported

import argparse
import csv
import itertools
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean

HELD_OUT_EVAL_SEEDS = set(range(7, 12))


def run_sweep_job(job: dict) -> dict:
    from leo_mappo import LEOMobilityEnv
    from leo_mappo.runner import summarize_metrics
    from cho_baselines import make_cho_policies

    cfg = eu.make_config([], "nominal", job["seed"], 7)
    lam = float(cfg["reward"]["lambda_resource"])
    env = LEOMobilityEnv(cfg)
    policy = make_cho_policies(job["params"])[job["variant"]]
    offset = (job["seed"] - 7) * eu.STREAM_LENGTH
    totals: dict[str, float] = defaultdict(float)
    steps = 0
    for episode in range(job["episodes"]):
        env.reset(seed=10_000 + episode + offset)
        done = False
        while not done:
            output = env.step(policy(env))
            for key, value in output.info.items():
                totals[key] += float(value)
            done = output.done
            steps += 1
    m = summarize_metrics(totals, steps)
    m = {k: float(v) for k, v in m.items()}
    m["common_reward"] = -m["service_deficit"] - lam * m["resource_cost"]
    return {**job, "metrics": m}


def run_grid(variant, grid, seeds, episodes, jobs_n, rows):
    jobs = [{"variant": variant, "params": params, "seed": s, "episodes": episodes}
            for params in grid for s in seeds]
    results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=jobs_n) as pool:
        futures = [pool.submit(run_sweep_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            key = json.dumps(r["params"], sort_keys=True)
            results[key].append(r["metrics"])
            print(f"[{variant} {i}/{len(jobs)}] {r['params']} seed={r['seed']} "
                  f"common={r['metrics']['common_reward']:.4f}", flush=True)
    scored = []
    for key, ms in results.items():
        params = json.loads(key)
        row = {"variant": variant, **params,
               **{f"{k}": mean(m[k] for m in ms) for k in
                  ("common_reward", "outage_ratio", "service_deficit", "resource_cost",
                   "handover_started", "pingpong_rate", "duplication_active_time_fraction")}}
        rows.append(row)
        scored.append((row["common_reward"], params))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[20, 21, 22])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--exec-offsets", nargs="+", type=float, default=[2.0, 3.0, 4.0])
    parser.add_argument("--ttts", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dup-thresholds", nargs="+", type=float, default=[2.0, 4.0, 6.0])
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    overlap = HELD_OUT_EVAL_SEEDS & set(args.seeds)
    if overlap:
        raise ValueError(f"Validation seeds overlap held-out evaluation seeds: {sorted(overlap)}")

    from cho_baselines import DEFAULT_PARAMS, PARAMS_FILE
    base = dict(DEFAULT_PARAMS)
    rows: list[dict] = []
    t0 = time.monotonic()

    grid1 = [{**base, "exec_offset_db": o, "ttt_steps": t}
             for o, t in itertools.product(args.exec_offsets, args.ttts)]
    best_a3 = run_grid("CHO-A3-TTT", grid1, args.seeds, args.episodes, args.jobs, rows)
    print(f"\nStage 1 best: exec_offset_db={best_a3['exec_offset_db']} ttt_steps={best_a3['ttt_steps']}")

    grid2 = [{**best_a3, "dup_threshold_db": d} for d in args.dup_thresholds]
    best_dup = run_grid("CHO-A3-TTT+Dup", grid2, args.seeds, args.episodes, args.jobs, rows)
    print(f"Stage 2 best: dup_threshold_db={best_dup['dup_threshold_db']}")

    chosen = {**best_a3, "dup_threshold_db": best_dup["dup_threshold_db"],
              "_tuning": {"validation_seeds": args.seeds, "episodes_per_seed": args.episodes,
                          "condition": "nominal", "metric": "common_reward"}}
    PARAMS_FILE.write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    out = eu.ROOT / "outputs" / "cho_sweep"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cho_sweep_results.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k))
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["variant"], -r["common_reward"])))
    print(f"\nFrozen parameters -> {PARAMS_FILE}")
    print(json.dumps(chosen, indent=2))
    print(f"Sweep table -> {out / 'cho_sweep_results.csv'} ({(time.monotonic() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
