"""Unified evaluation: models x training seeds x evaluation seeds x conditions.

Same protocol as eval_v3_compound_generalization_multiseed.py (V3 + eta 0.05 +
dwell 15, stochastic actions, action seed 7000+e, nominal vs compound stress),
but:
  * several training seeds per model (seed 7 = existing anneal_2000 checkpoints),
  * fewer episodes per evaluation seed (default 40). Episode streams are the
    FIRST N episodes of the original 200-episode streams, so results remain a
    subset of the earlier evaluations,
  * one process per (model, train_seed, condition, eval_seed) job, pinned to one
    BLAS thread, run in parallel; each finished job is saved as JSON (resumable),
  * optional V3-fair heuristic baselines (observed SINR) from
    build_common_reward_and_baselines.py on the same episode streams.

Usage:
    PYTHONPATH=src .venv/Scripts/python.exe eval_unified.py --jobs 8
    PYTHONPATH=src .venv/Scripts/python.exe eval_unified.py --summarize-only

Summary statistic: for learned models, each training seed is first averaged over
evaluation seeds; mean/std/95% CI are then taken ACROSS TRAINING SEEDS (Student t,
df = n_train - 1). Heuristics have no training seed, so their CI is across
evaluation seeds.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import ast
import csv
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parent
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))  # cho_baselines.py lives next to this script
CFG = ROOT / "configs"
BASE = CFG / "default.yaml"
COMMON = [
    CFG / "validation_agentic_challenge_v3.yaml",
    CFG / "eta_handover_005.yaml",
    CFG / "validation_dwell_15.yaml",
]
STRESS = CFG / "compound_eval.yaml"
MODEL_OVERRIDES = {
    "M1": [],
    "M2": [CFG / "memory_only.yaml"],
    "M3": [CFG / "memory_reasoning.yaml"],
    "M4": [CFG / "m4_coordination.yaml"],
    "M5": [CFG / "full_agentic.yaml", CFG / "dual_mid_0005.yaml"],
    # Ablation: plain MAPPO (M1) trained with the M5 reliability Lagrangian only.
    "M1D": [CFG / "m1_dual.yaml", CFG / "dual_mid_0005.yaml"],
    # Value-based MARL baseline (parameter-shared Double DQN + fingerprints),
    # trained by train_madqn.py; evaluated greedily.
    "MADQN": [],
}
SEED7_DIRS = {
    "M1": "m1_seed7", "M2": "m2_seed7", "M3": "m3_seed7",
    "M4": "m4_seed7", "M5": "m5_dual_only_seed7",
}
HEURISTICS = {
    "Max-Observed-SINR": "max_observed_sinr_policy",
    "Always-Dup": "always_duplication_observed_policy",
    "Threshold-Dup": "threshold_duplication_observed_policy",
}
CHO_NAMES = ("CHO-A3-TTT", "CHO-A3-TTT+Dup")  # from cho_baselines.py
STREAM_LENGTH = 200  # original per-seed stream length; keeps episode offsets identical
KEY_METRICS = [
    "outage_ratio", "service_deficit", "resource_cost", "handover_started",
    "handover_failure", "pingpong_rate", "duplication_active_time_fraction",
    "deadline_violation_rate", "team_reward",
]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def weights_path(model: str, train_seed: int) -> Path:
    if model == "MADQN":
        return ROOT / "outputs" / "v3_anneal_2000_multiseed" / f"madqn_seed{train_seed}" / "madqn_weights.npz"
    if model not in SEED7_DIRS:  # new ablation models: every seed lives in the multiseed dir
        return ROOT / "outputs" / "v3_anneal_2000_multiseed" / f"{model.lower()}_seed{train_seed}" / "mappo_weights.npz"
    if train_seed == 7:
        return ROOT / "outputs" / "v3_convergence_anneal_2000" / SEED7_DIRS[model] / "mappo_weights.npz"
    return ROOT / "outputs" / "v3_anneal_2000_multiseed" / f"{model.lower()}_seed{train_seed}" / "mappo_weights.npz"


def make_config(model_overrides, condition: str, eval_seed: int, agent_seed: int,
                ablation: str = "none"):
    from leo_mappo import load_config
    paths = [*COMMON, *model_overrides]
    if condition == "stress":
        paths.append(STRESS)  # must stay last: replaces the event schedule only
    elif condition != "nominal":
        raise ValueError(condition)
    cfg = load_config(str(BASE), [str(p) for p in paths])
    cfg["environment_seed"] = int(eval_seed)
    cfg["agent_seed"] = int(agent_seed)
    if ablation == "price_zero":
        # Same switch as eval_m4_price_ablation.py: coordination price stays 0.
        cfg["coordination"]["price_learning_rate"] = 0.0
        cfg["coordination"]["initial_price"] = 0.0
    elif ablation not in ("none", "reasoning_zero", "greedy"):
        raise ValueError(f"unknown ablation {ablation}")
    return cfg


def _uninformative_reasoning(original_analyze):
    """Wrap LookaheadReasoner.analyze so that the actor receives an uninformative
    reasoning feature: 0 for every feasible action and 1 for masked actions. This is
    exactly the feature the actor sees during training when all plan costs are tied,
    so it removes the planner's ranking without creating an out-of-range input."""
    from dataclasses import replace

    def analyze(self, env, masks=None, *args, **kwargs):
        result = original_analyze(self, env, masks, *args, **kwargs)
        m = env.action_masks() if masks is None else np.asarray(masks, dtype=np.float64)
        return replace(result, features=np.where(m > 0.0, 0.0, 1.0).astype(np.float64))
    return analyze


def load_heuristics() -> dict:
    """Pull the V3-fair heuristic functions out of build_common_reward_and_baselines.py
    without executing that script's module-level experiment code."""
    from leo_mappo.ue import MobilityState
    candidates = [
        ROOT / "build_common_reward_and_baselines.py",
        ROOT / "archive" / "legacy_scripts" / "build_common_reward_and_baselines.py",
    ]
    source_path = next((p for p in candidates if p.is_file()), None)
    if source_path is None:
        raise FileNotFoundError(
            "build_common_reward_and_baselines.py not found in:\n  "
            + "\n  ".join(str(p) for p in candidates))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"mask_action", *HEURISTICS.values()}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    found = {n.name for n in nodes}
    if found != wanted:
        raise RuntimeError(f"Heuristic functions not found: {sorted(wanted - found)}")
    namespace = {"np": np, "MobilityState": MobilityState}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    policies = {name: namespace[fn] for name, fn in HEURISTICS.items()}
    from cho_baselines import make_cho_policies  # stateful TTT timers: fresh per job
    policies.update(make_cho_policies())
    return policies


def label_of(job: dict) -> str:
    ablation = job.get("ablation", "none")
    return job["model"] if ablation == "none" else f"{job['model']}[{ablation}]"


def job_file(out_dir: Path, job: dict) -> Path:
    return out_dir / "raw" / (
        f"{label_of(job)}_t{job['train_seed']}_{job['condition']}_e{job['eval_seed']}.json")


def run_job(job: dict) -> dict:
    from leo_mappo import LEOMobilityEnv, MAPPO
    from leo_mappo.runner import evaluate_policy, summarize_metrics

    t0 = time.monotonic()
    offset = (job["eval_seed"] - 7) * STREAM_LENGTH
    episodes = job["episodes"]

    if job["kind"] == "heuristic":
        cfg = make_config([], job["condition"], job["eval_seed"], 7)
        env = LEOMobilityEnv(cfg)
        policy = load_heuristics()[job["model"]]
        totals: dict[str, float] = defaultdict(float)
        steps = 0
        for episode in range(episodes):
            env.reset(seed=10_000 + episode + offset)
            done = False
            while not done:
                output = env.step(policy(env))
                for key, value in output.info.items():
                    totals[key] += float(value)
                done = output.done
                steps += 1
        result = summarize_metrics(totals, steps)
    elif job["model"] == "MADQN":
        from leo_mappo.madqn import MADQN
        train_seed = job["train_seed"]
        cfg = make_config([], job["condition"], job["eval_seed"], train_seed)
        env = LEOMobilityEnv(cfg)
        agent = MADQN.from_checkpoint(weights_path("MADQN", train_seed), train_seed)
        rng = np.random.default_rng(7000 + job["eval_seed"])
        totals: dict[str, float] = defaultdict(float)
        steps = 0
        for episode in range(episodes):
            obs, _, masks = env.reset(seed=10_000 + episode + offset)
            done = False
            while not done:
                output = env.step(agent.greedy(obs, masks, rng))
                for key, value in output.info.items():
                    totals[key] += float(value)
                obs, masks, done = output.observations, output.action_masks, output.done
                steps += 1
        result = summarize_metrics(totals, steps)
    else:
        model, train_seed = job["model"], job["train_seed"]
        cfg = make_config(MODEL_OVERRIDES[model], job["condition"], job["eval_seed"], train_seed,
                          job.get("ablation", "none"))
        env = LEOMobilityEnv(cfg)
        agent = MAPPO(env.obs_dim, env.global_state_dim, env.action_dim, cfg["mappo"],
                      train_seed, num_satellites=env.num_cells,
                      candidate_count=env.candidate_count,
                      eta_handover=float(env.reward_cfg["eta_handover"]))
        weights = weights_path(model, train_seed)
        with np.load(weights, allow_pickle=False) as data:
            for flag, expected in (("model_use_memory", agent.use_memory),
                                   ("model_use_reasoning", agent.use_reasoning),
                                   ("model_use_coordination", agent.use_coordination),
                                   ("model_use_constraint", agent.use_constraint)):
                if flag in data.files and bool(int(np.asarray(data[flag]).item())) != expected:
                    raise ValueError(f"{model} t{train_seed}: capability mismatch {flag}")
        agent.load(weights)

        original_reset = env.reset

        def seeded_reset(seed=None):
            return original_reset(seed=None if seed is None else int(seed) + offset)

        env.reset = seeded_reset  # type: ignore[method-assign]
        from leo_mappo.reasoning import LookaheadReasoner
        original_analyze = LookaheadReasoner.analyze
        if job.get("ablation", "none") == "reasoning_zero":
            if not agent.use_reasoning:
                raise ValueError(f"{model} has no reasoning module to ablate")
            LookaheadReasoner.analyze = _uninformative_reasoning(original_analyze)
        try:
            # "greedy": argmax actions instead of sampling (same protocol as MADQN)
            result = evaluate_policy(env, agent, episodes=episodes,
                                     deterministic=job.get("ablation", "none") == "greedy",
                                     evaluation_action_seed=7000 + job["eval_seed"])
        finally:
            LookaheadReasoner.analyze = original_analyze  # worker processes are reused

    if job["model"] in CHO_NAMES:
        from cho_baselines import load_cho_params
        job = {**job, "cho_params": load_cho_params()}
    record = {**job, "elapsed_s": time.monotonic() - t0,
              "metrics": {k: float(v) for k, v in result.items()}}
    return record


def t_ci(values: list[float]) -> tuple[float, float, float, float]:
    m = mean(values)
    if len(values) < 2:
        return m, float("nan"), float("nan"), float("nan")
    s = stdev(values)
    half = T95.get(len(values) - 1, 1.96) * s / math.sqrt(len(values))
    return m, s, m - half, m + half


def summarize(out_dir: Path) -> None:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_dir / "raw").glob("*.json"))]
    if not records:
        print("No results yet.")
        return
    metrics = sorted({k for r in records for k in r["metrics"]})
    groups: dict[tuple[str, str], dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    kinds = {}
    for r in records:
        groups[(label_of(r), r["condition"])][r["train_seed"]][r["eval_seed"]] = r["metrics"]
        kinds[label_of(r)] = r["kind"]

    rows = []
    for (model, condition), by_train in sorted(groups.items()):
        for metric in metrics:
            if kinds[model] == "heuristic":
                values = [m[metric] for m in by_train[0].values() if metric in m]
                unit = "eval_seed"
            else:
                values = []
                for evals in by_train.values():
                    per_eval = [ev[metric] for ev in evals.values() if metric in ev]
                    if per_eval:
                        values.append(mean(per_eval))
                unit = "train_seed"
            if not values:
                continue
            m, s, lo, hi = t_ci(values)
            rows.append({"model": model, "kind": kinds[model], "condition": condition,
                         "metric": metric, "n": len(values), "ci_unit": unit,
                         "mean": m, "std": s, "ci95_low": lo, "ci95_high": hi,
                         "n_train_seeds": len(by_train),
                         "n_eval_seeds": len({e for ev in by_train.values() for e in ev})})

    with (out_dir / "summary_long.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(r["model"], r["condition"], r["metric"]): r for r in rows}
    known = [*MODEL_OVERRIDES, *HEURISTICS, *CHO_NAMES]
    labels = {k[0] for k in groups}
    order = [m for m in known if m in labels] + sorted(labels - set(known))
    lines = []
    for condition in ("nominal", "stress"):
        if not any(k[1] == condition for k in groups):
            continue
        lines.append(f"\n### {condition}  (mean ± 95% CI half-width; n = seeds used for CI)\n")
        lines.append("| model | n | " + " | ".join(KEY_METRICS) + " |")
        lines.append("|---|---|" + "---|" * len(KEY_METRICS))
        for model in order:
            if (model, condition, KEY_METRICS[0]) not in lookup:
                continue
            cells = []
            for metric in KEY_METRICS:
                r = lookup.get((model, condition, metric))
                if r is None:
                    cells.append("-")
                elif math.isnan(r["ci95_high"]):
                    cells.append(f"{r['mean']:.4f}")
                else:
                    cells.append(f"{r['mean']:.4f} ± {(r['ci95_high'] - r['mean']):.4f}")
            n = lookup[(model, condition, KEY_METRICS[0])]["n"]
            lines.append(f"| {model} | {n} | " + " | ".join(cells) + " |")
    # Paired ablation effect: (ablated - normal) per training seed, averaged over
    # evaluation seeds, CI across training seeds.
    ablated = sorted(l for l in labels if "[" in l)
    for label in ablated:
        base = label.split("[")[0]
        for condition in ("nominal", "stress"):
            a, b = groups.get((label, condition)), groups.get((base, condition))
            if not a or not b:
                continue
            common_ts = sorted(set(a) & set(b))
            lines.append(f"\n### paired effect {label} minus {base}, {condition} "
                         f"(train seeds {common_ts}; mean ± 95% CI half-width; per-seed values)\n")
            lines.append("| metric | mean ± CI | per train seed |")
            lines.append("|---|---|---|")
            for metric in KEY_METRICS:
                diffs = []
                for ts in common_ts:
                    es = sorted(set(a[ts]) & set(b[ts]))
                    if not es or metric not in a[ts][es[0]]:
                        continue
                    diffs.append(mean(a[ts][e][metric] - b[ts][e][metric] for e in es))
                if not diffs:
                    continue
                m, _, lo, hi = t_ci(diffs)
                ci = "" if math.isnan(hi) else f" ± {hi - m:.4f}"
                lines.append(f"| {metric} | {m:+.4f}{ci} | " + ", ".join(f"{d:+.4f}" for d in diffs) + " |")
    (out_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'summary_long.csv'} and {out_dir / 'summary_table.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["M1", "M2", "M3", "M4", "M5"],
                        choices=list(MODEL_OVERRIDES))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[7, 8, 9, 10, 11])
    parser.add_argument("--conditions", nargs="+", default=["nominal", "stress"],
                        choices=["nominal", "stress"])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--heuristics", action="store_true",
                        help="also evaluate the V3-fair heuristic baselines")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default="outputs/unified_eval")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--ablations", nargs="+", default=["none"], choices=["none", "price_zero", "reasoning_zero", "greedy"],
                        help="evaluate each learned model normally and/or with the coordination price forced to 0")
    args = parser.parse_args()

    if args.episodes > STREAM_LENGTH:
        raise ValueError(f"--episodes must be <= {STREAM_LENGTH} to keep streams disjoint")
    out_dir = (ROOT / args.out).resolve()
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out_dir)
        return

    jobs, skipped_missing = [], []
    for model in args.models:
        for ts in args.train_seeds:
            if not weights_path(model, ts).is_file():
                skipped_missing.append(f"{model} t{ts}")
                continue
            for ablation in args.ablations:
                for cond in args.conditions:
                    for es in args.eval_seeds:
                        jobs.append({"kind": "model", "model": model, "train_seed": ts,
                                     "condition": cond, "eval_seed": es, "episodes": args.episodes,
                                     "ablation": ablation})
    if args.heuristics:
        load_heuristics()  # fail fast if extraction breaks
        for name in [*HEURISTICS, *CHO_NAMES]:
            for cond in args.conditions:
                for es in args.eval_seeds:
                    jobs.append({"kind": "heuristic", "model": name, "train_seed": 0,
                                 "condition": cond, "eval_seed": es, "episodes": args.episodes})

    from cho_baselines import load_cho_params
    current_cho = load_cho_params()
    if args.heuristics:
        print(f"CHO parameters in use: {current_cho}")
    todo = []
    for job in jobs:
        path = job_file(out_dir, job)
        if path.is_file():
            done = json.loads(path.read_text(encoding="utf-8"))
            stale_cho = job["model"] in CHO_NAMES and done.get("cho_params") != current_cho
            if done.get("episodes") == job["episodes"] and not stale_cho:
                continue
        todo.append(job)

    if skipped_missing:
        print("Missing checkpoints (skipped): " + ", ".join(skipped_missing))
    print(f"jobs total={len(jobs)} todo={len(todo)} parallel={args.jobs} episodes={args.episodes}")
    start = time.monotonic()
    # Heavier jobs first for better packing.
    weight = {"M5": 5, "M4": 4, "M3": 3, "M2": 2, "M1": 1, "M1D": 1, "MADQN": 1}
    todo.sort(key=lambda j: -weight.get(j["model"], 0))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_job, job): job for job in todo}
        for i, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # keep going; report at the end
                print(f"[FAILED] {job}: {exc!r}", flush=True)
                continue
            job_file(out_dir, job).write_text(json.dumps(record), encoding="utf-8")
            print(f"[{i}/{len(todo)}] {label_of(job)} t{job['train_seed']} {job['condition']} "
                  f"e{job['eval_seed']} ({record['elapsed_s'] / 60:.1f} min) "
                  f"| elapsed {(time.monotonic() - start) / 60:.1f} min", flush=True)
    summarize(out_dir)


if __name__ == "__main__":
    main()
