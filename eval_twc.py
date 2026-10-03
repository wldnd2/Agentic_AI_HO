"""Evaluate trained runs and rule baselines on common TWC episodes.

Examples
  # rule baselines on the validation seeds (nominal)
  PYTHONPATH=src python eval_twc.py --methods a3 t1 d2 maxsinr --seeds 20 21 22 --episodes 4
  # trained runs (directories written by train_twc.py), both conditions
  PYTHONPATH=src python eval_twc.py --runs outputs/twc/proposed_seed7 outputs/twc/madqn_seed7 \
      --methods a3 t1 --seeds 20 21 22 --episodes 4 --conditions nominal stress

Rows are written to outputs/twc/eval/<tag>.jsonl and a summary table is printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from cho_baselines import A3ConditionalHandover, LocationBasedCHO, TimeBasedCHO  # noqa: E402
from leo_mappo import LEOMobilityEnv, load_config  # noqa: E402
from leo_mappo.baselines import max_sinr_policy  # noqa: E402
from leo_mappo.egcmadqn import EGCExecutor, EGCMADQN  # noqa: E402
from train_twc import CFG, TWC_STACK  # noqa: E402

# Rule baselines tuned on validation seeds 20-22 (outputs/twc/cho_sweep_*.csv).
RULES = {
    "a3": lambda: A3ConditionalHandover(prep_offset_db=1.0, exec_offset_db=1.0, ttt_steps=1),
    "t1": lambda: TimeBasedCHO(service_elevation_deg=75.0, lead_s=15.0, target_rule="geometry"),
    "d2": lambda: LocationBasedCHO(thresh1_km=400.0, thresh2_km=1500.0, target_rule="load"),
    "maxsinr": lambda: max_sinr_policy,
}
# Compound-stress disturbances are active in epochs [60, 130) (twc_compound_eval.yaml).
STRESS_WINDOW = (60, 130)
WINDOW_KEYS = ("outage_ratio", "service_deficit", "signaling_cost", "handover_started")
SUM_KEYS = (
    "outage_ratio", "service_deficit", "signaling_cost", "resource_cost", "handover_started",
    "handover_failure", "handover_success", "pingpong", "deadline_misses", "arrivals",
    "duplication_active_time_fraction", "preparation_requested", "preparation_rejected",
)


def build_env(condition: str, seed: int) -> LEOMobilityEnv:
    stack = [p for p in TWC_STACK if p.exists()]
    if condition == "stress":
        stack.append(CFG / "twc_compound_eval.yaml")
    cfg = load_config(CFG / "default.yaml", stack)
    cfg["environment_seed"] = seed
    return LEOMobilityEnv(cfg)


def run_job(job: tuple) -> dict:
    method, run_dir, condition, seed, episodes = job
    env = build_env(condition, seed)
    steps = int(env.env_cfg["episode_steps"])
    executor = None
    if run_dir is not None:
        run_config = json.loads((Path(run_dir) / "run_config.json").read_text())
        agent = EGCMADQN.from_checkpoint(Path(run_dir) / "egcmadqn_weights.npz", seed=seed)
        executor = EGCExecutor(env, agent, run_config["components"]["price_input"], np.random.default_rng(seed))
    else:
        policy = RULES[method]()
    sums = {k: 0.0 for k in SUM_KEYS}
    window = {k: 0.0 for k in WINDOW_KEYS}
    window_steps = 0
    reward = 0.0
    for episode in range(episodes):
        obs, _, masks = env.reset(seed=10_000 * seed + episode)
        if executor is not None and executor.coordinator is not None:
            executor.coordinator.reset()
        for _ in range(steps):
            actions = executor.act(obs, masks) if executor is not None else policy(env)
            out = env.step(actions)
            if executor is not None:
                executor.observe(out)
            obs, masks = out.observations, out.action_masks
            reward += float(np.mean(out.rewards))
            for k in SUM_KEYS:
                sums[k] += float(out.info.get(k, 0.0))
            if STRESS_WINDOW[0] <= env.step_index - 1 < STRESS_WINDOW[1]:
                window_steps += 1
                for k in WINDOW_KEYS:
                    window[k] += float(out.info.get(k, 0.0))
    n = episodes * steps
    return {
        "method": method, "run": None if run_dir is None else str(run_dir), "condition": condition, "seed": seed,
        "common_reward": reward / n, "outage": sums["outage_ratio"] / n, "deficit": sums["service_deficit"] / n,
        "signaling": sums["signaling_cost"] / n, "resource": sums["resource_cost"] / n,
        "ho_per_epoch": sums["handover_started"] / n, "ho_failure_per_epoch": sums["handover_failure"] / n,
        "pingpong_ratio": sums["pingpong"] / max(sums["handover_success"], 1.0),
        "deadline_violation": sums["deadline_misses"] / max(sums["arrivals"], 1.0),
        "duplication_fraction": sums["duplication_active_time_fraction"] / n,
        "prep_rejection": sums["preparation_rejected"] / max(sums["preparation_requested"], 1.0),
        # Same epochs in both conditions, so nominal vs stress is a paired comparison.
        "window_outage": window["outage_ratio"] / max(window_steps, 1),
        "window_deficit": window["service_deficit"] / max(window_steps, 1),
        "window_signaling": window["signaling_cost"] / max(window_steps, 1),
        "window_ho_per_epoch": window["handover_started"] / max(window_steps, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=[], choices=sorted(RULES))
    parser.add_argument("--runs", nargs="*", default=[], help="train_twc.py output directories")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20, 21, 22])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--conditions", nargs="+", default=["nominal"], choices=["nominal", "stress"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default="eval")
    args = parser.parse_args()

    jobs = [(m, None, c, s, args.episodes) for m in args.methods for c in args.conditions for s in args.seeds]
    jobs += [(Path(r).name, r, c, s, args.episodes) for r in args.runs for c in args.conditions for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    out = ROOT / "outputs" / "twc" / "eval" / f"{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    keys = ("common_reward", "outage", "deficit", "signaling", "ho_per_epoch", "pingpong_ratio",
            "prep_rejection", "window_outage")
    print(f"{'condition':9s} {'method':24s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
    for condition in args.conditions:
        for name in dict.fromkeys(r["method"] for r in rows):
            sel = [r for r in rows if r["method"] == name and r["condition"] == condition]
            print(f"{condition:9s} {name:24s} " + " ".join(f"{np.mean([r[k] for r in sel]):12.4f}" for k in keys))
    print(f"rows -> {out}")


if __name__ == "__main__":
    main()
