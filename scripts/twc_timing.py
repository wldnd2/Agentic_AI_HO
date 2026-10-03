"""Measure wall time per training iteration in the TWC environment.

Usage: PYTHONPATH=src python scripts/twc_timing.py [--iterations 8]
Times MADQN (incl. replay updates after warm-up) and MAPPO M1 / M4-style
(planner + price) for a few iterations and extrapolates to 2000 iterations.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from leo_mappo import LEOMobilityEnv, MAPPO, load_config  # noqa: E402
from leo_mappo.madqn import MADQN  # noqa: E402
from leo_mappo.runner import RolloutRunner  # noqa: E402
from train_madqn import DEFAULT_CFG  # noqa: E402

CFG = ROOT / "configs"
TWC = [
    CFG / "validation_agentic_challenge_v3.yaml", CFG / "eta_handover_005.yaml", CFG / "validation_dwell_15.yaml",
    CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set1.yaml", CFG / "twc_load_24ue.yaml", CFG / "twc_events.yaml",
]
ROLLOUT = 128


def time_madqn(iterations: int) -> float:
    cfg = load_config(CFG / "default.yaml", TWC)
    env = LEOMobilityEnv(cfg)
    agent = MADQN(env.obs_dim, env.action_dim, {**DEFAULT_CFG, "learning_rate": 1e-3}, 7)
    obs, _, masks = env.reset()
    times = []
    for it in range(iterations):
        t0 = time.perf_counter()
        eps, _ = agent.set_iteration(it, 2000)
        for _ in range(ROLLOUT):
            actions = agent.act(obs, masks, eps)
            step = env.step(actions)
            agent.store(obs, actions, float(step.rewards.mean()), step.observations, step.action_masks, step.done)
            obs, masks = step.observations, step.action_masks
            if step.done:
                obs, _, masks = env.reset()
        agent.update()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times[-max(1, iterations // 2):]))  # after replay warm-up


def time_mappo(iterations: int, extra: list[Path]) -> float:
    cfg = load_config(CFG / "default.yaml", [*TWC, *extra])
    env = LEOMobilityEnv(cfg)
    agent = MAPPO(env.obs_dim, env.global_state_dim, env.action_dim, cfg["mappo"], 7,
                  num_satellites=env.num_cells, candidate_count=env.candidate_count,
                  eta_handover=float(env.reward_cfg["eta_handover"]))
    runner = RolloutRunner(env, agent)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        batch, _ = runner.collect(ROLLOUT)
        agent.update(batch)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    results = {
        "MADQN": time_madqn(args.iterations),
        "MAPPO (M1)": time_mappo(args.iterations, []),
        "MAPPO + planner + price (M4)": time_mappo(args.iterations, [CFG / "m4_coordination.yaml"]),
    }
    for name, sec in results.items():
        print(f"{name:30s} {sec:6.2f} s/iteration  ->  2000 iterations ~ {sec * 2000 / 3600:5.1f} h")


if __name__ == "__main__":
    main()
