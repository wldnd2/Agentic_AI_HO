"""Per-step decision latency of M1-M5 (seed-7 checkpoints), single CPU thread.

Measures, per control epoch for all UEs of one satellite scenario:
  * actor   : MAPPO actor forward pass (act / act_recurrent), batched over UEs
  * reason  : LookaheadReasoner.analyze (ephemeris-based lookahead), if enabled
  * total   : actor + reason (the agent-side decision path; simulator excluded)

Usage:
    PYTHONPATH=src .venv/Scripts/python.exe latency_bench.py --episodes 3
Output: outputs/latency/latency.csv and a printed table.
"""

from __future__ import annotations

import eval_unified as eu  # sets BLAS threads = 1 before numpy import

import argparse
import csv
import time
from statistics import mean

import numpy as np


class Timer:
    def __init__(self) -> None:
        self.samples: list[float] = []
        self.pending = 0.0

    def wrap(self, fn, accumulate_only: bool = False):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            if accumulate_only:
                self.pending += dt
            else:
                self.samples.append(dt)
            return out
        return timed


def bench(model: str, episodes: int) -> dict:
    from leo_mappo import LEOMobilityEnv, MAPPO
    from leo_mappo import reasoning as reasoning_mod
    from leo_mappo.runner import evaluate_policy

    cfg = eu.make_config(eu.MODEL_OVERRIDES[model], "nominal", 7, 7)
    env = LEOMobilityEnv(cfg)
    agent = MAPPO(env.obs_dim, env.global_state_dim, env.action_dim, cfg["mappo"], 7,
                  num_satellites=env.num_cells, candidate_count=env.candidate_count,
                  eta_handover=float(env.reward_cfg["eta_handover"]))
    agent.load(eu.weights_path(model, 7))

    actor_t, reason_t = Timer(), Timer()
    step_reason: list[float] = []

    original_analyze = reasoning_mod.LookaheadReasoner.analyze
    timed_analyze = reason_t.wrap(original_analyze, accumulate_only=True)
    reasoning_mod.LookaheadReasoner.analyze = timed_analyze  # class-level patch

    def close_step(fn):
        def wrapped(*args, **kwargs):
            # reasoning for this step happens before the actor call
            step_reason.append(reason_t.pending)
            reason_t.pending = 0.0
            return fn(*args, **kwargs)
        return wrapped

    agent.act = close_step(actor_t.wrap(agent.act))
    agent.act_recurrent = close_step(actor_t.wrap(agent.act_recurrent))
    try:
        evaluate_policy(env, agent, episodes=episodes, deterministic=False,
                        evaluation_action_seed=7007)
    finally:
        reasoning_mod.LookaheadReasoner.analyze = original_analyze

    actor_ms = np.array(actor_t.samples) * 1e3
    reason_ms = np.array(step_reason[: len(actor_ms)]) * 1e3
    total_ms = actor_ms + reason_ms
    n_ue = env.num_agents
    return {
        "model": model, "steps": len(actor_ms), "ues": n_ue,
        "actor_ms_mean": actor_ms.mean(), "actor_ms_p95": np.percentile(actor_ms, 95),
        "reason_ms_mean": reason_ms.mean(), "reason_ms_p95": np.percentile(reason_ms, 95),
        "total_ms_mean": total_ms.mean(), "total_ms_p95": np.percentile(total_ms, 95),
        "total_ms_per_ue": total_ms.mean() / n_ue,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["M1", "M2", "M3", "M4", "M5"])
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        print(f"benchmarking {model} ...", flush=True)
        rows.append(bench(model, args.episodes))

    out = eu.ROOT / "outputs" / "latency"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "latency.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n| model | steps | actor ms (mean/p95) | reasoning ms (mean/p95) | total ms (mean/p95) | per UE ms |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['steps']} | {r['actor_ms_mean']:.3f} / {r['actor_ms_p95']:.3f} | "
              f"{r['reason_ms_mean']:.3f} / {r['reason_ms_p95']:.3f} | "
              f"{r['total_ms_mean']:.3f} / {r['total_ms_p95']:.3f} | {r['total_ms_per_ue']:.4f} |")
    print(f"\nWrote {out / 'latency.csv'}  (single thread, control epoch = 1 s)")


if __name__ == "__main__":
    main()
