"""Tune the A3/TTT CHO baseline on validation seeds in the TWC environment.

Usage: PYTHONPATH=src python scripts/twc_cho_sweep.py [--episodes 2] [--workers 4]
Writes outputs/twc/cho_sweep.csv and prints the best (exec offset, TTT) by
common reward, together with its outage ratio.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from cho_baselines import A3ConditionalHandover  # noqa: E402
from leo_mappo import LEOMobilityEnv, load_config  # noqa: E402

CFG = ROOT / "configs"
TWC = [
    CFG / "validation_agentic_challenge_v3.yaml",
    CFG / "eta_handover_005.yaml",
    CFG / "validation_dwell_15.yaml",
    CFG / "twc_walker_p3.yaml",
    CFG / "twc_3gpp_set1.yaml",
    CFG / "twc_load_24ue.yaml",
]
VALIDATION_SEEDS = (20, 21, 22)


def run(job):
    exec_offset, ttt, seed, episodes, rate, max_steps = job
    cfg = load_config(CFG / "default.yaml", TWC)
    if max_steps:
        cfg["environment"]["episode_steps"] = max_steps  # smoke test only
    if rate is not None:
        cfg["traffic"]["mean_arrival_mbps"] = rate
    cfg["environment_seed"] = seed
    env = LEOMobilityEnv(cfg)
    policy = A3ConditionalHandover(prep_offset_db=1.0, exec_offset_db=exec_offset, ttt_steps=ttt)
    steps = int(cfg["environment"]["episode_steps"])
    keys = ("outage_ratio", "signaling_cost", "handover_started", "pingpong", "handover_success")
    sums = {k: 0.0 for k in keys}
    reward = 0.0
    for episode in range(episodes):
        env.reset(seed=1000 * seed + episode)
        for _ in range(steps):
            out = env.step(policy(env))
            reward += float(np.mean(out.rewards))
            for k in keys:
                sums[k] += float(out.info.get(k, 0.0))
    n = episodes * steps
    return {
        "exec_offset_db": exec_offset, "ttt_steps": ttt, "seed": seed, "rate_mbps": rate,
        "common_reward": reward / n, "outage": sums["outage_ratio"] / n,
        "signaling": sums["signaling_cost"] / n, "ho_per_epoch": sums["handover_started"] / n,
        "pingpong_ratio": sums["pingpong"] / max(sums["handover_success"], 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate", type=float, default=None, help="override per-UE traffic (Mb/s)")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the script")
    parser.add_argument("--offsets", nargs="+", type=float, default=[2.0, 3.0, 4.0], help="execution offsets (dB)")
    parser.add_argument("--ttts", nargs="+", type=int, default=[1, 2, 3], help="time-to-trigger (epochs)")
    parser.add_argument("--tag", default="", help="suffix for the output CSV")
    args = parser.parse_args()
    offsets, ttts, seeds = (tuple(args.offsets), tuple(args.ttts), VALIDATION_SEEDS)
    if args.smoke:
        offsets, ttts, seeds = (2.0,), (1, 2), (20,)
    jobs = [
        (e, t, s, args.episodes, args.rate, 20 if args.smoke else 0)
        for e, t, s in itertools.product(offsets, ttts, seeds)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, jobs))
    out = ROOT / "outputs" / "twc" / ("cho_sweep_smoke.csv" if args.smoke else f"cho_sweep{args.tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grid = {}
    for row in rows:
        grid.setdefault((row["exec_offset_db"], row["ttt_steps"]), []).append(row)
    print(f"{'offset':>6} {'TTT':>3} {'reward':>8} {'outage':>7} {'signal':>7} {'HO/ep':>6} {'pp':>6}")
    best = None
    for key, rs in sorted(grid.items()):
        m = {k: float(np.mean([r[k] for r in rs])) for k in ("common_reward", "outage", "signaling", "ho_per_epoch", "pingpong_ratio")}
        print(f"{key[0]:6.1f} {key[1]:3d} {m['common_reward']:8.4f} {m['outage']:7.3f} {m['signaling']:7.3f} {m['ho_per_epoch']:6.3f} {m['pingpong_ratio']:6.3f}")
        if best is None or m["common_reward"] > best[1]["common_reward"]:
            best = (key, m)
    print(f"best: exec_offset={best[0][0]} dB, TTT={best[0][1]} -> outage {best[1]['outage']:.3f}, reward {best[1]['common_reward']:.4f}")


if __name__ == "__main__":
    main()
