"""Compare the JSAC V3 environment with Walker-slice TWC environments.

Usage:  PYTHONPATH=src python scripts/diagnose_twc_env.py [--ues 12] [--episodes 4]
Reports satellites kept, visible satellites per UE, serving elevation and SINR,
load concentration, Max-SINR handover rate and outage, and wall time per step.
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.baselines import max_sinr_policy

CFG = Path(__file__).resolve().parents[3] / "configs"
V3 = [CFG / "validation_agentic_challenge_v3.yaml", CFG / "eta_handover_005.yaml", CFG / "validation_dwell_15.yaml"]
ENVS = {
    "JSAC V3": [],
    "Walker P2": [CFG / "twc_walker_p2.yaml"],
    "Walker P3": [CFG / "twc_walker_p3.yaml"],
    "P3+Set1": [CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set1.yaml"],
    "P3+Set2": [CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set2.yaml"],
}


def measure(name: str, extra: list[Path], ues: int, episodes: int) -> None:
    cfg = load_config(CFG / "default.yaml", [*V3, *extra])
    cfg["environment"]["num_ues"] = ues
    env = LEOMobilityEnv(cfg)
    steps_per_episode = int(cfg["environment"]["episode_steps"])
    vis, elev, sinr, share, press, outage = [], [], [], [], [], []
    handovers, steps, t0 = 0.0, 0, time.time()
    for episode in range(episodes):
        env.reset(seed=100 + episode)
        for _ in range(steps_per_episode):
            prim = [u.primary_cell for u in env.ues]
            vis.append(np.mean([sum(l.visible for l in env.links[i]) for i in range(env.num_agents)]))
            elev.append(np.mean([env.links[u.ue_id][u.primary_cell].elevation_deg for u in env.ues]))
            sinr.append(np.mean([env.links[u.ue_id][u.primary_cell].sinr_db for u in env.ues]))
            share.append(max(collections.Counter(prim).values()) / len(prim))
            press.append(float(np.max(env.last_requested_pressure)))
            out = env.step(max_sinr_policy(env))
            handovers += out.info["handover_started"]
            outage.append(out.info["outage_ratio"])
            steps += 1
    dt = (time.time() - t0) / steps
    print(
        f"{name:9s} sats={env.num_cells:3d} visible/UE={np.mean(vis):.2f} elev={np.mean(elev):5.1f}deg "
        f"SINR={np.mean(sinr):5.1f}dB max-share={np.mean(share):.2f} peak-pressure={np.mean(press):.2f} "
        f"HO/UE/min={60 * handovers / (steps * ues):.2f} outage(MaxSINR)={np.mean(outage):.3f} ms/step={1000 * dt:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ues", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None, help="subset of environment names")
    args = parser.parse_args()
    for name, extra in ENVS.items():
        if args.only and name not in args.only:
            continue
        measure(name, extra, args.ues, args.episodes)


if __name__ == "__main__":
    main()
