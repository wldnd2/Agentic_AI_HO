from __future__ import annotations

import json
from pathlib import Path

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import evaluate_policy

ROOT = Path(__file__).resolve().parent

COMMON = [
    ROOT / "configs" / "validation_agentic_balanced_v2.yaml",
    ROOT / "configs" / "validation_reliability_stress_055.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
    ROOT / "outputs" / "seed7_compare_1500" / "seed7.yaml",
]

MODELS = {
    "M1": {
        "overrides": [
            *COMMON,
        ],
        "weights": (
            ROOT / "outputs" / "seed7_compare_1500"
            / "m1" / "mappo_weights.npz"
        ),
    },
    "M2": {
        "overrides": [
            ROOT / "configs" / "memory_only.yaml",
            *COMMON,
        ],
        "weights": (
            ROOT / "outputs" / "seed7_compare_1500"
            / "m2" / "mappo_weights.npz"
        ),
    },
    "M3": {
        "overrides": [
            ROOT / "configs" / "memory_reasoning.yaml",
            *COMMON,
        ],
        "weights": (
            ROOT / "outputs" / "seed7_compare_1500"
            / "m3" / "mappo_weights.npz"
        ),
    },
}

KEYS = [
    "outage_ratio",
    "service_deficit",
    "service_ratio_mean",
    "resource_cost",
    "dro",
    "handover_started",
    "handover_failure_rate",
    "pingpong_rate",
    "deadline_violation_rate",
    "team_reward",
    "action_stay_fraction",
    "action_prepare_fraction",
    "action_dup_on_fraction",
    "action_commit_fraction",
    "action_release_fraction",
    "reasoning_planning_risk_action_spread",
    "reasoning_planning_risk_discriminating_fraction",
]

results = {}

for name, spec in MODELS.items():

    cfg = load_config(
        str(ROOT / "configs" / "default.yaml"),
        [str(x) for x in spec["overrides"]],
    )

    env = LEOMobilityEnv(cfg)

    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        cfg["mappo"],
        int(cfg["seed"]),
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(
            env.reward_cfg["eta_handover"]
        ),
    )

    agent.load(str(spec["weights"]))

    print()
    print("=" * 60)
    print(f"{name}: 200 stochastic episodes")
    print("=" * 60)

    r = evaluate_policy(
        env,
        agent,
        episodes=200,
        deterministic=False,
    )

    results[name] = r

    for key in KEYS:
        if key in r:
            print(f"{key}={r[key]}")

out = (
    ROOT
    / "outputs"
    / "seed7_compare_1500"
    / "evaluation_200_stochastic_m1_m3.json"
)

with out.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print()
print(f"saved: {out}")
