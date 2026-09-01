from __future__ import annotations

import json
from pathlib import Path

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import evaluate_policy


ROOT = Path(__file__).resolve().parent

BASE_CONFIG = ROOT / "configs" / "default.yaml"

COMMON_OVERRIDES = [
    ROOT / "configs" / "validation_agentic_balanced_v2.yaml",
    ROOT / "configs" / "validation_reliability_stress_055.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
]

SEED_CONFIG = (
    ROOT
    / "outputs"
    / "seed7_compare_1500"
    / "seed7.yaml"
)

MODELS = {
    "M4": {
        "overrides": [
            ROOT / "configs" / "m4_coordination.yaml",
            *COMMON_OVERRIDES,
            SEED_CONFIG,
        ],
        "weights": (
            ROOT
            / "outputs"
            / "seed7_compare_1500"
            / "m4"
            / "mappo_weights.npz"
        ),
    },

    "M5": {
        "overrides": [
            ROOT / "configs" / "full_agentic.yaml",
            *COMMON_OVERRIDES,
            SEED_CONFIG,
            ROOT / "configs" / "dual_mid_0005.yaml",
        ],
        "weights": (
            ROOT
            / "outputs"
            / "seed7_compare_1500"
            / "m5"
            / "mappo_weights.npz"
        ),
    },
}


METRIC_KEYS = [
    "outage_ratio",
    "service_deficit",
    "service_ratio_mean",
    "resource_cost",
    "dro",
    "handover_started",
    "handover_failure_rate",
    "pingpong_rate",
    "deadline_violation_rate",
    "outage_dual_mu",
    "team_reward",

    "action_stay_fraction",
    "action_prepare_fraction",
    "action_dup_on_fraction",
    "action_dup_off_fraction",
    "action_commit_fraction",
    "action_release_fraction",
    "action_rollback_fraction",

    "reasoning_planning_risk_action_spread",
    "reasoning_planning_risk_discriminating_fraction",
]


def check_files():
    required = [BASE_CONFIG, SEED_CONFIG]

    for spec in MODELS.values():
        required.extend(spec["overrides"])
        required.append(spec["weights"])

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "다음 파일을 찾을 수 없습니다:\n"
            + "\n".join(missing)
        )


def evaluate_model(name: str, spec: dict) -> dict:
    print()
    print("=" * 70)
    print(f"{name} - 200 STOCHASTIC EPISODES START")
    print("=" * 70)
    print(f"weights = {spec['weights']}")
    print()

    cfg = load_config(
        str(BASE_CONFIG),
        [
            str(path)
            for path in spec["overrides"]
        ],
    )

    print(
        f"{name} config:"
        f" seed={cfg['seed']},"
        f" memory={cfg['mappo'].get('use_memory', False)},"
        f" reasoning={cfg['mappo'].get('use_reasoning', False)},"
        f" coordination={cfg['mappo'].get('use_coordination', False)},"
        f" constraint={cfg['mappo'].get('use_constraint', False)}"
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

    agent.load(
        str(spec["weights"])
    )

    if name == "M5":
        print(
            f"loaded M5 mu = "
            f"{agent.outage_dual_mu:.9f}"
        )

    print()
    print(
        f"[{name}] 200 stochastic episodes 평가 중..."
    )

    result = evaluate_policy(
        env,
        agent,
        episodes=200,
        deterministic=False,
    )

    print()
    print("=" * 70)
    print(f"{name} - 200 STOCHASTIC EPISODES RESULT")
    print("=" * 70)

    for key in METRIC_KEYS:
        if key in result:
            print(
                f"{key}={result[key]}"
            )

    outage = float(
        result["outage_ratio"]
    )

    print()
    print(
        "CONSTRAINT="
        + (
            "PASS"
            if outage <= 0.10
            else "FAIL"
        )
    )

    print(
        f"OUTAGE GAP="
        f"{outage - 0.10:+.9f}"
    )

    return {
        key: float(value)
        if isinstance(value, (int, float))
        else value
        for key, value in result.items()
    }


def main():
    check_files()

    all_results = {}

    for name in ["M4", "M5"]:
        all_results[name] = evaluate_model(
            name,
            MODELS[name],
        )

    print()
    print()
    print("#" * 70)
    print("FINAL M4 vs M5")
    print("#" * 70)

    m4 = all_results["M4"]
    m5 = all_results["M5"]

    print(
        f"M4 outage_ratio = "
        f"{m4['outage_ratio']:.9f}"
    )

    print(
        f"M5 outage_ratio = "
        f"{m5['outage_ratio']:.9f}"
    )

    print(
        f"M4 service_deficit = "
        f"{m4['service_deficit']:.9f}"
    )

    print(
        f"M5 service_deficit = "
        f"{m5['service_deficit']:.9f}"
    )

    print(
        f"M4 resource_cost = "
        f"{m4['resource_cost']:.9f}"
    )

    print(
        f"M5 resource_cost = "
        f"{m5['resource_cost']:.9f}"
    )

    outage_improvement = (
        float(m4["outage_ratio"])
        - float(m5["outage_ratio"])
    )

    print()
    print(
        f"M4 - M5 outage difference = "
        f"{outage_improvement:+.9f}"
    )

    print(
        "M4 constraint = "
        + (
            "PASS"
            if float(m4["outage_ratio"]) <= 0.10
            else "FAIL"
        )
    )

    print(
        "M5 constraint = "
        + (
            "PASS"
            if float(m5["outage_ratio"]) <= 0.10
            else "FAIL"
        )
    )

    output_path = (
        ROOT
        / "outputs"
        / "seed7_compare_1500"
        / "evaluation_200_stochastic.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print()
    print(
        f"결과 저장 완료:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()
