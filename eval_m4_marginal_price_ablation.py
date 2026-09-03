from copy import deepcopy
from pathlib import Path

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import evaluate_policy


ROOT = Path(__file__).resolve().parent

BASE_CONFIG = ROOT / "configs" / "default.yaml"

OVERRIDES = [
    ROOT / "configs" / "validation_agentic_challenge_v3.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
    ROOT / "configs" / "m4_coordination.yaml",
]

WEIGHTS = (
    ROOT
    / "outputs"
    / "v3_validation_420"
    / "m4_marginal"
    / "mappo_weights.npz"
)

ENV_SEED = 7
AGENT_SEED = 7
ACTION_SEED = 7007
EPISODES = 200


def make_config(zero_price=False):
    cfg = load_config(
        str(BASE_CONFIG),
        [str(path) for path in OVERRIDES],
    )

    cfg["environment_seed"] = ENV_SEED
    cfg["agent_seed"] = AGENT_SEED

    if zero_price:
        cfg = deepcopy(cfg)

        cfg["coordination"]["price_learning_rate"] = 0.0
        cfg["coordination"]["initial_price"] = 0.0

    return cfg


def evaluate(zero_price=False):
    cfg = make_config(zero_price)

    env = LEOMobilityEnv(cfg)

    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        cfg["mappo"],
        AGENT_SEED,
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(
            env.reward_cfg["eta_handover"]
        ),
    )

    agent.load(str(WEIGHTS))

    if not agent.use_coordination:
        raise RuntimeError(
            "Loaded checkpoint is not an M4 coordination policy."
        )

    return evaluate_policy(
        env,
        agent,
        episodes=EPISODES,
        deterministic=False,
        evaluation_action_seed=ACTION_SEED,
    )


def print_result(title, result):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    keys = [
        "outage_ratio",
        "service_deficit",
        "service_ratio_mean",
        "resource_cost",
        "dro",
        "handover_started",
        "handover_failure_rate",
        "pingpong_rate",
        "deadline_violation_rate",
        "preparation_rejection_rate",
        "prepare_to_release_without_commit_rate",
        "prepare_to_commit_rate",
        "duplication_active_time_fraction",
        "action_stay_fraction",
        "action_prepare_fraction",
        "action_commit_fraction",
        "action_release_fraction",
        "resource_price_mean",
        "resource_price_max",
        "price_nonzero_fraction",
        "reasoning_service_cost",
        "reasoning_coordination_cost",
        "reasoning_total_cost",
        "reasoning_score_gap",
        "team_reward",
    ]

    for key in keys:
        if key in result:
            print(f"{key}={result[key]}")


normal = evaluate(zero_price=False)
zero = evaluate(zero_price=True)

print_result(
    "M4 NORMAL COORDINATION - 200 STOCHASTIC",
    normal,
)

print_result(
    "M4 ZERO PRICE - 200 STOCHASTIC",
    zero,
)

print()
print("=" * 78)
print("COORDINATION EFFECT")
print("=" * 78)

keys = [
    "outage_ratio",
    "service_deficit",
    "resource_cost",
    "dro",
    "handover_started",
    "pingpong_rate",
    "deadline_violation_rate",
    "preparation_rejection_rate",
    "prepare_to_release_without_commit_rate",
    "prepare_to_commit_rate",
    "duplication_active_time_fraction",
    "team_reward",
]

for key in keys:
    normal_value = float(normal[key])
    zero_value = float(zero[key])

    print(
        f"{key}: "
        f"normal={normal_value:.9f} | "
        f"zero={zero_value:.9f} | "
        f"zero-normal={zero_value-normal_value:+.9f}"
    )
