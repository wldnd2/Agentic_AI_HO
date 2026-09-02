from dataclasses import replace
from pathlib import Path

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.reasoning import LookaheadReasoner
from leo_mappo.runner import evaluate_policy


ROOT = Path(__file__).resolve().parent

BASE_CONFIG = ROOT / "configs" / "default.yaml"

OVERRIDES = [
    ROOT / "configs" / "validation_agentic_challenge_v3.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
    ROOT / "configs" / "memory_reasoning.yaml",
]

WEIGHTS = (
    ROOT
    / "outputs"
    / "v3_validation_420"
    / "m3"
    / "mappo_weights.npz"
)

ENV_SEED = 7
AGENT_SEED = 7
ACTION_SEED = 7007
EPISODES = 200


def make_env_agent():
    cfg = load_config(
        str(BASE_CONFIG),
        [str(path) for path in OVERRIDES],
    )

    cfg["environment_seed"] = ENV_SEED
    cfg["agent_seed"] = AGENT_SEED

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

    if not agent.use_memory:
        raise RuntimeError("M3 must use memory.")

    if not agent.use_reasoning:
        raise RuntimeError("Checkpoint is not an M3 reasoning policy.")

    return env, agent


def evaluate_normal():
    env, agent = make_env_agent()

    return evaluate_policy(
        env,
        agent,
        episodes=EPISODES,
        deterministic=False,
        evaluation_action_seed=ACTION_SEED,
    )


def evaluate_reasoning_zero():
    env, agent = make_env_agent()

    original_analyze = LookaheadReasoner.analyze

    def analyze_without_reasoning_signal(
        self,
        *args,
        **kwargs,
    ):
        result = original_analyze(
            self,
            *args,
            **kwargs,
        )

        # Keep the analytical calculation and diagnostics alive,
        # but remove all action-discriminating reasoning input
        # supplied to the actor.
        return replace(
            result,
            features=np.zeros_like(
                result.features
            ),
        )

    LookaheadReasoner.analyze = (
        analyze_without_reasoning_signal
    )

    try:
        result = evaluate_policy(
            env,
            agent,
            episodes=EPISODES,
            deterministic=False,
            evaluation_action_seed=ACTION_SEED,
        )
    finally:
        LookaheadReasoner.analyze = (
            original_analyze
        )

    return result


def print_result(title, result):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)

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
        "duplication_activation_rate",
        "action_stay_fraction",
        "action_prepare_fraction",
        "action_dup_on_fraction",
        "action_commit_fraction",
        "action_release_fraction",
        "hidden_state_norm",
        "reasoning_score_gap",
        "reasoning_planning_risk_action_spread",
        "reasoning_planning_risk_discriminating_fraction",
        "team_reward",
    ]

    for key in keys:
        if key in result:
            print(f"{key}={result[key]}")


normal = evaluate_normal()
zero = evaluate_reasoning_zero()

print_result(
    "M3 NORMAL REASONING - 200 STOCHASTIC",
    normal,
)

print_result(
    "M3 ZERO REASONING FEATURES - 200 STOCHASTIC",
    zero,
)

print()
print("=" * 76)
print("REASONING EFFECT")
print("=" * 76)

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
