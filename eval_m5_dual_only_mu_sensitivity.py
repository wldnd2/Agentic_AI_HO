from pathlib import Path

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import evaluate_policy


ROOT = Path(__file__).resolve().parent

BASE_CONFIG = ROOT / "configs" / "default.yaml"

OVERRIDES = [
    ROOT / "configs" / "validation_agentic_challenge_v3.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
    ROOT / "configs" / "full_agentic.yaml",
    ROOT / "configs" / "dual_mid_0005.yaml",
]

WEIGHTS = (
    ROOT
    / "outputs"
    / "v3_final_1500"
    / "m5_dual_only_seed7"
    / "mappo_weights.npz"
)

ENV_SEED = 7
AGENT_SEED = 7
ACTION_SEED = 7007
EPISODES = 200

MU_VALUES = [
    0.0,
    0.10,
    0.20,
    0.30,
    0.3787,
    0.45,
]


def make_env_agent(mu):
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

    if not agent.use_constraint:
        raise RuntimeError(
            "Checkpoint is not an M5 constrained policy."
        )

    # evaluate_policy() constructs its fixed evaluation
    # ReliabilityDualController from this value.
    agent.outage_dual_mu = float(mu)

    return env, agent


results = []

for mu in MU_VALUES:
    env, agent = make_env_agent(mu)

    result = evaluate_policy(
        env,
        agent,
        episodes=EPISODES,
        deterministic=False,
        evaluation_action_seed=ACTION_SEED,
    )

    results.append((mu, result))

    print()
    print("=" * 80)
    print(f"FIXED MU = {mu:.9f}")
    print("=" * 80)

    for key in [
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
        "resource_price_mean",
        "reasoning_service_cost",
        "reasoning_coordination_cost",
        "reasoning_reliability_cost",
        "reasoning_selected_planning_risk",
        "reasoning_score_gap",
        "team_reward",
    ]:
        if key in result:
            print(f"{key}={result[key]}")


print()
print("=" * 80)
print("MU SENSITIVITY SUMMARY")
print("=" * 80)

print(
    "mu,outage_ratio,service_deficit,resource_cost,"
    "handover_started,pingpong_rate,planning_risk"
)

for mu, result in results:
    print(
        f"{mu:.9f},"
        f"{result['outage_ratio']:.9f},"
        f"{result['service_deficit']:.9f},"
        f"{result['resource_cost']:.9f},"
        f"{result['handover_started']:.9f},"
        f"{result['pingpong_rate']:.9f},"
        f"{result.get('reasoning_selected_planning_risk', 0.0):.9f}"
    )
