from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.planner import (
    _counterfactual_cost_deltas,
    _fixed_scale_features,
)
from leo_mappo.reasoning import LookaheadReasoner
from leo_mappo.residual_world_model import ResidualLinkPredictor
from leo_mappo.ue import MobilityState, PacketBatch


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["episode_steps"] = 12
    return config


def _case(
    *,
    counterfactual: bool = True,
    use_coordination: bool = False,
    use_constraint: bool = False,
) -> tuple[LEOMobilityEnv, LookaheadReasoner]:
    config = _config()
    if counterfactual:
        config["reasoning"]["planner_objective"] = "counterfactual"
    env = LEOMobilityEnv(config)
    env.reset(seed=707)
    for ue in env.ues:
        ue.queue.append(PacketBatch(1_000))
    reasoner = LookaheadReasoner(
        config["reasoning"],
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=use_coordination,
        use_constraint=use_constraint,
    )
    return env, reasoner


def _ue_with_candidate(env: LEOMobilityEnv) -> tuple[int, int]:
    ue_id = next(index for index, candidates in enumerate(env.candidates) if candidates)
    return ue_id, env.candidates[ue_id][0]


def test_omitted_objective_matches_explicit_legacy_outputs() -> None:
    config = _config()
    env = LEOMobilityEnv(config)
    env.reset(seed=707)
    for ue in env.ues:
        ue.queue.append(PacketBatch(1_000))
    omitted = LookaheadReasoner(config["reasoning"], env.reward_cfg).analyze(env)
    explicit_config = deepcopy(config["reasoning"])
    explicit_config["planner_objective"] = "legacy"
    explicit = LookaheadReasoner(explicit_config, env.reward_cfg).analyze(env)
    for field in (
        "features",
        "raw_costs",
        "predicted_deficits",
        "best_scores",
        "score_gaps",
        "coordination_costs",
        "predicted_outages",
        "planning_reliability_risks",
        "predicted_control_costs",
        "predicted_handover_costs",
        "predicted_preparation_costs",
        "service_costs",
        "reliability_costs",
        "total_costs",
    ):
        np.testing.assert_array_equal(getattr(omitted, field), getattr(explicit, field))


def test_counterfactual_stay_is_exact_zero() -> None:
    env, reasoner = _case()
    result = reasoner.analyze(env)
    assert result.delta_total_cost is not None
    np.testing.assert_array_equal(
        result.delta_total_cost[:, 0], np.zeros(env.num_agents)
    )
    np.testing.assert_array_equal(result.raw_costs[:, 0], np.zeros(env.num_agents))
    np.testing.assert_array_equal(result.features[:, 0], np.zeros(env.num_agents))
    feasible = env.action_masks() > 0.0
    np.testing.assert_allclose(
        result.raw_costs[feasible],
        np.clip(
            result.delta_total_cost[feasible],
            -reasoner.score_clip,
            reasoner.score_clip,
        ),
    )
    np.testing.assert_allclose(
        result.features[feasible], result.raw_costs[feasible] / reasoner.score_clip
    )


def test_common_service_constant_cancels_from_deltas() -> None:
    service = np.asarray([0.4, 0.1, 0.8])
    reliability = np.asarray([0.2, 0.3, 0.1])
    control = np.asarray([0.0, 0.02, 0.03])
    coordination = np.asarray([0.0, 0.04, 0.01])
    baseline = _counterfactual_cost_deltas(
        service, reliability, control, coordination
    )
    shifted = _counterfactual_cost_deltas(
        service + 123.0, reliability, control, coordination
    )
    for expected, actual in zip(baseline, shifted):
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_reliability_dual_is_zero_at_zero_and_operational_when_positive() -> None:
    env, reasoner = _case(use_constraint=True)
    ue_id, target = _ue_with_candidate(env)
    ue = env.ues[ue_id]
    ue.secondary_cell = target
    ue.state = MobilityState.PREPARED
    ue.protocol_timer = 0
    env.remaining_visibility[ue_id, target] = 1e9
    env.min_elevation_rad = -np.pi / 2.0

    def contrasting_link(
        _env: LEOMobilityEnv,
        _ue_id: int,
        cell_id: int | None,
        _offset: int,
    ) -> tuple[float, float]:
        return (1.0, 1.0 if cell_id == target else 0.1)

    with patch.object(reasoner, "_future_link", side_effect=contrasting_link):
        zero = reasoner.analyze(env, outage_dual_mu=0.0)
        positive = reasoner.analyze(env, outage_dual_mu=2.0)
    assert zero.delta_reliability_cost is not None
    assert positive.delta_reliability_cost is not None
    feasible = env.action_masks()[ue_id] > 0.0
    np.testing.assert_array_equal(
        zero.delta_reliability_cost[ue_id, feasible],
        np.zeros(np.count_nonzero(feasible)),
    )
    assert positive.delta_reliability_cost[ue_id, env.commit_action] < 0.0


def test_existing_prepare_control_cost_enters_counterfactual_delta() -> None:
    env, reasoner = _case()
    ue_id, _ = _ue_with_candidate(env)
    result = reasoner.analyze(env)
    assert result.delta_control_cost is not None
    assert result.delta_control_cost[ue_id, env.prepare_start] > 0.0
    assert result.delta_control_cost[ue_id, 0] == 0.0


def test_prepared_release_differs_only_through_existing_control_cost() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    ue = env.ues[ue_id]
    ue.secondary_cell = target
    ue.state = MobilityState.PREPARED
    ue.protocol_timer = 0
    env.min_elevation_rad = -np.pi / 2.0
    result = reasoner.analyze(env)
    assert result.delta_service_cost is not None
    assert result.delta_control_cost is not None
    assert result.delta_total_cost is not None
    release = env.release_action
    assert result.delta_service_cost[ue_id, release] == 0.0
    assert result.delta_control_cost[ue_id, release] < 0.0
    assert result.delta_total_cost[ue_id, release] < 0.0
    assert result.delta_control_cost[ue_id, env.commit_action] != 0.0


def test_coordination_delta_enters_only_when_coordination_is_enabled() -> None:
    env, disabled_reasoner = _case(use_coordination=False)
    enabled_reasoner = LookaheadReasoner(
        env.config["reasoning"],
        env.reward_cfg,
        env.config.get("coordination", {}),
        use_coordination=True,
    )
    ue_id, target = _ue_with_candidate(env)
    prices = np.zeros(env.num_cells)
    prices[target] = 2.0
    disabled = disabled_reasoner.analyze(env, prices=prices)
    enabled = enabled_reasoner.analyze(env, prices=prices)
    assert disabled.delta_coordination_cost is not None
    assert enabled.delta_coordination_cost is not None
    assert disabled.delta_coordination_cost[ue_id, env.prepare_start] == 0.0
    assert enabled.delta_coordination_cost[ue_id, env.prepare_start] > 0.0


def test_fixed_feature_scaling_preserves_tiny_costs() -> None:
    raw = np.asarray([0.0, 1e-5, -1e-5, 10.0])
    features = _fixed_scale_features(raw, score_clip=5.0)
    np.testing.assert_allclose(features, [0.0, 2e-6, -2e-6, 1.0])


def test_better_than_stay_plan_can_have_negative_raw_cost() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    env.remaining_visibility[ue_id, target] = 1e9
    env.min_elevation_rad = -np.pi / 2.0

    def better_target_link(
        _env: LEOMobilityEnv,
        _ue_id: int,
        cell_id: int | None,
        _offset: int,
    ) -> tuple[float, float]:
        return (1.0, 1.0 if cell_id == target else 0.0)

    with patch.object(reasoner, "_future_link", side_effect=better_target_link):
        result = reasoner.analyze(env)
    assert result.raw_costs[ue_id, env.prepare_start] < 0.0


def test_geometry_and_hybrid_support_counterfactual_objective(tmp_path: Path) -> None:
    env, geometry_reasoner = _case()
    geometry = geometry_reasoner.analyze(env)
    checkpoint = tmp_path / "model.npz"
    ResidualLinkPredictor(
        history_steps=1,
        horizon_steps=6,
        hidden_dim=4,
        mlp_hidden_dim=4,
        seed=5,
    ).save(checkpoint)
    hybrid_config = deepcopy(env.config["reasoning"])
    hybrid_config.update(
        planner_objective="counterfactual",
        world_model="hybrid",
        residual_model_path=str(checkpoint),
    )
    hybrid_reasoner = LookaheadReasoner(hybrid_config, env.reward_cfg)
    hybrid = hybrid_reasoner.analyze(env)
    assert geometry.delta_total_cost is not None
    assert hybrid.delta_total_cost is not None
    np.testing.assert_array_equal(
        geometry.delta_total_cost[:, 0], np.zeros(env.num_agents)
    )
    np.testing.assert_array_equal(
        hybrid.delta_total_cost[:, 0], np.zeros(env.num_agents)
    )
