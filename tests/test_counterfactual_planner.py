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


def _counterfactual_plans(
    env: LEOMobilityEnv, reasoner: LookaheadReasoner, ue_id: int
) -> dict[int, list[tuple[tuple[int, ...], int | None]]]:
    return reasoner.planner._counterfactual_plans(
        env, ue_id, env.action_masks()[ue_id]
    )


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


def test_every_counterfactual_continuation_preserves_first_action() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)

    def assert_first_actions() -> None:
        plans = _counterfactual_plans(env, reasoner, ue_id)
        for immediate_action, action_plans in plans.items():
            assert action_plans
            assert all(
                sequence[0] == immediate_action for sequence, _ in action_plans
            )

    assert_first_actions()
    env.ues[ue_id].state = MobilityState.PREPARED
    env.ues[ue_id].secondary_cell = target
    for timer in (2, 0):
        env.ues[ue_id].protocol_timer = timer
        assert_first_actions()
    env.ues[ue_id].state = MobilityState.SWITCHING
    env.ues[ue_id].protocol_timer = 1
    assert_first_actions()


def test_single_stay_has_pure_and_delayed_prepare_continuations() -> None:
    env, reasoner = _case()
    ue_id, _ = _ue_with_candidate(env)
    current_candidates = tuple(env.candidates[ue_id])
    plans = _counterfactual_plans(env, reasoner, ue_id)
    stay_plans = plans[0]
    assert any(all(action == 0 for action in sequence) for sequence, _ in stay_plans)
    for slot, target in enumerate(current_candidates):
        prepare_action = env.prepare_start + slot
        assert any(
            plan_target == target
            and sequence[0] == 0
            and sequence[1] == prepare_action
            and env.commit_action in sequence
            for sequence, plan_target in stay_plans
        )


def test_immediate_prepare_has_commit_abort_and_hold_continuations() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    prepare_action = env.prepare_start
    plans = _counterfactual_plans(env, reasoner, ue_id)[prepare_action]
    assert all(plan_target == target for _, plan_target in plans)
    assert any(env.commit_action in sequence for sequence, _ in plans)
    assert any(sequence[1] == env.release_action for sequence, _ in plans)
    assert any(all(action == 0 for action in sequence[1:]) for sequence, _ in plans)


def test_prepared_waiting_stay_commits_only_after_timer_wait() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    ue = env.ues[ue_id]
    ue.state = MobilityState.PREPARED
    ue.secondary_cell = target
    ue.protocol_timer = 2
    stay_plans = _counterfactual_plans(env, reasoner, ue_id)[0]
    commit_plan = next(
        sequence for sequence, _ in stay_plans if env.commit_action in sequence
    )
    assert commit_plan[:2] == (0, 0)
    assert commit_plan[2] == env.commit_action
    assert any(sequence[1] == env.release_action for sequence, _ in stay_plans)


def test_prepared_ready_stay_has_next_commit_and_release_options() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    ue = env.ues[ue_id]
    ue.state = MobilityState.PREPARED
    ue.secondary_cell = target
    ue.protocol_timer = 0
    stay_plans = _counterfactual_plans(env, reasoner, ue_id)[0]
    assert any(sequence[1] == env.commit_action for sequence, _ in stay_plans)
    assert any(sequence[1] == env.release_action for sequence, _ in stay_plans)
    assert any(all(action == 0 for action in sequence) for sequence, _ in stay_plans)


def test_release_can_reprepare_only_from_current_candidate_pool() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    current_candidates = tuple(env.candidates[ue_id])
    ue = env.ues[ue_id]
    ue.state = MobilityState.PREPARED
    ue.secondary_cell = target
    ue.protocol_timer = 0
    release_plans = _counterfactual_plans(
        env, reasoner, ue_id
    )[env.release_action]
    for slot, candidate in enumerate(current_candidates):
        assert any(
            sequence[0] == env.release_action
            and sequence[1] == env.prepare_start + slot
            and env.commit_action in sequence
            and plan_target == candidate
            for sequence, plan_target in release_plans
        )
    assert {
        plan_target for _, plan_target in release_plans if plan_target is not None
    }.issubset(set(current_candidates))


def test_plan_generation_does_not_reselect_future_candidates() -> None:
    env, reasoner = _case()
    ue_id, _ = _ue_with_candidate(env)
    current_candidates = tuple(env.candidates[ue_id])
    with patch.object(
        env,
        "_select_candidates",
        side_effect=AssertionError("future candidate selection must not be called"),
    ):
        plans = _counterfactual_plans(env, reasoner, ue_id)
    generated_targets = {
        target
        for action_plans in plans.values()
        for _, target in action_plans
        if target is not None
    }
    assert generated_targets.issubset(set(current_candidates))


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
    with patch.object(reasoner, "_future_link", return_value=(6.0, 1.0)):
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


def test_good_prepared_target_makes_stay_select_future_commit() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    ue = env.ues[ue_id]
    ue.state = MobilityState.PREPARED
    ue.secondary_cell = target
    ue.protocol_timer = 0
    env.remaining_visibility[ue_id, target] = 1e9
    env.min_elevation_rad = -np.pi / 2.0

    def good_target_link(
        _env: LEOMobilityEnv,
        _ue_id: int,
        cell_id: int | None,
        _offset: int,
    ) -> tuple[float, float]:
        return (1.0, 1.0 if cell_id == target else 0.0)

    with patch.object(reasoner, "_future_link", side_effect=good_target_link):
        result = reasoner.analyze(env)
    assert result.selected_action_sequences is not None
    selected_stay = result.selected_action_sequences[ue_id, 0]
    assert selected_stay[0] == 0
    assert selected_stay[1] == env.commit_action


def test_bad_prepared_target_can_legitimately_prefer_release() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    primary = env.ues[ue_id].primary_cell
    ue = env.ues[ue_id]
    ue.state = MobilityState.PREPARED
    ue.secondary_cell = target
    ue.protocol_timer = 0
    env.remaining_visibility[ue_id, target] = 1e9
    env.min_elevation_rad = -np.pi / 2.0

    def bad_target_link(
        _env: LEOMobilityEnv,
        _ue_id: int,
        cell_id: int | None,
        _offset: int,
    ) -> tuple[float, float]:
        return (1.0, 1.0 if cell_id == primary else 0.0)

    with patch.object(reasoner, "_future_link", side_effect=bad_target_link):
        result = reasoner.analyze(env)
    assert result.selected_action_sequences is not None
    assert result.selected_action_sequences[ue_id, 0, 1] == env.release_action
    assert result.raw_costs[ue_id, env.release_action] < 0.0


def test_single_stay_can_select_delayed_prepare_instead_of_never_preparing() -> None:
    env, reasoner = _case()
    ue_id, target = _ue_with_candidate(env)
    env.remaining_visibility[ue_id, target] = 1e9
    env.min_elevation_rad = -np.pi / 2.0

    def useful_soon_link(
        _env: LEOMobilityEnv,
        _ue_id: int,
        cell_id: int | None,
        _offset: int,
    ) -> tuple[float, float]:
        return (1.0, 1.0 if cell_id == target else 0.0)

    with patch.object(reasoner, "_future_link", side_effect=useful_soon_link):
        result = reasoner.analyze(env)
    assert result.selected_action_sequences is not None
    selected_stay = result.selected_action_sequences[ue_id, 0]
    assert selected_stay[0] == 0
    assert selected_stay[1] == env.prepare_start
    assert env.commit_action in selected_stay


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
