from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.coordination_diagnostics import (
    PRICE_INTERVENTIONS,
    average_ranks,
    canonical_preparation_admission_snapshot,
    correlation_summary,
    intervention_price_inputs,
    prepared_decision_row,
)
from leo_mappo.planner import ReasoningResult
from leo_mappo.ue import MobilityState
from scripts.diagnose_m4_coordination_mechanism import EPISODE_SEEDS, evaluate_condition


ROOT = Path(__file__).resolve().parents[1]


def _v3_config() -> dict:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["episode_steps"] = 3
    config["environment"]["num_ues"] = 4
    return config


def test_canonical_capacity_matches_environment_for_requested_target() -> None:
    config = _v3_config()
    env = LEOMobilityEnv(config)
    env.reset(seed=81)
    target = 0
    for ue in env.ues:
        ue.state = MobilityState.SINGLE
        ue.secondary_cell = None
        env.candidates[ue.ue_id] = [target]
        env.links[ue.ue_id][target].visible = True
    masks = env.action_masks()
    actions = np.full(env.num_agents, env.prepare_start, dtype=np.int64)

    snapshot = canonical_preparation_admission_snapshot(env, masks, actions)
    env.step(actions)

    assert snapshot.active_reservations[target] == 0
    assert snapshot.available_slots[target] == 2
    assert env.last_preparation_available_slots[target] == snapshot.available_slots[target]


def test_price_interventions_preserve_shapes_and_do_not_mutate_inputs() -> None:
    actor = np.arange(8, dtype=np.float64).reshape(2, 4)
    planner = np.arange(6, dtype=np.float64)
    actor_before = actor.copy()
    planner_before = planner.copy()

    outputs = {
        condition: intervention_price_inputs(actor, planner, condition)
        for condition in PRICE_INTERVENTIONS
    }

    np.testing.assert_array_equal(outputs["normal"][0], actor)
    np.testing.assert_array_equal(outputs["normal"][1], planner)
    np.testing.assert_array_equal(outputs["zero_actor_price"][0], 0.0)
    np.testing.assert_array_equal(outputs["zero_actor_price"][1], planner)
    np.testing.assert_array_equal(outputs["zero_planner_price"][0], actor)
    np.testing.assert_array_equal(outputs["zero_planner_price"][1], 0.0)
    np.testing.assert_array_equal(outputs["zero_actor_and_planner_price"][0], 0.0)
    np.testing.assert_array_equal(outputs["zero_actor_and_planner_price"][1], 0.0)
    np.testing.assert_array_equal(actor, actor_before)
    np.testing.assert_array_equal(planner, planner_before)


def test_prepared_logging_uses_exact_actor_and_planner_values() -> None:
    ue = SimpleNamespace(
        ue_id=0,
        state=MobilityState.PREPARED,
        primary_cell=0,
        secondary_cell=1,
    )
    env = SimpleNamespace(
        num_agents=1,
        action_dim=4,
        commit_action=2,
        release_action=3,
        ues=[ue],
        config={"coordination": {"price_max": 5.0}},
    )
    base = np.asarray([[0.0, 1.0, 2.0, 3.0]])
    reasoning = ReasoningResult(
        features=base.copy(),
        raw_costs=base + 10.0,
        predicted_deficits=base.copy(),
        best_scores=np.zeros(1),
        score_gaps=np.zeros(1),
        coordination_costs=base + 20.0,
        predicted_outages=base.copy(),
        planning_reliability_risks=base.copy(),
        predicted_control_costs=base + 30.0,
        predicted_handover_costs=base.copy(),
        predicted_preparation_costs=base.copy(),
        service_costs=base + 40.0,
        reliability_costs=base + 50.0,
        total_costs=base + 60.0,
    )
    actor_features = np.asarray(
        [[0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 0.9]]
    )
    logits = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    probabilities = np.asarray([[0.1, 0.0, 0.6, 0.3]])
    masks = np.asarray([[1.0, 0.0, 1.0, 1.0]])

    row = prepared_decision_row(
        env,  # type: ignore[arg-type]
        0,
        5001,
        7,
        np.asarray([2.0, 4.0]),
        masks,
        logits,
        probabilities,
        reasoning,
        actor_features,
        2,
    )

    assert row["prepared_price"] == 4.0
    assert row["actor_probability_commit"] == 0.6
    assert row["actor_logit_release"] == 4.0
    assert row["planner_raw_total_cost_commit"] == 12.0
    assert row["planner_coordination_cost_release"] == 23.0
    assert row["planner_control_cost_commit"] == 32.0
    assert row["reasoning_legacy_commit"] == 0.3
    assert row["reasoning_magnitude_release"] == 1.4
    assert row["selected_action_name"] == "COMMIT"


def test_rank_and_correlation_helpers_handle_ties() -> None:
    np.testing.assert_array_equal(
        average_ranks(np.asarray([0.0, 0.0, 2.0])),
        [1.5, 1.5, 3.0],
    )
    summary = correlation_summary([0.0, 1.0, 2.0], [2.0, 1.0, 0.0])
    assert summary["count"] == 3
    assert np.isclose(summary["pearson"], -1.0)
    assert np.isclose(summary["spearman"], -1.0)


def test_mechanism_diagnostic_is_frozen_and_uses_matched_seeds() -> None:
    source = inspect.getsource(evaluate_condition)
    assert "agent.update" not in source
    assert "agent.value" not in source
    assert EPISODE_SEEDS == tuple(range(5001, 5011))
