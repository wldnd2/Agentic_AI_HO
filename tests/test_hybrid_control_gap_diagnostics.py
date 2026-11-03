from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.predictive_control import (
    predictive_controller_step,
    select_greedy_feasible_actions,
)
from scripts.compare_geometry_hybrid_planning import (
    _preferred_metrics,
    _ranking,
    build_divergent_decision_rows,
    divergent_decision_summary,
    write_outputs,
)
from scripts.evaluate_geometry_planner_closed_loop import (
    EpisodeAccumulator,
    analyze_preparation_contention,
    build_geometry_reasoner,
    prep_rejection_rate,
)


ROOT = Path(__file__).resolve().parents[1]


def _contention_env() -> SimpleNamespace:
    return SimpleNamespace(
        num_cells=5,
        prepare_start=1,
        candidate_count=1,
        candidates=[[2], [2], [3], [2]],
    )


def _divergent_row(**overrides) -> dict:
    row = {
        "episode_seed": 5001,
        "episode": 0,
        "timestep": 7,
        "ue_id": 2,
        "protocol_state": "SINGLE",
        "preferred_action_differs": 1,
        "geometry_preferred_action": 1,
        "geometry_preferred_category": "PREP",
        "geometry_preferred_target_cell_id": 3,
        "geometry_preferred_raw_cost": -0.2,
        "geometry_second_best_cost": -0.1,
        "geometry_decision_margin": 0.1,
        "hybrid_preferred_action": 2,
        "hybrid_preferred_category": "PREP",
        "hybrid_preferred_target_cell_id": 4,
        "hybrid_preferred_raw_cost": -0.3,
        "hybrid_second_best_cost": -0.29999,
        "hybrid_decision_margin": 1e-5,
        "current_required_rate_bps": 2.5e6,
        "queue_packets": 12,
        "hol_delay_s": 0.4,
        "serving_cell_id": 1,
        "observed_serving_load": 0.7,
    }
    for model in ("geometry", "hybrid"):
        row[f"{model}_selected_target_observed_load"] = 0.8
        row[f"{model}_selected_target_remaining_visibility_s"] = 120.0
        for component, value in (
            ("service", -0.2),
            ("reliability", 0.0),
            ("control", 0.01),
            ("coordination", 0.0),
        ):
            row[
                f"{model}_preferred_counterfactual_{component}_contribution"
            ] = value
    row.update(overrides)
    return row


def test_contention_analysis_does_not_modify_selected_actions() -> None:
    env = _contention_env()
    actions = np.asarray([1, 1, 1, 1], dtype=np.int64)
    before = actions.copy()
    analyze_preparation_contention(env, actions)  # type: ignore[arg-type]
    np.testing.assert_array_equal(actions, before)


def test_same_target_unique_target_and_collision_fraction_counting() -> None:
    snapshot = analyze_preparation_contention(
        _contention_env(),  # type: ignore[arg-type]
        np.asarray([1, 1, 1, 1], dtype=np.int64),
    )
    assert snapshot.total_requests == 4
    assert snapshot.unique_targets == 2
    assert snapshot.max_same_target_requests == 3
    assert snapshot.collision_target_count == 1
    assert snapshot.collision_request_count == 3

    accumulator = EpisodeAccumulator(num_agents=4, interval_s=1.0)
    accumulator.steps = 1
    accumulator.record_preparation_contention(snapshot, None)
    metrics = accumulator.preparation_contention_metrics()
    assert metrics["fraction_prep_requests_in_same_target_collision"] == 0.75
    assert metrics["mean_unique_prep_targets_per_active_prep_epoch"] == 2.0


def test_rejection_rate_handles_zero_and_nonzero_denominators() -> None:
    assert prep_rejection_rate(3.0, 0.0) == 0.0
    assert prep_rejection_rate(3.0, 12.0) == 0.25


def test_canonical_admission_capacity_snapshot_drives_excess_metrics() -> None:
    snapshot = analyze_preparation_contention(
        _contention_env(),  # type: ignore[arg-type]
        np.asarray([1, 1, 1, 1], dtype=np.int64),
    )
    canonical_available = np.asarray([-1, -1, 2, 4, -1], dtype=np.int64)
    accumulator = EpisodeAccumulator(num_agents=4, interval_s=1.0)
    accumulator.steps = 1
    accumulator.record_preparation_contention(snapshot, canonical_available)
    metrics = accumulator.preparation_contention_metrics()
    assert metrics["prep_admission_capacity_observed_epochs"] == 1
    assert metrics["mean_excess_prep_requests"] == 1.0
    assert metrics["fraction_prep_requests_to_oversubscribed_targets"] == 0.75
    assert metrics["mean_oversubscribed_prep_targets_per_epoch"] == 1.0


def test_environment_exposes_canonical_v3_admission_slots() -> None:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["num_ues"] = 3
    config["environment"]["episode_steps"] = 2
    env = LEOMobilityEnv(config)
    env.reset(seed=43)
    serving_cells = {ue.primary_cell for ue in env.ues}
    target = next(
        cell_id
        for cell_id in range(env.num_cells)
        if cell_id not in serving_cells
    )
    env.candidates = [[target] for _ in env.ues]
    actions = np.full(env.num_agents, env.prepare_start, dtype=np.int64)
    env.step(actions)
    assert env.last_preparation_available_slots[target] == 2


def test_divergent_rows_and_prep_target_only_summary() -> None:
    row = _divergent_row()
    records = build_divergent_decision_rows([row])
    assert len(records) == 1
    assert records[0]["episode_seed"] == 5001
    assert records[0]["time_index"] == 7
    assert records[0]["prep_target_only_change"] == 1
    assert records[0]["geometry_service_contribution"] == -0.2

    summary = divergent_decision_summary([row])
    assert summary["divergent_decision_count"] == 1
    assert summary["divergent_decision_fraction"] == 1.0
    assert summary["category_change_count"] == 0
    assert summary["prep_target_only_change_count"] == 1
    assert summary["hybrid_decision_margins"]["fraction_le_1e-5"] == 1.0


def test_divergent_decisions_csv_is_written(tmp_path: Path) -> None:
    row = _divergent_row()
    with patch(
        "scripts.compare_geometry_hybrid_planning.build_top_divergence_examples",
        return_value=[],
    ):
        write_outputs(tmp_path, [row], [], {"diagnostic": True})
    path = tmp_path / "divergent_decisions.csv"
    assert path.is_file()
    with path.open(newline="", encoding="utf-8") as handle:
        stored = list(csv.DictReader(handle))
    assert len(stored) == 1
    assert stored[0]["geometry_selected_action_category"] == "PREP"
    assert stored[0]["hybrid_selected_prep_target_cell"] == "4"


def test_decision_margin_and_exact_tie_keep_lower_action_index() -> None:
    costs = np.asarray([[0.4, 0.1, 0.1, 0.2]])
    masks = np.ones_like(costs)
    actions = select_greedy_feasible_actions(costs, masks)
    assert actions.tolist() == [1]
    ranking = _ranking(costs[0], np.flatnonzero(masks[0] > 0.0))
    assert ranking[:2] == (1, 2)
    assert costs[0, ranking[1]] - costs[0, ranking[0]] == 0.0


def test_cost_decomposition_reporting_does_not_change_planner_costs() -> None:
    config = load_config(
        ROOT / "configs/default.yaml",
        [
            ROOT / "configs/validation_agentic_challenge_v3.yaml",
            ROOT / "configs/counterfactual_planner.yaml",
        ],
    )
    config["environment"]["num_ues"] = 1
    config["environment"]["episode_steps"] = 2
    env = LEOMobilityEnv(config)
    env.reset(seed=41)
    reasoner = build_geometry_reasoner(config, env)
    masks = env.action_masks()
    result = reasoner.analyze(env, masks)
    before = result.raw_costs.copy()
    feasible = np.flatnonzero(masks[0] > 0.0)
    action = _ranking(result.raw_costs[0], feasible)[0]
    metrics = _preferred_metrics(result, 0, action)
    np.testing.assert_array_equal(result.raw_costs, before)
    contribution_sum = sum(
        metrics[f"counterfactual_{name}_contribution"]
        for name in ("service", "reliability", "control", "coordination")
    )
    assert np.isclose(contribution_sum, result.delta_total_cost[0, action])


def test_closed_loop_controller_callback_preserves_direct_action_execution() -> None:
    class TinyEnvironment:
        def __init__(self) -> None:
            self.received: np.ndarray | None = None

        def action_masks(self) -> np.ndarray:
            return np.asarray([[1.0, 1.0, 1.0]])

        def step(self, actions: np.ndarray) -> SimpleNamespace:
            self.received = np.asarray(actions).copy()
            return SimpleNamespace(done=True)

    class TinyReasoner:
        def analyze(
            self, env: TinyEnvironment, masks: np.ndarray
        ) -> SimpleNamespace:
            return SimpleNamespace(raw_costs=np.asarray([[0.3, 0.1, 0.2]]))

    env = TinyEnvironment()
    captured: list[np.ndarray] = []

    def capture_without_control(
        _: TinyEnvironment, selected: np.ndarray
    ) -> None:
        captured.append(selected.copy())
        selected.fill(0)

    _, actions, _, _ = predictive_controller_step(  # type: ignore[arg-type]
        env,
        TinyReasoner(),  # type: ignore[arg-type]
        before_step=capture_without_control,  # type: ignore[arg-type]
    )
    assert actions.tolist() == [1]
    np.testing.assert_array_equal(captured[0], actions)
    np.testing.assert_array_equal(env.received, actions)
