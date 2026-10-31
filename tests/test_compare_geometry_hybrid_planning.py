from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.reasoning import LookaheadReasoner
from leo_mappo.residual_world_model import FEATURE_DIM
from scripts.compare_geometry_hybrid_planning import (
    _action_difference_type,
    _action_transition_matrix,
    _best_second_margin,
    _divergence_vs_agreement,
    _exact_action_divergence,
    _ranking,
    _service_deficit_audit,
    build_summary,
    compare_current_state,
    inspect_then_step,
    write_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


def _summary_row(**overrides):
    row = {
        "episode": 0,
        "timestep": 0,
        "ue_id": 0,
        "protocol_state": "SINGLE",
        "geometry_preferred_action": 0,
        "geometry_preferred_category": "STAY",
        "geometry_preferred_target_cell_id": -1,
        "hybrid_preferred_action": 0,
        "hybrid_preferred_category": "STAY",
        "hybrid_preferred_target_cell_id": -1,
        "preferred_action_differs": 0,
        "action_difference_type": "identical_exact_action",
        "ranking_changed": 0,
        "history_ready": 1,
        "hybrid_active": 1,
        "geometry_decision_margin": 0.1,
        "hybrid_decision_margin": 0.1,
        "geometry_cost_of_hybrid_action_minus_best": 0.0,
        "hybrid_cost_of_geometry_action_minus_best": 0.0,
        "cross_selected_cost_penalty_sum": 0.0,
        "mean_abs_sinr_correction_db": 0.0,
        "max_abs_sinr_correction_db": 0.0,
        "geometry_preferred_predicted_deficit": 0.0,
        "hybrid_preferred_predicted_deficit": 0.0,
        "geometry_preferred_predicted_outage": 0.0,
        "hybrid_preferred_predicted_outage": 0.0,
        "geometry_preferred_planning_reliability_risk": 0.0,
        "hybrid_preferred_planning_reliability_risk": 0.0,
        "geometry_preferred_total_cost": 0.0,
        "hybrid_preferred_total_cost": 0.0,
        "geometry_feasible_total_costs": '{"0":1.0}',
        "hybrid_feasible_total_costs": '{"0":1.0}',
        "geometry_feasible_predicted_deficits": '{"0":0.0}',
        "hybrid_feasible_predicted_deficits": '{"0":0.0}',
        "geometry_feasible_predicted_outages": '{"0":0.0}',
        "hybrid_feasible_predicted_outages": '{"0":0.0}',
        "geometry_feasible_planning_reliability_risks": '{"0":0.0}',
        "hybrid_feasible_planning_reliability_risks": '{"0":0.0}',
        "geometry_feasible_coordination_costs": '{"0":0.0}',
        "hybrid_feasible_coordination_costs": '{"0":0.0}',
    }
    row.update(overrides)
    return row


class _FakeResidualPredictor:
    feature_dim = FEATURE_DIM

    def __init__(self, residuals: list[float], history_steps: int = 1) -> None:
        self.history_steps = history_steps
        self.horizon_steps = len(residuals)
        self.residuals = np.asarray(residuals, dtype=np.float64)

    def predict(self, histories: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            self.residuals,
            (histories.shape[0], self.horizon_steps),
        ).copy()


class _TinyStepEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.actions: list[np.ndarray] = []

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self.actions.append(np.asarray(actions).copy())
        self.step_index += 1
        return SimpleNamespace(done=False)


def _env() -> LEOMobilityEnv:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["num_ues"] = 1
    config["environment"]["episode_steps"] = 2
    env = LEOMobilityEnv(config)
    env.reset(seed=73)
    env.min_elevation_rad = -np.pi / 2.0
    env.observed_sinr[0, :] = -5.0
    return env


def _reasoners(
    env: LEOMobilityEnv,
    predictor: _FakeResidualPredictor,
) -> tuple[LookaheadReasoner, LookaheadReasoner]:
    reasoning_config = {"horizon_steps": predictor.horizon_steps}
    geometry_reasoner = LookaheadReasoner(reasoning_config, env.reward_cfg)
    hybrid_reasoner = LookaheadReasoner(reasoning_config, env.reward_cfg)
    hybrid_model = HybridWorldModel(hybrid_reasoner.world_model, predictor)
    hybrid_reasoner.world_model = hybrid_model
    hybrid_reasoner.planner.world_model = hybrid_model
    return geometry_reasoner, hybrid_reasoner


def test_both_inspections_happen_before_single_environment_step() -> None:
    env = _TinyStepEnv()
    observed_steps: list[tuple[str, int]] = []

    def inspection(current_env: _TinyStepEnv):
        observed_steps.append(("geometry", current_env.step_index))
        observed_steps.append(("hybrid", current_env.step_index))
        return [], []

    inspect_then_step(
        env,  # type: ignore[arg-type]
        inspection,  # type: ignore[arg-type]
        lambda _: np.asarray([0]),
    )
    assert observed_steps == [("geometry", 0), ("hybrid", 0)]
    assert env.step_index == 1
    assert len(env.actions) == 1


def test_behavior_action_is_independent_of_reasoner_recommendations() -> None:
    stepped_actions: list[int] = []
    for recommendations in ((0, 1), (3, 2)):
        env = _TinyStepEnv()

        def inspection(_: _TinyStepEnv):
            return ([{"recommendations": recommendations}], [])

        _, actions, _ = inspect_then_step(
            env,  # type: ignore[arg-type]
            inspection,  # type: ignore[arg-type]
            lambda _: np.asarray([2]),
        )
        stepped_actions.append(int(actions[0]))
    assert stepped_actions == [2, 2]


def test_zero_residual_has_equivalent_decisions_and_predictions() -> None:
    env = _env()
    geometry, hybrid = _reasoners(
        env, _FakeResidualPredictor([0.0, 0.0], history_steps=1)
    )
    decisions, predictions = compare_current_state(
        env,
        geometry,
        hybrid,
        episode=0,
        timestep=0,
        prices=np.zeros(env.num_cells),
        outage_dual_mu=0.0,
    )
    assert all(row["preferred_action_differs"] == 0 for row in decisions)
    assert all(row["ranking_changed"] == 0 for row in decisions)
    assert all(row["residual_correction_db"] == 0.0 for row in predictions)
    assert all(row["hybrid_active"] == 1 for row in predictions)


def test_known_residual_produces_prediction_difference() -> None:
    env = _env()
    geometry, hybrid = _reasoners(
        env, _FakeResidualPredictor([2.0, 2.0], history_steps=1)
    )
    _, predictions = compare_current_state(
        env,
        geometry,
        hybrid,
        episode=0,
        timestep=0,
        prices=np.zeros(env.num_cells),
        outage_dual_mu=0.0,
    )
    active = [row for row in predictions if row["hybrid_active"]]
    assert active
    assert all(np.isclose(row["residual_correction_db"], 2.0) for row in active)


def test_warmup_predictions_are_marked_separately() -> None:
    env = _env()
    geometry, hybrid = _reasoners(
        env, _FakeResidualPredictor([2.0, 2.0], history_steps=2)
    )
    decisions, predictions = compare_current_state(
        env,
        geometry,
        hybrid,
        episode=0,
        timestep=0,
        prices=np.zeros(env.num_cells),
        outage_dual_mu=0.0,
    )
    assert all(row["history_warmup"] == 1 for row in decisions)
    assert all(row["hybrid_active"] == 0 for row in decisions)
    assert all(row["prediction_mode"] == "history_warmup" for row in predictions)
    assert all(row["residual_correction_db"] == 0.0 for row in predictions)


def test_action_category_transition_matrix_counts_transitions() -> None:
    rows = [
        _summary_row(
            geometry_preferred_category="STAY",
            hybrid_preferred_category="PREP",
            preferred_action_differs=1,
        ),
        _summary_row(
            geometry_preferred_category="STAY",
            hybrid_preferred_category="STAY",
        ),
        _summary_row(
            geometry_preferred_category="PREP",
            hybrid_preferred_category="STAY",
            preferred_action_differs=1,
        ),
    ]
    matrix = _action_transition_matrix(rows)
    assert matrix["STAY"]["PREP"]["count"] == 1
    assert matrix["STAY"]["PREP"]["fraction_of_all_decision_points"] == 1 / 3
    assert matrix["STAY"]["STAY"]["fraction_within_geometry_category"] == 0.5
    assert matrix["PREP"]["STAY"]["count"] == 1


def test_prep_to_different_prep_is_same_category_different_exact_action() -> None:
    difference_type = _action_difference_type(1, 2, "PREP", "PREP")
    row = _summary_row(
        geometry_preferred_action=1,
        hybrid_preferred_action=2,
        geometry_preferred_category="PREP",
        hybrid_preferred_category="PREP",
        geometry_preferred_target_cell_id=3,
        hybrid_preferred_target_cell_id=4,
        preferred_action_differs=1,
        action_difference_type=difference_type,
    )
    metrics = _exact_action_divergence([row])
    assert difference_type == "same_category_different_exact_action"
    assert metrics["same_category_different_exact_action_count"] == 1
    assert metrics["same_PREP_category_different_candidate_count"] == 1
    assert metrics["same_PREP_category_different_candidate_rate"] == 1.0


def test_decision_margin_uses_best_and_second_best_feasible_costs() -> None:
    costs = np.asarray([0.4, 0.1, 0.3, 0.2])
    feasible = np.asarray([0, 1, 3])
    ranking = _ranking(costs, feasible)
    best, second, margin = _best_second_margin(costs, ranking)
    assert ranking == (1, 3, 0)
    assert best == 0.1
    assert second == 0.2
    assert np.isclose(margin, 0.1)


def test_service_deficit_audit_reports_zero_saturation() -> None:
    audit = _service_deficit_audit([_summary_row(), _summary_row()])
    all_points = audit["all_decision_points"]
    assert all_points["geometry"]["fraction_exactly_zero"] == 1.0
    assert all_points["hybrid"]["fraction_greater_than_zero"] == 0.0
    assert all_points["different_service_deficit_count"] == 0


def test_service_deficit_audit_reports_nonzero_distribution() -> None:
    rows = [
        _summary_row(
            geometry_preferred_predicted_deficit=0.1,
            hybrid_preferred_predicted_deficit=0.2,
        ),
        _summary_row(
            geometry_preferred_predicted_deficit=0.3,
            hybrid_preferred_predicted_deficit=0.3,
        ),
    ]
    audit = _service_deficit_audit(rows)["all_decision_points"]
    assert audit["geometry"]["fraction_greater_than_zero"] == 1.0
    assert np.isclose(audit["geometry"]["mean"], 0.2)
    assert audit["geometry"]["max"] == 0.3
    assert audit["different_service_deficit_count"] == 1


def test_divergence_vs_agreement_aggregates_prediction_differences() -> None:
    divergent = _summary_row(
        preferred_action_differs=1,
        geometry_preferred_predicted_outage=0.1,
        hybrid_preferred_predicted_outage=0.4,
        geometry_preferred_planning_reliability_risk=0.2,
        hybrid_preferred_planning_reliability_risk=0.5,
        geometry_preferred_total_cost=0.3,
        hybrid_preferred_total_cost=0.9,
        mean_abs_sinr_correction_db=2.0,
        max_abs_sinr_correction_db=3.0,
    )
    agreeing = _summary_row(mean_abs_sinr_correction_db=0.5)
    metrics = _divergence_vs_agreement([divergent, agreeing])[
        "all_decision_points"
    ]
    assert metrics["divergent"]["count"] == 1
    assert np.isclose(
        metrics["divergent"]["mean_absolute_predicted_outage_difference"],
        0.3,
    )
    assert np.isclose(
        metrics["divergent"]["mean_absolute_total_cost_difference"],
        0.6,
    )
    assert metrics["divergent"]["median_decision_mean_abs_sinr_correction_db"] == 2.0
    assert metrics["agreeing"]["count"] == 1


def test_tiny_diagnostic_records_write_csv_and_json(tmp_path: Path) -> None:
    decision = _summary_row(
        geometry_feasible_total_costs='{"0":1.0}',
        hybrid_feasible_total_costs='{"0":1.5}',
        geometry_feasible_predicted_deficits='{"0":0.2}',
        hybrid_feasible_predicted_deficits='{"0":0.1}',
        geometry_feasible_predicted_outages='{"0":0.3}',
        hybrid_feasible_predicted_outages='{"0":0.2}',
        geometry_feasible_planning_reliability_risks='{"0":0.4}',
        hybrid_feasible_planning_reliability_risks='{"0":0.3}',
    )
    prediction = {
        "offset": 1,
        "hybrid_active": 1,
        "residual_correction_db": 2.0,
    }
    diagnostics = {
        "hybrid_prediction_count": 1,
        "geometry_fallback_count": 0,
        "history_warmup_fallback_count": 0,
    }
    summary = build_summary([decision], [prediction], diagnostics, "max_sinr")
    write_outputs(tmp_path, [decision], [prediction], summary)
    assert (tmp_path / "decision_comparison.csv").is_file()
    assert (tmp_path / "prediction_comparison.csv").is_file()
    assert (tmp_path / "top_divergence_examples.csv").is_file()
    stored = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert stored["decision_metrics"]["total_ue_decision_points"] == 1
    assert stored["prediction_metrics"]["hybrid_active_predictions"]["count"] == 1
