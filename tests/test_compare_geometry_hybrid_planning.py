from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.demand import remaining_service_window_s
from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.reasoning import LookaheadReasoner
from leo_mappo.residual_world_model import FEATURE_DIM
from scripts.compare_geometry_hybrid_planning import (
    _action_difference_type,
    _action_transition_matrix,
    _best_second_margin,
    _classify_selected_continuation,
    _clipped_by_demand,
    _deadline_aware_diagnostic_required_rate,
    _divergence_vs_agreement,
    _exact_action_divergence,
    _geometry_service_decision_diagnostics,
    _immediate_prep_delay_analysis,
    _prepared_release_analysis,
    _ranking,
    _resolve_shared_outage_dual,
    _selected_continuation_analysis,
    _service_deficit_audit,
    build_summary,
    compare_current_state,
    inspect_then_step,
    parse_args,
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
        "geometry_selected_continuation": "pure_stay",
        "hybrid_selected_continuation": "pure_stay",
        "geometry_best_immediate_prep_cost": None,
        "hybrid_best_immediate_prep_cost": None,
        "geometry_selected_stay_cost": None,
        "hybrid_selected_stay_cost": None,
        "geometry_selected_stay_contains_delayed_prep": 0,
        "hybrid_selected_stay_contains_delayed_prep": 0,
        "geometry_immediate_prep_minus_stay_cost": None,
        "hybrid_immediate_prep_minus_stay_cost": None,
        "geometry_stay_reference_cost": None,
        "hybrid_stay_reference_cost": None,
        "geometry_release_cost": None,
        "hybrid_release_cost": None,
        "geometry_selected_release_contains_reprep": 0,
        "hybrid_selected_release_contains_reprep": 0,
        "geometry_selected_stay_contains_future_commit": 0,
        "hybrid_selected_stay_contains_future_commit": 0,
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


def _diagnostic_ue(
    *,
    queue_packets: int = 1000,
    packet_size_bits: int = 8000,
    mean_arrival_mbps: float = 1.0,
    deadline_s: float = 1.0,
    head_of_line_delay_s: float = 0.5,
) -> SimpleNamespace:
    return SimpleNamespace(
        queue_packets=queue_packets,
        packet_size_bits=packet_size_bits,
        mean_arrival_mbps=mean_arrival_mbps,
        deadline_s=deadline_s,
        head_of_line_delay_s=head_of_line_delay_s,
    )


def test_service_rate_clipping_classification() -> None:
    assert _clipped_by_demand(10.0, 10.0)
    assert _clipped_by_demand(10.0, 12.0)
    assert not _clipped_by_demand(10.0, 9.999)
    assert not _clipped_by_demand(0.0, 12.0)


def test_deadline_aware_diagnostic_demand_is_zero_for_empty_queue() -> None:
    ue = _diagnostic_ue(queue_packets=0)
    assert _deadline_aware_diagnostic_required_rate(ue, 0.1) == 0.0


def test_deadline_aware_diagnostic_demand_formula() -> None:
    ue = _diagnostic_ue()
    # queue_bits=8e6, service_window=0.6, arrival=1e6, cap=80e6
    assert np.isclose(
        _deadline_aware_diagnostic_required_rate(ue, 0.1),
        8e6 / 0.6,
    )


def test_remaining_service_window_at_deadline_is_one_interval() -> None:
    assert remaining_service_window_s(
        deadline_s=1.0,
        head_of_line_delay_s=1.0,
        interval_s=0.1,
    ) == 0.1


def test_window_one_interval_before_deadline_is_two_intervals() -> None:
    assert np.isclose(
        remaining_service_window_s(
            deadline_s=1.0,
            head_of_line_delay_s=0.9,
            interval_s=0.1,
        ),
        0.2,
    )


def test_remaining_service_window_at_zero_age_includes_next_interval() -> None:
    assert np.isclose(
        remaining_service_window_s(
            deadline_s=1.0,
            head_of_line_delay_s=0.0,
            interval_s=0.1,
        ),
        1.1,
    )


def test_deadline_aware_demand_increases_as_remaining_deadline_shrinks() -> None:
    early = _diagnostic_ue(
        queue_packets=100,
        head_of_line_delay_s=0.0,
    )
    urgent = _diagnostic_ue(
        queue_packets=100,
        head_of_line_delay_s=0.8,
    )
    early_rate = _deadline_aware_diagnostic_required_rate(early, 0.1)
    urgent_rate = _deadline_aware_diagnostic_required_rate(urgent, 0.1)
    assert urgent_rate > early_rate


def test_deadline_aware_demand_never_exceeds_one_step_queue_cap() -> None:
    ue = _diagnostic_ue(
        queue_packets=100,
        mean_arrival_mbps=1000.0,
        head_of_line_delay_s=0.99,
    )
    demand = _deadline_aware_diagnostic_required_rate(ue, 0.1)
    queue_cap = ue.queue_packets * ue.packet_size_bits / 0.1
    assert demand <= queue_cap


def test_geometry_service_diagnostics_do_not_change_planner_or_environment() -> None:
    env = _env()
    geometry, _ = _reasoners(
        env, _FakeResidualPredictor([0.0, 0.0], history_steps=1)
    )
    masks = env.action_masks()
    before_costs = geometry.analyze(
        env,
        masks,
        np.zeros(env.num_cells),
        0.0,
    ).raw_costs.copy()
    before_state = (
        env.step_index,
        env.time_s,
        env.observed_required_rates()[1].copy(),
        masks.copy(),
        tuple(
            (
                ue.primary_cell,
                ue.secondary_cell,
                ue.state,
                ue.protocol_timer,
                ue.queue_packets,
                ue.head_of_line_delay_s,
            )
            for ue in env.ues
        ),
    )

    diagnostics = _geometry_service_decision_diagnostics(env, geometry, masks)

    after_costs = geometry.analyze(
        env,
        masks,
        np.zeros(env.num_cells),
        0.0,
    ).raw_costs
    assert diagnostics
    assert np.array_equal(after_costs, before_costs)
    assert env.step_index == before_state[0]
    assert env.time_s == before_state[1]
    assert np.array_equal(env.observed_required_rates()[1], before_state[2])
    assert np.array_equal(env.action_masks(), before_state[3])
    assert tuple(
        (
            ue.primary_cell,
            ue.secondary_cell,
            ue.state,
            ue.protocol_timer,
            ue.queue_packets,
            ue.head_of_line_delay_s,
        )
        for ue in env.ues
    ) == before_state[4]


def test_selected_continuation_classification_rules() -> None:
    env = _env()
    stay = 0
    prep = env.prepare_start
    commit = env.commit_action
    release = env.release_action
    cases = (
        ("SINGLE", stay, (stay,) * 6, "pure_stay"),
        (
            "SINGLE",
            stay,
            (stay, prep, stay, commit, stay, stay),
            "stay_then_delayed_prep_commit",
        ),
        (
            "SINGLE",
            prep,
            (prep, release, stay, stay, stay, stay),
            "prep_then_release",
        ),
        (
            "PREPARED",
            stay,
            (stay, commit, stay, stay, stay, stay),
            "stay_then_future_commit",
        ),
        (
            "PREPARED",
            release,
            (release, prep, stay, commit, stay, stay),
            "release_then_reprep_commit",
        ),
    )
    for state, immediate_action, sequence, expected in cases:
        continuation, categories = _classify_selected_continuation(
            env, state, immediate_action, sequence
        )
        assert continuation == expected
        assert len(categories) == len(sequence)


def test_selected_continuation_summary_is_split_by_reasoner_and_state() -> None:
    rows = [
        _summary_row(
            geometry_selected_continuation="pure_stay",
            hybrid_selected_continuation="stay_then_delayed_prep_commit",
        ),
        _summary_row(
            protocol_state="PREPARED",
            geometry_preferred_category="RELEASE",
            hybrid_preferred_category="RELEASE",
            geometry_selected_continuation="pure_release",
            hybrid_selected_continuation="release_then_reprep_commit",
        ),
    ]
    analysis = _selected_continuation_analysis(rows)
    assert analysis["geometry"]["SINGLE"]["classes"]["pure_stay"]["count"] == 1
    assert (
        analysis["hybrid"]["SINGLE"]["classes"]
        ["stay_then_delayed_prep_commit"]["fraction_within_state"]
        == 1.0
    )
    release_class = analysis["hybrid"]["PREPARED"]["classes"][
        "release_then_reprep_commit"
    ]
    assert release_class["count"] == 1
    assert release_class["by_preferred_immediate_action_category"]["RELEASE"][
        "fraction_within_preferred_immediate_action_category"
    ] == 1.0


def test_immediate_prep_and_prepared_release_analyses() -> None:
    single = _summary_row(
        geometry_best_immediate_prep_cost=0.3,
        geometry_selected_stay_cost=0.2,
        geometry_selected_stay_contains_delayed_prep=1,
        geometry_immediate_prep_minus_stay_cost=0.1,
        hybrid_best_immediate_prep_cost=0.1,
        hybrid_selected_stay_cost=0.2,
        hybrid_selected_stay_contains_delayed_prep=0,
        hybrid_immediate_prep_minus_stay_cost=-0.1,
    )
    prepared = _summary_row(
        protocol_state="PREPARED",
        geometry_stay_reference_cost=0.3,
        geometry_release_cost=0.2,
        geometry_selected_release_contains_reprep=0,
        geometry_selected_stay_contains_future_commit=1,
        hybrid_stay_reference_cost=0.3,
        hybrid_release_cost=0.4,
        hybrid_selected_release_contains_reprep=1,
        hybrid_selected_stay_contains_future_commit=0,
    )
    prep_analysis = _immediate_prep_delay_analysis([single, prepared])
    assert prep_analysis["geometry"]["selected_stay_delayed_prep_count"] == 1
    assert prep_analysis["geometry"][
        "best_immediate_prep_minus_selected_stay_cost"
    ]["mean"] == 0.1
    assert prep_analysis["hybrid"][
        "best_immediate_prep_minus_selected_stay_cost"
    ]["fraction_negative"] == 1.0

    release_analysis = _prepared_release_analysis([single, prepared])
    assert release_analysis["geometry"]["selected_stay_future_commit_count"] == 1
    assert release_analysis["hybrid"]["selected_release_reprep_count"] == 1
    assert release_analysis["geometry"]["release_minus_stay_cost"]["mean"] < 0.0


def test_fixed_outage_dual_override_and_default_resolution() -> None:
    controller = SimpleNamespace(mu=0.4, mu_max=10.0)
    assert _resolve_shared_outage_dual(None, controller, False) == (
        0.0,
        "config/default",
    )
    assert _resolve_shared_outage_dual(None, controller, True) == (
        0.4,
        "config/default",
    )
    assert _resolve_shared_outage_dual(2.5, controller, False) == (
        2.5,
        "cli_fixed_override",
    )
    with pytest.raises(ValueError, match="configured mu_max"):
        _resolve_shared_outage_dual(10.1, controller, True)
    with pytest.raises(ValueError, match="configured mu_max"):
        _resolve_shared_outage_dual(-0.1, controller, True)


def test_outage_dual_cli_option_is_optional_and_accepts_fixed_value() -> None:
    required = [
        "compare_geometry_hybrid_planning.py",
        "--residual-model",
        "model.npz",
        "--output",
        "diagnostic-output",
    ]
    with patch("sys.argv", required):
        assert parse_args().outage_dual_mu is None
    with patch("sys.argv", [*required, "--outage-dual-mu", "1.25"]):
        assert parse_args().outage_dual_mu == 1.25


def test_same_outage_dual_is_supplied_to_both_reasoners() -> None:
    env = _env()
    geometry, hybrid = _reasoners(
        env, _FakeResidualPredictor([0.0, 0.0], history_steps=1)
    )
    shared_dual = 1.75
    with patch.object(geometry, "analyze", wraps=geometry.analyze) as geometry_analyze:
        with patch.object(hybrid, "analyze", wraps=hybrid.analyze) as hybrid_analyze:
            compare_current_state(
                env,
                geometry,
                hybrid,
                episode=0,
                timestep=0,
                prices=np.zeros(env.num_cells),
                outage_dual_mu=shared_dual,
            )
    assert geometry_analyze.call_args.args[3] == shared_dual
    assert hybrid_analyze.call_args.args[3] == shared_dual


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
    assert all(row["geometry_selected_continuation"] for row in decisions)
    assert all(row["hybrid_selected_continuation"] for row in decisions)
    assert all(row["geometry_selected_sequence"] for row in decisions)
    assert all(row["hybrid_selected_sequence"] for row in decisions)
    assert all(row["geometry_feasible_selected_continuations"] for row in decisions)
    assert all(row["hybrid_feasible_selected_continuations"] for row in decisions)
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
    assert stored["scientific_control"]["source"] == "config/default"
    assert stored["scientific_control"]["shared_outage_dual_mu"] == 0.0
    assert "selected_continuation_analysis" in stored
    assert "immediate_prep_delay_analysis" in stored
    assert "prepared_release_analysis" in stored
    assert "service_saturation_analysis" in stored
    assert "deadline_demand_analysis" in stored
    assert "capacity_sufficiency_analysis" in stored
    assert "prep_conditioning_analysis" in stored
