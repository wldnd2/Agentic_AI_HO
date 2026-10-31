from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from leo_mappo.baselines import max_sinr_policy
from leo_mappo.config import load_config
from leo_mappo.coordinator import (
    ReliabilityDualController,
    SatellitePriceCoordinator,
)
from leo_mappo.environment import LEOMobilityEnv, StepOutput
from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.planner import ReasoningResult
from leo_mappo.reasoning import LookaheadReasoner


Inspection = tuple[list[dict[str, Any]], list[dict[str, Any]]]
BehaviorPolicy = Callable[[LEOMobilityEnv], np.ndarray]


def _action_category(env: LEOMobilityEnv, action: int) -> str:
    if action == 0:
        return "STAY"
    if env.prepare_start <= action < env.prepare_start + env.candidate_count:
        return "PREP"
    if action == env.commit_action:
        return "COMMIT"
    if action == env.release_action:
        return "RELEASE"
    raise ValueError(f"unknown action index {action}")


def _ranking(values: np.ndarray, feasible: np.ndarray) -> tuple[int, ...]:
    return tuple(
        sorted(
            (int(action) for action in feasible),
            key=lambda action: (float(values[action]), action),
        )
    )


def _action_values(values: np.ndarray, feasible: np.ndarray) -> str:
    return json.dumps(
        {str(int(action)): float(values[action]) for action in feasible},
        separators=(",", ":"),
        sort_keys=True,
    )


def _best_second_margin(
    values: np.ndarray,
    ranking: tuple[int, ...],
) -> tuple[float, float, float]:
    best = float(values[ranking[0]])
    second = float(values[ranking[1] if len(ranking) > 1 else ranking[0]])
    return best, second, second - best


def _preferred_target_cell(
    env: LEOMobilityEnv,
    ue_id: int,
    action: int,
) -> int:
    if env.prepare_start <= action < env.prepare_start + env.candidate_count:
        slot = action - env.prepare_start
        if slot < len(env.candidates[ue_id]):
            return int(env.candidates[ue_id][slot])
    return -1


def _action_difference_type(
    geometry_action: int,
    hybrid_action: int,
    geometry_category: str,
    hybrid_category: str,
) -> str:
    if geometry_action == hybrid_action:
        return "identical_exact_action"
    if geometry_category == hybrid_category:
        return "same_category_different_exact_action"
    return "different_action_category"


def _preferred_metrics(
    result: ReasoningResult,
    ue_id: int,
    action: int,
) -> dict[str, float]:
    return {
        "raw_cost": float(result.raw_costs[ue_id, action]),
        "total_cost": float(result.total_costs[ue_id, action]),
        "predicted_deficit": float(result.predicted_deficits[ue_id, action]),
        "predicted_outage": float(result.predicted_outages[ue_id, action]),
        "planning_reliability_risk": float(
            result.planning_reliability_risks[ue_id, action]
        ),
        "preparation_cost": float(
            result.predicted_preparation_costs[ue_id, action]
        ),
        "handover_cost": float(result.predicted_handover_costs[ue_id, action]),
        "control_cost": float(result.predicted_control_costs[ue_id, action]),
        "coordination_cost": float(result.coordination_costs[ue_id, action]),
        "service_cost": float(result.service_costs[ue_id, action]),
    }


def _decision_rows(
    env: LEOMobilityEnv,
    episode: int,
    timestep: int,
    masks: np.ndarray,
    geometry_result: ReasoningResult,
    hybrid_result: ReasoningResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vector_fields = {
        "raw_costs": "raw_costs",
        "total_costs": "total_costs",
        "predicted_deficits": "predicted_deficits",
        "predicted_outages": "predicted_outages",
        "planning_reliability_risks": "planning_reliability_risks",
        "coordination_costs": "coordination_costs",
    }
    for ue in env.ues:
        ue_id = ue.ue_id
        feasible = np.flatnonzero(masks[ue_id] > 0.0)
        geometry_ranking = _ranking(geometry_result.raw_costs[ue_id], feasible)
        hybrid_ranking = _ranking(hybrid_result.raw_costs[ue_id], feasible)
        geometry_action = geometry_ranking[0]
        hybrid_action = hybrid_ranking[0]
        geometry_category = _action_category(env, geometry_action)
        hybrid_category = _action_category(env, hybrid_action)
        (
            geometry_best_cost,
            geometry_second_best_cost,
            geometry_decision_margin,
        ) = _best_second_margin(
            geometry_result.raw_costs[ue_id], geometry_ranking
        )
        (
            hybrid_best_cost,
            hybrid_second_best_cost,
            hybrid_decision_margin,
        ) = _best_second_margin(
            hybrid_result.raw_costs[ue_id], hybrid_ranking
        )
        geometry_cross_penalty = float(
            geometry_result.raw_costs[ue_id, hybrid_action] - geometry_best_cost
        )
        hybrid_cross_penalty = float(
            hybrid_result.raw_costs[ue_id, geometry_action] - hybrid_best_cost
        )
        geometry_metrics = _preferred_metrics(
            geometry_result, ue_id, geometry_action
        )
        hybrid_metrics = _preferred_metrics(hybrid_result, ue_id, hybrid_action)
        row: dict[str, Any] = {
            "episode": episode,
            "timestep": timestep,
            "ue_id": ue_id,
            "protocol_state": ue.state.name,
            "feasible_action_mask": json.dumps(
                [int(value > 0.0) for value in masks[ue_id]], separators=(",", ":")
            ),
            "geometry_preferred_action": geometry_action,
            "geometry_preferred_category": geometry_category,
            "geometry_preferred_target_cell_id": _preferred_target_cell(
                env, ue_id, geometry_action
            ),
            "hybrid_preferred_action": hybrid_action,
            "hybrid_preferred_category": hybrid_category,
            "hybrid_preferred_target_cell_id": _preferred_target_cell(
                env, ue_id, hybrid_action
            ),
            "preferred_action_differs": int(geometry_action != hybrid_action),
            "action_difference_type": _action_difference_type(
                geometry_action,
                hybrid_action,
                geometry_category,
                hybrid_category,
            ),
            "ranking_changed": int(geometry_ranking != hybrid_ranking),
            "geometry_action_ranking": json.dumps(geometry_ranking),
            "hybrid_action_ranking": json.dumps(hybrid_ranking),
            "geometry_score_gap": float(geometry_result.score_gaps[ue_id]),
            "hybrid_score_gap": float(hybrid_result.score_gaps[ue_id]),
            "feasible_action_count": int(len(feasible)),
            "geometry_best_cost": geometry_best_cost,
            "geometry_second_best_cost": geometry_second_best_cost,
            "geometry_decision_margin": geometry_decision_margin,
            "hybrid_best_cost": hybrid_best_cost,
            "hybrid_second_best_cost": hybrid_second_best_cost,
            "hybrid_decision_margin": hybrid_decision_margin,
            "geometry_cost_of_hybrid_action_minus_best": geometry_cross_penalty,
            "hybrid_cost_of_geometry_action_minus_best": hybrid_cross_penalty,
            "cross_selected_cost_penalty_sum": (
                geometry_cross_penalty + hybrid_cross_penalty
            ),
            "history_ready": 0,
            "hybrid_active": 0,
            "history_warmup": 1,
            "hybrid_active_prediction_count": 0,
            "mean_abs_sinr_correction_db": 0.0,
            "max_abs_sinr_correction_db": 0.0,
        }
        for name, value in geometry_metrics.items():
            row[f"geometry_preferred_{name}"] = value
        for name, value in hybrid_metrics.items():
            row[f"hybrid_preferred_{name}"] = value
        for output_name, attribute_name in vector_fields.items():
            geometry_values = getattr(geometry_result, attribute_name)[ue_id]
            hybrid_values = getattr(hybrid_result, attribute_name)[ue_id]
            row[f"geometry_feasible_{output_name}"] = _action_values(
                geometry_values, feasible
            )
            row[f"hybrid_feasible_{output_name}"] = _action_values(
                hybrid_values, feasible
            )
        rows.append(row)
    return rows


def _prediction_rows(
    env: LEOMobilityEnv,
    episode: int,
    timestep: int,
    geometry_reasoner: LookaheadReasoner,
    hybrid_reasoner: LookaheadReasoner,
) -> list[dict[str, Any]]:
    hybrid_model = hybrid_reasoner.world_model
    if not isinstance(hybrid_model, HybridWorldModel):
        raise TypeError("hybrid reasoner must contain HybridWorldModel")
    geometry_model = geometry_reasoner.world_model
    maximum_offset = min(geometry_reasoner.horizon, hybrid_model.horizon_steps)
    rows: list[dict[str, Any]] = []
    for ue in env.ues:
        ue_id = ue.ue_id
        relevant_cells = set(env.candidates[ue_id])
        if ue.primary_cell is not None:
            relevant_cells.add(ue.primary_cell)
        for cell_id in sorted(relevant_cells):
            history_ready = (
                len(hybrid_model.history(ue_id, cell_id))
                >= hybrid_model.history_steps
            )
            for offset in range(1, maximum_offset + 1):
                geometry = geometry_model.predict_link(env, ue_id, cell_id, offset)
                hybrid = hybrid_model.predict_link(env, ue_id, cell_id, offset)
                hybrid_active = bool(hybrid_model.last_prediction_used_hybrid)
                if not history_ready:
                    mode = "history_warmup"
                elif hybrid_active:
                    mode = "hybrid"
                elif not geometry.visible:
                    mode = "invisible_geometry"
                else:
                    mode = "geometry_fallback"
                rows.append(
                    {
                        "episode": episode,
                        "timestep": timestep,
                        "ue_id": ue_id,
                        "cell_id": cell_id,
                        "is_serving": int(ue.primary_cell == cell_id),
                        "is_candidate": int(cell_id in env.candidates[ue_id]),
                        "offset": offset,
                        "geometry_sinr_db": geometry.sinr_db,
                        "hybrid_sinr_db": hybrid.sinr_db,
                        "residual_correction_db": hybrid.sinr_db - geometry.sinr_db,
                        "geometry_spectral_efficiency": geometry.spectral_efficiency,
                        "hybrid_spectral_efficiency": hybrid.spectral_efficiency,
                        "geometry_success_probability": geometry.success_probability,
                        "hybrid_success_probability": hybrid.success_probability,
                        "visible": int(geometry.visible),
                        "history_ready": int(history_ready),
                        "history_warmup": int(not history_ready),
                        "hybrid_active": int(hybrid_active),
                        "prediction_mode": mode,
                    }
                )
    return rows


def compare_current_state(
    env: LEOMobilityEnv,
    geometry_reasoner: LookaheadReasoner,
    hybrid_reasoner: LookaheadReasoner,
    episode: int,
    timestep: int,
    prices: np.ndarray,
    outage_dual_mu: float,
) -> Inspection:
    masks = env.action_masks()
    geometry_result = geometry_reasoner.analyze(
        env, masks, prices, outage_dual_mu
    )
    hybrid_result = hybrid_reasoner.analyze(env, masks, prices, outage_dual_mu)
    decisions = _decision_rows(
        env,
        episode,
        timestep,
        masks,
        geometry_result,
        hybrid_result,
    )
    predictions = _prediction_rows(
        env,
        episode,
        timestep,
        geometry_reasoner,
        hybrid_reasoner,
    )
    active_by_ue = {
        ue_id: any(
            bool(row["hybrid_active"])
            for row in predictions
            if int(row["ue_id"]) == ue_id
        )
        for ue_id in range(env.num_agents)
    }
    ready_by_ue = {
        ue_id: any(
            bool(row["history_ready"])
            for row in predictions
            if int(row["ue_id"]) == ue_id
        )
        for ue_id in range(env.num_agents)
    }
    for row in decisions:
        ue_id = int(row["ue_id"])
        ue_predictions = [
            prediction
            for prediction in predictions
            if int(prediction["ue_id"]) == ue_id
        ]
        absolute_corrections = np.asarray(
            [abs(float(prediction["residual_correction_db"])) for prediction in ue_predictions],
            dtype=np.float64,
        )
        row["hybrid_active"] = int(active_by_ue[ue_id])
        row["history_ready"] = int(ready_by_ue[ue_id])
        row["history_warmup"] = int(not ready_by_ue[ue_id])
        row["hybrid_active_prediction_count"] = sum(
            int(prediction["hybrid_active"]) for prediction in ue_predictions
        )
        row["mean_abs_sinr_correction_db"] = (
            float(np.mean(absolute_corrections))
            if absolute_corrections.size
            else 0.0
        )
        row["max_abs_sinr_correction_db"] = (
            float(np.max(absolute_corrections))
            if absolute_corrections.size
            else 0.0
        )
    return decisions, predictions


def inspect_then_step(
    env: LEOMobilityEnv,
    inspection: Callable[[LEOMobilityEnv], Inspection],
    behavior_policy: BehaviorPolicy,
) -> tuple[Inspection, np.ndarray, StepOutput]:
    """Inspect one shared state, then step once using only the behavior policy."""
    step_before = int(env.step_index)
    records = inspection(env)
    if env.step_index != step_before:
        raise RuntimeError("diagnostic inspection changed the environment timestep")
    actions = np.asarray(behavior_policy(env), dtype=np.int64)
    output = env.step(actions)
    return records, actions, output


def _random_feasible_policy(rng: np.random.Generator) -> BehaviorPolicy:
    def policy(env: LEOMobilityEnv) -> np.ndarray:
        return np.asarray(
            [rng.choice(np.flatnonzero(mask > 0.0)) for mask in env.action_masks()],
            dtype=np.int64,
        )

    return policy


def _mean_absolute_vector_difference(
    rows: list[dict[str, Any]],
    geometry_field: str,
    hybrid_field: str,
) -> float:
    differences: list[float] = []
    for row in rows:
        geometry = json.loads(row[geometry_field])
        hybrid = json.loads(row[hybrid_field])
        for action, value in geometry.items():
            differences.append(abs(float(value) - float(hybrid[action])))
    return float(np.mean(differences)) if differences else 0.0


def _correction_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    corrections = np.asarray(
        [float(row["residual_correction_db"]) for row in rows], dtype=np.float64
    )
    if corrections.size == 0:
        return {
            "count": 0,
            "mean_absolute_sinr_correction_db": 0.0,
            "median_absolute_sinr_correction_db": 0.0,
            "p95_absolute_sinr_correction_db": 0.0,
            "signed_mean_sinr_correction_db": 0.0,
        }
    absolute = np.abs(corrections)
    return {
        "count": int(corrections.size),
        "mean_absolute_sinr_correction_db": float(np.mean(absolute)),
        "median_absolute_sinr_correction_db": float(np.median(absolute)),
        "p95_absolute_sinr_correction_db": float(np.percentile(absolute, 95)),
        "signed_mean_sinr_correction_db": float(np.mean(corrections)),
    }


def _plan_cost_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_absolute_total_cost_difference": _mean_absolute_vector_difference(
            rows,
            "geometry_feasible_total_costs",
            "hybrid_feasible_total_costs",
        ),
        "mean_absolute_predicted_service_deficit_difference": (
            _mean_absolute_vector_difference(
                rows,
                "geometry_feasible_predicted_deficits",
                "hybrid_feasible_predicted_deficits",
            )
        ),
        "mean_absolute_predicted_outage_difference": (
            _mean_absolute_vector_difference(
                rows,
                "geometry_feasible_predicted_outages",
                "hybrid_feasible_predicted_outages",
            )
        ),
        "mean_absolute_planning_reliability_risk_difference": (
            _mean_absolute_vector_difference(
                rows,
                "geometry_feasible_planning_reliability_risks",
                "hybrid_feasible_planning_reliability_risks",
            )
        ),
        "mean_absolute_coordination_cost_difference": (
            _mean_absolute_vector_difference(
                rows,
                "geometry_feasible_coordination_costs",
                "hybrid_feasible_coordination_costs",
            )
        ),
    }


def _action_transition_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = ("STAY", "PREP", "COMMIT", "RELEASE")
    total = len(rows)
    matrix: dict[str, Any] = {}
    for geometry_category in categories:
        geometry_rows = [
            row
            for row in rows
            if row["geometry_preferred_category"] == geometry_category
        ]
        row_total = len(geometry_rows)
        matrix[geometry_category] = {}
        for hybrid_category in categories:
            count = sum(
                int(row["hybrid_preferred_category"] == hybrid_category)
                for row in geometry_rows
            )
            matrix[geometry_category][hybrid_category] = {
                "count": count,
                "fraction": count / total if total else 0.0,
                "fraction_of_all_decision_points": count / total if total else 0.0,
                "fraction_within_geometry_category": (
                    count / row_total if row_total else 0.0
                ),
            }
    return matrix


def _exact_action_divergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    identical = sum(
        int(row["action_difference_type"] == "identical_exact_action")
        for row in rows
    )
    same_category_different = sum(
        int(
            row["action_difference_type"]
            == "same_category_different_exact_action"
        )
        for row in rows
    )
    different_category = sum(
        int(row["action_difference_type"] == "different_action_category")
        for row in rows
    )
    both_prep = [
        row
        for row in rows
        if row["geometry_preferred_category"] == "PREP"
        and row["hybrid_preferred_category"] == "PREP"
    ]
    different_prep_candidate = sum(
        int(
            int(row["geometry_preferred_target_cell_id"])
            != int(row["hybrid_preferred_target_cell_id"])
        )
        for row in both_prep
    )

    def rate(count: int) -> float:
        return count / total if total else 0.0

    return {
        "total_decision_points": total,
        "identical_exact_action_count": identical,
        "identical_exact_action_rate": rate(identical),
        "same_category_different_exact_action_count": same_category_different,
        "same_category_different_exact_action_rate": rate(same_category_different),
        "different_action_category_count": different_category,
        "different_action_category_rate": rate(different_category),
        "both_prefer_PREP_count": len(both_prep),
        "same_PREP_category_different_candidate_count": different_prep_candidate,
        "same_PREP_category_different_candidate_rate": (
            different_prep_candidate / len(both_prep) if both_prep else 0.0
        ),
        "same_PREP_category_different_candidate_fraction_of_all": rate(
            different_prep_candidate
        ),
    }


_MARGIN_BINS = (
    ("<=1e-4", 1e-4),
    ("(1e-4,1e-3]", 1e-3),
    ("(1e-3,1e-2]", 1e-2),
    ("(1e-2,1e-1]", 1e-1),
    (">1e-1", float("inf")),
)


def _margin_bin(value: float) -> str:
    margin = max(0.0, float(value))
    for label, upper_bound in _MARGIN_BINS:
        if margin <= upper_bound:
            return label
    raise RuntimeError("decision margin did not match a fixed bin")


def _margin_bins_for_field(
    rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, _ in _MARGIN_BINS:
        selected = [row for row in rows if _margin_bin(float(row[field])) == label]
        divergent = sum(int(row["preferred_action_differs"]) for row in selected)
        result[label] = {
            "count": len(selected),
            "different_preferred_action": divergent,
            "divergence_rate": divergent / len(selected) if selected else 0.0,
        }
    return result


def _decision_margin_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    divergent = [row for row in rows if bool(row["preferred_action_differs"])]

    def mean_field(field: str) -> float:
        return (
            float(np.mean([float(row[field]) for row in divergent]))
            if divergent
            else 0.0
        )

    return {
        "margin_definition": (
            "second-lowest feasible raw_cost minus lowest feasible raw_cost; "
            "single-feasible-action points use margin 0"
        ),
        "fixed_margin_bins": [label for label, _ in _MARGIN_BINS],
        "geometry_margin_bins": _margin_bins_for_field(
            rows, "geometry_decision_margin"
        ),
        "hybrid_margin_bins": _margin_bins_for_field(
            rows, "hybrid_decision_margin"
        ),
        "divergent_point_count": len(divergent),
        "divergent_mean_geometry_decision_margin": mean_field(
            "geometry_decision_margin"
        ),
        "divergent_mean_hybrid_decision_margin": mean_field(
            "hybrid_decision_margin"
        ),
        "divergent_mean_geometry_cross_selected_cost_penalty": mean_field(
            "geometry_cost_of_hybrid_action_minus_best"
        ),
        "divergent_mean_hybrid_cross_selected_cost_penalty": mean_field(
            "hybrid_cost_of_geometry_action_minus_best"
        ),
    }


_SERVICE_DEFICIT_TOLERANCE = 1e-12


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "fraction_exactly_zero": 0.0,
            "fraction_greater_than_zero": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "fraction_exactly_zero": float(np.mean(array == 0.0)),
        "fraction_greater_than_zero": float(np.mean(array > 0.0)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _service_deficit_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    geometry = [
        float(row["geometry_preferred_predicted_deficit"]) for row in rows
    ]
    hybrid = [float(row["hybrid_preferred_predicted_deficit"]) for row in rows]
    unequal = sum(
        int(
            not np.isclose(
                geometry_value,
                hybrid_value,
                atol=_SERVICE_DEFICIT_TOLERANCE,
                rtol=1e-9,
            )
        )
        for geometry_value, hybrid_value in zip(geometry, hybrid)
    )
    return {
        "geometry": _distribution(geometry),
        "hybrid": _distribution(hybrid),
        "different_service_deficit_count": unequal,
        "different_service_deficit_rate": unequal / len(rows) if rows else 0.0,
        "comparison_absolute_tolerance": _SERVICE_DEFICIT_TOLERANCE,
        "comparison_relative_tolerance": 1e-9,
    }


def _service_deficit_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    post_warmup = [row for row in rows if bool(row["history_ready"])]
    return {
        "value_definition": (
            "predicted_deficits at each reasoner's own preferred feasible action"
        ),
        "all_decision_points": _service_deficit_subset(rows),
        "post_warmup_decision_points": _service_deficit_subset(post_warmup),
    }


def _difference_and_correction_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    def mean_absolute_difference(geometry_field: str, hybrid_field: str) -> float:
        return (
            float(
                np.mean(
                    [
                        abs(float(row[geometry_field]) - float(row[hybrid_field]))
                        for row in rows
                    ]
                )
            )
            if rows
            else 0.0
        )

    corrections = [float(row["mean_abs_sinr_correction_db"]) for row in rows]
    maxima = [float(row["max_abs_sinr_correction_db"]) for row in rows]
    correction_array = np.asarray(corrections, dtype=np.float64)
    return {
        "count": len(rows),
        "mean_absolute_predicted_outage_difference": mean_absolute_difference(
            "geometry_preferred_predicted_outage",
            "hybrid_preferred_predicted_outage",
        ),
        "mean_absolute_planning_reliability_risk_difference": (
            mean_absolute_difference(
                "geometry_preferred_planning_reliability_risk",
                "hybrid_preferred_planning_reliability_risk",
            )
        ),
        "mean_absolute_total_cost_difference": mean_absolute_difference(
            "geometry_preferred_total_cost",
            "hybrid_preferred_total_cost",
        ),
        "mean_decision_mean_abs_sinr_correction_db": (
            float(np.mean(correction_array)) if correction_array.size else 0.0
        ),
        "median_decision_mean_abs_sinr_correction_db": (
            float(np.median(correction_array)) if correction_array.size else 0.0
        ),
        "p95_decision_mean_abs_sinr_correction_db": (
            float(np.percentile(correction_array, 95))
            if correction_array.size
            else 0.0
        ),
        "mean_decision_max_abs_sinr_correction_db": (
            float(np.mean(maxima)) if maxima else 0.0
        ),
    }


def _divergence_vs_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    divergent = [row for row in rows if bool(row["preferred_action_differs"])]
    agreeing = [row for row in rows if not bool(row["preferred_action_differs"])]
    post_warmup_divergent = [row for row in divergent if bool(row["history_ready"])]
    post_warmup_agreeing = [row for row in agreeing if bool(row["history_ready"])]
    return {
        "difference_definition": (
            "absolute difference between quantities at each reasoner's own "
            "preferred feasible action"
        ),
        "correction_definition": (
            "distribution of per-decision mean absolute SINR correction across "
            "already-recorded serving/candidate/horizon prediction rows"
        ),
        "all_decision_points": {
            "divergent": _difference_and_correction_metrics(divergent),
            "agreeing": _difference_and_correction_metrics(agreeing),
        },
        "post_warmup_decision_points": {
            "divergent": _difference_and_correction_metrics(
                post_warmup_divergent
            ),
            "agreeing": _difference_and_correction_metrics(post_warmup_agreeing),
        },
    }


def _prep_specific_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def geometry_prep_outcome(row: dict[str, Any]) -> str:
        hybrid_category = str(row["hybrid_preferred_category"])
        if hybrid_category != "PREP":
            return f"PREP->{hybrid_category}"
        if (
            int(row["geometry_preferred_target_cell_id"])
            == int(row["hybrid_preferred_target_cell_id"])
        ):
            return "PREP->PREP_same_candidate"
        return "PREP->PREP_different_candidate"

    def hybrid_prep_source(row: dict[str, Any]) -> str:
        geometry_category = str(row["geometry_preferred_category"])
        if geometry_category != "PREP":
            return f"{geometry_category}->PREP"
        if (
            int(row["geometry_preferred_target_cell_id"])
            == int(row["hybrid_preferred_target_cell_id"])
        ):
            return "PREP_same_candidate->PREP"
        return "PREP_different_candidate->PREP"

    def counts(selected: list[dict[str, Any]], classifier) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for row in selected:
            label = classifier(row)
            output.setdefault(label, {"count": 0, "fraction": 0.0})
            output[label]["count"] += 1
        for values in output.values():
            values["fraction"] = values["count"] / len(selected) if selected else 0.0
        return output

    geometry_prep = [
        row for row in rows if row["geometry_preferred_category"] == "PREP"
    ]
    hybrid_prep = [
        row for row in rows if row["hybrid_preferred_category"] == "PREP"
    ]
    return {
        "geometry_prefers_PREP_count": len(geometry_prep),
        "when_geometry_prefers_PREP": counts(
            geometry_prep, geometry_prep_outcome
        ),
        "hybrid_prefers_PREP_count": len(hybrid_prep),
        "when_hybrid_prefers_PREP": counts(hybrid_prep, hybrid_prep_source),
    }


def build_top_divergence_examples(
    rows: list[dict[str, Any]], limit: int = 100
) -> list[dict[str, Any]]:
    divergent = [row for row in rows if bool(row["preferred_action_differs"])]
    ordered = sorted(
        divergent,
        key=lambda row: (
            -float(row["cross_selected_cost_penalty_sum"]),
            -float(row["max_abs_sinr_correction_db"]),
            int(row["episode"]),
            int(row["timestep"]),
            int(row["ue_id"]),
        ),
    )
    examples: list[dict[str, Any]] = []
    for row in ordered[:limit]:
        geometry_total = float(row["geometry_preferred_total_cost"])
        hybrid_total = float(row["hybrid_preferred_total_cost"])
        geometry_outage = float(row["geometry_preferred_predicted_outage"])
        hybrid_outage = float(row["hybrid_preferred_predicted_outage"])
        geometry_risk = float(
            row["geometry_preferred_planning_reliability_risk"]
        )
        hybrid_risk = float(row["hybrid_preferred_planning_reliability_risk"])
        examples.append(
            {
                "episode": row["episode"],
                "timestep": row["timestep"],
                "ue_id": row["ue_id"],
                "protocol_state": row["protocol_state"],
                "geometry_preferred_action": row["geometry_preferred_action"],
                "geometry_preferred_category": row[
                    "geometry_preferred_category"
                ],
                "geometry_preferred_target_cell_id": row[
                    "geometry_preferred_target_cell_id"
                ],
                "hybrid_preferred_action": row["hybrid_preferred_action"],
                "hybrid_preferred_category": row["hybrid_preferred_category"],
                "hybrid_preferred_target_cell_id": row[
                    "hybrid_preferred_target_cell_id"
                ],
                "action_difference_type": row["action_difference_type"],
                "geometry_decision_margin": row["geometry_decision_margin"],
                "hybrid_decision_margin": row["hybrid_decision_margin"],
                "geometry_cost_of_hybrid_action_minus_best": row[
                    "geometry_cost_of_hybrid_action_minus_best"
                ],
                "hybrid_cost_of_geometry_action_minus_best": row[
                    "hybrid_cost_of_geometry_action_minus_best"
                ],
                "cross_selected_cost_penalty_sum": row[
                    "cross_selected_cost_penalty_sum"
                ],
                "mean_abs_sinr_correction_db": row[
                    "mean_abs_sinr_correction_db"
                ],
                "max_abs_sinr_correction_db": row[
                    "max_abs_sinr_correction_db"
                ],
                "signed_preferred_total_cost_difference": (
                    hybrid_total - geometry_total
                ),
                "absolute_preferred_total_cost_difference": abs(
                    hybrid_total - geometry_total
                ),
                "absolute_preferred_outage_difference": abs(
                    hybrid_outage - geometry_outage
                ),
                "absolute_preferred_reliability_risk_difference": abs(
                    hybrid_risk - geometry_risk
                ),
            }
        )
    return examples


def build_summary(
    decision_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    diagnostics: dict[str, int],
    policy: str,
    coordination_price_mode: str = "shared zero vector",
    outage_dual_mu: float = 0.0,
) -> dict[str, Any]:
    total = len(decision_rows)
    post_warmup = [row for row in decision_rows if bool(row["history_ready"])]
    active_decisions = [row for row in decision_rows if bool(row["hybrid_active"])]
    divergent = [row for row in decision_rows if bool(row["preferred_action_differs"])]
    post_warmup_divergent = [
        row for row in post_warmup if bool(row["preferred_action_differs"])
    ]
    active_divergent = [
        row for row in active_decisions if bool(row["preferred_action_differs"])
    ]

    def divergence_group(field: str, values: tuple[str, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for value in values:
            selected = [row for row in decision_rows if row[field] == value]
            count = len(selected)
            different = sum(
                int(row["preferred_action_differs"]) for row in selected
            )
            result[value] = {
                "decision_points": count,
                "different_preferred_action": different,
                "divergence_rate": different / count if count else 0.0,
            }
        return result

    active_predictions = [
        row for row in prediction_rows if bool(row["hybrid_active"])
    ]
    per_horizon: dict[str, Any] = {}
    for offset in sorted({int(row["offset"]) for row in prediction_rows}):
        selected = [
            row
            for row in active_predictions
            if int(row["offset"]) == offset
        ]
        per_horizon[str(offset)] = _correction_metrics(selected)

    same_top_changed_ranking = sum(
        int(not row["preferred_action_differs"] and row["ranking_changed"])
        for row in decision_rows
    )
    return {
        "scientific_control": {
            "single_environment_per_episode": True,
            "behavior_policy": policy,
            "reasoner_recommendations_control_environment": False,
            "shared_coordination_prices": coordination_price_mode,
            "shared_outage_dual_mu": outage_dual_mu,
        },
        "decision_metrics": {
            "total_ue_decision_points": total,
            "post_warmup_decision_points": len(post_warmup),
            "hybrid_active_decision_points": len(active_decisions),
            "different_preferred_action": len(divergent),
            "action_recommendation_divergence_rate": (
                len(divergent) / total if total else 0.0
            ),
            "post_warmup_different_preferred_action": len(
                post_warmup_divergent
            ),
            "post_warmup_divergence_rate": (
                len(post_warmup_divergent) / len(post_warmup)
                if post_warmup
                else 0.0
            ),
            "hybrid_active_different_preferred_action": len(active_divergent),
            "hybrid_active_divergence_rate": (
                len(active_divergent) / len(active_decisions)
                if active_decisions
                else 0.0
            ),
            "divergence_by_protocol_state": divergence_group(
                "protocol_state",
                ("SINGLE", "PREPARED", "SWITCHING", "COOLDOWN"),
            ),
            "divergence_by_geometry_action_category": divergence_group(
                "geometry_preferred_category",
                ("STAY", "PREP", "COMMIT", "RELEASE"),
            ),
            "same_preferred_action_but_ranking_changed_count": (
                same_top_changed_ranking
            ),
            "same_preferred_action_but_ranking_changed_rate": (
                same_top_changed_ranking / total if total else 0.0
            ),
        },
        "prediction_metrics": {
            "all_predictions": _correction_metrics(prediction_rows),
            "hybrid_active_predictions": _correction_metrics(active_predictions),
            "per_horizon_hybrid_active": per_horizon,
        },
        "plan_cost_metrics": {
            "all_decision_points": _plan_cost_metrics(decision_rows),
            "post_warmup_decision_points": _plan_cost_metrics(post_warmup),
            "hybrid_active_decision_points": _plan_cost_metrics(active_decisions),
        },
        "action_transition_matrix": _action_transition_matrix(decision_rows),
        "exact_action_divergence": _exact_action_divergence(decision_rows),
        "decision_margin_analysis": _decision_margin_analysis(decision_rows),
        "service_deficit_audit": _service_deficit_audit(decision_rows),
        "divergence_vs_agreement": _divergence_vs_agreement(decision_rows),
        "prep_specific_analysis": _prep_specific_analysis(decision_rows),
        "top_divergence_examples": {
            "maximum_rows": 100,
            "row_count": min(
                100,
                sum(
                    int(row["preferred_action_differs"])
                    for row in decision_rows
                ),
            ),
            "sort_criterion": (
                "descending cross_selected_cost_penalty_sum, then descending "
                "max_abs_sinr_correction_db"
            ),
        },
        "hybrid_world_model_diagnostics": dict(diagnostics),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output: str | Path,
    decision_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "decision_comparison.csv", decision_rows)
    _write_csv(destination / "prediction_comparison.csv", prediction_rows)
    _write_csv(
        destination / "top_divergence_examples.csv",
        build_top_divergence_examples(decision_rows),
    )
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _print_summary(summary: dict[str, Any]) -> None:
    decisions = summary["decision_metrics"]
    predictions = summary["prediction_metrics"]["hybrid_active_predictions"]
    diagnostics = summary["hybrid_world_model_diagnostics"]
    print(f"decision points: {decisions['total_ue_decision_points']}")
    print(f"post-warmup points: {decisions['post_warmup_decision_points']}")
    print(
        "different preferred action: "
        f"{decisions['different_preferred_action']} "
        f"({100.0 * decisions['action_recommendation_divergence_rate']:.2f}%)"
    )
    print(
        "post-warmup mean |SINR correction|: "
        f"{predictions['mean_absolute_sinr_correction_db']:.6f} dB"
    )
    for offset, values in summary["prediction_metrics"][
        "per_horizon_hybrid_active"
    ].items():
        print(
            f"h={offset} mean |correction|: "
            f"{values['mean_absolute_sinr_correction_db']:.6f} dB "
            f"(n={values['count']})"
        )
    print(f"hybrid_prediction_count: {diagnostics['hybrid_prediction_count']}")
    print(f"geometry_fallback_count: {diagnostics['geometry_fallback_count']}")
    print(
        "history_warmup_fallback_count: "
        f"{diagnostics['history_warmup_fallback_count']}"
    )


def run(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    config = load_config(args.config, args.override)
    config["environment_seed"] = int(args.seed)
    env = LEOMobilityEnv(config)
    reasoning_config = dict(config.get("reasoning", {}))
    geometry_config = {**reasoning_config, "world_model": "geometry"}
    hybrid_config = {
        **reasoning_config,
        "world_model": "hybrid",
        "residual_model_path": args.residual_model,
    }
    mappo_config = config.get("mappo", {})
    use_coordination = bool(mappo_config.get("use_coordination", False))
    use_constraint = bool(mappo_config.get("use_constraint", False))
    price_coordinator = (
        SatellitePriceCoordinator(env.num_cells, config.get("coordination", {}))
        if use_coordination
        else None
    )
    reliability_controller = ReliabilityDualController(
        config.get("constraint", {})
    )
    geometry_reasoner = LookaheadReasoner(
        geometry_config,
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=use_coordination,
        use_constraint=use_constraint,
    )
    hybrid_reasoner = LookaheadReasoner(
        hybrid_config,
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=use_coordination,
        use_constraint=use_constraint,
    )
    if not isinstance(hybrid_reasoner.world_model, HybridWorldModel):
        raise RuntimeError("hybrid reasoner construction failed")

    behavior_rng = np.random.default_rng(args.seed)
    behavior_policy: BehaviorPolicy = (
        max_sinr_policy
        if args.policy == "max_sinr"
        else _random_feasible_policy(behavior_rng)
    )
    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    shared_outage_dual = reliability_controller.mu if use_constraint else 0.0
    for episode in range(args.episodes):
        env.reset(seed=args.seed + episode)
        if price_coordinator is not None:
            price_coordinator.reset()
        done = False
        while not done:
            timestep = int(env.step_index)
            shared_prices = (
                price_coordinator.prices
                if price_coordinator is not None
                else np.zeros(env.num_cells, dtype=np.float64)
            )

            def inspection(current_env: LEOMobilityEnv) -> Inspection:
                return compare_current_state(
                    current_env,
                    geometry_reasoner,
                    hybrid_reasoner,
                    episode,
                    timestep,
                    shared_prices,
                    shared_outage_dual,
                )

            (
                (step_decisions, step_predictions),
                behavior_actions,
                step_output,
            ) = inspect_then_step(
                env,
                inspection,
                behavior_policy,
            )
            for row in step_decisions:
                ue_id = int(row["ue_id"])
                behavior_action = int(behavior_actions[ue_id])
                row["behavior_action"] = behavior_action
                row["behavior_action_category"] = _action_category(
                    env, behavior_action
                )
            decisions.extend(step_decisions)
            predictions.extend(step_predictions)
            if price_coordinator is not None:
                if step_output.requested_pressure is None:
                    raise RuntimeError(
                        "environment did not return requested pressure for coordination"
                    )
                price_coordinator.update(step_output.requested_pressure)
            done = step_output.done

    summary = build_summary(
        decisions,
        predictions,
        hybrid_reasoner.world_model.diagnostics,
        args.policy,
        coordination_price_mode=(
            "shared behavior-trajectory SatellitePriceCoordinator"
            if price_coordinator is not None
            else "shared zero vector"
        ),
        outage_dual_mu=shared_outage_dual,
    )
    write_outputs(args.output, decisions, predictions, summary)
    _print_summary(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare geometry and hybrid planning on one shared trajectory."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="YAML override path; repeat to apply multiple overrides in order.",
    )
    parser.add_argument("--residual-model", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy", choices=("max_sinr", "random_feasible"), default="max_sinr"
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
