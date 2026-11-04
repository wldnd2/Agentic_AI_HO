from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .ue import MobilityState

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv
    from .planner import ReasoningResult


PRICE_INTERVENTIONS = {
    "normal": (False, False),
    "zero_actor_price": (True, False),
    "zero_planner_price": (False, True),
    "zero_actor_and_planner_price": (True, True),
}


@dataclass(frozen=True)
class PreparationAdmissionSnapshot:
    active_reservations: np.ndarray
    canonical_reservations_before_admission: np.ndarray
    selected_releases: np.ndarray
    maximum_slots: np.ndarray
    available_slots: np.ndarray
    candidate_ue_count: np.ndarray
    feasible_candidate_ue_count: np.ndarray


def intervention_price_inputs(
    actor_price_features: np.ndarray,
    planner_prices: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return copied diagnostic inputs without changing coordinator state."""
    if condition not in PRICE_INTERVENTIONS:
        raise ValueError(f"unknown price intervention: {condition}")
    zero_actor, zero_planner = PRICE_INTERVENTIONS[condition]
    actor = np.asarray(actor_price_features, dtype=np.float64)
    planner = np.asarray(planner_prices, dtype=np.float64)
    if actor.ndim != 2 or planner.ndim != 1:
        raise ValueError("price intervention inputs have invalid dimensions")
    return (
        np.zeros_like(actor) if zero_actor else actor.copy(),
        np.zeros_like(planner) if zero_planner else planner.copy(),
    )


def selected_preparation_targets(
    env: LEOMobilityEnv,
    masks: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Map valid selected PREP actions to absolute satellite IDs."""
    selected = np.full(env.num_agents, -1, dtype=np.int64)
    for ue_id, raw_action in enumerate(np.asarray(actions, dtype=np.int64)):
        action = int(raw_action)
        if not (
            0 <= action < env.action_dim
            and masks[ue_id, action] > 0.0
            and env.prepare_start
            <= action
            < env.prepare_start + env.candidate_count
        ):
            continue
        slot = action - env.prepare_start
        if slot < len(env.candidates[ue_id]):
            selected[ue_id] = int(env.candidates[ue_id][slot])
    return selected


def canonical_preparation_admission_snapshot(
    env: LEOMobilityEnv,
    masks: np.ndarray,
    actions: np.ndarray,
) -> PreparationAdmissionSnapshot:
    """Read-only replica of the V3 capacity calculation used by `_apply_actions`.

    The environment advances protocols before applying an action. A SWITCHING
    reservation with one remaining step and a PREPARED reservation whose link
    has disappeared therefore free capacity before admission. Valid RELEASE
    actions are then resolved before reservation occupancy is measured.
    """
    if not env.v3_enabled:
        raise ValueError("canonical admission diagnostics require V3")
    action_values = np.asarray(actions, dtype=np.int64)
    mask_values = np.asarray(masks, dtype=np.float64)
    if action_values.shape != (env.num_agents,):
        raise ValueError("actions have the wrong shape")
    if mask_values.shape != (env.num_agents, env.action_dim):
        raise ValueError("masks have the wrong shape")

    active = np.zeros(env.num_cells, dtype=np.int64)
    canonical = np.zeros(env.num_cells, dtype=np.int64)
    releases = np.zeros(env.num_cells, dtype=np.int64)
    for ue in env.ues:
        secondary = ue.secondary_cell
        reserves_now = secondary is not None and ue.state in (
            MobilityState.PREPARED,
            MobilityState.SWITCHING,
        )
        if reserves_now:
            active[int(secondary)] += 1

        survives_protocol_advance = False
        if ue.state == MobilityState.PREPARED and secondary is not None:
            survives_protocol_advance = bool(
                env.links[ue.ue_id][int(secondary)].visible
            )
        elif ue.state == MobilityState.SWITCHING and secondary is not None:
            survives_protocol_advance = int(ue.protocol_timer) > 1
        if not survives_protocol_advance:
            continue

        action = int(action_values[ue.ue_id])
        valid_release = (
            action == env.release_action
            and 0 <= action < env.action_dim
            and mask_values[ue.ue_id, action] > 0.0
        )
        if valid_release:
            releases[int(secondary)] += 1
        else:
            canonical[int(secondary)] += 1

    maximum = np.full(
        env.num_cells,
        int(
            env.preparation_admission_cfg.get(
                "max_concurrent_per_satellite", env.num_agents
            )
        ),
        dtype=np.int64,
    )
    rho = float(env.protocol_cfg["preparation_resource_fraction"])
    control_slots = np.maximum(maximum - canonical, 0)
    if rho <= 0.0:
        physical_slots = np.full(env.num_cells, env.num_agents, dtype=np.int64)
    else:
        reservation = canonical.astype(np.float64) * rho
        physical_slots = np.maximum(
            np.floor((1.0 + 1e-12 - reservation) / rho).astype(np.int64),
            0,
        )
    available = np.minimum(control_slots, physical_slots)

    candidate_count = np.zeros(env.num_cells, dtype=np.int64)
    feasible_count = np.zeros(env.num_cells, dtype=np.int64)
    for ue_id, candidates in enumerate(env.candidates):
        for slot, cell_id in enumerate(candidates[: env.candidate_count]):
            candidate_count[int(cell_id)] += 1
            action = env.prepare_start + slot
            if mask_values[ue_id, action] > 0.0:
                feasible_count[int(cell_id)] += 1

    return PreparationAdmissionSnapshot(
        active_reservations=active,
        canonical_reservations_before_admission=canonical,
        selected_releases=releases,
        maximum_slots=maximum,
        available_slots=available,
        candidate_ue_count=candidate_count,
        feasible_candidate_ue_count=feasible_count,
    )


def average_ranks(values: np.ndarray, *, ascending: bool = True) -> np.ndarray:
    """One-based average ranks with deterministic tie handling."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not np.all(np.isfinite(data)):
        raise ValueError("rank input must be a finite vector")
    ordered = np.argsort(data if ascending else -data, kind="stable")
    ranks = np.empty(len(data), dtype=np.float64)
    start = 0
    while start < len(data):
        end = start + 1
        while end < len(data) and data[ordered[end]] == data[ordered[start]]:
            end += 1
        ranks[ordered[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def correlation_summary(x: list[float], y: list[float]) -> dict[str, Any]:
    pairs = np.asarray(
        [(left, right) for left, right in zip(x, y) if np.isfinite(left) and np.isfinite(right)],
        dtype=np.float64,
    )
    if len(pairs) < 2:
        return {"count": int(len(pairs)), "pearson": None, "spearman": None}
    left, right = pairs[:, 0], pairs[:, 1]

    def coefficient(a: np.ndarray, b: np.ndarray) -> float | None:
        if np.std(a) <= 0.0 or np.std(b) <= 0.0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "count": int(len(pairs)),
        "pearson": coefficient(left, right),
        "spearman": coefficient(
            average_ranks(left),
            average_ranks(right),
        ),
    }


def distribution_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p05": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def prepared_decision_row(
    env: LEOMobilityEnv,
    ue_id: int,
    episode_seed: int,
    time_index: int,
    prices: np.ndarray,
    masks: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    reasoning: ReasoningResult,
    reasoning_features: np.ndarray,
    selected_action: int,
) -> dict[str, Any]:
    """Capture the exact actor and planner values for one PREPARED UE."""
    ue = env.ues[ue_id]
    if ue.state != MobilityState.PREPARED or ue.secondary_cell is None:
        raise ValueError("prepared decision row requires a PREPARED UE")
    action_dim = env.action_dim
    if reasoning_features.shape != (env.num_agents, 2 * action_dim + 1):
        raise ValueError("prepared diagnostics require confidence-aware features")
    action_items = (
        ("stay", 0),
        ("commit", env.commit_action),
        ("release", env.release_action),
    )
    feasible_names = [
        name.upper() for name, action in action_items if masks[ue_id, action] > 0.0
    ]
    row: dict[str, Any] = {
        "episode_seed": int(episode_seed),
        "time_index": int(time_index),
        "ue_id": int(ue_id),
        "serving_satellite": int(ue.primary_cell) if ue.primary_cell is not None else -1,
        "prepared_satellite": int(ue.secondary_cell),
        "serving_price": (
            float(prices[int(ue.primary_cell)]) if ue.primary_cell is not None else 0.0
        ),
        "prepared_price": float(prices[int(ue.secondary_cell)]),
        "serving_normalized_price": (
            float(prices[int(ue.primary_cell)] / env.config["coordination"]["price_max"])
            if ue.primary_cell is not None
            else 0.0
        ),
        "prepared_normalized_price": float(
            prices[int(ue.secondary_cell)] / env.config["coordination"]["price_max"]
        ),
        "feasible_actions": "|".join(feasible_names),
        "selected_action": int(selected_action),
        "selected_action_name": next(
            (name.upper() for name, action in action_items if action == selected_action),
            "OTHER",
        ),
        "reasoning_global_confidence": float(reasoning_features[ue_id, -1]),
    }
    delta_components = {
        "planner_service_cost": (
            reasoning.delta_service_cost
            if reasoning.delta_service_cost is not None
            else reasoning.service_costs
        ),
        "planner_control_cost": (
            reasoning.delta_control_cost
            if reasoning.delta_control_cost is not None
            else reasoning.predicted_control_costs
        ),
        "planner_reliability_cost": (
            reasoning.delta_reliability_cost
            if reasoning.delta_reliability_cost is not None
            else reasoning.reliability_costs
        ),
        "planner_coordination_cost": (
            reasoning.delta_coordination_cost
            if reasoning.delta_coordination_cost is not None
            else reasoning.coordination_costs
        ),
    }
    absolute_components = {
        "planner_raw_total_cost": reasoning.raw_costs,
        "planner_absolute_total_cost": reasoning.total_costs,
        "planner_absolute_service_cost": reasoning.service_costs,
        "planner_offset0_control_cost": reasoning.predicted_control_costs,
        "planner_absolute_reliability_cost": reasoning.reliability_costs,
        "planner_absolute_coordination_cost": reasoning.coordination_costs,
    }
    for name, action in action_items:
        row[f"feasible_{name}"] = float(masks[ue_id, action] > 0.0)
        row[f"actor_logit_{name}"] = float(logits[ue_id, action])
        row[f"actor_probability_{name}"] = float(probabilities[ue_id, action])
        for component_name, values in {
            **delta_components,
            **absolute_components,
        }.items():
            row[f"{component_name}_{name}"] = float(values[ue_id, action])
        row[f"reasoning_legacy_{name}"] = float(reasoning_features[ue_id, action])
        row[f"reasoning_magnitude_{name}"] = float(
            reasoning_features[ue_id, action_dim + action]
        )
        row[f"selected_{name}"] = float(selected_action == action)
    row["planner_raw_commit_minus_stay"] = (
        row["planner_raw_total_cost_commit"] - row["planner_raw_total_cost_stay"]
    )
    row["planner_raw_release_minus_stay"] = (
        row["planner_raw_total_cost_release"] - row["planner_raw_total_cost_stay"]
    )
    return row
