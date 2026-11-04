from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO
from leo_mappo.coordination_diagnostics import (
    PRICE_INTERVENTIONS,
    average_ranks,
    canonical_preparation_admission_snapshot,
    correlation_summary,
    distribution_summary,
    intervention_price_inputs,
    prepared_decision_row,
    selected_preparation_targets,
)
from leo_mappo.coordinator import SatellitePriceCoordinator
from leo_mappo.evaluation_diagnostics import (
    PolicyEvaluationDiagnostics,
    analyze_preparation_contention,
)
from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.policy_diagnostics import model_parameter_digest
from leo_mappo.reasoning import LookaheadReasoner, build_actor_reasoning_features
from leo_mappo.runner import evaluation_episode_seeds, summarize_metrics
from leo_mappo.ue import MobilityState


EPISODE_SEEDS = tuple(range(5001, 5011))
INTERVENTION_FIELDS = (
    "packet_delivery_ratio",
    "deadline_miss_rate",
    "requested_prep_actions",
    "rejected_prep_requests",
    "prep_rejection_rate",
    "requested_commit_actions",
    "requested_release_actions",
    "completed_handovers",
    "failed_handovers",
    "pingpong_events",
    "fraction_prep_requests_in_same_target_collision",
    "fraction_prep_requests_to_oversubscribed_targets",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_m4_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_json(run_dir / "checkpoint_metadata.json")
    config = metadata.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint metadata has no resolved_config mapping")
    if metadata.get("history_recurrent_mode") != "recurrent":
        raise ValueError("M4 diagnostic requires a recurrent checkpoint")
    if metadata.get("coordination_mode") != "satellite_price":
        raise ValueError("M4 diagnostic requires satellite-price coordination")
    if metadata.get("world_model_type") != "hybrid":
        raise ValueError("M4 diagnostic requires the Hybrid world model")
    if config.get("reasoning", {}).get("planner_objective") != "counterfactual":
        raise ValueError("M4 diagnostic requires the counterfactual planner")
    if bool(config.get("mappo", {}).get("use_constraint", False)):
        raise ValueError("M4 diagnostic requires reliability constraint disabled")
    if bool(config.get("reward", {}).get("charge_rejected_preparation_signaling", False)):
        raise ValueError("M4 diagnostic requires the original rejected-PREP reward")
    return deepcopy(config), metadata


def _build_agent(env: LEOMobilityEnv, config: dict[str, Any], weights: Path) -> MAPPO:
    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        config["mappo"],
        int(config.get("agent_seed", config["seed"])),
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )
    agent.load(weights)
    return agent


def _mean_rows(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if np.isfinite(float(row[field]))]
        result[field] = float(np.mean(values)) if values else None
    return result


def _price_admission_summary(
    satellite_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    correlations = {}
    for field in (
        "active_reservation_occupancy",
        "available_prep_admissions",
        "selected_prep_requests",
    ):
        correlations[f"price_vs_{field}"] = correlation_summary(
            [float(row["price"]) for row in satellite_rows],
            [float(row[field]) for row in satellite_rows],
        )
    requested_targets = [
        row for row in satellite_rows if int(row["selected_prep_requests"]) > 0
    ]
    correlations["price_vs_subsequent_rejection_probability"] = correlation_summary(
        [float(row["price"]) for row in requested_targets],
        [float(row["subsequent_rejection_probability"]) for row in requested_targets],
    )
    correlations["low_price_rank_vs_high_capacity_rank"] = correlation_summary(
        [float(row["price_rank_ascending"]) for row in satellite_rows],
        [float(row["available_capacity_rank_descending"]) for row in satellite_rows],
    )

    epoch_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in satellite_rows:
        epoch_groups.setdefault(
            (int(row["episode_seed"]), int(row["time_index"])), []
        ).append(row)
    rank_gaps = [
        abs(
            float(row["price_rank_ascending"])
            - float(row["available_capacity_rank_descending"])
        )
        for row in satellite_rows
    ]
    best_rank_matches = []
    for rows in epoch_groups.values():
        minimum_price = min(float(row["price"]) for row in rows)
        maximum_capacity = max(int(row["available_prep_admissions"]) for row in rows)
        low_price = {
            int(row["satellite_id"])
            for row in rows
            if float(row["price"]) == minimum_price
        }
        high_capacity = {
            int(row["satellite_id"])
            for row in rows
            if int(row["available_prep_admissions"]) == maximum_capacity
        }
        best_rank_matches.append(float(bool(low_price & high_capacity)))

    opportunity_fields = [
        "price",
        "normalized_price",
        "active_reservation_occupancy",
        "available_prep_admissions",
        "subsequent_rejection",
    ]
    selected = [row for row in candidate_rows if int(row["selected_target"]) == 1]
    nonselected = [
        row for row in candidate_rows if int(row["selected_target"]) == 0
    ]
    oversubscribed = [
        row for row in request_rows if int(row["oversubscribed_target"]) == 1
    ]
    not_oversubscribed = [
        row for row in request_rows if int(row["oversubscribed_target"]) == 0
    ]
    request_distribution_fields = (
        "target_price",
        "target_normalized_price",
        "target_available_prep_admissions",
        "target_active_reservation_occupancy",
    )

    return {
        "correlations": correlations,
        "price_capacity_rank": {
            "mean_absolute_rank_gap": float(np.mean(rank_gaps)) if rank_gaps else None,
            "lowest_price_has_maximum_capacity_fraction": (
                float(np.mean(best_rank_matches)) if best_rank_matches else None
            ),
        },
        "candidate_target_opportunities": {
            "selected": _mean_rows(selected, opportunity_fields),
            "feasible_nonselected": _mean_rows(nonselected, opportunity_fields),
        },
        "prep_request_target_distributions": {
            "oversubscribed": {
                field: distribution_summary([float(row[field]) for row in oversubscribed])
                for field in request_distribution_fields
            },
            "not_oversubscribed": {
                field: distribution_summary(
                    [float(row[field]) for row in not_oversubscribed]
                )
                for field in request_distribution_fields
            },
        },
    }


def _prepared_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"count": 0, "correlations": {}}
    prices = np.asarray([float(row["prepared_price"]) for row in rows])
    boundaries = np.quantile(prices, [0.25, 0.5, 0.75])
    labels = np.searchsorted(boundaries, prices, side="right") + 1
    fields = [
        "prepared_price",
        "serving_price",
        "actor_probability_stay",
        "actor_probability_commit",
        "actor_probability_release",
        "actor_logit_stay",
        "actor_logit_commit",
        "actor_logit_release",
        "feasible_stay",
        "feasible_commit",
        "feasible_release",
        "planner_raw_total_cost_stay",
        "planner_raw_total_cost_commit",
        "planner_raw_total_cost_release",
        "planner_raw_commit_minus_stay",
        "planner_raw_release_minus_stay",
        "selected_stay",
        "selected_commit",
        "selected_release",
        "reasoning_legacy_stay",
        "reasoning_legacy_commit",
        "reasoning_legacy_release",
        "reasoning_magnitude_stay",
        "reasoning_magnitude_commit",
        "reasoning_magnitude_release",
        "reasoning_global_confidence",
    ]
    for component in (
        "planner_service_cost",
        "planner_control_cost",
        "planner_reliability_cost",
        "planner_coordination_cost",
    ):
        fields.extend(f"{component}_{action}" for action in ("stay", "commit", "release"))
    quartiles = []
    for quartile in range(1, 5):
        group = [row for row, label in zip(rows, labels) if int(label) == quartile]
        summary = _mean_rows(group, fields)
        summary = {
            "prepared_price_quartile": f"Q{quartile}",
            "lower_bound": (
                float(np.min(prices)) if quartile == 1 else float(boundaries[quartile - 2])
            ),
            "upper_bound": (
                float(np.max(prices)) if quartile == 4 else float(boundaries[quartile - 1])
            ),
            **summary,
        }
        quartiles.append(summary)

    correlations = {}
    for field in (
        "planner_raw_commit_minus_stay",
        "planner_raw_release_minus_stay",
        "actor_probability_commit",
        "actor_probability_release",
        "selected_commit",
        "selected_release",
    ):
        correlations[f"prepared_price_vs_{field}"] = correlation_summary(
            prices.tolist(), [float(row[field]) for row in rows]
        )
    return quartiles, {
        "count": len(rows),
        "quartile_boundaries": [float(value) for value in boundaries],
        "correlations": correlations,
    }


def evaluate_condition(
    config: dict[str, Any],
    weights: Path,
    output_dir: Path,
    condition: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = LEOMobilityEnv(config)
    agent = _build_agent(env, config, weights)
    if not (agent.use_memory and agent.use_reasoning and agent.use_coordination):
        raise ValueError("diagnostic requires recurrent reasoning and coordination")
    if agent.reasoning_feature_mode != "confidence_aware":
        raise ValueError("diagnostic requires confidence-aware reasoning features")
    reasoner = LookaheadReasoner(
        config.get("reasoning", {}),
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=True,
        use_constraint=False,
    )
    if not isinstance(reasoner.world_model, HybridWorldModel):
        raise ValueError("diagnostic requires Hybrid reasoning")
    coordinator = SatellitePriceCoordinator(
        env.num_cells, config.get("coordination", {})
    )
    before_digest = model_parameter_digest(agent)
    totals: dict[str, float] = {}
    evaluation = PolicyEvaluationDiagnostics(
        interval_s=env.interval_s,
        packet_size_bits=int(env.traffic_cfg["packet_size_bits"]),
    )
    satellite_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    prepared_rows: list[dict[str, Any]] = []
    total_steps = 0

    for episode_seed in EPISODE_SEEDS:
        observations, _, masks = env.reset(seed=episode_seed)
        coordinator.reset()
        hidden_states = np.zeros(
            (env.num_agents, agent.recurrent_hidden_dim), dtype=np.float64
        )
        previous_actions = np.full(env.num_agents, -1, dtype=np.int64)
        previous_outcomes = np.zeros(
            (env.num_agents, agent.outcome_dim), dtype=np.float64
        )
        done = False
        time_index = 0
        while not done:
            prices = coordinator.prices
            normal_actor_prices = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            actor_prices, planner_prices = intervention_price_inputs(
                normal_actor_prices, prices, condition
            )
            reasoning_result = reasoner.analyze(env, masks, planner_prices, 0.0)
            reasoning_features = build_actor_reasoning_features(
                reasoning_result,
                masks,
                reasoner.score_clip,
                agent.reasoning_feature_mode,
            )
            logits, probabilities, new_hidden = agent.recurrent_policy(
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                reasoning_features,
                actor_prices,
            )
            actions = probabilities.argmax(axis=-1).astype(np.int64)
            actions_before_diagnostics = actions.copy()
            contention = analyze_preparation_contention(env, actions)

            normal_condition = condition == "normal"
            if normal_condition:
                admission = canonical_preparation_admission_snapshot(
                    env, masks, actions
                )
                request_targets = selected_preparation_targets(env, masks, actions)
                request_counts = np.bincount(
                    request_targets[request_targets >= 0], minlength=env.num_cells
                )
                candidates_before = [list(candidates) for candidates in env.candidates]
                candidate_ue_ids: list[list[int]] = [
                    [] for _ in range(env.num_cells)
                ]
                feasible_candidate_ue_ids: list[list[int]] = [
                    [] for _ in range(env.num_cells)
                ]
                for ue_id, candidates in enumerate(candidates_before):
                    for slot, target in enumerate(
                        candidates[: env.candidate_count]
                    ):
                        candidate_ue_ids[target].append(ue_id)
                        if masks[ue_id, env.prepare_start + slot] > 0.0:
                            feasible_candidate_ue_ids[target].append(ue_id)
                price_ranks = average_ranks(prices, ascending=True)
                capacity_ranks = average_ranks(
                    admission.available_slots.astype(np.float64), ascending=False
                )
                serving_demand = env.last_serving_demand.copy()
                preparation_demand = env.last_preparation_demand.copy()
                coordination_demand = env.last_coordination_demand.copy()
                for ue in env.ues:
                    if ue.state == MobilityState.PREPARED:
                        prepared_rows.append(
                            prepared_decision_row(
                                env,
                                ue.ue_id,
                                episode_seed,
                                time_index,
                                prices,
                                masks,
                                logits,
                                probabilities,
                                reasoning_result,
                                reasoning_features,
                                int(actions[ue.ue_id]),
                            )
                        )
                admissions_before = env.preparation_admissions_by_ue.copy()
                rejections_before = env.preparation_rejections_by_ue.copy()

            np.testing.assert_array_equal(actions, actions_before_diagnostics)
            output = env.step(actions)
            evaluation.record_step(env, actions, output, contention)
            for key, value in output.info.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            totals["resource_price_mean"] = totals.get("resource_price_mean", 0.0) + float(
                prices.mean()
            )
            totals["resource_price_max"] = totals.get("resource_price_max", 0.0) + float(
                prices.max(initial=0.0)
            )
            totals["resource_price_std"] = totals.get("resource_price_std", 0.0) + float(
                prices.std()
            )
            totals["price_nonzero_fraction"] = totals.get(
                "price_nonzero_fraction", 0.0
            ) + float(np.mean(prices > 1e-8))

            if normal_condition:
                admitted_delta = env.preparation_admissions_by_ue - admissions_before
                rejected_delta = env.preparation_rejections_by_ue - rejections_before
                admitted_by_target = np.zeros(env.num_cells, dtype=np.int64)
                rejected_by_target = np.zeros(env.num_cells, dtype=np.int64)
                for ue_id, target in enumerate(request_targets):
                    if target < 0:
                        continue
                    admitted_by_target[target] += int(admitted_delta[ue_id])
                    rejected_by_target[target] += int(rejected_delta[ue_id])
                requested_targets = request_counts > 0
                np.testing.assert_array_equal(
                    env.last_preparation_available_slots[requested_targets],
                    admission.available_slots[requested_targets],
                )
                if int(np.sum(request_counts)) != int(
                    output.info["preparation_requested"]
                ):
                    raise RuntimeError("diagnostic PREP request mapping diverged")

                for cell_id in range(env.num_cells):
                    requests = int(request_counts[cell_id])
                    rejection_probability = (
                        float(rejected_by_target[cell_id] / requests)
                        if requests > 0
                        else float("nan")
                    )
                    satellite_rows.append(
                        {
                            "episode_seed": episode_seed,
                            "time_index": time_index,
                            "satellite_id": cell_id,
                            "price": float(prices[cell_id]),
                            "normalized_price": float(
                                prices[cell_id] / coordinator.price_max
                            ),
                            "serving_demand": float(serving_demand[cell_id]),
                            "preparation_intent_demand": float(
                                preparation_demand[cell_id]
                            ),
                            "coordination_demand": float(
                                coordination_demand[cell_id]
                            ),
                            "active_reservation_occupancy": int(
                                admission.active_reservations[cell_id]
                            ),
                            "canonical_reservations_before_admission": int(
                                admission.canonical_reservations_before_admission[
                                    cell_id
                                ]
                            ),
                            "selected_releases_freeing_slots": int(
                                admission.selected_releases[cell_id]
                            ),
                            "maximum_preparation_slots": int(
                                admission.maximum_slots[cell_id]
                            ),
                            "available_prep_admissions": int(
                                admission.available_slots[cell_id]
                            ),
                            "candidate_ue_count": int(
                                admission.candidate_ue_count[cell_id]
                            ),
                            "candidate_ue_ids": "|".join(
                                str(ue_id) for ue_id in candidate_ue_ids[cell_id]
                            ),
                            "feasible_candidate_ue_count": int(
                                admission.feasible_candidate_ue_count[cell_id]
                            ),
                            "feasible_candidate_ue_ids": "|".join(
                                str(ue_id)
                                for ue_id in feasible_candidate_ue_ids[cell_id]
                            ),
                            "selected_prep_requests": requests,
                            "subsequent_prep_admitted": int(
                                admitted_by_target[cell_id]
                            ),
                            "subsequent_prep_rejected": int(
                                rejected_by_target[cell_id]
                            ),
                            "subsequent_rejection_probability": rejection_probability,
                            "oversubscribed_target": int(
                                requests > admission.available_slots[cell_id]
                            ),
                            "price_rank_ascending": float(price_ranks[cell_id]),
                            "available_capacity_rank_descending": float(
                                capacity_ranks[cell_id]
                            ),
                        }
                    )

                for ue_id, candidates in enumerate(candidates_before):
                    for slot, target in enumerate(candidates[: env.candidate_count]):
                        prep_action = env.prepare_start + slot
                        if masks[ue_id, prep_action] <= 0.0:
                            continue
                        selected = int(actions[ue_id] == prep_action)
                        candidate_rows.append(
                            {
                                "episode_seed": episode_seed,
                                "time_index": time_index,
                                "ue_id": ue_id,
                                "prep_action": prep_action,
                                "target_satellite": int(target),
                                "selected_target": selected,
                                "price": float(prices[target]),
                                "normalized_price": float(
                                    prices[target] / coordinator.price_max
                                ),
                                "active_reservation_occupancy": int(
                                    admission.active_reservations[target]
                                ),
                                "available_prep_admissions": int(
                                    admission.available_slots[target]
                                ),
                                "subsequent_rejection": (
                                    float(rejected_delta[ue_id])
                                    if selected
                                    else float("nan")
                                ),
                            }
                        )
                for ue_id, target in enumerate(request_targets):
                    if target < 0:
                        continue
                    request_rows.append(
                        {
                            "episode_seed": episode_seed,
                            "time_index": time_index,
                            "ue_id": ue_id,
                            "target_satellite": int(target),
                            "target_price": float(prices[target]),
                            "target_normalized_price": float(
                                prices[target] / coordinator.price_max
                            ),
                            "target_active_reservation_occupancy": int(
                                admission.active_reservations[target]
                            ),
                            "target_available_prep_admissions": int(
                                admission.available_slots[target]
                            ),
                            "target_selected_prep_requests": int(
                                request_counts[target]
                            ),
                            "oversubscribed_target": int(
                                request_counts[target]
                                > admission.available_slots[target]
                            ),
                            "admitted": int(admitted_delta[ue_id]),
                            "rejected": int(rejected_delta[ue_id]),
                        }
                    )

            hidden_states = new_hidden
            previous_actions = actions.copy()
            if output.agent_outcomes is None:
                raise RuntimeError("environment did not return recurrent outcomes")
            previous_outcomes = output.agent_outcomes.copy()
            if output.requested_pressure is None:
                raise RuntimeError("environment did not return coordination demand")
            coordinator.update(output.requested_pressure)
            observations, masks, done = (
                output.observations,
                output.action_masks,
                output.done,
            )
            total_steps += 1
            time_index += 1

    metrics = summarize_metrics(totals, total_steps)
    metrics.update(evaluation.metrics())
    after_digest = model_parameter_digest(agent)
    if before_digest != after_digest:
        raise RuntimeError("mechanism diagnostic mutated checkpoint parameters")
    provenance = {
        "condition": condition,
        "out_of_training_distribution": condition != "normal",
        "zero_actor_price_features": PRICE_INTERVENTIONS[condition][0],
        "zero_planner_coordination_price": PRICE_INTERVENTIONS[condition][1],
        "critic_used_for_action_selection": False,
        "checkpoint": str(weights.resolve()),
        "checkpoint_sha256": _sha256(weights),
        "evaluation_episode_seeds": list(EPISODE_SEEDS),
        "deterministic": True,
        "parameter_digest_before": before_digest,
        "parameter_digest_after": after_digest,
        "parameters_unchanged": True,
    }
    _write_json(output_dir / "evaluation.json", metrics)
    _write_json(output_dir / "provenance.json", provenance)

    mechanism_summary: dict[str, Any] = {}
    if condition == "normal":
        price_summary = _price_admission_summary(
            satellite_rows, candidate_rows, request_rows
        )
        prepared_quartiles, prepared_summary = _prepared_summaries(prepared_rows)
        _write_csv(output_dir / "satellite_epoch.csv", satellite_rows)
        _write_csv(
            output_dir / "candidate_target_opportunities.csv", candidate_rows
        )
        _write_csv(output_dir / "prep_requests.csv", request_rows)
        _write_csv(output_dir / "prepared_decisions.csv", prepared_rows)
        _write_csv(
            output_dir / "prepared_price_quartile_summary.csv",
            prepared_quartiles,
        )
        _write_json(output_dir / "price_admission_summary.json", price_summary)
        _write_json(output_dir / "prepared_price_summary.json", prepared_summary)
        mechanism_summary = {
            "price_admission": price_summary,
            "prepared_price": prepared_summary,
            "prepared_price_quartiles": prepared_quartiles,
        }
    return {**provenance, **metrics}, mechanism_summary


def run(args: argparse.Namespace) -> None:
    config, metadata = _load_m4_run(args.run_dir)
    if tuple(evaluation_episode_seeds(10, 5001)) != EPISODE_SEEDS:
        raise RuntimeError("matched evaluation seed definition changed")
    weights = args.run_dir / "mappo_weights_final.npz"
    intervention_rows = []
    normal_mechanism: dict[str, Any] = {}
    for condition in PRICE_INTERVENTIONS:
        result, mechanism = evaluate_condition(
            config,
            weights,
            args.output / condition,
            condition,
        )
        intervention_rows.append(
            {
                "condition": condition,
                "out_of_training_distribution": condition != "normal",
                **{field: result[field] for field in INTERVENTION_FIELDS},
            }
        )
        if condition == "normal":
            normal_mechanism = mechanism
    _write_csv(args.output / "intervention_summary.csv", intervention_rows)
    _write_json(
        args.output / "provenance.json",
        {
            "source_run": str(args.run_dir.resolve()),
            "source_metadata": metadata,
            "checkpoint": str(weights.resolve()),
            "checkpoint_sha256": _sha256(weights),
            "evaluation_episode_seeds": list(EPISODE_SEEDS),
            "conditions": list(PRICE_INTERVENTIONS),
            "warning": (
                "zero_actor_price, zero_planner_price, and zero_actor_and_planner_price "
                "are out-of-training-distribution mechanism diagnostics, not baselines"
            ),
        },
    )
    _write_json(
        args.output / "compact_summary.json",
        {
            "interventions": intervention_rows,
            "normal_mechanism": normal_mechanism,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen M4-H satellite-price mechanism diagnostics"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
