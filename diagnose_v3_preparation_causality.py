from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from diagnose_v3_outage_controllability import (  # noqa: E402
    AGENT_SEED,
    EPISODE_SEED_BASE,
    EVALUATION_ACTION_SEED,
    load_experiment,
    protocol_snapshot,
    ready_set_oracle_ratio,
    usable_true_leg,
)
from leo_mappo.coordinator import SatellitePriceCoordinator  # noqa: E402
from leo_mappo.reasoning import LookaheadReasoner  # noqa: E402
from leo_mappo.ue import MobilityState  # noqa: E402


CLASS_NAMES = {
    "A": "TARGET_NOT_IN_ACTION_SPACE_EARLY_ENOUGH",
    "B": "INHERENT_PREPARATION_TIMING_LIMITED",
    "C": "POLICY_MISSED_PREPARE_OPPORTUNITY",
    "D": "PREPARATION_CONTENTION_REJECTION",
    "E": "PREPARE_STARTED_TOO_LATE",
    "F": "PREMATURE_RELEASE_ROLLBACK",
    "G": "COMMIT_TRANSITION_LIMITED",
    "H": "OTHER",
}


def percentiles(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def true_visible_ratios(
    env: Any, ue_id: int, required_rate: float
) -> np.ndarray:
    ratios = np.zeros(env.num_cells, dtype=np.float64)
    if required_rate <= 0.0:
        return ratios
    epsilon = float(env.reward_cfg["epsilon"])
    for cell_id, satellite in enumerate(env.satellites):
        if not usable_true_leg(env, ue_id, cell_id):
            continue
        link = env.links[ue_id][cell_id]
        capacity = (
            satellite.bandwidth_hz
            * link.spectral_efficiency
            * satellite.effective_capacity_scale
        )
        ratios[cell_id] = (
            min(required_rate, capacity)
            * link.packet_success_probability
            / (required_rate + epsilon)
        )
    return ratios


def decision_snapshot(env: Any, masks: np.ndarray) -> list[dict[str, Any]]:
    snapshots = []
    for ue in env.ues:
        prepare_targets = {}
        for slot, target in enumerate(env.candidates[ue.ue_id]):
            action = env.prepare_start + slot
            prepare_targets[int(target)] = bool(masks[ue.ue_id, action] > 0.0)
        snapshots.append(
            {
                "primary": ue.primary_cell,
                "secondary": ue.secondary_cell,
                "state": int(ue.state),
                "timer": int(ue.protocol_timer),
                "candidates": [int(cell) for cell in env.candidates[ue.ue_id]],
                "prepare_targets": prepare_targets,
                "commit_mask": bool(masks[ue.ue_id, env.commit_action] > 0.0),
                "release_mask": bool(masks[ue.ue_id, env.release_action] > 0.0),
                "rollback_mask": bool(masks[ue.ue_id, env.rollback_action] > 0.0),
                "visible": [bool(link.visible) for link in env.links[ue.ue_id]],
            }
        )
    return snapshots


def preparation_lead_steps(env: Any) -> int:
    # Admission sets timer=preparation_steps. It is decremented at the start of
    # subsequent environment steps, and the actor receives the READY action mask
    # in the following decision state.
    return int(env.protocol_cfg["preparation_steps"]) + 1


def visible_through_ready(
    history: list[dict[str, Any]],
    ue_id: int,
    satellite: int,
    start: int,
    ready_step: int,
) -> bool:
    by_step = {row["step"]: row for row in history}
    return all(
        bool(by_step[step]["ues"][ue_id]["visible"][satellite])
        for step in range(start, ready_step + 1)
        if step in by_step
    )


def satellite_evidence(
    history: list[dict[str, Any]],
    ue_id: int,
    outage_step: int,
    satellite: int,
    required_lead: int,
) -> dict[str, Any]:
    rows = [row for row in history if row["step"] <= outage_step]
    first_visible = next(
        (
            row["step"]
            for row in rows
            if row["ues"][ue_id]["visible"][satellite]
        ),
        None,
    )
    first_candidate = next(
        (
            row["step"]
            for row in rows
            if satellite in row["ues"][ue_id]["candidates"]
        ),
        None,
    )
    legal_steps = [
        row["step"]
        for row in rows
        if row["ues"][ue_id]["prepare_targets"].get(satellite, False)
    ]
    sufficient_legal_steps = [
        step
        for step in legal_steps
        if step + required_lead <= outage_step
        and visible_through_ready(
            rows, ue_id, satellite, step, step + required_lead
        )
    ]
    request_events = [
        row["ues"][ue_id]["prepare_event"]
        for row in rows
        if row["ues"][ue_id].get("prepare_event") is not None
        and row["ues"][ue_id]["prepare_event"]["target"] == satellite
    ]
    timely_requests = [
        event
        for event in request_events
        if event["step"] in sufficient_legal_steps
    ]
    timely_admissions = [event for event in timely_requests if event["admitted"]]
    late_requests = [
        event
        for event in request_events
        if sufficient_legal_steps
        and event["step"] > sufficient_legal_steps[0]
        and event["step"] + required_lead > outage_step
    ]
    ready_steps = [
        row["step"]
        for row in rows
        if row["ues"][ue_id]["secondary"] == satellite
        and MobilityState(row["ues"][ue_id]["state"])
        in (MobilityState.PREPARED, MobilityState.DUPLICATING)
        and int(row["ues"][ue_id]["timer"]) == 0
    ]
    release_events = [
        event
        for row in rows
        for event in row["ues"][ue_id].get("discard_events", [])
        if event["target"] == satellite
    ]
    commit_legal_steps = [
        row["step"]
        for row in rows
        if row["ues"][ue_id]["secondary"] == satellite
        and row["ues"][ue_id]["commit_mask"]
    ]
    early_visible = any(
        row["step"] + required_lead <= outage_step
        and row["ues"][ue_id]["visible"][satellite]
        for row in rows
    )
    return {
        "satellite": satellite,
        "first_visible": first_visible,
        "first_candidate": first_candidate,
        "first_legal": legal_steps[0] if legal_steps else None,
        "sufficient_legal_steps": sufficient_legal_steps,
        "request_events": request_events,
        "timely_requests": timely_requests,
        "timely_admissions": timely_admissions,
        "late_requests": late_requests,
        "ready_steps": ready_steps,
        "release_events": release_events,
        "commit_legal_steps": commit_legal_steps,
        "early_visible": early_visible,
    }


def classify_outage(
    history: list[dict[str, Any]],
    outage_row: dict[str, Any],
    ue_id: int,
    rescue_satellites: list[int],
    required_lead: int,
) -> dict[str, Any]:
    outage_step = int(outage_row["step"])
    evidence = [
        satellite_evidence(
            history, ue_id, outage_step, satellite, required_lead
        )
        for satellite in rescue_satellites
    ]
    ready_oracle_succeeds = (
        outage_row["ues"][ue_id]["ready_ratio"] >= 0.8
    )
    sufficient = [item for item in evidence if item["sufficient_legal_steps"]]
    released_after_ready = [
        (item, event)
        for item in evidence
        for event in item["release_events"]
        if any(ready <= event["step"] for ready in item["ready_steps"])
        and event["step"] <= outage_step
        and not any(ready > event["step"] for ready in item["ready_steps"])
        and visible_through_ready(
            history,
            ue_id,
            int(item["satellite"]),
            int(event["step"]),
            outage_step,
        )
    ]
    timely_requests = [
        (item, event) for item in sufficient for event in item["timely_requests"]
    ]
    timely_admissions = [
        (item, event) for item in sufficient for event in item["timely_admissions"]
    ]
    late_requests = [
        (item, event) for item in sufficient for event in item["late_requests"]
    ]
    ready_before_outage = [
        item
        for item in sufficient
        if any(step <= outage_step for step in item["ready_steps"])
    ]
    causal_rejection_ids: list[int] = []
    release_lead: int | None = None

    if ready_oracle_succeeds:
        class_code = "G"
        reason = "READY-set counterfactual met the outage threshold"
    elif released_after_ready:
        class_code = "F"
        item, event = max(released_after_ready, key=lambda pair: pair[1]["step"])
        release_lead = outage_step - int(event["step"])
        reason = (
            f"satellite {item['satellite']} was READY before a "
            f"{event['action']} at step {event['step']}"
        )
    elif sufficient:
        if timely_admissions:
            class_code = "H"
            preparing_discards = [
                event
                for item, admission in timely_admissions
                for event in item["release_events"]
                if admission["step"] <= event["step"] < admission["step"] + required_lead
            ]
            reason = (
                "an admitted PREPARE was discarded while still PREPARING before READY"
                if preparing_discards
                else "a continuously viable timely admission did not produce an observed READY state"
            )
        elif timely_requests:
            class_code = "D"
            causal_rejection_ids = [
                int(event["event_id"])
                for _, event in timely_requests
                if event["rejected"]
            ]
            reason = "timely rescue-target PREPARE requests were rejected"
        elif late_requests:
            class_code = "E"
            reason = "a rescue-target PREPARE was requested only after the sufficient window"
        else:
            class_code = "C"
            reason = "a sufficient legal PREPARE opportunity was not used"
    elif any(item["early_visible"] for item in evidence):
        class_code = "A"
        reason = "a rescue target was visible early but never legally targetable early enough"
    elif evidence:
        class_code = "B"
        reason = "even the earliest legal/visible rescue opportunity was too late"
    else:
        class_code = "H"
        reason = "no rescue-capable satellite evidence was available"

    best_satellite = int(outage_row["ues"][ue_id]["best_rescue_satellite"])
    best_evidence = next(
        (item for item in evidence if item["satellite"] == best_satellite),
        evidence[0] if evidence else {},
    )
    first_request = (
        best_evidence["request_events"][0]["step"]
        if best_evidence and best_evidence["request_events"]
        else None
    )
    first_ready = (
        best_evidence["ready_steps"][0]
        if best_evidence and best_evidence["ready_steps"]
        else None
    )
    return {
        "class_code": class_code,
        "class_name": CLASS_NAMES[class_code],
        "reason": reason,
        "evidence": evidence,
        "sufficient_opportunity": bool(sufficient),
        "timely_request": bool(timely_requests),
        "timely_admission": bool(timely_admissions),
        "ready_before_outage": bool(ready_before_outage),
        "causal_rejection_ids": causal_rejection_ids,
        "release_lead": release_lead,
        "best_satellite": best_satellite,
        "first_visible_lead": (
            None
            if not best_evidence or best_evidence["first_visible"] is None
            else outage_step - int(best_evidence["first_visible"])
        ),
        "first_candidate_lead": (
            None
            if not best_evidence or best_evidence["first_candidate"] is None
            else outage_step - int(best_evidence["first_candidate"])
        ),
        "first_legal_lead": (
            None
            if not best_evidence or best_evidence["first_legal"] is None
            else outage_step - int(best_evidence["first_legal"])
        ),
        "first_request_lead": (
            None if first_request is None else outage_step - int(first_request)
        ),
        "first_ready_lead": (
            None if first_ready is None else outage_step - int(first_ready)
        ),
    }


def compact_trace(
    episode: int,
    outage_row: dict[str, Any],
    ue_id: int,
    result: dict[str, Any],
    required_lead: int,
) -> dict[str, Any]:
    return {
        "episode": episode,
        "ue_id": ue_id,
        "outage_step": int(outage_row["step"]),
        "rescue_satellite": result["best_satellite"],
        "rescue_capable_set": outage_row["ues"][ue_id]["rescue_satellites"],
        "first_visible_lead": result["first_visible_lead"],
        "first_candidate_lead": result["first_candidate_lead"],
        "first_legal_prepare_lead": result["first_legal_lead"],
        "first_actual_prepare_lead": result["first_request_lead"],
        "first_ready_lead": result["first_ready_lead"],
        "required_preparation_lead": required_lead,
        "classification": result["class_name"],
        "reason": result["reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose V3 preparation causality")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    config, env, agent = load_experiment()
    reasoner = LookaheadReasoner(
        config.get("reasoning", {}),
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=True,
        use_constraint=True,
    )
    coordinator = SatellitePriceCoordinator(
        env.num_cells, config.get("coordination", {})
    )
    agent.rng = np.random.default_rng(EVALUATION_ACTION_SEED)
    trajectory_mu = float(agent.outage_dual_mu)
    normalized_mu = trajectory_mu / float(config["constraint"]["mu_max"])
    required_lead = preparation_lead_steps(env)
    histories: list[list[dict[str, Any]]] = []
    request_events: list[dict[str, Any]] = []
    event_counter = 0
    capture: dict[str, Any] = {}
    original_apply_actions = env._apply_actions
    original_schedule = env._schedule_and_transmit

    def captured_apply_actions(actions: np.ndarray, masks: np.ndarray) -> dict[str, int]:
        nonlocal event_counter
        before_action = protocol_snapshot(env)
        valid_actions = np.zeros(env.num_agents, dtype=np.int64)
        raw_requests: list[tuple[int, int]] = []
        for ue, raw_action in zip(env.ues, actions):
            action = int(raw_action)
            if action < 0 or action >= env.action_dim or masks[ue.ue_id, action] == 0:
                action = 0
            valid_actions[ue.ue_id] = action
            if env.prepare_start <= action < env.prepare_start + env.candidate_count:
                slot = action - env.prepare_start
                if slot < len(env.candidates[ue.ue_id]):
                    raw_requests.append((ue.ue_id, int(env.candidates[ue.ue_id][slot])))

        occupancy = np.zeros(env.num_cells, dtype=np.int64)
        for ue in env.ues:
            action = int(valid_actions[ue.ue_id])
            state = MobilityState(int(before_action[ue.ue_id]["state"]))
            secondary = before_action[ue.ue_id]["secondary"]
            if action in (env.release_action, env.rollback_action):
                continue
            if secondary is not None and state in (
                MobilityState.PREPARED,
                MobilityState.DUPLICATING,
                MobilityState.SWITCHING,
            ):
                occupancy[int(secondary)] += 1
        request_counts = defaultdict(int)
        for _, target in raw_requests:
            request_counts[target] += 1
        capacity = int(
            env.preparation_admission_cfg.get(
                "max_concurrent_per_satellite", env.num_agents
            )
        )
        available = {
            target: max(0, capacity - int(occupancy[target]))
            for target in request_counts
        }
        counters = original_apply_actions(actions, masks)
        per_ue_event: list[dict[str, Any] | None] = [None] * env.num_agents
        for ue_id, target in raw_requests:
            admitted = bool(
                env.ues[ue_id].state == MobilityState.PREPARED
                and env.ues[ue_id].secondary_cell == target
            )
            event = {
                "event_id": event_counter,
                "episode": int(capture["episode"]),
                "step": int(capture["step"]),
                "ue_id": ue_id,
                "target": target,
                "concurrent_requests": int(request_counts[target]),
                "available_slots": int(available[target]),
                "admitted": admitted,
                "rejected": not admitted,
            }
            event_counter += 1
            request_events.append(event)
            per_ue_event[ue_id] = event
        capture["before_action"] = before_action
        capture["valid_actions"] = valid_actions
        capture["prepare_events"] = per_ue_event
        return counters

    def captured_schedule(
        required_rates: np.ndarray, active: np.ndarray
    ) -> dict[str, float]:
        before_action = capture["before_action"]
        visible_ratios = np.zeros((env.num_agents, env.num_cells), dtype=np.float64)
        ready_ratios = np.zeros(env.num_agents, dtype=np.float64)
        post_protocol = protocol_snapshot(env)
        for ue_id in np.flatnonzero(active > 0.0):
            visible_ratios[ue_id] = true_visible_ratios(
                env, int(ue_id), float(required_rates[ue_id])
            )
            ready_ratios[ue_id] = ready_set_oracle_ratio(
                env, int(ue_id), required_rates, before_action[ue_id]
            )
        delivered_before = np.asarray(
            [ue.delivered_packets for ue in env.ues], dtype=np.int64
        )
        metrics = original_schedule(required_rates, active)
        delivered_after = np.asarray(
            [ue.delivered_packets for ue in env.ues], dtype=np.int64
        )
        realized_goodput = (
            (delivered_after - delivered_before)
            * int(env.traffic_cfg["packet_size_bits"])
            / max(env.interval_s, 1e-8)
        )
        actual_ratios = np.where(
            active > 0.0,
            realized_goodput / (required_rates + env.reward_cfg["epsilon"]),
            0.0,
        )
        capture["transmission"] = {
            "active": active.copy(),
            "required_rates": required_rates.copy(),
            "visible_ratios": visible_ratios,
            "ready_ratios": ready_ratios,
            "actual_ratios": actual_ratios,
            "post_protocol": post_protocol,
        }
        return metrics

    env._apply_actions = captured_apply_actions  # type: ignore[method-assign]
    env._schedule_and_transmit = captured_schedule  # type: ignore[method-assign]

    for episode in range(args.episodes):
        observations, _, masks = env.reset(seed=EPISODE_SEED_BASE + episode)
        coordinator.reset()
        hidden_states = np.zeros(
            (env.num_agents, agent.recurrent_hidden_dim), dtype=np.float64
        )
        previous_actions = np.full(env.num_agents, -1, dtype=np.int64)
        previous_outcomes = np.zeros(
            (env.num_agents, agent.outcome_dim), dtype=np.float64
        )
        history: list[dict[str, Any]] = []
        done = False
        step = 0
        while not done:
            prices = coordinator.prices
            reasoning_result = reasoner.analyze(
                env, masks, prices, trajectory_mu
            )
            price_features = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            actions, _, new_hidden = agent.act_recurrent(
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                reasoning_result.features,
                price_features,
                normalized_mu,
                deterministic=False,
            )
            decisions = decision_snapshot(env, masks)
            capture["episode"] = episode
            capture["step"] = step
            output = env.step(actions)
            transmission = capture["transmission"]
            ues = []
            for ue_id in range(env.num_agents):
                decision = dict(decisions[ue_id])
                action = int(actions[ue_id])
                discard_events = []
                if action in (env.release_action, env.rollback_action):
                    target = decision["secondary"]
                    if target is not None:
                        discard_events.append(
                            {
                                "step": step,
                                "target": int(target),
                                "action": (
                                    "RELEASE"
                                    if action == env.release_action
                                    else "ROLLBACK"
                                ),
                            }
                        )
                ratios = transmission["visible_ratios"][ue_id]
                rescue_satellites = [
                    int(cell)
                    for cell in np.flatnonzero(ratios >= env.outage_service_ratio)
                ]
                best_rescue = (
                    int(np.argmax(ratios)) if rescue_satellites else -1
                )
                post = transmission["post_protocol"][ue_id]
                decision.update(
                    {
                        "action": action,
                        "valid_action": int(capture["valid_actions"][ue_id]),
                        "prepare_event": capture["prepare_events"][ue_id],
                        "discard_events": discard_events,
                        "active": bool(transmission["active"][ue_id] > 0.0),
                        "actual_ratio": float(transmission["actual_ratios"][ue_id]),
                        "ready_ratio": float(transmission["ready_ratios"][ue_id]),
                        "rescue_satellites": rescue_satellites,
                        "best_rescue_satellite": best_rescue,
                        "post_state": int(post["state"]),
                        "post_timer": int(post["timer"]),
                        "post_secondary": post["secondary"],
                    }
                )
                ues.append(decision)
            history.append({"step": step, "ues": ues})
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("environment omitted diagnostic outputs")
            expected_outages = np.asarray(
                [
                    ue["active"]
                    and ue["actual_ratio"] < env.outage_service_ratio
                    for ue in ues
                ],
                dtype=np.float64,
            )
            np.testing.assert_array_equal(
                expected_outages, output.agent_outcomes[:, 1]
            )
            coordinator.update(output.requested_pressure)
            hidden_states = new_hidden
            previous_actions = actions.copy()
            previous_outcomes = output.agent_outcomes.copy()
            observations = output.observations
            masks = output.action_masks
            done = output.done
            step += 1
        histories.append(history)
        print(
            f"completed episode {episode + 1}/{args.episodes}",
            file=sys.stderr,
            flush=True,
        )

    classifications: list[dict[str, Any]] = []
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    causal_rejection_ids: set[int] = set()
    release_leads: list[float] = []
    for episode, history in enumerate(histories):
        for row in history:
            for ue_id, ue_row in enumerate(row["ues"]):
                if not (
                    ue_row["active"]
                    and ue_row["actual_ratio"] < env.outage_service_ratio
                ):
                    continue
                rescue_satellites = ue_row["rescue_satellites"]
                result = classify_outage(
                    history,
                    row,
                    ue_id,
                    rescue_satellites,
                    required_lead,
                )
                result.update(
                    {
                        "episode": episode,
                        "step": row["step"],
                        "ue_id": ue_id,
                    }
                )
                classifications.append(result)
                causal_rejection_ids.update(result["causal_rejection_ids"])
                if result["release_lead"] is not None:
                    release_leads.append(float(result["release_lead"]))
                code = result["class_code"]
                if len(traces[code]) < 3:
                    traces[code].append(
                        compact_trace(
                            episode, row, ue_id, result, required_lead
                        )
                    )

    outage_count = len(classifications)
    class_counts = {
        code: sum(result["class_code"] == code for result in classifications)
        for code in CLASS_NAMES
    }
    causal_classes = {
        code: {
            "name": CLASS_NAMES[code],
            "count": int(class_counts[code]),
            "fraction": float(class_counts[code] / max(outage_count, 1)),
        }
        for code in CLASS_NAMES
    }
    other_reason_counts = {
        reason: sum(
            result["class_code"] == "H" and result["reason"] == reason
            for result in classifications
        )
        for reason in sorted(
            {
                result["reason"]
                for result in classifications
                if result["class_code"] == "H"
            }
        )
    }
    lead_fields = {
        "first_visible_to_outage": "first_visible_lead",
        "first_candidate_to_outage": "first_candidate_lead",
        "first_legal_prepare_to_outage": "first_legal_lead",
        "first_actual_prepare_to_outage": "first_request_lead",
        "first_ready_to_outage": "first_ready_lead",
    }
    lead_statistics = {
        name: percentiles(
            [
                float(result[field])
                for result in classifications
                if result[field] is not None
            ]
        )
        for name, field in lead_fields.items()
    }
    sufficient_count = sum(
        result["sufficient_opportunity"] for result in classifications
    )
    request_count = sum(result["timely_request"] for result in classifications)
    admitted_count = sum(result["timely_admission"] for result in classifications)
    ready_count = sum(result["ready_before_outage"] for result in classifications)
    funnel = {
        "sufficient_opportunity": {
            "count": int(sufficient_count),
            "fraction_of_outages": float(sufficient_count / max(outage_count, 1)),
        },
        "timely_actual_request": {
            "count": int(request_count),
            "fraction_of_outages": float(request_count / max(outage_count, 1)),
            "fraction_of_previous_stage": float(
                request_count / max(sufficient_count, 1)
            ),
        },
        "timely_admitted": {
            "count": int(admitted_count),
            "fraction_of_outages": float(admitted_count / max(outage_count, 1)),
            "fraction_of_previous_stage": float(
                admitted_count / max(request_count, 1)
            ),
        },
        "ready_before_outage": {
            "count": int(ready_count),
            "fraction_of_outages": float(ready_count / max(outage_count, 1)),
            "fraction_of_previous_stage": float(
                ready_count / max(admitted_count, 1)
            ),
        },
    }
    contention = {}
    for satellite in range(env.num_cells):
        events = [event for event in request_events if event["target"] == satellite]
        rejected = [event for event in events if event["rejected"]]
        causal = [
            event for event in rejected if event["event_id"] in causal_rejection_ids
        ]
        contention[str(satellite)] = {
            "request_count": len(events),
            "mean_concurrent_requests": float(
                np.mean([event["concurrent_requests"] for event in events])
            )
            if events
            else 0.0,
            "mean_available_slots": float(
                np.mean([event["available_slots"] for event in events])
            )
            if events
            else 0.0,
            "rejection_count": len(rejected),
            "rejected_fraction": float(len(rejected) / max(len(events), 1)),
            "outage_causal_rejection_count": len(causal),
            "fraction_of_rejections_linked_to_class_d_outage": float(
                len(causal) / max(len(rejected), 1)
            ),
        }

    report = {
        "configuration": {
            "episodes": args.episodes,
            "environment_seed": 7,
            "agent_seed": AGENT_SEED,
            "evaluation_action_seed": EVALUATION_ACTION_SEED,
            "episode_seed_base": EPISODE_SEED_BASE,
            "checkpoint": "outputs/v3_final_1500/m5_seed7/mappo_weights.npz",
            "checkpoint_mu": trajectory_mu,
            "preparation_steps": int(env.protocol_cfg["preparation_steps"]),
            "switching_steps": int(env.protocol_cfg["switching_steps"]),
            "reasoning_horizon": int(config["reasoning"]["horizon_steps"]),
            "history_lookback": "full episode history",
            "required_preparation_lead_steps": required_lead,
        },
        "evaluated_actual_outages": outage_count,
        "causal_classes": causal_classes,
        "other_reason_breakdown": {
            reason: {
                "count": int(count),
                "fraction_of_outages": float(count / max(outage_count, 1)),
            }
            for reason, count in other_reason_counts.items()
        },
        "earliest_legal_prepare_counterfactual": {
            "ready_possible_before_outage_count": int(sufficient_count),
            "ready_possible_before_outage_fraction": float(
                sufficient_count / max(outage_count, 1)
            ),
            "still_too_late_count": int(outage_count - sufficient_count),
            "still_too_late_fraction": float(
                (outage_count - sufficient_count) / max(outage_count, 1)
            ),
        },
        "lead_time_statistics_steps": lead_statistics,
        "opportunity_funnel": funnel,
        "preparation_contention_by_satellite": contention,
        "prepare_release_churn": {
            "causal_outage_count": int(class_counts["F"]),
            "fraction_of_outages": float(class_counts["F"] / max(outage_count, 1)),
            "release_to_outage_lead_steps": percentiles(release_leads),
        },
        "sample_traces": dict(traces),
        "classification_semantics": {
            "rescue_set": (
                "all true currently visible satellites whose independent optimistic "
                "instantaneous expected service ratio is at least 0.8"
            ),
            "counterfactual": (
                "earliest policy-visible legal PREPARE, full state-machine preparation "
                "lead, and continuous target visibility through READY; admission is "
                "evaluated separately as contention"
            ),
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
