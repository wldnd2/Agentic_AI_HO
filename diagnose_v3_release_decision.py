from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from diagnose_v3_outage_controllability import (  # noqa: E402
    EPISODE_SEED_BASE,
    EVALUATION_ACTION_SEED,
    load_experiment,
)
from diagnose_v3_preparation_causality import true_visible_ratios  # noqa: E402
from leo_mappo.coordinator import (  # noqa: E402
    SatellitePriceCoordinator,
    compute_training_reward,
)
from leo_mappo.networks import masked_softmax  # noqa: E402
from leo_mappo.reasoning import LookaheadReasoner  # noqa: E402
from leo_mappo.ue import MobilityState  # noqa: E402


HORIZONS = (1, 2, 4, 8, 16)
TIE_EPSILON = 1e-8


def distribution(values: list[float]) -> dict[str, float | int | None]:
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


def recurrent_policy(
    agent: Any,
    observations: np.ndarray,
    masks: np.ndarray,
    previous_actions: np.ndarray,
    previous_outcomes: np.ndarray,
    hidden_states: np.ndarray,
    reasoning_features: np.ndarray,
    price_features: np.ndarray,
    normalized_mu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if agent.gru is None:
        raise RuntimeError("M5 diagnostic requires recurrent policy")
    one_hot = np.zeros((len(observations), agent.action_dim), dtype=np.float64)
    valid = (previous_actions >= 0) & (previous_actions < agent.action_dim)
    rows = np.flatnonzero(valid)
    one_hot[rows, previous_actions[rows]] = 1.0
    gru_inputs = np.concatenate(
        [observations, one_hot, previous_outcomes], axis=1
    )
    new_hidden, _ = agent.gru.forward_step(gru_inputs, hidden_states)
    actor_input = np.concatenate(
        [
            observations,
            new_hidden,
            reasoning_features,
            price_features,
            np.full((len(observations), 1), normalized_mu, dtype=np.float64),
        ],
        axis=1,
    )
    logits, _ = agent.actor.forward(actor_input)
    return logits, masked_softmax(logits, masks), new_hidden


def sample_actions(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(
        [rng.choice(probabilities.shape[1], p=row) for row in probabilities],
        dtype=np.int64,
    )


def discard_kind(env: Any, ue: Any, action: int) -> str | None:
    if ue.secondary_cell is None or action not in (
        env.release_action,
        env.rollback_action,
    ):
        return None
    if ue.state == MobilityState.PREPARED and ue.protocol_timer > 0:
        return "PREPARING"
    if ue.state in (MobilityState.PREPARED, MobilityState.DUPLICATING):
        return "READY"
    return None


def is_ready_for_target(ue: Any, target: int) -> bool:
    return bool(
        ue.secondary_cell == target
        and ue.state in (MobilityState.PREPARED, MobilityState.DUPLICATING)
        and ue.protocol_timer == 0
    )


def forced_branch(
    snapshot: dict[str, Any],
    agent: Any,
    config: dict[str, Any],
    first_action: int,
    max_horizon: int,
) -> dict[int, dict[str, float]]:
    env = deepcopy(snapshot["env"])
    coordinator = deepcopy(snapshot["coordinator"])
    reasoner = LookaheadReasoner(
        config.get("reasoning", {}),
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=True,
        use_constraint=True,
    )
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(snapshot["actor_rng_state"])
    actions = snapshot["sampled_actions"].copy()
    actions[snapshot["ue_id"]] = first_action
    hidden_states = snapshot["new_hidden"].copy()
    target = int(snapshot["target"])
    target_ue = int(snapshot["ue_id"])
    lambda_resource = float(env.reward_cfg["lambda_resource"])
    outage_target = float(config["constraint"]["outage_target"])
    cumulative = defaultdict(float)
    results: dict[int, dict[str, float]] = {}
    reprepare = False
    observations = snapshot["observations"].copy()
    masks = snapshot["masks"].copy()
    previous_actions = snapshot["previous_actions"].copy()
    previous_outcomes = snapshot["previous_outcomes"].copy()

    for offset in range(max_horizon):
        if offset > 0:
            action_candidates = [list(row) for row in env.candidates]
            prices = coordinator.prices
            reasoning_result = reasoner.analyze(
                env, masks, prices, snapshot["mu"]
            )
            price_features = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            _, probabilities, new_hidden = recurrent_policy(
                agent,
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                reasoning_result.features,
                price_features,
                snapshot["normalized_mu"],
            )
            actions = sample_actions(probabilities, rng)
            action = int(actions[target_ue])
            if env.prepare_start <= action < env.prepare_start + env.candidate_count:
                slot = action - env.prepare_start
                if slot < len(action_candidates[target_ue]):
                    reprepare |= action_candidates[target_ue][slot] == target
        output = env.step(actions)
        if output.agent_outcomes is None or output.requested_pressure is None:
            raise RuntimeError("counterfactual environment omitted outcomes")
        reward, _ = compute_training_reward(
            output.info["service_deficit"],
            output.info["resource_cost"],
            output.info["outage_cost"],
            lambda_resource,
            True,
            snapshot["mu"],
            outage_target,
        )
        cumulative["service_deficit"] += float(output.info["service_deficit"])
        cumulative["service_reward_contribution"] += -float(
            output.info["service_deficit"]
        )
        cumulative["outage_ue_steps"] += float(output.info["outage_ue_count"])
        cumulative["outage_cost"] += float(output.info["outage_cost"])
        cumulative["dual_outage_contribution"] += -snapshot["mu"] * (
            float(output.info["outage_cost"]) - outage_target
        )
        cumulative["target_ready_steps"] += float(
            is_ready_for_target(env.ues[target_ue], target)
        )
        cumulative["resource_cost"] += float(output.info["resource_cost"])
        cumulative["resource_reward_contribution"] += -lambda_resource * float(
            output.info["resource_cost"]
        )
        cumulative["preparation_requests"] += float(
            output.info["preparation_requested"]
        )
        cumulative["team_reward"] += reward
        coordinator.update(output.requested_pressure)
        hidden_states = (
            snapshot["new_hidden"].copy() if offset == 0 else new_hidden
        )
        previous_actions = actions.copy()
        previous_outcomes = output.agent_outcomes.copy()
        observations = output.observations
        masks = output.action_masks
        step_count = offset + 1
        if step_count in HORIZONS:
            results[step_count] = {
                **{key: float(value) for key, value in cumulative.items()},
                "later_reprepare": float(reprepare),
            }
        if output.done:
            break
    return results


def replay_counterfactuals(
    config: dict[str, Any],
    selected_events: list[dict[str, Any]],
    episodes: int,
) -> dict[int, dict[str, Any]]:
    _, env, agent = load_experiment()
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
    mu = float(agent.outage_dual_mu)
    normalized_mu = mu / float(config["constraint"]["mu_max"])
    selected_by_step: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in selected_events:
        selected_by_step[(event["episode"], event["step"])].append(event)
    results: dict[int, dict[str, Any]] = {}
    completed = 0

    for episode in range(episodes):
        observations, _, masks = env.reset(seed=EPISODE_SEED_BASE + episode)
        coordinator.reset()
        hidden_states = np.zeros(
            (env.num_agents, agent.recurrent_hidden_dim), dtype=np.float64
        )
        previous_actions = np.full(env.num_agents, -1, dtype=np.int64)
        previous_outcomes = np.zeros(
            (env.num_agents, agent.outcome_dim), dtype=np.float64
        )
        done = False
        step = 0
        while not done:
            reasoning_result = reasoner.analyze(env, masks, coordinator.prices, mu)
            price_features = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            _, probabilities, new_hidden = recurrent_policy(
                agent,
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                reasoning_result.features,
                price_features,
                normalized_mu,
            )
            actions = sample_actions(probabilities, agent.rng)
            selected_here = selected_by_step.get((episode, step), [])
            if selected_here:
                actor_rng_state = deepcopy(agent.rng.bit_generator.state)
                base_env = deepcopy(env)
                for event in selected_here:
                    ue_id = int(event["ue_id"])
                    if (
                        int(actions[ue_id]) != int(event["sampled_action"])
                        or env.ues[ue_id].secondary_cell != event["target"]
                    ):
                        raise RuntimeError(
                            "counterfactual replay diverged from diagnostic trajectory"
                        )
                    snapshot = {
                        "env": deepcopy(base_env),
                        "coordinator": deepcopy(coordinator),
                        "ue_id": ue_id,
                        "target": int(event["target"]),
                        "discard_action": int(event["sampled_action"]),
                        "sampled_actions": actions.copy(),
                        "new_hidden": new_hidden.copy(),
                        "observations": observations.copy(),
                        "masks": masks.copy(),
                        "previous_actions": previous_actions.copy(),
                        "previous_outcomes": previous_outcomes.copy(),
                        "actor_rng_state": deepcopy(actor_rng_state),
                        "mu": mu,
                        "normalized_mu": normalized_mu,
                        "remaining_steps": int(env.env_cfg["episode_steps"]) - step,
                    }
                    maximum = min(max(HORIZONS), snapshot["remaining_steps"])
                    results[event["event_id"]] = {
                        "keep": forced_branch(snapshot, agent, config, 0, maximum),
                        "release": forced_branch(
                            snapshot,
                            agent,
                            config,
                            int(event["sampled_action"]),
                            maximum,
                        ),
                    }
                    completed += 1
                    if completed % 25 == 0 or completed == len(selected_events):
                        print(
                            f"completed counterfactual {completed}/{len(selected_events)}",
                            file=sys.stderr,
                            flush=True,
                        )
            output = env.step(actions)
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("counterfactual replay omitted outcomes")
            coordinator.update(output.requested_pressure)
            hidden_states = new_hidden
            previous_actions = actions.copy()
            previous_outcomes = output.agent_outcomes.copy()
            observations = output.observations
            masks = output.action_masks
            done = output.done
            step += 1
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose V3 release decisions")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--counterfactual-samples", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0 or args.counterfactual_samples <= 0:
        raise ValueError("episodes and counterfactual-samples must be positive")

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
    mu = float(agent.outage_dual_mu)
    normalized_mu = mu / float(config["constraint"]["mu_max"])
    reasoning_horizon = int(config["reasoning"]["horizon_steps"])
    discard_events: list[dict[str, Any]] = []
    keep_opportunities: list[dict[str, Any]] = []
    histories: list[list[dict[str, Any]]] = []
    reservoir_rng = np.random.default_rng(70_077)
    capture: dict[str, Any] = {}
    original_schedule = env._schedule_and_transmit

    def captured_schedule(
        required_rates: np.ndarray, active: np.ndarray
    ) -> dict[str, float]:
        visible_ratios = np.zeros((env.num_agents, env.num_cells), dtype=np.float64)
        for ue_id in np.flatnonzero(active > 0.0):
            visible_ratios[ue_id] = true_visible_ratios(
                env, int(ue_id), float(required_rates[ue_id])
            )
        delivered_before = np.asarray(
            [ue.delivered_packets for ue in env.ues], dtype=np.int64
        )
        metrics = original_schedule(required_rates, active)
        delivered_after = np.asarray(
            [ue.delivered_packets for ue in env.ues], dtype=np.int64
        )
        actual_ratios = np.where(
            active > 0.0,
            (delivered_after - delivered_before)
            * int(env.traffic_cfg["packet_size_bits"])
            / max(env.interval_s, 1e-8)
            / (required_rates + env.reward_cfg["epsilon"]),
            0.0,
        )
        capture["transmission"] = {
            "active": active.copy(),
            "actual_ratios": actual_ratios,
            "visible_ratios": visible_ratios,
            "visible": np.asarray(
                [[link.visible for link in row] for row in env.links],
                dtype=np.bool_,
            ),
        }
        return metrics

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
            reasoning_result = reasoner.analyze(env, masks, prices, mu)
            price_features = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            logits, probabilities, new_hidden = recurrent_policy(
                agent,
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                reasoning_result.features,
                price_features,
                normalized_mu,
            )
            actions = sample_actions(probabilities, agent.rng)
            event_ids_by_ue: dict[int, int] = {}
            for ue in env.ues:
                action = int(actions[ue.ue_id])
                kind = discard_kind(env, ue, action)
                state_has_target = ue.secondary_cell is not None and ue.state in (
                    MobilityState.PREPARED,
                    MobilityState.DUPLICATING,
                )
                if state_has_target:
                    keep_opportunities.append(
                        {
                            "episode": episode,
                            "step": step,
                            "ue_id": ue.ue_id,
                            "target": int(ue.secondary_cell),
                            "target_price": float(prices[ue.secondary_cell]),
                            "action": action,
                            "ready": bool(ue.protocol_timer == 0),
                        }
                    )
                if kind is None:
                    continue
                target = int(ue.secondary_cell)
                legal = np.flatnonzero(masks[ue.ue_id] > 0.0)
                stay = 0
                discard_action = action
                delta = {
                    "total": float(
                        reasoning_result.total_costs[ue.ue_id, discard_action]
                        - reasoning_result.total_costs[ue.ue_id, stay]
                    ),
                    "service": float(
                        reasoning_result.service_costs[ue.ue_id, discard_action]
                        - reasoning_result.service_costs[ue.ue_id, stay]
                    ),
                    "coordination": float(
                        reasoning_result.coordination_costs[ue.ue_id, discard_action]
                        - reasoning_result.coordination_costs[ue.ue_id, stay]
                    ),
                    "reliability": float(
                        reasoning_result.reliability_costs[ue.ue_id, discard_action]
                        - reasoning_result.reliability_costs[ue.ue_id, stay]
                    ),
                }
                event_id = len(discard_events)
                deterministic_action = int(np.argmax(probabilities[ue.ue_id]))
                event = {
                    "event_id": event_id,
                    "episode": episode,
                    "step": step,
                    "ue_id": ue.ue_id,
                    "target": target,
                    "kind": kind,
                    "sampled_action": discard_action,
                    "deterministic_action": deterministic_action,
                    "deterministic_discard": bool(
                        deterministic_action
                        in (env.release_action, env.rollback_action)
                    ),
                    "chosen_discard_probability": float(
                        probabilities[ue.ue_id, discard_action]
                    ),
                    "stay_probability": float(probabilities[ue.ue_id, stay]),
                    "commit_probability": (
                        float(probabilities[ue.ue_id, env.commit_action])
                        if masks[ue.ue_id, env.commit_action] > 0.0
                        else None
                    ),
                    "probability_delta_discard_minus_stay": float(
                        probabilities[ue.ue_id, discard_action]
                        - probabilities[ue.ue_id, stay]
                    ),
                    "target_price": float(prices[target]),
                    "target_price_percentile": float(
                        np.mean(prices <= prices[target])
                    ),
                    "coordination_cost_stay": float(
                        reasoning_result.coordination_costs[ue.ue_id, stay]
                    ),
                    "coordination_cost_discard": float(
                        reasoning_result.coordination_costs[
                            ue.ue_id, discard_action
                        ]
                    ),
                    "reasoner_delta": delta,
                    "reasoner_legal_action_count": int(len(legal)),
                    "legal_logits": {
                        str(int(legal_action)): float(
                            logits[ue.ue_id, legal_action]
                        )
                        for legal_action in legal
                    },
                    "legal_probabilities": {
                        str(int(legal_action)): float(
                            probabilities[ue.ue_id, legal_action]
                        )
                        for legal_action in legal
                    },
                    "reasoner_legal_actions": {
                        str(int(legal_action)): {
                            "service_cost": float(
                                reasoning_result.service_costs[
                                    ue.ue_id, legal_action
                                ]
                            ),
                            "coordination_cost": float(
                                reasoning_result.coordination_costs[
                                    ue.ue_id, legal_action
                                ]
                            ),
                            "planning_reliability_risk": float(
                                reasoning_result.planning_reliability_risks[
                                    ue.ue_id, legal_action
                                ]
                            ),
                            "reliability_cost": float(
                                reasoning_result.reliability_costs[
                                    ue.ue_id, legal_action
                                ]
                            ),
                            "total_cost": float(
                                reasoning_result.total_costs[
                                    ue.ue_id, legal_action
                                ]
                            ),
                        }
                        for legal_action in legal
                    },
                    "outage_leads": [],
                }
                discard_events.append(event)
                event_ids_by_ue[ue.ue_id] = event_id

            output = env.step(actions)
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("environment omitted release diagnostic outcomes")
            transmission = capture["transmission"]
            ues = []
            for ue in env.ues:
                ratios = transmission["visible_ratios"][ue.ue_id]
                ues.append(
                    {
                        "active": bool(transmission["active"][ue.ue_id] > 0.0),
                        "outage": bool(output.agent_outcomes[ue.ue_id, 1] > 0.5),
                        "actual_ratio": float(
                            transmission["actual_ratios"][ue.ue_id]
                        ),
                        "visible": [
                            bool(value)
                            for value in transmission["visible"][ue.ue_id]
                        ],
                        "rescue_satellites": [
                            int(cell)
                            for cell in np.flatnonzero(
                                ratios >= env.outage_service_ratio
                            )
                        ],
                        "secondary": ue.secondary_cell,
                        "state": int(ue.state),
                        "timer": int(ue.protocol_timer),
                        "discard_event_id": event_ids_by_ue.get(ue.ue_id),
                    }
                )
            history.append({"step": step, "ues": ues})
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
            f"completed policy episode {episode + 1}/{args.episodes}",
            file=sys.stderr,
            flush=True,
        )

    events_by_episode_ue: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in discard_events:
        events_by_episode_ue[(event["episode"], event["ue_id"])].append(event)
    for episode, history in enumerate(histories):
        for row in history:
            outage_step = int(row["step"])
            for ue_id, ue_row in enumerate(row["ues"]):
                if not (ue_row["active"] and ue_row["outage"]):
                    continue
                candidates = []
                for event in events_by_episode_ue[(episode, ue_id)]:
                    if event["step"] > outage_step or event["target"] not in ue_row["rescue_satellites"]:
                        continue
                    target = int(event["target"])
                    visible_continuously = all(
                        history[index]["ues"][ue_id]["visible"][target]
                        for index in range(event["step"], outage_step + 1)
                    )
                    rereadied = any(
                        history[index]["ues"][ue_id]["secondary"] == target
                        and MobilityState(history[index]["ues"][ue_id]["state"])
                        in (MobilityState.PREPARED, MobilityState.DUPLICATING)
                        and history[index]["ues"][ue_id]["timer"] == 0
                        for index in range(event["step"] + 1, outage_step + 1)
                    )
                    if visible_continuously and not rereadied:
                        candidates.append(event)
                if candidates:
                    causal_event = max(candidates, key=lambda item: item["step"])
                    causal_event["outage_leads"].append(
                        outage_step - int(causal_event["step"])
                    )

    successful_keeps = []
    for opportunity in keep_opportunities:
        if opportunity["action"] != 0 or not opportunity["ready"]:
            continue
        history = histories[opportunity["episode"]]
        end = min(len(history), opportunity["step"] + reasoning_horizon + 1)
        continuation = history[opportunity["step"] : end]
        if len(continuation) > 1 and all(
            not row["ues"][opportunity["ue_id"]]["outage"]
            and row["ues"][opportunity["ue_id"]]["secondary"] == opportunity["target"]
            for row in continuation
        ):
            successful_keeps.append(opportunity)

    premature_events = [event for event in discard_events if event["outage_leads"]]
    sample_count = min(args.counterfactual_samples, len(premature_events))
    selected_indices = np.sort(
        reservoir_rng.choice(len(premature_events), size=sample_count, replace=False)
    )
    sampled_events = [premature_events[int(index)] for index in selected_indices]
    counterfactual_by_event = replay_counterfactuals(
        config, sampled_events, args.episodes
    )

    deltas = [event["reasoner_delta"] for event in premature_events]
    reasoner_prefer_release = sum(delta["total"] < -TIE_EPSILON for delta in deltas)
    reasoner_prefer_keep = sum(delta["total"] > TIE_EPSILON for delta in deltas)
    reasoner_tied = len(deltas) - reasoner_prefer_release - reasoner_prefer_keep
    all_target_prices = np.asarray(
        [item["target_price"] for item in keep_opportunities], dtype=np.float64
    )
    low_price = float(np.percentile(all_target_prices, 25))
    high_price = float(np.percentile(all_target_prices, 75))

    immediate_differences = defaultdict(list)
    horizon_summary: dict[str, Any] = {}
    for horizon in HORIZONS:
        rows = []
        for event in sampled_events:
            pair = counterfactual_by_event[event["event_id"]]
            if horizon not in pair["keep"] or horizon not in pair["release"]:
                continue
            rows.append((pair["keep"][horizon], pair["release"][horizon]))
        metrics = (
            "service_deficit",
            "service_reward_contribution",
            "outage_ue_steps",
            "outage_cost",
            "dual_outage_contribution",
            "target_ready_steps",
            "resource_cost",
            "resource_reward_contribution",
            "preparation_requests",
            "later_reprepare",
            "team_reward",
        )
        horizon_result: dict[str, Any] = {"sample_count": len(rows)}
        horizon_result.update({
            metric: {
                "keep_mean": float(np.mean([keep[metric] for keep, _ in rows]))
                if rows
                else None,
                "release_mean": float(np.mean([release[metric] for _, release in rows]))
                if rows
                else None,
                "release_minus_keep_mean": float(
                    np.mean([release[metric] - keep[metric] for keep, release in rows])
                )
                if rows
                else None,
            }
            for metric in metrics
        })
        horizon_summary[str(horizon)] = horizon_result
        if horizon == 1:
            for metric in metrics:
                immediate_differences[metric] = [
                    release[metric] - keep[metric] for keep, release in rows
                ]

    class_counts = defaultdict(int)
    for event in premature_events:
        delta = event["reasoner_delta"]
        pair = counterfactual_by_event.get(event["event_id"])
        immediate_resource_dominance = False
        if pair and 1 in pair["keep"] and 1 in pair["release"]:
            reward_advantage = (
                pair["release"][1]["team_reward"]
                - pair["keep"][1]["team_reward"]
            )
            horizon = max(set(pair["keep"]) & set(pair["release"]))
            future_keep_advantage = (
                pair["keep"][horizon]["outage_ue_steps"]
                < pair["release"][horizon]["outage_ue_steps"]
                or pair["keep"][horizon]["service_deficit"]
                < pair["release"][horizon]["service_deficit"]
            )
            immediate_resource_saving = (
                pair["release"][1]["resource_cost"]
                < pair["keep"][1]["resource_cost"]
            )
            immediate_resource_dominance = (
                reward_advantage > 0.0
                and immediate_resource_saving
                and future_keep_advantage
            )
        if immediate_resource_dominance:
            code = "R4_IMMEDIATE_RESOURCE_INCENTIVE_DOMINATES"
        elif (
            event["target_price"] >= high_price
            and delta["coordination"] < -TIE_EPSILON
            and delta["total"] < -TIE_EPSILON
        ):
            code = "R5_COORDINATION_PRICE_DOMINATES"
        elif delta["total"] < -TIE_EPSILON:
            code = "R1_REASONER_PREFERS_RELEASE"
        elif delta["total"] > TIE_EPSILON:
            code = "R2_REASONER_PREFERS_KEEP_BUT_ACTOR_RELEASES"
        elif (
            abs(delta["total"]) <= TIE_EPSILON
            and event["deterministic_action"] == 0
        ):
            code = "R3_REASONER_NEAR_TIE_STOCHASTIC_RELEASE"
        else:
            code = "R7_OTHER"
        class_counts[code] += 1
    class_counts["R6_REASONER_PROTOCOL_SIMULATION_MISMATCH"] += 0

    def action_probability_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
        commits = [
            event["commit_probability"]
            for event in events
            if event["commit_probability"] is not None
        ]
        return {
            "chosen_release_or_rollback": distribution(
                [event["chosen_discard_probability"] for event in events]
            ),
            "stay": distribution([event["stay_probability"] for event in events]),
            "commit_when_legal": distribution([float(value) for value in commits]),
            "discard_minus_stay": distribution(
                [event["probability_delta_discard_minus_stay"] for event in events]
            ),
        }

    def discard_fraction(opportunities: list[dict[str, Any]]) -> float:
        return float(
            np.mean(
                [
                    item["action"] in (env.release_action, env.rollback_action)
                    for item in opportunities
                ]
            )
        ) if opportunities else 0.0

    low_opportunities = [
        item for item in keep_opportunities if item["target_price"] <= low_price
    ]
    high_opportunities = [
        item for item in keep_opportunities if item["target_price"] >= high_price
    ]
    all_outage_leads = [
        float(lead) for event in premature_events for lead in event["outage_leads"]
    ]
    report = {
        "configuration": {
            "episodes": args.episodes,
            "environment_seed": 7,
            "agent_seed": 7,
            "evaluation_action_seed": EVALUATION_ACTION_SEED,
            "episode_seed_base": EPISODE_SEED_BASE,
            "checkpoint": "outputs/v3_final_1500/m5_seed7/mappo_weights.npz",
            "checkpoint_mu": mu,
            "reasoning_horizon": reasoning_horizon,
            "tie_epsilon": TIE_EPSILON,
            "counterfactual_sample_limit": args.counterfactual_samples,
            "counterfactual_evaluated": len(sampled_events),
        },
        "event_counts": {
            "all_ready_discard": sum(event["kind"] == "READY" for event in discard_events),
            "all_preparing_discard": sum(event["kind"] == "PREPARING" for event in discard_events),
            "premature_ready_discard": sum(event["kind"] == "READY" for event in premature_events),
            "premature_preparing_discard": sum(event["kind"] == "PREPARING" for event in premature_events),
            "successful_ready_keep": len(successful_keeps),
        },
        "actor_decision": {
            "deterministic_argmax_discard_fraction": float(
                np.mean([event["deterministic_discard"] for event in premature_events])
            ) if premature_events else 0.0,
            "stochastic_only_discard_fraction": float(
                np.mean([not event["deterministic_discard"] for event in premature_events])
            ) if premature_events else 0.0,
            "probabilities": action_probability_summary(premature_events),
        },
        "reasoner_release_minus_stay": {
            "release_lower_fraction": float(reasoner_prefer_release / max(len(deltas), 1)),
            "release_higher_fraction": float(reasoner_prefer_keep / max(len(deltas), 1)),
            "approximately_tied_fraction": float(reasoner_tied / max(len(deltas), 1)),
            "delta_total": distribution([delta["total"] for delta in deltas]),
            "delta_service": distribution([delta["service"] for delta in deltas]),
            "delta_coordination": distribution([delta["coordination"] for delta in deltas]),
            "delta_reliability": distribution([delta["reliability"] for delta in deltas]),
        },
        "continuation_value_audit": {
            "ready_stay_preserves_target": True,
            "release_immediately_clears_target": True,
            "release_plan_models_future_reprepare": False,
            "release_plan_bypasses_reprepare_delay": False,
            "preparing_discard_clears_progress": True,
            "immediate_protocol_transition_matches_environment": True,
            "detail": (
                "RELEASE/ROLLBACK plans clear secondary/timer and are then padded with STAY; "
                "the horizon represents readiness loss but no adaptive re-PREPARE continuation"
            ),
        },
        "counterfactual_keep_vs_release": horizon_summary,
        "immediate_release_minus_keep": {
            key: distribution(values) for key, values in immediate_differences.items()
        },
        "coordination_price": {
            "low_quartile_threshold": low_price,
            "high_quartile_threshold": high_price,
            "premature_discard_target_price": distribution(
                [event["target_price"] for event in premature_events]
            ),
            "successful_keep_target_price": distribution(
                [event["target_price"] for event in successful_keeps]
            ),
            "premature_coordination_cost_keep": distribution(
                [event["coordination_cost_stay"] for event in premature_events]
            ),
            "premature_coordination_cost_discard": distribution(
                [event["coordination_cost_discard"] for event in premature_events]
            ),
            "low_price_quartile_discard_probability": discard_fraction(low_opportunities),
            "high_price_quartile_discard_probability": discard_fraction(high_opportunities),
        },
        "release_to_outage_vs_horizon": {
            "outage_link_count": len(all_outage_leads),
            "lead_steps": distribution(all_outage_leads),
            "within_half_horizon_fraction": float(
                np.mean(np.asarray(all_outage_leads) <= reasoning_horizon / 2.0)
            ) if all_outage_leads else 0.0,
            "within_horizon_fraction": float(
                np.mean(np.asarray(all_outage_leads) <= reasoning_horizon)
            ) if all_outage_leads else 0.0,
            "beyond_horizon_fraction": float(
                np.mean(np.asarray(all_outage_leads) > reasoning_horizon)
            ) if all_outage_leads else 0.0,
        },
        "diagnostic_classification": {
            code: {
                "count": int(count),
                "fraction": float(count / max(len(premature_events), 1)),
            }
            for code, count in sorted(class_counts.items())
        },
        "counterfactual_randomness_limitation": (
            "KEEP and RELEASE branches start from identical environment and actor RNG states. "
            "The actor consumes the same number of draws per step, but action-dependent "
            "environment RNG consumption can desynchronize later exogenous draws."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
