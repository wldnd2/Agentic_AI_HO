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

from diagnose_m5_reliability_alignment import (  # noqa: E402
    AGENT_SEED,
    EPISODE_SEED_BASE,
    EVALUATION_ACTION_SEED,
    HORIZONS,
    add_future_labels,
    choose_counterfactual_records,
    correlation,
    counterfactual_branch,
    event_flags,
    kendall,
    load_experiment,
    protocol_group,
    recurrent_policy,
    sample_actions,
    spearman,
)
from leo_mappo.coordinator import SatellitePriceCoordinator  # noqa: E402
from leo_mappo.pdcp import joint_delivery_probability  # noqa: E402
from leo_mappo.reasoning import (  # noqa: E402
    LookaheadReasoner,
    _HypotheticalState,
    _duplicated_stream_rates,
)
from leo_mappo.satellite import elevation_and_distance  # noqa: E402
from leo_mappo.ue import MobilityState  # noqa: E402


PLANNING_SCENARIO_SEED = 9107
SCENARIO_COUNT = 32
APPROX_ZERO = 1e-12


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "fraction_approximately_zero": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "fraction_approximately_zero": float(np.mean(array < APPROX_ZERO)),
    }


def sample_range(
    rng: np.random.Generator,
    specification: dict[str, Any],
    name: str,
    integer: bool,
) -> Any:
    bounds = specification.get(f"{name}_range")
    if bounds is None:
        return specification.get(name)
    low, high = bounds
    if integer:
        return int(rng.integers(int(low), int(high) + 1))
    return float(rng.uniform(float(low), float(high)))


def scenario_effects(
    env: Any,
    config: dict[str, Any],
    ue_id: int,
    current_step: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon = max(HORIZONS)
    loss = np.zeros((SCENARIO_COUNT, horizon, env.num_cells), dtype=np.float64)
    capacity = np.ones((SCENARIO_COUNT, horizon, env.num_cells), dtype=np.float64)
    target_traffic = np.ones((SCENARIO_COUNT, horizon), dtype=np.float64)
    aggregate_traffic = np.ones((SCENARIO_COUNT, horizon), dtype=np.float64)
    latent = config["v3"]["latent_degradation"]
    initial_probability = float(latent["initial_degraded_probability"])
    entry_probability = float(latent["entry_probability_per_step"])
    degraded_low, degraded_high = latent["degraded_duration_steps"]
    recovery_low, recovery_high = latent["recovering_duration_steps"]
    loss_low, loss_high = latent["loss_db_range"]

    for scenario in range(SCENARIO_COUNT):
        latent_state = np.zeros(env.num_cells, dtype=np.int8)
        latent_timer = np.zeros(env.num_cells, dtype=np.int64)
        latent_loss = np.zeros(env.num_cells, dtype=np.float64)
        initial = rng.random(env.num_cells) < initial_probability
        if np.any(initial):
            latent_state[initial] = 1
            latent_timer[initial] = rng.integers(
                int(degraded_low), int(degraded_high) + 1, size=np.sum(initial)
            )
            latent_loss[initial] = rng.uniform(
                float(loss_low), float(loss_high), size=np.sum(initial)
            )

        sampled_events = []
        for template in config.get("events", []):
            specification = deepcopy(template)
            for name, integer in (
                ("start_step", True),
                ("duration_steps", True),
                ("severity", False),
                ("ue_fraction", False),
            ):
                sampled = sample_range(rng, template, name, integer)
                if sampled is not None:
                    specification[name] = sampled
            choices = template.get("satellite_choices")
            if choices is not None:
                specification["satellite_id"] = int(rng.choice(choices))
            fraction = float(specification.get("ue_fraction", 1.0))
            count = min(env.num_agents, max(1, int(round(env.num_agents * fraction))))
            targets = set(
                int(value)
                for value in rng.choice(env.num_agents, size=count, replace=False)
            )
            sampled_events.append((specification, targets))

        for offset in range(horizon):
            loss[scenario, offset] += latent_loss
            traffic_multipliers = np.ones(env.num_agents, dtype=np.float64)
            absolute_step = current_step + offset
            for specification, targets in sampled_events:
                start = int(specification.get("start_step", 0))
                duration = int(specification.get("duration_steps", 0))
                if not start <= absolute_step < start + duration:
                    continue
                event_type = str(specification.get("type", ""))
                severity = float(specification.get("severity", 1.0))
                cell = int(specification.get("satellite_id", 0)) % env.num_cells
                if event_type == "blockage" and ue_id in targets:
                    loss[scenario, offset, cell] += severity
                elif event_type == "capacity_drop":
                    capacity[scenario, offset, cell] *= np.clip(
                        severity, 0.05, 1.0
                    )
                elif event_type == "traffic_burst":
                    traffic_multipliers[list(targets)] *= severity
            target_traffic[scenario, offset] = traffic_multipliers[ue_id]
            aggregate_traffic[scenario, offset] = float(
                np.mean(traffic_multipliers)
            )

            active = latent_state > 0
            latent_timer[active] -= 1
            degraded_done = (latent_state == 1) & (latent_timer <= 0)
            if np.any(degraded_done):
                latent_state[degraded_done] = 2
                latent_timer[degraded_done] = rng.integers(
                    int(recovery_low),
                    int(recovery_high) + 1,
                    size=np.sum(degraded_done),
                )
            recovering = latent_state == 2
            if np.any(recovering):
                steps = np.maximum(latent_timer[recovering], 1)
                latent_loss[recovering] *= (steps - 1) / steps
            recovered = recovering & (latent_timer <= 0)
            latent_state[recovered] = 0
            latent_loss[recovered] = 0.0
            starts = (latent_state == 0) & (
                rng.random(env.num_cells) < entry_probability
            )
            if np.any(starts):
                latent_state[starts] = 1
                latent_timer[starts] = rng.integers(
                    int(degraded_low),
                    int(degraded_high) + 1,
                    size=np.sum(starts),
                )
                latent_loss[starts] = rng.uniform(
                    float(loss_low), float(loss_high), size=np.sum(starts)
                )
    return loss, capacity, target_traffic, aggregate_traffic


def selected_action_plans(
    reasoner: LookaheadReasoner,
    env: Any,
    ue_id: int,
    masks: np.ndarray,
    prices: np.ndarray,
    required_rate: float,
    mu: float,
) -> dict[int, tuple[tuple[int, ...], int | None]]:
    available = np.ones(env.num_cells, dtype=np.float64)
    output = {}
    for action, plans in reasoner._plans(env, ue_id, masks[ue_id]).items():
        if len(plans) == 1:
            output[action] = plans[0]
            continue
        evaluations = [
            reasoner._evaluate_plan(
                env,
                ue_id,
                plan,
                target,
                required_rate,
                available,
                prices,
                mu,
                0.0,
                0.0,
            )
            for plan, target in plans
        ]
        output[action] = plans[
            min(range(len(plans)), key=lambda index: evaluations[index][0])
        ]
    return output


def scenario_leg(
    reasoner: LookaheadReasoner,
    env: Any,
    ue_id: int,
    cell_id: int | None,
    offset: int,
    required_rate: float,
    extra_loss_db: float,
    capacity_multiplier: float,
    traffic_load_multiplier: float,
) -> tuple[float, float]:
    if cell_id is None or required_rate <= 0.0:
        return 0.0, 0.0
    ue = env.ues[ue_id]
    satellite = env.satellites[cell_id]
    elevation, future_distance = elevation_and_distance(
        ue.position,
        satellite.position_ecef(env.time_s + offset * env.interval_s),
    )
    if elevation < env.min_elevation_rad:
        return 0.0, 0.0
    current = env.links[ue_id][cell_id]
    loss_delta = 20.0 * np.log10(
        future_distance / (current.distance_m + reasoner.epsilon)
    )
    sinr_db = env.observed_sinr[ue_id, cell_id] - loss_delta - extra_loss_db
    sinr_linear = 10.0 ** (np.clip(sinr_db, -120.0, 80.0) / 10.0)
    efficiency = min(
        float(env.channel_cfg["spectral_efficiency_cap"]),
        0.75 * np.log2(1.0 + sinr_linear),
    )
    success = 1.0 / (
        1.0 + np.exp(np.clip(-0.75 * (sinr_db + 2.0), -40.0, 40.0))
    )
    pressure = max(
        0.0,
        float(env.observed_load[ue_id, cell_id]) * traffic_load_multiplier,
    )
    load_scale = min(1.0, 1.0 / max(1.0, pressure, reasoner.epsilon))
    raw_capacity = (
        satellite.bandwidth_hz
        * efficiency
        * satellite.base_capacity_scale
        * capacity_multiplier
    )
    potential_rate = raw_capacity * load_scale
    return float(potential_rate), float(success)


def stochastic_action_risks(
    reasoner: LookaheadReasoner,
    env: Any,
    config: dict[str, Any],
    ue_id: int,
    masks: np.ndarray,
    prices: np.ndarray,
    required_rate: float,
    mu: float,
    rng: np.random.Generator,
) -> dict[int, dict[int, float]]:
    plans = selected_action_plans(
        reasoner, env, ue_id, masks, prices, required_rate, mu
    )
    loss, capacity, target_traffic, aggregate_traffic = scenario_effects(
        env, config, ue_id, env.step_index, rng
    )
    beta = float(env.outage_service_ratio)
    output = {horizon: {} for horizon in HORIZONS}
    for action, (plan, prepare_target) in plans.items():
        scenario_cumulative = np.zeros((SCENARIO_COUNT, max(HORIZONS)))
        for scenario in range(SCENARIO_COUNT):
            ue = env.ues[ue_id]
            state = _HypotheticalState(
                ue.primary_cell,
                ue.secondary_cell,
                ue.state,
                int(ue.protocol_timer),
                int(ue.last_switch_step),
            )
            cumulative = 0.0
            for offset, plan_action in enumerate(plan):
                reasoner._advance_hypothetical(env, state, offset, ue_id)
                target = (
                    prepare_target
                    if env.prepare_start
                    <= plan_action
                    < env.prepare_start + env.candidate_count
                    else None
                )
                reasoner._apply_hypothetical_action(
                    env, state, plan_action, target, ue_id, offset
                )
                scenario_required = required_rate * target_traffic[scenario, offset]
                primary_rate, primary_success = scenario_leg(
                    reasoner,
                    env,
                    ue_id,
                    state.primary_cell,
                    offset,
                    scenario_required,
                    loss[scenario, offset, state.primary_cell]
                    if state.primary_cell is not None
                    else 0.0,
                    capacity[scenario, offset, state.primary_cell]
                    if state.primary_cell is not None
                    else 1.0,
                    aggregate_traffic[scenario, offset],
                )
                potential_rate = primary_rate
                probability = primary_success
                if (
                    state.mobility_state == MobilityState.DUPLICATING
                    and state.secondary_cell is not None
                ):
                    secondary_rate, secondary_success = scenario_leg(
                        reasoner,
                        env,
                        ue_id,
                        state.secondary_cell,
                        offset,
                        scenario_required,
                        loss[scenario, offset, state.secondary_cell],
                        capacity[scenario, offset, state.secondary_cell],
                        aggregate_traffic[scenario, offset],
                    )
                    _, potential_rate = _duplicated_stream_rates(
                        scenario_required,
                        primary_rate,
                        scenario_required,
                        secondary_rate,
                    )
                    probability = joint_delivery_probability(
                        primary_success,
                        secondary_success,
                        failure_correlation=float(
                            env.duplication_cfg.get("failure_correlation", 0.15)
                        ),
                    )
                goodput = potential_rate * probability
                outage = float(goodput < beta * scenario_required)
                cumulative += reasoner.discount**offset * outage
                scenario_cumulative[scenario, offset] = cumulative
        for horizon in HORIZONS:
            output[horizon][action] = float(
                np.mean(scenario_cumulative[:, horizon - 1])
            )
    return output


def collect_state_records(
    config: dict[str, Any], env: Any, agent: Any, episodes: int
) -> list[dict[str, Any]]:
    reasoner = LookaheadReasoner(
        config["reasoning"],
        env.reward_cfg,
        config["coordination"],
        use_coordination=True,
        use_constraint=True,
    )
    coordinator = SatellitePriceCoordinator(env.num_cells, config["coordination"])
    agent.rng = np.random.default_rng(EVALUATION_ACTION_SEED)
    mu = float(agent.outage_dual_mu)
    normalized_mu = mu / float(config["constraint"]["mu_max"])
    original_schedule = env._schedule_and_transmit
    capture: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    def captured_schedule(
        required_rates: np.ndarray, active: np.ndarray
    ) -> dict[str, float]:
        delivered_before = np.asarray(
            [ue.delivered_packets for ue in env.ues], dtype=np.int64
        )
        metrics = original_schedule(required_rates, active)
        delivered_after = np.asarray(
            [ue.delivered_packets for ue in env.ues], dtype=np.int64
        )
        capture["active"] = active.copy()
        capture["actual_ratio"] = np.where(
            active > 0.0,
            (delivered_after - delivered_before)
            * int(env.traffic_cfg["packet_size_bits"])
            / max(env.interval_s, 1e-8)
            / (required_rates + env.reward_cfg["epsilon"]),
            0.0,
        )
        return metrics

    env._schedule_and_transmit = captured_schedule  # type: ignore[method-assign]
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
            prices = coordinator.prices
            _, required_rates = env.observed_required_rates()
            result = reasoner.analyze(env, masks, prices, mu)
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
                result.features,
                price_features,
                normalized_mu,
            )
            actions = sample_actions(probabilities, agent.rng)
            pending = []
            for ue in env.ues:
                legal = [int(value) for value in np.flatnonzero(masks[ue.ue_id] > 0.0)]
                risks = {
                    str(action): float(
                        result.planning_reliability_risks[ue.ue_id, action]
                    )
                    for action in legal
                }
                pending.append(
                    {
                        "episode": episode,
                        "step": step,
                        "ue_id": ue.ue_id,
                        "selected_action": int(actions[ue.ue_id]),
                        "legal_actions": legal,
                        "action_risks": risks,
                        "risk_spread": max(risks.values()) - min(risks.values()),
                        "risk": float(
                            result.planning_reliability_risks[
                                ue.ue_id, actions[ue.ue_id]
                            ]
                        ),
                        "protocol_group": protocol_group(
                            MobilityState(ue.state), int(ue.protocol_timer)
                        ),
                        "observed_required_rate": float(required_rates[ue.ue_id]),
                        "event_flags": event_flags(env, ue.ue_id),
                    }
                )
            output = env.step(actions)
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("environment omitted diagnostic outcomes")
            for ue_id in np.flatnonzero(capture["active"] > 0.0):
                row = pending[int(ue_id)]
                row["record_id"] = len(records)
                row["actual_ratio"] = float(capture["actual_ratio"][ue_id])
                row["actual_outage"] = bool(output.agent_outcomes[ue_id, 1] > 0.5)
                records.append(row)
            coordinator.update(output.requested_pressure)
            hidden_states = new_hidden
            previous_actions = actions.copy()
            previous_outcomes = output.agent_outcomes.copy()
            observations = output.observations
            masks = output.action_masks
            done = output.done
            step += 1
        print(
            f"completed state sampling episode {episode + 1}/{episodes}",
            file=sys.stderr,
            flush=True,
        )
    add_future_labels(records)
    return records


def replay_scenarios_and_counterfactuals(
    config: dict[str, Any], selected: list[dict[str, Any]], episodes: int
) -> tuple[
    dict[int, dict[int, dict[int, float]]],
    dict[int, dict[int, dict[int, dict[str, float]]]],
]:
    _, env, agent = load_experiment()
    reasoner = LookaheadReasoner(
        config["reasoning"],
        env.reward_cfg,
        config["coordination"],
        use_coordination=True,
        use_constraint=True,
    )
    coordinator = SatellitePriceCoordinator(env.num_cells, config["coordination"])
    agent.rng = np.random.default_rng(EVALUATION_ACTION_SEED)
    scenario_rng = np.random.default_rng(PLANNING_SCENARIO_SEED)
    mu = float(agent.outage_dual_mu)
    normalized_mu = mu / float(config["constraint"]["mu_max"])
    selected_by_step: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_step[(row["episode"], row["step"])].append(row)
    scenario_results = {}
    counterfactual_results = {}
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
            prices = coordinator.prices
            _, required_rates = env.observed_required_rates()
            result = reasoner.analyze(env, masks, prices, mu)
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
                result.features,
                price_features,
                normalized_mu,
            )
            actions = sample_actions(probabilities, agent.rng)
            selected_here = selected_by_step.get((episode, step), [])
            if selected_here:
                base_env = deepcopy(env)
                actor_rng_state = deepcopy(agent.rng.bit_generator.state)
                for row in selected_here:
                    ue_id = int(row["ue_id"])
                    if int(actions[ue_id]) != int(row["selected_action"]):
                        raise RuntimeError("scenario replay diverged")
                    scenario_results[row["record_id"]] = stochastic_action_risks(
                        reasoner,
                        env,
                        config,
                        ue_id,
                        masks,
                        prices,
                        float(required_rates[ue_id]),
                        mu,
                        scenario_rng,
                    )
                    snapshot = {
                        "env": deepcopy(base_env),
                        "coordinator": deepcopy(coordinator),
                        "ue_id": ue_id,
                        "sampled_actions": actions.copy(),
                        "new_hidden": new_hidden.copy(),
                        "observations": observations.copy(),
                        "masks": masks.copy(),
                        "previous_actions": previous_actions.copy(),
                        "previous_outcomes": previous_outcomes.copy(),
                        "actor_rng_state": deepcopy(actor_rng_state),
                        "mu": mu,
                        "normalized_mu": normalized_mu,
                    }
                    counterfactual_results[row["record_id"]] = {
                        action: counterfactual_branch(snapshot, agent, config, action)
                        for action in row["legal_actions"]
                    }
                    completed += 1
                    if completed % 25 == 0 or completed == len(selected):
                        print(
                            f"completed scenario/counterfactual state "
                            f"{completed}/{len(selected)}",
                            file=sys.stderr,
                            flush=True,
                        )
            output = env.step(actions)
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("scenario replay omitted outcomes")
            coordinator.update(output.requested_pressure)
            hidden_states = new_hidden
            previous_actions = actions.copy()
            previous_outcomes = output.agent_outcomes.copy()
            observations = output.observations
            masks = output.action_masks
            done = output.done
            step += 1
    return scenario_results, counterfactual_results


def regret(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "p90": float(np.percentile(array, 90)) if len(array) else None,
    }


def comparison_summary(
    selected: list[dict[str, Any]],
    scenarios: dict[int, dict[int, dict[int, float]]],
    counterfactuals: dict[int, dict[int, dict[int, dict[str, float]]]],
) -> dict[str, Any]:
    output = {}
    for horizon in HORIZONS:
        accumulators: dict[str, list[Any]] = defaultdict(list)
        strict_pairs = defaultdict(int)
        evaluated = 0
        for row in selected:
            legal = row["legal_actions"]
            branches = counterfactuals[row["record_id"]]
            if any(horizon not in branches[action] for action in legal):
                continue
            current = np.asarray(
                [row["action_risks"][str(action)] for action in legal]
            )
            scenario = np.asarray(
                [scenarios[row["record_id"]][horizon][action] for action in legal]
            )
            outage = np.asarray(
                [branches[action][horizon]["target_outage_steps"] for action in legal]
            )
            deficit = np.asarray(
                [branches[action][horizon]["service_deficit"] for action in legal]
            )
            current_choice = int(np.argmin(current))
            scenario_choice = int(np.argmin(scenario))
            accumulators["current_outage_regret"].append(
                float(outage[current_choice] - np.min(outage))
            )
            accumulators["scenario_outage_regret"].append(
                float(outage[scenario_choice] - np.min(outage))
            )
            accumulators["current_deficit_regret"].append(
                float(deficit[current_choice] - np.min(deficit))
            )
            accumulators["scenario_deficit_regret"].append(
                float(deficit[scenario_choice] - np.min(deficit))
            )
            accumulators["current_outage_agree_all"].append(
                outage[current_choice] <= np.min(outage) + 1e-12
            )
            accumulators["scenario_outage_agree_all"].append(
                outage[scenario_choice] <= np.min(outage) + 1e-12
            )
            if np.ptp(outage) > 1e-12:
                accumulators["current_outage_agree_nontied"].append(
                    outage[current_choice] <= np.min(outage) + 1e-12
                )
                accumulators["scenario_outage_agree_nontied"].append(
                    outage[scenario_choice] <= np.min(outage) + 1e-12
                )
                for name, risks in (("current", current), ("scenario", scenario)):
                    value = spearman(risks, outage)
                    tau = kendall(risks, outage)
                    if value is not None:
                        accumulators[f"{name}_outage_spearman"].append(value)
                    if tau is not None:
                        accumulators[f"{name}_outage_kendall"].append(tau)
            if np.ptp(deficit) > 1e-12:
                accumulators["current_deficit_agree_nontied"].append(
                    deficit[current_choice] <= np.min(deficit) + 1e-12
                )
                accumulators["scenario_deficit_agree_nontied"].append(
                    deficit[scenario_choice] <= np.min(deficit) + 1e-12
                )
                for name, risks in (("current", current), ("scenario", scenario)):
                    value = spearman(risks, deficit)
                    tau = kendall(risks, deficit)
                    if value is not None:
                        accumulators[f"{name}_deficit_spearman"].append(value)
                    if tau is not None:
                        accumulators[f"{name}_deficit_kendall"].append(tau)
            for first in range(len(legal)):
                for second in range(first + 1, len(legal)):
                    for outcome_name, actual in (("outage", outage), ("deficit", deficit)):
                        actual_delta = actual[first] - actual[second]
                        if abs(actual_delta) <= 1e-12:
                            continue
                        for risk_name, risks in (("current", current), ("scenario", scenario)):
                            strict_pairs[f"{outcome_name}_pairs"] += int(
                                risk_name == "current"
                            )
                            predicted_delta = risks[first] - risks[second]
                            strict_pairs[
                                f"{risk_name}_{outcome_name}_correct"
                            ] += int(predicted_delta * actual_delta > 0.0)
            evaluated += 1

        def mean(name: str) -> float | None:
            return (
                float(np.mean(accumulators[name]))
                if accumulators[name]
                else None
            )

        outage_pair_count = strict_pairs["outage_pairs"]
        deficit_pair_count = strict_pairs["deficit_pairs"]
        output[str(horizon)] = {
            "state_count": evaluated,
            "all_state_lowest_outage_agreement": {
                "current": mean("current_outage_agree_all"),
                "scenario": mean("scenario_outage_agree_all"),
            },
            "non_tied_outage_state_count": len(
                accumulators["current_outage_agree_nontied"]
            ),
            "non_tied_lowest_outage_agreement": {
                "current": mean("current_outage_agree_nontied"),
                "scenario": mean("scenario_outage_agree_nontied"),
            },
            "non_tied_outage_rank": {
                "current_spearman": mean("current_outage_spearman"),
                "scenario_spearman": mean("scenario_outage_spearman"),
                "current_kendall": mean("current_outage_kendall"),
                "scenario_kendall": mean("scenario_outage_kendall"),
            },
            "non_tied_deficit_state_count": len(
                accumulators["current_deficit_agree_nontied"]
            ),
            "non_tied_lowest_deficit_agreement": {
                "current": mean("current_deficit_agree_nontied"),
                "scenario": mean("scenario_deficit_agree_nontied"),
            },
            "non_tied_deficit_rank": {
                "current_spearman": mean("current_deficit_spearman"),
                "scenario_spearman": mean("scenario_deficit_spearman"),
                "current_kendall": mean("current_deficit_kendall"),
                "scenario_kendall": mean("scenario_deficit_kendall"),
            },
            "strict_pairwise": {
                "outage_pair_count": outage_pair_count,
                "current_outage_accuracy": strict_pairs[
                    "current_outage_correct"
                ] / max(outage_pair_count, 1),
                "scenario_outage_accuracy": strict_pairs[
                    "scenario_outage_correct"
                ] / max(outage_pair_count, 1),
                "deficit_pair_count": deficit_pair_count,
                "current_deficit_accuracy": strict_pairs[
                    "current_deficit_correct"
                ] / max(deficit_pair_count, 1),
                "scenario_deficit_accuracy": strict_pairs[
                    "scenario_deficit_correct"
                ] / max(deficit_pair_count, 1),
            },
            "outage_regret": {
                "current": regret(accumulators["current_outage_regret"]),
                "scenario": regret(accumulators["scenario_outage_regret"]),
            },
            "deficit_regret": {
                "current": regret(accumulators["current_deficit_regret"]),
                "scenario": regret(accumulators["scenario_deficit_regret"]),
            },
        }
    return output


def event_family_quality(
    selected: list[dict[str, Any]],
    scenarios: dict[int, dict[int, dict[int, float]]],
    counterfactuals: dict[int, dict[int, dict[int, dict[str, float]]]],
) -> dict[str, Any]:
    output = {}
    horizon = max(HORIZONS)
    for event_name in (
        "latent_degraded",
        "recovering",
        "random_blockage",
        "capacity_drop",
        "traffic_burst",
        "switching_transition",
    ):
        state_count = 0
        pair_count = 0
        current_correct = 0
        scenario_correct = 0
        current_correlations = []
        scenario_correlations = []
        for row in selected:
            if not row["event_flags"][event_name]:
                continue
            legal = row["legal_actions"]
            branches = counterfactuals[row["record_id"]]
            if any(horizon not in branches[action] for action in legal):
                continue
            actual = np.asarray(
                [branches[action][horizon]["target_outage_steps"] for action in legal]
            )
            if np.ptp(actual) <= 1e-12:
                continue
            current = np.asarray(
                [row["action_risks"][str(action)] for action in legal]
            )
            scenario = np.asarray(
                [scenarios[row["record_id"]][horizon][action] for action in legal]
            )
            for destination, risks in (
                (current_correlations, current),
                (scenario_correlations, scenario),
            ):
                value = spearman(risks, actual)
                if value is not None:
                    destination.append(value)
            for first in range(len(legal)):
                for second in range(first + 1, len(legal)):
                    actual_delta = actual[first] - actual[second]
                    if abs(actual_delta) <= 1e-12:
                        continue
                    pair_count += 1
                    current_correct += int(
                        (current[first] - current[second]) * actual_delta > 0.0
                    )
                    scenario_correct += int(
                        (scenario[first] - scenario[second]) * actual_delta > 0.0
                    )
            state_count += 1
        output[event_name] = {
            "non_tied_state_count": state_count,
            "strict_pair_count": pair_count,
            "current_mean_spearman": float(np.mean(current_correlations))
            if current_correlations
            else None,
            "scenario_mean_spearman": float(np.mean(scenario_correlations))
            if scenario_correlations
            else None,
            "current_strict_pair_accuracy": current_correct / max(pair_count, 1),
            "scenario_strict_pair_accuracy": scenario_correct / max(pair_count, 1),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a diagnostic stochastic M5 reliability candidate"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--states", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0 or args.states <= 0:
        raise ValueError("episodes and states must be positive")

    config, env, agent = load_experiment()
    records = collect_state_records(config, env, agent, args.episodes)
    selected = choose_counterfactual_records(records, args.states)
    scenarios, counterfactuals = replay_scenarios_and_counterfactuals(
        config, selected, args.episodes
    )
    comparison = comparison_summary(selected, scenarios, counterfactuals)
    current_values = [
        row["action_risks"][str(action)]
        for row in selected
        for action in row["legal_actions"]
    ]
    scenario_values = [
        scenarios[row["record_id"]][max(HORIZONS)][action]
        for row in selected
        for action in row["legal_actions"]
    ]
    report = {
        "configuration": {
            "episodes": args.episodes,
            "sampled_states": len(selected),
            "action_evaluations": len(current_values),
            "scenario_count_per_action": SCENARIO_COUNT,
            "total_scenario_action_evaluations": len(current_values)
            * SCENARIO_COUNT,
            "planning_scenario_seed": PLANNING_SCENARIO_SEED,
            "environment_seed": 7,
            "agent_seed": AGENT_SEED,
            "evaluation_action_seed": EVALUATION_ACTION_SEED,
            "checkpoint": (
                "outputs/v3_validation_420/m5_marginal_headroom/mappo_weights.npz"
            ),
            "checkpoint_mu": float(agent.outage_dual_mu),
            "horizons": list(HORIZONS),
        },
        "sample_composition": {
            "pre_outage_1_3": sum(row["pre_outage_1_3"] for row in selected),
            "preparing_ready": sum(
                row["protocol_group"] == "PREPARING_PREPARED_READY"
                for row in selected
            ),
            "nonzero_current_risk_spread": sum(
                row["risk_spread"] > APPROX_ZERO for row in selected
            ),
        },
        "saturation": {
            "approximately_zero_threshold": APPROX_ZERO,
            "current": distribution(current_values),
            "scenario_h6": distribution(scenario_values),
        },
        "comparison": comparison,
        "event_family_h6": event_family_quality(
            selected, scenarios, counterfactuals
        ),
        "scenario_information_audit": {
            "uses_current_observed_sinr": True,
            "uses_current_observed_load": True,
            "uses_ephemeris_and_protocol_state": True,
            "uses_configured_event_distributions": True,
            "uses_configured_latent_transition_distribution": True,
            "uses_true_current_latent_state": False,
            "uses_actual_future_events": False,
            "copies_environment_future_rng_state": False,
            "actual_event_flags_used_only_for_posthoc_evaluation": True,
        },
        "counterfactual_randomness_limitation": (
            "Legal-action branches start from identical environment and actor RNG states, "
            "with other UEs' first actions fixed. Action-dependent environment RNG "
            "consumption can desynchronize later exogenous draws."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
