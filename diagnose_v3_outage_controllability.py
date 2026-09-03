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

from leo_mappo import LEOMobilityEnv, MAPPO, load_config  # noqa: E402
from leo_mappo.coordinator import SatellitePriceCoordinator  # noqa: E402
from leo_mappo.pdcp import joint_delivery_probability  # noqa: E402
from leo_mappo.reasoning import (  # noqa: E402
    LookaheadReasoner,
    _HypotheticalState,
    _duplicated_stream_rates,
)
from leo_mappo.scheduler import LegRequest  # noqa: E402
from leo_mappo.ue import MobilityState  # noqa: E402


ENVIRONMENT_SEED = 7
AGENT_SEED = 7
EVALUATION_ACTION_SEED = 7007
EPISODE_SEED_BASE = 10_000
DIAGNOSTIC_MU = 0.325419922
LOWER_RISK_THRESHOLD = 1e-4


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def load_experiment() -> tuple[dict[str, Any], LEOMobilityEnv, MAPPO]:
    config = load_config(
        ROOT / "configs/default.yaml",
        [
            ROOT / "configs/validation_agentic_challenge_v3.yaml",
            ROOT / "configs/eta_handover_005.yaml",
            ROOT / "configs/validation_dwell_15.yaml",
            ROOT / "configs/full_agentic.yaml",
            ROOT / "configs/dual_mid_0005.yaml",
        ],
    )
    config["environment_seed"] = ENVIRONMENT_SEED
    config["agent_seed"] = AGENT_SEED
    env = LEOMobilityEnv(config)
    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        config["mappo"],
        seed=AGENT_SEED,
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )
    checkpoint = ROOT / "outputs/v3_final_1500/m5_seed7/mappo_weights.npz"
    agent.load(checkpoint)
    if not (
        agent.use_memory
        and agent.use_reasoning
        and agent.use_coordination
        and agent.use_constraint
    ):
        raise RuntimeError("checkpoint/config combination is not M5")
    return config, env, agent


def protocol_snapshot(env: LEOMobilityEnv) -> list[dict[str, int | None]]:
    return [
        {
            "primary": ue.primary_cell,
            "secondary": ue.secondary_cell,
            "state": int(ue.state),
            "timer": int(ue.protocol_timer),
        }
        for ue in env.ues
    ]


def usable_true_leg(env: LEOMobilityEnv, ue_id: int, cell_id: int | None) -> bool:
    if cell_id is None:
        return False
    link = env.links[ue_id][cell_id]
    unit_capacity = (
        env.satellites[cell_id].bandwidth_hz
        * link.spectral_efficiency
        * env.satellites[cell_id].effective_capacity_scale
    )
    return bool(
        link.visible
        and link.spectral_efficiency > env.min_schedulable_spectral_efficiency
        and unit_capacity > env.epsilon_capacity
    )


def actual_requests_except(
    env: LEOMobilityEnv,
    required_rates: np.ndarray,
    excluded_ue: int,
) -> list[LegRequest]:
    requests: list[LegRequest] = []
    for ue in env.ues:
        if ue.ue_id == excluded_ue or required_rates[ue.ue_id] <= 0.0:
            continue
        cells = [(ue.primary_cell, False)]
        if ue.state == MobilityState.DUPLICATING:
            cells.append((ue.secondary_cell, True))
        for cell_id, secondary in cells:
            if not usable_true_leg(env, ue.ue_id, cell_id):
                continue
            assert cell_id is not None
            requests.append(
                LegRequest(
                    ue.ue_id,
                    cell_id,
                    env.links[ue.ue_id][cell_id].spectral_efficiency,
                    ue.qos_priority,
                    float(required_rates[ue.ue_id]),
                    secondary,
                )
            )
    return requests


def ready_options(
    env: LEOMobilityEnv,
    ue_id: int,
    before_action: dict[str, int | None],
) -> list[tuple[int, ...]]:
    primary = before_action["primary"]
    secondary = before_action["secondary"]
    state = MobilityState(int(before_action["state"]))
    timer = int(before_action["timer"])
    options: list[tuple[int, ...]] = []
    if usable_true_leg(env, ue_id, primary):
        assert primary is not None
        options.append((primary,))
    duplication_ready = (
        state == MobilityState.DUPLICATING
        or (state == MobilityState.PREPARED and timer == 0)
    )
    if (
        duplication_ready
        and usable_true_leg(env, ue_id, primary)
        and usable_true_leg(env, ue_id, secondary)
    ):
        assert primary is not None and secondary is not None
        options.append((primary, secondary))
    return options


def target_reservation(
    env: LEOMobilityEnv,
    ue_id: int,
    before_action: dict[str, int | None],
) -> np.ndarray:
    reservation = np.zeros(env.num_cells, dtype=np.float64)
    rho = float(env.protocol_cfg["preparation_resource_fraction"])
    for ue in env.ues:
        if ue.ue_id == ue_id:
            continue
        if ue.secondary_cell is not None and ue.state in (
            MobilityState.PREPARED,
            MobilityState.DUPLICATING,
            MobilityState.SWITCHING,
        ):
            reservation[ue.secondary_cell] += rho
    state = MobilityState(int(before_action["state"]))
    secondary = before_action["secondary"]
    if secondary is not None and state in (
        MobilityState.PREPARED,
        MobilityState.DUPLICATING,
        MobilityState.SWITCHING,
    ):
        reservation[int(secondary)] += rho
    return np.clip(reservation, 0.0, 1.0)


def scheduled_option_ratio(
    env: LEOMobilityEnv,
    ue_id: int,
    required_rates: np.ndarray,
    before_action: dict[str, int | None],
    option: tuple[int, ...],
) -> float:
    required_rate = float(required_rates[ue_id])
    if required_rate <= 0.0 or not option:
        return 0.0
    requests = actual_requests_except(env, required_rates, ue_id)
    for index, cell_id in enumerate(option):
        requests.append(
            LegRequest(
                ue_id,
                cell_id,
                env.links[ue_id][cell_id].spectral_efficiency,
                env.ues[ue_id].qos_priority,
                required_rate,
                index > 0,
            )
        )
    bandwidth = np.asarray(
        [satellite.bandwidth_hz for satellite in env.satellites], dtype=np.float64
    )
    capacity_scale = np.asarray(
        [satellite.effective_capacity_scale for satellite in env.satellites],
        dtype=np.float64,
    )
    reservation = target_reservation(env, ue_id, before_action)
    schedule = env.scheduler.allocate(
        requests,
        bandwidth,
        capacity_scale,
        1.0 - reservation,
    )
    rates = [schedule.rates_bps.get((ue_id, cell_id), 0.0) for cell_id in option]
    if len(option) == 1:
        rate = rates[0]
        probability = env.links[ue_id][option[0]].packet_success_probability
    else:
        rate = min(rates)
        probability = joint_delivery_probability(
            env.links[ue_id][option[0]].packet_success_probability,
            env.links[ue_id][option[1]].packet_success_probability,
            failure_correlation=float(
                env.duplication_cfg.get("failure_correlation", 0.15)
            ),
        )
    return float(rate * probability / (required_rate + env.reward_cfg["epsilon"]))


def ready_set_oracle_ratio(
    env: LEOMobilityEnv,
    ue_id: int,
    required_rates: np.ndarray,
    before_action: dict[str, int | None],
) -> float:
    options = ready_options(env, ue_id, before_action)
    return max(
        (
            scheduled_option_ratio(
                env, ue_id, required_rates, before_action, option
            )
            for option in options
        ),
        default=0.0,
    )


def visible_set_optimistic_ratio(
    env: LEOMobilityEnv, ue_id: int, required_rate: float
) -> float:
    if required_rate <= 0.0:
        return 0.0
    ratios = []
    for cell_id, satellite in enumerate(env.satellites):
        if not usable_true_leg(env, ue_id, cell_id):
            continue
        link = env.links[ue_id][cell_id]
        capacity = (
            satellite.bandwidth_hz
            * link.spectral_efficiency
            * satellite.effective_capacity_scale
        )
        rate = min(required_rate, capacity)
        ratios.append(
            rate
            * link.packet_success_probability
            / (required_rate + env.reward_cfg["epsilon"])
        )
    return float(max(ratios, default=0.0))


def plan_potential_ratio(
    reasoner: LookaheadReasoner,
    env: LEOMobilityEnv,
    ue_id: int,
    plan: tuple[int, ...],
    prepare_target: int | None,
    required_rate: float,
    available: np.ndarray,
) -> float:
    if required_rate <= 0.0:
        return 0.0
    ue = env.ues[ue_id]
    state = _HypotheticalState(
        ue.primary_cell,
        ue.secondary_cell,
        ue.state,
        int(ue.protocol_timer),
        int(ue.last_switch_step),
    )
    weighted_ratio = 0.0
    weight_sum = 0.0
    for offset, action in enumerate(plan):
        reasoner._advance_hypothetical(env, state, offset, ue_id)
        target = (
            prepare_target
            if env.prepare_start <= action < env.prepare_start + env.candidate_count
            else None
        )
        reasoner._apply_hypothetical_action(
            env, state, action, target, ue_id, offset
        )
        _, primary_potential, primary_success, _ = reasoner._predicted_leg(
            env, ue_id, state.primary_cell, offset, required_rate, available
        )
        potential_rate = primary_potential
        probability = primary_success
        if (
            state.mobility_state == MobilityState.DUPLICATING
            and state.secondary_cell is not None
        ):
            _, secondary_potential, secondary_success, _ = reasoner._predicted_leg(
                env, ue_id, state.secondary_cell, offset, required_rate, available
            )
            _, potential_rate = _duplicated_stream_rates(
                required_rate,
                primary_potential,
                required_rate,
                secondary_potential,
            )
            probability = joint_delivery_probability(
                primary_success,
                secondary_success,
                failure_correlation=0.15,
            )
        weight = reasoner.discount**offset
        weighted_ratio += weight * potential_rate * probability / (
            required_rate + reasoner.epsilon
        )
        weight_sum += weight
    return float(weighted_ratio / max(weight_sum, reasoner.epsilon))


def legal_action_potential_ratios(
    reasoner: LookaheadReasoner,
    env: LEOMobilityEnv,
    masks: np.ndarray,
    prices: np.ndarray,
    required_rates: np.ndarray,
    outage_dual_mu: float,
) -> list[np.ndarray]:
    available = np.ones(env.num_cells, dtype=np.float64)
    output: list[np.ndarray] = []
    for ue in env.ues:
        plans = reasoner._plans(env, ue.ue_id, masks[ue.ue_id])
        action_ratios = np.zeros(env.action_dim, dtype=np.float64)
        for action, action_plans in plans.items():
            if len(action_plans) == 1:
                chosen_plan, chosen_target = action_plans[0]
            else:
                evaluated = [
                    reasoner._evaluate_plan(
                        env,
                        ue.ue_id,
                        plan,
                        target,
                        float(required_rates[ue.ue_id]),
                        available,
                        prices,
                        outage_dual_mu,
                        0.0,
                        0.0,
                    )
                    for plan, target in action_plans
                ]
                chosen = min(range(len(evaluated)), key=lambda index: evaluated[index][0])
                chosen_plan, chosen_target = action_plans[chosen]
            action_ratios[action] = plan_potential_ratio(
                reasoner,
                env,
                ue.ue_id,
                chosen_plan,
                chosen_target,
                float(required_rates[ue.ue_id]),
                available,
            )
        output.append(action_ratios)
    return output


def analytical_records(
    env: LEOMobilityEnv,
    reasoner: LookaheadReasoner,
    masks: np.ndarray,
    prices: np.ndarray,
    mu: float,
) -> tuple[list[dict[str, Any]], Any]:
    _, required_rates = env.observed_required_rates()
    reference = reasoner.analyze(env, masks, prices, 0.0)
    with_mu = reasoner.analyze(env, masks, prices, mu)
    potential_ratios = legal_action_potential_ratios(
        reasoner, env, masks, prices, required_rates, mu
    )
    records = []
    for ue_id in range(env.num_agents):
        legal = np.flatnonzero(masks[ue_id] > 0.0)
        risks = reference.planning_reliability_risks[ue_id, legal]
        reference_costs = reference.total_costs[ue_id, legal]
        best_position = int(np.argmin(reference_costs))
        lower_reduction = float(
            max(0.0, risks[best_position] - float(np.min(risks)))
        )
        mu_risks = with_mu.planning_reliability_risks[ue_id, legal]
        ratios = potential_ratios[ue_id][legal]
        records.append(
            {
                "risk_spread": float(np.max(risks) - np.min(risks)),
                "lower_risk_reduction": lower_reduction,
                "service_spread": float(
                    np.max(with_mu.service_costs[ue_id, legal])
                    - np.min(with_mu.service_costs[ue_id, legal])
                ),
                "coordination_spread": float(
                    np.max(with_mu.coordination_costs[ue_id, legal])
                    - np.min(with_mu.coordination_costs[ue_id, legal])
                ),
                "reliability_contribution_spread": float(
                    mu * (np.max(mu_risks) - np.min(mu_risks))
                ),
                "potential_ratios": ratios.tolist(),
                "observed_required_rate": float(required_rates[ue_id]),
            }
        )
    return records, with_mu


def event_flags(
    env: LEOMobilityEnv,
    ue_id: int,
    visible_ratio: float,
) -> dict[str, bool]:
    ue = env.ues[ue_id]
    serving_cells = [ue.primary_cell]
    if ue.state == MobilityState.DUPLICATING:
        serving_cells.append(ue.secondary_cell)
    cells = [cell for cell in serving_cells if cell is not None]
    return {
        "latent_degraded": any(
            env.latent_degradation_state[ue_id, cell] == 1 for cell in cells
        ),
        "recovering": any(
            env.latent_degradation_state[ue_id, cell] == 2 for cell in cells
        ),
        "random_blockage": bool(
            np.any(env._active_effects.link_extra_loss_db[ue_id] > 0.0)
        ),
        "capacity_drop": any(
            env._active_effects.cell_capacity_multiplier[cell] < 1.0
            for cell in cells
        ),
        "traffic_burst": bool(
            env._active_effects.traffic_multiplier[ue_id] > 1.0
        ),
        "no_sufficiently_strong_visible_candidate": visible_ratio < env.outage_service_ratio,
        "switching_or_transition": ue.state == MobilityState.SWITCHING,
        "duplication_active": ue.state == MobilityState.DUPLICATING,
    }


def subset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state_count": len(records),
        "risk_spread": distribution([row["risk_spread"] for row in records]),
        "lower_risk_legal_alternative_fraction": float(
            np.mean(
                [row["lower_risk_reduction"] > LOWER_RISK_THRESHOLD for row in records]
            )
        )
        if records
        else 0.0,
        "service_cost_spread": distribution(
            [row["service_spread"] for row in records]
        ),
        "coordination_cost_spread": distribution(
            [row["coordination_spread"] for row in records]
        ),
        "mu_reliability_contribution_spread": distribution(
            [row["reliability_contribution_spread"] for row in records]
        ),
    }


def headroom_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in records if row["observed_required_rate"] > 0.0]
    best = [max(row["potential_ratios"]) for row in usable]
    medians = [float(np.median(row["potential_ratios"])) for row in usable]
    spreads = [
        max(row["potential_ratios"]) - min(row["potential_ratios"])
        for row in usable
    ]
    return {
        "states_with_observed_required_rate": len(usable),
        "best_legal_ratio": distribution(best),
        "median_legal_ratio": distribution(medians),
        "max_minus_min_ratio_spread": distribution(spreads),
        **{
            f"fraction_with_any_legal_ratio_ge_{threshold:g}": float(
                np.mean(np.asarray(best) >= threshold)
            )
            if best
            else 0.0
            for threshold in (0.8, 1.0, 1.2)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose V3 outage controllability")
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
    all_records: list[dict[str, Any]] = []

    original_apply_actions = env._apply_actions
    original_schedule = env._schedule_and_transmit
    capture: dict[str, Any] = {}

    def captured_apply_actions(actions: np.ndarray, masks: np.ndarray) -> dict[str, int]:
        capture["protocol_before_action"] = protocol_snapshot(env)
        return original_apply_actions(actions, masks)

    def captured_schedule(
        required_rates: np.ndarray, active: np.ndarray
    ) -> dict[str, float]:
        before_action = capture["protocol_before_action"]
        ready = np.zeros(env.num_agents, dtype=np.float64)
        visible = np.zeros(env.num_agents, dtype=np.float64)
        flags = []
        for ue_id in range(env.num_agents):
            if active[ue_id] <= 0.0:
                flags.append({})
                continue
            ready[ue_id] = ready_set_oracle_ratio(
                env, ue_id, required_rates, before_action[ue_id]
            )
            visible[ue_id] = visible_set_optimistic_ratio(
                env, ue_id, float(required_rates[ue_id])
            )
            flags.append(event_flags(env, ue_id, visible[ue_id]))
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
        actual_ratio = np.where(
            active > 0.0,
            realized_goodput / (required_rates + env.reward_cfg["epsilon"]),
            0.0,
        )
        capture["true_step"] = {
            "active": active.copy(),
            "actual_ratio": actual_ratio,
            "ready_ratio": ready,
            "visible_ratio": visible,
            "event_flags": flags,
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
        episode_records: list[dict[str, Any]] = []
        done = False
        step = 0
        while not done:
            prices = coordinator.prices
            analytical, reasoning_result = analytical_records(
                env, reasoner, masks, prices, DIAGNOSTIC_MU
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
            rejection_before = env.preparation_rejections_by_ue.copy()
            output = env.step(actions)
            true_step = capture["true_step"]
            rejection = env.preparation_rejections_by_ue - rejection_before
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("environment omitted required diagnostic outputs")
            for ue_id in np.flatnonzero(true_step["active"] > 0.0):
                actual_outage = bool(output.agent_outcomes[ue_id, 1] > 0.5)
                recomputed_outage = bool(
                    true_step["actual_ratio"][ue_id] < env.outage_service_ratio
                )
                if actual_outage != recomputed_outage:
                    raise RuntimeError("captured actual outage disagrees with environment")
                row = dict(analytical[ue_id])
                row.update(
                    {
                        "episode": episode,
                        "step": step,
                        "ue_id": int(ue_id),
                        "actual_ratio": float(true_step["actual_ratio"][ue_id]),
                        "ready_ratio": float(true_step["ready_ratio"][ue_id]),
                        "visible_ratio": float(true_step["visible_ratio"][ue_id]),
                        "actual_outage": actual_outage,
                        "event_flags": {
                            **true_step["event_flags"][ue_id],
                            "preparation_rejection": bool(rejection[ue_id] > 0),
                        },
                    }
                )
                episode_records.append(row)
            coordinator.update(output.requested_pressure)
            hidden_states = new_hidden
            previous_actions = actions.copy()
            previous_outcomes = output.agent_outcomes.copy()
            observations = output.observations
            masks = output.action_masks
            done = output.done
            step += 1

        lookup = {(row["step"], row["ue_id"]): row for row in episode_records}
        for row in episode_records:
            row["pre_outage"] = (
                not row["actual_outage"]
                and any(
                    lookup.get((row["step"] + lead, row["ue_id"]), {}).get(
                        "actual_outage", False
                    )
                    for lead in range(1, 6)
                )
            )
        all_records.extend(episode_records)
        print(
            f"completed episode {episode + 1}/{args.episodes}",
            file=sys.stderr,
            flush=True,
        )

    outage_records = [row for row in all_records if row["actual_outage"]]
    pre_outage_records = [row for row in all_records if row["pre_outage"]]
    non_outage_records = [
        row
        for row in all_records
        if not row["actual_outage"] and not row["pre_outage"]
    ]
    active_count = len(all_records)
    outage_count = len(outage_records)
    ready_outage_count = sum(
        row["ready_ratio"] < env.outage_service_ratio for row in all_records
    )
    visible_outage_count = sum(
        row["visible_ratio"] < env.outage_service_ratio for row in all_records
    )
    decomposition_counts = {
        "physical_exogenous_unavoidable": sum(
            row["visible_ratio"] < env.outage_service_ratio for row in outage_records
        ),
        "protocol_preparation_limited": sum(
            row["visible_ratio"] >= env.outage_service_ratio
            and row["ready_ratio"] < env.outage_service_ratio
            for row in outage_records
        ),
        "policy_decision_limited": sum(
            row["ready_ratio"] >= env.outage_service_ratio
            for row in outage_records
        ),
    }
    decomposition = {
        name: {
            "count": int(count),
            "fraction_of_active_states": float(count / max(active_count, 1)),
            "fraction_of_actual_outage_states": float(count / max(outage_count, 1)),
        }
        for name, count in decomposition_counts.items()
    }
    event_names = sorted(
        {name for row in outage_records for name in row["event_flags"]}
    )
    event_overlap = {
        name: {
            "count": int(sum(row["event_flags"][name] for row in outage_records)),
            "fraction_of_actual_outage_states": float(
                np.mean([row["event_flags"][name] for row in outage_records])
            )
            if outage_records
            else 0.0,
        }
        for name in event_names
    }
    report = {
        "configuration": {
            "episodes": args.episodes,
            "environment_seed": ENVIRONMENT_SEED,
            "agent_seed": AGENT_SEED,
            "evaluation_action_seed": EVALUATION_ACTION_SEED,
            "episode_seed_base": EPISODE_SEED_BASE,
            "checkpoint": "outputs/v3_final_1500/m5_seed7/mappo_weights.npz",
            "checkpoint_mu": trajectory_mu,
            "diagnostic_mu": DIAGNOSTIC_MU,
            "outage_service_ratio": env.outage_service_ratio,
        },
        "counts": {
            "active_ue_states": active_count,
            "actual_outage_states": outage_count,
            "pre_outage_states": len(pre_outage_records),
            "non_outage_non_pre_outage_states": len(non_outage_records),
        },
        "outage_ratios": {
            "actual": float(outage_count / max(active_count, 1)),
            "ready_set_oracle": float(ready_outage_count / max(active_count, 1)),
            "visible_set_optimistic_oracle": float(
                visible_outage_count / max(active_count, 1)
            ),
        },
        "outage_decomposition": decomposition,
        "actual_outage_event_overlap": event_overlap,
        "conditional_action_discrimination": {
            "actual_outage": subset_summary(outage_records),
            "pre_outage_1_to_5_steps": subset_summary(pre_outage_records),
            "non_outage_excluding_pre_outage": subset_summary(non_outage_records),
        },
        "potential_goodput_headroom": {
            "actual_outage": headroom_summary(outage_records),
            "pre_outage_1_to_5_steps": headroom_summary(pre_outage_records),
        },
        "oracle_semantics": {
            "actual": "realized packet delivery under the trained stochastic policy",
            "ready_set": (
                "per-UE counterfactual scheduling with other UEs' actual requests and "
                "shared capacity retained; prepared paths are usable immediately only "
                "through protocol-valid duplication, and switching delay is not bypassed"
            ),
            "visible_set": (
                "optimistic per-UE instantaneous bound using any true visible usable "
                "satellite at full cell capacity; independent choices need not be jointly feasible"
            ),
            "analytical_headroom": (
                "discounted mean potential-goodput ratio of each legal action's selected "
                "M5 analytical plan, using only the production reasoner's allowed inputs"
            ),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=False)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
