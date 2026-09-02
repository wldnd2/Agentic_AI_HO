from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .pdcp import joint_delivery_probability
from .satellite import elevation_and_distance
from .ue import MobilityState

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv


def _predicted_outage_probability(
    transmission_rate_bps: float,
    success_probability: float,
    required_rate_bps: float,
    interval_s: float,
    packet_size_bits: int,
    outage_service_ratio: float,
    epsilon: float,
) -> float:
    """Continuity-corrected normal approximation of packet-level outage risk."""
    numeric_inputs = (
        transmission_rate_bps,
        success_probability,
        required_rate_bps,
        interval_s,
        float(packet_size_bits),
        outage_service_ratio,
        epsilon,
    )
    if not all(math.isfinite(float(value)) for value in numeric_inputs):
        raise FloatingPointError("predicted outage inputs must be finite")
    if interval_s <= 0.0 or packet_size_bits <= 0 or epsilon <= 0.0:
        raise ValueError("interval, packet size, and epsilon must be positive")
    if required_rate_bps <= 0.0:
        return 0.0

    packet_budget = math.floor(
        max(0.0, transmission_rate_bps) * interval_s / packet_size_bits
    )
    outage_success_limit = (
        math.ceil(
            outage_service_ratio
            * required_rate_bps
            * interval_s
            / packet_size_bits
        )
        - 1
    )
    if outage_success_limit < 0:
        return 0.0
    if packet_budget <= outage_success_limit:
        return 1.0
    if success_probability <= 0.0:
        return 1.0
    if success_probability >= 1.0:
        return 0.0

    mean = packet_budget * success_probability
    variance = packet_budget * success_probability * (1.0 - success_probability)
    z_score = (outage_success_limit + 0.5 - mean) / math.sqrt(variance)
    probability = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    if not math.isfinite(probability):
        raise FloatingPointError("predicted outage probability is non-finite")
    return float(min(1.0, max(0.0, probability)))


def _planning_reliability_risk(
    expected_goodput_bps: float,
    required_rate_bps: float,
    outage_service_ratio: float,
    temperature: float,
    epsilon: float,
) -> float:
    """Dense deterministic planning risk around the expected-goodput boundary."""
    numeric_inputs = (
        expected_goodput_bps,
        required_rate_bps,
        outage_service_ratio,
        temperature,
        epsilon,
    )
    if not all(math.isfinite(float(value)) for value in numeric_inputs):
        raise FloatingPointError("planning reliability-risk inputs must be finite")
    if temperature <= 0.0 or epsilon <= 0.0:
        raise ValueError("planning reliability-risk temperature and epsilon must be positive")
    if required_rate_bps <= 0.0:
        return 0.0
    margin = (
        outage_service_ratio * required_rate_bps - expected_goodput_bps
    ) / (temperature * required_rate_bps + epsilon)
    clipped_margin = min(40.0, max(-40.0, margin))
    risk = 1.0 / (1.0 + math.exp(-clipped_margin))
    if not math.isfinite(risk):
        raise FloatingPointError("planning reliability risk is non-finite")
    return float(min(1.0, max(0.0, risk)))


@dataclass
class ReasoningResult:
    features: np.ndarray
    raw_costs: np.ndarray
    predicted_deficits: np.ndarray
    best_scores: np.ndarray
    score_gaps: np.ndarray
    coordination_costs: np.ndarray
    predicted_outages: np.ndarray
    planning_reliability_risks: np.ndarray
    predicted_control_costs: np.ndarray
    predicted_duplication_costs: np.ndarray
    predicted_handover_costs: np.ndarray
    predicted_preparation_costs: np.ndarray


@dataclass
class _HypotheticalState:
    primary_cell: int | None
    secondary_cell: int | None
    mobility_state: MobilityState
    timer: int
    last_switch_step: int


class LookaheadReasoner:
    """State-free ephemeris lookahead over protocol-aware mobility plans.

    Team-level duplication and handover costs are analytical incremental
    proxies based only on current observable demand; future event labels are
    deliberately excluded.
    """

    def __init__(
        self,
        config: dict,
        reward_config: dict,
        coordination_config: dict | None = None,
        use_coordination: bool = False,
        use_constraint: bool = False,
    ):
        self.horizon = int(config.get("horizon_steps", 6))
        self.discount = float(config.get("discount", 0.95))
        self.epsilon = float(config.get("score_epsilon", 1e-8))
        self.score_clip = float(config.get("score_clip", 5.0))
        self.duplication_hold_steps = int(config.get("duplication_hold_steps", 1))
        self.reliability_risk_temperature = float(
            config.get("reliability_risk_temperature", 0.10)
        )
        if self.horizon <= 0 or self.score_clip <= 0.0 or self.duplication_hold_steps < 0:
            raise ValueError("reasoning horizon/score_clip must be positive and hold steps non-negative")
        if self.reliability_risk_temperature <= 0.0:
            raise ValueError("reasoning reliability_risk_temperature must be positive")
        self.lambda_resource = float(reward_config.get("lambda_resource", 0.25))
        self.eta_preparation = float(reward_config.get("eta_preparation", 0.20))
        self.eta_handover = float(reward_config.get("eta_handover", 0.02))
        coordination_config = coordination_config or {}
        self.reasoning_price_weight = float(
            coordination_config.get("reasoning_price_weight", 1.0)
        )
        self.use_coordination = bool(use_coordination)
        self.use_constraint = bool(use_constraint)

    def analyze(
        self,
        env: LEOMobilityEnv,
        masks: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        outage_dual_mu: float = 0.0,
    ) -> ReasoningResult:
        masks = env.action_masks() if masks is None else np.asarray(masks, dtype=np.float64)
        active, required_rates = env.current_required_rates()
        active_count = float(np.sum(active))
        aggregate_required_rate = float(np.sum(required_rates))
        features = np.ones((env.num_agents, env.action_dim), dtype=np.float64)
        raw_costs = np.full((env.num_agents, env.action_dim), self.score_clip, dtype=np.float64)
        predicted = np.ones((env.num_agents, env.action_dim), dtype=np.float64)
        best_scores = np.zeros(env.num_agents, dtype=np.float64)
        score_gaps = np.zeros(env.num_agents, dtype=np.float64)
        coordination_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        predicted_outages = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        planning_reliability_risks = np.zeros(
            (env.num_agents, env.action_dim), dtype=np.float64
        )
        predicted_control_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        predicted_duplication_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        predicted_handover_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        predicted_preparation_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        current_prices = (
            np.zeros(env.num_cells, dtype=np.float64)
            if prices is None
            else np.asarray(prices, dtype=np.float64)
        )
        if current_prices.shape != (env.num_cells,) or not np.all(np.isfinite(current_prices)):
            raise ValueError("prices must be a finite per-satellite vector")
        preparation = env._compute_preparation_reservations()
        available = np.maximum(0.0, 1.0 - preparation)

        for ue in env.ues:
            plans = self._plans(env, ue.ue_id, masks[ue.ue_id])
            for action, action_plans in plans.items():
                evaluated = [
                    self._evaluate_plan(
                        env,
                        ue.ue_id,
                        plan,
                        target,
                        required_rates[ue.ue_id],
                        available,
                        current_prices,
                        outage_dual_mu,
                        aggregate_required_rate,
                        active_count,
                    )
                    for plan, target in action_plans
                ]
                best_evaluation = min(evaluated, key=lambda item: item[0])
                (
                    cost,
                    first_deficit,
                    coordination_cost,
                    predicted_outage,
                    planning_reliability_risk,
                    predicted_control_cost,
                    predicted_duplication_cost,
                    predicted_handover_cost,
                    predicted_preparation_cost,
                ) = best_evaluation
                raw_costs[ue.ue_id, action] = float(np.clip(cost, 0.0, self.score_clip))
                predicted[ue.ue_id, action] = float(np.clip(first_deficit, 0.0, 1.0))
                coordination_costs[ue.ue_id, action] = coordination_cost
                predicted_outages[ue.ue_id, action] = predicted_outage
                planning_reliability_risks[ue.ue_id, action] = planning_reliability_risk
                predicted_control_costs[ue.ue_id, action] = predicted_control_cost
                predicted_duplication_costs[ue.ue_id, action] = predicted_duplication_cost
                predicted_handover_costs[ue.ue_id, action] = predicted_handover_cost
                predicted_preparation_costs[ue.ue_id, action] = predicted_preparation_cost
            feasible = np.flatnonzero(masks[ue.ue_id] > 0.0)
            feasible_costs = raw_costs[ue.ue_id, feasible]
            ordered = np.sort(feasible_costs)
            minimum = float(ordered[0])
            maximum = float(ordered[-1])
            best_scores[ue.ue_id] = minimum
            score_gaps[ue.ue_id] = float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
            if maximum - minimum <= self.epsilon:
                features[ue.ue_id, feasible] = 0.0
            else:
                features[ue.ue_id, feasible] = np.clip(
                    (feasible_costs - minimum) / (maximum - minimum + self.epsilon), 0.0, 1.0
                )
        if not all(
            np.all(np.isfinite(values))
            for values in (
                features,
                raw_costs,
                predicted,
                best_scores,
                score_gaps,
                coordination_costs,
                predicted_outages,
                planning_reliability_risks,
                predicted_control_costs,
                predicted_duplication_costs,
                predicted_handover_costs,
                predicted_preparation_costs,
            )
        ):
            raise FloatingPointError("reasoner produced a non-finite result")
        return ReasoningResult(
            features,
            raw_costs,
            predicted,
            best_scores,
            score_gaps,
            coordination_costs,
            predicted_outages,
            planning_reliability_risks,
            predicted_control_costs,
            predicted_duplication_costs,
            predicted_handover_costs,
            predicted_preparation_costs,
        )

    def _pad(self, actions: list[int], stay: int = 0) -> tuple[int, ...]:
        return tuple((actions + [stay] * self.horizon)[: self.horizon])

    def _plans(
        self, env: LEOMobilityEnv, ue_id: int, mask: np.ndarray
    ) -> dict[int, list[tuple[tuple[int, ...], int | None]]]:
        ue = env.ues[ue_id]
        plans: dict[int, list[tuple[tuple[int, ...], int | None]]] = {}

        def add(action: int, sequence: list[int], target: int | None = None) -> None:
            if mask[action] > 0.0:
                plans.setdefault(action, []).append((self._pad(sequence), target))

        add(0, [0])
        if ue.state == MobilityState.SINGLE:
            for slot, target in enumerate(env.candidates[ue_id]):
                action = env.prepare_start + slot
                prep_wait = [0] * int(env.protocol_cfg["preparation_steps"])
                switch_wait = [0] * int(env.protocol_cfg["switching_steps"])
                add(action, [action, *prep_wait, env.commit_action, *switch_wait], target)
                add(
                    action,
                    [
                        action,
                        *prep_wait,
                        env.dup_on_action,
                        *([0] * self.duplication_hold_steps),
                        env.commit_action,
                    ],
                    target,
                )
        elif ue.state == MobilityState.PREPARED and ue.protocol_timer > 0:
            add(env.release_action, [env.release_action])
        elif ue.state == MobilityState.PREPARED:
            add(
                env.dup_on_action,
                [env.dup_on_action, *([0] * self.duplication_hold_steps), env.commit_action],
            )
            add(env.commit_action, [env.commit_action])
            add(env.release_action, [env.release_action])
        elif ue.state == MobilityState.DUPLICATING:
            add(env.commit_action, [env.commit_action])
            add(env.dup_off_action, [env.dup_off_action])
            add(env.rollback_action, [env.rollback_action])
        for action in np.flatnonzero(mask > 0.0):
            if int(action) not in plans:
                add(int(action), [int(action)])
        return plans

    def _advance_hypothetical(
        self, env: LEOMobilityEnv, state: _HypotheticalState, offset: int, ue_id: int
    ) -> None:
        if (
            state.mobility_state in (MobilityState.PREPARED, MobilityState.DUPLICATING)
            and state.secondary_cell is not None
        ):
            elevation, _ = elevation_and_distance(
                env.ues[ue_id].position,
                env.satellites[state.secondary_cell].position_ecef(
                    env.time_s + offset * env.interval_s
                ),
            )
            if elevation < env.min_elevation_rad:
                state.secondary_cell = None
                state.mobility_state = MobilityState.SINGLE
                state.timer = 0
        if state.timer > 0:
            state.timer -= 1
        if state.mobility_state == MobilityState.SWITCHING and state.timer == 0:
            state.primary_cell = state.secondary_cell
            state.secondary_cell = None
            state.mobility_state = MobilityState.COOLDOWN
            state.timer = int(env.protocol_cfg["cooldown_steps"])
            state.last_switch_step = env.step_index + offset
        elif state.mobility_state == MobilityState.COOLDOWN and state.timer == 0:
            state.mobility_state = MobilityState.SINGLE

    def _apply_hypothetical_action(
        self,
        env: LEOMobilityEnv,
        state: _HypotheticalState,
        action: int,
        prepare_target: int | None,
        ue_id: int,
        offset: int,
    ) -> bool:
        handover_started = False
        if env.prepare_start <= action < env.prepare_start + env.candidate_count:
            state.secondary_cell = prepare_target
            state.mobility_state = MobilityState.PREPARED
            state.timer = int(env.protocol_cfg["preparation_steps"])
        elif (
            action == env.dup_on_action
            and state.mobility_state == MobilityState.PREPARED
            and state.timer == 0
            and state.secondary_cell is not None
        ):
            state.mobility_state = MobilityState.DUPLICATING
        elif action == env.dup_off_action:
            state.mobility_state = MobilityState.PREPARED
            state.timer = 0
        elif (
            action == env.commit_action
            and state.mobility_state in (MobilityState.PREPARED, MobilityState.DUPLICATING)
            and state.timer == 0
            and state.secondary_cell is not None
        ):
            dwell_ok = env.step_index + offset - state.last_switch_step >= int(
                env.protocol_cfg.get("dwell_steps", 0)
            )
            visibility_left = max(
                0.0,
                env.remaining_visibility[ue_id, state.secondary_cell] - offset * env.interval_s,
            )
            visibility_ok = visibility_left >= (
                int(env.protocol_cfg["switching_steps"])
                + int(env.protocol_cfg.get("visibility_margin_steps", 0))
            ) * env.interval_s
            if dwell_ok and visibility_ok:
                state.mobility_state = MobilityState.SWITCHING
                state.timer = int(env.protocol_cfg["switching_steps"])
                handover_started = True
        elif action in (env.release_action, env.rollback_action):
            state.secondary_cell = None
            state.mobility_state = MobilityState.SINGLE
            state.timer = 0
        return handover_started

    def _future_link(
        self, env: LEOMobilityEnv, ue_id: int, cell_id: int | None, offset: int
    ) -> tuple[float, float]:
        if cell_id is None:
            return 0.0, 0.0
        ue = env.ues[ue_id]
        satellite = env.satellites[cell_id]
        future_time = env.time_s + offset * env.interval_s
        elevation, future_distance = elevation_and_distance(
            ue.position, satellite.position_ecef(future_time)
        )
        if elevation < env.min_elevation_rad:
            return 0.0, 0.0
        current = env.links[ue_id][cell_id]
        loss_delta = 20.0 * np.log10(future_distance / (current.distance_m + self.epsilon))
        current_sinr = (
            env.observed_sinr[ue_id, cell_id] if env.v3_enabled else current.sinr_db
        )
        sinr_db = current_sinr - loss_delta
        sinr_linear = 10.0 ** (np.clip(sinr_db, -120.0, 80.0) / 10.0)
        efficiency = min(
            float(env.channel_cfg["spectral_efficiency_cap"]),
            0.75 * np.log2(1.0 + sinr_linear),
        )
        success = 1.0 / (1.0 + np.exp(np.clip(-0.75 * (sinr_db + 2.0), -40.0, 40.0)))
        return float(efficiency), float(success)

    def _predicted_leg(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int | None,
        offset: int,
        required_rate: float,
        available: np.ndarray,
    ) -> tuple[float, float, float]:
        if cell_id is None or required_rate <= 0.0:
            return 0.0, 0.0, 0.0
        efficiency, success = self._future_link(env, ue_id, cell_id, offset)
        fraction = available[cell_id]
        pressure = max(
            0.0,
            (
                env.observed_load[ue_id, cell_id]
                if env.v3_enabled
                else env.last_pressure[cell_id]
            ),
        )
        load_scale = min(1.0, fraction / max(fraction, pressure, self.epsilon))
        raw_capacity = (
            env.satellites[cell_id].bandwidth_hz
            * efficiency
            * env.satellites[cell_id].effective_capacity_scale
        )
        rate = min(required_rate, raw_capacity * fraction * load_scale)
        resource_fraction = rate / max(raw_capacity, self.epsilon)
        return float(rate), success, float(max(0.0, resource_fraction))

    def _evaluate_plan(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        plan: tuple[int, ...],
        prepare_target: int | None,
        required_rate: float,
        available: np.ndarray,
        prices: np.ndarray,
        outage_dual_mu: float,
        aggregate_required_rate: float,
        active_count: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        ue = env.ues[ue_id]
        state = _HypotheticalState(
            ue.primary_cell,
            ue.secondary_cell,
            ue.state,
            int(ue.protocol_timer),
            int(ue.last_switch_step),
        )
        total = 0.0
        total_coordination = 0.0
        total_predicted_outage = 0.0
        total_planning_reliability_risk = 0.0
        total_control_cost = 0.0
        total_duplication_cost = 0.0
        total_handover_cost = 0.0
        total_preparation_cost = 0.0
        total_discount_weight = 0.0
        first_deficit = 0.0
        rho_preparation = float(env.protocol_cfg["preparation_resource_fraction"])
        for offset, action in enumerate(plan):
            self._advance_hypothetical(env, state, offset, ue_id)
            target = prepare_target if env.prepare_start <= action < env.prepare_start + env.candidate_count else None
            handover_started = self._apply_hypothetical_action(
                env, state, action, target, ue_id, offset
            )
            primary_rate, primary_success, primary_fraction = self._predicted_leg(
                env, ue_id, state.primary_cell, offset, required_rate, available
            )
            transmission_rate = primary_rate
            delivery_probability = primary_success
            goodput = transmission_rate * delivery_probability
            coordination_step = (
                prices[state.primary_cell] * primary_fraction
                if state.primary_cell is not None
                else 0.0
            )
            duplication_cost = 0.0
            if state.mobility_state == MobilityState.DUPLICATING and state.secondary_cell is not None:
                secondary_rate, secondary_success, secondary_fraction = self._predicted_leg(
                    env, ue_id, state.secondary_cell, offset, required_rate, available
                )
                stream_rate = min(primary_rate, secondary_rate)
                probability = joint_delivery_probability(
                    primary_success, secondary_success, failure_correlation=0.15
                )
                transmission_rate = stream_rate
                delivery_probability = probability
                goodput = transmission_rate * delivery_probability
                # Incremental team-level proxy: current observable aggregate demand
                # is held persistent over the horizon without peeking at future events.
                network_base_rate = max(0.0, aggregate_required_rate - required_rate) + stream_rate
                duplication_cost = stream_rate / (
                    network_base_rate + stream_rate + self.epsilon
                )
                duplication_cost = float(np.clip(duplication_cost, 0.0, 1.0))
                coordination_step += prices[state.secondary_cell] * secondary_fraction
            deficit = (
                max(0.0, required_rate - goodput) / (required_rate + self.epsilon)
                if required_rate > 0.0
                else 0.0
            )
            if offset == 0:
                first_deficit = deficit
            reservation_active = state.mobility_state in (
                MobilityState.PREPARED,
                MobilityState.DUPLICATING,
                MobilityState.SWITCHING,
            )
            preparation_cost = rho_preparation / env.num_cells if reservation_active else 0.0
            if reservation_active and state.secondary_cell is not None:
                coordination_step += prices[state.secondary_cell] * rho_preparation
            handover_cost = float(handover_started) / max(active_count, 1.0)
            control_cost = (
                duplication_cost
                + self.eta_preparation * preparation_cost
                + self.eta_handover * handover_cost
            )
            predicted_outage = _predicted_outage_probability(
                transmission_rate,
                delivery_probability,
                required_rate,
                env.interval_s,
                int(env.traffic_cfg["packet_size_bits"]),
                env.outage_service_ratio,
                self.epsilon,
            )
            planning_reliability_risk = _planning_reliability_risk(
                goodput,
                required_rate,
                env.outage_service_ratio,
                self.reliability_risk_temperature,
                self.epsilon,
            )
            base_step_cost = deficit + self.lambda_resource * control_cost
            # The omitted -mu*outage_target term is identical for every
            # hypothetical plan and therefore cannot change action ranking.
            step_cost = (
                base_step_cost + outage_dual_mu * planning_reliability_risk
                if self.use_constraint
                else base_step_cost
            )
            if self.use_coordination:
                step_cost += self.reasoning_price_weight * coordination_step
            discount_weight = self.discount**offset
            total += discount_weight * step_cost
            total_coordination += discount_weight * coordination_step
            total_predicted_outage += discount_weight * predicted_outage
            total_planning_reliability_risk += (
                discount_weight * planning_reliability_risk
            )
            total_control_cost += discount_weight * control_cost
            total_duplication_cost += discount_weight * duplication_cost
            total_handover_cost += discount_weight * handover_cost
            total_preparation_cost += discount_weight * preparation_cost
            total_discount_weight += discount_weight
        diagnostic_scale = max(total_discount_weight, self.epsilon)
        return (
            float(total),
            float(first_deficit),
            float(total_coordination),
            float(np.clip(total_predicted_outage / diagnostic_scale, 0.0, 1.0)),
            float(
                np.clip(
                    total_planning_reliability_risk / diagnostic_scale,
                    0.0,
                    1.0,
                )
            ),
            float(total_control_cost / diagnostic_scale),
            float(np.clip(total_duplication_cost / diagnostic_scale, 0.0, 1.0)),
            float(np.clip(total_handover_cost / diagnostic_scale, 0.0, 1.0)),
            float(np.clip(total_preparation_cost / diagnostic_scale, 0.0, 1.0)),
        )
