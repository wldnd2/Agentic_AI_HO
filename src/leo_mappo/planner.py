from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .ue import MobilityState
from .world_model import FutureLinkPredictor, GeometryWorldModel

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv


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
    predicted_handover_costs: np.ndarray
    predicted_preparation_costs: np.ndarray
    service_costs: np.ndarray
    reliability_costs: np.ndarray
    total_costs: np.ndarray


@dataclass
class _HypotheticalState:
    primary_cell: int | None
    secondary_cell: int | None
    mobility_state: MobilityState
    timer: int
    last_switch_step: int


class CHOPlanner:
    """Protocol-aware CHO lookahead using a supplied deterministic world model."""

    def __init__(
        self,
        config: dict,
        reward_config: dict,
        coordination_config: dict | None = None,
        use_coordination: bool = False,
        use_constraint: bool = False,
        world_model: GeometryWorldModel | None = None,
    ):
        self.horizon = int(config.get("horizon_steps", 6))
        self.discount = float(config.get("discount", 0.95))
        self.epsilon = float(config.get("score_epsilon", 1e-8))
        self.score_clip = float(config.get("score_clip", 5.0))
        if self.horizon <= 0 or self.score_clip <= 0.0:
            raise ValueError("reasoning horizon and score_clip must be positive")
        self.lambda_resource = float(reward_config.get("lambda_resource", 0.25))
        self.eta_preparation = float(reward_config.get("eta_preparation", 0.20))
        self.eta_handover = float(reward_config.get("eta_handover", 0.02))
        coordination_config = coordination_config or {}
        self.reasoning_price_weight = float(
            coordination_config.get("reasoning_price_weight", 1.0)
        )
        self.use_coordination = bool(use_coordination)
        self.use_constraint = bool(use_constraint)
        if world_model is None:
            raise ValueError("CHOPlanner requires a GeometryWorldModel")
        self.world_model = world_model
        self.future_link_predictor: FutureLinkPredictor | None = None

    def analyze(
        self,
        env: LEOMobilityEnv,
        masks: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        outage_dual_mu: float = 0.0,
    ) -> ReasoningResult:
        masks = env.action_masks() if masks is None else np.asarray(masks, dtype=np.float64)
        active, required_rates = env.observed_required_rates()
        if env.v3_enabled:
            # V3 M3 is local: dynamic cross-UE activity/demand enters decisions
            # only through M4 prices, never through a direct aggregate shortcut.
            active_count = 0.0
        else:
            active_count = float(np.sum(active))
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
        predicted_handover_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        predicted_preparation_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        service_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        reliability_costs = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
        total_costs = np.full((env.num_agents, env.action_dim), self.score_clip, dtype=np.float64)
        current_prices = (
            np.zeros(env.num_cells, dtype=np.float64)
            if prices is None
            else np.asarray(prices, dtype=np.float64)
        )
        if current_prices.shape != (env.num_cells,) or not np.all(np.isfinite(current_prices)):
            raise ValueError("prices must be a finite per-satellite vector")
        if env.v3_enabled:
            # Exact system-wide preparation occupancy is not an M3 oracle.
            # V3 capacity pressure is represented by actor-consistent observed
            # load; only nominal data capacity anchors the geometry forecast.
            available = np.ones(env.num_cells, dtype=np.float64)
        else:
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
                        active_count,
                    )
                    for plan, target in action_plans
                ]
                best_evaluation = min(evaluated, key=lambda item: item[0])
                (
                    cost,
                    first_deficit,
                    service_cost,
                    coordination_cost,
                    reliability_cost,
                    predicted_outage,
                    planning_reliability_risk,
                    predicted_control_cost,
                    predicted_handover_cost,
                    predicted_preparation_cost,
                ) = best_evaluation
                raw_costs[ue.ue_id, action] = float(np.clip(cost, 0.0, self.score_clip))
                total_costs[ue.ue_id, action] = cost
                predicted[ue.ue_id, action] = float(np.clip(first_deficit, 0.0, 1.0))
                service_costs[ue.ue_id, action] = service_cost
                coordination_costs[ue.ue_id, action] = coordination_cost
                reliability_costs[ue.ue_id, action] = reliability_cost
                predicted_outages[ue.ue_id, action] = predicted_outage
                planning_reliability_risks[ue.ue_id, action] = planning_reliability_risk
                predicted_control_costs[ue.ue_id, action] = predicted_control_cost
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
                predicted_handover_costs,
                predicted_preparation_costs,
                service_costs,
                reliability_costs,
                total_costs,
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
            predicted_handover_costs,
            predicted_preparation_costs,
            service_costs,
            reliability_costs,
            total_costs,
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
        elif ue.state == MobilityState.PREPARED and ue.protocol_timer > 0:
            add(env.release_action, [env.release_action])
        elif ue.state == MobilityState.PREPARED:
            add(env.commit_action, [env.commit_action])
            add(env.release_action, [env.release_action])
        for action in np.flatnonzero(mask > 0.0):
            if int(action) not in plans:
                add(int(action), [int(action)])
        return plans

    def _advance_hypothetical(
        self, env: LEOMobilityEnv, state: _HypotheticalState, offset: int, ue_id: int
    ) -> None:
        if (
            state.mobility_state == MobilityState.PREPARED
            and state.secondary_cell is not None
        ):
            elevation, _ = self.world_model.future_elevation_and_distance(
                env, ue_id, state.secondary_cell, offset
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
            action == env.commit_action
            and state.mobility_state == MobilityState.PREPARED
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
        elif action == env.release_action:
            state.secondary_cell = None
            state.mobility_state = MobilityState.SINGLE
            state.timer = 0
        return handover_started

    def _predicted_leg(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int | None,
        offset: int,
        required_rate: float,
        available: np.ndarray,
    ) -> tuple[float, float, float, float]:
        return self.world_model.predicted_leg(
            env,
            ue_id,
            cell_id,
            offset,
            required_rate,
            available,
            self.future_link_predictor,
        )

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
        active_count: float,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        ue = env.ues[ue_id]
        state = _HypotheticalState(
            ue.primary_cell,
            ue.secondary_cell,
            ue.state,
            int(ue.protocol_timer),
            int(ue.last_switch_step),
        )
        baseline_state = _HypotheticalState(
            ue.primary_cell,
            ue.secondary_cell,
            ue.state,
            int(ue.protocol_timer),
            int(ue.last_switch_step),
        )
        total = 0.0
        total_service = 0.0
        total_coordination = 0.0
        total_reliability = 0.0
        total_predicted_outage = 0.0
        total_planning_reliability_risk = 0.0
        total_control_cost = 0.0
        total_handover_cost = 0.0
        total_preparation_cost = 0.0
        total_discount_weight = 0.0
        first_deficit = 0.0
        rho_preparation = float(env.protocol_cfg["preparation_resource_fraction"])
        for offset, action in enumerate(plan):
            self._advance_hypothetical(env, state, offset, ue_id)
            if env.v3_enabled:
                self._advance_hypothetical(env, baseline_state, offset, ue_id)
                self._apply_hypothetical_action(
                    env, baseline_state, 0, None, ue_id, offset
                )
            target = prepare_target if env.prepare_start <= action < env.prepare_start + env.candidate_count else None
            handover_started = self._apply_hypothetical_action(
                env, state, action, target, ue_id, offset
            )
            (
                primary_rate,
                primary_potential_rate,
                primary_success,
                primary_fraction,
            ) = self._predicted_leg(
                env, ue_id, state.primary_cell, offset, required_rate, available
            )
            transmission_rate = primary_rate
            potential_transmission_rate = primary_potential_rate
            delivery_probability = primary_success
            action_demand = np.zeros(env.num_cells, dtype=np.float64)
            if state.primary_cell is not None:
                action_demand[state.primary_cell] += primary_fraction
            service_goodput, reliability_goodput = self.world_model.predicted_goodputs(
                transmission_rate,
                potential_transmission_rate,
                delivery_probability,
                preserve_reliability_headroom=(
                    env.v3_enabled and self.use_constraint
                ),
            )
            deficit = (
                max(0.0, required_rate - service_goodput)
                / (required_rate + self.epsilon)
                if required_rate > 0.0
                else 0.0
            )
            if offset == 0:
                first_deficit = deficit
            reservation_active = state.mobility_state in (
                MobilityState.PREPARED,
                MobilityState.SWITCHING,
            )
            preparation_cost = rho_preparation / env.num_cells if reservation_active else 0.0
            if reservation_active and state.secondary_cell is not None:
                action_demand[state.secondary_cell] += rho_preparation
            if env.v3_enabled:
                baseline_demand = np.zeros(env.num_cells, dtype=np.float64)
                if baseline_state.primary_cell is not None:
                    *_, baseline_primary_fraction = self._predicted_leg(
                        env,
                        ue_id,
                        baseline_state.primary_cell,
                        offset,
                        required_rate,
                        available,
                    )
                    baseline_demand[baseline_state.primary_cell] += (
                        baseline_primary_fraction
                    )
                baseline_reservation_active = baseline_state.mobility_state in (
                    MobilityState.PREPARED,
                    MobilityState.SWITCHING,
                )
                if (
                    baseline_reservation_active
                    and baseline_state.secondary_cell is not None
                ):
                    baseline_demand[baseline_state.secondary_cell] += rho_preparation
                incremental_demand = np.maximum(
                    action_demand - baseline_demand, 0.0
                )
                coordination_step = float(np.dot(prices, incremental_demand))
            else:
                coordination_step = float(np.dot(prices, action_demand))
            handover_normalizer = (
                float(max(env.num_agents, 1))
                if env.v3_enabled
                else max(active_count, 1.0)
            )
            handover_cost = float(handover_started) / handover_normalizer
            control_cost = (
                self.eta_preparation * preparation_cost
                + self.eta_handover * handover_cost
            )
            predicted_outage = self.world_model.predicted_outage_probability(
                transmission_rate,
                delivery_probability,
                required_rate,
                env.interval_s,
                int(env.traffic_cfg["packet_size_bits"]),
                env.outage_service_ratio,
            )
            planning_reliability_risk = self.world_model.planning_reliability_risk(
                reliability_goodput,
                required_rate,
                env.outage_service_ratio,
            )
            service_step_cost = deficit
            base_step_cost = (
                service_step_cost
                if env.v3_enabled
                else service_step_cost + self.lambda_resource * control_cost
            )
            # V3 M5 deliberately keeps analytical reliability risk diagnostic-only.
            # Reliability enters production decisions through realized-outage
            # constrained rewards, recurrent outcomes, and the actor's normalized
            # mu input. Preserve the historical analytical term for V2 compatibility.
            reliability_step_cost = (
                outage_dual_mu * planning_reliability_risk
                if self.use_constraint and not env.v3_enabled
                else 0.0
            )
            applied_coordination_cost = (
                self.reasoning_price_weight * coordination_step
                if self.use_coordination
                else 0.0
            )
            step_cost = (
                base_step_cost
                + applied_coordination_cost
                + reliability_step_cost
            )
            discount_weight = self.discount**offset
            total += discount_weight * step_cost
            total_service += discount_weight * service_step_cost
            total_coordination += discount_weight * applied_coordination_cost
            total_reliability += discount_weight * reliability_step_cost
            total_predicted_outage += discount_weight * predicted_outage
            total_planning_reliability_risk += (
                discount_weight * planning_reliability_risk
            )
            total_control_cost += discount_weight * control_cost
            total_handover_cost += discount_weight * handover_cost
            total_preparation_cost += discount_weight * preparation_cost
            total_discount_weight += discount_weight
        diagnostic_scale = max(total_discount_weight, self.epsilon)
        return (
            float(total),
            float(first_deficit),
            float(total_service),
            float(total_coordination),
            float(total_reliability),
            float(np.clip(total_predicted_outage / diagnostic_scale, 0.0, 1.0)),
            float(
                np.clip(
                    total_planning_reliability_risk / diagnostic_scale,
                    0.0,
                    1.0,
                )
            ),
            float(total_control_cost / diagnostic_scale),
            float(np.clip(total_handover_cost / diagnostic_scale, 0.0, 1.0)),
            float(np.clip(total_preparation_cost / diagnostic_scale, 0.0, 1.0)),
        )
