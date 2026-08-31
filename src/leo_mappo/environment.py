from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from .channel import ChannelModel, LinkResult
from .coordinator import compute_training_reward
from .events import EventGenerator
from .pdcp import joint_delivery_probability
from .satellite import SatelliteCell, estimate_remaining_visibility
from .scheduler import LegRequest, QoSAwareFairScheduler
from .ue import MobilityState, UE


class Action(IntEnum):
    STAY = 0


@dataclass
class StepOutput:
    observations: np.ndarray
    global_state: np.ndarray
    action_masks: np.ndarray
    rewards: np.ndarray
    done: bool
    info: dict[str, float]
    agent_outcomes: np.ndarray | None = None
    agent_active: np.ndarray | None = None
    requested_pressure: np.ndarray | None = None


class LEOMobilityEnv:
    """Multi-UE LEO mobility environment with shared cell capacity.

    MAPPO controls target preparation, PDCP duplication, and primary switching.
    A fixed scheduler owns radio-resource allocation.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.env_cfg = config["environment"]
        self.sat_cfg = config["satellite"]
        self.channel_cfg = config["channel"]
        self.traffic_cfg = config["traffic"]
        self.protocol_cfg = config["protocol"]
        self.qos_cfg = {"outage_service_ratio": 0.80, **config.get("qos", {})}
        self.scheduler_cfg = {
            "min_schedulable_spectral_efficiency": 1e-4,
            "epsilon_capacity": 1e-8,
            **config.get("scheduler", {}),
        }
        self.reward_cfg = {
            "lambda_resource": 0.25,
            "eta_preparation": 0.20,
            "eta_handover": 0.02,
            "epsilon": 1e-8,
            **config["reward"],
        }
        self.num_agents = int(self.env_cfg["num_ues"])
        self.num_cells = int(self.env_cfg["num_satellites"])
        self.candidate_count = int(self.env_cfg["candidate_count"])
        self.action_dim = self.candidate_count + 6
        self.obs_dim = 16 + 7 * self.candidate_count
        self.global_state_dim = self.num_agents * self.obs_dim + 2 * self.num_cells
        self.rng = np.random.default_rng(int(config.get("seed", 0)))
        self.interval_s = float(self.env_cfg["control_interval_s"])
        self.min_elevation_rad = np.radians(float(self.env_cfg["min_elevation_deg"]))
        self.satellites = self._make_satellites()
        self.ues = self._make_ues()
        self.channel = ChannelModel(
            carrier_hz=float(self.channel_cfg["carrier_ghz"]) * 1e9,
            ue_gain_dbi=float(self.channel_cfg["ue_gain_dbi"]),
            noise_figure_db=float(self.channel_cfg["noise_figure_db"]),
            implementation_loss_db=float(self.channel_cfg["implementation_loss_db"]),
            atmospheric_loss_db=float(self.channel_cfg["atmospheric_loss_db"]),
            shadow_std_db=float(self.channel_cfg["shadow_std_db"]),
            shadow_correlation=float(self.channel_cfg["shadow_correlation"]),
            interference_dbm=float(self.channel_cfg["interference_dbm"]),
            spectral_efficiency_cap=float(self.channel_cfg["spectral_efficiency_cap"]),
            rng=self.rng,
        )
        self.outage_service_ratio = float(self.qos_cfg["outage_service_ratio"])
        if not 0.0 < self.outage_service_ratio <= 1.0:
            raise ValueError("qos.outage_service_ratio must be in (0, 1]")
        self.min_schedulable_spectral_efficiency = float(
            self.scheduler_cfg["min_schedulable_spectral_efficiency"]
        )
        self.epsilon_capacity = float(self.scheduler_cfg["epsilon_capacity"])
        if float(self.reward_cfg["eta_handover"]) < 0.0:
            raise ValueError("reward.eta_handover must be non-negative")
        if self.min_schedulable_spectral_efficiency < 0.0 or self.epsilon_capacity <= 0.0:
            raise ValueError("scheduler thresholds must be non-negative with positive epsilon_capacity")
        self.scheduler = QoSAwareFairScheduler(request_epsilon=self.epsilon_capacity)
        self.events = EventGenerator(
            config.get("events", []), self.num_agents, self.num_cells, self.rng
        )
        self.step_index = 0
        self.time_s = 0.0
        self.last_utilization = np.zeros(self.num_cells, dtype=np.float64)
        self.last_pressure = np.zeros(self.num_cells, dtype=np.float64)
        self.last_requested_pressure = np.zeros(self.num_cells, dtype=np.float64)
        self.links: list[list[LinkResult]] = []
        self.remaining_visibility = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self.candidates: list[list[int]] = []
        self._active_effects = self.events.effects(0)
        self._last_ho_outcomes = np.zeros(self.num_agents, dtype=np.float64)
        self._last_handover_started = np.zeros(self.num_agents, dtype=np.float64)
        self._last_agent_outcomes = np.zeros((self.num_agents, 6), dtype=np.float64)
        self._last_agent_active = np.zeros(self.num_agents, dtype=np.float64)

    @property
    def prepare_start(self) -> int:
        return 1

    @property
    def dup_on_action(self) -> int:
        return self.candidate_count + 1

    @property
    def dup_off_action(self) -> int:
        return self.candidate_count + 2

    @property
    def commit_action(self) -> int:
        return self.candidate_count + 3

    @property
    def release_action(self) -> int:
        return self.candidate_count + 4

    @property
    def rollback_action(self) -> int:
        return self.candidate_count + 5

    def _make_satellites(self) -> list[SatelliteCell]:
        altitude_m = float(self.sat_cfg["altitude_km"]) * 1e3
        inclination = np.radians(float(self.sat_cfg["inclination_deg"]))
        period = float(self.sat_cfg["orbital_period_s"])
        phases = np.linspace(-0.38, 0.38, self.num_cells)
        return [
            SatelliteCell(
                cell_id=index,
                altitude_m=altitude_m,
                inclination_rad=inclination,
                phase_rad=float(phase),
                # Local multi-plane constellation slice: cells on either side
                # of the service region traverse it in opposite ground-track
                # directions, preserving overlapping handover candidates.
                angular_rate_rad_s=(2.0 * np.pi / period) * (1.0 if phase <= 0.0 else -1.0),
                bandwidth_hz=float(self.sat_cfg["bandwidth_mhz"]) * 1e6,
                tx_power_dbm=float(self.sat_cfg["tx_power_dbm"]),
                tx_gain_dbi=float(self.sat_cfg["tx_gain_dbi"]),
                base_capacity_scale=float(self.sat_cfg.get("capacity_scale", 1.0)),
            )
            for index, phase in enumerate(phases)
        ]

    def _make_ues(self) -> list[UE]:
        base_rate = float(self.traffic_cfg["mean_arrival_mbps"])
        spread = float(self.traffic_cfg.get("arrival_spread", 0.0))
        ues = []
        for index in range(self.num_agents):
            rate = base_rate * self.rng.uniform(1.0 - spread, 1.0 + spread)
            ues.append(
                UE(
                    ue_id=index,
                    latitude_rad=float(self.rng.uniform(-0.04, 0.04)),
                    longitude_rad=float(self.rng.uniform(-0.05, 0.05)),
                    mean_arrival_mbps=rate,
                    packet_size_bits=int(self.traffic_cfg["packet_size_bits"]),
                    deadline_s=float(self.traffic_cfg["deadline_ms"]) / 1000.0,
                    reliability_target=float(self.traffic_cfg["reliability_target"]),
                    qos_priority=float(self.traffic_cfg["qos_priority"]),
                    max_queue_packets=int(self.env_cfg["max_queue_packets"]),
                )
            )
        return ues

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.channel.rng = self.rng
            self.events.rng = self.rng
        self.step_index = 0
        self.time_s = 0.0
        self.last_utilization.fill(0.0)
        self.last_pressure.fill(0.0)
        self.last_requested_pressure.fill(0.0)
        self.channel.reset()
        self.events.reset()
        self._active_effects = self.events.effects(0)
        self._last_ho_outcomes.fill(0.0)
        self._last_handover_started.fill(0.0)
        self._last_agent_outcomes.fill(0.0)
        self._last_agent_active.fill(0.0)
        for satellite in self.satellites:
            satellite.capacity_multiplier = 1.0
        for ue in self.ues:
            ue.reset_runtime()
        self._refresh_links()
        for ue in self.ues:
            visible = [cell for cell in range(self.num_cells) if self.links[ue.ue_id][cell].visible]
            if not visible:
                ue.primary_cell = int(np.argmax([link.elevation_deg for link in self.links[ue.ue_id]]))
            else:
                ue.primary_cell = max(visible, key=lambda cell: self.links[ue.ue_id][cell].sinr_db)
        return self._build_output_state()

    def _refresh_links(self) -> None:
        effects = self._active_effects
        self.links = []
        for ue in self.ues:
            ue_links = []
            for cell_id, satellite in enumerate(self.satellites):
                ue_links.append(
                    self.channel.evaluate(
                        ue.ue_id,
                        ue.position,
                        satellite,
                        self.time_s,
                        self.min_elevation_rad,
                        extra_loss_db=float(effects.link_extra_loss_db[ue.ue_id, cell_id]),
                    )
                )
            self.links.append(ue_links)
        for ue in self.ues:
            for cell_id, satellite in enumerate(self.satellites):
                self.remaining_visibility[ue.ue_id, cell_id] = estimate_remaining_visibility(
                    satellite,
                    ue.position,
                    self.time_s,
                    self.min_elevation_rad,
                    resolution_s=10.0,
                )
        self.candidates = []
        for ue in self.ues:
            choices = [
                cell_id
                for cell_id in range(self.num_cells)
                if cell_id != ue.primary_cell and self.links[ue.ue_id][cell_id].visible
            ]
            choices.sort(
                key=lambda cell_id: (
                    self.links[ue.ue_id][cell_id].sinr_db
                    + 0.01 * self.remaining_visibility[ue.ue_id, cell_id]
                    - 5.0 * self.last_pressure[cell_id]
                ),
                reverse=True,
            )
            self.candidates.append(choices[: self.candidate_count])

    def action_masks(self) -> np.ndarray:
        masks = np.zeros((self.num_agents, self.action_dim), dtype=np.float64)
        for ue in self.ues:
            mask = masks[ue.ue_id]
            mask[Action.STAY] = 1.0
            if ue.state == MobilityState.SINGLE:
                for slot, cell_id in enumerate(self.candidates[ue.ue_id]):
                    enough_visibility = self.remaining_visibility[ue.ue_id, cell_id] >= (
                        self.protocol_cfg["preparation_steps"]
                        + self.protocol_cfg["switching_steps"]
                        + int(self.protocol_cfg.get("visibility_margin_steps", 0))
                    ) * self.interval_s
                    if enough_visibility:
                        mask[self.prepare_start + slot] = 1.0
            elif ue.state == MobilityState.PREPARED and ue.protocol_timer > 0:
                mask[self.release_action] = 1.0
            elif ue.state == MobilityState.PREPARED:
                secondary = ue.secondary_cell
                if secondary is not None and self.links[ue.ue_id][secondary].visible:
                    mask[self.dup_on_action] = 1.0
                    if self._commit_feasible(ue):
                        mask[self.commit_action] = 1.0
                mask[self.release_action] = 1.0
            elif ue.state == MobilityState.DUPLICATING:
                secondary = ue.secondary_cell
                if secondary is not None and self.links[ue.ue_id][secondary].visible:
                    mask[self.dup_off_action] = 1.0
                    if self._commit_feasible(ue):
                        mask[self.commit_action] = 1.0
                    mask[self.rollback_action] = 1.0
        return masks

    def _commit_feasible(self, ue: UE) -> bool:
        secondary = ue.secondary_cell
        if secondary is None or not self.links[ue.ue_id][secondary].visible:
            return False
        enough_visibility = self.remaining_visibility[ue.ue_id, secondary] >= (
            int(self.protocol_cfg["switching_steps"])
            + int(self.protocol_cfg.get("visibility_margin_steps", 0))
        ) * self.interval_s
        dwell_ok = self.step_index - ue.last_switch_step >= int(self.protocol_cfg.get("dwell_steps", 0))
        return bool(enough_visibility and dwell_ok)

    def _advance_protocols(self) -> dict[str, int]:
        counters = {
            "handover_success": 0,
            "handover_failure": 0,
            "pingpong": 0,
            "secondary_link_lost": 0,
        }
        self._last_ho_outcomes.fill(0.0)
        for ue in self.ues:
            if ue.state in (MobilityState.PREPARED, MobilityState.DUPLICATING):
                secondary = ue.secondary_cell
                if secondary is None or not self.links[ue.ue_id][secondary].visible:
                    ue.secondary_cell = None
                    ue.state = MobilityState.SINGLE
                    ue.protocol_timer = 0
                    counters["secondary_link_lost"] += 1
                    continue
            if ue.protocol_timer > 0:
                ue.protocol_timer -= 1
            if ue.state == MobilityState.SWITCHING and ue.protocol_timer == 0:
                target = ue.secondary_cell
                target_link = self.links[ue.ue_id][target] if target is not None else None
                pressure = self.last_pressure[target] if target is not None else 10.0
                probability = 0.0
                if target_link and target_link.visible:
                    probability = (
                        float(self.protocol_cfg["handover_base_success"])
                        * target_link.packet_success_probability
                        * np.exp(-max(0.0, pressure - 1.0))
                    )
                if target is not None and self.rng.random() < probability:
                    old_primary = ue.primary_cell
                    returning_to_previous = target == ue.last_primary_cell
                    ue.primary_cell = target
                    ue.secondary_cell = None
                    if (
                        returning_to_previous
                        and self.step_index - ue.last_switch_step <= int(self.protocol_cfg["pingpong_window_steps"])
                    ):
                        counters["pingpong"] += 1
                    ue.last_primary_cell = old_primary
                    ue.last_switch_step = self.step_index
                    ue.state = MobilityState.COOLDOWN
                    ue.protocol_timer = int(self.protocol_cfg["cooldown_steps"])
                    counters["handover_success"] += 1
                    self._last_ho_outcomes[ue.ue_id] = 1.0
                else:
                    ue.secondary_cell = None
                    ue.state = MobilityState.SINGLE
                    counters["handover_failure"] += 1
                    self._last_ho_outcomes[ue.ue_id] = -1.0
            elif ue.state == MobilityState.COOLDOWN and ue.protocol_timer == 0:
                ue.state = MobilityState.SINGLE
        return counters

    def _apply_actions(self, actions: np.ndarray, masks: np.ndarray) -> dict[str, int]:
        counters = {
            "handover_started": 0,
            "invalid_actions": 0,
            "preparation_requested": 0,
            "preparation_rejected": 0,
        }
        valid_actions = np.full(self.num_agents, int(Action.STAY), dtype=np.int64)
        self._last_handover_started.fill(0.0)
        for ue, raw_action in zip(self.ues, actions):
            action = int(raw_action)
            if action < 0 or action >= self.action_dim or masks[ue.ue_id, action] == 0:
                counters["invalid_actions"] += 1
                action = int(Action.STAY)
            valid_actions[ue.ue_id] = action
        # Release capacity first, then admit simultaneous preparations in UE-id order.
        for ue in self.ues:
            action = int(valid_actions[ue.ue_id])
            if action in (self.release_action, self.rollback_action):
                ue.secondary_cell = None
                ue.state = MobilityState.SINGLE
                ue.protocol_timer = 0
        reservation = self._compute_preparation_reservations()
        rho = float(self.protocol_cfg["preparation_resource_fraction"])
        for ue in self.ues:
            action = int(valid_actions[ue.ue_id])
            if self.prepare_start <= action < self.prepare_start + self.candidate_count:
                counters["preparation_requested"] += 1
                slot = action - self.prepare_start
                if slot < len(self.candidates[ue.ue_id]):
                    target = self.candidates[ue.ue_id][slot]
                    if reservation[target] + rho <= 1.0 + 1e-12:
                        ue.secondary_cell = target
                        ue.state = MobilityState.PREPARED
                        ue.protocol_timer = int(self.protocol_cfg["preparation_steps"])
                        reservation[target] += rho
                    else:
                        ue.secondary_cell = None
                        ue.state = MobilityState.SINGLE
                        ue.protocol_timer = 0
                        counters["preparation_rejected"] += 1
            elif action == self.dup_on_action:
                ue.state = MobilityState.DUPLICATING
            elif action == self.dup_off_action:
                ue.state = MobilityState.PREPARED
            elif action == self.commit_action:
                ue.state = MobilityState.SWITCHING
                ue.protocol_timer = int(self.protocol_cfg["switching_steps"])
                counters["handover_started"] += 1
                self._last_handover_started[ue.ue_id] = 1.0
        return counters

    def _compute_preparation_reservations(self) -> np.ndarray:
        reservation = np.zeros(self.num_cells, dtype=np.float64)
        rho = float(self.protocol_cfg["preparation_resource_fraction"])
        for ue in self.ues:
            if ue.state in (
                MobilityState.PREPARED,
                MobilityState.DUPLICATING,
                MobilityState.SWITCHING,
            ) and ue.secondary_cell is not None:
                reservation[ue.secondary_cell] += rho
        return reservation

    def _compute_required_rates(self, traffic_multiplier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        queue_packets = np.asarray([ue.queue_packets for ue in self.ues], dtype=np.float64)
        queue_limited = queue_packets * int(self.traffic_cfg["packet_size_bits"]) / max(self.interval_s, 1e-8)
        traffic_rate = np.asarray(
            [ue.mean_arrival_mbps * 1e6 for ue in self.ues], dtype=np.float64
        ) * np.asarray(traffic_multiplier, dtype=np.float64)
        active = queue_packets > 0.0
        return active.astype(np.float64), np.where(active, np.minimum(traffic_rate, queue_limited), 0.0)

    def _schedule_and_transmit(self, required_rates: np.ndarray, active: np.ndarray) -> dict[str, float]:
        requests: list[LegRequest] = []
        bandwidth = np.array([satellite.bandwidth_hz for satellite in self.satellites])
        capacity_scale = np.array([satellite.effective_capacity_scale for satellite in self.satellites])
        unschedulable_data_leg_count = 0

        def add_usable_request(ue: UE, cell_id: int, demand: float, is_secondary: bool) -> None:
            nonlocal unschedulable_data_leg_count
            link = self.links[ue.ue_id][cell_id]
            unit_capacity = bandwidth[cell_id] * link.spectral_efficiency * capacity_scale[cell_id]
            if (
                not link.visible
                or link.spectral_efficiency <= self.min_schedulable_spectral_efficiency
                or unit_capacity <= self.epsilon_capacity
            ):
                unschedulable_data_leg_count += 1
                return
            requests.append(
                LegRequest(
                    ue.ue_id,
                    cell_id,
                    link.spectral_efficiency,
                    ue.qos_priority,
                    demand,
                    is_secondary,
                )
            )

        for ue in self.ues:
            demand = float(required_rates[ue.ue_id])
            if demand <= 0.0:
                continue
            if ue.primary_cell is not None:
                add_usable_request(ue, ue.primary_cell, demand, False)
            if ue.state == MobilityState.DUPLICATING and ue.secondary_cell is not None:
                add_usable_request(ue, ue.secondary_cell, demand, True)
        preparation = self._compute_preparation_reservations()
        if np.any(preparation > 1.0 + 1e-9):
            raise RuntimeError("preparation admission violated the shared-capacity constraint")
        schedule = self.scheduler.allocate(requests, bandwidth, capacity_scale, 1.0 - preparation)
        expected_goodput = np.zeros(self.num_agents, dtype=np.float64)
        delivered_by_ue = np.zeros(self.num_agents, dtype=np.float64)
        duplicated_bits = 0.0
        useful_bits = 0.0
        delivered = 0
        queue_snapshot_rates = np.asarray(
            [ue.queue_packets * ue.packet_size_bits / max(self.interval_s, 1e-8) for ue in self.ues],
            dtype=np.float64,
        )
        for ue in self.ues:
            primary = ue.primary_cell
            primary_rate = schedule.rates_bps.get((ue.ue_id, primary), 0.0)
            primary_success = self.links[ue.ue_id][primary].packet_success_probability if primary is not None else 0.0
            if ue.state == MobilityState.DUPLICATING and ue.secondary_cell is not None:
                secondary = ue.secondary_cell
                secondary_rate = schedule.rates_bps.get((ue.ue_id, secondary), 0.0)
                rate = min(primary_rate, secondary_rate)
                probability = joint_delivery_probability(
                    primary_success,
                    self.links[ue.ue_id][secondary].packet_success_probability,
                    failure_correlation=0.15,
                )
            else:
                rate = primary_rate
                probability = primary_success
            stream_rate = min(rate, ue.queue_packets * ue.packet_size_bits / max(self.interval_s, 1e-8))
            expected_goodput[ue.ue_id] = stream_rate * probability
            useful_bits += stream_rate * self.interval_s
            if ue.state == MobilityState.DUPLICATING and ue.secondary_cell is not None:
                duplicated_bits += stream_rate * self.interval_s
            packet_budget = int(rate * self.interval_s / ue.packet_size_bits)
            delivered_by_ue[ue.ue_id] = ue.deliver(packet_budget, probability, self.rng)
            delivered += int(delivered_by_ue[ue.ue_id])
        allocation = schedule.utilization
        total_fraction = preparation + allocation
        violations = int(np.count_nonzero(total_fraction > 1.0 + 1e-9))
        self.last_utilization = total_fraction
        self.last_requested_pressure = preparation + schedule.pressure
        self.last_pressure = self.last_requested_pressure.copy()
        epsilon = float(self.reward_cfg.get("epsilon", 1e-8))
        deficits = np.where(
            active > 0.0,
            np.maximum(required_rates - expected_goodput, 0.0) / (required_rates + epsilon),
            0.0,
        )
        deficits = np.clip(deficits, 0.0, 1.0)
        service_deficit = float(np.clip(deficits[active > 0.0].mean() if np.any(active > 0.0) else 0.0, 0.0, 1.0))
        realized_goodput = (
            delivered_by_ue
            * int(self.traffic_cfg["packet_size_bits"])
            / max(self.interval_s, 1e-8)
        )
        service_ratios = np.where(
            active > 0.0,
            realized_goodput / (required_rates + epsilon),
            0.0,
        )
        service_ratio_mean = float(
            np.clip(service_ratios[active > 0.0], 0.0, 2.0).mean()
            if np.any(active > 0.0)
            else 0.0
        )
        service_ratio_sum = float(np.clip(service_ratios[active > 0.0], 0.0, 2.0).sum())
        # Dense training deficit targets full R_req. Hard outage instead tests the
        # configurable minimum acceptable service beta_out * R_req.
        strict_shortfall = (active > 0.0) & (realized_goodput < required_rates)
        outage = (active > 0.0) & (
            realized_goodput < self.outage_service_ratio * required_rates
        )
        duplication_train_cost = (
            duplicated_bits / (useful_bits + duplicated_bits + epsilon)
            if useful_bits > 0.0 or duplicated_bits > 0.0
            else 0.0
        )
        preparation_train_cost = float(preparation.mean())
        duplication_gain = np.zeros(self.num_agents, dtype=np.float64)
        resource_outcome = np.zeros(self.num_agents, dtype=np.float64)
        congestion_outcome = np.zeros(self.num_agents, dtype=np.float64)
        eta_preparation = float(self.reward_cfg["eta_preparation"])
        for ue in self.ues:
            ue_id = ue.ue_id
            primary = ue.primary_cell
            primary_rate = schedule.rates_bps.get((ue_id, primary), 0.0)
            queue_rate = queue_snapshot_rates[ue_id]
            primary_stream = min(primary_rate, queue_rate)
            primary_success = self.links[ue_id][primary].packet_success_probability if primary is not None else 0.0
            primary_goodput = primary_stream * primary_success
            if ue.state == MobilityState.DUPLICATING:
                duplication_gain[ue_id] = np.clip(
                    max(0.0, expected_goodput[ue_id] - primary_goodput)
                    / (required_rates[ue_id] + epsilon),
                    0.0,
                    1.0,
                )
            duplication_indicator = float(ue.state == MobilityState.DUPLICATING)
            preparation_indicator = float(
                ue.state in (MobilityState.PREPARED, MobilityState.DUPLICATING)
            )
            resource_outcome[ue_id] = (
                duplication_indicator + eta_preparation * preparation_indicator
            ) / (1.0 + eta_preparation)
            experienced_pressure = self.last_pressure[primary] if primary is not None else 0.0
            if ue.secondary_cell is not None:
                experienced_pressure = max(experienced_pressure, self.last_pressure[ue.secondary_cell])
            congestion_outcome[ue_id] = experienced_pressure / (1.0 + experienced_pressure)
        self._last_agent_active = active.copy()
        self._last_agent_outcomes = np.column_stack(
            [
                self._last_ho_outcomes,
                outage.astype(np.float64),
                deficits,
                duplication_gain,
                resource_outcome,
                congestion_outcome,
            ]
        )
        return {
            "delivered_packets": float(delivered),
            "service_deficit": service_deficit,
            "duplication_train_cost": float(np.clip(duplication_train_cost, 0.0, 1.0)),
            "preparation_train_cost": preparation_train_cost,
            "active_ue_count": float(active.sum()),
            "outage_ue_count": float(outage.sum()),
            "strict_shortfall_ue_count": float(strict_shortfall.sum()),
            "service_ratio_mean": service_ratio_mean,
            "service_ratio_sum": service_ratio_sum,
            "outage_service_ratio": self.outage_service_ratio,
            "useful_bits": float(useful_bits),
            "duplicated_bits": float(duplicated_bits),
            "mean_utilization": float(self.last_utilization.mean()),
            "max_pressure": float(self.last_pressure.max(initial=0.0)),
            "requested_pressure_mean": float(self.last_requested_pressure.mean()),
            "requested_pressure_max": float(self.last_requested_pressure.max(initial=0.0)),
            "overloaded_satellite_count": float(
                np.count_nonzero(self.last_requested_pressure > 1.0)
            ),
            "data_overload_event_rate": float(
                np.count_nonzero(self.last_requested_pressure > 1.0) / self.num_cells
            ),
            "capacity_violation_count": float(violations),
            "unschedulable_data_leg_count": float(unschedulable_data_leg_count),
            "request_saturation_count": float(schedule.request_saturation_count),
            "mean_duplication_gain": float(duplication_gain.mean()),
            "mean_memory_resource_outcome": float(resource_outcome.mean()),
            "mean_memory_congestion_outcome": float(congestion_outcome.mean()),
        }

    def step(self, actions: np.ndarray) -> StepOutput:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_agents,):
            raise ValueError(f"actions must have shape {(self.num_agents,)}, got {actions.shape}")
        protocol_counts = self._advance_protocols()
        masks = self.action_masks()
        valid_action_count = float(np.sum(masks))
        action_fractions = {
            "action_stay_fraction": float(np.count_nonzero(actions == Action.STAY)) / self.num_agents,
            "action_prepare_fraction": float(
                np.count_nonzero((actions >= self.prepare_start) & (actions < self.prepare_start + self.candidate_count))
            ) / self.num_agents,
            "action_dup_on_fraction": float(np.count_nonzero(actions == self.dup_on_action)) / self.num_agents,
            "action_dup_off_fraction": float(np.count_nonzero(actions == self.dup_off_action)) / self.num_agents,
            "action_commit_fraction": float(np.count_nonzero(actions == self.commit_action)) / self.num_agents,
            "action_release_fraction": float(np.count_nonzero(actions == self.release_action)) / self.num_agents,
            "action_rollback_fraction": float(np.count_nonzero(actions == self.rollback_action)) / self.num_agents,
        }
        action_counts = self._apply_actions(actions, masks)
        effects = self._active_effects
        arrivals = 0
        overflow_before = sum(ue.overflow_packets for ue in self.ues)
        for ue in self.ues:
            arrivals += ue.generate_traffic(self.interval_s, effects.traffic_multiplier[ue.ue_id], self.rng)
        active, required_rates = self._compute_required_rates(effects.traffic_multiplier)
        tx_metrics = self._schedule_and_transmit(required_rates, active)
        missed = sum(ue.age_and_drop_expired(self.interval_s) for ue in self.ues)
        overflow = sum(ue.overflow_packets for ue in self.ues) - overflow_before
        epsilon = float(self.reward_cfg.get("epsilon", 1e-8))
        active_count = tx_metrics["active_ue_count"]
        handover_train_cost = (
            float(np.sum(self._last_handover_started * active)) / (active_count + epsilon)
            if active_count > 0.0
            else 0.0
        )
        # One-time normalized abstraction of signaling, switching interruption,
        # and mobility-management overhead for valid active-UE COMMIT starts.
        handover_train_cost = float(np.clip(handover_train_cost, 0.0, 1.0))
        handover_control_cost = float(self.reward_cfg["eta_handover"]) * handover_train_cost
        control_cost = (
            tx_metrics["duplication_train_cost"]
            + float(self.reward_cfg["eta_preparation"]) * tx_metrics["preparation_train_cost"]
            + handover_control_cost
        )
        outage_cost = tx_metrics["outage_ue_count"] / (tx_metrics["active_ue_count"] + epsilon)
        reward, _ = compute_training_reward(
            tx_metrics["service_deficit"],
            control_cost,
            outage_cost,
            float(self.reward_cfg["lambda_resource"]),
            use_constraint=False,
        )
        delivered_bits = tx_metrics["delivered_packets"] * int(self.traffic_cfg["packet_size_bits"])
        traffic_target_bits = float(required_rates.sum() * self.interval_s)
        throughput_deficit = max(0.0, 1.0 - delivered_bits / max(traffic_target_bits, epsilon)) if traffic_target_bits else 0.0
        self.step_index += 1
        self.time_s += self.interval_s
        self._active_effects = self.events.effects(self.step_index)
        for cell_id, satellite in enumerate(self.satellites):
            satellite.capacity_multiplier = float(self._active_effects.cell_capacity_multiplier[cell_id])
        self._refresh_links()
        observations, global_state, next_masks = self._build_output_state()
        done = self.step_index >= int(self.env_cfg["episode_steps"])
        info = {
            **tx_metrics,
            **{key: float(value) for key, value in protocol_counts.items()},
            **{key: float(value) for key, value in action_counts.items()},
            **action_fractions,
            "arrivals": float(arrivals),
            "deadline_misses": float(missed),
            "overflow_packets": float(overflow),
            "throughput_deficit": float(throughput_deficit),
            "handover_train_cost": handover_train_cost,
            "handover_control_cost": handover_control_cost,
            "active_handover_started": float(np.sum(self._last_handover_started * active)),
            "control_cost": float(control_cost),
            "resource_cost": float(control_cost),
            "outage_cost": float(outage_cost),
            "outage_ratio": float(outage_cost),
            "dro": float(tx_metrics["duplicated_bits"] / (tx_metrics["useful_bits"] + epsilon)),
            "mean_valid_action_count": valid_action_count / self.num_agents,
            "team_reward": float(reward),
            "constraint_resource_cost": float(control_cost),
            "constraint_outage_cost": float(outage_cost),
            "constraint_penalty": 0.0,
            "constraint_reward": 0.0,
        }
        return StepOutput(
            observations,
            global_state,
            next_masks,
            np.full(self.num_agents, reward, dtype=np.float64),
            done,
            info,
            agent_outcomes=self._last_agent_outcomes.copy(),
            agent_active=self._last_agent_active.copy(),
            requested_pressure=self.last_requested_pressure.copy(),
        )

    def current_required_rates(self) -> tuple[np.ndarray, np.ndarray]:
        """Current observable demand proxy for the non-mutating analytical reasoner."""
        return self._compute_required_rates(self._active_effects.traffic_multiplier)

    def _build_output_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = np.stack([self._observation(ue) for ue in self.ues]).astype(np.float64)
        global_state = np.concatenate(
            [
                observations.reshape(-1),
                self.last_utilization,
                np.clip(self.last_pressure / 3.0, 0.0, 2.0),
            ]
        ).astype(np.float64)
        return observations, global_state, self.action_masks()

    def _observation(self, ue: UE) -> np.ndarray:
        duplication = float(ue.state == MobilityState.DUPLICATING)
        one_hot_state = np.eye(len(MobilityState), dtype=np.float64)[int(ue.state)]
        values: list[float] = [
            ue.queue_packets / max(1, ue.max_queue_packets),
            np.clip(ue.head_of_line_delay_s / max(ue.deadline_s, 1e-6), 0.0, 2.0),
            np.clip(ue.mean_arrival_mbps / 5.0, 0.0, 2.0),
            ue.reliability_target,
            duplication,
            *one_hot_state.tolist(),
            np.clip(ue.protocol_timer / 10.0, 0.0, 1.0),
        ]
        primary = ue.primary_cell
        if primary is None:
            values.extend([0.0] * 5)
        else:
            link = self.links[ue.ue_id][primary]
            values.extend(
                [
                    float(link.visible),
                    np.clip((link.sinr_db + 20.0) / 50.0, 0.0, 1.0),
                    link.spectral_efficiency / max(1.0, float(self.channel_cfg["spectral_efficiency_cap"])),
                    np.clip(self.remaining_visibility[ue.ue_id, primary] / 600.0, 0.0, 1.0),
                    np.clip(self.last_pressure[primary] / 2.0, 0.0, 1.5),
                ]
            )
        for slot in range(self.candidate_count):
            if slot >= len(self.candidates[ue.ue_id]):
                values.extend([0.0] * 7)
                continue
            cell_id = self.candidates[ue.ue_id][slot]
            link = self.links[ue.ue_id][cell_id]
            values.extend(
                [
                    1.0,
                    float(cell_id == ue.secondary_cell),
                    np.clip((link.sinr_db + 20.0) / 50.0, 0.0, 1.0),
                    link.spectral_efficiency / max(1.0, float(self.channel_cfg["spectral_efficiency_cap"])),
                    np.clip(self.remaining_visibility[ue.ue_id, cell_id] / 600.0, 0.0, 1.0),
                    np.clip(self.last_pressure[cell_id] / 2.0, 0.0, 1.5),
                    np.clip(self.satellites[cell_id].effective_capacity_scale, 0.0, 1.0),
                ]
            )
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.obs_dim,):
            raise RuntimeError(f"observation shape mismatch: expected {self.obs_dim}, got {result.shape}")
        return result

    def sample_valid_actions(self) -> np.ndarray:
        masks = self.action_masks()
        return np.array(
            [self.rng.choice(np.flatnonzero(mask > 0.0)) for mask in masks], dtype=np.int64
        )
