from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from enum import IntEnum
from typing import Any

import numpy as np

from .channel import ChannelModel, LinkResult
from .coordinator import compute_training_reward
from .events import EventGenerator
from .pdcp import joint_delivery_probability
from .satellite import (
    SatelliteCell,
    circular_orbital_period_s,
    elevation_and_distance,
    estimate_remaining_visibility,
    ground_position_ecef,
)
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
        self.challenge_version = str(self.env_cfg.get("challenge_version", "v2")).lower()
        self.v3_enabled = self.challenge_version == "v3"
        self.v3_cfg = config.get("v3", {}) if self.v3_enabled else {}
        self.telemetry_cfg = self.v3_cfg.get("observation", {})
        self.latent_cfg = self.v3_cfg.get("latent_degradation", {})
        self.preparation_admission_cfg = self.v3_cfg.get("preparation_admission", {})
        self.duplication_cfg = self.v3_cfg.get("duplication", {})
        self.coordination_demand_cfg = self.v3_cfg.get("coordination_demand", {})
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
        self.environment_seed = int(config.get("environment_seed", config.get("seed", 0)))
        self.rng = np.random.default_rng(self.environment_seed)
        self.interval_s = float(self.env_cfg["control_interval_s"])
        self.min_elevation_rad = np.radians(float(self.env_cfg["min_elevation_deg"]))
        self.satellites = self._make_satellites()
        if len(self.satellites) != self.num_cells:
            # A Walker slice determines its own (visibility-filtered) size.
            self.num_cells = len(self.satellites)
            self.global_state_dim = self.num_agents * self.obs_dim + 2 * self.num_cells
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
            config.get("events", []),
            self.num_agents,
            self.num_cells,
            self.rng,
            randomized=self.v3_enabled and bool(self.v3_cfg.get("randomize_events", True)),
        )
        self.step_index = 0
        self.time_s = 0.0
        self.last_utilization = np.zeros(self.num_cells, dtype=np.float64)
        self.last_pressure = np.zeros(self.num_cells, dtype=np.float64)
        self.last_requested_pressure = np.zeros(self.num_cells, dtype=np.float64)
        self.last_serving_demand = np.zeros(self.num_cells, dtype=np.float64)
        self.last_preparation_demand = np.zeros(self.num_cells, dtype=np.float64)
        self.last_preparation_intent_count = np.zeros(self.num_cells, dtype=np.float64)
        self.last_duplication_demand = np.zeros(self.num_cells, dtype=np.float64)
        self.last_coordination_demand = np.zeros(self.num_cells, dtype=np.float64)
        self.links: list[list[LinkResult]] = []
        self.remaining_visibility = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self.candidates: list[list[int]] = []
        self._active_effects = self.events.effects(0)
        self._last_ho_outcomes = np.zeros(self.num_agents, dtype=np.float64)
        self._last_handover_started = np.zeros(self.num_agents, dtype=np.float64)
        # Fraction of the current epoch lost to handover execution
        # (TR 38.821 Sec. 7.3.2.1.1: DL interruption ~ 2 RTT + processing).
        self._ho_interruption = np.zeros(self.num_agents, dtype=np.float64)
        self._last_agent_outcomes = np.zeros((self.num_agents, 6), dtype=np.float64)
        self._last_agent_active = np.zeros(self.num_agents, dtype=np.float64)
        self.observed_sinr = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self.observed_efficiency = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self.observed_elevation = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self.observed_sinr_trend = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self.observed_load = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self._telemetry_history: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque()
        self._sinr_noise_state = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self._load_noise_state = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self.latent_degradation_state = np.zeros((self.num_agents, self.num_cells), dtype=np.int8)
        self.latent_degradation_timer = np.zeros((self.num_agents, self.num_cells), dtype=np.int64)
        self.latent_degradation_loss_db = np.zeros((self.num_agents, self.num_cells), dtype=np.float64)
        self._prepared_since = np.full(self.num_agents, -1, dtype=np.int64)
        self.preparation_requests_by_ue = np.zeros(self.num_agents, dtype=np.int64)
        self.preparation_admissions_by_ue = np.zeros(self.num_agents, dtype=np.int64)
        self.preparation_rejections_by_ue = np.zeros(self.num_agents, dtype=np.int64)

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

    # Service region used for UE placement (radians); shared with the
    # Walker-slice visibility filter.
    UE_LAT_HALF_RAD = 0.04
    UE_LON_HALF_RAD = 0.05

    def _constellation_cfg(self) -> dict:
        return dict(self.sat_cfg.get("constellation", {}) or {})

    def _service_radius_rad(self) -> float | None:
        """Radius of a circular service area (one earth-fixed beam); None keeps
        the legacy rectangular area."""
        radius_km = (self.config.get("service_area", {}) or {}).get("radius_km")
        return None if radius_km is None else float(radius_km) * 1e3 / 6_371_000.0

    def _region_probe_offsets(self) -> list[tuple[float, float]]:
        radius = self._service_radius_rad()
        if radius is None:
            return [
                (la, lo)
                for la in (-self.UE_LAT_HALF_RAD, 0.0, self.UE_LAT_HALF_RAD)
                for lo in (-self.UE_LON_HALF_RAD, 0.0, self.UE_LON_HALF_RAD)
            ]
        return [(0.0, 0.0)] + [
            (radius * np.sin(a), radius * np.cos(a)) for a in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ]

    def _link_budget(self) -> tuple[float, float, float]:
        """Return (bandwidth_hz, tx_power_dbm, tx_gain_dbi). With
        eirp_density_dbw_per_mhz set (3GPP TR 38.821 Table 6.1.1.1-1/2), the
        transmit power follows from EIRP density, bandwidth and boresight gain."""
        bandwidth_mhz = float(self.sat_cfg["bandwidth_mhz"])
        gain = float(self.sat_cfg["tx_gain_dbi"])
        density = self.sat_cfg.get("eirp_density_dbw_per_mhz")
        if density is None:
            return bandwidth_mhz * 1e6, float(self.sat_cfg["tx_power_dbm"]), gain
        eirp_dbm = float(density) + 30.0 + 10.0 * np.log10(bandwidth_mhz)
        return bandwidth_mhz * 1e6, eirp_dbm - gain, gain

    def _make_satellites(self) -> list[SatelliteCell]:
        if self._constellation_cfg().get("type", "local_slice") == "walker_slice":
            return self._make_walker_slice()
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

    def _make_walker_slice(self) -> list[SatelliteCell]:
        """Adjacent planes of a Walker shell, restricted to satellites that can
        become visible from the service region during an episode."""
        cfg = self._constellation_cfg()
        altitude_m = float(self.sat_cfg["altitude_km"]) * 1e3
        inclination = np.radians(float(self.sat_cfg["inclination_deg"]))
        period = self.sat_cfg.get("orbital_period_s", "auto")
        period = circular_orbital_period_s(altitude_m) if period in (None, "auto") else float(period)
        planes = int(cfg.get("planes", 2))
        per_plane = int(cfg.get("sats_per_plane", 22))
        raan_spacing = np.radians(float(cfg.get("raan_spacing_deg", 5.0)))
        in_plane = 2.0 * np.pi / per_plane
        phase_offset = np.radians(float(cfg.get("phase_offset_deg", np.degrees(in_plane) / 2.0)))
        rate = 2.0 * np.pi / period
        self.constellation_period_s = period
        self.max_epoch_offset_s = period / per_plane if bool(cfg.get("random_epoch", True)) else 0.0
        horizon_s = float(self.env_cfg["episode_steps"]) * float(self.env_cfg["control_interval_s"])
        horizon_s += self.max_epoch_offset_s
        margin = np.radians(float(cfg.get("visibility_margin_deg", 5.0)))
        threshold = np.radians(float(self.env_cfg["min_elevation_deg"])) - margin
        probes = [ground_position_ecef(la, lo) for la, lo in self._region_probe_offsets()]
        times = np.arange(0.0, horizon_s + 1e-9, 5.0)
        bandwidth_hz, tx_power_dbm, tx_gain_dbi = self._link_budget()
        beam_cfg = dict(self.sat_cfg.get("beam", {}) or {})
        earth_fixed = beam_cfg.get("type") == "earth_fixed"
        beam_center = ground_position_ecef(0.0, 0.0) if earth_fixed else None
        aperture_radius = float(beam_cfg.get("aperture_diameter_m", 0.0)) / 2.0 if earth_fixed else 0.0
        cells: list[SatelliteCell] = []
        for k in range(planes):
            raan = (k - (planes - 1) / 2.0) * raan_spacing
            for slot in range(per_plane):
                phase = float(np.angle(np.exp(1j * (slot * in_plane + k * phase_offset))))
                cell = SatelliteCell(
                    cell_id=len(cells),
                    altitude_m=altitude_m,
                    inclination_rad=inclination,
                    phase_rad=phase,
                    angular_rate_rad_s=rate,
                    bandwidth_hz=bandwidth_hz,
                    tx_power_dbm=tx_power_dbm,
                    tx_gain_dbi=tx_gain_dbi,
                    base_capacity_scale=float(self.sat_cfg.get("capacity_scale", 1.0)),
                    raan_rad=float(raan),
                    plane_index=k,
                    beam_center_ecef=beam_center,
                    aperture_radius_m=aperture_radius,
                )
                visible = any(
                    elevation_and_distance(q, cell.position_ecef(t))[0] >= threshold
                    for t in times
                    for q in probes
                )
                if visible:
                    cells.append(cell)
        if not cells:
            raise ValueError("walker_slice: no satellite becomes visible from the service region")
        for index, cell in enumerate(cells):
            cell.cell_id = index
        return cells

    def _resample_constellation_epoch(self) -> None:
        offset = float(getattr(self, "max_epoch_offset_s", 0.0))
        if offset <= 0.0:
            return
        epoch = float(self.rng.uniform(0.0, offset))
        for cell in self.satellites:
            cell.epoch_offset_s = epoch

    def _make_ues(self) -> list[UE]:
        base_rate = float(self.traffic_cfg["mean_arrival_mbps"])
        spread = float(self.traffic_cfg.get("arrival_spread", 0.0))
        ues = []
        radius = self._service_radius_rad()
        for index in range(self.num_agents):
            rate = base_rate * self.rng.uniform(1.0 - spread, 1.0 + spread)
            if radius is None:
                lat = float(self.rng.uniform(-self.UE_LAT_HALF_RAD, self.UE_LAT_HALF_RAD))
                lon = float(self.rng.uniform(-self.UE_LON_HALF_RAD, self.UE_LON_HALF_RAD))
            else:
                # Uniform in a disk (3GPP calibration: UEs uniform in the beam cell).
                r = radius * np.sqrt(self.rng.uniform())
                phi = self.rng.uniform(0.0, 2.0 * np.pi)
                lat, lon = float(r * np.sin(phi)), float(r * np.cos(phi))
            ues.append(
                UE(
                    ue_id=index,
                    latitude_rad=lat,
                    longitude_rad=lon,
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
        self._resample_constellation_epoch()
        self.step_index = 0
        self.time_s = 0.0
        self.last_utilization.fill(0.0)
        self.last_pressure.fill(0.0)
        self.last_requested_pressure.fill(0.0)
        self.last_serving_demand.fill(0.0)
        self.last_preparation_demand.fill(0.0)
        self.last_preparation_intent_count.fill(0.0)
        self.last_duplication_demand.fill(0.0)
        self.last_coordination_demand.fill(0.0)
        self.channel.reset()
        self.events.reset()
        self._active_effects = self.events.effects(0)
        self._last_ho_outcomes.fill(0.0)
        self._last_handover_started.fill(0.0)
        self._last_agent_outcomes.fill(0.0)
        self._last_agent_active.fill(0.0)
        self._prepared_since.fill(-1)
        self.preparation_requests_by_ue.fill(0)
        self.preparation_admissions_by_ue.fill(0)
        self.preparation_rejections_by_ue.fill(0)
        self._reset_v3_runtime()
        for satellite in self.satellites:
            satellite.capacity_multiplier = 1.0
        for ue in self.ues:
            ue.reset_runtime()
            if self.v3_enabled:
                # V3 seeded episode resets must not inherit a serving cell from
                # the preceding episode; V2 keeps its historical reset path.
                ue.primary_cell = None
        self._refresh_links()
        for ue in self.ues:
            visible = [cell for cell in range(self.num_cells) if self.links[ue.ue_id][cell].visible]
            if not visible:
                ue.primary_cell = int(np.argmax([link.elevation_deg for link in self.links[ue.ue_id]]))
            else:
                ue.primary_cell = max(visible, key=lambda cell: self.links[ue.ue_id][cell].sinr_db)
        if self.v3_enabled:
            self.candidates = [self._select_candidates(ue) for ue in self.ues]
        return self._build_output_state()

    def _reset_v3_runtime(self) -> None:
        self._telemetry_history.clear()
        self._sinr_noise_state.fill(0.0)
        self._load_noise_state.fill(0.0)
        self.observed_sinr.fill(0.0)
        self.observed_efficiency.fill(0.0)
        self.observed_elevation.fill(0.0)
        self.observed_sinr_trend.fill(0.0)
        self.observed_load.fill(0.0)
        self.latent_degradation_state.fill(0)
        self.latent_degradation_timer.fill(0)
        self.latent_degradation_loss_db.fill(0.0)
        if not self.v3_enabled:
            return
        initial_probability = float(self.latent_cfg.get("initial_degraded_probability", 0.05))
        initially_degraded = self.rng.random(self.latent_degradation_state.shape) < initial_probability
        self._enter_degraded(initially_degraded)

    def _sample_duration(self, name: str, default: tuple[int, int]) -> np.ndarray:
        low, high = self.latent_cfg.get(name, list(default))
        return self.rng.integers(int(low), int(high) + 1, size=(self.num_agents, self.num_cells))

    def _enter_degraded(self, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        low, high = self.latent_cfg.get("loss_db_range", [7.0, 15.0])
        losses = self.rng.uniform(float(low), float(high), size=mask.shape)
        durations = self._sample_duration("degraded_duration_steps", (8, 24))
        self.latent_degradation_state[mask] = 1
        self.latent_degradation_timer[mask] = durations[mask]
        self.latent_degradation_loss_db[mask] = losses[mask]

    def _advance_latent_regimes(self) -> None:
        if not self.v3_enabled:
            return
        active = self.latent_degradation_state > 0
        self.latent_degradation_timer[active] -= 1
        degraded_done = (self.latent_degradation_state == 1) & (self.latent_degradation_timer <= 0)
        if np.any(degraded_done):
            durations = self._sample_duration("recovering_duration_steps", (4, 10))
            self.latent_degradation_state[degraded_done] = 2
            self.latent_degradation_timer[degraded_done] = durations[degraded_done]
        recovering = self.latent_degradation_state == 2
        if np.any(recovering):
            recovery_steps = np.maximum(self.latent_degradation_timer[recovering], 1)
            self.latent_degradation_loss_db[recovering] *= (
                (recovery_steps - 1) / recovery_steps
            )
        recovered = recovering & (self.latent_degradation_timer <= 0)
        self.latent_degradation_state[recovered] = 0
        self.latent_degradation_loss_db[recovered] = 0.0
        starts = (self.latent_degradation_state == 0) & (
            self.rng.random(self.latent_degradation_state.shape)
            < float(self.latent_cfg.get("entry_probability_per_step", 0.006))
        )
        self._enter_degraded(starts)

    def _update_observed_telemetry(self) -> None:
        true_sinr = np.asarray([[link.sinr_db for link in row] for row in self.links])
        true_efficiency = np.asarray([[link.spectral_efficiency for link in row] for row in self.links])
        true_elevation = np.asarray([[link.elevation_deg for link in row] for row in self.links])
        true_load = np.broadcast_to(self.last_requested_pressure, true_sinr.shape).copy()
        self._telemetry_history.append((true_sinr, true_efficiency, true_elevation, true_load))
        lag = max(0, int(self.telemetry_cfg.get("lag_steps", 1)))
        while len(self._telemetry_history) > lag + 1:
            delayed = self._telemetry_history.popleft()
        if len(self._telemetry_history) <= lag:
            delayed = self._telemetry_history[0]
        else:
            delayed = self._telemetry_history[-(lag + 1)]
        sinr, efficiency, elevation, load = (value.copy() for value in delayed)
        correlation = float(self.telemetry_cfg.get("noise_correlation", 0.80))
        innovation = np.sqrt(max(0.0, 1.0 - correlation**2))
        self._sinr_noise_state = correlation * self._sinr_noise_state + self.rng.normal(
            0.0,
            float(self.telemetry_cfg.get("sinr_noise_std_db", 1.5)) * innovation,
            self._sinr_noise_state.shape,
        )
        self._load_noise_state = correlation * self._load_noise_state + self.rng.normal(
            0.0,
            float(self.telemetry_cfg.get("load_noise_std", 0.08)) * innovation,
            self._load_noise_state.shape,
        )
        previous = self.observed_sinr.copy()
        self.observed_sinr = sinr + self._sinr_noise_state
        self.observed_efficiency = np.clip(
            efficiency * np.power(2.0, self._sinr_noise_state / 10.0),
            0.0,
            float(self.channel_cfg["spectral_efficiency_cap"]),
        )
        self.observed_elevation = elevation
        self.observed_load = np.maximum(0.0, load + self._load_noise_state)
        self.observed_sinr_trend = self.observed_sinr - previous

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
                        extra_loss_db=float(
                            effects.link_extra_loss_db[ue.ue_id, cell_id]
                            + (
                                self.latent_degradation_loss_db[ue.ue_id, cell_id]
                                if self.v3_enabled
                                else 0.0
                            )
                        ),
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
        if self.v3_enabled:
            self._update_observed_telemetry()
        self.candidates = []
        for ue in self.ues:
            self.candidates.append(self._select_candidates(ue))

    def _select_candidates(self, ue: UE) -> list[int]:
        choices = [
            cell_id
            for cell_id in range(self.num_cells)
            if cell_id != ue.primary_cell and self.links[ue.ue_id][cell_id].visible
        ]
        if self.v3_enabled:
            # V3 candidate discovery is radio-only. Future visibility, true load,
            # latent state, and coordinator prices remain decision-layer inputs.
            choices.sort(
                key=lambda cell_id: self.observed_sinr[ue.ue_id, cell_id], reverse=True
            )
        else:
            choices.sort(
                key=lambda cell_id: (
                    self.links[ue.ue_id][cell_id].sinr_db
                    + 0.01 * self.remaining_visibility[ue.ue_id, cell_id]
                    - 5.0 * self.last_pressure[cell_id]
                ),
                reverse=True,
            )
        return choices[: self.candidate_count]

    def action_masks(self) -> np.ndarray:
        masks = np.zeros((self.num_agents, self.action_dim), dtype=np.float64)
        for ue in self.ues:
            mask = masks[ue.ue_id]
            mask[Action.STAY] = 1.0
            if ue.state == MobilityState.SINGLE:
                for slot, cell_id in enumerate(self.candidates[ue.ue_id]):
                    enough_visibility = self.v3_enabled or self.remaining_visibility[
                        ue.ue_id, cell_id
                    ] >= (
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
        # V3 keeps physical/protocol invariants in the hard mask but leaves
        # future lifetime quality to the actor and analytical reasoner.
        enough_visibility = self.v3_enabled or self.remaining_visibility[ue.ue_id, secondary] >= (
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
                    self._prepared_since[ue.ue_id] = -1
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
                self._ho_interruption[ue.ue_id] = self._ho_interruption_fraction()
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

    def _ho_interruption_fraction(self) -> float:
        interruption_ms = float(self.protocol_cfg.get("ho_interruption_ms", 0.0))
        processing_ms = float(self.protocol_cfg.get("ho_processing_ms", 0.0))
        return float(np.clip((interruption_ms + processing_ms) / 1000.0 / self.interval_s, 0.0, 1.0))

    def _apply_actions(self, actions: np.ndarray, masks: np.ndarray) -> dict[str, int]:
        counters = {
            "handover_started": 0,
            "invalid_actions": 0,
            "preparation_requested": 0,
            "preparation_rejected": 0,
            "preparation_admitted": 0,
            "release_count": 0,
            "rollback_count": 0,
            "prepare_to_release_without_commit": 0,
            "prepare_to_commit": 0,
            "prepared_duration_sum": 0,
            "prepared_duration_count": 0,
            "duplication_activation_count": 0,
        }
        valid_actions = np.full(self.num_agents, int(Action.STAY), dtype=np.int64)
        self.last_preparation_intent_count.fill(0.0)
        self._last_handover_started.fill(0.0)
        for ue, raw_action in zip(self.ues, actions):
            action = int(raw_action)
            if action < 0 or action >= self.action_dim or masks[ue.ue_id, action] == 0:
                counters["invalid_actions"] += 1
                action = int(Action.STAY)
            valid_actions[ue.ue_id] = action
        # Releases are resolved before preparation occupancy is measured.
        for ue in self.ues:
            action = int(valid_actions[ue.ue_id])
            if action in (self.release_action, self.rollback_action):
                if action == self.release_action:
                    counters["release_count"] += 1
                    if self._prepared_since[ue.ue_id] >= 0:
                        counters["prepare_to_release_without_commit"] += 1
                else:
                    counters["rollback_count"] += 1
                if self._prepared_since[ue.ue_id] >= 0:
                    counters["prepared_duration_sum"] += (
                        self.step_index - int(self._prepared_since[ue.ue_id])
                    )
                    counters["prepared_duration_count"] += 1
                self._prepared_since[ue.ue_id] = -1
                ue.secondary_cell = None
                ue.state = MobilityState.SINGLE
                ue.protocol_timer = 0
        reservation = self._compute_preparation_reservations()
        preparation_counts = np.zeros(self.num_cells, dtype=np.int64)
        for existing in self.ues:
            if existing.secondary_cell is not None and existing.state in (
                MobilityState.PREPARED,
                MobilityState.DUPLICATING,
                MobilityState.SWITCHING,
            ):
                preparation_counts[existing.secondary_cell] += 1
        control_capacity = int(
            self.preparation_admission_cfg.get("max_concurrent_per_satellite", self.num_agents)
        )
        rho = float(self.protocol_cfg["preparation_resource_fraction"])

        def apply_non_prepare_action(ue: UE, action: int) -> None:
            if action == self.dup_on_action:
                ue.state = MobilityState.DUPLICATING
                counters["duplication_activation_count"] += 1
            elif action == self.dup_off_action:
                ue.state = MobilityState.PREPARED
            elif action == self.commit_action:
                if self._prepared_since[ue.ue_id] >= 0:
                    counters["prepare_to_commit"] += 1
                    counters["prepared_duration_sum"] += (
                        self.step_index - int(self._prepared_since[ue.ue_id])
                    )
                    counters["prepared_duration_count"] += 1
                self._prepared_since[ue.ue_id] = -1
                ue.state = MobilityState.SWITCHING
                ue.protocol_timer = int(self.protocol_cfg["switching_steps"])
                counters["handover_started"] += 1
                self._last_handover_started[ue.ue_id] = 1.0

        if self.v3_enabled:
            # Atomic V3 semantics: collect every new request before selecting
            # winners. Existing reservations occupy slots, while competing new
            # requests are sampled uniformly without replacement.
            requests_by_target: list[list[UE]] = [[] for _ in range(self.num_cells)]
            for ue in self.ues:
                action = int(valid_actions[ue.ue_id])
                if self.prepare_start <= action < self.prepare_start + self.candidate_count:
                    counters["preparation_requested"] += 1
                    slot = action - self.prepare_start
                    if slot < len(self.candidates[ue.ue_id]):
                        target = self.candidates[ue.ue_id][slot]
                        requests_by_target[target].append(ue)
                        self.last_preparation_intent_count[target] += 1.0
                        self.preparation_requests_by_ue[ue.ue_id] += 1
                else:
                    apply_non_prepare_action(ue, action)

            for target, requesters in enumerate(requests_by_target):
                if not requesters:
                    continue
                control_slots = max(0, control_capacity - int(preparation_counts[target]))
                physical_slots = (
                    self.num_agents
                    if rho <= 0.0
                    else max(0, int(np.floor((1.0 + 1e-12 - reservation[target]) / rho)))
                )
                available_slots = min(control_slots, physical_slots)
                if len(requesters) <= available_slots:
                    admitted_ids = {ue.ue_id for ue in requesters}
                elif available_slots > 0:
                    requester_ids = np.asarray([ue.ue_id for ue in requesters], dtype=np.int64)
                    admitted_ids = set(
                        int(value)
                        for value in self.rng.choice(
                            requester_ids, size=available_slots, replace=False
                        )
                    )
                else:
                    admitted_ids = set()
                for ue in requesters:
                    if ue.ue_id in admitted_ids:
                        ue.secondary_cell = target
                        ue.state = MobilityState.PREPARED
                        ue.protocol_timer = int(self.protocol_cfg["preparation_steps"])
                        counters["preparation_admitted"] += 1
                        self.preparation_admissions_by_ue[ue.ue_id] += 1
                        self._prepared_since[ue.ue_id] = self.step_index
                    else:
                        ue.secondary_cell = None
                        ue.state = MobilityState.SINGLE
                        ue.protocol_timer = 0
                        counters["preparation_rejected"] += 1
                        self.preparation_rejections_by_ue[ue.ue_id] += 1
            return counters

        # V2 preserves the historical sequential admission behavior exactly.
        for ue in self.ues:
            action = int(valid_actions[ue.ue_id])
            if self.prepare_start <= action < self.prepare_start + self.candidate_count:
                counters["preparation_requested"] += 1
                slot = action - self.prepare_start
                if slot < len(self.candidates[ue.ue_id]):
                    target = self.candidates[ue.ue_id][slot]
                    self.last_preparation_intent_count[target] += 1.0
                    if reservation[target] + rho <= 1.0 + 1e-12:
                        ue.secondary_cell = target
                        ue.state = MobilityState.PREPARED
                        ue.protocol_timer = int(self.protocol_cfg["preparation_steps"])
                        reservation[target] += rho
                        preparation_counts[target] += 1
                        counters["preparation_admitted"] += 1
                        self._prepared_since[ue.ue_id] = self.step_index
                    else:
                        ue.secondary_cell = None
                        ue.state = MobilityState.SINGLE
                        ue.protocol_timer = 0
                        counters["preparation_rejected"] += 1
            else:
                apply_non_prepare_action(ue, action)
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
        serving_demand = np.zeros(self.num_cells, dtype=np.float64)
        duplication_demand = np.zeros(self.num_cells, dtype=np.float64)
        for request in requests:
            unit_capacity = (
                bandwidth[request.cell_id]
                * max(0.0, request.spectral_efficiency)
                * max(0.0, capacity_scale[request.cell_id])
            )
            fraction = min(1.0, max(0.0, request.demand_bps / (unit_capacity + self.epsilon_capacity)))
            if request.is_secondary:
                duplication_demand[request.cell_id] += fraction
            else:
                serving_demand[request.cell_id] += fraction
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
                    failure_correlation=float(
                        self.duplication_cfg.get("failure_correlation", 0.15)
                    ),
                )
            else:
                rate = primary_rate
                probability = primary_success
            rate *= 1.0 - self._ho_interruption[ue.ue_id]
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
        self.last_serving_demand = serving_demand
        if self.v3_enabled:
            preparation_capacity = max(
                1.0,
                float(
                    self.preparation_admission_cfg.get(
                        "max_concurrent_per_satellite", self.num_agents
                    )
                ),
            )
            self.last_preparation_demand = (
                self.last_preparation_intent_count / preparation_capacity
            )
        else:
            self.last_preparation_demand = preparation.copy()
        self.last_duplication_demand = duplication_demand
        if self.v3_enabled:
            self.last_coordination_demand = (
                float(self.coordination_demand_cfg.get("serving_weight", 1.0)) * serving_demand
                + float(self.coordination_demand_cfg.get("preparation_weight", 1.0))
                * self.last_preparation_demand
                + float(self.coordination_demand_cfg.get("duplication_weight", 1.0)) * duplication_demand
            )
        else:
            self.last_coordination_demand = self.last_requested_pressure.copy()
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
            "useful_duplication_interval_count": float(np.count_nonzero(duplication_gain > 1e-8)),
            "duplication_active_ue_count": float(
                sum(ue.state == MobilityState.DUPLICATING for ue in self.ues)
            ),
            "duplication_active_time_fraction": float(
                np.mean([ue.state == MobilityState.DUPLICATING for ue in self.ues])
            ),
            "serving_demand_mean": float(serving_demand.mean()),
            "serving_demand_max": float(serving_demand.max(initial=0.0)),
            "preparation_demand_mean": float(self.last_preparation_demand.mean()),
            "preparation_demand_max": float(
                self.last_preparation_demand.max(initial=0.0)
            ),
            "duplication_demand_mean": float(duplication_demand.mean()),
            "duplication_demand_max": float(duplication_demand.max(initial=0.0)),
            "aggregate_intended_demand_mean": float(self.last_coordination_demand.mean()),
            "aggregate_intended_demand_max": float(
                self.last_coordination_demand.max(initial=0.0)
            ),
            "mean_memory_resource_outcome": float(resource_outcome.mean()),
            "mean_memory_congestion_outcome": float(congestion_outcome.mean()),
        }

    def step(self, actions: np.ndarray) -> StepOutput:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_agents,):
            raise ValueError(f"actions must have shape {(self.num_agents,)}, got {actions.shape}")
        self._ho_interruption.fill(0.0)
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
        self._advance_latent_regimes()
        self._active_effects = self.events.effects(self.step_index)
        for cell_id, satellite in enumerate(self.satellites):
            satellite.capacity_multiplier = float(self._active_effects.cell_capacity_multiplier[cell_id])
        self._refresh_links()
        if self.v3_enabled:
            true_sinr = np.asarray([[link.sinr_db for link in row] for row in self.links])
            telemetry_sinr_mae = float(np.mean(np.abs(self.observed_sinr - true_sinr)))
            true_load = np.broadcast_to(self.last_requested_pressure, self.observed_load.shape)
            telemetry_load_mae = float(np.mean(np.abs(self.observed_load - true_load)))
            latent_degraded_fraction = float(np.mean(self.latent_degradation_state == 1))
            latent_recovering_fraction = float(np.mean(self.latent_degradation_state == 2))
        else:
            telemetry_sinr_mae = 0.0
            telemetry_load_mae = 0.0
            latent_degraded_fraction = 0.0
            latent_recovering_fraction = 0.0
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
            # Preparation requests per active UE and epoch (signaling constraint).
            "signaling_cost": float(action_counts.get("preparation_requested", 0)) / (active_count + epsilon),
            "outage_ratio": float(outage_cost),
            "dro": float(tx_metrics["duplicated_bits"] / (tx_metrics["useful_bits"] + epsilon)),
            "mean_valid_action_count": valid_action_count / self.num_agents,
            "prepare_count": float(action_counts["preparation_requested"]),
            "commit_count": float(action_counts["handover_started"]),
            "telemetry_sinr_mae_db": telemetry_sinr_mae,
            "telemetry_load_mae": telemetry_load_mae,
            "latent_degraded_fraction": latent_degraded_fraction,
            "latent_recovering_fraction": latent_recovering_fraction,
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
            requested_pressure=self.last_coordination_demand.copy(),
        )

    def current_required_rates(self) -> tuple[np.ndarray, np.ndarray]:
        """True physical demand, retained for V2 reasoning and diagnostics."""
        return self._compute_required_rates(self._active_effects.traffic_multiplier)

    def observed_required_rates(self) -> tuple[np.ndarray, np.ndarray]:
        """Actor-consistent demand estimate for V3 analytical reasoning.

        Queue occupancy and each UE's configured mean arrival rate are actor
        inputs. Hidden traffic-event multipliers remain physical-only.
        """
        if not self.v3_enabled:
            return self.current_required_rates()
        return self._compute_required_rates(np.ones(self.num_agents, dtype=np.float64))

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
            if self.v3_enabled:
                values.extend(
                    [
                        float(link.visible),
                        np.clip((self.observed_sinr[ue.ue_id, primary] + 20.0) / 50.0, 0.0, 1.0),
                        self.observed_efficiency[ue.ue_id, primary]
                        / max(1.0, float(self.channel_cfg["spectral_efficiency_cap"])),
                        np.clip((self.observed_elevation[ue.ue_id, primary] + 10.0) / 100.0, 0.0, 1.0),
                        np.clip(self.observed_load[ue.ue_id, primary] / 2.0, 0.0, 1.5),
                    ]
                )
            else:
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
            if self.v3_enabled:
                values.extend(
                    [
                        1.0,
                        float(cell_id == ue.secondary_cell),
                        np.clip((self.observed_sinr[ue.ue_id, cell_id] + 20.0) / 50.0, 0.0, 1.0),
                        self.observed_efficiency[ue.ue_id, cell_id]
                        / max(1.0, float(self.channel_cfg["spectral_efficiency_cap"])),
                        np.clip((self.observed_elevation[ue.ue_id, cell_id] + 10.0) / 100.0, 0.0, 1.0),
                        np.clip(self.observed_load[ue.ue_id, cell_id] / 2.0, 0.0, 1.5),
                        np.clip(
                            (self.observed_sinr_trend[ue.ue_id, cell_id] + 10.0) / 20.0,
                            0.0,
                            1.0,
                        ),
                    ]
                )
            else:
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
