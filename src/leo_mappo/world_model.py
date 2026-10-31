from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from .satellite import elevation_and_distance

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


def _predicted_goodputs(
    service_transmission_rate_bps: float,
    potential_transmission_rate_bps: float,
    delivery_probability: float,
    preserve_reliability_headroom: bool,
) -> tuple[float, float]:
    """Keep served-traffic clipping separate from M5 reliability headroom."""
    service_goodput = service_transmission_rate_bps * delivery_probability
    reliability_transmission_rate = (
        potential_transmission_rate_bps
        if preserve_reliability_headroom
        else service_transmission_rate_bps
    )
    reliability_goodput = reliability_transmission_rate * delivery_probability
    if not math.isfinite(service_goodput) or not math.isfinite(reliability_goodput):
        raise FloatingPointError("predicted goodputs must be finite")
    return float(service_goodput), float(reliability_goodput)


FutureLinkPredictor = Callable[
    ["LEOMobilityEnv", int, int | None, int], tuple[float, float]
]


@dataclass(frozen=True)
class GeometryLinkPrediction:
    """Deterministic future-link prediction from information available now."""

    visible: bool
    elevation_rad: float
    distance_m: float
    sinr_db: float
    spectral_efficiency: float
    success_probability: float


class GeometryWorldModel:
    """Current deterministic geometry, link, service, and reliability model."""

    def __init__(self, epsilon: float, reliability_risk_temperature: float):
        self.epsilon = float(epsilon)
        self.reliability_risk_temperature = float(reliability_risk_temperature)
        self.outage_probability_fn = _predicted_outage_probability
        self.reliability_risk_fn = _planning_reliability_risk
        self.goodputs_fn = _predicted_goodputs

    def future_elevation_and_distance(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int,
        offset: int,
    ) -> tuple[float, float]:
        future_time = env.time_s + offset * env.interval_s
        return elevation_and_distance(
            env.ues[ue_id].position,
            env.satellites[cell_id].position_ecef(future_time),
        )

    def future_link(
        self, env: LEOMobilityEnv, ue_id: int, cell_id: int | None, offset: int
    ) -> tuple[float, float]:
        if cell_id is None:
            return 0.0, 0.0
        prediction = self.predict_link(env, ue_id, cell_id, offset)
        return prediction.spectral_efficiency, prediction.success_probability

    def predict_link(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int,
        offset: int,
    ) -> GeometryLinkPrediction:
        """Expose the existing geometry/link forecast without changing its math."""
        elevation, future_distance = self.future_elevation_and_distance(
            env, ue_id, cell_id, offset
        )
        if elevation < env.min_elevation_rad:
            return GeometryLinkPrediction(
                visible=False,
                elevation_rad=float(elevation),
                distance_m=float(future_distance),
                sinr_db=0.0,
                spectral_efficiency=0.0,
                success_probability=0.0,
            )
        current = env.links[ue_id][cell_id]
        loss_delta = 20.0 * np.log10(
            future_distance / (current.distance_m + self.epsilon)
        )
        current_sinr = (
            env.observed_sinr[ue_id, cell_id] if env.v3_enabled else current.sinr_db
        )
        sinr_db = current_sinr - loss_delta
        sinr_linear = 10.0 ** (np.clip(sinr_db, -120.0, 80.0) / 10.0)
        efficiency = min(
            float(env.channel_cfg["spectral_efficiency_cap"]),
            0.75 * np.log2(1.0 + sinr_linear),
        )
        success = 1.0 / (
            1.0 + np.exp(np.clip(-0.75 * (sinr_db + 2.0), -40.0, 40.0))
        )
        return GeometryLinkPrediction(
            visible=True,
            elevation_rad=float(elevation),
            distance_m=float(future_distance),
            sinr_db=float(sinr_db),
            spectral_efficiency=float(efficiency),
            success_probability=float(success),
        )

    def predicted_leg(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int | None,
        offset: int,
        required_rate: float,
        available: np.ndarray,
        future_link_predictor: FutureLinkPredictor | None = None,
    ) -> tuple[float, float, float, float]:
        if cell_id is None or required_rate <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        predictor = (
            self.future_link
            if future_link_predictor is None
            else future_link_predictor
        )
        efficiency, success = predictor(env, ue_id, cell_id, offset)
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
        capacity_scale = (
            env.satellites[cell_id].base_capacity_scale
            if env.v3_enabled
            else env.satellites[cell_id].effective_capacity_scale
        )
        raw_capacity = (
            env.satellites[cell_id].bandwidth_hz
            * efficiency
            * capacity_scale
        )
        potential_rate = raw_capacity * fraction * load_scale
        service_rate = min(required_rate, potential_rate)
        resource_fraction = service_rate / max(raw_capacity, self.epsilon)
        return (
            float(service_rate),
            float(potential_rate),
            success,
            float(max(0.0, resource_fraction)),
        )

    def predicted_goodputs(
        self,
        service_transmission_rate_bps: float,
        potential_transmission_rate_bps: float,
        delivery_probability: float,
        preserve_reliability_headroom: bool,
    ) -> tuple[float, float]:
        return self.goodputs_fn(
            service_transmission_rate_bps,
            potential_transmission_rate_bps,
            delivery_probability,
            preserve_reliability_headroom,
        )

    def predicted_outage_probability(
        self,
        transmission_rate_bps: float,
        success_probability: float,
        required_rate_bps: float,
        interval_s: float,
        packet_size_bits: int,
        outage_service_ratio: float,
    ) -> float:
        return self.outage_probability_fn(
            transmission_rate_bps,
            success_probability,
            required_rate_bps,
            interval_s,
            packet_size_bits,
            outage_service_ratio,
            self.epsilon,
        )

    def planning_reliability_risk(
        self,
        expected_goodput_bps: float,
        required_rate_bps: float,
        outage_service_ratio: float,
    ) -> float:
        return self.reliability_risk_fn(
            expected_goodput_bps,
            required_rate_bps,
            outage_service_ratio,
            self.reliability_risk_temperature,
            self.epsilon,
        )
