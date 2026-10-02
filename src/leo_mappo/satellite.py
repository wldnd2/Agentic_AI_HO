from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
EARTH_MU_M3_S2 = 3.986004418e14


def circular_orbital_period_s(altitude_m: float) -> float:
    """Kepler period of a circular orbit at the given altitude."""
    radius = EARTH_RADIUS_M + altitude_m
    return float(2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU_M3_S2))


@dataclass
class SatelliteCell:
    cell_id: int
    altitude_m: float
    inclination_rad: float
    phase_rad: float
    angular_rate_rad_s: float
    bandwidth_hz: float
    tx_power_dbm: float
    tx_gain_dbi: float
    base_capacity_scale: float = 1.0
    capacity_multiplier: float = 1.0
    # Right ascension of the ascending node. 0 reproduces the legacy
    # single-plane geometry exactly.
    raan_rad: float = 0.0
    # Per-episode constellation epoch offset (seconds); 0 for the legacy model.
    epoch_offset_s: float = 0.0
    plane_index: int = 0

    def position_ecef(self, time_s: float) -> np.ndarray:
        angle = self.phase_rad + self.angular_rate_rad_s * (time_s + self.epoch_offset_s)
        radius = EARTH_RADIUS_M + self.altitude_m
        x = np.cos(angle)
        y = np.sin(angle) * np.cos(self.inclination_rad)
        z = np.sin(angle) * np.sin(self.inclination_rad)
        if self.raan_rad != 0.0:
            c, s = np.cos(self.raan_rad), np.sin(self.raan_rad)
            x, y = c * x - s * y, s * x + c * y
        return radius * np.array([x, y, z], dtype=np.float64)

    @property
    def effective_capacity_scale(self) -> float:
        return max(0.05, self.base_capacity_scale * self.capacity_multiplier)


def ground_position_ecef(latitude_rad: float, longitude_rad: float) -> np.ndarray:
    clat = np.cos(latitude_rad)
    return EARTH_RADIUS_M * np.array(
        [clat * np.cos(longitude_rad), clat * np.sin(longitude_rad), np.sin(latitude_rad)],
        dtype=np.float64,
    )


def elevation_and_distance(ue_position: np.ndarray, satellite_position: np.ndarray) -> tuple[float, float]:
    delta = satellite_position - ue_position
    distance = float(np.linalg.norm(delta))
    up = ue_position / np.linalg.norm(ue_position)
    elevation = float(np.arcsin(np.clip(np.dot(delta / distance, up), -1.0, 1.0)))
    return elevation, distance


def estimate_remaining_visibility(
    satellite: SatelliteCell,
    ue_position: np.ndarray,
    time_s: float,
    min_elevation_rad: float,
    horizon_s: float = 1200.0,
    resolution_s: float = 2.0,
) -> float:
    elevation, _ = elevation_and_distance(ue_position, satellite.position_ecef(time_s))
    if elevation < min_elevation_rad:
        return 0.0
    elapsed = resolution_s
    while elapsed <= horizon_s:
        future_elevation, _ = elevation_and_distance(
            ue_position, satellite.position_ecef(time_s + elapsed)
        )
        if future_elevation < min_elevation_rad:
            return elapsed
        elapsed += resolution_s
    return horizon_s

