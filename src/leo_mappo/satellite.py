from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


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

    def position_ecef(self, time_s: float) -> np.ndarray:
        angle = self.phase_rad + self.angular_rate_rad_s * time_s
        radius = EARTH_RADIUS_M + self.altitude_m
        return radius * np.array(
            [
                np.cos(angle),
                np.sin(angle) * np.cos(self.inclination_rad),
                np.sin(angle) * np.sin(self.inclination_rad),
            ],
            dtype=np.float64,
        )

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

