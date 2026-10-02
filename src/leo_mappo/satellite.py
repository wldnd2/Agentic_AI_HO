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
    # Earth-fixed beam (TR 38.811 Sec. 6.4.1 circular-aperture pattern). When
    # beam_center_ecef is None the legacy constant-gain model is used.
    beam_center_ecef: np.ndarray | None = None
    aperture_radius_m: float = 0.0

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



def bessel_j1(x: float) -> float:
    """First-kind Bessel function J1 (Abramowitz & Stegun 9.4.4 / 9.4.6)."""
    ax = abs(x)
    if ax <= 3.0:
        y = (x / 3.0) ** 2
        value = x * (
            0.5 - 0.56249985 * y + 0.21093573 * y**2 - 0.03954289 * y**3
            + 0.00443319 * y**4 - 0.00031761 * y**5 + 0.00001109 * y**6
        )
        return float(value)
    y = 3.0 / ax
    f1 = (0.79788456 + 0.00000156 * y + 0.01659667 * y**2 + 0.00017105 * y**3
          - 0.00249511 * y**4 + 0.00113653 * y**5 - 0.00020033 * y**6)
    t1 = (ax - 2.35619449 + 0.12499612 * y + 0.00005650 * y**2 - 0.00637879 * y**3
          + 0.00074348 * y**4 + 0.00079824 * y**5 - 0.00029166 * y**6)
    value = f1 * np.cos(t1) / np.sqrt(ax)
    return float(value if x >= 0.0 else -value)


def beam_pattern_loss_db(off_axis_rad: float, aperture_radius_m: float, wavelength_m: float) -> float:
    """Gain reduction relative to boresight for a circular aperture:
    G(theta)/G_max = 4 |J1(ka sin theta) / (ka sin theta)|^2."""
    u = 2.0 * np.pi / wavelength_m * aperture_radius_m * np.sin(abs(off_axis_rad))
    if u < 1e-9:
        return 0.0
    relative = 4.0 * (bessel_j1(u) / u) ** 2
    return float(-10.0 * np.log10(max(relative, 1e-12)))


def off_axis_angle_rad(satellite_position: np.ndarray, beam_center: np.ndarray, ue_position: np.ndarray) -> float:
    to_center = beam_center - satellite_position
    to_ue = ue_position - satellite_position
    cosine = np.dot(to_center, to_ue) / (np.linalg.norm(to_center) * np.linalg.norm(to_ue))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))
