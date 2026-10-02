from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .satellite import SatelliteCell, beam_pattern_loss_db, elevation_and_distance, off_axis_angle_rad

SPEED_OF_LIGHT = 299_792_458.0


@dataclass
class LinkResult:
    visible: bool
    elevation_deg: float
    distance_m: float
    sinr_db: float
    spectral_efficiency: float
    packet_success_probability: float


@dataclass
class ChannelModel:
    carrier_hz: float
    ue_gain_dbi: float
    noise_figure_db: float
    implementation_loss_db: float
    atmospheric_loss_db: float
    shadow_std_db: float
    shadow_correlation: float
    interference_dbm: float
    spectral_efficiency_cap: float
    rng: np.random.Generator
    shadow_state: dict[tuple[int, int], float] = field(default_factory=dict)

    def reset(self) -> None:
        self.shadow_state.clear()

    def _shadowing(self, ue_id: int, cell_id: int, extra_loss_db: float) -> float:
        key = (ue_id, cell_id)
        old = self.shadow_state.get(key, self.rng.normal(0.0, self.shadow_std_db))
        innovation_scale = self.shadow_std_db * np.sqrt(max(0.0, 1.0 - self.shadow_correlation**2))
        new = self.shadow_correlation * old + self.rng.normal(0.0, innovation_scale)
        self.shadow_state[key] = new
        return new + extra_loss_db

    def evaluate(
        self,
        ue_id: int,
        ue_position: np.ndarray,
        satellite: SatelliteCell,
        time_s: float,
        min_elevation_rad: float,
        extra_loss_db: float = 0.0,
    ) -> LinkResult:
        satellite_position = satellite.position_ecef(time_s)
        elevation, distance = elevation_and_distance(ue_position, satellite_position)
        visible = elevation >= min_elevation_rad
        wavelength = SPEED_OF_LIGHT / self.carrier_hz
        pattern_loss_db = 0.0
        if satellite.beam_center_ecef is not None and satellite.aperture_radius_m > 0.0:
            pattern_loss_db = beam_pattern_loss_db(
                off_axis_angle_rad(satellite_position, satellite.beam_center_ecef, ue_position),
                satellite.aperture_radius_m,
                wavelength,
            )
        fspl_db = 20.0 * np.log10(max(1.0, 4.0 * np.pi * distance / wavelength))
        elevation_loss = 1.5 / max(0.2, np.sin(max(elevation, 0.01)))
        shadow_db = self._shadowing(ue_id, satellite.cell_id, extra_loss_db)
        rx_dbm = (
            satellite.tx_power_dbm
            + satellite.tx_gain_dbi
            - pattern_loss_db
            + self.ue_gain_dbi
            - fspl_db
            - self.atmospheric_loss_db
            - self.implementation_loss_db
            - elevation_loss
            - shadow_db
        )
        noise_dbm = -174.0 + 10.0 * np.log10(satellite.bandwidth_hz) + self.noise_figure_db
        noise_mw = 10.0 ** (noise_dbm / 10.0)
        interference_mw = 10.0 ** (self.interference_dbm / 10.0)
        sinr_linear = 10.0 ** (rx_dbm / 10.0) / (noise_mw + interference_mw)
        if not visible:
            sinr_linear = 1e-12
        sinr_db = 10.0 * np.log10(max(sinr_linear, 1e-12))
        efficiency = min(self.spectral_efficiency_cap, 0.75 * np.log2(1.0 + sinr_linear))
        # Smooth BLER abstraction: approximately 50% packet success around -2 dB.
        success = float(1.0 / (1.0 + np.exp(-0.75 * (sinr_db + 2.0)))) if visible else 0.0
        return LinkResult(visible, np.degrees(elevation), distance, sinr_db, float(efficiency), success)

