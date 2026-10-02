from pathlib import Path

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.satellite import (
    EARTH_RADIUS_M,
    SatelliteCell,
    circular_orbital_period_s,
    elevation_and_distance,
)

CFG = Path(__file__).resolve().parents[1] / "configs"
V3 = [
    CFG / "validation_agentic_challenge_v3.yaml",
    CFG / "eta_handover_005.yaml",
    CFG / "validation_dwell_15.yaml",
]


def _cfg(extra=()):
    return load_config(CFG / "default.yaml", [*V3, *extra])


def test_zero_raan_reproduces_legacy_geometry():
    cell = SatelliteCell(0, 600e3, np.radians(53.0), 0.3, 1e-3, 20e6, 46.0, 30.0)
    t = 123.0
    angle = 0.3 + 1e-3 * t
    r = EARTH_RADIUS_M + 600e3
    legacy = r * np.array(
        [np.cos(angle), np.sin(angle) * np.cos(np.radians(53.0)), np.sin(angle) * np.sin(np.radians(53.0))]
    )
    assert np.allclose(cell.position_ecef(t), legacy)


def test_raan_rotates_about_polar_axis():
    a = SatelliteCell(0, 550e3, np.radians(53.0), 0.4, 1e-3, 20e6, 46.0, 30.0)
    b = SatelliteCell(0, 550e3, np.radians(53.0), 0.4, 1e-3, 20e6, 46.0, 30.0, raan_rad=np.radians(5.0))
    pa, pb = a.position_ecef(50.0), b.position_ecef(50.0)
    assert np.isclose(pa[2], pb[2])
    assert np.isclose(np.linalg.norm(pa), np.linalg.norm(pb))
    assert np.isclose(np.arctan2(pb[1], pb[0]) - np.arctan2(pa[1], pa[0]), np.radians(5.0))


def test_kepler_period_at_550_km():
    # 2*pi*sqrt((6371 km + 550 km)^3 / mu) = 5730 s (mean Earth radius)
    assert abs(circular_orbital_period_s(550e3) - 5730.0) < 2.0


def test_legacy_config_keeps_six_satellites():
    env = LEOMobilityEnv(_cfg())
    assert env.num_cells == 6
    assert all(cell.raan_rad == 0.0 and cell.epoch_offset_s == 0.0 for cell in env.satellites)


def test_walker_slice_is_visibility_filtered_and_consistent():
    env = LEOMobilityEnv(_cfg([CFG / "twc_walker_p3.yaml"]))
    assert 0 < env.num_cells < 66
    assert len(env.satellites) == env.num_cells
    assert env.global_state_dim == env.num_agents * env.obs_dim + 2 * env.num_cells
    assert {cell.plane_index for cell in env.satellites} == {0, 1, 2}
    env.reset(seed=1)
    obs, _, masks = env._build_output_state()
    assert obs.shape == (env.num_agents, env.obs_dim)
    assert masks.shape == (env.num_agents, env.action_dim)


def test_every_kept_satellite_can_become_visible():
    env = LEOMobilityEnv(_cfg([CFG / "twc_walker_p2.yaml"]))
    horizon = int(env.env_cfg["episode_steps"]) + int(env.max_epoch_offset_s)
    threshold = env.min_elevation_rad - np.radians(5.0)
    centre = env.ues[0].position * 0.0 + np.array([EARTH_RADIUS_M, 0.0, 0.0])
    for cell in env.satellites:
        best = max(elevation_and_distance(centre, cell.position_ecef(t))[0] for t in range(0, horizon, 10))
        assert best >= threshold - np.radians(10.0)


def test_epoch_offset_is_random_but_seed_reproducible():
    env = LEOMobilityEnv(_cfg([CFG / "twc_walker_p2.yaml"]))
    env.reset(seed=3)
    first = env.satellites[0].epoch_offset_s
    env.reset(seed=4)
    second = env.satellites[0].epoch_offset_s
    env.reset(seed=3)
    assert env.satellites[0].epoch_offset_s == first
    assert first != second
    assert 0.0 <= first <= env.max_epoch_offset_s
