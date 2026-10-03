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


def test_bessel_j1_reference_values():
    from leo_mappo.satellite import bessel_j1

    assert abs(bessel_j1(1.0) - 0.4400506) < 1e-6
    assert abs(bessel_j1(5.0) + 0.3275791) < 1e-6
    assert abs(bessel_j1(-1.0) + 0.4400506) < 1e-6


def test_beam_pattern_matches_3gpp_beamwidths():
    from leo_mappo.satellite import beam_pattern_loss_db

    wavelength = 299_792_458.0 / 2e9
    for diameter, beamwidth_deg in [(2.0, 4.4127), (1.0, 8.8320)]:
        half = np.radians(beamwidth_deg / 2.0)
        assert abs(beam_pattern_loss_db(half, diameter / 2.0, wavelength) - 3.0) < 0.05
        assert beam_pattern_loss_db(0.0, diameter / 2.0, wavelength) == 0.0


def test_set1_link_budget_and_service_area():
    env = LEOMobilityEnv(_cfg([CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set1.yaml"]))
    cell = env.satellites[0]
    assert np.isclose(cell.bandwidth_hz, 10e6)
    # EIRP density 34 dBW/MHz over 10 MHz = 74 dBm; minus 30 dBi gain.
    assert np.isclose(cell.tx_power_dbm, 44.0)
    assert cell.beam_center_ecef is not None and np.isclose(cell.aperture_radius_m, 1.0)
    centre = np.array([EARTH_RADIUS_M, 0.0, 0.0])
    for ue in env.ues:
        assert np.linalg.norm(ue.position - centre) <= 25.0e3 + 1.0


def test_load_config_sets_population():
    env = LEOMobilityEnv(
        _cfg([CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set1.yaml", CFG / "twc_load_24ue.yaml"])
    )
    assert env.num_agents == 24


TWC = [CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set1.yaml", CFG / "twc_load_24ue.yaml", CFG / "twc_events.yaml"]


def test_ho_interruption_is_off_by_default_and_2rtt_in_twc():
    assert LEOMobilityEnv(_cfg())._ho_interruption_fraction() == 0.0
    env = LEOMobilityEnv(_cfg(TWC))
    assert np.isclose(env._ho_interruption_fraction(), 0.05154)


def test_interruption_reduces_the_delivered_rate_in_the_resolution_epoch():
    env = LEOMobilityEnv(_cfg(TWC))
    env.reset(seed=5)
    stay = np.zeros(env.num_agents, dtype=np.int64)
    env.step(stay)
    ue = env.ues[0]
    # Force UE 0 to resolve a switch in the next epoch.
    from leo_mappo.ue import MobilityState

    target = env.candidates[0][0]
    ue.state, ue.secondary_cell, ue.protocol_timer = MobilityState.SWITCHING, target, 1
    env.step(stay)
    assert np.isclose(env._ho_interruption[0], env._ho_interruption_fraction())
    assert np.all(env._ho_interruption[1:] == 0.0)


def test_signaling_cost_counts_preparation_requests_per_active_ue():
    env = LEOMobilityEnv(_cfg(TWC))
    env.reset(seed=6)
    out = env.step(np.zeros(env.num_agents, dtype=np.int64))
    assert out.info["signaling_cost"] == 0.0
    actions = np.zeros(env.num_agents, dtype=np.int64)
    actions[:] = env.prepare_start  # request preparation towards candidate 1
    masks = env._build_output_state()[2]
    valid = masks[np.arange(env.num_agents), actions] > 0
    out = env.step(np.where(valid, actions, 0))
    assert out.info["signaling_cost"] > 0.0
    expected = out.info["preparation_requested"] / (out.info["active_ue_count"] + 1e-8)
    assert np.isclose(out.info["signaling_cost"], expected)


def test_twc_events_hit_a_serving_satellite():
    env = LEOMobilityEnv(_cfg(TWC))
    env.reset(seed=8)
    hits = 0
    for _ in range(int(env.env_cfg["episode_steps"]) - 1):
        out = env.step(np.zeros(env.num_agents, dtype=np.int64))
        for spec in env.events.realized_specs:
            if spec["type"] == "capacity_drop" and spec.get("_resolved"):
                start, end = spec["start_step"], spec["start_step"] + spec["duration_steps"]
                if start <= env.step_index < end:
                    cell = spec["satellite_id"]
                    assert env.satellites[cell].capacity_multiplier < 1.0
                    hits += 1
    assert hits > 0


def test_compound_capacity_drop_targets_the_busiest_satellite():
    env = LEOMobilityEnv(_cfg([*TWC, CFG / "twc_compound_eval.yaml"]))
    env.reset(seed=9)
    stay = np.zeros(env.num_agents, dtype=np.int64)
    for _ in range(80):
        env.step(stay)
    serving = [u.primary_cell for u in env.ues]
    busiest = max(set(serving), key=serving.count)
    spec = next(s for s in env.events.realized_specs if s["type"] == "capacity_drop")
    assert spec.get("_resolved") and spec["satellite_id"] == busiest
    assert np.isclose(env.satellites[busiest].capacity_multiplier, 0.45)


def test_kept_satellites_do_not_depend_on_episode_length():
    full = LEOMobilityEnv(_cfg(TWC))
    cfg = _cfg(TWC)
    cfg["environment"]["episode_steps"] = 40
    short = LEOMobilityEnv(cfg)
    assert short.num_cells == full.num_cells == 14
