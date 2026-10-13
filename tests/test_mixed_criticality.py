from pathlib import Path

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.egcmadqn import VARIANTS, EGCTrainer, TrainerConfig

CFG = Path(__file__).resolve().parents[1] / "configs"
TWC = [
    CFG / "validation_agentic_challenge_v3.yaml", CFG / "eta_handover_005.yaml", CFG / "validation_dwell_15.yaml",
    CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set1.yaml", CFG / "twc_load_24ue.yaml", CFG / "twc_events.yaml",
]


def _env(mixed: bool, ues: int = 48, rate: float = 0.6, steps: int | None = None) -> LEOMobilityEnv:
    cfg = load_config(CFG / "default.yaml", [*TWC, *([CFG / "twc_mixed_criticality.yaml"] if mixed else [])])
    cfg["environment"]["num_ues"], cfg["traffic"]["mean_arrival_mbps"] = ues, rate
    if steps:
        cfg["environment"]["episode_steps"] = steps
    return LEOMobilityEnv(cfg)


def test_default_scenario_is_unchanged():
    env = _env(False)
    assert not env.traffic_classes and env.obs_dim == 16 + 7 * env.candidate_count
    assert all(not u.critical and u.reward_weight == 1.0 and u.traffic_class == "default" for u in env.ues)
    env.reset(seed=1)
    info = env.step(np.zeros(env.num_agents, dtype=np.int64)).info
    assert not any(k.startswith("critical_") for k in info)


def test_class_assignment_rates_and_observation_flag():
    env = _env(True)
    crit = [u for u in env.ues if u.critical]
    assert len(crit) == 12 and env.obs_dim == 16 + 7 * env.candidate_count + 1
    assert all(u.deadline_s == 1.0 and u.qos_priority == 2.0 and u.reward_weight == 5.0 for u in crit)
    # Mean offered load stays at 48 x 0.6 = 28.8 Mb/s (class rate scales, before the +-35 % spread).
    nominal = sum(0.6 * (0.3333 if u.critical else 1.2222) for u in env.ues)
    assert abs(nominal - 28.8) < 0.01
    obs, _, _ = env.reset(seed=2)
    flags = obs[:, -1]
    assert np.array_equal(flags, np.array([float(u.critical) for u in env.ues]))


def test_class_weighted_deficit_and_per_class_metrics():
    env = _env(True)
    env.reset(seed=3)
    out = env.step(np.zeros(env.num_agents, dtype=np.int64))
    info = out.info
    for k in ("critical_outage_ratio", "noncritical_outage_ratio", "critical_duplication_fraction",
              "critical_deadline_misses", "critical_arrivals"):
        assert k in info and info[k] >= 0.0
    deficits = out.agent_outcomes[:, 2]
    active = out.agent_active > 0.0
    w = np.array([u.reward_weight for u in env.ues])
    expected = (w[active] * deficits[active]).sum() / w[active].sum()
    assert np.isclose(info["service_deficit"], expected)


def test_trainer_constrains_the_critical_outage():
    small = dict(warmup_transitions=50, batch_size=32, updates_per_iteration=2, buffer_capacity=5_000)
    tc = TrainerConfig(rollout_steps=6, outage_metric="critical_outage_ratio")
    trainer = EGCTrainer(_env(True, steps=30), VARIANTS["no_c1"], small, 1, tc)
    m = trainer.run_iteration(99, 100)
    assert 0.0 <= m["greedy_outage"] <= 1.0 and np.isfinite(m["lambda_out"])


def test_duplication_can_be_disabled_for_the_ablation():
    cfg = load_config(CFG / "default.yaml", [*TWC, CFG / "twc_mixed_criticality.yaml", CFG / "twc_no_duplication.yaml"])
    env = LEOMobilityEnv(cfg)
    env.reset(seed=4)
    for _ in range(60):
        masks = env.action_masks()
        assert np.all(masks[:, env.dup_on_action] == 0.0)
        env.step(env.sample_valid_actions())
    assert _env(True).duplication_enabled
