from pathlib import Path

import numpy as np
import pytest

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.egcmadqn import VARIANTS, DualVariable, EGCMADQN, EGCTrainer, TrainerConfig

CFG = Path(__file__).resolve().parents[1] / "configs"
TWC = [
    CFG / "validation_agentic_challenge_v3.yaml", CFG / "eta_handover_005.yaml", CFG / "validation_dwell_15.yaml",
    CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set1.yaml", CFG / "twc_load_24ue.yaml", CFG / "twc_events.yaml",
]
SMALL = dict(warmup_transitions=50, batch_size=32, updates_per_iteration=2, buffer_capacity=5_000)


def _env(steps=30):
    cfg = load_config(CFG / "default.yaml", TWC)
    cfg["environment"]["episode_steps"] = steps
    return LEOMobilityEnv(cfg)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_trains_and_reports_finite_metrics(variant):
    trainer = EGCTrainer(_env(), VARIANTS[variant], SMALL, 3, TrainerConfig(rollout_steps=8))
    for it in range(3):
        metrics = trainer.run_iteration(it, 3)
    assert all(np.isfinite(v) for v in metrics.values())
    assert ("lambda_out" in metrics) == VARIANTS[variant].dual_constraints


def test_guided_exploration_follows_planner_costs_and_masks():
    agent = EGCMADQN(5, 4, {"buffer_capacity": 10}, 0, constraint_fingerprint=False)
    obs = np.zeros((200, 5))
    masks = np.tile(np.array([1.0, 1.0, 0.0, 1.0]), (200, 1))
    costs = np.tile(np.array([5.0, 0.0, 0.0, 5.0]), (200, 1))  # action 2 cheap but infeasible
    actions = agent.act_guided(obs, masks, epsilon=1.0, guidance=1.0, planner_costs=costs, temperature=0.5)
    assert np.all(masks[np.arange(200), actions] > 0)
    assert np.mean(actions == 1) > 0.99
    uniform = agent.act_guided(obs, masks, epsilon=1.0, guidance=0.0, planner_costs=costs, temperature=0.5)
    assert 0.2 < np.mean(uniform == 1) < 0.5


def test_shaping_is_zero_without_c2_and_potential_ignores_masked_actions():
    trainer = EGCTrainer(_env(), VARIANTS["no_c2"], SMALL, 1, TrainerConfig(rollout_steps=6))
    assert trainer.run_iteration(0, 2)["shaping"] == 0.0
    proposed = EGCTrainer(_env(), VARIANTS["proposed"], SMALL, 1, TrainerConfig(rollout_steps=6))
    costs = np.array([[0.1, 3.0], [2.0, 4.0]])
    masks = np.array([[0.0, 1.0], [1.0, 1.0]])
    proposed.env.num_agents  # noqa: B018
    phi = EGCTrainer._potential(proposed, costs, masks)
    assert np.allclose(phi, [-3.0, -2.0])


def test_dual_variable_ascent_and_projection():
    dual = DualVariable(target=0.2, learning_rate=1.0, maximum=0.5)
    assert dual.update(0.1) == 0.0
    assert np.isclose(dual.update(0.5), 0.3)
    assert dual.update(2.0) == 0.5


def test_checkpoint_roundtrip(tmp_path):
    agent = EGCMADQN(7, 9, {"buffer_capacity": 10}, 0, constraint_fingerprint=True)
    agent.set_multipliers(0.1, 0.2)
    path = tmp_path / "w.npz"
    agent.save(path)
    loaded = EGCMADQN.from_checkpoint(path, seed=1)
    x = np.random.default_rng(0).normal(size=(4, 7))
    m = np.ones((4, 9))
    assert loaded.constraint_fingerprint and np.allclose(loaded.fingerprint, agent.fingerprint)
    assert np.allclose(loaded.q.forward(loaded.augment(x))[0], agent.q.forward(agent.augment(x))[0])
