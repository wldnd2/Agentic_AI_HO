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


def test_mappo_variants_train_and_reload(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    for variant in ("mappo", "mappo_c234"):
        out = tmp_path / variant
        subprocess.run(
            [sys.executable, str(root / "train_twc.py"), "--variant", variant, "--iterations", "2", "--smoke",
             "--outdir", str(out)],
            check=True, cwd=root, env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
        )
        sys.path.insert(0, str(root))
        from eval_twc import run_job

        row = run_job((variant, str(out), "nominal", 20, 1, False))
        assert 0.0 <= row["outage"] <= 1.0


def test_relabeling_uses_current_multipliers():
    from leo_mappo.egcmadqn import RelabelingReplayBuffer

    lam = {"out": 0.0, "sig": 0.0}
    buf = RelabelingReplayBuffer(10, 3, 4, np.random.default_rng(0),
                                 lambda: (lam["out"], lam["sig"], 0.2, 0.16))
    z = np.zeros((2, 3))
    buf.add_components(z, np.array([0, 1]), np.full(2, -0.1), np.array([0.01, -0.01]),
                       np.full(2, 0.4), np.full(2, 0.36), z, np.ones((2, 4)), np.zeros(2))
    idx = np.array([0, 1])
    assert np.allclose(buf.relabeled(idx), [-0.09, -0.11])
    lam.update(out=1.0, sig=2.0)  # multipliers grow after the transitions were stored
    assert np.allclose(buf.relabeled(idx), [-0.09 - 0.2 - 0.4, -0.11 - 0.2 - 0.4])


def test_trainer_relabels_and_estimates_constraints_on_greedy_agents():
    trainer = EGCTrainer(_env(), VARIANTS["proposed"], SMALL, 2, TrainerConfig(rollout_steps=8))
    from leo_mappo.egcmadqn import RelabelingReplayBuffer

    assert isinstance(trainer.agent.buffer, RelabelingReplayBuffer)
    m = trainer.run_iteration(0, 100)  # epsilon = 1: nobody is greedy -> fall back to behaviour averages
    assert m["greedy_signaling"] == m["signaling"]
    m = trainer.run_iteration(99, 100)  # epsilon = 0.05: mostly greedy agents
    assert 0.0 <= m["greedy_outage"] <= 1.0 and 0.0 <= m["greedy_signaling"] <= 1.0


def test_v1_behaviour_is_still_available():
    tc = TrainerConfig(rollout_steps=6, relabel_rewards=False, greedy_constraint_estimates=False, outage_target=0.15)
    trainer = EGCTrainer(_env(), VARIANTS["proposed"], SMALL, 4, tc)
    from leo_mappo.egcmadqn import RelabelingReplayBuffer

    assert not isinstance(trainer.agent.buffer, RelabelingReplayBuffer)
    assert np.isfinite(trainer.run_iteration(0, 2)["lambda_out"])


def test_fixed_penalty_keeps_multipliers_constant():
    tc = TrainerConfig(rollout_steps=6, fixed_lambda_out=0.6, fixed_lambda_sig=1.0)
    trainer = EGCTrainer(_env(), VARIANTS["c3_only"], SMALL, 5, tc)
    for it in range(3):
        m = trainer.run_iteration(it, 3)
    assert m["lambda_out"] == 0.6 and m["lambda_sig"] == 1.0


def test_hysteretic_update_only_damps_value_decreases():
    from leo_mappo.madqn import MADQN

    cfg = dict(SMALL, warmup_transitions=4, batch_size=4, updates_per_iteration=1, target_update_interval=10**6)
    base = MADQN(3, 2, cfg, 0)
    hyst = MADQN(3, 2, {**cfg, "hysteretic_beta": 0.0}, 0)
    obs = np.ones((4, 3))
    for agent, reward in ((base, -50.0), (hyst, -50.0)):  # very negative reward -> Q must decrease
        agent.store(obs, np.zeros(4, dtype=np.int64), reward, obs, np.ones((4, 2)), True)
    q = lambda agent: agent.q.forward(agent.augment(obs))[0][0, 0]
    before_h, before_b = q(hyst), q(base)
    base.update(); hyst.update()
    assert np.isclose(q(hyst), before_h)        # beta = 0: a value decrease is fully damped
    assert q(base) < before_b                   # the standard update decreases the value
