from pathlib import Path
from copy import deepcopy
import tempfile
import unittest

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import RolloutRunner


ROOT = Path(__file__).resolve().parents[1]


class MAPPOTest(unittest.TestCase):
    def test_learning_rates_remain_constant_when_annealing_is_absent_or_disabled(self) -> None:
        for lr_anneal in (None, {"enabled": False}):
            with self.subTest(lr_anneal=lr_anneal):
                config = deepcopy(load_config(ROOT / "configs/default.yaml")["mappo"])
                if lr_anneal is None:
                    config.pop("lr_anneal", None)
                else:
                    config["lr_anneal"] = lr_anneal
                agent = MAPPO(4, 6, 3, config, seed=1)
                initial = agent.learning_rates
                self.assertEqual(
                    initial,
                    (float(config["actor_lr"]), float(config["critic_lr"])),
                )
                self.assertEqual(agent.set_training_iteration(9, 10), initial)

    def test_linear_learning_rate_annealing_reaches_configured_endpoints(self) -> None:
        config = deepcopy(load_config(ROOT / "configs/default.yaml")["mappo"])
        config["lr_anneal"] = {
            "enabled": True,
            "schedule": "linear",
            "actor_initial_lr": 1.0e-3,
            "critic_initial_lr": 2.0e-3,
            "actor_final_lr": 1.0e-4,
            "critic_final_lr": 2.0e-4,
        }
        agent = MAPPO(4, 6, 3, config, seed=1)
        np.testing.assert_allclose(agent.set_training_iteration(0, 3), (1.0e-3, 2.0e-3))
        np.testing.assert_allclose(agent.set_training_iteration(1, 3), (5.5e-4, 1.1e-3))
        np.testing.assert_allclose(agent.set_training_iteration(2, 3), (1.0e-4, 2.0e-4))

    def test_learning_rate_annealing_defaults_to_ten_percent_final_rate(self) -> None:
        config = deepcopy(load_config(ROOT / "configs/default.yaml")["mappo"])
        config["lr_anneal"] = {"enabled": True, "schedule": "linear"}
        agent = MAPPO(4, 6, 3, config, seed=1)
        actor_lr, critic_lr = agent.set_training_iteration(4, 5)
        self.assertAlmostEqual(actor_lr, 0.1 * float(config["actor_lr"]))
        self.assertAlmostEqual(critic_lr, 0.1 * float(config["critic_lr"]))

    def test_numpy_mappo_update_and_save(self) -> None:
        config = load_config(ROOT / "configs/default.yaml")
        config["environment"]["episode_steps"] = 20
        config["mappo"]["rollout_steps"] = 16
        config["mappo"]["update_epochs"] = 1
        env = LEOMobilityEnv(config)
        agent = MAPPO(env.obs_dim, env.global_state_dim, env.action_dim, config["mappo"], seed=3)
        runner = RolloutRunner(env, agent)
        batch, _ = runner.collect(16)
        before = agent.actor.parameters["w1"].copy()
        losses = agent.update(batch)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        self.assertFalse(np.array_equal(before, agent.actor.parameters["w1"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            agent.save(path)
            agent.load(path)


if __name__ == "__main__":
    unittest.main()
