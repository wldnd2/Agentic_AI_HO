from pathlib import Path
import tempfile
import unittest

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import RolloutRunner


ROOT = Path(__file__).resolve().parents[1]


class MAPPOTest(unittest.TestCase):
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
