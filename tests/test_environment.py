from pathlib import Path
import unittest

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.ue import MobilityState


ROOT = Path(__file__).resolve().parents[1]


def make_env() -> LEOMobilityEnv:
    config = load_config(ROOT / "configs/default.yaml")
    config["environment"]["episode_steps"] = 20
    return LEOMobilityEnv(config)


class EnvironmentTest(unittest.TestCase):
    def test_shapes_and_valid_random_steps(self) -> None:
        env = make_env()
        observations, global_state, masks = env.reset(seed=1)
        self.assertEqual(observations.shape, (env.num_agents, env.obs_dim))
        self.assertEqual(global_state.shape, (env.global_state_dim,))
        self.assertEqual(masks.shape, (env.num_agents, env.action_dim))
        for _ in range(10):
            output = env.step(env.sample_valid_actions())
            self.assertTrue(np.isfinite(output.observations).all())
            self.assertTrue(np.isfinite(output.rewards).all())
            self.assertEqual(output.info["invalid_actions"], 0)

    def test_prepare_duplicate_commit_sequence(self) -> None:
        env = make_env()
        env.reset(seed=2)
        ue = env.ues[0]
        actions = np.zeros(env.num_agents, dtype=np.int64)
        actions[0] = env.prepare_start
        env.step(actions)
        self.assertEqual(ue.state, MobilityState.PREPARED)
        env.step(np.zeros(env.num_agents, dtype=np.int64))
        actions[0] = env.dup_on_action
        env.step(actions)
        self.assertEqual(ue.state, MobilityState.DUPLICATING)
        actions[0] = env.commit_action
        env.step(actions)
        self.assertEqual(ue.state, MobilityState.SWITCHING)


if __name__ == "__main__":
    unittest.main()
