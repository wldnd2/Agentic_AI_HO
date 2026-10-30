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
    def test_cho_only_protocol_and_dimensions(self) -> None:
        env = make_env()
        self.assertEqual(
            list(MobilityState),
            [
                MobilityState.SINGLE,
                MobilityState.PREPARED,
                MobilityState.SWITCHING,
                MobilityState.COOLDOWN,
            ],
        )
        self.assertEqual(env.action_dim, env.candidate_count + 3)
        self.assertEqual(env.obs_dim, 14 + 7 * env.candidate_count)

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

    def test_prepare_commit_sequence(self) -> None:
        env = make_env()
        _, _, initial_masks = env.reset(seed=2)
        ue = env.ues[0]
        old_primary = ue.primary_cell
        slot = next(
            slot
            for slot, cell_id in enumerate(env.candidates[0])
            if cell_id != old_primary
            and initial_masks[0, env.prepare_start + slot] > 0.0
        )
        actions = np.zeros(env.num_agents, dtype=np.int64)
        actions[0] = env.prepare_start + slot
        env.step(actions)
        self.assertEqual(ue.state, MobilityState.PREPARED)
        target = ue.secondary_cell
        self.assertIsNotNone(target)
        env.step(np.zeros(env.num_agents, dtype=np.int64))
        mask = env.action_masks()[0]
        self.assertEqual(mask[env.commit_action], 1.0)
        self.assertEqual(mask[env.release_action], 1.0)
        self.assertEqual(np.count_nonzero(mask), 3)
        actions[0] = env.commit_action
        env.step(actions)
        self.assertEqual(ue.state, MobilityState.SWITCHING)

        env.protocol_cfg["handover_base_success"] = 1.0
        env.last_pressure[int(target)] = 0.0
        env.links[0][int(target)].visible = True
        env.links[0][int(target)].packet_success_probability = 1.0
        ue.protocol_timer = 1
        env._advance_protocols()
        self.assertEqual(ue.primary_cell, target)
        self.assertNotEqual(ue.primary_cell, old_primary)
        self.assertEqual(ue.state, MobilityState.COOLDOWN)

    def test_release_clears_prepared_target(self) -> None:
        env = make_env()
        _, _, initial_masks = env.reset(seed=3)
        ue = env.ues[0]
        slot = next(
            slot
            for slot, cell_id in enumerate(env.candidates[0])
            if cell_id != ue.primary_cell
            and initial_masks[0, env.prepare_start + slot] > 0.0
        )
        actions = np.zeros(env.num_agents, dtype=np.int64)
        actions[0] = env.prepare_start + slot
        env.step(actions)
        self.assertEqual(ue.state, MobilityState.PREPARED)
        actions[0] = env.release_action
        env.step(actions)
        self.assertEqual(ue.state, MobilityState.SINGLE)
        self.assertIsNone(ue.secondary_cell)


if __name__ == "__main__":
    unittest.main()
