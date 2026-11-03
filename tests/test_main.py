import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
from main import _rolling_mean, _save_training_plots, build_checkpoint_metadata


class MainTest(unittest.TestCase):
    def test_checkpoint_metadata_records_reasoning_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "residual.npz"
            checkpoint.write_bytes(b"frozen residual")
            config = {
                "seed": 3,
                "agent_seed": 7,
                "environment_seed": 11,
                "mappo": {
                    "use_memory": True,
                    "use_reasoning": True,
                    "reasoning_feature_mode": "confidence_aware",
                    "use_coordination": False,
                },
                "reasoning": {
                    "world_model": "hybrid",
                    "residual_model_path": str(checkpoint),
                },
            }
            args = argparse.Namespace(
                config="configs/default.yaml",
                override=["configs/reasoning_aware_mappo.yaml"],
            )

            metadata = build_checkpoint_metadata(config, args)

            self.assertEqual(metadata["world_model_type"], "hybrid")
            self.assertEqual(
                metadata["residual_checkpoint_path"], str(checkpoint.resolve())
            )
            self.assertEqual(
                metadata["residual_checkpoint_sha256"],
                hashlib.sha256(b"frozen residual").hexdigest(),
            )
            self.assertEqual(metadata["reasoning_feature_mode"], "confidence_aware")
            self.assertEqual(metadata["history_recurrent_mode"], "recurrent")
            self.assertEqual(metadata["coordination_mode"], "none")
            self.assertEqual(metadata["training_seed"], 7)

    def test_rolling_mean_uses_trailing_window_with_minimum_one_sample(self) -> None:
        self.assertEqual(_rolling_mean([1.0, 2.0, 3.0, 4.0], window=3), [1.0, 1.5, 2.0, 3.0])

    def test_reward_plot_is_saved_for_history_shorter_than_window(self) -> None:
        history = [
            {"iteration": 1, "team_reward": -2.0},
            {"iteration": 2, "team_reward": -1.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _save_training_plots(history, output_dir)
            self.assertTrue((output_dir / "reward.png").is_file())

    def test_training_writes_periodic_final_and_legacy_checkpoints(self) -> None:
        config = {
            "seed": 1,
            "agent_seed": 1,
            "mappo": {
                "training_iterations": 2,
                "rollout_steps": 1,
                "checkpoint_interval": 1,
            },
        }

        class FakeEnvironment:
            obs_dim = 2
            global_state_dim = 3
            action_dim = 2
            num_cells = 1
            candidate_count = 1
            reward_cfg = {"eta_handover": 0.02}

            def __init__(self, unused_config):
                pass

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def set_training_iteration(self, iteration, total_iterations):
                return 0.001 - iteration * 0.0005, 0.002 - iteration * 0.001

            def update(self, batch):
                return {
                    "actor_loss": 0.0,
                    "critic_loss": 0.0,
                    "entropy": 0.0,
                    "approx_kl": 0.0,
                }

            def save(self, path):
                checkpoint = Path(path)
                if not checkpoint.parent.is_dir():
                    raise AssertionError("checkpoint directory was not created")
                checkpoint.touch()

        class FakeRunner:
            def __init__(self, env, agent):
                pass

            def collect(self, steps):
                return {}, {
                    "team_reward": -1.0,
                    "outage_ratio": 0.2,
                    "deadline_misses": 0.0,
                    "preparation_train_cost": 0.0,
                }

            def update_constraint(self):
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            args = argparse.Namespace(
                config="unused.yaml",
                override=None,
                environment_seed=None,
                agent_seed=None,
                iterations=None,
                output=str(output_dir),
                eval_episodes=1,
                evaluation_action_seed=None,
                stochastic_eval_episodes=0,
            )
            with (
                patch.object(main, "load_config", return_value=config),
                patch.object(main, "LEOMobilityEnv", FakeEnvironment),
                patch.object(main, "MAPPO", FakeAgent),
                patch.object(main, "RolloutRunner", FakeRunner),
                patch.object(main, "evaluate_policy", return_value={"team_reward": -1.0}),
                patch.object(main, "_save_training_plots"),
            ):
                main.train(args)

            self.assertTrue(
                (output_dir / "checkpoints" / "mappo_weights_iter_0001.npz").is_file()
            )
            self.assertTrue(
                (output_dir / "checkpoints" / "mappo_weights_iter_0002.npz").is_file()
            )
            self.assertTrue((output_dir / "mappo_weights_final.npz").is_file())
            self.assertTrue((output_dir / "mappo_weights.npz").is_file())
            with (output_dir / "checkpoint_metadata.json").open(
                encoding="utf-8"
            ) as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["training_seed"], 1)
            with (output_dir / "training_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("actor_lr", rows[0])
            self.assertIn("critic_lr", rows[0])


if __name__ == "__main__":
    unittest.main()
