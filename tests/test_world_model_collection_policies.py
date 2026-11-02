from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.predictive_control import (
    predictive_controller_step,
    select_greedy_feasible_actions,
)
from leo_mappo.residual_world_model import ResidualLinkPredictor
from leo_mappo.world_model import GeometryWorldModel
import scripts.collect_world_model_data as collector
import scripts.evaluate_geometry_planner_closed_loop as evaluator


def _counterfactual_config() -> dict:
    return {
        "reasoning": {"planner_objective": "counterfactual"},
        "coordination": {},
    }


@pytest.mark.parametrize(
    "policy", ("max_sinr", "geometry_planner", "hybrid_planner")
)
def test_collection_cli_accepts_diagnostic_policy_modes(policy: str) -> None:
    args = collector.parse_args(
        ["--policy", policy, "--output", "unused.npz"]
    )
    assert args.policy == policy


def test_max_sinr_policy_dispatch_remains_unchanged() -> None:
    expected = np.asarray([2, 0], dtype=np.int64)
    assert (
        collector.build_behavior_reasoner(
            _counterfactual_config(),
            SimpleNamespace(reward_cfg={}),  # type: ignore[arg-type]
            "max_sinr",
        )
        is None
    )

    class TinyEnvironment:
        received_actions: np.ndarray | None = None

        def step(self, actions: np.ndarray) -> SimpleNamespace:
            self.received_actions = np.asarray(actions).copy()
            return SimpleNamespace(done=False)

    env = TinyEnvironment()
    with patch.object(
        collector,
        "_observable_max_sinr_actions",
        return_value=expected,
    ) as max_sinr_actions:
        actions = collector._step_behavior_policy(  # type: ignore[arg-type]
            env,
            "max_sinr",
            np.random.default_rng(7),
            reasoner=None,
        )
    max_sinr_actions.assert_called_once_with(env)
    np.testing.assert_array_equal(actions, expected)
    np.testing.assert_array_equal(env.received_actions, expected)


def test_geometry_planner_uses_geometry_without_loading_residual() -> None:
    env = SimpleNamespace(reward_cfg={})
    with patch.object(
        HybridWorldModel,
        "from_checkpoint",
        side_effect=AssertionError("residual checkpoint was loaded"),
    ) as loader:
        reasoner = collector.build_behavior_reasoner(
            _counterfactual_config(),
            env,  # type: ignore[arg-type]
            "geometry_planner",
        )
    loader.assert_not_called()
    assert reasoner is not None
    assert type(reasoner.world_model) is GeometryWorldModel


def test_hybrid_planner_requires_residual_checkpoint() -> None:
    with pytest.raises(ValueError, match="--residual-model"):
        collector.build_behavior_reasoner(
            _counterfactual_config(),
            SimpleNamespace(reward_cfg={}),  # type: ignore[arg-type]
            "hybrid_planner",
        )


def test_hybrid_planner_loads_existing_hybrid_world_model(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.npz"
    ResidualLinkPredictor(
        history_steps=2,
        horizon_steps=3,
        hidden_dim=4,
        mlp_hidden_dim=4,
        seed=23,
    ).save(checkpoint)
    reasoner = collector.build_behavior_reasoner(
        _counterfactual_config(),
        SimpleNamespace(reward_cfg={}),  # type: ignore[arg-type]
        "hybrid_planner",
        checkpoint,
    )
    assert reasoner is not None
    assert isinstance(reasoner.world_model, HybridWorldModel)
    assert reasoner.planner.world_model is reasoner.world_model


def test_hybrid_planner_load_failure_has_no_geometry_fallback(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.npz"
    checkpoint.touch()
    with patch.object(
        HybridWorldModel,
        "from_checkpoint",
        side_effect=RuntimeError("checkpoint load failed"),
    ):
        with pytest.raises(RuntimeError, match="checkpoint load failed"):
            collector.build_behavior_reasoner(
                _counterfactual_config(),
                SimpleNamespace(reward_cfg={}),  # type: ignore[arg-type]
                "hybrid_planner",
                checkpoint,
            )


def test_collector_and_evaluator_share_argmin_and_direct_step() -> None:
    assert collector.predictive_controller_step is predictive_controller_step
    assert evaluator.predictive_controller_step is predictive_controller_step
    assert evaluator.select_greedy_feasible_actions is (
        select_greedy_feasible_actions
    )

    class TinyEnvironment:
        received_actions: np.ndarray | None = None

        def action_masks(self) -> np.ndarray:
            return np.asarray([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])

        def step(self, actions: np.ndarray) -> SimpleNamespace:
            self.received_actions = np.asarray(actions).copy()
            return SimpleNamespace(done=True)

    class TinyReasoner:
        def analyze(
            self, env: TinyEnvironment, masks: np.ndarray
        ) -> SimpleNamespace:
            assert np.array_equal(masks, env.action_masks())
            return SimpleNamespace(
                raw_costs=np.asarray([[0.4, 0.1, 0.1], [0.2, -9.0, 0.1]])
            )

    env = TinyEnvironment()
    actions = collector._step_behavior_policy(  # type: ignore[arg-type]
        env,
        "geometry_planner",
        np.random.default_rng(11),
        TinyReasoner(),  # type: ignore[arg-type]
    )
    assert actions.tolist() == [1, 2]
    np.testing.assert_array_equal(env.received_actions, actions)
