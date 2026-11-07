from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from leo_mappo.runner import _evaluate_policy_impl, evaluation_episode_seeds
from scripts.evaluate_training_budget_checkpoints import (
    CHECKPOINT_ITERATIONS,
    DETERMINISTIC_EPISODES,
    EVALUATION_ACTION_SEED,
    EVALUATION_SEED_BASE,
    STOCHASTIC_EPISODES,
    RunSpec,
    build_evaluation_plan,
    evaluate_task,
    validate_matched_configs,
    validate_run_metadata,
)


def _metadata(coordination_enabled: bool) -> dict:
    return {
        "history_recurrent_mode": "recurrent",
        "world_model_type": "hybrid",
        "coordination_mode": (
            "satellite_price" if coordination_enabled else "none"
        ),
        "training_seed": 7,
        "environment_seed": 7,
        "resolved_config": {
            "mappo": {
                "use_memory": True,
                "use_reasoning": True,
                "reasoning_feature_mode": "confidence_aware",
                "use_coordination": coordination_enabled,
                "use_constraint": False,
                "training_iterations": 500,
                "checkpoint_interval": 100,
            },
            "reasoning": {
                "world_model": "hybrid",
                "planner_objective": "counterfactual",
            },
            "reward": {"charge_rejected_preparation_signaling": False},
        },
    }


def test_plan_is_exact_matched_twenty_row_matrix(tmp_path: Path) -> None:
    plan = build_evaluation_plan(
        (
            RunSpec("M3-H", tmp_path / "m3", False),
            RunSpec("M4-H", tmp_path / "m4", True),
        )
    )

    assert len(plan) == 20
    assert CHECKPOINT_ITERATIONS == (100, 200, 300, 400, 500)
    assert {task.method for task in plan} == {"M3-H", "M4-H"}
    assert {task.evaluation_mode for task in plan} == {
        "deterministic",
        "stochastic",
    }
    assert all(
        task.checkpoint_path.name
        == f"mappo_weights_iter_{task.checkpoint_iteration:04d}.npz"
        for task in plan
    )
    assert all("final" not in task.checkpoint_path.name for task in plan)
    assert DETERMINISTIC_EPISODES == STOCHASTIC_EPISODES == 10
    assert EVALUATION_SEED_BASE == 5001
    assert EVALUATION_ACTION_SEED == 7007
    assert evaluation_episode_seeds(10, EVALUATION_SEED_BASE) == tuple(
        range(5001, 5011)
    )


@pytest.mark.parametrize("coordination_enabled", [False, True])
def test_metadata_validation_reconstructs_m3_and_m4(
    coordination_enabled: bool,
) -> None:
    config = validate_run_metadata(
        _metadata(coordination_enabled),
        coordination_enabled=coordination_enabled,
    )
    assert config["mappo"]["use_coordination"] is coordination_enabled
    assert config["mappo"]["use_constraint"] is False
    assert config["reasoning"]["world_model"] == "hybrid"


def test_metadata_validation_rejects_cross_loading_m3_as_m4() -> None:
    with pytest.raises(ValueError, match="coordination"):
        validate_run_metadata(_metadata(False), coordination_enabled=True)


def test_matched_config_validation_allows_only_coordination_toggle() -> None:
    m3 = _metadata(False)["resolved_config"]
    m4 = _metadata(True)["resolved_config"]
    validate_matched_configs(m3, m4)
    m4["reward"]["charge_rejected_preparation_signaling"] = True
    with pytest.raises(ValueError, match="differ beyond"):
        validate_matched_configs(m3, m4)


def test_checkpoint_evaluator_has_no_training_or_critic_action_path() -> None:
    source = inspect.getsource(evaluate_task)
    assert ".update(" not in source
    assert ".value(" not in source
    assert "evaluate_policy(" in source
    assert "evaluation_action_seed=EVALUATION_ACTION_SEED" in source

    core_source = inspect.getsource(_evaluate_policy_impl)
    assert "agent.update" not in core_source
    assert "agent.value" not in core_source
    assert "reliability_controller.update" not in core_source
    assert core_source.count("reasoner.analyze(") == 1
