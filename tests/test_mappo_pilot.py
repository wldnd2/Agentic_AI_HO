from __future__ import annotations

import argparse
import inspect
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import main
from leo_mappo import load_config
from leo_mappo.evaluation_diagnostics import (
    PolicyEvaluationDiagnostics,
    analyze_preparation_contention,
)
from leo_mappo.runner import (
    RolloutRunner,
    _evaluate_policy_impl,
    evaluation_episode_seeds,
)
from scripts.compare_mappo_pilots import METRICS, compare_pilot_outputs


ROOT = Path(__file__).resolve().parents[1]


def _changed_paths(before: dict, after: dict, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key in before.keys() | after.keys():
        path = f"{prefix}.{key}" if prefix else key
        left = before.get(key)
        right = after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            paths.update(_changed_paths(left, right, path))
        elif left != right:
            paths.add(path)
    return paths


def _pilot_config(*capabilities: str) -> dict:
    return load_config(
        ROOT / "configs/default.yaml",
        [
            ROOT / "configs/validation_agentic_challenge_v3.yaml",
            ROOT / "configs/deadline_aware_demand.yaml",
            ROOT / "configs/recurrent_mappo.yaml",
            *[ROOT / path for path in capabilities],
            ROOT / "configs/pilot_mappo.yaml",
        ],
    )


def test_pilot_override_changes_only_budget_and_artifact_cadence() -> None:
    base = load_config(ROOT / "configs/default.yaml")
    pilot = load_config(
        ROOT / "configs/default.yaml", ROOT / "configs/pilot_mappo.yaml"
    )

    assert _changed_paths(base, pilot) == {
        "mappo.training_iterations",
        "mappo.checkpoint_interval",
        "evaluation",
    }
    assert pilot["mappo"]["training_iterations"] == 100
    assert pilot["mappo"]["checkpoint_interval"] == 25
    assert pilot["evaluation"]["episodes"] == 10


def test_three_pilot_variants_have_matched_interactions_and_no_coordination() -> None:
    variants = {
        "M2": _pilot_config(),
        "M3-G": _pilot_config(
            "configs/counterfactual_planner.yaml",
            "configs/reasoning_aware_mappo.yaml",
        ),
        "M3-H": _pilot_config(
            "configs/counterfactual_planner.yaml",
            "configs/reasoning_aware_mappo.yaml",
            "configs/hybrid_world_model.yaml",
        ),
    }

    assert {config["environment"]["episode_steps"] for config in variants.values()} == {
        420
    }
    assert {
        config["mappo"]["training_iterations"]
        * config["mappo"]["rollout_steps"]
        for config in variants.values()
    } == {12_800}
    optimizer_fields = (
        "rollout_steps",
        "bptt_steps",
        "update_epochs",
        "minibatch_size",
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "actor_lr",
        "critic_lr",
        "lr_anneal",
    )
    assert len(
        {
            json.dumps(
                {field: config["mappo"][field] for field in optimizer_fields},
                sort_keys=True,
            )
            for config in variants.values()
        }
    ) == 1
    assert {config["seed"] for config in variants.values()} == {7}
    assert all(
        not config["mappo"]["use_coordination"]
        and not config["mappo"]["use_constraint"]
        for config in variants.values()
    )
    assert variants["M2"]["mappo"]["use_reasoning"] is False
    assert variants["M3-G"]["reasoning"].get("world_model", "geometry") == "geometry"
    assert variants["M3-H"]["reasoning"]["world_model"] == "hybrid"
    assert variants["M3-H"]["reasoning"]["residual_model_path"] == (
        "outputs/world_model_deadline_aware/models/model_best/model.npz"
    )


def test_metadata_records_matched_evaluation_seeds_for_all_variants(
    tmp_path: Path,
) -> None:
    residual = tmp_path / "model.npz"
    residual.write_bytes(b"frozen residual checkpoint")
    configs = {
        "M2": _pilot_config(),
        "M3-G": _pilot_config(
            "configs/counterfactual_planner.yaml",
            "configs/reasoning_aware_mappo.yaml",
        ),
        "M3-H": _pilot_config(
            "configs/counterfactual_planner.yaml",
            "configs/reasoning_aware_mappo.yaml",
            "configs/hybrid_world_model.yaml",
        ),
    }
    configs["M3-H"] = deepcopy(configs["M3-H"])
    configs["M3-H"]["reasoning"]["residual_model_path"] = str(residual)
    args = argparse.Namespace(
        config="configs/default.yaml",
        override=[],
        eval_episodes=None,
        evaluation_seed_base=5001,
        evaluation_action_seed=7007,
    )

    metadata = {
        label: main.build_checkpoint_metadata(config, args)
        for label, config in configs.items()
    }

    assert {tuple(item["evaluation_episode_seeds"]) for item in metadata.values()} == {
        evaluation_episode_seeds(10, 5001)
    }
    assert {item["total_training_interactions"] for item in metadata.values()} == {
        12_800
    }
    assert metadata["M3-H"]["residual_checkpoint_sha256"] is not None


def test_contention_diagnostics_are_evaluation_only_and_do_not_change_actions() -> None:
    assert "analyze_preparation_contention" not in inspect.getsource(
        RolloutRunner.collect
    )
    assert "PolicyEvaluationDiagnostics" not in inspect.getsource(
        RolloutRunner.collect
    )
    assert "analyze_preparation_contention" in inspect.getsource(
        _evaluate_policy_impl
    )
    actions = np.asarray([1, 1, 0], dtype=np.int64)
    before = actions.copy()
    ues = [
        SimpleNamespace(queue_packets=10, head_of_line_delay_s=0.5),
        SimpleNamespace(queue_packets=20, head_of_line_delay_s=1.0),
        SimpleNamespace(queue_packets=0, head_of_line_delay_s=0.0),
    ]
    env = SimpleNamespace(
        num_cells=2,
        prepare_start=1,
        candidate_count=1,
        commit_action=2,
        release_action=3,
        candidates=[[0], [0], [1]],
        last_preparation_available_slots=np.asarray([1, -1]),
        ues=ues,
    )
    contention = analyze_preparation_contention(env, actions)
    diagnostics = PolicyEvaluationDiagnostics(interval_s=1.0, packet_size_bits=12_000)
    output = SimpleNamespace(
        info={
            "arrivals": 10.0,
            "delivered_packets": 7.0,
            "deadline_misses": 1.0,
            "overflow_packets": 1.0,
            "preparation_requested": 2.0,
            "preparation_admitted": 1.0,
            "preparation_rejected": 1.0,
        }
    )

    diagnostics.record_step(env, actions, output, contention)
    metrics = diagnostics.metrics()

    np.testing.assert_array_equal(actions, before)
    assert metrics["fraction_prep_requests_in_same_target_collision"] == 1.0
    assert metrics["fraction_prep_requests_to_oversubscribed_targets"] == 1.0
    assert metrics["total_excess_prep_requests"] == 1.0
    assert metrics["packet_delivery_ratio"] == 0.7


def _write_pilot_artifacts(
    directory: Path,
    label: str,
    seeds: list[int],
) -> None:
    world_model = {"M2": None, "M3-G": "geometry", "M3-H": "hybrid"}[label]
    metadata = {
        "training_seed": 7,
        "environment_seed": 7,
        "total_training_interactions": 12_800,
        "evaluation_episode_seeds": seeds,
        "history_recurrent_mode": "recurrent",
        "coordination_mode": "none",
        "reasoning_feature_mode": "none" if label == "M2" else "confidence_aware",
        "world_model_type": world_model,
        "residual_checkpoint_sha256": "abc" if label == "M3-H" else None,
    }
    evaluation = {metric: 0.0 for metric in METRICS}
    directory.mkdir()
    (directory / "checkpoint_metadata.json").write_text(json.dumps(metadata))
    (directory / "evaluation.json").write_text(json.dumps(evaluation))


def test_comparison_requires_matched_evaluation_seeds(tmp_path: Path) -> None:
    directories = {label: tmp_path / label for label in ("M2", "M3-G", "M3-H")}
    for label, directory in directories.items():
        _write_pilot_artifacts(directory, label, [5001, 5002])
    rows = compare_pilot_outputs(directories)
    assert [row["variant"] for row in rows] == ["M2", "M3-G", "M3-H"]

    metadata_path = directories["M3-H"] / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["evaluation_episode_seeds"] = [6001, 6002]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="evaluation_episode_seeds"):
        compare_pilot_outputs(directories)
