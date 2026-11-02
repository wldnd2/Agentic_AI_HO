from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from leo_mappo import load_config
from leo_mappo.prediction_dataset import (
    EpisodePredictionTrajectory,
    PredictionDataset,
    build_prediction_dataset,
)
from leo_mappo.residual_world_model import FEATURE_DIM, OBSERVABLE_LINK_FEATURES
from scripts.collect_world_model_data import (
    _metadata_path,
    build_collection_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def _trajectory(features: np.ndarray | None = None) -> EpisodePredictionTrajectory:
    time_count, ue_count, cell_count, horizon_steps = 6, 1, 2, 2
    if features is None:
        features = np.arange(
            time_count * ue_count * cell_count * FEATURE_DIM, dtype=np.float64
        ).reshape(time_count, ue_count, cell_count, FEATURE_DIM)
    true_sinr = np.full((time_count, ue_count, cell_count), 8.0)
    visible = np.ones_like(true_sinr, dtype=bool)
    selected = np.zeros_like(visible)
    selected[:, 0, 0] = True
    geometry = np.full(
        (time_count, ue_count, cell_count, horizon_steps), 5.0, dtype=np.float64
    )
    geometry_visible = np.ones_like(geometry, dtype=bool)
    return EpisodePredictionTrajectory(
        episode_id=4,
        features=features,
        true_sinr_db=true_sinr,
        geometric_visible=visible,
        selected_pairs=selected,
        geometry_sinr_db=geometry,
        geometry_visible=geometry_visible,
    )


def test_future_changes_do_not_modify_earlier_history_tensor() -> None:
    original = _trajectory()
    dataset_a = build_prediction_dataset([original], history_steps=2, horizon_steps=2)
    changed_features = original.features.copy()
    changed_features[2:] += 50_000.0
    dataset_b = build_prediction_dataset(
        [_trajectory(changed_features)], history_steps=2, horizon_steps=2
    )
    earlier_a = dataset_a.histories[dataset_a.time_index == 1]
    earlier_b = dataset_b.histories[dataset_b.time_index == 1]
    assert np.array_equal(earlier_a, earlier_b)


def test_invisible_future_link_has_zero_target_mask() -> None:
    trajectory = _trajectory()
    trajectory.geometric_visible[2, 0, 0] = False
    dataset = build_prediction_dataset([trajectory], history_steps=2, horizon_steps=2)
    first = np.flatnonzero(dataset.time_index == 1)[0]
    assert dataset.target_mask[first, 0] == 0.0
    assert dataset.target_mask[first, 1] == 1.0


def test_dataset_residual_target_and_npz_round_trip(tmp_path: Path) -> None:
    dataset = build_prediction_dataset([_trajectory()], history_steps=2, horizon_steps=2)
    valid = dataset.target_mask > 0.0
    assert np.allclose(
        dataset.target_residual_db[valid],
        dataset.target_sinr_db[valid] - dataset.geometry_sinr_db[valid],
    )
    path = tmp_path / "prediction_data.npz"
    dataset.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "histories",
            "geometry_sinr_db",
            "target_sinr_db",
            "target_residual_db",
            "target_mask",
            "episode_id",
            "time_index",
            "ue_id",
            "cell_id",
        }
    restored = PredictionDataset.load(path)
    for name in (
        "histories",
        "geometry_sinr_db",
        "target_sinr_db",
        "target_residual_db",
        "target_mask",
        "episode_id",
        "time_index",
        "ue_id",
        "cell_id",
    ):
        assert np.array_equal(getattr(dataset, name), getattr(restored, name))


def _collection_args(output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=ROOT / "configs/default.yaml",
        override=[
            ROOT / "configs/validation_agentic_challenge_v3.yaml",
            ROOT / "configs/deadline_aware_demand.yaml",
        ],
        episodes=20,
        seed=1001,
        history_steps=8,
        horizon_steps=6,
        policy="max_sinr",
        residual_model=None,
        output=output,
    )


def test_deadline_aware_collection_metadata_records_provenance(
    tmp_path: Path,
) -> None:
    args = _collection_args(tmp_path / "train.npz")
    config = load_config(args.config, args.override)
    metadata = build_collection_metadata(args, config)
    assert metadata["required_rate_model"] == "deadline_aware"
    assert metadata["base_seed"] == 1001
    assert metadata["episode_count"] == 20
    assert metadata["episode_seeds"] == list(range(1001, 1021))
    assert metadata["seed_range"] == {"first": 1001, "last": 1020}
    assert metadata["history_steps"] == 8
    assert metadata["prediction_horizon_steps"] == 6
    assert metadata["behavior_policy"] == "max_sinr"
    assert metadata["world_model_controlling_environment"] is None
    assert metadata["reasoner_controls_environment"] is False
    assert metadata["feature_order"] == list(OBSERVABLE_LINK_FEATURES)
    assert metadata["prediction_target"] == (
        "true_realized_sinr_db(t+h) - "
        "geometry_predicted_sinr_db(t+h | t)"
    )
    assert metadata["overrides"][-1].endswith(
        "configs/deadline_aware_demand.yaml"
    )
    assert "3GPP-inspired system-level CHO control abstraction" in metadata[
        "control_abstraction"
    ]["description"]
    assert _metadata_path(args.output) == Path(
        f"{args.output}.metadata.json"
    )


def test_legacy_collection_metadata_keeps_arrival_capped_default(
    tmp_path: Path,
) -> None:
    args = _collection_args(tmp_path / "legacy.npz")
    args.override = [ROOT / "configs/validation_agentic_challenge_v3.yaml"]
    config = load_config(args.config, args.override)
    metadata = build_collection_metadata(args, config)
    assert metadata["required_rate_model"] == "arrival_capped"


def test_planner_collection_metadata_identifies_controlling_world_model(
    tmp_path: Path,
) -> None:
    args = _collection_args(tmp_path / "planner.npz")
    config = load_config(args.config, args.override)

    args.policy = "geometry_planner"
    geometry = build_collection_metadata(args, config)
    assert geometry["behavior_policy"] == "geometry_planner"
    assert geometry["world_model_controlling_environment"] == "geometry"
    assert geometry["reasoner_controls_environment"] is True
    assert "residual_model_path" not in geometry

    checkpoint = tmp_path / "model.npz"
    checkpoint.write_bytes(b"frozen residual checkpoint")
    args.policy = "hybrid_planner"
    args.residual_model = checkpoint
    hybrid = build_collection_metadata(args, config)
    assert hybrid["behavior_policy"] == "hybrid_planner"
    assert hybrid["world_model_controlling_environment"] == "hybrid"
    assert hybrid["reasoner_controls_environment"] is True
    assert hybrid["residual_model_path"] == str(checkpoint.resolve())
    assert hybrid["residual_model_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert hybrid["prediction_target"] == geometry["prediction_target"]
