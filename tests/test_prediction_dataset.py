from __future__ import annotations

from pathlib import Path

import numpy as np

from leo_mappo.prediction_dataset import (
    EpisodePredictionTrajectory,
    PredictionDataset,
    build_prediction_dataset,
)
from leo_mappo.residual_world_model import FEATURE_DIM


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
