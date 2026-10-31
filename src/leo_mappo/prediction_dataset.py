from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class PredictionDataset:
    histories: np.ndarray
    geometry_sinr_db: np.ndarray
    target_sinr_db: np.ndarray
    target_residual_db: np.ndarray
    target_mask: np.ndarray
    episode_id: np.ndarray
    time_index: np.ndarray
    ue_id: np.ndarray
    cell_id: np.ndarray

    def __post_init__(self) -> None:
        self.histories = np.asarray(self.histories, dtype=np.float64)
        self.geometry_sinr_db = np.asarray(self.geometry_sinr_db, dtype=np.float64)
        self.target_sinr_db = np.asarray(self.target_sinr_db, dtype=np.float64)
        self.target_residual_db = np.asarray(self.target_residual_db, dtype=np.float64)
        self.target_mask = np.asarray(self.target_mask, dtype=np.float64)
        self.episode_id = np.asarray(self.episode_id, dtype=np.int64)
        self.time_index = np.asarray(self.time_index, dtype=np.int64)
        self.ue_id = np.asarray(self.ue_id, dtype=np.int64)
        self.cell_id = np.asarray(self.cell_id, dtype=np.int64)
        self.validate()

    @property
    def sample_count(self) -> int:
        return int(self.histories.shape[0])

    @property
    def history_steps(self) -> int:
        return int(self.histories.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.histories.shape[2])

    @property
    def horizon_steps(self) -> int:
        return int(self.geometry_sinr_db.shape[1])

    def validate(self) -> None:
        if self.histories.ndim != 3:
            raise ValueError("histories must have shape [samples, history, features]")
        sample_count = self.histories.shape[0]
        if self.histories.shape[1] <= 0 or self.histories.shape[2] <= 0:
            raise ValueError("history and feature dimensions must be positive")
        if self.geometry_sinr_db.ndim != 2 or self.geometry_sinr_db.shape[0] != sample_count:
            raise ValueError("geometry_sinr_db must have shape [samples, horizon]")
        expected_targets = self.geometry_sinr_db.shape
        for name, values in (
            ("target_sinr_db", self.target_sinr_db),
            ("target_residual_db", self.target_residual_db),
            ("target_mask", self.target_mask),
        ):
            if values.shape != expected_targets:
                raise ValueError(f"{name} must have shape {expected_targets}")
        if expected_targets[1] <= 0:
            raise ValueError("horizon dimension must be positive")
        for name, values in (
            ("histories", self.histories),
            ("geometry_sinr_db", self.geometry_sinr_db),
            ("target_sinr_db", self.target_sinr_db),
            ("target_residual_db", self.target_residual_db),
            ("target_mask", self.target_mask),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        if np.any((self.target_mask < 0.0) | (self.target_mask > 1.0)):
            raise ValueError("target_mask must lie in [0, 1]")
        for name, values in (
            ("episode_id", self.episode_id),
            ("time_index", self.time_index),
            ("ue_id", self.ue_id),
            ("cell_id", self.cell_id),
        ):
            if values.shape != (sample_count,):
                raise ValueError(f"{name} must have shape ({sample_count},)")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            histories=self.histories,
            geometry_sinr_db=self.geometry_sinr_db,
            target_sinr_db=self.target_sinr_db,
            target_residual_db=self.target_residual_db,
            target_mask=self.target_mask,
            episode_id=self.episode_id,
            time_index=self.time_index,
            ue_id=self.ue_id,
            cell_id=self.cell_id,
        )

    @classmethod
    def load(cls, path: str | Path) -> PredictionDataset:
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                histories=data["histories"],
                geometry_sinr_db=data["geometry_sinr_db"],
                target_sinr_db=data["target_sinr_db"],
                target_residual_db=data["target_residual_db"],
                target_mask=data["target_mask"],
                episode_id=data["episode_id"],
                time_index=data["time_index"],
                ue_id=data["ue_id"],
                cell_id=data["cell_id"],
            )


@dataclass(frozen=True)
class EpisodePredictionTrajectory:
    """One complete rollout captured before supervised samples are built."""

    episode_id: int
    features: np.ndarray
    true_sinr_db: np.ndarray
    geometric_visible: np.ndarray
    selected_pairs: np.ndarray
    geometry_sinr_db: np.ndarray
    geometry_visible: np.ndarray

    def validate(self, horizon_steps: int) -> None:
        features = np.asarray(self.features)
        if features.ndim != 4:
            raise ValueError("trajectory features must have shape [time, UE, cell, feature]")
        prefix = features.shape[:3]
        if any(size <= 0 for size in features.shape):
            raise ValueError("trajectory dimensions must be positive")
        for name, values in (
            ("true_sinr_db", self.true_sinr_db),
            ("geometric_visible", self.geometric_visible),
            ("selected_pairs", self.selected_pairs),
        ):
            if np.asarray(values).shape != prefix:
                raise ValueError(f"trajectory {name} must have shape {prefix}")
        future_shape = (*prefix, horizon_steps)
        if np.asarray(self.geometry_sinr_db).shape != future_shape:
            raise ValueError(f"trajectory geometry_sinr_db must have shape {future_shape}")
        if np.asarray(self.geometry_visible).shape != future_shape:
            raise ValueError(f"trajectory geometry_visible must have shape {future_shape}")
        if not np.all(np.isfinite(features)):
            raise ValueError("trajectory observable features must be finite")


def build_prediction_dataset(
    trajectories: Iterable[EpisodePredictionTrajectory],
    history_steps: int,
    horizon_steps: int,
) -> PredictionDataset:
    """Construct samples only from already-complete episode trajectories."""
    if history_steps <= 0 or horizon_steps <= 0:
        raise ValueError("history_steps and horizon_steps must be positive")
    histories: list[np.ndarray] = []
    geometry_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    residual_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    episode_ids: list[int] = []
    time_indices: list[int] = []
    ue_ids: list[int] = []
    cell_ids: list[int] = []

    for trajectory in trajectories:
        trajectory.validate(horizon_steps)
        features = np.asarray(trajectory.features, dtype=np.float64)
        true_sinr = np.asarray(trajectory.true_sinr_db, dtype=np.float64)
        visible = np.asarray(trajectory.geometric_visible, dtype=bool)
        selected = np.asarray(trajectory.selected_pairs, dtype=bool)
        predicted_geometry = np.asarray(trajectory.geometry_sinr_db, dtype=np.float64)
        predicted_visible = np.asarray(trajectory.geometry_visible, dtype=bool)
        time_count, ue_count, cell_count, _ = features.shape
        final_prediction_time = time_count - horizon_steps - 1
        for time_index in range(history_steps - 1, final_prediction_time + 1):
            for ue_id, cell_id in np.argwhere(selected[time_index]):
                geometry = predicted_geometry[time_index, ue_id, cell_id].copy()
                # Zeroes are finite storage placeholders for masked horizons,
                # never synthetic low-SINR labels.
                targets = np.zeros(horizon_steps, dtype=np.float64)
                residuals = np.zeros(horizon_steps, dtype=np.float64)
                masks = np.zeros(horizon_steps, dtype=np.float64)
                for horizon_index in range(horizon_steps):
                    future_index = time_index + horizon_index + 1
                    target = true_sinr[future_index, ue_id, cell_id]
                    baseline = geometry[horizon_index]
                    valid = (
                        visible[future_index, ue_id, cell_id]
                        and predicted_visible[time_index, ue_id, cell_id, horizon_index]
                        and np.isfinite(target)
                        and np.isfinite(baseline)
                    )
                    if valid:
                        targets[horizon_index] = target
                        residuals[horizon_index] = target - baseline
                        masks[horizon_index] = 1.0
                    elif not np.isfinite(baseline):
                        geometry[horizon_index] = 0.0
                histories.append(
                    features[
                        time_index - history_steps + 1 : time_index + 1,
                        ue_id,
                        cell_id,
                    ].copy()
                )
                geometry_rows.append(geometry)
                target_rows.append(targets)
                residual_rows.append(residuals)
                mask_rows.append(masks)
                episode_ids.append(int(trajectory.episode_id))
                time_indices.append(time_index)
                ue_ids.append(int(ue_id))
                cell_ids.append(int(cell_id))

    if not histories:
        raise ValueError("complete trajectories produced no prediction samples")
    return PredictionDataset(
        histories=np.stack(histories),
        geometry_sinr_db=np.stack(geometry_rows),
        target_sinr_db=np.stack(target_rows),
        target_residual_db=np.stack(residual_rows),
        target_mask=np.stack(mask_rows),
        episode_id=np.asarray(episode_ids),
        time_index=np.asarray(time_indices),
        ue_id=np.asarray(ue_ids),
        cell_id=np.asarray(cell_ids),
    )
