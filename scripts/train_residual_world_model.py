from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from leo_mappo.prediction_dataset import PredictionDataset
from leo_mappo.residual_world_model import (
    DISTANCE_SCALE_M,
    ELEVATION_OFFSET_DEG,
    ELEVATION_SCALE_DEG,
    LOAD_SCALE,
    OBSERVABLE_LINK_FEATURES,
    SINR_OFFSET_DB,
    SINR_SCALE_DB,
    VISIBILITY_SCALE_S,
    ResidualLinkPredictor,
)


def _masked_mean(values: np.ndarray, masks: np.ndarray) -> float:
    count = float(np.sum(masks))
    if count <= 0.0:
        raise ValueError("metric has no valid targets")
    return float(np.sum(values * masks) / count)


def _validation_metrics(
    dataset: PredictionDataset, residual_prediction: np.ndarray
) -> tuple[float, float, float]:
    residual_error = residual_prediction - dataset.target_residual_db
    hybrid_prediction = dataset.geometry_sinr_db + residual_prediction
    hybrid_error = hybrid_prediction - dataset.target_sinr_db
    return (
        _masked_mean(residual_error**2, dataset.target_mask),
        _masked_mean(np.abs(residual_error), dataset.target_mask),
        _masked_mean(np.abs(hybrid_error), dataset.target_mask),
    )


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch size, and hidden dimension must be positive")
    train_data = PredictionDataset.load(args.train)
    validation_data = PredictionDataset.load(args.validation)
    train_shape = (
        train_data.history_steps,
        train_data.feature_dim,
        train_data.horizon_steps,
    )
    validation_shape = (
        validation_data.history_steps,
        validation_data.feature_dim,
        validation_data.horizon_steps,
    )
    if train_shape != validation_shape:
        raise ValueError(
            "training and validation datasets must have matching history, feature, "
            "and horizon dimensions"
        )
    if np.sum(train_data.target_mask) <= 0 or np.sum(validation_data.target_mask) <= 0:
        raise ValueError("training and validation datasets must each have valid targets")

    model = ResidualLinkPredictor(
        feature_dim=train_data.feature_dim,
        history_steps=train_data.history_steps,
        horizon_steps=train_data.horizon_steps,
        hidden_dim=args.hidden_dim,
        mlp_hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    metrics: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_data.sample_count)
        weighted_loss = 0.0
        valid_count = 0.0
        for start in range(0, train_data.sample_count, args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_mask = train_data.target_mask[indices]
            batch_valid_count = float(np.sum(batch_mask))
            batch_loss = model.train_batch(
                train_data.histories[indices],
                train_data.target_residual_db[indices],
                batch_mask,
            )
            weighted_loss += batch_loss * batch_valid_count
            valid_count += batch_valid_count
        train_masked_mse = weighted_loss / max(valid_count, 1e-12)
        validation_prediction = model.predict(validation_data.histories)
        validation_mse, validation_residual_mae, validation_hybrid_mae = (
            _validation_metrics(validation_data, validation_prediction)
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_masked_mse": train_masked_mse,
            "validation_masked_mse": validation_mse,
            "validation_residual_mae": validation_residual_mae,
            "validation_hybrid_sinr_mae": validation_hybrid_mae,
        }
        metrics.append(row)
        print(
            f"epoch={epoch:04d} train_mse={train_masked_mse:.6f} "
            f"validation_mse={validation_mse:.6f} "
            f"validation_hybrid_mae={validation_hybrid_mae:.6f}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.npz")
    model_config = {
        "architecture": "GRU final hidden state -> tanh MLP -> residual SINR horizons",
        "feature_dim": model.feature_dim,
        "feature_order": list(OBSERVABLE_LINK_FEATURES),
        "history_steps": model.history_steps,
        "horizon_steps": model.horizon_steps,
        "gru_hidden_dim": model.hidden_dim,
        "mlp_hidden_dim": model.mlp_hidden_dim,
        "learning_rate": model.learning_rate,
        "max_grad_norm": model.max_grad_norm,
        "seed": model.seed,
        "normalization": {
            "measured_sinr": f"0 if unavailable, else clip((dB + {SINR_OFFSET_DB}) / {SINR_SCALE_DB}, 0, 1)",
            "observed_load": f"clip(load / {LOAD_SCALE}, 0, 1.5)",
            "elevation": f"clip((degrees + {ELEVATION_OFFSET_DEG}) / {ELEVATION_SCALE_DEG}, 0, 1)",
            "distance": f"clip(metres / {DISTANCE_SCALE_M}, 0, 2)",
            "remaining_visibility": f"clip(seconds / {VISIBILITY_SCALE_S}, 0, 1)",
            "indicators": "binary 0 or 1",
        },
    }
    with (output / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)
    with (output / "training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the residual SINR world model.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
