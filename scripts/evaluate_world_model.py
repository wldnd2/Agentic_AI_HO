from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from leo_mappo.prediction_dataset import PredictionDataset
from leo_mappo.residual_world_model import ResidualLinkPredictor


def _error_metrics(errors: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(errors**2))),
        float(np.mean(errors)),
    )


def evaluate(args: argparse.Namespace) -> None:
    dataset = PredictionDataset.load(args.dataset)
    model = ResidualLinkPredictor.load(args.model)
    expected_shape = (
        model.history_steps,
        model.feature_dim,
        model.horizon_steps,
    )
    dataset_shape = (
        dataset.history_steps,
        dataset.feature_dim,
        dataset.horizon_steps,
    )
    if dataset_shape != expected_shape:
        raise ValueError("dataset dimensions do not match the saved model")

    residual = model.predict(dataset.histories)
    hybrid = dataset.geometry_sinr_db + residual
    rows: list[dict[str, float | int]] = []
    for horizon_index in range(dataset.horizon_steps):
        valid = dataset.target_mask[:, horizon_index] > 0.0
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            raise ValueError(f"horizon {horizon_index + 1} has no valid targets")
        target = dataset.target_sinr_db[valid, horizon_index]
        geometry_error = dataset.geometry_sinr_db[valid, horizon_index] - target
        hybrid_error = hybrid[valid, horizon_index] - target
        geometry_mae, geometry_rmse, geometry_bias = _error_metrics(geometry_error)
        hybrid_mae, hybrid_rmse, hybrid_bias = _error_metrics(hybrid_error)
        absolute_improvement = geometry_mae - hybrid_mae
        percentage_improvement = (
            100.0 * absolute_improvement / geometry_mae
            if geometry_mae > 0.0
            else 0.0
        )
        rows.append(
            {
                "horizon": horizon_index + 1,
                "valid_count": valid_count,
                "geometry_mae": geometry_mae,
                "geometry_rmse": geometry_rmse,
                "geometry_bias": geometry_bias,
                "hybrid_mae": hybrid_mae,
                "hybrid_rmse": hybrid_rmse,
                "hybrid_bias": hybrid_bias,
                "absolute_mae_improvement": absolute_improvement,
                "percentage_mae_improvement": percentage_improvement,
            }
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_valid = dataset.target_mask > 0.0
    all_target = dataset.target_sinr_db[all_valid]
    all_geometry_error = dataset.geometry_sinr_db[all_valid] - all_target
    all_hybrid_error = hybrid[all_valid] - all_target
    geometry_mae, geometry_rmse, geometry_bias = _error_metrics(all_geometry_error)
    hybrid_mae, hybrid_rmse, hybrid_bias = _error_metrics(all_hybrid_error)
    absolute_improvement = geometry_mae - hybrid_mae
    summary = {
        "valid_count": int(np.count_nonzero(all_valid)),
        "geometry": {
            "mae": geometry_mae,
            "rmse": geometry_rmse,
            "bias": geometry_bias,
        },
        "hybrid": {
            "mae": hybrid_mae,
            "rmse": hybrid_rmse,
            "bias": hybrid_bias,
        },
        "absolute_mae_improvement": absolute_improvement,
        "percentage_mae_improvement": (
            100.0 * absolute_improvement / geometry_mae if geometry_mae > 0.0 else 0.0
        ),
        "per_horizon": rows,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "horizon valid geometry_mae geometry_rmse geometry_bias "
        "hybrid_mae hybrid_rmse hybrid_bias improvement improvement_pct"
    )
    for row in rows:
        print(
            f"{row['horizon']:>7d} {row['valid_count']:>5d} "
            f"{row['geometry_mae']:>12.5f} {row['geometry_rmse']:>13.5f} "
            f"{row['geometry_bias']:>13.5f} {row['hybrid_mae']:>10.5f} "
            f"{row['hybrid_rmse']:>11.5f} {row['hybrid_bias']:>11.5f} "
            f"{row['absolute_mae_improvement']:>11.5f} "
            f"{row['percentage_mae_improvement']:>15.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate geometry-only and residual-corrected SINR forecasts."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
