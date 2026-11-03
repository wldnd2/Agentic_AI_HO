from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    "packet_delivery_ratio",
    "packet_drop_rate",
    "deadline_miss_rate",
    "delivered_goodput_bps",
    "mean_queue_packets",
    "p95_queue_packets",
    "mean_hol_delay_s",
    "p95_hol_delay_s",
    "requested_prep_actions",
    "successful_prep_transitions",
    "rejected_prep_requests",
    "prep_rejection_rate",
    "requested_commit_actions",
    "completed_handovers",
    "failed_handovers",
    "requested_release_actions",
    "pingpong_events",
    "fraction_prep_requests_in_same_target_collision",
    "fraction_prep_requests_to_oversubscribed_targets",
    "total_excess_prep_requests",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required pilot artifact does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"pilot artifact must contain a JSON object: {path}")
    return value


def compare_pilot_outputs(run_directories: dict[str, Path]) -> list[dict[str, Any]]:
    expected_labels = {"M2", "M3-G", "M3-H"}
    if set(run_directories) != expected_labels:
        raise ValueError(f"pilot labels must be exactly {sorted(expected_labels)}")
    artifacts = {
        label: (
            _read_json(directory / "checkpoint_metadata.json"),
            _read_json(directory / "evaluation.json"),
        )
        for label, directory in run_directories.items()
    }
    metadata = {label: values[0] for label, values in artifacts.items()}
    shared_fields = (
        "training_seed",
        "environment_seed",
        "total_training_interactions",
        "evaluation_episode_seeds",
    )
    for field in shared_fields:
        values = {json.dumps(item[field], sort_keys=True) for item in metadata.values()}
        if len(values) != 1:
            raise ValueError(f"pilot runs are not matched for {field}")
    for label, item in metadata.items():
        if item.get("history_recurrent_mode") != "recurrent":
            raise ValueError(f"{label} is not recurrent")
        if item.get("coordination_mode") != "none":
            raise ValueError(f"{label} unexpectedly enables coordination")
    if metadata["M2"].get("reasoning_feature_mode") != "none":
        raise ValueError("M2 unexpectedly enables reasoning")
    if metadata["M3-G"].get("world_model_type") != "geometry":
        raise ValueError("M3-G is not Geometry reasoning")
    if metadata["M3-H"].get("world_model_type") != "hybrid":
        raise ValueError("M3-H is not Hybrid reasoning")
    for label in ("M3-G", "M3-H"):
        if metadata[label].get("reasoning_feature_mode") != "confidence_aware":
            raise ValueError(f"{label} is not confidence-aware")
    if not metadata["M3-H"].get("residual_checkpoint_sha256"):
        raise ValueError("M3-H metadata is missing the residual checkpoint SHA-256")

    rows: list[dict[str, Any]] = []
    for label in ("M2", "M3-G", "M3-H"):
        item, evaluation = artifacts[label]
        missing = [metric for metric in METRICS if metric not in evaluation]
        if missing:
            raise ValueError(f"{label} evaluation is missing metrics: {missing}")
        rows.append(
            {
                "variant": label,
                "training_seed": item["training_seed"],
                "environment_seed": item["environment_seed"],
                "training_interactions": item["total_training_interactions"],
                "evaluation_episode_seeds": ",".join(
                    str(seed) for seed in item["evaluation_episode_seeds"]
                ),
                "world_model_type": item.get("world_model_type") or "none",
                "reasoning_feature_mode": item["reasoning_feature_mode"],
                "residual_checkpoint_sha256": (
                    item.get("residual_checkpoint_sha256") or ""
                ),
                **{metric: evaluation[metric] for metric in METRICS},
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched M2/M3-G/M3-H MAPPO pilot outputs"
    )
    parser.add_argument("--m2", type=Path, required=True)
    parser.add_argument("--m3-g", type=Path, required=True)
    parser.add_argument("--m3-h", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_pilot_outputs(
        {"M2": args.m2, "M3-G": args.m3_g, "M3-H": args.m3_h}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"comparison={args.output} variants={len(rows)}")


if __name__ == "__main__":
    main()
