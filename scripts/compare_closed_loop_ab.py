from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261102

TRAFFIC_QOS_METRICS = (
    "packet_delivery_ratio",
    "packet_drop_rate",
    "deadline_miss_rate",
    "delivered_goodput_bps",
)
QUEUE_DELAY_METRICS = (
    "mean_queue_packets",
    "p95_queue_packets",
    "mean_hol_delay_s",
    "p95_hol_delay_s",
)
CHO_CONTROL_METRICS = (
    "requested_prep_actions",
    "requested_commit_actions",
    "requested_release_actions",
    "successful_prep_transitions",
    "rejected_prep_requests",
    "successful_commit_starts",
    "completed_handovers",
    "failed_handovers",
    "release_transitions",
    "pingpong_events",
    "mean_handovers_per_ue",
    "mean_time_between_completed_handovers_s",
    "mean_target_serving_duration_after_handover_s",
)
SERVICE_METRICS = (
    "mean_required_rate_bps",
    "p95_required_rate_bps",
    "fraction_potential_service_insufficient",
    "mean_primary_service_deficit",
)
METRIC_GROUPS = {
    "traffic_qos": TRAFFIC_QOS_METRICS,
    "queue_delay": QUEUE_DELAY_METRICS,
    "handover_control": CHO_CONTROL_METRICS,
    "service": SERVICE_METRICS,
}
METRICS = tuple(
    metric for metrics in METRIC_GROUPS.values() for metric in metrics
)
HIGHER_IS_BETTER = frozenset(
    ("packet_delivery_ratio", "delivered_goodput_bps")
)
LOWER_IS_BETTER = frozenset(
    (
        "packet_drop_rate",
        "deadline_miss_rate",
        "mean_queue_packets",
        "p95_queue_packets",
        "mean_hol_delay_s",
        "p95_hol_delay_s",
    )
)
DIRECTIONAL_METRICS = HIGHER_IS_BETTER | LOWER_IS_BETTER
REQUIRED_COLUMNS = frozenset(("episode_seed", *METRICS))


def metric_group(metric: str) -> str:
    for group, metrics in METRIC_GROUPS.items():
        if metric in metrics:
            return group
    raise KeyError(f"unknown metric {metric!r}")


def metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    return "operational_tradeoff"


def read_episode_csv(path: str | Path) -> list[dict[str, float | int]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"episode CSV does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"{source} is missing required columns: {', '.join(missing)}"
            )
        rows: list[dict[str, float | int]] = []
        seeds: set[int] = set()
        for row_number, raw in enumerate(reader, start=2):
            try:
                seed = int(raw["episode_seed"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{source}:{row_number} has invalid episode_seed"
                ) from error
            if seed in seeds:
                raise ValueError(
                    f"{source} contains duplicate episode seed {seed}"
                )
            seeds.add(seed)
            parsed: dict[str, float | int] = {"episode_seed": seed}
            for metric in METRICS:
                try:
                    value = float(raw[metric])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{source}:{row_number} has invalid {metric}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"{source}:{row_number} has non-finite {metric}"
                    )
                parsed[metric] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{source} contains no episode rows")
    return rows


def match_episode_rows(
    baseline_rows: list[dict[str, float | int]],
    treatment_rows: list[dict[str, float | int]],
) -> list[
    tuple[dict[str, float | int], dict[str, float | int]]
]:
    if len(baseline_rows) != len(treatment_rows):
        raise ValueError(
            "baseline and treatment episode counts differ: "
            f"{len(baseline_rows)} != {len(treatment_rows)}"
        )

    def indexed(
        rows: list[dict[str, float | int]], label: str
    ) -> dict[int, dict[str, float | int]]:
        result: dict[int, dict[str, float | int]] = {}
        for row in rows:
            seed = int(row["episode_seed"])
            if seed in result:
                raise ValueError(
                    f"{label} contains duplicate episode seed {seed}"
                )
            result[seed] = row
        return result

    baseline_by_seed = indexed(baseline_rows, "baseline")
    treatment_by_seed = indexed(treatment_rows, "treatment")
    baseline_seeds = set(baseline_by_seed)
    treatment_seeds = set(treatment_by_seed)
    if baseline_seeds != treatment_seeds:
        baseline_only = sorted(baseline_seeds - treatment_seeds)
        treatment_only = sorted(treatment_seeds - baseline_seeds)
        raise ValueError(
            "baseline and treatment episode seeds differ; "
            f"baseline_only={baseline_only}, treatment_only={treatment_only}"
        )
    return [
        (row, treatment_by_seed[int(row["episode_seed"])])
        for row in baseline_rows
    ]


def paired_bootstrap_mean_ci(
    paired_differences: np.ndarray,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    differences = np.asarray(paired_differences, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("paired_differences must be a non-empty 1-D array")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        differences.size,
        size=(samples, differences.size),
    )
    bootstrap_means = np.mean(differences[indices], axis=1)
    lower, upper = np.percentile(bootstrap_means, (2.5, 97.5))
    return float(lower), float(upper)


def compare_metric(
    metric: str,
    baseline_values: np.ndarray,
    treatment_values: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_values, dtype=np.float64)
    treatment = np.asarray(treatment_values, dtype=np.float64)
    if (
        baseline.shape != treatment.shape
        or baseline.ndim != 1
        or baseline.size == 0
    ):
        raise ValueError(
            "baseline and treatment values must be matching non-empty 1-D arrays"
        )
    differences = treatment - baseline
    baseline_mean = float(np.mean(baseline))
    treatment_mean = float(np.mean(treatment))
    mean_difference = float(np.mean(differences))
    relative_difference = (
        100.0 * mean_difference / baseline_mean
        if baseline_mean != 0.0
        else None
    )
    ci_lower, ci_upper = paired_bootstrap_mean_ci(differences)
    treatment_greater = int(np.count_nonzero(differences > 0.0))
    treatment_lower = int(np.count_nonzero(differences < 0.0))
    equal = int(np.count_nonzero(differences == 0.0))
    direction = metric_direction(metric)
    if direction == "higher_is_better":
        treatment_better = treatment_greater
        baseline_better = treatment_lower
    elif direction == "lower_is_better":
        treatment_better = treatment_lower
        baseline_better = treatment_greater
    else:
        treatment_better = None
        baseline_better = None

    return {
        "metric": metric,
        "group": metric_group(metric),
        "direction": direction,
        "matched_episode_count": int(differences.size),
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "mean_paired_difference": mean_difference,
        "absolute_mean_difference": abs(mean_difference),
        "relative_mean_difference_percent": relative_difference,
        "median_paired_difference": float(np.median(differences)),
        "paired_difference_std": float(
            np.std(differences, ddof=1) if differences.size > 1 else 0.0
        ),
        "mean_difference_bootstrap_ci_95": {
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "fraction_treatment_greater": treatment_greater / differences.size,
        "fraction_treatment_lower": treatment_lower / differences.size,
        "fraction_equal": equal / differences.size,
        "fraction_treatment_better": (
            treatment_better / differences.size
            if treatment_better is not None
            else None
        ),
        "paired_sign_counts": (
            {
                "treatment_better": treatment_better,
                "baseline_better": baseline_better,
                "equal": equal,
            }
            if treatment_better is not None
            else None
        ),
    }


def build_comparison(
    baseline_path: str | Path,
    treatment_path: str | Path,
) -> dict[str, Any]:
    baseline_rows = read_episode_csv(baseline_path)
    treatment_rows = read_episode_csv(treatment_path)
    pairs = match_episode_rows(baseline_rows, treatment_rows)
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        baseline_values = np.asarray(
            [float(baseline[metric]) for baseline, _ in pairs]
        )
        treatment_values = np.asarray(
            [float(treatment[metric]) for _, treatment in pairs]
        )
        metrics[metric] = compare_metric(
            metric, baseline_values, treatment_values
        )
    return {
        "analysis_control": {
            "baseline": str(baseline_path),
            "treatment": str(treatment_path),
            "matched_episode_count": len(pairs),
            "matched_episode_seeds": [
                int(baseline["episode_seed"]) for baseline, _ in pairs
            ],
            "paired_difference": "treatment - baseline",
            "paired_difference_std": "sample standard deviation (ddof=1)",
            "bootstrap": {
                "method": "paired percentile bootstrap of mean differences",
                "samples": BOOTSTRAP_SAMPLES,
                "rng_seed": BOOTSTRAP_SEED,
                "percentiles": [2.5, 97.5],
            },
            "interpretation": (
                "Operational CHO/control and service metrics have no assigned "
                "success direction."
            ),
        },
        "metrics": metrics,
    }


def write_outputs(output: str | Path, comparison: dict[str, Any]) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    csv_rows: list[dict[str, Any]] = []
    for result in comparison["metrics"].values():
        ci = result["mean_difference_bootstrap_ci_95"]
        sign_counts = result["paired_sign_counts"] or {}
        csv_rows.append(
            {
                **{
                    key: value
                    for key, value in result.items()
                    if key
                    not in (
                        "mean_difference_bootstrap_ci_95",
                        "paired_sign_counts",
                    )
                },
                "bootstrap_ci_95_lower": ci["lower"],
                "bootstrap_ci_95_upper": ci["upper"],
                "treatment_better_count": sign_counts.get(
                    "treatment_better"
                ),
                "baseline_better_count": sign_counts.get("baseline_better"),
                "equal_count": sign_counts.get("equal"),
            }
        )
    with (destination / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def print_compact_summary(comparison: dict[str, Any]) -> None:
    for metric in METRICS:
        if metric not in DIRECTIONAL_METRICS:
            continue
        result = comparison["metrics"][metric]
        ci = result["mean_difference_bootstrap_ci_95"]
        print(
            f"{metric}: baseline={result['baseline_mean']:.6g} "
            f"treatment={result['treatment_mean']:.6g} "
            f"delta={result['mean_paired_difference']:.6g} "
            f"ci95=[{ci['lower']:.6g},{ci['upper']:.6g}] "
            f"fraction_better={result['fraction_treatment_better']:.3f}"
        )


def run(args: argparse.Namespace) -> None:
    comparison = build_comparison(args.baseline, args.treatment)
    write_outputs(args.output, comparison)
    print_compact_summary(comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched closed-loop evaluation episodes."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
