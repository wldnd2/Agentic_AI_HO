from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_closed_loop_ab import (
    CHO_CONTROL_METRICS,
    METRICS,
    compare_metric,
    match_episode_rows,
    metric_direction,
    paired_bootstrap_mean_ci,
    read_episode_csv,
)


def _row(seed: int, value: float = 0.0) -> dict[str, float | int]:
    return {
        "episode_seed": seed,
        **{metric: value for metric in METRICS},
    }


def _write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["episode_seed", *METRICS]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_episode_rows_are_matched_exactly_by_seed() -> None:
    baseline = [_row(11, 1.0), _row(12, 2.0)]
    treatment = [_row(12, 20.0), _row(11, 10.0)]
    pairs = match_episode_rows(baseline, treatment)
    assert [int(left["episode_seed"]) for left, _ in pairs] == [11, 12]
    assert [int(right["episode_seed"]) for _, right in pairs] == [11, 12]
    assert [float(right["packet_delivery_ratio"]) for _, right in pairs] == [
        10.0,
        20.0,
    ]


def test_mismatched_episode_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="episode seeds differ"):
        match_episode_rows(
            [_row(11), _row(12)],
            [_row(11), _row(13)],
        )


def test_mismatched_episode_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="episode counts differ"):
        match_episode_rows([_row(11)], [_row(11), _row(12)])


def test_duplicate_episode_seeds_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "episodes.csv"
    _write_rows(source, [_row(11), _row(11)])
    with pytest.raises(ValueError, match="duplicate episode seed 11"):
        read_episode_csv(source)


def test_missing_requested_metric_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "episodes.csv"
    columns = ["episode_seed", *METRICS[:-1]]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: 0 for column in columns})
    with pytest.raises(ValueError, match="missing required columns"):
        read_episode_csv(source)


def test_paired_delta_statistics_are_correct() -> None:
    result = compare_metric(
        "packet_delivery_ratio",
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 4.0, 6.0]),
    )
    assert result["baseline_mean"] == 2.0
    assert result["treatment_mean"] == 4.0
    assert result["mean_paired_difference"] == 2.0
    assert result["absolute_mean_difference"] == 2.0
    assert result["relative_mean_difference_percent"] == 100.0
    assert result["median_paired_difference"] == 2.0
    assert result["paired_difference_std"] == 1.0


def test_higher_is_better_fraction_and_sign_counts() -> None:
    result = compare_metric(
        "packet_delivery_ratio",
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([2.0, 0.0, 1.0]),
    )
    assert result["direction"] == "higher_is_better"
    assert result["fraction_treatment_better"] == 1 / 3
    assert result["paired_sign_counts"] == {
        "treatment_better": 1,
        "baseline_better": 1,
        "equal": 1,
    }


def test_lower_is_better_fraction_and_sign_counts() -> None:
    result = compare_metric(
        "deadline_miss_rate",
        np.asarray([1.0, 1.0, 1.0, 1.0]),
        np.asarray([0.0, 0.5, 2.0, 1.0]),
    )
    assert result["direction"] == "lower_is_better"
    assert result["fraction_treatment_better"] == 0.5
    assert result["paired_sign_counts"] == {
        "treatment_better": 2,
        "baseline_better": 1,
        "equal": 1,
    }


def test_paired_bootstrap_is_deterministic() -> None:
    differences = np.asarray([-2.0, -1.0, 0.0, 1.0, 4.0])
    first = paired_bootstrap_mean_ci(differences)
    second = paired_bootstrap_mean_ci(differences)
    assert first == second


def test_paired_bootstrap_generates_ordered_ci_containing_sample_mean() -> None:
    differences = np.asarray([-1.0, 0.0, 1.0, 2.0])
    lower, upper = paired_bootstrap_mean_ci(differences)
    assert lower <= float(np.mean(differences)) <= upper
    assert lower < upper


def test_operational_metrics_have_no_assigned_success_direction() -> None:
    result = compare_metric(
        "requested_prep_actions",
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([30.0, 20.0, 10.0]),
    )
    assert result["direction"] == "operational_tradeoff"
    assert result["fraction_treatment_better"] is None
    assert result["paired_sign_counts"] is None
    assert result["fraction_treatment_greater"] == 1 / 3
    assert result["fraction_treatment_lower"] == 1 / 3
    assert result["fraction_equal"] == 1 / 3


@pytest.mark.parametrize(
    "metric",
    (
        "successful_prep_transitions",
        "rejected_prep_requests",
        "successful_commit_starts",
        "failed_handovers",
        "pingpong_events",
        "mean_handovers_per_ue",
        "mean_time_between_completed_handovers_s",
        "mean_target_serving_duration_after_handover_s",
    ),
)
def test_extended_operational_metrics_are_direction_neutral(metric: str) -> None:
    assert metric in CHO_CONTROL_METRICS
    assert metric_direction(metric) == "operational_tradeoff"
    result = compare_metric(
        metric,
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 2.0, 1.0]),
    )
    assert result["direction"] == "operational_tradeoff"
    assert result["fraction_treatment_better"] is None
    assert result["paired_sign_counts"] is None
