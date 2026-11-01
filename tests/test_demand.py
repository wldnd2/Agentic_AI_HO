from __future__ import annotations

import pytest

from leo_mappo.demand import (
    ARRIVAL_CAPPED,
    DEADLINE_AWARE,
    compute_required_rate_bps,
    normalize_required_rate_model,
    remaining_service_window_s,
)


def _required_rate(model: str, **overrides: float) -> float:
    values = {
        "queue_packets": 200,
        "packet_size_bits": 12_000,
        "interval_s": 1.0,
        "mean_arrival_mbps": 1.0,
        "traffic_multiplier": 1.0,
        "deadline_s": 2.0,
        "head_of_line_delay_s": 0.0,
    }
    values.update(overrides)
    return compute_required_rate_bps(model=model, **values)


@pytest.mark.parametrize("model", (ARRIVAL_CAPPED, DEADLINE_AWARE))
def test_empty_queue_has_exactly_zero_required_rate(model: str) -> None:
    assert _required_rate(model, queue_packets=0) == 0.0


def test_arrival_capped_matches_legacy_formula_exactly() -> None:
    queue_packets = 137
    packet_size_bits = 12_000
    interval_s = 0.75
    mean_arrival_mbps = 1.3
    traffic_multiplier = 1.7
    expected = min(
        mean_arrival_mbps * 1e6 * traffic_multiplier,
        queue_packets * packet_size_bits / max(interval_s, 1e-8),
    )
    actual = compute_required_rate_bps(
        model=ARRIVAL_CAPPED,
        queue_packets=queue_packets,
        packet_size_bits=packet_size_bits,
        interval_s=interval_s,
        mean_arrival_mbps=mean_arrival_mbps,
        traffic_multiplier=traffic_multiplier,
        deadline_s=2.0,
        head_of_line_delay_s=0.0,
    )
    assert actual == expected


def test_deadline_aware_rate_never_exceeds_one_step_queue_cap() -> None:
    queue_packets = 10
    packet_size_bits = 12_000
    interval_s = 1.0
    actual = _required_rate(
        DEADLINE_AWARE,
        queue_packets=queue_packets,
        packet_size_bits=packet_size_bits,
        interval_s=interval_s,
        mean_arrival_mbps=1000.0,
        head_of_line_delay_s=100.0,
    )
    assert actual <= queue_packets * packet_size_bits / interval_s


def test_deadline_aware_matches_arrival_target_without_deadline_pressure() -> None:
    legacy = _required_rate(
        ARRIVAL_CAPPED,
        queue_packets=200,
        mean_arrival_mbps=1.0,
        deadline_s=10.0,
    )
    deadline_aware = _required_rate(
        DEADLINE_AWARE,
        queue_packets=200,
        mean_arrival_mbps=1.0,
        deadline_s=10.0,
    )
    assert deadline_aware == legacy


def test_deadline_pressure_is_non_decreasing_with_hol_delay() -> None:
    rates = [
        _required_rate(
            DEADLINE_AWARE,
            queue_packets=200,
            mean_arrival_mbps=0.1,
            head_of_line_delay_s=hol_delay,
        )
        for hol_delay in (0.0, 1.0, 2.0, 3.0)
    ]
    assert all(left <= right for left, right in zip(rates, rates[1:]))


@pytest.mark.parametrize(
    ("hol_delay_s", "expected_window_s"),
    ((0.0, 3.0), (1.0, 2.0), (2.0, 1.0)),
)
def test_remaining_service_window_matches_simulator_timing(
    hol_delay_s: float,
    expected_window_s: float,
) -> None:
    assert remaining_service_window_s(
        deadline_s=2.0,
        head_of_line_delay_s=hol_delay_s,
        interval_s=1.0,
    ) == expected_window_s


def test_hol_beyond_deadline_uses_one_interval_service_window() -> None:
    assert remaining_service_window_s(
        deadline_s=2.0,
        head_of_line_delay_s=5.0,
        interval_s=1.0,
    ) == 1.0


def test_required_rate_model_defaults_and_rejects_unknown_values() -> None:
    assert normalize_required_rate_model(None) == ARRIVAL_CAPPED
    with pytest.raises(ValueError, match="traffic.required_rate_model"):
        normalize_required_rate_model("unsupported")
