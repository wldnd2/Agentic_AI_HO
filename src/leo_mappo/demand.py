from __future__ import annotations


ARRIVAL_CAPPED = "arrival_capped"
DEADLINE_AWARE = "deadline_aware"
SUPPORTED_REQUIRED_RATE_MODELS = frozenset(
    (ARRIVAL_CAPPED, DEADLINE_AWARE)
)


def normalize_required_rate_model(value: str | None) -> str:
    """Return a supported required-rate model, defaulting to legacy behavior."""
    model = ARRIVAL_CAPPED if value is None else str(value).strip().lower()
    if model not in SUPPORTED_REQUIRED_RATE_MODELS:
        supported = ", ".join(sorted(SUPPORTED_REQUIRED_RATE_MODELS))
        raise ValueError(
            f"traffic.required_rate_model must be one of: {supported}"
        )
    return model


def remaining_service_window_s(
    *,
    deadline_s: float,
    head_of_line_delay_s: float,
    interval_s: float,
) -> float:
    """Observable service window implied by transmit-before-expire timing."""
    safe_interval_s = max(float(interval_s), 1e-8)
    return float(
        max(
            float(deadline_s)
            - float(head_of_line_delay_s)
            + safe_interval_s,
            safe_interval_s,
        )
    )


def compute_required_rate_bps(
    *,
    model: str | None,
    queue_packets: int | float,
    packet_size_bits: int,
    interval_s: float,
    mean_arrival_mbps: float,
    traffic_multiplier: float,
    deadline_s: float,
    head_of_line_delay_s: float,
) -> float:
    """Compute scalar queue service demand from physical/observable inputs.

    ``deadline_aware`` is a queue/deadline-aware backlog-drain demand. It uses
    total backlog and HOL slack as an urgency proxy; it is not a formal
    per-packet deadline guarantee.
    """
    required_rate_model = normalize_required_rate_model(model)
    queue_packet_count = float(queue_packets)
    if queue_packet_count <= 0.0:
        return 0.0

    safe_interval_s = max(float(interval_s), 1e-8)
    queue_bits = queue_packet_count * int(packet_size_bits)
    queue_one_step_cap = queue_bits / safe_interval_s
    arrival_rate = (
        float(mean_arrival_mbps) * 1e6 * float(traffic_multiplier)
    )
    if required_rate_model == ARRIVAL_CAPPED:
        return float(min(arrival_rate, queue_one_step_cap))

    remaining_window = remaining_service_window_s(
        deadline_s=deadline_s,
        head_of_line_delay_s=head_of_line_delay_s,
        interval_s=safe_interval_s,
    )
    deadline_backlog_rate = queue_bits / remaining_window
    return float(
        min(
            queue_one_step_cap,
            max(arrival_rate, deadline_backlog_rate),
        )
    )
