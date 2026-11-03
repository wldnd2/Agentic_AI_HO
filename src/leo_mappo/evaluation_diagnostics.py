from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv, StepOutput


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


@dataclass(frozen=True)
class PreparationContentionSnapshot:
    request_counts_by_target: np.ndarray
    total_requests: int
    unique_targets: int
    max_same_target_requests: int
    collision_target_count: int
    collision_request_count: int


def analyze_preparation_contention(
    env: LEOMobilityEnv,
    actions: np.ndarray,
) -> PreparationContentionSnapshot:
    """Summarize a selected joint PREP action without changing it."""
    request_counts = np.zeros(env.num_cells, dtype=np.int64)
    for ue_id, action_value in enumerate(np.asarray(actions, dtype=np.int64)):
        action = int(action_value)
        if not (
            env.prepare_start
            <= action
            < env.prepare_start + env.candidate_count
        ):
            continue
        slot = action - env.prepare_start
        if slot < len(env.candidates[ue_id]):
            target = int(env.candidates[ue_id][slot])
            request_counts[target] += 1
    active_targets = request_counts > 0
    collision_targets = request_counts >= 2
    return PreparationContentionSnapshot(
        request_counts_by_target=request_counts,
        total_requests=int(np.sum(request_counts)),
        unique_targets=int(np.count_nonzero(active_targets)),
        max_same_target_requests=int(request_counts.max(initial=0)),
        collision_target_count=int(np.count_nonzero(collision_targets)),
        collision_request_count=int(np.sum(request_counts[collision_targets])),
    )


@dataclass
class PolicyEvaluationDiagnostics:
    """Evaluation-only QoS, CHO, and joint-PREP diagnostic accumulator."""

    interval_s: float
    packet_size_bits: int
    steps: int = 0
    arrivals: float = 0.0
    delivered_packets: float = 0.0
    deadline_misses: float = 0.0
    overflow_packets: float = 0.0
    requested_prep_actions: int = 0
    requested_commit_actions: int = 0
    requested_release_actions: int = 0
    preparation_requested: float = 0.0
    preparation_admitted: float = 0.0
    preparation_rejected: float = 0.0
    handover_started: float = 0.0
    handover_success: float = 0.0
    handover_failure: float = 0.0
    release_count: float = 0.0
    pingpong: float = 0.0
    queue_samples: list[float] = field(default_factory=list)
    hol_samples: list[float] = field(default_factory=list)
    active_prep_epochs: int = 0
    prep_request_total: int = 0
    prep_unique_target_samples: list[float] = field(default_factory=list)
    prep_max_same_target_samples: list[float] = field(default_factory=list)
    prep_collision_target_total: int = 0
    prep_collision_request_total: int = 0
    prep_capacity_observed_epochs: int = 0
    prep_available_slots_total: int = 0
    prep_requested_minus_available_total: int = 0
    prep_excess_request_total: int = 0
    prep_oversubscribed_target_total: int = 0
    prep_oversubscribed_request_total: int = 0

    def record_step(
        self,
        env: LEOMobilityEnv,
        actions: np.ndarray,
        output: StepOutput,
        contention: PreparationContentionSnapshot,
    ) -> None:
        self.steps += 1
        info = output.info
        self.arrivals += float(info.get("arrivals", 0.0))
        self.delivered_packets += float(info.get("delivered_packets", 0.0))
        self.deadline_misses += float(info.get("deadline_misses", 0.0))
        self.overflow_packets += float(info.get("overflow_packets", 0.0))
        self.preparation_requested += float(info.get("preparation_requested", 0.0))
        self.preparation_admitted += float(info.get("preparation_admitted", 0.0))
        self.preparation_rejected += float(info.get("preparation_rejected", 0.0))
        self.handover_started += float(info.get("handover_started", 0.0))
        self.handover_success += float(info.get("handover_success", 0.0))
        self.handover_failure += float(info.get("handover_failure", 0.0))
        self.release_count += float(info.get("release_count", 0.0))
        self.pingpong += float(info.get("pingpong", 0.0))

        selected = np.asarray(actions, dtype=np.int64)
        self.requested_prep_actions += int(
            np.count_nonzero(
                (selected >= env.prepare_start)
                & (selected < env.prepare_start + env.candidate_count)
            )
        )
        self.requested_commit_actions += int(
            np.count_nonzero(selected == env.commit_action)
        )
        self.requested_release_actions += int(
            np.count_nonzero(selected == env.release_action)
        )
        for ue in env.ues:
            self.queue_samples.append(float(ue.queue_packets))
            self.hol_samples.append(float(ue.head_of_line_delay_s))
        self._record_contention(
            contention,
            getattr(env, "last_preparation_available_slots", None),
        )

    def _record_contention(
        self,
        snapshot: PreparationContentionSnapshot,
        available_slots_by_target: np.ndarray | None,
    ) -> None:
        self.prep_request_total += snapshot.total_requests
        self.prep_collision_target_total += snapshot.collision_target_count
        self.prep_collision_request_total += snapshot.collision_request_count
        if snapshot.total_requests <= 0:
            return
        self.active_prep_epochs += 1
        self.prep_unique_target_samples.append(float(snapshot.unique_targets))
        self.prep_max_same_target_samples.append(
            float(snapshot.max_same_target_requests)
        )
        if available_slots_by_target is None:
            return
        available = np.asarray(available_slots_by_target, dtype=np.int64)
        counts = snapshot.request_counts_by_target
        requested_targets = counts > 0
        if available.shape != counts.shape or np.any(available[requested_targets] < 0):
            return
        requested = counts[requested_targets]
        target_capacity = available[requested_targets]
        gaps = requested - target_capacity
        oversubscribed = gaps > 0
        self.prep_capacity_observed_epochs += 1
        self.prep_available_slots_total += int(np.sum(target_capacity))
        self.prep_requested_minus_available_total += int(np.sum(gaps))
        self.prep_excess_request_total += int(np.sum(np.maximum(gaps, 0)))
        self.prep_oversubscribed_target_total += int(
            np.count_nonzero(oversubscribed)
        )
        self.prep_oversubscribed_request_total += int(
            np.sum(requested[oversubscribed])
        )

    def metrics(self) -> dict[str, float]:
        dropped = self.deadline_misses + self.overflow_packets
        duration_s = self.steps * self.interval_s
        return {
            "generated_packets": self.arrivals,
            "dropped_packets": dropped,
            "packet_delivery_ratio": _safe_ratio(
                self.delivered_packets, self.arrivals
            ),
            "packet_drop_rate": _safe_ratio(dropped, self.arrivals),
            "deadline_miss_rate": _safe_ratio(
                self.deadline_misses, self.arrivals
            ),
            "delivered_goodput_bps": _safe_ratio(
                self.delivered_packets * self.packet_size_bits,
                duration_s,
            ),
            "mean_queue_packets": _mean(self.queue_samples),
            "p95_queue_packets": _p95(self.queue_samples),
            "mean_hol_delay_s": _mean(self.hol_samples),
            "p95_hol_delay_s": _p95(self.hol_samples),
            "requested_prep_actions": float(self.requested_prep_actions),
            "successful_prep_transitions": self.preparation_admitted,
            "rejected_prep_requests": self.preparation_rejected,
            "prep_rejection_rate": _safe_ratio(
                self.preparation_rejected, self.preparation_requested
            ),
            "requested_commit_actions": float(self.requested_commit_actions),
            "successful_commit_starts": self.handover_started,
            "completed_handovers": self.handover_success,
            "failed_handovers": self.handover_failure,
            "requested_release_actions": float(self.requested_release_actions),
            "release_transitions": self.release_count,
            "pingpong_events": self.pingpong,
            "active_prep_epochs": float(self.active_prep_epochs),
            "diagnostic_prep_request_count": float(self.prep_request_total),
            "same_target_collision_prep_request_count": float(
                self.prep_collision_request_total
            ),
            "mean_prep_requests_per_active_prep_epoch": _safe_ratio(
                self.prep_request_total, self.active_prep_epochs
            ),
            "mean_unique_prep_targets_per_active_prep_epoch": _mean(
                self.prep_unique_target_samples
            ),
            "mean_max_same_target_prep_requests": _mean(
                self.prep_max_same_target_samples
            ),
            "p95_max_same_target_prep_requests": _p95(
                self.prep_max_same_target_samples
            ),
            "fraction_prep_requests_in_same_target_collision": _safe_ratio(
                self.prep_collision_request_total, self.prep_request_total
            ),
            "mean_same_target_collision_satellites_per_epoch": _safe_ratio(
                self.prep_collision_target_total, self.steps
            ),
            "prep_admission_capacity_observed_epochs": float(
                self.prep_capacity_observed_epochs
            ),
            "mean_available_prep_admissions_on_requested_targets": _safe_ratio(
                self.prep_available_slots_total,
                self.prep_capacity_observed_epochs,
            ),
            "mean_requested_minus_available_prep_capacity": _safe_ratio(
                self.prep_requested_minus_available_total,
                self.prep_capacity_observed_epochs,
            ),
            "mean_excess_prep_requests": _safe_ratio(
                self.prep_excess_request_total, self.steps
            ),
            "total_excess_prep_requests": float(self.prep_excess_request_total),
            "fraction_prep_requests_to_oversubscribed_targets": _safe_ratio(
                self.prep_oversubscribed_request_total, self.prep_request_total
            ),
            "mean_oversubscribed_prep_targets_per_epoch": _safe_ratio(
                self.prep_oversubscribed_target_total, self.steps
            ),
        }
