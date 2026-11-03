from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from leo_mappo.config import load_config
from leo_mappo.environment import LEOMobilityEnv, StepOutput
from leo_mappo.evaluation_diagnostics import (
    PreparationContentionSnapshot,
    analyze_preparation_contention,
)
from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.planner import ReasoningResult
from leo_mappo.predictive_control import (
    build_predictive_reasoner as _build_predictive_reasoner,
    predictive_controller_step,
    select_greedy_feasible_actions,
)
from leo_mappo.reasoning import LookaheadReasoner
from leo_mappo.ue import MobilityState


CONTROLLER_NAME = "geometry_counterfactual_greedy"
PROTOCOL_STATES = tuple(state.name for state in MobilityState)
CANONICAL_CONTROL_COUNTERS = (
    "preparation_requested",
    "preparation_admitted",
    "preparation_rejected",
    "secondary_link_lost",
    "handover_started",
    "handover_success",
    "handover_failure",
    "release_count",
    "pingpong",
)


def geometry_controller_step(
    env: LEOMobilityEnv,
    reasoner: LookaheadReasoner,
    before_step: Callable[[LEOMobilityEnv, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, ReasoningResult, StepOutput]:
    """Backward-compatible name for the shared predictive controller."""
    return predictive_controller_step(env, reasoner, before_step=before_step)


def build_predictive_reasoner(
    config: dict[str, Any],
    env: LEOMobilityEnv,
    world_model: str = "geometry",
    residual_model: str | Path | None = None,
) -> LookaheadReasoner:
    return _build_predictive_reasoner(
        config,
        env,
        world_model=world_model,
        residual_model=residual_model,
        reasoner_factory=LookaheadReasoner,
    )


def build_geometry_reasoner(
    config: dict[str, Any],
    env: LEOMobilityEnv,
) -> LookaheadReasoner:
    """Build the historical Geometry-only reasoner."""
    return build_predictive_reasoner(config, env, world_model="geometry")


def load_evaluation_config(
    config_path: str | Path,
    override_paths: list[str | Path],
    seed: int,
) -> dict[str, Any]:
    config = load_config(config_path, override_paths)
    config["environment_seed"] = int(seed)
    return config


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def prep_rejection_rate(rejected: float, requested: float) -> float:
    """Return the diagnostic rejection fraction with a safe empty denominator."""
    return _safe_ratio(rejected, requested)


def _current_primary_service_samples(
    env: LEOMobilityEnv,
    reasoner: LookaheadReasoner,
) -> list[tuple[float, bool, float]]:
    """Return physical-demand service samples without changing state."""
    active, required_rates = env.current_required_rates()
    available = (
        np.ones(env.num_cells, dtype=np.float64)
        if env.v3_enabled
        else np.maximum(0.0, 1.0 - env._compute_preparation_reservations())
    )
    samples: list[tuple[float, bool, float]] = []
    for ue in env.ues:
        ue_id = int(ue.ue_id)
        required_rate = float(required_rates[ue_id])
        if active[ue_id] <= 0.0 or required_rate <= 0.0:
            continue
        service_rate, potential_rate, success_probability, _ = (
            reasoner.world_model.predicted_leg(
                env,
                ue_id,
                ue.primary_cell,
                0,
                required_rate,
                available,
            )
        )
        service_goodput, potential_goodput = (
            reasoner.world_model.predicted_goodputs(
                service_rate,
                potential_rate,
                success_probability,
                preserve_reliability_headroom=True,
            )
        )
        service_deficit = max(0.0, required_rate - service_goodput) / (
            required_rate + reasoner.epsilon
        )
        samples.append(
            (
                required_rate,
                potential_goodput < required_rate,
                float(service_deficit),
            )
        )
    return samples


@dataclass
class EpisodeAccumulator:
    num_agents: int
    interval_s: float
    requested_actions: dict[str, int] = field(
        default_factory=lambda: {
            "STAY": 0,
            "PREP": 0,
            "COMMIT": 0,
            "RELEASE": 0,
        }
    )
    control_counters: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    state_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    queue_samples: list[float] = field(default_factory=list)
    hol_samples: list[float] = field(default_factory=list)
    hol_deadline_samples: list[float] = field(default_factory=list)
    required_rate_samples: list[float] = field(default_factory=list)
    potential_insufficient_samples: list[float] = field(default_factory=list)
    service_deficit_samples: list[float] = field(default_factory=list)
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
    handover_completion_times: list[list[float]] = field(init=False)
    steps: int = 0

    def __post_init__(self) -> None:
        self.handover_completion_times = [
            [] for _ in range(self.num_agents)
        ]

    def record_service_samples(
        self,
        samples: list[tuple[float, bool, float]],
    ) -> None:
        for required_rate, insufficient, deficit in samples:
            self.required_rate_samples.append(float(required_rate))
            self.potential_insufficient_samples.append(float(insufficient))
            self.service_deficit_samples.append(float(deficit))

    def record_preparation_contention(
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

    def record_step(
        self,
        env: LEOMobilityEnv,
        actions: np.ndarray,
        output: StepOutput,
        contention: PreparationContentionSnapshot | None = None,
    ) -> None:
        self.steps += 1
        if contention is not None:
            self.record_preparation_contention(
                contention,
                getattr(env, "last_preparation_available_slots", None),
            )
        for action in np.asarray(actions, dtype=np.int64):
            value = int(action)
            if value == 0:
                self.requested_actions["STAY"] += 1
            elif env.prepare_start <= value < env.prepare_start + env.candidate_count:
                self.requested_actions["PREP"] += 1
            elif value == env.commit_action:
                self.requested_actions["COMMIT"] += 1
            elif value == env.release_action:
                self.requested_actions["RELEASE"] += 1
            else:
                raise RuntimeError(f"controller emitted unknown action {value}")

        for key in CANONICAL_CONTROL_COUNTERS:
            self.control_counters[key] += float(output.info.get(key, 0.0))

        completion_time = float(env.time_s - env.interval_s)
        if output.agent_outcomes is not None:
            completed = np.flatnonzero(output.agent_outcomes[:, 0] > 0.0)
            for ue_id in completed:
                self.handover_completion_times[int(ue_id)].append(completion_time)

        for ue in env.ues:
            self.state_counts[ue.state.name] += 1
            self.queue_samples.append(float(ue.queue_packets))
            self.hol_samples.append(float(ue.head_of_line_delay_s))
            ratio = (
                ue.head_of_line_delay_s / ue.deadline_s
                if ue.deadline_s > 0.0
                else 0.0
            )
            self.hol_deadline_samples.append(float(ratio))

    def preparation_contention_metrics(self) -> dict[str, float | int]:
        return {
            "active_prep_epochs": self.active_prep_epochs,
            "diagnostic_prep_request_count": self.prep_request_total,
            "same_target_collision_prep_request_count": (
                self.prep_collision_request_total
            ),
            "mean_prep_requests_per_active_prep_epoch": _safe_ratio(
                float(self.prep_request_total), float(self.active_prep_epochs)
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
                float(self.prep_collision_request_total),
                float(self.prep_request_total),
            ),
            "mean_same_target_collision_satellites_per_epoch": _safe_ratio(
                float(self.prep_collision_target_total), float(self.steps)
            ),
            "prep_admission_capacity_observed_epochs": (
                self.prep_capacity_observed_epochs
            ),
            "mean_available_prep_admissions_on_requested_targets": _safe_ratio(
                float(self.prep_available_slots_total),
                float(self.prep_capacity_observed_epochs),
            ),
            "mean_requested_minus_available_prep_capacity": _safe_ratio(
                float(self.prep_requested_minus_available_total),
                float(self.prep_capacity_observed_epochs),
            ),
            "mean_excess_prep_requests": _safe_ratio(
                float(self.prep_excess_request_total), float(self.steps)
            ),
            "total_excess_prep_requests": self.prep_excess_request_total,
            "fraction_prep_requests_to_oversubscribed_targets": _safe_ratio(
                float(self.prep_oversubscribed_request_total),
                float(self.prep_request_total),
            ),
            "mean_oversubscribed_prep_targets_per_epoch": _safe_ratio(
                float(self.prep_oversubscribed_target_total), float(self.steps)
            ),
        }

    def finalize(
        self,
        env: LEOMobilityEnv,
        episode: int,
        episode_seed: int,
    ) -> dict[str, Any]:
        generated = float(sum(ue.generated_packets for ue in env.ues))
        delivered = float(sum(ue.delivered_packets for ue in env.ues))
        deadline_expired = float(sum(ue.missed_packets for ue in env.ues))
        overflow = float(sum(ue.overflow_packets for ue in env.ues))
        dropped = deadline_expired + overflow
        duration_s = float(self.steps * self.interval_s)
        delivered_bits = float(
            sum(
                ue.delivered_packets * ue.packet_size_bits
                for ue in env.ues
            )
        )

        handovers_per_ue = [
            len(times) for times in self.handover_completion_times
        ]
        inter_handover_times = [
            right - left
            for times in self.handover_completion_times
            for left, right in zip(times, times[1:])
        ]
        target_serving_durations = [
            duration
            for times in self.handover_completion_times
            for duration in (
                [right - left for left, right in zip(times, times[1:])]
                + ([duration_s - times[-1]] if times else [])
            )
        ]
        total_state_samples = self.steps * self.num_agents

        row: dict[str, Any] = {
            "episode": episode,
            "episode_seed": episode_seed,
            "steps": self.steps,
            "duration_s": duration_s,
            "generated_packets": generated,
            "delivered_packets": delivered,
            "dropped_packets": dropped,
            "deadline_expired_packets": deadline_expired,
            "overflow_dropped_packets": overflow,
            "packet_delivery_ratio": _safe_ratio(delivered, generated),
            "packet_drop_rate": _safe_ratio(dropped, generated),
            "deadline_miss_rate": _safe_ratio(deadline_expired, generated),
            "delivered_goodput_bps": _safe_ratio(delivered_bits, duration_s),
            "delivered_throughput_bps": _safe_ratio(
                delivered_bits, duration_s
            ),
            "mean_queue_packets": _mean(self.queue_samples),
            "p95_queue_packets": _p95(self.queue_samples),
            "mean_hol_delay_s": _mean(self.hol_samples),
            "p95_hol_delay_s": _p95(self.hol_samples),
            "mean_hol_deadline_ratio": _mean(self.hol_deadline_samples),
            "fraction_hol_deadline_at_least_one": _mean(
                [float(value >= 1.0) for value in self.hol_deadline_samples]
            ),
            "requested_stay_actions": self.requested_actions["STAY"],
            "requested_prep_actions": self.requested_actions["PREP"],
            "requested_commit_actions": self.requested_actions["COMMIT"],
            "requested_release_actions": self.requested_actions["RELEASE"],
            "successful_prep_transitions": self.control_counters[
                "preparation_admitted"
            ],
            "rejected_prep_requests": self.control_counters[
                "preparation_rejected"
            ],
            "prep_rejection_rate": prep_rejection_rate(
                self.control_counters["preparation_rejected"],
                float(self.requested_actions["PREP"]),
            ),
            "prepared_link_losses": self.control_counters[
                "secondary_link_lost"
            ],
            "successful_commit_starts": self.control_counters[
                "handover_started"
            ],
            "completed_handovers": self.control_counters["handover_success"],
            "failed_handovers": self.control_counters["handover_failure"],
            "release_transitions": self.control_counters["release_count"],
            "pingpong_events": self.control_counters["pingpong"],
            "handovers_per_ue": handovers_per_ue,
            "mean_handovers_per_ue": _mean(
                [float(value) for value in handovers_per_ue]
            ),
            "fraction_ues_with_handover": _mean(
                [float(value > 0) for value in handovers_per_ue]
            ),
            "mean_time_between_completed_handovers_s": _mean(
                inter_handover_times
            ),
            "mean_target_serving_duration_after_handover_s": _mean(
                target_serving_durations
            ),
            "active_service_prediction_count": len(
                self.required_rate_samples
            ),
            "mean_required_rate_bps": _mean(self.required_rate_samples),
            "p95_required_rate_bps": _p95(self.required_rate_samples),
            "fraction_potential_service_insufficient": _mean(
                self.potential_insufficient_samples
            ),
            "mean_primary_service_deficit": _mean(
                self.service_deficit_samples
            ),
            **self.preparation_contention_metrics(),
        }
        for state in PROTOCOL_STATES:
            count = self.state_counts[state]
            name = state.lower()
            row[f"time_in_{name}_ue_seconds"] = count * self.interval_s
            row[f"fraction_time_in_{name}"] = _safe_ratio(
                float(count), float(total_state_samples)
            )
        return row


TRAFFIC_QOS_FIELDS = (
    "generated_packets",
    "delivered_packets",
    "dropped_packets",
    "deadline_expired_packets",
    "overflow_dropped_packets",
    "packet_delivery_ratio",
    "packet_drop_rate",
    "deadline_miss_rate",
    "delivered_goodput_bps",
    "delivered_throughput_bps",
)
QUEUE_DELAY_FIELDS = (
    "mean_queue_packets",
    "p95_queue_packets",
    "mean_hol_delay_s",
    "p95_hol_delay_s",
    "mean_hol_deadline_ratio",
    "fraction_hol_deadline_at_least_one",
)
HANDOVER_CONTROL_FIELDS = (
    "requested_stay_actions",
    "requested_prep_actions",
    "requested_commit_actions",
    "requested_release_actions",
    "successful_prep_transitions",
    "rejected_prep_requests",
    "prepared_link_losses",
    "successful_commit_starts",
    "completed_handovers",
    "failed_handovers",
    "release_transitions",
    "pingpong_events",
    "handovers_per_ue",
    "mean_handovers_per_ue",
    "fraction_ues_with_handover",
    "mean_time_between_completed_handovers_s",
    "mean_target_serving_duration_after_handover_s",
    "prep_rejection_rate",
)
PROTOCOL_STATE_FIELDS = tuple(
    field_name
    for state in PROTOCOL_STATES
    for field_name in (
        f"time_in_{state.lower()}_ue_seconds",
        f"fraction_time_in_{state.lower()}",
    )
)
SERVICE_DEMAND_FIELDS = (
    "active_service_prediction_count",
    "mean_required_rate_bps",
    "p95_required_rate_bps",
    "fraction_potential_service_insufficient",
    "mean_primary_service_deficit",
)
PREPARATION_CONTENTION_FIELDS = (
    "active_prep_epochs",
    "diagnostic_prep_request_count",
    "same_target_collision_prep_request_count",
    "mean_prep_requests_per_active_prep_epoch",
    "mean_unique_prep_targets_per_active_prep_epoch",
    "mean_max_same_target_prep_requests",
    "p95_max_same_target_prep_requests",
    "fraction_prep_requests_in_same_target_collision",
    "mean_same_target_collision_satellites_per_epoch",
    "prep_admission_capacity_observed_epochs",
    "mean_available_prep_admissions_on_requested_targets",
    "mean_requested_minus_available_prep_capacity",
    "mean_excess_prep_requests",
    "total_excess_prep_requests",
    "fraction_prep_requests_to_oversubscribed_targets",
    "mean_oversubscribed_prep_targets_per_epoch",
)


def evaluate_episode(
    env: LEOMobilityEnv,
    reasoner: LookaheadReasoner,
    episode: int,
    episode_seed: int,
) -> dict[str, Any]:
    env.reset(seed=episode_seed)
    accumulator = EpisodeAccumulator(env.num_agents, env.interval_s)
    done = False
    while not done:
        accumulator.record_service_samples(
            _current_primary_service_samples(env, reasoner)
        )
        contention: list[PreparationContentionSnapshot] = []
        _, actions, _, output = geometry_controller_step(
            env,
            reasoner,
            before_step=lambda current_env, selected_actions: contention.append(
                analyze_preparation_contention(current_env, selected_actions)
            ),
        )
        if len(contention) != 1:
            raise RuntimeError("controller did not capture PREP contention once")
        accumulator.record_step(env, actions, output, contention[0])
        done = bool(output.done)
    return accumulator.finalize(env, episode, episode_seed)


def _numeric_aggregate(
    episode_rows: list[dict[str, Any]],
    field_name: str,
) -> dict[str, float]:
    values = np.asarray(
        [float(row[field_name]) for row in episode_rows], dtype=np.float64
    )
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _summary_section(
    episode_rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    per_episode = [
        {
            "episode": row["episode"],
            "episode_seed": row["episode_seed"],
            **{field: row[field] for field in fields},
        }
        for row in episode_rows
    ]
    numeric_fields = [
        field
        for field in fields
        if all(isinstance(row[field], (int, float)) for row in episode_rows)
    ]
    return {
        "per_episode": per_episode,
        "aggregate": {
            field: _numeric_aggregate(episode_rows, field)
            for field in numeric_fields
        },
    }


def build_summary(
    episode_rows: list[dict[str, Any]],
    config_paths: list[str],
    seed: int,
    reasoner: LookaheadReasoner,
    required_rate_model: str,
    world_model: str = "geometry",
    residual_model: str | Path | None = None,
) -> dict[str, Any]:
    scientific_control: dict[str, Any] = {
        "configs": config_paths,
        "seed": seed,
        "episodes": len(episode_rows),
        "episode_seed_progression": "seed + episode_index",
        "controller": CONTROLLER_NAME,
        "world_model": world_model,
        "planner_objective": reasoner.planner_objective,
        "required_rate_model": required_rate_model,
        "reasoner_controls_environment": True,
        "use_coordination": reasoner.use_coordination,
        "use_constraint": reasoner.use_constraint,
        "outage_dual_mu": 0.0,
        "control_abstraction": (
            "3GPP-inspired system-level CHO abstraction; COMMIT = "
            "EXECUTE_CHO and COOLDOWN = POST_HO_GUARD, not literal "
            "message-level RRC states/actions"
        ),
    }
    if world_model == "hybrid":
        if residual_model is None:
            raise ValueError("hybrid summary requires a residual model path")
        if not isinstance(reasoner.world_model, HybridWorldModel):
            raise RuntimeError(
                "hybrid summary requires a reasoner with HybridWorldModel"
            )
        checkpoint = Path(residual_model).expanduser().resolve()
        scientific_control.update(
            {
                "residual_model_path": str(checkpoint),
                "residual_model_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "residual_history_length": reasoner.world_model.history_steps,
                "residual_prediction_horizon": (
                    reasoner.world_model.horizon_steps
                ),
            }
        )
    return {
        "scientific_control": scientific_control,
        "traffic_qos": _summary_section(episode_rows, TRAFFIC_QOS_FIELDS),
        "queue_delay": _summary_section(episode_rows, QUEUE_DELAY_FIELDS),
        "handover_control": _summary_section(
            episode_rows, HANDOVER_CONTROL_FIELDS
        ),
        "protocol_state_occupancy": _summary_section(
            episode_rows, PROTOCOL_STATE_FIELDS
        ),
        "service_demand": _summary_section(
            episode_rows, SERVICE_DEMAND_FIELDS
        ),
        "preparation_contention": _summary_section(
            episode_rows, PREPARATION_CONTENTION_FIELDS
        ),
    }


def write_outputs(
    output: str | Path,
    episode_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for row in episode_rows:
        csv_row = dict(row)
        csv_row["handovers_per_ue"] = json.dumps(
            row["handovers_per_ue"], separators=(",", ":")
        )
        csv_rows.append(csv_row)
    with (destination / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    config = load_evaluation_config(args.config, args.override, args.seed)
    env = LEOMobilityEnv(config)
    reasoner = build_predictive_reasoner(
        config,
        env,
        world_model=args.world_model,
        residual_model=args.residual_model,
    )
    episode_rows = [
        evaluate_episode(env, reasoner, episode, args.seed + episode)
        for episode in range(args.episodes)
    ]
    config_paths = [str(args.config), *(str(path) for path in args.override)]
    summary = build_summary(
        episode_rows,
        config_paths,
        args.seed,
        reasoner,
        env.required_rate_model,
        world_model=args.world_model,
        residual_model=args.residual_model,
    )
    write_outputs(args.output, episode_rows, summary)
    traffic = summary["traffic_qos"]["aggregate"]
    control = summary["handover_control"]["aggregate"]
    print(
        f"packet_delivery_ratio={traffic['packet_delivery_ratio']['mean']:.6f} "
        f"deadline_miss_rate={traffic['deadline_miss_rate']['mean']:.6f} "
        f"completed_handovers={control['completed_handovers']['mean']:.3f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a predictive counterfactual planner as a closed-loop "
            "diagnostic controller."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="YAML override path; repeat to compose overrides in order.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--world-model",
        choices=("geometry", "hybrid"),
        default="geometry",
    )
    parser.add_argument(
        "--residual-model",
        help="ResidualLinkPredictor checkpoint required for hybrid mode.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
