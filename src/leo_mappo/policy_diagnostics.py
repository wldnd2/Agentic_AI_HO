from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .hybrid_world_model import HybridWorldModel
from .reasoning import LookaheadReasoner, build_actor_reasoning_features
from .ue import MobilityState

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv
    from .mappo import MAPPO
    from .planner import ReasoningResult


def _summary(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _action_category(env: LEOMobilityEnv, action: int) -> str:
    if action == 0:
        return "STAY"
    if env.prepare_start <= action < env.prepare_start + env.candidate_count:
        return "PREP"
    if action == env.commit_action:
        return "COMMIT"
    if action == env.release_action:
        return "RELEASE"
    return "UNKNOWN"


def _action_label(env: LEOMobilityEnv, action: int) -> str:
    category = _action_category(env, action)
    if category == "PREP":
        return f"PREP_{action - env.prepare_start}"
    return category


def select_policy_actions(
    agent: MAPPO,
    probabilities: np.ndarray,
    deterministic: bool,
) -> np.ndarray:
    """Apply the production argmax/sampling rule without changing probabilities."""
    if deterministic:
        return probabilities.argmax(axis=-1).astype(np.int64)
    return np.asarray(
        [agent.rng.choice(agent.action_dim, p=row) for row in probabilities],
        dtype=np.int64,
    )


def intervene_reasoning_features(
    normal_features: np.ndarray,
    geometry_features: np.ndarray,
    mode: str,
    expected_dim: int,
) -> np.ndarray:
    """Select a frozen-actor diagnostic input while preserving its exact width."""
    if normal_features.shape != geometry_features.shape:
        raise ValueError("normal and Geometry reasoning feature shapes differ")
    if mode == "normal":
        selected = normal_features
    elif mode == "geometry":
        selected = geometry_features
    elif mode == "zero":
        selected = np.zeros_like(normal_features)
    else:
        raise ValueError("reasoning intervention must be normal, geometry, or zero")
    if selected.ndim != 2 or selected.shape[1] != int(expected_dim):
        raise ValueError("reasoning intervention changed actor input dimensionality")
    return selected.copy()


@dataclass
class PolicyProbabilityDiagnostics:
    values: dict[tuple[str, str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(
        self,
        env: LEOMobilityEnv,
        masks: np.ndarray,
        logits: np.ndarray,
        probabilities: np.ndarray,
        following_prep_outcome: list[str | None],
    ) -> None:
        for ue in env.ues:
            ue_id = int(ue.ue_id)
            feasible = np.flatnonzero(masks[ue_id] > 0.0)
            feasible_probabilities = probabilities[ue_id, feasible]
            feasible_logits = logits[ue_id, feasible]
            order = np.argsort(feasible_probabilities)
            top_probability = float(feasible_probabilities[order[-1]])
            second_probability = (
                float(feasible_probabilities[order[-2]]) if len(order) > 1 else 0.0
            )
            logit_order = np.argsort(feasible_logits)
            top_logit = float(feasible_logits[logit_order[-1]])
            second_logit = (
                float(feasible_logits[logit_order[-2]]) if len(logit_order) > 1 else top_logit
            )
            entropy = float(
                -np.sum(
                    feasible_probabilities
                    * np.log(np.maximum(feasible_probabilities, 1e-12))
                )
            )
            argmax_action = int(feasible[np.argmax(feasible_probabilities)])
            state = ue.state.name
            metrics = {
                "policy_entropy": entropy,
                "top1_minus_top2_probability_margin": (
                    top_probability - second_probability
                ),
                "top1_minus_top2_logit_margin": top_logit - second_logit,
                "feasible_action_count": float(len(feasible)),
            }
            if ue.state == MobilityState.SINGLE:
                prep_actions = feasible[
                    (feasible >= env.prepare_start)
                    & (feasible < env.prepare_start + env.candidate_count)
                ]
                prep_probabilities = probabilities[ue_id, prep_actions]
                metrics.update(
                    {
                        "probability_stay": float(probabilities[ue_id, 0]),
                        "probability_all_prep": float(np.sum(prep_probabilities)),
                        "probability_highest_prep": (
                            float(np.max(prep_probabilities))
                            if len(prep_probabilities)
                            else 0.0
                        ),
                        "argmax_is_prep": float(
                            env.prepare_start
                            <= argmax_action
                            < env.prepare_start + env.candidate_count
                        ),
                    }
                )
            elif ue.state == MobilityState.PREPARED:
                metrics.update(
                    {
                        "probability_stay": float(probabilities[ue_id, 0]),
                        "probability_commit": float(
                            probabilities[ue_id, env.commit_action]
                        ),
                        "probability_release": float(
                            probabilities[ue_id, env.release_action]
                        ),
                        "argmax_is_stay": float(argmax_action == 0),
                        "argmax_is_commit": float(
                            argmax_action == env.commit_action
                        ),
                        "argmax_is_release": float(
                            argmax_action == env.release_action
                        ),
                    }
                )
            cohorts = ["all"]
            if following_prep_outcome[ue_id] is not None:
                cohorts.append(str(following_prep_outcome[ue_id]))
            for cohort in cohorts:
                for metric, value in metrics.items():
                    self.values[(state, cohort, metric)].append(value)

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for (state, cohort, metric), values in sorted(self.values.items()):
            rows.append(
                {
                    "protocol_state": state,
                    "following_prep_outcome": cohort,
                    "metric": metric,
                    **_summary(values),
                }
            )
        return rows


@dataclass
class ReasoningFeatureDiagnostics:
    values: dict[tuple[str, str, str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(
        self,
        env: LEOMobilityEnv,
        masks: np.ndarray,
        actor_features: np.ndarray,
    ) -> None:
        action_dim = env.action_dim
        expected_dim = 2 * action_dim + 1
        if actor_features.shape != (env.num_agents, expected_dim):
            raise ValueError(
                "confidence-aware reasoning diagnostics require exact 2A+1 actor input"
            )
        for ue in env.ues:
            ue_id = int(ue.ue_id)
            state = ue.state.name
            feasible = np.flatnonzero(masks[ue_id] > 0.0)
            for action in feasible:
                action_value = int(action)
                key = (
                    state,
                    _action_category(env, action_value),
                    _action_label(env, action_value),
                )
                self.values[(*key, "legacy")].append(
                    float(actor_features[ue_id, action_value])
                )
                self.values[(*key, "best_relative_magnitude")].append(
                    float(actor_features[ue_id, action_dim + action_value])
                )
            self.values[(state, "GLOBAL", "GLOBAL", "confidence_margin")].append(
                float(actor_features[ue_id, -1])
            )

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for (state, category, action, component), values in sorted(
            self.values.items()
        ):
            rows.append(
                {
                    "protocol_state": state,
                    "action_category": category,
                    "action": action,
                    "feature_component": component,
                    **_summary(values),
                }
            )
        return rows


def analyze_matched_reasoners(
    env: LEOMobilityEnv,
    masks: np.ndarray,
    hybrid_reasoner: LookaheadReasoner,
    geometry_reasoner: LookaheadReasoner,
) -> tuple[
    ReasoningResult,
    np.ndarray,
    ReasoningResult,
    np.ndarray,
]:
    if not isinstance(hybrid_reasoner.world_model, HybridWorldModel):
        raise ValueError("matched diagnostic requires a Hybrid primary reasoner")
    hybrid_result = hybrid_reasoner.analyze(env, masks)
    history_lengths_before = {
        key: len(value)
        for key, value in hybrid_reasoner.world_model._histories.items()
    }
    geometry_result = geometry_reasoner.analyze(env, masks)
    history_lengths_after = {
        key: len(value)
        for key, value in hybrid_reasoner.world_model._histories.items()
    }
    if history_lengths_before != history_lengths_after:
        raise RuntimeError("Geometry diagnostic advanced Hybrid history")
    hybrid_features = build_actor_reasoning_features(
        hybrid_result,
        masks,
        hybrid_reasoner.score_clip,
        "confidence_aware",
    )
    geometry_features = build_actor_reasoning_features(
        geometry_result,
        masks,
        geometry_reasoner.score_clip,
        "confidence_aware",
    )
    if hybrid_features.shape != geometry_features.shape:
        raise RuntimeError("Geometry and Hybrid actor feature dimensions differ")
    return hybrid_result, hybrid_features, geometry_result, geometry_features


@dataclass
class MatchedReasoningDiagnostics:
    rows_data: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        env: LEOMobilityEnv,
        masks: np.ndarray,
        episode_seed: int,
        time_index: int,
        hybrid_result: ReasoningResult,
        hybrid_features: np.ndarray,
        geometry_result: ReasoningResult,
        geometry_features: np.ndarray,
    ) -> None:
        for ue in env.ues:
            ue_id = int(ue.ue_id)
            feasible = np.flatnonzero(masks[ue_id] > 0.0)
            hybrid_action = int(
                feasible[np.argmin(hybrid_result.raw_costs[ue_id, feasible])]
            )
            geometry_action = int(
                feasible[np.argmin(geometry_result.raw_costs[ue_id, feasible])]
            )
            prep_actions = feasible[
                (feasible >= env.prepare_start)
                & (feasible < env.prepare_start + env.candidate_count)
            ]

            def prep_values(
                result: ReasoningResult, features: np.ndarray
            ) -> tuple[float, float, float]:
                if len(prep_actions) == 0:
                    return float("nan"), float("nan"), float("nan")
                best_prep = int(
                    prep_actions[np.argmin(result.raw_costs[ue_id, prep_actions])]
                )
                return (
                    float(result.raw_costs[ue_id, best_prep] - result.raw_costs[ue_id, 0]),
                    float(features[ue_id, best_prep]),
                    float(features[ue_id, env.action_dim + best_prep]),
                )

            hybrid_prep_delta, hybrid_prep_legacy, hybrid_prep_magnitude = prep_values(
                hybrid_result, hybrid_features
            )
            geometry_prep_delta, geometry_prep_legacy, geometry_prep_magnitude = prep_values(
                geometry_result, geometry_features
            )
            hybrid_category = _action_category(env, hybrid_action)
            geometry_category = _action_category(env, geometry_action)
            self.rows_data.append(
                {
                    "episode_seed": episode_seed,
                    "time_index": time_index,
                    "ue_id": ue_id,
                    "protocol_state": ue.state.name,
                    "hybrid_preferred_action": hybrid_action,
                    "hybrid_preferred_category": hybrid_category,
                    "geometry_preferred_action": geometry_action,
                    "geometry_preferred_category": geometry_category,
                    "preferred_category_changed": float(
                        hybrid_category != geometry_category
                    ),
                    "hybrid_preferred_raw_cost": float(
                        hybrid_result.raw_costs[ue_id, hybrid_action]
                    ),
                    "geometry_preferred_raw_cost": float(
                        geometry_result.raw_costs[ue_id, geometry_action]
                    ),
                    "hybrid_best_second_margin": float(
                        hybrid_result.score_gaps[ue_id]
                    ),
                    "geometry_best_second_margin": float(
                        geometry_result.score_gaps[ue_id]
                    ),
                    "hybrid_confidence_feature": float(hybrid_features[ue_id, -1]),
                    "geometry_confidence_feature": float(geometry_features[ue_id, -1]),
                    "hybrid_best_prep_minus_stay_raw_cost": hybrid_prep_delta,
                    "geometry_best_prep_minus_stay_raw_cost": geometry_prep_delta,
                    "hybrid_best_prep_legacy_feature": hybrid_prep_legacy,
                    "geometry_best_prep_legacy_feature": geometry_prep_legacy,
                    "hybrid_best_prep_magnitude_feature": hybrid_prep_magnitude,
                    "geometry_best_prep_magnitude_feature": geometry_prep_magnitude,
                }
            )

    def summary(self) -> dict[str, Any]:
        rows = self.rows_data
        if not rows:
            return {"count": 0}
        result: dict[str, Any] = {
            "count": len(rows),
            "preferred_category_change_fraction": float(
                np.mean([row["preferred_category_changed"] for row in rows])
            ),
            "hybrid_preferred_prep_fraction": float(
                np.mean(
                    [row["hybrid_preferred_category"] == "PREP" for row in rows]
                )
            ),
            "geometry_preferred_prep_fraction": float(
                np.mean(
                    [row["geometry_preferred_category"] == "PREP" for row in rows]
                )
            ),
        }
        for field_name in (
            "hybrid_preferred_raw_cost",
            "geometry_preferred_raw_cost",
            "hybrid_best_second_margin",
            "geometry_best_second_margin",
            "hybrid_confidence_feature",
            "geometry_confidence_feature",
            "hybrid_best_prep_minus_stay_raw_cost",
            "geometry_best_prep_minus_stay_raw_cost",
            "hybrid_best_prep_legacy_feature",
            "geometry_best_prep_legacy_feature",
            "hybrid_best_prep_magnitude_feature",
            "geometry_best_prep_magnitude_feature",
        ):
            result[field_name] = _summary(
                [float(row[field_name]) for row in rows]
            )
        return result


def model_parameter_digest(agent: MAPPO) -> str:
    digest = hashlib.sha256()
    parameter_groups = (
        ("actor", agent.actor.parameters),
        ("critic", agent.critic.parameters),
        ("gru", agent.gru.parameters if agent.gru is not None else {}),
    )
    for group_name, parameters in parameter_groups:
        for name in sorted(parameters):
            values = np.asarray(parameters[name])
            digest.update(f"{group_name}:{name}:{values.shape}:{values.dtype}".encode())
            digest.update(values.tobytes(order="C"))
    return digest.hexdigest()
