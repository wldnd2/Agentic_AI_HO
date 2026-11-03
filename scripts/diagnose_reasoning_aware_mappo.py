from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO
from leo_mappo.evaluation_diagnostics import (
    PolicyEvaluationDiagnostics,
    analyze_preparation_contention,
)
from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.policy_diagnostics import (
    MatchedReasoningDiagnostics,
    PolicyProbabilityDiagnostics,
    ReasoningFeatureDiagnostics,
    analyze_matched_reasoners,
    intervene_reasoning_features,
    model_parameter_digest,
    select_policy_actions,
)
from leo_mappo.reasoning import LookaheadReasoner, build_actor_reasoning_features
from leo_mappo.runner import evaluation_episode_seeds, summarize_metrics


CHECKPOINTS = (
    ("iter_0025", "checkpoints/mappo_weights_iter_0025.npz"),
    ("iter_0050", "checkpoints/mappo_weights_iter_0050.npz"),
    ("iter_0075", "checkpoints/mappo_weights_iter_0075.npz"),
    ("final_0100", "mappo_weights_final.npz"),
)
STOCHASTIC_ACTION_SEEDS = (7007, 7008, 7009)
COMPARISON_FIELDS = (
    "packet_delivery_ratio",
    "deadline_miss_rate",
    "team_reward",
    "requested_prep_actions",
    "successful_prep_transitions",
    "rejected_prep_requests",
    "prep_rejection_rate",
    "requested_commit_actions",
    "completed_handovers",
    "requested_release_actions",
    "pingpong_events",
    "fraction_prep_requests_in_same_target_collision",
    "fraction_prep_requests_to_oversubscribed_targets",
    "total_excess_prep_requests",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object in {path}")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_json(run_dir / "checkpoint_metadata.json")
    config = metadata.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint metadata has no resolved_config mapping")
    if metadata.get("history_recurrent_mode") != "recurrent":
        raise ValueError("diagnostic requires a recurrent checkpoint")
    if metadata.get("coordination_mode") != "none":
        raise ValueError("diagnostic requires coordination_mode=none")
    if bool(config.get("mappo", {}).get("use_constraint", False)):
        raise ValueError("diagnostic requires reliability constraint disabled")
    return deepcopy(config), metadata


def _build_agent(env: LEOMobilityEnv, config: dict[str, Any], weights: Path) -> MAPPO:
    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        config["mappo"],
        int(config.get("agent_seed", config["seed"])),
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )
    agent.load(weights)
    return agent


def _build_geometry_reasoner(
    env: LEOMobilityEnv, config: dict[str, Any]
) -> LookaheadReasoner:
    reasoning_config = deepcopy(config.get("reasoning", {}))
    reasoning_config["world_model"] = "geometry"
    reasoning_config.pop("residual_model_path", None)
    return LookaheadReasoner(
        reasoning_config,
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=False,
        use_constraint=False,
    )


def _build_primary_reasoner(
    env: LEOMobilityEnv, config: dict[str, Any]
) -> LookaheadReasoner:
    return LookaheadReasoner(
        config.get("reasoning", {}),
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=False,
        use_constraint=False,
    )


def _policy_row_value(
    rows: list[dict[str, Any]], state: str, metric: str
) -> float:
    for row in rows:
        if (
            row["protocol_state"] == state
            and row["following_prep_outcome"] == "all"
            and row["metric"] == metric
        ):
            return float(row["mean"])
    return 0.0


def _checkpoint_equivalence(run_dir: Path) -> dict[str, Any]:
    periodic = run_dir / "checkpoints/mappo_weights_iter_0100.npz"
    final = run_dir / "mappo_weights_final.npz"
    with np.load(periodic) as periodic_data, np.load(final) as final_data:
        arrays_identical = set(periodic_data.files) == set(final_data.files) and all(
            np.array_equal(periodic_data[name], final_data[name])
            for name in periodic_data.files
        )
    return {
        "iteration_100_path": str(periodic.resolve()),
        "final_path": str(final.resolve()),
        "iteration_100_sha256": _sha256(periodic),
        "final_sha256": _sha256(final),
        "arrays_identical": arrays_identical,
    }


def evaluate_frozen_checkpoint(
    config: dict[str, Any],
    weights: Path,
    output_dir: Path,
    *,
    deterministic: bool,
    action_seed: int | None,
    episode_seeds: tuple[int, ...],
    reasoning_input: str = "normal",
) -> dict[str, Any]:
    env = LEOMobilityEnv(config)
    agent = _build_agent(env, config, weights)
    if not agent.use_memory or not agent.use_reasoning:
        raise ValueError("reasoning diagnostic requires recurrent reasoning MAPPO")
    if agent.reasoning_feature_mode != "confidence_aware":
        raise ValueError("reasoning diagnostic requires confidence-aware features")
    if action_seed is not None:
        agent.rng = np.random.default_rng(int(action_seed))
    primary_reasoner = _build_primary_reasoner(env, config)
    primary_is_hybrid = isinstance(primary_reasoner.world_model, HybridWorldModel)
    geometry_reasoner = (
        _build_geometry_reasoner(env, config) if primary_is_hybrid else None
    )
    if reasoning_input == "geometry" and geometry_reasoner is None:
        raise ValueError("Geometry intervention requires a Hybrid checkpoint")
    if reasoning_input not in ("normal", "geometry", "zero"):
        raise ValueError("reasoning_input must be normal, geometry, or zero")

    parameter_digest_before = model_parameter_digest(agent)
    totals: dict[str, float] = {}
    evaluation = PolicyEvaluationDiagnostics(
        interval_s=env.interval_s,
        packet_size_bits=int(env.traffic_cfg["packet_size_bits"]),
    )
    policy_diagnostics = PolicyProbabilityDiagnostics()
    feature_diagnostics = ReasoningFeatureDiagnostics()
    matched_diagnostics = MatchedReasoningDiagnostics()
    matched_hybrid_features = ReasoningFeatureDiagnostics()
    matched_geometry_features = ReasoningFeatureDiagnostics()
    lifecycle_rows: list[dict[str, Any]] = []
    total_steps = 0

    for episode_seed in episode_seeds:
        observations, _, masks = env.reset(seed=episode_seed)
        hidden_states = np.zeros(
            (env.num_agents, agent.recurrent_hidden_dim), dtype=np.float64
        )
        previous_actions = np.full(env.num_agents, -1, dtype=np.int64)
        previous_outcomes = np.zeros(
            (env.num_agents, agent.outcome_dim), dtype=np.float64
        )
        following_prep_outcome: list[str | None] = [None] * env.num_agents
        done = False
        episode_step = 0
        while not done:
            history_entries_before = (
                sum(
                    len(history)
                    for history in primary_reasoner.world_model._histories.values()
                )
                if primary_is_hybrid
                else 0
            )
            if primary_is_hybrid:
                assert geometry_reasoner is not None
                (
                    primary_result,
                    primary_features,
                    geometry_result,
                    geometry_features,
                ) = analyze_matched_reasoners(
                    env,
                    masks,
                    primary_reasoner,
                    geometry_reasoner,
                )
                matched_diagnostics.record(
                    env,
                    masks,
                    episode_seed,
                    episode_step,
                    primary_result,
                    primary_features,
                    geometry_result,
                    geometry_features,
                )
                matched_hybrid_features.record(env, masks, primary_features)
                matched_geometry_features.record(env, masks, geometry_features)
            else:
                primary_result = primary_reasoner.analyze(env, masks)
                primary_features = build_actor_reasoning_features(
                    primary_result,
                    masks,
                    primary_reasoner.score_clip,
                    agent.reasoning_feature_mode,
                )
                geometry_features = primary_features
            actor_reasoning = intervene_reasoning_features(
                primary_features,
                geometry_features,
                reasoning_input,
                agent.reasoning_feature_dim,
            )

            logits, probabilities, new_hidden = agent.recurrent_policy(
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                actor_reasoning,
            )
            policy_diagnostics.record(
                env,
                masks,
                logits,
                probabilities,
                following_prep_outcome,
            )
            feature_diagnostics.record(env, masks, actor_reasoning)
            actions = select_policy_actions(
                agent,
                probabilities,
                deterministic,
            )
            actions_before_diagnostics = actions.copy()
            contention = analyze_preparation_contention(env, actions)
            np.testing.assert_array_equal(actions, actions_before_diagnostics)
            output = env.step(actions)
            evaluation.record_step(env, actions, output, contention)
            for key, value in output.info.items():
                totals[key] = totals.get(key, 0.0) + float(value)

            next_following: list[str | None] = [None] * env.num_agents
            for ue_id, action in enumerate(actions):
                if env.prepare_start <= action < env.prepare_start + env.candidate_count:
                    next_following[ue_id] = (
                        "admitted"
                        if env.ues[ue_id].state.name == "PREPARED"
                        else "rejected"
                    )
            following_prep_outcome = next_following
            hidden_states = new_hidden
            previous_actions = actions.copy()
            if output.agent_outcomes is None:
                raise RuntimeError("environment did not return recurrent outcomes")
            previous_outcomes = output.agent_outcomes.copy()
            observations, masks, done = (
                output.observations,
                output.action_masks,
                output.done,
            )
            if primary_is_hybrid and episode_step == 0:
                history_entries_after = sum(
                    len(history)
                    for history in primary_reasoner.world_model._histories.values()
                )
                lifecycle_rows.append(
                    {
                        "episode_seed": episode_seed,
                        "history_entries_before_first_observe": history_entries_before,
                        "history_entries_after_first_observe": history_entries_after,
                        "expected_entries_after_first_observe": (
                            env.num_agents * env.num_cells
                        ),
                    }
                )
            total_steps += 1
            episode_step += 1

    metrics = summarize_metrics(totals, total_steps)
    metrics.update(evaluation.metrics())
    policy_rows = policy_diagnostics.rows()
    feature_rows = feature_diagnostics.rows()
    parameter_digest_after = model_parameter_digest(agent)
    if parameter_digest_before != parameter_digest_after:
        raise RuntimeError("diagnostic evaluation mutated model parameters")
    metrics.update(
        {
            "single_probability_stay": _policy_row_value(
                policy_rows, "SINGLE", "probability_stay"
            ),
            "single_probability_all_prep": _policy_row_value(
                policy_rows, "SINGLE", "probability_all_prep"
            ),
            "single_probability_highest_prep": _policy_row_value(
                policy_rows, "SINGLE", "probability_highest_prep"
            ),
            "single_argmax_prep_fraction": _policy_row_value(
                policy_rows, "SINGLE", "argmax_is_prep"
            ),
            "single_policy_entropy": _policy_row_value(
                policy_rows, "SINGLE", "policy_entropy"
            ),
            "single_top1_top2_probability_margin": _policy_row_value(
                policy_rows, "SINGLE", "top1_minus_top2_probability_margin"
            ),
        }
    )
    provenance = {
        "checkpoint_path": str(weights.resolve()),
        "checkpoint_sha256": _sha256(weights),
        "deterministic": deterministic,
        "action_seed": action_seed,
        "evaluation_episode_seeds": list(episode_seeds),
        "reasoning_input": reasoning_input,
        "reasoning_input_is_out_of_training_distribution": reasoning_input
        in ("geometry", "zero"),
        "world_model": "hybrid" if primary_is_hybrid else "geometry",
        "parameter_digest_before": parameter_digest_before,
        "parameter_digest_after": parameter_digest_after,
        "parameters_unchanged": True,
        "world_model_diagnostics": (
            primary_reasoner.world_model.diagnostics if primary_is_hybrid else {}
        ),
    }
    _write_json(output_dir / "evaluation.json", metrics)
    _write_json(output_dir / "provenance.json", provenance)
    _write_csv(output_dir / "policy_probability_summary.csv", policy_rows)
    _write_csv(output_dir / "reasoning_feature_summary.csv", feature_rows)
    _write_csv(output_dir / "world_model_lifecycle.csv", lifecycle_rows)
    if primary_is_hybrid:
        _write_csv(
            output_dir / "matched_geometry_hybrid_decisions.csv",
            matched_diagnostics.rows_data,
        )
        _write_json(
            output_dir / "matched_geometry_hybrid_summary.json",
            matched_diagnostics.summary(),
        )
        matched_feature_rows = [
            {"world_model": "hybrid", **row}
            for row in matched_hybrid_features.rows()
        ] + [
            {"world_model": "geometry", **row}
            for row in matched_geometry_features.rows()
        ]
        _write_csv(
            output_dir / "matched_geometry_hybrid_feature_summary.csv",
            matched_feature_rows,
        )
    return {**provenance, **metrics}


def run_checkpoint_series(args: argparse.Namespace) -> None:
    config, metadata = _load_run_config(args.run_dir)
    expected_world_model = args.world_model
    configured_world_model = (
        str(config.get("reasoning", {}).get("world_model", "geometry")).lower()
    )
    if configured_world_model != expected_world_model:
        raise ValueError(
            f"run world model is {configured_world_model}, not {expected_world_model}"
        )
    episode_seeds = evaluation_episode_seeds(10, 5001)
    combined: list[dict[str, Any]] = []
    for checkpoint_label, relative_path in CHECKPOINTS:
        weights = args.run_dir / relative_path
        evaluations = [("deterministic", True, None)] + [
            (f"stochastic_seed{seed}", False, seed)
            for seed in STOCHASTIC_ACTION_SEEDS
        ]
        for evaluation_label, deterministic, action_seed in evaluations:
            run_output = args.output / checkpoint_label / evaluation_label
            result = evaluate_frozen_checkpoint(
                config,
                weights,
                run_output,
                deterministic=deterministic,
                action_seed=action_seed,
                episode_seeds=episode_seeds,
            )
            combined.append(
                {
                    "world_model": expected_world_model,
                    "checkpoint": checkpoint_label,
                    "evaluation_mode": evaluation_label,
                    "reasoning_input": "normal",
                    **{field: result[field] for field in COMPARISON_FIELDS},
                    "single_probability_stay": result["single_probability_stay"],
                    "single_probability_all_prep": result[
                        "single_probability_all_prep"
                    ],
                    "single_probability_highest_prep": result[
                        "single_probability_highest_prep"
                    ],
                    "single_argmax_prep_fraction": result[
                        "single_argmax_prep_fraction"
                    ],
                    "single_policy_entropy": result["single_policy_entropy"],
                    "single_top1_top2_probability_margin": result[
                        "single_top1_top2_probability_margin"
                    ],
                }
            )
    _write_csv(args.output / "checkpoint_diagnostics.csv", combined)
    _write_json(
        args.output / "series_provenance.json",
        {
            "source_run": str(args.run_dir.resolve()),
            "source_metadata": metadata,
            "evaluation_episode_seeds": list(episode_seeds),
            "stochastic_action_seeds": list(STOCHASTIC_ACTION_SEEDS),
            "iteration_100_final_equivalence": _checkpoint_equivalence(args.run_dir),
        },
    )


def run_interventions(args: argparse.Namespace) -> None:
    config, metadata = _load_run_config(args.run_dir)
    if str(config.get("reasoning", {}).get("world_model", "geometry")).lower() != "hybrid":
        raise ValueError("input interventions require the M3-H run")
    weights = args.run_dir / "mappo_weights_final.npz"
    episode_seeds = evaluation_episode_seeds(10, 5001)
    combined: list[dict[str, Any]] = []
    for reasoning_input in ("normal", "geometry", "zero"):
        result = evaluate_frozen_checkpoint(
            config,
            weights,
            args.output / reasoning_input,
            deterministic=True,
            action_seed=None,
            episode_seeds=episode_seeds,
            reasoning_input=reasoning_input,
        )
        combined.append(
            {
                "world_model": "hybrid_actor",
                "checkpoint": "final_0100",
                "evaluation_mode": "deterministic",
                "reasoning_input": reasoning_input,
                **{field: result[field] for field in COMPARISON_FIELDS},
                "single_probability_stay": result["single_probability_stay"],
                "single_probability_all_prep": result[
                    "single_probability_all_prep"
                ],
                "single_probability_highest_prep": result[
                    "single_probability_highest_prep"
                ],
                "single_argmax_prep_fraction": result[
                    "single_argmax_prep_fraction"
                ],
                "single_policy_entropy": result["single_policy_entropy"],
                "single_top1_top2_probability_margin": result[
                    "single_top1_top2_probability_margin"
                ],
            }
        )
    _write_csv(args.output / "intervention_diagnostics.csv", combined)
    _write_json(
        args.output / "intervention_provenance.json",
        {
            "source_run": str(args.run_dir.resolve()),
            "source_metadata": metadata,
            "checkpoint": str(weights.resolve()),
            "evaluation_episode_seeds": list(episode_seeds),
            "warning": (
                "geometry and zero are out-of-training-distribution diagnostic "
                "input interventions, not performance baselines"
            ),
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic table does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_comparison(args: argparse.Namespace) -> None:
    geometry = _read_csv(args.geometry / "checkpoint_diagnostics.csv")
    hybrid = _read_csv(args.hybrid / "checkpoint_diagnostics.csv")
    interventions = _read_csv(args.interventions / "intervention_diagnostics.csv")
    rows = geometry + hybrid + interventions
    _write_csv(args.output, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen reasoning-aware MAPPO collapse diagnostics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    checkpoints = subparsers.add_parser("checkpoints")
    checkpoints.add_argument("--run-dir", type=Path, required=True)
    checkpoints.add_argument(
        "--world-model", choices=("geometry", "hybrid"), required=True
    )
    checkpoints.add_argument("--output", type=Path, required=True)
    interventions = subparsers.add_parser("interventions")
    interventions.add_argument("--run-dir", type=Path, required=True)
    interventions.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--geometry", type=Path, required=True)
    compare.add_argument("--hybrid", type=Path, required=True)
    compare.add_argument("--interventions", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "checkpoints":
        run_checkpoint_series(args)
    elif args.command == "interventions":
        run_interventions(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
