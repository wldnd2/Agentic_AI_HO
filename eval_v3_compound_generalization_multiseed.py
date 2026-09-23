"""Zero-shot V3 nominal versus compound-stress evaluation for M1, M2, and M4.

This script only loads existing checkpoints and calls the stochastic evaluator. It
does not expose a training path or modify any checkpoint/configuration file.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import numpy as np
import yaml

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import evaluate_policy


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "configs" / "default.yaml"
COMPOUND_OVERRIDE = ROOT / "configs" / "compound_eval.yaml"

COMMON_OVERRIDES = (
    ROOT / "configs" / "validation_agentic_challenge_v3.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
)

MODEL_SPECS = {
    "M1": {
        "override": None,
        "weights": ROOT
        / "outputs"
        / "v3_convergence_anneal_2000"
        / "m1_seed7"
        / "mappo_weights.npz",
    },
    "M2": {
        "override": ROOT / "configs" / "memory_only.yaml",
        "weights": ROOT
        / "outputs"
        / "v3_convergence_anneal_2000"
        / "m2_seed7"
        / "mappo_weights.npz",
    },
    "M4": {
        "override": ROOT / "configs" / "m4_coordination.yaml",
        "weights": ROOT
        / "outputs"
        / "v3_convergence_anneal_2000"
        / "m4_seed7"
        / "mappo_weights.npz",
    },
}

CONDITIONS = ("nominal", "stress")
EVALUATION_SEEDS = (7, 8, 9, 10, 11)
AGENT_SEED = 7
EPISODES = 200
T_95_DF4 = 2.7764451051977987

RAW_CSV = ROOT / "outputs" / "v3_compound_generalization_multiseed_raw.csv"
SUMMARY_CSV = ROOT / "outputs" / "v3_compound_generalization_multiseed_summary.csv"

REQUIRED_METRICS = (
    "outage_ratio",
    "service_deficit",
    "service_ratio_mean",
    "resource_cost",
    "dro",
    "handover_started",
    "handover_failure_rate",
    "pingpong_rate",
    "deadline_violation_rate",
    "preparation_rejection_rate",
    "prepare_to_commit_rate",
    "prepare_to_release_without_commit_rate",
    "duplication_active_time_fraction",
    "team_reward",
)

STRESS_EFFECT_METRICS = (
    "outage_ratio",
    "service_deficit",
    "resource_cost",
    "dro",
    "handover_started",
    "pingpong_rate",
    "duplication_active_time_fraction",
    "team_reward",
)

CSV_METADATA = (
    "model",
    "condition",
    "evaluation_seed",
    "environment_seed",
    "action_seed",
    "agent_seed",
    "episodes",
    "checkpoint",
    "config_overrides",
)

SUMMARY_FIELDS = (
    "comparison",
    "model",
    "condition",
    "metric",
    "n",
    "mean",
    "std",
    "ci95_low",
    "ci95_high",
)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def override_paths(model: str, condition: str) -> tuple[Path, ...]:
    paths = list(COMMON_OVERRIDES)
    model_override = MODEL_SPECS[model]["override"]
    if model_override is not None:
        paths.append(model_override)
    if condition == "stress":
        # This must remain the final override: stress changes the event schedule only.
        paths.append(COMPOUND_OVERRIDE)
    elif condition != "nominal":
        raise ValueError(f"Unknown condition: {condition}")
    return tuple(paths)


def make_config(model: str, condition: str, environment_seed: int) -> dict[str, Any]:
    config = load_config(
        BASE_CONFIG,
        [str(path) for path in override_paths(model, condition)],
    )
    config["environment_seed"] = int(environment_seed)
    config["agent_seed"] = AGENT_SEED
    return config


def action_definition(env: LEOMobilityEnv) -> dict[str, Any]:
    return {
        "stay": 0,
        "prepare": tuple(range(env.prepare_start, env.prepare_start + env.candidate_count)),
        "dup_on": env.dup_on_action,
        "dup_off": env.dup_off_action,
        "commit": env.commit_action,
        "release": env.release_action,
        "rollback": env.rollback_action,
        "action_dim": env.action_dim,
    }


def structural_signature(env: LEOMobilityEnv) -> dict[str, Any]:
    return {
        "num_ues": env.num_agents,
        "num_satellites": env.num_cells,
        "candidate_count": env.candidate_count,
        "obs_dim": env.obs_dim,
        "action_dim": env.action_dim,
        "protocol": deepcopy(env.protocol_cfg),
        "action_definition": action_definition(env),
    }


def make_agent(env: LEOMobilityEnv, config: dict[str, Any]) -> MAPPO:
    return MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        config["mappo"],
        AGENT_SEED,
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )


def validate_checkpoint_metadata(
    model: str, env: LEOMobilityEnv, agent: MAPPO, weights: Path
) -> None:
    """Fail explicitly before evaluation if checkpoint architecture is incompatible."""
    expected_flags = {
        "model_use_memory": agent.use_memory,
        "model_use_reasoning": agent.use_reasoning,
        "model_use_coordination": agent.use_coordination,
        "model_use_constraint": agent.use_constraint,
    }
    with np.load(weights, allow_pickle=False) as data:
        for key, expected in expected_flags.items():
            if key not in data.files or bool(int(np.asarray(data[key]).item())) != expected:
                raise ValueError(f"{model}: checkpoint capability mismatch for {key}")
        expected_dimensions = {
            "model_num_satellites": env.num_cells,
            "model_candidate_count": env.candidate_count,
            "model_recurrent_hidden_dim": agent.recurrent_hidden_dim,
        }
        for key, expected in expected_dimensions.items():
            if key not in data.files or int(np.asarray(data[key]).item()) != expected:
                raise ValueError(
                    f"{model}: checkpoint dimension mismatch for {key}; "
                    f"expected {expected}"
                )
        if "reward_eta_handover" not in data.files or not math.isclose(
            float(np.asarray(data["reward_eta_handover"]).item()),
            float(env.reward_cfg["eta_handover"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"{model}: checkpoint eta_handover disagrees with config")

    # MAPPO.load performs exact actor, critic, and GRU tensor-shape checks. Those
    # checks cover obs_dim/action_dim and the complete configured architecture.
    agent.load(weights)


def preflight() -> None:
    required_files = [BASE_CONFIG, COMPOUND_OVERRIDE, *COMMON_OVERRIDES]
    for spec in MODEL_SPECS.values():
        if spec["override"] is not None:
            required_files.append(spec["override"])
        required_files.append(spec["weights"])
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    with COMPOUND_OVERRIDE.open(encoding="utf-8") as handle:
        compound_document = yaml.safe_load(handle) or {}
    if set(compound_document) != {"events"}:
        raise ValueError("compound_eval.yaml must override only the event schedule")

    print("\n========== PREFLIGHT STRUCTURAL INVARIANCE ==========")
    for model, spec in MODEL_SPECS.items():
        nominal_config = make_config(model, "nominal", EVALUATION_SEEDS[0])
        stress_config = make_config(model, "stress", EVALUATION_SEEDS[0])

        stress_without_event_change = deepcopy(stress_config)
        stress_without_event_change["events"] = deepcopy(nominal_config.get("events", []))
        if stress_without_event_change != nominal_config:
            raise ValueError(
                f"{model}: nominal/stress configs differ outside the events schedule"
            )
        if stress_config.get("events") != compound_document["events"]:
            raise ValueError(f"{model}: compound event schedule was not applied exactly")

        nominal_env = LEOMobilityEnv(nominal_config)
        stress_env = LEOMobilityEnv(stress_config)
        nominal_signature = structural_signature(nominal_env)
        stress_signature = structural_signature(stress_env)
        if nominal_signature != stress_signature:
            raise ValueError(
                f"{model}: nominal/stress structural mismatch: "
                f"nominal={nominal_signature}, stress={stress_signature}"
            )
        if nominal_env.reward_cfg != stress_env.reward_cfg:
            raise ValueError(f"{model}: nominal/stress reward coefficients differ")

        for condition, env, config in (
            ("nominal", nominal_env, nominal_config),
            ("stress", stress_env, stress_config),
        ):
            agent = make_agent(env, config)
            validate_checkpoint_metadata(model, env, agent, spec["weights"])
            print(
                f"[{model} {condition}] PASS | "
                f"num_ues={env.num_agents}, num_satellites={env.num_cells}, "
                f"candidate_count={env.candidate_count}, obs_dim={env.obs_dim}, "
                f"action_dim={env.action_dim}, checkpoint_architecture=matched"
            )
        print(f"[{model}] reward={nominal_env.reward_cfg}")
        print(f"[{model}] protocol/action={nominal_signature['protocol']} / "
              f"{nominal_signature['action_definition']}")

    print(f"compound event config applied: {compound_document['events']}")
    print("=====================================================\n")


def install_seeded_episode_reset(
    env: LEOMobilityEnv, evaluation_seed: int, progress: Callable[[int], None]
) -> None:
    """Use disjoint episode streams while preserving the legacy seed-7 stream.

    evaluate_policy normally resets episodes with seeds 10000..10199. Seed 7
    retains that exact final-V3 evaluation stream; later evaluation seeds use
    consecutive, non-overlapping 200-episode streams. Nominal/stress pairs use
    the same stream by construction.
    """
    original_reset = env.reset
    episode_offset = (evaluation_seed - EVALUATION_SEEDS[0]) * EPISODES
    started_episodes = 0

    def seeded_reset(seed: int | None = None):
        nonlocal started_episodes
        progress(started_episodes)
        actual_seed = None if seed is None else int(seed) + episode_offset
        result = original_reset(seed=actual_seed)
        started_episodes += 1
        return result

    env.reset = seeded_reset  # type: ignore[method-assign]


def evaluate_combination(model: str, condition: str, evaluation_seed: int) -> dict[str, float]:
    spec = MODEL_SPECS[model]
    config = make_config(model, condition, evaluation_seed)
    env = LEOMobilityEnv(config)
    agent = make_agent(env, config)
    validate_checkpoint_metadata(model, env, agent, spec["weights"])
    action_seed = 7000 + evaluation_seed
    overrides = override_paths(model, condition)

    print("\n" + "=" * 72)
    print(f"model={model}")
    print(f"condition={condition}")
    print(f"weights path={_relative(spec['weights'])}")
    print(f"config overrides={[ _relative(path) for path in overrides ]}")
    print(f"environment_seed={evaluation_seed}")
    print(f"agent_seed={AGENT_SEED}")
    print(f"evaluation_action_seed={action_seed}")
    print(f"episodes={EPISODES} stochastic")
    if condition == "stress":
        print(f"compound event config applied={env.config['events']}")
    print("=" * 72)

    def report_progress(completed: int) -> None:
        if completed % 10 == 0:
            print(
                f"[{model} {condition} seed={evaluation_seed}] "
                f"{completed}/{EPISODES} episodes complete",
                flush=True,
            )

    install_seeded_episode_reset(env, evaluation_seed, report_progress)
    result = evaluate_policy(
        env,
        agent,
        episodes=EPISODES,
        deterministic=False,
        evaluation_action_seed=action_seed,
    )
    print(
        f"[{model} {condition} seed={evaluation_seed}] "
        f"{EPISODES}/{EPISODES} episodes complete",
        flush=True,
    )
    missing = set(REQUIRED_METRICS) - set(result)
    if missing:
        raise KeyError(f"Evaluation did not produce required metrics: {sorted(missing)}")
    return {key: float(value) for key, value in result.items()}


def load_completed() -> tuple[
    dict[tuple[str, str, int], dict[str, float]], list[str] | None
]:
    if not RAW_CSV.exists():
        return {}, None

    results: dict[tuple[str, str, int], dict[str, float]] = {}
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not set(CSV_METADATA).issubset(fieldnames):
            raise ValueError(f"Invalid resume CSV header: {RAW_CSV}")
        metric_keys = [key for key in fieldnames if key not in CSV_METADATA]
        if not set(REQUIRED_METRICS).issubset(metric_keys):
            raise ValueError(f"Required metrics are missing from {RAW_CSV}")

        for row in reader:
            if None in row or any(row.get(key) in (None, "") for key in fieldnames):
                raise ValueError(f"Incomplete row in resume CSV: {RAW_CSV}")
            model = row["model"]
            condition = row["condition"]
            evaluation_seed = int(row["evaluation_seed"])
            if model not in MODEL_SPECS or condition not in CONDITIONS:
                raise ValueError(f"Unexpected model/condition in {RAW_CSV}")
            expected_checkpoint = _relative(MODEL_SPECS[model]["weights"])
            expected_overrides = ";".join(
                _relative(path) for path in override_paths(model, condition)
            )
            if (
                evaluation_seed not in EVALUATION_SEEDS
                or int(row["environment_seed"]) != evaluation_seed
                or int(row["action_seed"]) != 7000 + evaluation_seed
                or int(row["agent_seed"]) != AGENT_SEED
                or int(row["episodes"]) != EPISODES
                or row["checkpoint"] != expected_checkpoint
                or row["config_overrides"] != expected_overrides
            ):
                raise ValueError(f"Resume CSV settings disagree with this run: {RAW_CSV}")
            key = (model, condition, evaluation_seed)
            if key in results:
                raise ValueError(f"Duplicate completed evaluation in {RAW_CSV}: {key}")
            results[key] = {metric: float(row[metric]) for metric in metric_keys}
    return results, metric_keys


def save_completed(
    results: dict[tuple[str, str, int], dict[str, float]], metric_keys: list[str]
) -> None:
    temporary_csv = RAW_CSV.with_name(RAW_CSV.name + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*CSV_METADATA, *metric_keys])
        writer.writeheader()
        for model in MODEL_SPECS:
            for condition in CONDITIONS:
                for evaluation_seed in EVALUATION_SEEDS:
                    key = (model, condition, evaluation_seed)
                    if key not in results:
                        continue
                    writer.writerow(
                        {
                            "model": model,
                            "condition": condition,
                            "evaluation_seed": evaluation_seed,
                            "environment_seed": evaluation_seed,
                            "action_seed": 7000 + evaluation_seed,
                            "agent_seed": AGENT_SEED,
                            "episodes": EPISODES,
                            "checkpoint": _relative(MODEL_SPECS[model]["weights"]),
                            "config_overrides": ";".join(
                                _relative(path)
                                for path in override_paths(model, condition)
                            ),
                            **results[key],
                        }
                    )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_csv, RAW_CSV)


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != len(EVALUATION_SEEDS):
        raise ValueError("95% CI requires all five configured evaluation seeds")
    average = mean(values)
    standard_deviation = stdev(values)
    margin = T_95_DF4 * standard_deviation / math.sqrt(len(values))
    return average, standard_deviation, average - margin, average + margin


def summary_row(
    comparison: str,
    model: str,
    condition: str,
    metric: str,
    values: list[float],
) -> dict[str, Any]:
    average, standard_deviation, low, high = summarize(values)
    return {
        "comparison": comparison,
        "model": model,
        "condition": condition,
        "metric": metric,
        "n": len(values),
        "mean": average,
        "std": standard_deviation,
        "ci95_low": low,
        "ci95_high": high,
    }


def write_and_print_summary(
    results: dict[tuple[str, str, int], dict[str, float]], metric_keys: list[str]
) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODEL_SPECS:
        for condition in CONDITIONS:
            for metric in metric_keys:
                values = [
                    results[model, condition, seed][metric]
                    for seed in EVALUATION_SEEDS
                ]
                rows.append(
                    summary_row("condition_summary", model, condition, metric, values)
                )
        for metric in metric_keys:
            differences = [
                results[model, "stress", seed][metric]
                - results[model, "nominal", seed][metric]
                for seed in EVALUATION_SEEDS
            ]
            rows.append(
                summary_row(
                    "paired_stress_effect",
                    model,
                    "stress_minus_nominal",
                    metric,
                    differences,
                )
            )

    for left, right in (("M2", "M1"), ("M4", "M2"), ("M4", "M1")):
        for metric in metric_keys:
            differences = [
                results[left, "stress", seed][metric]
                - results[right, "stress", seed][metric]
                for seed in EVALUATION_SEEDS
            ]
            rows.append(
                summary_row(
                    "descriptive_same_evaluation_seed",
                    f"{left}_minus_{right}",
                    "stress",
                    metric,
                    differences,
                )
            )

    temporary_csv = SUMMARY_CSV.with_name(SUMMARY_CSV.name + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_csv, SUMMARY_CSV)

    print("\n========== PAIRED STRESS EFFECTS (STRESS - NOMINAL) ==========")
    print("95% CI: two-sided Student t interval, df=4")
    for model in MODEL_SPECS:
        print(f"\n{model}")
        for metric in STRESS_EFFECT_METRICS:
            nominal = [results[model, "nominal", seed][metric] for seed in EVALUATION_SEEDS]
            stress = [results[model, "stress", seed][metric] for seed in EVALUATION_SEEDS]
            differences = [right - left for left, right in zip(nominal, stress)]
            nominal_mean = mean(nominal)
            stress_mean = mean(stress)
            difference_mean, _, low, high = summarize(differences)
            print(
                f"{metric}: nominal mean={nominal_mean:.9g}, "
                f"stress mean={stress_mean:.9g}, stress - nominal={difference_mean:.9g}, "
                f"95% CI=[{low:.9g}, {high:.9g}]"
            )

    print("\n========== STRESS MODEL DIFFERENCES ==========")
    print(
        "Descriptive same-evaluation-seed comparisons only; these are not "
        "paired training-seed significance tests."
    )
    for left, right in (("M2", "M1"), ("M4", "M2"), ("M4", "M1")):
        print(f"\n{left} - {right}")
        for metric in STRESS_EFFECT_METRICS:
            differences = [
                results[left, "stress", seed][metric]
                - results[right, "stress", seed][metric]
                for seed in EVALUATION_SEEDS
            ]
            average, standard_deviation, low, high = summarize(differences)
            print(
                f"{metric}: mean={average:.9g}, std={standard_deviation:.9g}, "
                f"95% descriptive interval=[{low:.9g}, {high:.9g}]"
            )


def main() -> None:
    preflight()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    results, metric_keys = load_completed()

    for model in MODEL_SPECS:
        for condition in CONDITIONS:
            for evaluation_seed in EVALUATION_SEEDS:
                key = (model, condition, evaluation_seed)
                if key in results:
                    print(
                        f"[{model} {condition} seed={evaluation_seed}] complete; skipping",
                        flush=True,
                    )
                    continue
                metrics = evaluate_combination(model, condition, evaluation_seed)
                if metric_keys is None:
                    metric_keys = sorted(metrics)
                elif set(metrics) != set(metric_keys):
                    raise ValueError("Evaluation metric keys changed between combinations")
                missing = set(REQUIRED_METRICS) - set(metric_keys)
                if missing:
                    raise KeyError(f"Missing required metrics: {sorted(missing)}")
                results[key] = metrics
                save_completed(results, metric_keys)
                print(f"saved completed combination to {RAW_CSV}", flush=True)

    if metric_keys is None:
        raise RuntimeError("No evaluation metrics were produced")
    write_and_print_summary(results, metric_keys)
    print(f"\nRaw metrics: {RAW_CSV}")
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
