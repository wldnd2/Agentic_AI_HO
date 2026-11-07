from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO
from leo_mappo.policy_diagnostics import model_parameter_digest
from leo_mappo.runner import evaluate_policy, evaluation_episode_seeds


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ITERATIONS = (100, 200, 300, 400, 500)
DETERMINISTIC_EPISODES = 10
STOCHASTIC_EPISODES = 10
EVALUATION_SEED_BASE = 5001
EVALUATION_ACTION_SEED = 7007
METRIC_FIELDS = (
    "packet_delivery_ratio",
    "packet_drop_rate",
    "deadline_miss_rate",
    "delivered_goodput_bps",
    "mean_queue_packets",
    "p95_queue_packets",
    "mean_hol_delay_s",
    "team_reward",
    "service_deficit",
    "control_cost",
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
    "resource_price_mean",
    "resource_price_max",
    "resource_price_std",
    "price_nonzero_fraction",
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    run_dir: Path
    coordination_enabled: bool


@dataclass(frozen=True)
class EvaluationTask:
    method: str
    run_dir: Path
    coordination_enabled: bool
    checkpoint_iteration: int
    checkpoint_path: Path
    evaluation_mode: str
    deterministic: bool
    episodes: int


def checkpoint_path(run_dir: Path, iteration: int) -> Path:
    return run_dir / "checkpoints" / f"mappo_weights_iter_{iteration:04d}.npz"


def build_evaluation_plan(run_specs: tuple[RunSpec, ...]) -> tuple[EvaluationTask, ...]:
    tasks = []
    for spec in run_specs:
        for iteration in CHECKPOINT_ITERATIONS:
            weights = checkpoint_path(spec.run_dir, iteration)
            tasks.extend(
                (
                    EvaluationTask(
                        method=spec.method,
                        run_dir=spec.run_dir,
                        coordination_enabled=spec.coordination_enabled,
                        checkpoint_iteration=iteration,
                        checkpoint_path=weights,
                        evaluation_mode="deterministic",
                        deterministic=True,
                        episodes=DETERMINISTIC_EPISODES,
                    ),
                    EvaluationTask(
                        method=spec.method,
                        run_dir=spec.run_dir,
                        coordination_enabled=spec.coordination_enabled,
                        checkpoint_iteration=iteration,
                        checkpoint_path=weights,
                        evaluation_mode="stochastic",
                        deterministic=False,
                        episodes=STOCHASTIC_EPISODES,
                    ),
                )
            )
    return tuple(tasks)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_path(configured_path: str | Path) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_run_metadata(
    metadata: dict[str, Any], *, coordination_enabled: bool
) -> dict[str, Any]:
    config = metadata.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint metadata has no resolved_config mapping")
    mappo = config.get("mappo", {})
    reasoning = config.get("reasoning", {})
    reward = config.get("reward", {})
    expected_coordination_mode = (
        "satellite_price" if coordination_enabled else "none"
    )
    checks = {
        "history_recurrent_mode": metadata.get("history_recurrent_mode")
        == "recurrent",
        "world_model_type": metadata.get("world_model_type") == "hybrid",
        "coordination_mode": metadata.get("coordination_mode")
        == expected_coordination_mode,
        "use_memory": bool(mappo.get("use_memory", False)),
        "use_reasoning": bool(mappo.get("use_reasoning", False)),
        "confidence_aware": mappo.get("reasoning_feature_mode")
        == "confidence_aware",
        "use_coordination": bool(mappo.get("use_coordination", False))
        == coordination_enabled,
        "constraint_disabled": not bool(mappo.get("use_constraint", False)),
        "hybrid_world_model": reasoning.get("world_model") == "hybrid",
        "counterfactual_planner": reasoning.get("planner_objective")
        == "counterfactual",
        "rejected_prep_correction_disabled": not bool(
            reward.get("charge_rejected_preparation_signaling", False)
        ),
        "training_iterations": int(mappo.get("training_iterations", -1)) == 500,
        "checkpoint_interval": int(mappo.get("checkpoint_interval", -1)) == 100,
        "training_seed": int(metadata.get("training_seed", -1)) == 7,
        "environment_seed": int(metadata.get("environment_seed", -1)) == 7,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(
            "run metadata does not match the frozen budget study: "
            + ", ".join(failed)
        )
    return deepcopy(config)


def validate_matched_configs(
    m3_config: dict[str, Any], m4_config: dict[str, Any]
) -> None:
    normalized_m3 = deepcopy(m3_config)
    normalized_m4 = deepcopy(m4_config)
    normalized_m3["mappo"]["use_coordination"] = False
    normalized_m4["mappo"]["use_coordination"] = False
    if normalized_m3 != normalized_m4:
        raise ValueError(
            "M3-H and M4-H resolved configs differ beyond mappo.use_coordination"
        )


def _load_run(spec: RunSpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    metadata_path = spec.run_dir / "checkpoint_metadata.json"
    metadata = _read_json(metadata_path)
    config = validate_run_metadata(
        metadata, coordination_enabled=spec.coordination_enabled
    )
    configured_residual = config["reasoning"].get("residual_model_path")
    if not configured_residual:
        raise ValueError(f"{spec.method} metadata has no residual model path")
    residual_path = _resolve_repo_path(configured_residual)
    if not residual_path.is_file():
        raise FileNotFoundError(f"residual model does not exist: {residual_path}")
    residual_sha256 = _sha256(residual_path)
    recorded_sha256 = metadata.get("residual_checkpoint_sha256")
    if not recorded_sha256 or residual_sha256 != recorded_sha256:
        raise ValueError(
            f"{spec.method} residual checkpoint SHA-256 does not match metadata"
        )
    residual = {
        "path": str(residual_path),
        "sha256": residual_sha256,
    }
    return config, metadata, residual


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


def _optimizer_digest(agent: MAPPO) -> str:
    digest = hashlib.sha256()
    optimizers = (
        ("actor", agent.actor_optimizer),
        ("gru", agent.gru_optimizer),
        ("critic", agent.critic_optimizer),
    )
    for label, optimizer in optimizers:
        if optimizer is None:
            digest.update(f"{label}:none".encode())
            continue
        digest.update(
            f"{label}:lr={optimizer.learning_rate}:t={optimizer.t}".encode()
        )
        for state_name, state in (("m", optimizer.m), ("v", optimizer.v)):
            for name in sorted(state):
                values = np.asarray(state[name])
                digest.update(
                    f"{label}:{state_name}:{name}:{values.shape}:{values.dtype}".encode()
                )
                digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _rng_state(agent: MAPPO) -> str:
    return json.dumps(agent.rng.bit_generator.state, sort_keys=True)


def evaluate_task(
    task: EvaluationTask,
    config: dict[str, Any],
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = LEOMobilityEnv(deepcopy(config))
    agent = _build_agent(env, config, task.checkpoint_path)
    parameter_before = model_parameter_digest(agent)
    optimizer_before = _optimizer_digest(agent)
    rng_before = _rng_state(agent)
    dual_before = float(agent.outage_dual_mu)

    metrics = evaluate_policy(
        env,
        agent,
        episodes=task.episodes,
        deterministic=task.deterministic,
        evaluation_action_seed=EVALUATION_ACTION_SEED,
        evaluation_seed_base=EVALUATION_SEED_BASE,
    )

    safeguards = {
        "parameters_unchanged": model_parameter_digest(agent) == parameter_before,
        "optimizers_unchanged": _optimizer_digest(agent) == optimizer_before,
        "agent_rng_restored": _rng_state(agent) == rng_before,
        "dual_unchanged": float(agent.outage_dual_mu) == dual_before,
        "checkpoint_unchanged": _sha256(task.checkpoint_path)
        == checkpoint_sha256,
    }
    failed = [name for name, valid in safeguards.items() if not valid]
    if failed:
        raise RuntimeError(
            f"evaluation mutated frozen state for {task.method} iteration "
            f"{task.checkpoint_iteration}: {', '.join(failed)}"
        )
    missing = [field for field in METRIC_FIELDS if field not in metrics]
    if missing:
        raise KeyError("evaluation result is missing metrics: " + ", ".join(missing))
    row = {
        "method": task.method,
        "checkpoint_iteration": task.checkpoint_iteration,
        "evaluation_mode": task.evaluation_mode,
        "coordination_enabled": task.coordination_enabled,
        "episodes": task.episodes,
        "evaluation_seed_base": EVALUATION_SEED_BASE,
        "evaluation_action_seed": EVALUATION_ACTION_SEED,
        **{field: metrics[field] for field in METRIC_FIELDS},
    }
    return row, safeguards


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_provenance() -> dict[str, Any]:
    tags = [tag for tag in _git_output("tag", "--points-at", "HEAD").splitlines() if tag]
    nearest_tag = _git_output("describe", "--tags", "--abbrev=0")
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "tag": tags[0] if len(tags) == 1 else nearest_tag,
        "tags": tags,
        "describe": _git_output("describe", "--tags", "--always", "--dirty"),
        "working_tree_dirty": bool(_git_output("status", "--porcelain")),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def run(args: argparse.Namespace) -> None:
    run_specs = (
        RunSpec("M3-H", args.m3_run.resolve(), False),
        RunSpec("M4-H", args.m4_run.resolve(), True),
    )
    plan = build_evaluation_plan(run_specs)
    if len(plan) != 20:
        raise RuntimeError("matched checkpoint evaluation plan must contain 20 rows")

    run_data: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, str]]] = {}
    checkpoint_records: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in run_specs:
        run_data[spec.method] = _load_run(spec)
        for iteration in CHECKPOINT_ITERATIONS:
            weights = checkpoint_path(spec.run_dir, iteration)
            if not weights.is_file():
                raise FileNotFoundError(f"required checkpoint does not exist: {weights}")
            checkpoint_records[(spec.method, iteration)] = {
                "iteration": iteration,
                "path": str(weights.resolve()),
                "sha256": _sha256(weights),
            }

    validate_matched_configs(run_data["M3-H"][0], run_data["M4-H"][0])
    if run_data["M3-H"][2]["sha256"] != run_data["M4-H"][2]["sha256"]:
        raise ValueError("M3-H and M4-H do not use the same residual checkpoint")

    rows: list[dict[str, Any]] = []
    safeguard_records = []
    for task in plan:
        print(
            f"method={task.method} iteration={task.checkpoint_iteration:04d} "
            f"mode={task.evaluation_mode}",
            flush=True,
        )
        config = run_data[task.method][0]
        checkpoint_record = checkpoint_records[
            (task.method, task.checkpoint_iteration)
        ]
        row, safeguards = evaluate_task(
            task,
            config,
            str(checkpoint_record["sha256"]),
        )
        rows.append(row)
        safeguard_records.append(
            {
                "method": task.method,
                "checkpoint_iteration": task.checkpoint_iteration,
                "evaluation_mode": task.evaluation_mode,
                **safeguards,
            }
        )

    if len(rows) != 20:
        raise RuntimeError("matched checkpoint evaluation produced an incomplete table")
    _write_csv(args.output, rows)

    provenance_runs = {}
    for spec in run_specs:
        _, metadata, residual = run_data[spec.method]
        provenance_runs[spec.method] = {
            "run_directory": str(spec.run_dir),
            "metadata_path": str((spec.run_dir / "checkpoint_metadata.json").resolve()),
            "coordination_enabled": spec.coordination_enabled,
            "coordination_mode": (
                "satellite_price" if spec.coordination_enabled else "none"
            ),
            "history_recurrent_mode": "recurrent",
            "reasoning_feature_mode": "confidence_aware",
            "world_model_type": "hybrid",
            "rejected_prep_correction_enabled": False,
            "reliability_constraint_enabled": False,
            "residual_model": residual,
            "checkpoint_metadata": metadata,
            "checkpoints": [
                checkpoint_records[(spec.method, iteration)]
                for iteration in CHECKPOINT_ITERATIONS
            ],
        }
    _write_json(
        args.provenance_output,
        {
            "git": _git_provenance(),
            "evaluation": {
                "checkpoint_iterations": list(CHECKPOINT_ITERATIONS),
                "deterministic_episodes": DETERMINISTIC_EPISODES,
                "stochastic_episodes": STOCHASTIC_EPISODES,
                "evaluation_seed_base": EVALUATION_SEED_BASE,
                "environment_seeds": list(
                    evaluation_episode_seeds(
                        DETERMINISTIC_EPISODES, EVALUATION_SEED_BASE
                    )
                ),
                "evaluation_action_seed": EVALUATION_ACTION_SEED,
                "critic_used_for_action_selection": False,
                "final_checkpoint_aliases_used": False,
            },
            "runs": provenance_runs,
            "safeguards": safeguard_records,
            "summary_csv": str(args.output.resolve()),
        },
    )
    print(f"summary={args.output.resolve()}")
    print(f"provenance={args.provenance_output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen M3-H/M4-H training-budget checkpoints"
    )
    parser.add_argument("--m3-run", type=Path, required=True)
    parser.add_argument("--m4-run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/training_budget_500/checkpoint_evaluation_summary.csv"),
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path(
            "outputs/training_budget_500/checkpoint_evaluation_provenance.json"
        ),
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
