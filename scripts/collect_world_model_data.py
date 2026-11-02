from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from leo_mappo.baselines import max_sinr_policy
from leo_mappo.config import load_config
from leo_mappo.demand import normalize_required_rate_model
from leo_mappo.environment import LEOMobilityEnv
from leo_mappo.prediction_dataset import (
    EpisodePredictionTrajectory,
    build_prediction_dataset,
)
from leo_mappo.predictive_control import (
    build_predictive_reasoner,
    predictive_controller_step,
)
from leo_mappo.reasoning import LookaheadReasoner
from leo_mappo.residual_world_model import (
    FEATURE_DIM,
    OBSERVABLE_LINK_FEATURES,
    build_observable_link_features,
)
from leo_mappo.ue import MobilityState
from leo_mappo.world_model import GeometryWorldModel


def _world_model(config: dict) -> GeometryWorldModel:
    reasoning = config.get("reasoning", {})
    return GeometryWorldModel(
        epsilon=float(reasoning.get("score_epsilon", 1e-8)),
        reliability_risk_temperature=float(
            reasoning.get("reliability_risk_temperature", 0.10)
        ),
    )


def _metadata_path(dataset_path: str | Path) -> Path:
    return Path(f"{Path(dataset_path)}.metadata.json")


def build_collection_metadata(
    args: argparse.Namespace,
    config: dict,
) -> dict:
    """Describe one collection run without changing the NPZ dataset schema."""
    base_seed = int(args.seed)
    episode_count = int(args.episodes)
    overrides = [str(path) for path in getattr(args, "override", [])]
    behavior_policy = str(args.policy)
    controlling_world_model = {
        "geometry_planner": "geometry",
        "hybrid_planner": "hybrid",
    }.get(behavior_policy)
    reasoner_controls_environment = controlling_world_model is not None
    metadata = {
        "schema_version": 1,
        "dataset_path": str(args.output),
        "required_rate_model": normalize_required_rate_model(
            config["traffic"].get("required_rate_model")
        ),
        "config": str(args.config),
        "overrides": overrides,
        "base_seed": base_seed,
        "episode_count": episode_count,
        "episode_seed_progression": "base_seed + episode_id",
        "episode_seeds": list(range(base_seed, base_seed + episode_count)),
        "seed_range": {
            "first": base_seed,
            "last": base_seed + episode_count - 1,
        },
        "history_steps": int(args.history_steps),
        "prediction_horizon_steps": int(args.horizon_steps),
        "behavior_policy": behavior_policy,
        "world_model_controlling_environment": controlling_world_model,
        "reasoner_controls_environment": reasoner_controls_environment,
        "feature_dim": FEATURE_DIM,
        "feature_order": list(OBSERVABLE_LINK_FEATURES),
        "prediction_target": (
            "true_realized_sinr_db(t+h) - "
            "geometry_predicted_sinr_db(t+h | t)"
        ),
        "control_abstraction": {
            "description": (
                "3GPP-inspired system-level CHO control abstraction; not a "
                "message-level RRC procedure"
            ),
            "research_facing_terminology": {
                "COMMIT": (
                    "EXECUTE_CHO; system-level abstraction of CHO execution"
                ),
                "COOLDOWN": (
                    "POST_HO_GUARD; system-level anti-ping-pong guard, not a "
                    "literal 3GPP state"
                ),
                "PREP": "retains its existing meaning",
                "PREPARED": "retains its existing meaning",
                "RELEASE": "retains its existing meaning",
            },
        },
    }
    if behavior_policy == "hybrid_planner":
        configured_path = getattr(args, "residual_model", None)
        if not configured_path:
            raise ValueError(
                "--residual-model is required with --policy hybrid_planner"
            )
        checkpoint = Path(configured_path).expanduser().resolve()
        metadata.update(
            {
                "residual_model_path": str(checkpoint),
                "residual_model_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
        )
    return metadata


def _random_feasible_actions(
    env: LEOMobilityEnv, rng: np.random.Generator
) -> np.ndarray:
    return np.asarray(
        [rng.choice(np.flatnonzero(mask > 0.0)) for mask in env.action_masks()],
        dtype=np.int64,
    )


def _observable_max_sinr_actions(env: LEOMobilityEnv) -> np.ndarray:
    """Use the existing baseline in V2 and delayed/noisy telemetry in V3."""
    if not env.v3_enabled:
        return max_sinr_policy(env)
    actions = np.zeros(env.num_agents, dtype=np.int64)
    for ue in env.ues:
        ue_id = ue.ue_id
        if ue.state == MobilityState.SINGLE and env.candidates[ue_id]:
            primary = ue.primary_cell
            if primary is None:
                continue
            slot = max(
                range(len(env.candidates[ue_id])),
                key=lambda index: env.observed_sinr[
                    ue_id, env.candidates[ue_id][index]
                ],
            )
            target = env.candidates[ue_id][slot]
            if env.observed_sinr[ue_id, target] > env.observed_sinr[ue_id, primary] + 1.0:
                actions[ue_id] = env.prepare_start + slot
        elif ue.state == MobilityState.PREPARED and ue.protocol_timer <= 0:
            actions[ue_id] = env.commit_action
    masks = env.action_masks()
    invalid = masks[np.arange(env.num_agents), actions] <= 0.0
    actions[invalid] = 0
    return actions


def build_behavior_reasoner(
    config: dict,
    env: LEOMobilityEnv,
    policy: str,
    residual_model: str | Path | None = None,
) -> LookaheadReasoner | None:
    """Build a reasoner only when it directly controls the rollout."""
    if policy in ("max_sinr", "random_feasible"):
        return None
    if policy == "geometry_planner":
        return build_predictive_reasoner(config, env, world_model="geometry")
    if policy == "hybrid_planner":
        if residual_model is None:
            raise ValueError(
                "--residual-model is required with --policy hybrid_planner"
            )
        return build_predictive_reasoner(
            config,
            env,
            world_model="hybrid",
            residual_model=residual_model,
        )
    raise ValueError(f"unknown collection policy {policy!r}")


def _step_behavior_policy(
    env: LEOMobilityEnv,
    policy: str,
    policy_rng: np.random.Generator,
    reasoner: LookaheadReasoner | None,
) -> np.ndarray:
    if policy == "max_sinr":
        actions = _observable_max_sinr_actions(env)
        env.step(actions)
        return actions
    if policy == "random_feasible":
        actions = _random_feasible_actions(env, policy_rng)
        env.step(actions)
        return actions
    if reasoner is None:
        raise RuntimeError(f"{policy} requires a predictive reasoner")
    _, actions, _, _ = predictive_controller_step(env, reasoner)
    return actions


def _collect_complete_trajectory(
    env: LEOMobilityEnv,
    model: GeometryWorldModel,
    episode_id: int,
    episode_seed: int,
    horizon_steps: int,
    policy: str,
    policy_rng: np.random.Generator,
    reasoner: LookaheadReasoner | None = None,
) -> EpisodePredictionTrajectory:
    env.reset(seed=episode_seed)
    time_count = int(env.env_cfg["episode_steps"])
    feature_history = np.empty(
        (time_count, env.num_agents, env.num_cells, FEATURE_DIM), dtype=np.float64
    )
    true_sinr_history = np.empty(
        (time_count, env.num_agents, env.num_cells), dtype=np.float64
    )
    visible_history = np.empty(
        (time_count, env.num_agents, env.num_cells), dtype=bool
    )
    selected_pairs = np.zeros_like(visible_history)
    geometry_sinr = np.zeros(
        (time_count, env.num_agents, env.num_cells, horizon_steps), dtype=np.float64
    )
    geometry_visible = np.zeros_like(geometry_sinr, dtype=bool)

    for time_index in range(time_count):
        for ue in env.ues:
            ue_id = ue.ue_id
            pair_cells = set(env.candidates[ue_id])
            if ue.primary_cell is not None:
                pair_cells.add(ue.primary_cell)
            for cell_id in range(env.num_cells):
                feature_history[time_index, ue_id, cell_id] = (
                    build_observable_link_features(env, ue_id, cell_id, model)
                )
                # Current realized SINR is retained only in this offline-label
                # array. It is never passed to the feature builder or model.
                link = env.links[ue_id][cell_id]
                true_sinr_history[time_index, ue_id, cell_id] = link.sinr_db
                visible_history[time_index, ue_id, cell_id] = link.visible
            for cell_id in pair_cells:
                selected_pairs[time_index, ue_id, cell_id] = True
                for horizon_index in range(horizon_steps):
                    prediction = model.predict_link(
                        env, ue_id, cell_id, horizon_index + 1
                    )
                    geometry_sinr[
                        time_index, ue_id, cell_id, horizon_index
                    ] = prediction.sinr_db
                    geometry_visible[
                        time_index, ue_id, cell_id, horizon_index
                    ] = prediction.visible

        _step_behavior_policy(env, policy, policy_rng, reasoner)

    return EpisodePredictionTrajectory(
        episode_id=episode_id,
        features=feature_history,
        true_sinr_db=true_sinr_history,
        geometric_visible=visible_history,
        selected_pairs=selected_pairs,
        geometry_sinr_db=geometry_sinr,
        geometry_visible=geometry_visible,
    )


def collect(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.history_steps <= 0 or args.horizon_steps <= 0:
        raise ValueError("history and horizon steps must be positive")
    config = load_config(args.config, args.override)
    config["environment_seed"] = int(args.seed)
    env = LEOMobilityEnv(config)
    model = _world_model(config)
    policy_rng = np.random.default_rng(args.seed)
    reasoner = build_behavior_reasoner(
        config,
        env,
        args.policy,
        getattr(args, "residual_model", None),
    )
    metadata = build_collection_metadata(args, config)

    # No supervised windows are made inside the rollout loop. Each episode is
    # fully materialized first, then the pure post-processing builder runs.
    trajectories = [
        _collect_complete_trajectory(
            env=env,
            model=model,
            episode_id=episode_id,
            episode_seed=args.seed + episode_id,
            horizon_steps=args.horizon_steps,
            policy=args.policy,
            policy_rng=policy_rng,
            reasoner=reasoner,
        )
        for episode_id in range(args.episodes)
    ]
    dataset = build_prediction_dataset(
        trajectories,
        history_steps=args.history_steps,
        horizon_steps=args.horizon_steps,
    )
    dataset.save(args.output)
    metadata_path = _metadata_path(args.output)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(
        f"saved {dataset.sample_count} samples from {args.episodes} complete "
        f"episodes to {Path(args.output)}; metadata={metadata_path}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect offline residual world-model prediction data."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="YAML override path; repeat the option to apply multiple files in order.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument(
        "--policy",
        choices=(
            "max_sinr",
            "random_feasible",
            "geometry_planner",
            "hybrid_planner",
        ),
        default="max_sinr",
    )
    parser.add_argument(
        "--residual-model",
        help="Frozen residual checkpoint required for hybrid_planner.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    collect(parse_args())
