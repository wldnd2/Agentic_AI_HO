from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .hybrid_world_model import HybridWorldModel
from .planner import CHOPlanner, ReasoningResult
from .world_model import (
    GeometryWorldModel,
    _planning_reliability_risk,
    _predicted_goodputs,
    _predicted_outage_probability,
)

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv


REASONING_FEATURE_MODES = ("legacy", "confidence_aware")


def reasoning_feature_dim(action_dim: int, mode: str) -> int:
    """Return the per-agent actor feature width for a reasoning mode."""
    normalized_mode = str(mode).lower()
    if normalized_mode == "legacy":
        return int(action_dim)
    if normalized_mode == "confidence_aware":
        return 2 * int(action_dim) + 1
    raise ValueError(
        "mappo.reasoning_feature_mode must be either 'legacy' or "
        "'confidence_aware'"
    )


def build_actor_reasoning_features(
    result: ReasoningResult,
    masks: np.ndarray,
    score_clip: float,
    mode: str,
) -> np.ndarray:
    """Construct actor inputs from already-computed planner outputs.

    ``legacy`` is byte-for-byte the historical feature matrix. The
    confidence-aware mode preserves that matrix and appends, for each action,
    its clipped raw-cost separation from the best feasible action plus one
    best-vs-second-best raw-cost margin. No planner score is recomputed.
    """
    normalized_mode = str(mode).lower()
    legacy = np.asarray(result.features, dtype=np.float64)
    action_masks = np.asarray(masks, dtype=np.float64)
    raw_costs = np.asarray(result.raw_costs, dtype=np.float64)
    score_gaps = np.asarray(result.score_gaps, dtype=np.float64)
    if legacy.ndim != 2 or action_masks.shape != legacy.shape:
        raise ValueError("reasoning features and action masks must be matching matrices")
    if raw_costs.shape != legacy.shape or score_gaps.shape != (legacy.shape[0],):
        raise ValueError("reasoning result dimensions do not match actor action dimensions")
    if not np.isfinite(score_clip) or score_clip <= 0.0:
        raise ValueError("reasoning score_clip must be finite and positive")
    if normalized_mode == "legacy":
        return legacy.copy()
    reasoning_feature_dim(legacy.shape[1], normalized_mode)

    magnitude = np.zeros_like(raw_costs)
    confidence = np.zeros((legacy.shape[0], 1), dtype=np.float64)
    for agent_id in range(legacy.shape[0]):
        feasible = np.flatnonzero(action_masks[agent_id] > 0.0)
        if len(feasible) == 0:
            raise ValueError("action mask contains a row with no feasible action")
        feasible_costs = raw_costs[agent_id, feasible]
        best_cost = float(np.min(feasible_costs))
        magnitude[agent_id, feasible] = np.clip(
            (feasible_costs - best_cost) / score_clip,
            0.0,
            1.0,
        )
        confidence[agent_id, 0] = np.clip(
            score_gaps[agent_id] / score_clip,
            0.0,
            1.0,
        )
    features = np.concatenate([legacy, magnitude, confidence], axis=1)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("actor reasoning features contain non-finite values")
    return features


class LookaheadReasoner:
    """Backward-compatible facade for configured world-model CHO planning."""

    def __init__(
        self,
        config: dict,
        reward_config: dict,
        coordination_config: dict | None = None,
        use_coordination: bool = False,
        use_constraint: bool = False,
    ):
        horizon = int(config.get("horizon_steps", 6))
        score_clip = float(config.get("score_clip", 5.0))
        reliability_risk_temperature = float(
            config.get("reliability_risk_temperature", 0.10)
        )
        if horizon <= 0 or score_clip <= 0.0:
            raise ValueError("reasoning horizon and score_clip must be positive")
        if reliability_risk_temperature <= 0.0:
            raise ValueError("reasoning reliability_risk_temperature must be positive")

        geometry_model = GeometryWorldModel(
            epsilon=float(config.get("score_epsilon", 1e-8)),
            reliability_risk_temperature=reliability_risk_temperature,
        )
        world_model_mode = str(config.get("world_model", "geometry")).lower()
        if world_model_mode == "geometry":
            self.world_model = geometry_model
        elif world_model_mode == "hybrid":
            configured_path = config.get("residual_model_path")
            if not configured_path:
                raise ValueError(
                    "reasoning.residual_model_path is required when "
                    "reasoning.world_model=hybrid"
                )
            checkpoint_path = Path(configured_path).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "hybrid residual model checkpoint does not exist: "
                    f"{checkpoint_path}"
                )
            self.world_model = HybridWorldModel.from_checkpoint(
                geometry_model,
                checkpoint_path,
            )
        else:
            raise ValueError(
                "reasoning.world_model must be either 'geometry' or 'hybrid'"
            )
        self.planner = CHOPlanner(
            config,
            reward_config,
            coordination_config,
            use_coordination,
            use_constraint,
            world_model=self.world_model,
        )

        # Preserve the existing attribute surface for diagnostics and tests.
        self.horizon = self.planner.horizon
        self.discount = self.planner.discount
        self.epsilon = self.planner.epsilon
        self.score_clip = self.planner.score_clip
        self.planner_objective = self.planner.planner_objective
        self.reliability_risk_temperature = (
            self.world_model.reliability_risk_temperature
        )
        self.lambda_resource = self.planner.lambda_resource
        self.eta_preparation = self.planner.eta_preparation
        self.eta_handover = self.planner.eta_handover
        self.reasoning_price_weight = self.planner.reasoning_price_weight
        self.use_coordination = self.planner.use_coordination
        self.use_constraint = self.planner.use_constraint

    def analyze(
        self,
        env: LEOMobilityEnv,
        masks: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        outage_dual_mu: float = 0.0,
    ) -> ReasoningResult:
        # Keep historical monkey-patching seams while delegating all work.
        self.world_model.observe(env)
        self.world_model.outage_probability_fn = _predicted_outage_probability
        self.world_model.reliability_risk_fn = _planning_reliability_risk
        self.world_model.goodputs_fn = _predicted_goodputs
        self.planner.future_link_predictor = self._future_link
        return self.planner.analyze(env, masks, prices, outage_dual_mu)

    def _future_link(
        self, env: LEOMobilityEnv, ue_id: int, cell_id: int | None, offset: int
    ) -> tuple[float, float]:
        return self.world_model.future_link(env, ue_id, cell_id, offset)
