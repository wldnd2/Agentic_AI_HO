from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .planner import CHOPlanner, ReasoningResult
from .world_model import (
    GeometryWorldModel,
    _planning_reliability_risk,
    _predicted_goodputs,
    _predicted_outage_probability,
)

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv


class LookaheadReasoner:
    """Backward-compatible facade for deterministic world-model CHO planning."""

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

        self.world_model = GeometryWorldModel(
            epsilon=float(config.get("score_epsilon", 1e-8)),
            reliability_risk_temperature=reliability_risk_temperature,
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
        self.world_model.outage_probability_fn = _predicted_outage_probability
        self.world_model.reliability_risk_fn = _planning_reliability_risk
        self.world_model.goodputs_fn = _predicted_goodputs
        self.planner.future_link_predictor = self._future_link
        return self.planner.analyze(env, masks, prices, outage_dual_mu)

    def _future_link(
        self, env: LEOMobilityEnv, ue_id: int, cell_id: int | None, offset: int
    ) -> tuple[float, float]:
        return self.world_model.future_link(env, ue_id, cell_id, offset)
