from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .environment import LEOMobilityEnv, StepOutput
from .hybrid_world_model import HybridWorldModel
from .planner import ReasoningResult
from .reasoning import LookaheadReasoner


# Research-facing terminology: this simulator is a 3GPP-inspired system-level
# CHO control abstraction, not a message-level RRC implementation. The legacy
# numeric action/state names remain unchanged: COMMIT corresponds to
# EXECUTE_CHO, while COOLDOWN corresponds to the POST_HO_GUARD anti-ping-pong
# guard. PREP, PREPARED, and RELEASE retain their existing meanings.


def select_greedy_feasible_actions(
    raw_costs: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Choose minimum-cost feasible actions with lowest-index tie-breaking."""
    costs = np.asarray(raw_costs, dtype=np.float64)
    feasible_masks = np.asarray(masks, dtype=np.float64)
    if costs.shape != feasible_masks.shape or costs.ndim != 2:
        raise ValueError("raw_costs and masks must be matching 2-D arrays")
    actions = np.empty(costs.shape[0], dtype=np.int64)
    for ue_id in range(costs.shape[0]):
        feasible = np.flatnonzero(feasible_masks[ue_id] > 0.0)
        if feasible.size == 0:
            raise RuntimeError(f"UE {ue_id} has no feasible action")
        actions[ue_id] = min(
            (int(action) for action in feasible),
            key=lambda action: (float(costs[ue_id, action]), action),
        )
    return actions


def predictive_controller_step(
    env: LEOMobilityEnv,
    reasoner: LookaheadReasoner,
) -> tuple[np.ndarray, np.ndarray, ReasoningResult, StepOutput]:
    """Analyze once and execute the feasible raw-cost argmin joint action."""
    masks = env.action_masks()
    result = reasoner.analyze(env, masks)
    actions = select_greedy_feasible_actions(result.raw_costs, masks)
    output = env.step(actions)
    return masks, actions, result, output


def build_predictive_reasoner(
    config: dict[str, Any],
    env: LEOMobilityEnv,
    world_model: str = "geometry",
    residual_model: str | Path | None = None,
    reasoner_factory: type[LookaheadReasoner] = LookaheadReasoner,
) -> LookaheadReasoner:
    """Build the shared counterfactual Geometry or Hybrid reasoner."""
    model_mode = str(world_model).lower()
    if model_mode not in ("geometry", "hybrid"):
        raise ValueError("world_model must be either 'geometry' or 'hybrid'")
    if model_mode == "hybrid" and residual_model is None:
        raise ValueError("--residual-model is required with a Hybrid planner")

    reasoning_config = dict(config.get("reasoning", {}))
    reasoning_config["world_model"] = model_mode
    if model_mode == "geometry":
        reasoning_config.pop("residual_model_path", None)
    else:
        reasoning_config["residual_model_path"] = str(residual_model)
    reasoner = reasoner_factory(
        reasoning_config,
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=False,
        use_constraint=False,
    )
    if reasoner.planner_objective != "counterfactual":
        raise ValueError(
            "predictive controller requires "
            "reasoning.planner_objective=counterfactual"
        )
    if model_mode == "hybrid" and not isinstance(
        reasoner.world_model, HybridWorldModel
    ):
        raise RuntimeError(
            "hybrid evaluation did not construct the required HybridWorldModel"
        )
    return reasoner
