from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from leo_mappo.demand import normalize_required_rate_model
import scripts.evaluate_geometry_planner_closed_loop as evaluator


ROOT = Path(__file__).resolve().parents[1]


def test_selected_actions_are_feasible_according_to_mask() -> None:
    costs = np.asarray([[0.0, -100.0, 2.0], [5.0, 4.0, 3.0]])
    masks = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    actions = evaluator.select_greedy_feasible_actions(costs, masks)
    assert all(masks[ue_id, action] > 0.0 for ue_id, action in enumerate(actions))


def test_minimum_raw_cost_feasible_action_is_selected() -> None:
    costs = np.asarray([[0.7, 0.2, -5.0, 0.4]])
    masks = np.asarray([[1.0, 1.0, 0.0, 1.0]])
    actions = evaluator.select_greedy_feasible_actions(costs, masks)
    assert actions.tolist() == [1]


def test_lowest_action_index_breaks_equal_cost_tie() -> None:
    costs = np.asarray([[0.8, 0.1, 0.1, 0.4]])
    masks = np.ones_like(costs)
    actions = evaluator.select_greedy_feasible_actions(costs, masks)
    assert actions.tolist() == [1]


def test_reasoner_recommendation_is_passed_directly_to_environment_step() -> None:
    class TinyEnvironment:
        def __init__(self) -> None:
            self.received_actions: np.ndarray | None = None

        def action_masks(self) -> np.ndarray:
            return np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

        def step(self, actions: np.ndarray) -> SimpleNamespace:
            self.received_actions = np.asarray(actions).copy()
            return SimpleNamespace(done=True)

    class TinyReasoner:
        def analyze(
            self, env: TinyEnvironment, masks: np.ndarray
        ) -> SimpleNamespace:
            assert np.array_equal(masks, env.action_masks())
            return SimpleNamespace(
                raw_costs=np.asarray([[0.5, 0.1, -9.0], [0.3, -8.0, 0.2]])
            )

    env = TinyEnvironment()
    _, actions, _, _ = evaluator.geometry_controller_step(  # type: ignore[arg-type]
        env,  # type: ignore[arg-type]
        TinyReasoner(),  # type: ignore[arg-type]
    )
    assert actions.tolist() == [1, 2]
    assert env.received_actions is not None
    np.testing.assert_array_equal(env.received_actions, actions)


def test_geometry_reasoner_never_receives_hybrid_or_residual_configuration() -> None:
    captured: dict[str, object] = {}

    class FakeReasoner:
        planner_objective = "counterfactual"

        def __init__(
            self,
            reasoning_config: dict,
            reward_config: dict,
            coordination_config: dict,
            use_coordination: bool,
            use_constraint: bool,
        ) -> None:
            captured["reasoning_config"] = reasoning_config
            captured["use_coordination"] = use_coordination
            captured["use_constraint"] = use_constraint

    config = {
        "reasoning": {
            "planner_objective": "counterfactual",
            "world_model": "hybrid",
            "residual_model_path": "must-not-be-loaded.npz",
        },
        "coordination": {},
    }
    env = SimpleNamespace(reward_cfg={})
    with patch.object(evaluator, "LookaheadReasoner", FakeReasoner):
        evaluator.build_geometry_reasoner(config, env)  # type: ignore[arg-type]

    reasoning_config = captured["reasoning_config"]
    assert isinstance(reasoning_config, dict)
    assert reasoning_config["world_model"] == "geometry"
    assert "residual_model_path" not in reasoning_config
    assert captured["use_coordination"] is False
    assert captured["use_constraint"] is False
    assert "HybridWorldModel" not in evaluator.__dict__


def test_legacy_and_deadline_aware_config_composition() -> None:
    common = [
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
        ROOT / "configs/counterfactual_planner.yaml",
    ]
    legacy = evaluator.load_evaluation_config(
        ROOT / "configs/default.yaml", common, seed=17
    )
    deadline_aware = evaluator.load_evaluation_config(
        ROOT / "configs/default.yaml",
        [*common, ROOT / "configs/deadline_aware_demand.yaml"],
        seed=17,
    )
    assert normalize_required_rate_model(
        legacy["traffic"].get("required_rate_model")
    ) == "arrival_capped"
    assert deadline_aware["traffic"]["required_rate_model"] == "deadline_aware"
    assert legacy["reasoning"]["planner_objective"] == "counterfactual"
    assert deadline_aware["reasoning"]["planner_objective"] == "counterfactual"


def test_episode_accumulators_do_not_share_counters() -> None:
    first = evaluator.EpisodeAccumulator(num_agents=2, interval_s=1.0)
    first.requested_actions["PREP"] = 7
    first.control_counters["handover_success"] = 3.0
    first.queue_samples.append(99.0)
    first.handover_completion_times[0].append(4.0)

    second = evaluator.EpisodeAccumulator(num_agents=2, interval_s=1.0)
    assert second.requested_actions["PREP"] == 0
    assert second.control_counters["handover_success"] == 0.0
    assert second.queue_samples == []
    assert second.handover_completion_times == [[], []]
