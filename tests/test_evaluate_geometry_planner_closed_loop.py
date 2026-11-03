from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from leo_mappo.demand import normalize_required_rate_model
from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.residual_world_model import ResidualLinkPredictor
from leo_mappo.world_model import GeometryWorldModel
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


@pytest.mark.parametrize("world_model", ("geometry", "hybrid"))
def test_same_reasoner_argmin_reaches_environment_step(
    world_model: str,
) -> None:
    class TinyEnvironment:
        def __init__(self) -> None:
            self.received_actions: np.ndarray | None = None

        def action_masks(self) -> np.ndarray:
            return np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

        def step(self, actions: np.ndarray) -> SimpleNamespace:
            self.received_actions = np.asarray(actions).copy()
            return SimpleNamespace(done=True)

    class TinyReasoner:
        def __init__(self, model_mode: str) -> None:
            self.world_model = model_mode

        def analyze(
            self, env: TinyEnvironment, masks: np.ndarray
        ) -> SimpleNamespace:
            assert np.array_equal(masks, env.action_masks())
            return SimpleNamespace(
                raw_costs=np.asarray([[0.5, 0.1, -9.0], [0.3, -8.0, 0.2]])
            )

    env = TinyEnvironment()
    _, actions, _, _ = evaluator.predictive_controller_step(  # type: ignore[arg-type]
        env,  # type: ignore[arg-type]
        TinyReasoner(world_model),  # type: ignore[arg-type]
    )
    assert actions.tolist() == [1, 2]
    assert env.received_actions is not None
    np.testing.assert_array_equal(env.received_actions, actions)


def test_omitted_world_model_defaults_to_geometry() -> None:
    args = evaluator.parse_args(["--output", "unused"])
    assert args.world_model == "geometry"
    assert args.residual_model is None


def test_geometry_mode_does_not_load_residual_checkpoint() -> None:
    config = {
        "reasoning": {
            "planner_objective": "counterfactual",
            "world_model": "hybrid",
            "residual_model_path": "must-not-be-loaded.npz",
        },
        "coordination": {},
    }
    env = SimpleNamespace(reward_cfg={})
    with patch.object(
        HybridWorldModel,
        "from_checkpoint",
        side_effect=AssertionError("residual checkpoint was loaded"),
    ) as loader:
        reasoner = evaluator.build_predictive_reasoner(
            config,
            env,  # type: ignore[arg-type]
        )
    loader.assert_not_called()
    assert type(reasoner.world_model) is GeometryWorldModel


def test_hybrid_mode_requires_residual_model() -> None:
    config = {"reasoning": {"planner_objective": "counterfactual"}}
    with pytest.raises(ValueError, match="--residual-model"):
        evaluator.build_predictive_reasoner(
            config,
            SimpleNamespace(reward_cfg={}),  # type: ignore[arg-type]
            world_model="hybrid",
        )


def test_valid_residual_checkpoint_reaches_hybrid_world_model(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.npz"
    ResidualLinkPredictor(
        history_steps=2,
        horizon_steps=3,
        hidden_dim=4,
        mlp_hidden_dim=4,
        seed=17,
    ).save(checkpoint)
    reasoner = evaluator.build_predictive_reasoner(
        {"reasoning": {"planner_objective": "counterfactual"}},
        SimpleNamespace(reward_cfg={}),  # type: ignore[arg-type]
        world_model="hybrid",
        residual_model=checkpoint,
    )
    assert isinstance(reasoner.world_model, HybridWorldModel)
    assert reasoner.planner.world_model is reasoner.world_model
    assert reasoner.world_model.history_steps == 2
    assert reasoner.world_model.horizon_steps == 3


def test_hybrid_checkpoint_load_failure_does_not_fall_back(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.npz"
    checkpoint.touch()
    with patch.object(
        HybridWorldModel,
        "from_checkpoint",
        side_effect=RuntimeError("checkpoint load failed"),
    ):
        with pytest.raises(RuntimeError, match="checkpoint load failed"):
            evaluator.build_predictive_reasoner(
                {"reasoning": {"planner_objective": "counterfactual"}},
                SimpleNamespace(reward_cfg={}),  # type: ignore[arg-type]
                world_model="hybrid",
                residual_model=checkpoint,
            )


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
    original_config = deepcopy(config)
    env = SimpleNamespace(reward_cfg={})
    with patch.object(evaluator, "LookaheadReasoner", FakeReasoner):
        evaluator.build_geometry_reasoner(config, env)  # type: ignore[arg-type]

    reasoning_config = captured["reasoning_config"]
    assert isinstance(reasoning_config, dict)
    assert reasoning_config["world_model"] == "geometry"
    assert "residual_model_path" not in reasoning_config
    assert captured["use_coordination"] is False
    assert captured["use_constraint"] is False
    assert config == original_config


def test_hybrid_summary_records_checkpoint_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.npz"
    ResidualLinkPredictor(
        history_steps=2,
        horizon_steps=3,
        hidden_dim=4,
        mlp_hidden_dim=4,
        seed=19,
    ).save(checkpoint)
    reasoner = evaluator.build_predictive_reasoner(
        {"reasoning": {"planner_objective": "counterfactual"}},
        SimpleNamespace(reward_cfg={}),  # type: ignore[arg-type]
        world_model="hybrid",
        residual_model=checkpoint,
    )
    row: dict[str, object] = {"episode": 0, "episode_seed": 5001}
    fields = (
        evaluator.TRAFFIC_QOS_FIELDS
        + evaluator.QUEUE_DELAY_FIELDS
        + evaluator.HANDOVER_CONTROL_FIELDS
        + evaluator.PROTOCOL_STATE_FIELDS
        + evaluator.SERVICE_DEMAND_FIELDS
        + evaluator.PREPARATION_CONTENTION_FIELDS
    )
    for field_name in fields:
        row[field_name] = [] if field_name == "handovers_per_ue" else 0.0

    summary = evaluator.build_summary(
        [row],  # type: ignore[list-item]
        ["configs/default.yaml"],
        5001,
        reasoner,
        "deadline_aware",
        world_model="hybrid",
        residual_model=checkpoint,
    )
    control = summary["scientific_control"]
    assert control["world_model"] == "hybrid"
    assert control["residual_model_path"] == str(checkpoint.resolve())
    assert control["residual_model_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert control["residual_history_length"] == 2
    assert control["residual_prediction_horizon"] == 3
    assert control["reasoner_controls_environment"] is True


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
