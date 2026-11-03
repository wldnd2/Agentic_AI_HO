from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.policy_diagnostics import (
    PolicyProbabilityDiagnostics,
    ReasoningFeatureDiagnostics,
    analyze_matched_reasoners,
    intervene_reasoning_features,
    model_parameter_digest,
    select_policy_actions,
)
from leo_mappo.reasoning import LookaheadReasoner
from leo_mappo.residual_world_model import ResidualLinkPredictor
from leo_mappo.runner import evaluation_episode_seeds
from leo_mappo.ue import MobilityState
from scripts.diagnose_reasoning_aware_mappo import evaluate_frozen_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def _agent(seed: int = 7) -> MAPPO:
    config = deepcopy(load_config(ROOT / "configs/default.yaml")["mappo"])
    config.update(
        use_memory=True,
        use_reasoning=True,
        reasoning_feature_mode="confidence_aware",
        recurrent_hidden_dim=4,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
    )
    return MAPPO(2, 5, 3, config, seed=seed)


def test_checkpoint_load_and_diagnostics_do_not_update_parameters(
    tmp_path: Path,
) -> None:
    source = _agent(seed=3)
    checkpoint = tmp_path / "weights.npz"
    source.save(checkpoint)
    loaded = _agent(seed=99)
    loaded.load(checkpoint)
    before = model_parameter_digest(loaded)
    reasoning = np.zeros((1, loaded.reasoning_feature_dim))
    loaded.recurrent_policy(
        np.zeros((1, 2)),
        np.ones((1, 3)),
        np.asarray([-1]),
        np.zeros((1, loaded.outcome_dim)),
        np.zeros((1, loaded.recurrent_hidden_dim)),
        reasoning,
    )

    assert model_parameter_digest(loaded) == before
    assert "agent.update" not in inspect.getsource(evaluate_frozen_checkpoint)


def test_deterministic_and_stochastic_action_modes_are_reproducible() -> None:
    probabilities = np.asarray([[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]])
    deterministic_agent = _agent(seed=4)
    np.testing.assert_array_equal(
        select_policy_actions(deterministic_agent, probabilities, True),
        [1, 0],
    )
    first = _agent(seed=8)
    second = _agent(seed=8)
    np.testing.assert_array_equal(
        select_policy_actions(first, probabilities, False),
        select_policy_actions(second, probabilities, False),
    )


def test_probability_logging_does_not_change_actions_or_probabilities() -> None:
    env = type(
        "DiagnosticEnv",
        (),
        {
            "prepare_start": 1,
            "candidate_count": 2,
            "commit_action": 3,
            "release_action": 4,
            "ues": [type("UE", (), {"ue_id": 0, "state": MobilityState.SINGLE})()],
        },
    )()
    masks = np.asarray([[1.0, 1.0, 1.0, 0.0, 0.0]])
    logits = np.asarray([[0.1, 0.2, 0.3, 9.0, 9.0]])
    probabilities = np.asarray([[0.30, 0.32, 0.38, 0.0, 0.0]])
    actions = probabilities.argmax(axis=-1)
    actions_before = actions.copy()
    probabilities_before = probabilities.copy()

    PolicyProbabilityDiagnostics().record(
        env, masks, logits, probabilities, [None]  # type: ignore[arg-type]
    )

    np.testing.assert_array_equal(actions, actions_before)
    np.testing.assert_array_equal(probabilities, probabilities_before)


def test_reasoning_feature_logging_uses_exact_actor_input() -> None:
    agent = _agent(seed=5)
    reasoning = np.asarray([[0.1, 0.2, 0.3, 0.0, 0.4, 0.7, 0.08]])
    captured: list[np.ndarray] = []
    original_forward = agent.actor.forward

    def capture(values: np.ndarray):
        captured.append(values.copy())
        return original_forward(values)

    with patch.object(agent.actor, "forward", side_effect=capture):
        agent.recurrent_policy(
            np.zeros((1, 2)),
            np.ones((1, 3)),
            np.asarray([-1]),
            np.zeros((1, agent.outcome_dim)),
            np.zeros((1, agent.recurrent_hidden_dim)),
            reasoning,
        )
    np.testing.assert_array_equal(captured[0][:, -7:], reasoning)

    env = type(
        "DiagnosticEnv",
        (),
        {
            "action_dim": 3,
            "num_agents": 1,
            "prepare_start": 1,
            "candidate_count": 1,
            "commit_action": 2,
            "release_action": 3,
            "ues": [
                type(
                    "UE",
                    (),
                    {"ue_id": 0, "state": type("State", (), {"name": "SINGLE"})()},
                )()
            ],
        },
    )()
    diagnostics = ReasoningFeatureDiagnostics()
    diagnostics.record(env, np.ones((1, 3)), reasoning)  # type: ignore[arg-type]
    rows = diagnostics.rows()
    prep_legacy = next(
        row
        for row in rows
        if row["action"] == "PREP_0" and row["feature_component"] == "legacy"
    )
    assert prep_legacy["mean"] == reasoning[0, 1]


def test_reasoning_interventions_preserve_actor_input_dimensionality() -> None:
    normal = np.arange(26, dtype=np.float64).reshape(2, 13)
    geometry = normal + 100.0
    for mode in ("normal", "geometry", "zero"):
        selected = intervene_reasoning_features(normal, geometry, mode, 13)
        assert selected.shape == (2, 13)
    np.testing.assert_array_equal(
        intervene_reasoning_features(normal, geometry, "zero", 13),
        0.0,
    )


def test_matched_comparison_does_not_double_advance_and_reset_clears_history(
    tmp_path: Path,
) -> None:
    config = load_config(
        ROOT / "configs/default.yaml",
        [
            ROOT / "configs/validation_agentic_challenge_v3.yaml",
            ROOT / "configs/deadline_aware_demand.yaml",
            ROOT / "configs/counterfactual_planner.yaml",
        ],
    )
    config["environment"]["num_ues"] = 2
    config["environment"]["episode_steps"] = 3
    checkpoint = tmp_path / "residual.npz"
    ResidualLinkPredictor(
        history_steps=2,
        horizon_steps=6,
        hidden_dim=4,
        mlp_hidden_dim=4,
        seed=2,
    ).save(checkpoint)
    hybrid_config = deepcopy(config["reasoning"])
    hybrid_config.update(
        world_model="hybrid",
        residual_model_path=str(checkpoint),
    )
    env = LEOMobilityEnv(config)
    _, _, masks = env.reset(seed=5001)
    hybrid = LookaheadReasoner(hybrid_config, env.reward_cfg)
    geometry = LookaheadReasoner(config["reasoning"], env.reward_cfg)

    analyze_matched_reasoners(env, masks, hybrid, geometry)
    first_lengths = [len(history) for history in hybrid.world_model._histories.values()]
    analyze_matched_reasoners(env, masks, hybrid, geometry)
    second_lengths = [len(history) for history in hybrid.world_model._histories.values()]
    assert first_lengths == second_lengths
    assert set(first_lengths) == {1}

    env.step(np.zeros(env.num_agents, dtype=np.int64))
    analyze_matched_reasoners(env, env.action_masks(), hybrid, geometry)
    assert set(len(history) for history in hybrid.world_model._histories.values()) == {2}
    env.reset(seed=5002)
    analyze_matched_reasoners(env, env.action_masks(), hybrid, geometry)
    assert set(len(history) for history in hybrid.world_model._histories.values()) == {1}


def test_matched_evaluation_seed_list_is_exact() -> None:
    assert evaluation_episode_seeds(10, 5001) == tuple(range(5001, 5011))
