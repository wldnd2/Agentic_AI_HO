from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from leo_mappo import MAPPO, load_config
from leo_mappo.reasoning import (
    build_actor_reasoning_features,
    reasoning_feature_dim,
)


ROOT = Path(__file__).resolve().parents[1]


def _reasoning_result(
    features: np.ndarray,
    raw_costs: np.ndarray,
    score_gaps: np.ndarray,
) -> SimpleNamespace:
    return SimpleNamespace(
        features=np.asarray(features, dtype=np.float64),
        raw_costs=np.asarray(raw_costs, dtype=np.float64),
        score_gaps=np.asarray(score_gaps, dtype=np.float64),
    )


def _recurrent_config(mode: str) -> dict:
    config = deepcopy(load_config(ROOT / "configs/default.yaml")["mappo"])
    config.update(
        use_memory=True,
        use_reasoning=True,
        reasoning_feature_mode=mode,
        recurrent_hidden_dim=4,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
    )
    return config


def test_legacy_reasoning_features_remain_exactly_available() -> None:
    legacy = np.asarray([[0.0, 0.25, 1.0]])
    result = _reasoning_result(legacy, [[-0.1, 0.0, 0.2]], [0.1])
    masks = np.ones((1, 3), dtype=np.float64)

    actual = build_actor_reasoning_features(result, masks, 5.0, "legacy")

    np.testing.assert_array_equal(actual, legacy)
    assert actual is not result.features
    assert reasoning_feature_dim(3, "legacy") == 3


def test_confidence_aware_features_are_finite_deterministic_and_tie_aware() -> None:
    legacy = np.asarray([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    raw = np.asarray([[0.0, 0.0, 5.0], [0.0, 1.0e-9, 5.0]])
    gaps = np.asarray([0.0, 1.0e-9])
    masks = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    result = _reasoning_result(legacy, raw.copy(), gaps)

    first = build_actor_reasoning_features(
        result, masks.copy(), 5.0, "confidence_aware"
    )
    second = build_actor_reasoning_features(
        result, masks.copy(), 5.0, "confidence_aware"
    )

    assert first.shape == (2, reasoning_feature_dim(3, "confidence_aware"))
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, :3], legacy)
    np.testing.assert_array_equal(first[0, 3:6], 0.0)
    np.testing.assert_allclose(first[1, 3:6], [0.0, 2.0e-10, 0.0])
    np.testing.assert_allclose(first[:, -1], [0.0, 2.0e-10])
    np.testing.assert_array_equal(result.raw_costs, raw)
    np.testing.assert_array_equal(masks[:, 2], 0.0)


def test_actor_can_disagree_with_reasoner_while_mask_remains_authoritative() -> None:
    agent = MAPPO(2, 5, 3, _recurrent_config("confidence_aware"), seed=4)
    for parameter in agent.actor.parameters.values():
        parameter.fill(0.0)
    output_bias = agent.actor.parameters[f"b{len(agent.actor.hidden_dims) + 1}"]
    output_bias[:] = [0.0, 10.0, 100.0]
    masks = np.asarray([[1.0, 1.0, 0.0]])
    # The signal says action 0 is the analytical best, but does not execute it.
    reasoning = np.asarray([[0.0, 1.0, 1.0, 0.0, 0.2, 0.4, 0.04]])

    actions, _, _ = agent.act_recurrent(
        np.zeros((1, 2)),
        masks,
        np.asarray([-1]),
        np.zeros((1, agent.outcome_dim)),
        np.zeros((1, agent.recurrent_hidden_dim)),
        reasoning,
        deterministic=True,
    )

    # Learned logits select feasible action 1; infeasible action 2 cannot win.
    np.testing.assert_array_equal(actions, [1])


def test_reasoning_mode_does_not_change_recurrent_hidden_state_path() -> None:
    legacy_agent = MAPPO(2, 5, 3, _recurrent_config("legacy"), seed=12)
    confidence_agent = MAPPO(
        2, 5, 3, _recurrent_config("confidence_aware"), seed=12
    )
    assert legacy_agent.gru is not None
    assert confidence_agent.gru is not None
    for key in legacy_agent.gru.parameters:
        np.testing.assert_array_equal(
            legacy_agent.gru.parameters[key], confidence_agent.gru.parameters[key]
        )
    common = (
        np.asarray([[0.2, -0.1]]),
        np.ones((1, 3)),
        np.asarray([1]),
        np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        np.asarray([[0.3, -0.2, 0.1, 0.0]]),
    )

    *_, legacy_hidden = legacy_agent.act_recurrent(
        *common, np.zeros((1, 3)), deterministic=True
    )
    *_, confidence_hidden = confidence_agent.act_recurrent(
        *common, np.zeros((1, 7)), deterministic=True
    )

    np.testing.assert_array_equal(legacy_hidden, confidence_hidden)


def test_critic_input_stays_centralized_and_actor_input_stays_local() -> None:
    agent = MAPPO(2, 11, 3, _recurrent_config("confidence_aware"), seed=8)

    assert agent.critic.input_dim == 11
    assert agent.actor.input_dim == 2 + agent.recurrent_hidden_dim + 7
    assert agent.gru is not None
    assert agent.gru.input_dim == 2 + 3 + agent.outcome_dim
