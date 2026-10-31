from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import leo_mappo.planner as planner_module
import leo_mappo.reasoning as reasoning_module
from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.residual_world_model import (
    FEATURE_DIM,
    MEASURED_SINR_INDEX,
    MEASUREMENT_AVAILABLE_INDEX,
    ResidualLinkPredictor,
    build_observable_link_features,
    normalize_observable_link_features,
)
from leo_mappo.world_model import GeometryWorldModel


ROOT = Path(__file__).resolve().parents[1]


def _v3_env() -> LEOMobilityEnv:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["num_ues"] = 1
    config["environment"]["episode_steps"] = 8
    env = LEOMobilityEnv(config)
    env.reset(seed=41)
    return env


def _geometry_model() -> GeometryWorldModel:
    return GeometryWorldModel(epsilon=1e-8, reliability_risk_temperature=0.10)


def test_predict_link_matches_existing_future_link_calculation() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    config["environment"]["num_ues"] = 1
    env = LEOMobilityEnv(config)
    env.reset(seed=3)
    model = _geometry_model()
    cell_id = env.ues[0].primary_cell
    assert cell_id is not None
    prediction = model.predict_link(env, 0, cell_id, offset=0)
    current = env.links[0][cell_id]
    loss_delta = 20.0 * np.log10(
        prediction.distance_m / (current.distance_m + model.epsilon)
    )
    expected_sinr = current.sinr_db - loss_delta
    expected_linear = 10.0 ** (np.clip(expected_sinr, -120.0, 80.0) / 10.0)
    expected_efficiency = min(
        float(env.channel_cfg["spectral_efficiency_cap"]),
        0.75 * np.log2(1.0 + expected_linear),
    )
    expected_success = 1.0 / (
        1.0 + np.exp(np.clip(-0.75 * (expected_sinr + 2.0), -40.0, 40.0))
    )
    assert prediction.visible
    assert np.isclose(prediction.sinr_db, expected_sinr)
    assert np.isclose(prediction.spectral_efficiency, expected_efficiency)
    assert np.isclose(prediction.success_probability, expected_success)
    assert np.allclose(
        model.future_link(env, 0, cell_id, 0),
        (prediction.spectral_efficiency, prediction.success_probability),
    )


def test_observable_features_ignore_hidden_v3_current_sinr() -> None:
    env = _v3_env()
    cell_id = env.ues[0].primary_cell
    assert cell_id is not None
    before = build_observable_link_features(env, 0, cell_id, _geometry_model())
    env.links[0][cell_id].sinr_db += 45.0
    after = build_observable_link_features(env, 0, cell_id, _geometry_model())
    assert np.array_equal(before, after)


def test_missing_measurement_is_zero_and_does_not_reveal_hidden_sinr() -> None:
    env = _v3_env()
    cell_id = env.ues[0].primary_cell
    assert cell_id is not None
    before = build_observable_link_features(
        env, 0, cell_id, _geometry_model(), measurement_available=False
    )
    env.links[0][cell_id].sinr_db -= 70.0
    after = build_observable_link_features(
        env, 0, cell_id, _geometry_model(), measurement_available=False
    )
    assert before[MEASURED_SINR_INDEX] == 0.0
    assert before[MEASUREMENT_AVAILABLE_INDEX] == 0.0
    assert np.array_equal(before, after)
    explicitly_missing = normalize_observable_link_features(
        measured_sinr_db=np.nan,
        observed_load=0.0,
        elevation_deg=0.0,
        distance_m=1_000_000.0,
        remaining_visibility_s=0.0,
        serving=False,
        candidate=False,
        measurement_available=False,
    )
    assert explicitly_missing[MEASURED_SINR_INDEX] == 0.0
    assert explicitly_missing[MEASUREMENT_AVAILABLE_INDEX] == 0.0


def test_residual_predictor_shape_and_finite_predictions() -> None:
    model = ResidualLinkPredictor(seed=9)
    histories = np.zeros((4, 8, FEATURE_DIM), dtype=np.float64)
    predictions = model.predict(histories)
    assert predictions.shape == (4, 6)
    assert np.all(np.isfinite(predictions))


def test_masked_targets_do_not_affect_loss_or_update() -> None:
    histories = np.linspace(0.0, 1.0, 3 * 8 * FEATURE_DIM).reshape(3, 8, FEATURE_DIM)
    masks = np.asarray(
        [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 0]],
        dtype=np.float64,
    )
    targets_a = np.zeros((3, 6), dtype=np.float64)
    targets_b = targets_a.copy()
    targets_b[masks == 0.0] = 1_000.0
    model_a = ResidualLinkPredictor(seed=17)
    model_b = ResidualLinkPredictor(seed=17)
    loss_a = model_a.train_batch(histories, targets_a, masks)
    loss_b = model_b.train_batch(histories, targets_b, masks)
    assert loss_a == loss_b
    for name in model_a.parameters:
        assert np.array_equal(model_a.parameters[name], model_b.parameters[name])


def test_model_save_load_reproduces_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    histories = rng.normal(size=(2, 8, FEATURE_DIM))
    model = ResidualLinkPredictor(seed=23)
    expected = model.predict(histories)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = ResidualLinkPredictor.load(path)
    assert np.array_equal(expected, restored.predict(histories))


def test_planner_and_reasoning_do_not_import_residual_predictor() -> None:
    assert "ResidualLinkPredictor" not in inspect.getsource(planner_module)
    assert "ResidualLinkPredictor" not in inspect.getsource(reasoning_module)
    assert planner_module.CHOPlanner.__init__.__annotations__["world_model"] in (
        "GeometryWorldModel | None",
        GeometryWorldModel | None,
    )
