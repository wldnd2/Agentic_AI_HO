from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import leo_mappo.planner as planner_module
from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.hybrid_world_model import HybridWorldModel
from leo_mappo.reasoning import LookaheadReasoner
from leo_mappo.residual_world_model import FEATURE_DIM, ResidualLinkPredictor
from leo_mappo.world_model import GeometryWorldModel


ROOT = Path(__file__).resolve().parents[1]


class _FakeResidualPredictor:
    feature_dim = FEATURE_DIM

    def __init__(
        self,
        residuals: list[float],
        history_steps: int = 1,
    ) -> None:
        self.history_steps = history_steps
        self.horizon_steps = len(residuals)
        self.residuals = np.asarray(residuals, dtype=np.float64)
        self.call_count = 0
        self.last_history: np.ndarray | None = None

    def predict(self, histories: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self.last_history = histories.copy()
        return np.broadcast_to(self.residuals, (histories.shape[0], self.horizon_steps)).copy()


def _geometry() -> GeometryWorldModel:
    return GeometryWorldModel(epsilon=1e-8, reliability_risk_temperature=0.10)


def _v3_env() -> LEOMobilityEnv:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["num_ues"] = 1
    config["environment"]["episode_steps"] = 8
    env = LEOMobilityEnv(config)
    env.reset(seed=61)
    env.min_elevation_rad = -np.pi / 2.0
    return env


def _visible_cell(env: LEOMobilityEnv, geometry: GeometryWorldModel, offset: int = 1) -> int:
    visible = [
        cell_id
        for cell_id in range(env.num_cells)
        if geometry.predict_link(env, 0, cell_id, offset).visible
    ]
    return min(
        visible,
        key=lambda cell_id: geometry.predict_link(env, 0, cell_id, offset).sinr_db,
    )


def test_default_reasoner_remains_geometry_only() -> None:
    reasoner = LookaheadReasoner({}, {})
    assert type(reasoner.world_model) is GeometryWorldModel
    assert reasoner.planner.world_model is reasoner.world_model


def test_hybrid_reasoner_loads_frozen_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.npz"
    ResidualLinkPredictor(
        history_steps=1,
        horizon_steps=2,
        hidden_dim=4,
        mlp_hidden_dim=4,
        seed=5,
    ).save(checkpoint)
    reasoner = LookaheadReasoner(
        {
            "world_model": "hybrid",
            "residual_model_path": str(checkpoint),
        },
        {},
    )
    assert isinstance(reasoner.world_model, HybridWorldModel)
    assert reasoner.planner.world_model is reasoner.world_model


def test_hybrid_mode_requires_valid_checkpoint_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="residual_model_path"):
        LookaheadReasoner({"world_model": "hybrid"}, {})
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LookaheadReasoner(
            {
                "world_model": "hybrid",
                "residual_model_path": str(tmp_path / "missing.npz"),
            },
            {},
        )


def test_zero_residual_matches_geometry_prediction() -> None:
    env = _v3_env()
    geometry = _geometry()
    cell_id = _visible_cell(env, geometry)
    hybrid = HybridWorldModel(geometry, _FakeResidualPredictor([0.0, 0.0]))
    assert hybrid.predict_link(env, 0, cell_id, 1) == geometry.predict_link(
        env, 0, cell_id, 1
    )
    assert hybrid.hybrid_prediction_count == 1


def test_known_residual_updates_sinr_efficiency_and_success() -> None:
    env = _v3_env()
    env.observed_sinr[0, :] = -5.0
    geometry = _geometry()
    cell_id = min(
        (
            cell
            for cell in range(env.num_cells)
            if geometry.predict_link(env, 0, cell, 1).visible
            and geometry.predict_link(env, 0, cell, 2).visible
        ),
        key=lambda cell: geometry.predict_link(env, 0, cell, 1).sinr_db,
    )
    hybrid = HybridWorldModel(geometry, _FakeResidualPredictor([2.0, -1.5]))
    baseline = geometry.predict_link(env, 0, cell_id, 1)
    prediction = hybrid.predict_link(env, 0, cell_id, 1)
    expected_sinr = baseline.sinr_db + 2.0
    expected_efficiency, expected_success = geometry.link_quality_from_sinr(
        env, expected_sinr
    )
    assert prediction.sinr_db == expected_sinr
    assert prediction.spectral_efficiency == expected_efficiency
    assert prediction.success_probability == expected_success
    assert prediction.spectral_efficiency != baseline.spectral_efficiency
    assert prediction.success_probability != baseline.success_probability
    baseline_h2 = geometry.predict_link(env, 0, cell_id, 2)
    prediction_h2 = hybrid.predict_link(env, 0, cell_id, 2)
    assert prediction_h2.sinr_db == baseline_h2.sinr_db - 1.5


def test_warmup_uses_geometry_only() -> None:
    env = _v3_env()
    geometry = _geometry()
    cell_id = _visible_cell(env, geometry)
    predictor = _FakeResidualPredictor([4.0, 4.0], history_steps=2)
    hybrid = HybridWorldModel(geometry, predictor)
    prediction = hybrid.predict_link(env, 0, cell_id, 1)
    assert prediction == geometry.predict_link(env, 0, cell_id, 1)
    assert predictor.call_count == 0
    assert hybrid.history_warmup_fallback_count == 1


def test_horizon_bounds_use_geometry_only() -> None:
    env = _v3_env()
    geometry = _geometry()
    cell_id = _visible_cell(env, geometry)
    predictor = _FakeResidualPredictor([3.0, 3.0])
    hybrid = HybridWorldModel(geometry, predictor)
    assert hybrid.predict_link(env, 0, cell_id, 0) == geometry.predict_link(
        env, 0, cell_id, 0
    )
    assert hybrid.predict_link(env, 0, cell_id, 3) == geometry.predict_link(
        env, 0, cell_id, 3
    )
    assert predictor.call_count == 0


def test_invisible_geometry_link_cannot_be_corrected_visible() -> None:
    env = _v3_env()
    env.min_elevation_rad = np.pi / 2.0
    geometry = _geometry()
    cell_id = 0
    predictor = _FakeResidualPredictor([100.0, 100.0])
    hybrid = HybridWorldModel(geometry, predictor)
    prediction = hybrid.predict_link(env, 0, cell_id, 1)
    assert not prediction.visible
    assert prediction == geometry.predict_link(env, 0, cell_id, 1)
    assert predictor.call_count == 0


def test_repeated_predictions_do_not_duplicate_current_history() -> None:
    env = _v3_env()
    geometry = _geometry()
    cell_id = _visible_cell(env, geometry)
    hybrid = HybridWorldModel(
        geometry,
        _FakeResidualPredictor([1.0, 1.0], history_steps=2),
    )
    hybrid.predict_link(env, 0, cell_id, 1)
    hybrid.predict_link(env, 0, cell_id, 1)
    hybrid.observe(env)
    assert hybrid.history(0, cell_id).shape == (1, FEATURE_DIM)


def test_history_uses_observable_not_hidden_v3_sinr() -> None:
    env = _v3_env()
    geometry = _geometry()
    cell_id = _visible_cell(env, geometry)
    hybrid = HybridWorldModel(
        geometry,
        _FakeResidualPredictor([0.0, 0.0], history_steps=2),
    )
    hybrid.predict_link(env, 0, cell_id, 1)
    first = hybrid.history(0, cell_id)[0].copy()
    env.links[0][cell_id].sinr_db += 80.0
    env.step_index += 1
    hybrid.predict_link(env, 0, cell_id, 1)
    second = hybrid.history(0, cell_id)[1]
    assert np.array_equal(first, second)


def test_planner_has_no_residual_predictor_dependency() -> None:
    source = inspect.getsource(planner_module)
    assert "ResidualLinkPredictor" not in source
    assert "HybridWorldModel" not in source
