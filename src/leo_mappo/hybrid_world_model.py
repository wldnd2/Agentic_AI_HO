from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .residual_world_model import (
    FEATURE_DIM,
    ResidualLinkPredictor,
    build_observable_link_features,
)
from .world_model import (
    FutureLinkPredictor,
    GeometryLinkPrediction,
    GeometryWorldModel,
)

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv


class ResidualInferencePredictor(Protocol):
    feature_dim: int
    history_steps: int
    horizon_steps: int

    def predict(self, histories: np.ndarray) -> np.ndarray: ...


class HybridWorldModel:
    """Geometry world model corrected by a frozen residual predictor."""

    def __init__(
        self,
        geometry_model: GeometryWorldModel,
        residual_predictor: ResidualInferencePredictor,
    ):
        self.geometry_model = geometry_model
        self._residual_predictor = residual_predictor
        self.history_steps = int(residual_predictor.history_steps)
        self.horizon_steps = int(residual_predictor.horizon_steps)
        feature_dim = int(residual_predictor.feature_dim)
        if self.history_steps <= 0 or self.horizon_steps <= 0:
            raise ValueError("residual predictor history and horizon must be positive")
        if feature_dim != FEATURE_DIM:
            raise ValueError(
                "residual predictor feature dimension does not match observable "
                f"feature construction: expected {FEATURE_DIM}, got {feature_dim}"
            )

        self.epsilon = geometry_model.epsilon
        self.reliability_risk_temperature = (
            geometry_model.reliability_risk_temperature
        )
        self._histories: defaultdict[tuple[int, int], deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.history_steps)
        )
        self._last_env: LEOMobilityEnv | None = None
        self._last_step_index: int | None = None
        self._last_links: object | None = None
        self._residual_cache: dict[tuple[int, int], np.ndarray | None] = {}

        self.hybrid_prediction_count = 0
        self.geometry_fallback_count = 0
        self.history_warmup_fallback_count = 0
        self.last_prediction_used_hybrid = False

    @classmethod
    def from_checkpoint(
        cls,
        geometry_model: GeometryWorldModel,
        checkpoint_path: str | Path,
    ) -> HybridWorldModel:
        predictor = ResidualLinkPredictor.load(checkpoint_path)
        return cls(geometry_model, predictor)

    @property
    def outage_probability_fn(self):
        return self.geometry_model.outage_probability_fn

    @outage_probability_fn.setter
    def outage_probability_fn(self, value) -> None:
        self.geometry_model.outage_probability_fn = value

    @property
    def reliability_risk_fn(self):
        return self.geometry_model.reliability_risk_fn

    @reliability_risk_fn.setter
    def reliability_risk_fn(self, value) -> None:
        self.geometry_model.reliability_risk_fn = value

    @property
    def goodputs_fn(self):
        return self.geometry_model.goodputs_fn

    @goodputs_fn.setter
    def goodputs_fn(self, value) -> None:
        self.geometry_model.goodputs_fn = value

    @property
    def diagnostics(self) -> dict[str, int]:
        return {
            "hybrid_prediction_count": self.hybrid_prediction_count,
            "geometry_fallback_count": self.geometry_fallback_count,
            "history_warmup_fallback_count": self.history_warmup_fallback_count,
        }

    def history(self, ue_id: int, cell_id: int) -> np.ndarray:
        """Return a diagnostic copy of the observable history for one pair."""
        entries = self._histories.get((ue_id, cell_id))
        if not entries:
            return np.empty((0, FEATURE_DIM), dtype=np.float64)
        return np.stack(entries).copy()

    def _clear_history(self) -> None:
        self._histories.clear()
        self._residual_cache.clear()

    def observe(self, env: LEOMobilityEnv) -> None:
        """Append every UE-cell observable feature at most once per env step."""
        step_index = int(env.step_index)
        same_snapshot = (
            env is self._last_env
            and step_index == self._last_step_index
            and env.links is self._last_links
        )
        if same_snapshot:
            return

        sequential = (
            env is self._last_env
            and self._last_step_index is not None
            and step_index == self._last_step_index + 1
        )
        if not sequential:
            self._clear_history()
        else:
            self._residual_cache.clear()

        for ue_id in range(env.num_agents):
            for cell_id in range(env.num_cells):
                feature = build_observable_link_features(
                    env,
                    ue_id,
                    cell_id,
                    self.geometry_model,
                )
                self._histories[(ue_id, cell_id)].append(feature.copy())

        self._last_env = env
        self._last_step_index = step_index
        self._last_links = env.links

    def _fallback(
        self,
        prediction: GeometryLinkPrediction,
        *,
        warmup: bool = False,
    ) -> GeometryLinkPrediction:
        self.geometry_fallback_count += 1
        if warmup:
            self.history_warmup_fallback_count += 1
        self.last_prediction_used_hybrid = False
        return prediction

    def _residuals_for_pair(self, ue_id: int, cell_id: int) -> np.ndarray | None:
        key = (ue_id, cell_id)
        if key in self._residual_cache:
            return self._residual_cache[key]
        history = self._histories[key]
        if len(history) < self.history_steps:
            return None
        batch = np.stack(history)[None, :, :]
        try:
            prediction = np.asarray(
                self._residual_predictor.predict(batch), dtype=np.float64
            )
        except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError):
            self._residual_cache[key] = None
            return None
        expected_shape = (1, self.horizon_steps)
        if prediction.shape != expected_shape or not np.all(np.isfinite(prediction)):
            self._residual_cache[key] = None
            return None
        residuals = prediction[0].copy()
        self._residual_cache[key] = residuals
        return residuals

    def future_elevation_and_distance(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int,
        offset: int,
    ) -> tuple[float, float]:
        return self.geometry_model.future_elevation_and_distance(
            env, ue_id, cell_id, offset
        )

    def predict_link(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int,
        offset: int,
    ) -> GeometryLinkPrediction:
        geometry = self.geometry_model.predict_link(env, ue_id, cell_id, offset)
        self.observe(env)
        if not geometry.visible or offset <= 0 or offset > self.horizon_steps:
            return self._fallback(geometry)

        history = self._histories[(ue_id, cell_id)]
        if len(history) < self.history_steps:
            return self._fallback(geometry, warmup=True)
        residuals = self._residuals_for_pair(ue_id, cell_id)
        if residuals is None:
            return self._fallback(geometry)
        corrected_sinr = geometry.sinr_db + float(residuals[offset - 1])
        if not np.isfinite(corrected_sinr):
            return self._fallback(geometry)
        efficiency, success = self.geometry_model.link_quality_from_sinr(
            env, corrected_sinr
        )
        self.hybrid_prediction_count += 1
        self.last_prediction_used_hybrid = True
        return GeometryLinkPrediction(
            visible=geometry.visible,
            elevation_rad=geometry.elevation_rad,
            distance_m=geometry.distance_m,
            sinr_db=float(corrected_sinr),
            spectral_efficiency=efficiency,
            success_probability=success,
        )

    def future_link(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int | None,
        offset: int,
    ) -> tuple[float, float]:
        if cell_id is None:
            self.geometry_fallback_count += 1
            self.last_prediction_used_hybrid = False
            return 0.0, 0.0
        prediction = self.predict_link(env, ue_id, cell_id, offset)
        return prediction.spectral_efficiency, prediction.success_probability

    def predicted_leg(
        self,
        env: LEOMobilityEnv,
        ue_id: int,
        cell_id: int | None,
        offset: int,
        required_rate: float,
        available: np.ndarray,
        future_link_predictor: FutureLinkPredictor | None = None,
    ) -> tuple[float, float, float, float]:
        predictor = self.future_link if future_link_predictor is None else future_link_predictor
        return self.geometry_model.predicted_leg(
            env,
            ue_id,
            cell_id,
            offset,
            required_rate,
            available,
            predictor,
        )

    def predicted_goodputs(
        self,
        service_transmission_rate_bps: float,
        potential_transmission_rate_bps: float,
        delivery_probability: float,
        preserve_reliability_headroom: bool,
    ) -> tuple[float, float]:
        return self.geometry_model.predicted_goodputs(
            service_transmission_rate_bps,
            potential_transmission_rate_bps,
            delivery_probability,
            preserve_reliability_headroom,
        )

    def predicted_outage_probability(
        self,
        transmission_rate_bps: float,
        success_probability: float,
        required_rate_bps: float,
        interval_s: float,
        packet_size_bits: int,
        outage_service_ratio: float,
    ) -> float:
        return self.geometry_model.predicted_outage_probability(
            transmission_rate_bps,
            success_probability,
            required_rate_bps,
            interval_s,
            packet_size_bits,
            outage_service_ratio,
        )

    def planning_reliability_risk(
        self,
        expected_goodput_bps: float,
        required_rate_bps: float,
        outage_service_ratio: float,
    ) -> float:
        return self.geometry_model.planning_reliability_risk(
            expected_goodput_bps,
            required_rate_bps,
            outage_service_ratio,
        )
