from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .networks import Adam, GRU, MLP
from .world_model import GeometryWorldModel

if TYPE_CHECKING:
    from .environment import LEOMobilityEnv


OBSERVABLE_LINK_FEATURES = (
    "measured_sinr",
    "observed_load",
    "elevation",
    "distance",
    "remaining_visibility",
    "serving_indicator",
    "candidate_indicator",
    "measurement_available",
)
MEASURED_SINR_INDEX = 0
MEASUREMENT_AVAILABLE_INDEX = 7
FEATURE_DIM = len(OBSERVABLE_LINK_FEATURES)

# Fixed, dataset-independent physical normalization. These constants match the
# active observation scales where equivalents already exist.
SINR_OFFSET_DB = 20.0
SINR_SCALE_DB = 50.0
LOAD_SCALE = 2.0
ELEVATION_OFFSET_DEG = 10.0
ELEVATION_SCALE_DEG = 100.0
DISTANCE_SCALE_M = 3_000_000.0
VISIBILITY_SCALE_S = 600.0


def normalize_observable_link_features(
    measured_sinr_db: float,
    observed_load: float,
    elevation_deg: float,
    distance_m: float,
    remaining_visibility_s: float,
    serving: bool,
    candidate: bool,
    measurement_available: bool,
) -> np.ndarray:
    """Build one fixed-scale feature vector from controller-visible values.

    Normalization, in feature order:
      sinr=clip((dB+20)/50, 0, 1), or exactly 0 when unavailable;
      load=clip(load/2, 0, 1.5);
      elevation=clip((degrees+10)/100, 0, 1);
      distance=clip(metres/3,000,000, 0, 2);
      remaining visibility=clip(seconds/600, 0, 1);
      the final three indicators are binary.
    """
    available = bool(measurement_available)
    sinr_for_validation = measured_sinr_db if available else 0.0
    numeric = np.asarray(
        [
            sinr_for_validation,
            observed_load,
            elevation_deg,
            distance_m,
            remaining_visibility_s,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("observable link feature inputs must be finite")
    normalized_sinr = (
        float(np.clip((measured_sinr_db + SINR_OFFSET_DB) / SINR_SCALE_DB, 0.0, 1.0))
        if available
        else 0.0
    )
    features = np.asarray(
        [
            normalized_sinr,
            np.clip(observed_load / LOAD_SCALE, 0.0, 1.5),
            np.clip(
                (elevation_deg + ELEVATION_OFFSET_DEG) / ELEVATION_SCALE_DEG,
                0.0,
                1.0,
            ),
            np.clip(distance_m / DISTANCE_SCALE_M, 0.0, 2.0),
            np.clip(remaining_visibility_s / VISIBILITY_SCALE_S, 0.0, 1.0),
            float(bool(serving)),
            float(bool(candidate)),
            float(available),
        ],
        dtype=np.float64,
    )
    if features.shape != (FEATURE_DIM,) or not np.all(np.isfinite(features)):
        raise FloatingPointError("normalized observable features must be finite")
    return features


def build_observable_link_features(
    env: LEOMobilityEnv,
    ue_id: int,
    cell_id: int,
    world_model: GeometryWorldModel | None = None,
    measurement_available: bool | None = None,
) -> np.ndarray:
    """Extract only information available to the active controller.

    V3 uses delayed/noisy telemetry. It never falls back to the current hidden
    ``LinkResult.sinr_db``. V2's current link estimate is controller-visible and
    is therefore its closest legitimate measured-SINR equivalent.
    """
    if not 0 <= ue_id < env.num_agents or not 0 <= cell_id < env.num_cells:
        raise IndexError("UE or cell index is out of range")
    link = env.links[ue_id][cell_id]
    if env.v3_enabled:
        measured_sinr_db = float(env.observed_sinr[ue_id, cell_id])
        observed_load = float(env.observed_load[ue_id, cell_id])
        elevation_deg = float(env.observed_elevation[ue_id, cell_id])
        inferred_available = elevation_deg >= float(np.degrees(env.min_elevation_rad))
    else:
        measured_sinr_db = float(link.sinr_db)
        observed_load = float(env.last_pressure[cell_id])
        elevation_deg = float(link.elevation_deg)
        inferred_available = bool(link.visible)
    available = inferred_available if measurement_available is None else bool(measurement_available)

    geometry = world_model or GeometryWorldModel(
        epsilon=1e-8,
        reliability_risk_temperature=0.10,
    )
    _, distance_m = geometry.future_elevation_and_distance(env, ue_id, cell_id, 0)
    primary_cell = env.ues[ue_id].primary_cell
    return normalize_observable_link_features(
        measured_sinr_db=measured_sinr_db,
        observed_load=observed_load,
        elevation_deg=elevation_deg,
        distance_m=float(distance_m),
        remaining_visibility_s=float(env.remaining_visibility[ue_id, cell_id]),
        serving=primary_cell == cell_id,
        candidate=cell_id in env.candidates[ue_id],
        measurement_available=available,
    )


class ResidualLinkPredictor:
    """NumPy GRU/MLP predictor for future residual SINR in dB."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        history_steps: int = 8,
        horizon_steps: int = 6,
        hidden_dim: int = 64,
        mlp_hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 5.0,
        seed: int = 0,
    ):
        dimensions = (feature_dim, history_steps, horizon_steps, hidden_dim, mlp_hidden_dim)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("all model dimensions must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        self.feature_dim = int(feature_dim)
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)
        self.hidden_dim = int(hidden_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.gru = GRU(self.feature_dim, self.hidden_dim, rng)
        self.mlp = MLP(self.hidden_dim, self.mlp_hidden_dim, self.horizon_steps, rng)
        self.parameters = {
            **{f"gru.{name}": value for name, value in self.gru.parameters.items()},
            **{f"mlp.{name}": value for name, value in self.mlp.parameters.items()},
        }
        self.optimizer = Adam(self.parameters, learning_rate=self.learning_rate)

    def _validate_histories(self, histories: np.ndarray) -> np.ndarray:
        values = np.asarray(histories, dtype=np.float64)
        expected_tail = (self.history_steps, self.feature_dim)
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                "histories must have shape "
                f"[batch, {self.history_steps}, {self.feature_dim}], got {values.shape}"
            )
        if values.shape[0] == 0:
            raise ValueError("histories batch must not be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("histories must contain only finite values")
        return values

    def _forward(
        self, histories: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]], tuple[list[np.ndarray], list[np.ndarray]]]:
        sequence = np.transpose(histories, (1, 0, 2))
        initial_hidden = np.zeros((histories.shape[0], self.hidden_dim), dtype=np.float64)
        _, final_hidden, gru_cache = self.gru.forward_sequence(sequence, initial_hidden)
        predictions, mlp_cache = self.mlp.forward(final_hidden)
        if predictions.shape != (histories.shape[0], self.horizon_steps):
            raise RuntimeError("residual predictor produced an unexpected output shape")
        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError("residual predictor produced non-finite values")
        return predictions, gru_cache, mlp_cache

    def predict(self, histories: np.ndarray) -> np.ndarray:
        values = self._validate_histories(histories)
        predictions, _, _ = self._forward(values)
        return predictions

    def train_batch(
        self,
        histories: np.ndarray,
        target_residuals: np.ndarray,
        masks: np.ndarray,
    ) -> float:
        values = self._validate_histories(histories)
        targets = np.asarray(target_residuals, dtype=np.float64)
        valid_masks = np.asarray(masks, dtype=np.float64)
        expected = (values.shape[0], self.horizon_steps)
        if targets.shape != expected or valid_masks.shape != expected:
            raise ValueError(f"targets and masks must have shape {expected}")
        if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(valid_masks)):
            raise ValueError("targets and masks must contain only finite values")
        if np.any((valid_masks < 0.0) | (valid_masks > 1.0)):
            raise ValueError("masks must lie in [0, 1]")

        predictions, gru_cache, mlp_cache = self._forward(values)
        denominator = float(np.sum(valid_masks))
        if denominator <= 0.0:
            return 0.0
        errors = np.where(valid_masks > 0.0, predictions - targets, 0.0)
        loss = float(np.sum(valid_masks * errors**2) / denominator)
        grad_predictions = 2.0 * valid_masks * errors / denominator
        mlp_gradients, grad_final_hidden = self.mlp.backward_with_input(
            grad_predictions, mlp_cache
        )
        grad_hidden_sequence = np.zeros(
            (self.history_steps, values.shape[0], self.hidden_dim), dtype=np.float64
        )
        gru_gradients, _, _ = self.gru.backward_sequence(
            grad_hidden_sequence,
            gru_cache,
            grad_final_hidden=grad_final_hidden,
        )
        gradients = {
            **{f"gru.{name}": value for name, value in gru_gradients.items()},
            **{f"mlp.{name}": value for name, value in mlp_gradients.items()},
        }
        self.optimizer.step(gradients, max_grad_norm=self.max_grad_norm)
        if not np.isfinite(loss):
            raise FloatingPointError("masked residual loss is non-finite")
        return loss

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "feature_dim": np.asarray(self.feature_dim, dtype=np.int64),
            "history_steps": np.asarray(self.history_steps, dtype=np.int64),
            "horizon_steps": np.asarray(self.horizon_steps, dtype=np.int64),
            "hidden_dim": np.asarray(self.hidden_dim, dtype=np.int64),
            "mlp_hidden_dim": np.asarray(self.mlp_hidden_dim, dtype=np.int64),
            "learning_rate": np.asarray(self.learning_rate, dtype=np.float64),
            "max_grad_norm": np.asarray(self.max_grad_norm, dtype=np.float64),
            "seed": np.asarray(self.seed, dtype=np.int64),
        }
        arrays.update({name: value for name, value in self.parameters.items()})
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ResidualLinkPredictor:
        with np.load(Path(path), allow_pickle=False) as data:
            model = cls(
                feature_dim=int(data["feature_dim"]),
                history_steps=int(data["history_steps"]),
                horizon_steps=int(data["horizon_steps"]),
                hidden_dim=int(data["hidden_dim"]),
                mlp_hidden_dim=int(data["mlp_hidden_dim"]),
                learning_rate=float(data["learning_rate"]),
                max_grad_norm=float(data["max_grad_norm"]),
                seed=int(data["seed"]),
            )
            for name, parameter in model.parameters.items():
                if name not in data or data[name].shape != parameter.shape:
                    raise ValueError(f"model archive has no compatible parameter {name!r}")
                parameter[...] = data[name]
        return model
