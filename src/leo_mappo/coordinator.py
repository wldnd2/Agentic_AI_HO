from __future__ import annotations

import numpy as np


def compute_training_reward(
    service_deficit: float,
    resource_cost: float,
    outage_cost: float,
    lambda_resource: float,
    use_constraint: bool,
    outage_dual_mu: float = 0.0,
    outage_target: float = 0.0,
) -> tuple[float, float]:
    """Single source of truth for Core and reliability-constrained rewards.

    C_ctrl = eta_preparation*C_prep + eta_handover*C_HO
    M1-M4: -C_def - lambda_resource*C_ctrl
    M5:    -C_def - lambda_resource*C_ctrl - mu*(C_out-outage_target)

    M5 is exactly the M4 base reward plus the reliability Lagrangian term;
    when mu == 0, the M5 reward reduces exactly to the M4 reward. The hard
    per-satellite physical-capacity constraint remains enforced by the scheduler.
    """
    base_reward = -service_deficit - lambda_resource * resource_cost
    constraint_penalty = (
        outage_dual_mu * (outage_cost - outage_target)
        if use_constraint
        else 0.0
    )
    reward = base_reward - constraint_penalty
    if not np.isfinite(reward) or not np.isfinite(constraint_penalty):
        raise FloatingPointError("training reward is non-finite")
    return float(reward), float(constraint_penalty)


class SatellitePriceCoordinator:
    """Episode-local slot-level congestion-price controller."""

    def __init__(self, num_satellites: int, config: dict):
        self.num_satellites = int(num_satellites)
        self.learning_rate = float(config.get("price_learning_rate", 0.05))
        self.price_max = float(config.get("price_max", 5.0))
        self.initial_price = float(config.get("initial_price", 0.0))
        self.pressure_clip = float(config.get("pressure_clip", 10.0))
        if self.num_satellites <= 0 or self.price_max <= 0.0 or self.learning_rate < 0.0:
            raise ValueError("invalid satellite price coordinator configuration")
        self._prices = np.zeros(self.num_satellites, dtype=np.float64)
        self.reset()

    @property
    def prices(self) -> np.ndarray:
        return self._prices.copy()

    def reset(self) -> None:
        self._prices.fill(np.clip(self.initial_price, 0.0, self.price_max))

    def update(self, requested_pressure: np.ndarray) -> np.ndarray:
        pressure = np.asarray(requested_pressure, dtype=np.float64)
        if pressure.shape != (self.num_satellites,) or not np.all(np.isfinite(pressure)):
            raise ValueError("requested_pressure must be a finite per-satellite vector")
        if np.any(pressure < 0.0):
            raise ValueError("requested_pressure cannot be negative")
        self._prices = np.clip(
            self._prices + self.learning_rate * (pressure - 1.0),
            0.0,
            self.price_max,
        )
        if not np.all(np.isfinite(self._prices)):
            raise FloatingPointError("satellite price update produced non-finite values")
        return self.prices

    def normalized_prices(self) -> np.ndarray:
        return self._prices / self.price_max

    def local_features(
        self,
        primary_cells: list[int | None],
        candidates: list[list[int]],
        candidate_count: int,
    ) -> np.ndarray:
        result = np.zeros((len(primary_cells), int(candidate_count) + 1), dtype=np.float64)
        normalized = self.normalized_prices()
        for ue_id, primary in enumerate(primary_cells):
            if primary is not None:
                result[ue_id, 0] = normalized[primary]
            for slot, cell_id in enumerate(candidates[ue_id][:candidate_count]):
                result[ue_id, slot + 1] = normalized[cell_id]
        return result


class ReliabilityDualController:
    """Training-global rollout-level reliability multiplier."""

    def __init__(self, config: dict, initial_override: float | None = None):
        self.outage_target = float(config.get("outage_target", 0.10))
        self.learning_rate = float(config.get("dual_learning_rate", 0.02))
        self.mu_initial = float(config.get("mu_initial", 0.0))
        self.mu_max = float(config.get("mu_max", 10.0))
        if not 0.0 <= self.outage_target <= 1.0 or self.learning_rate < 0.0 or self.mu_max <= 0.0:
            raise ValueError("invalid reliability dual configuration")
        initial = self.mu_initial if initial_override is None else float(initial_override)
        self._mu = float(np.clip(initial, 0.0, self.mu_max))

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def normalized_mu(self) -> float:
        return self._mu / self.mu_max

    def reset_training(self) -> None:
        self._mu = float(np.clip(self.mu_initial, 0.0, self.mu_max))

    def update(self, outage_count: float, active_count: float, epsilon: float = 1e-8) -> float:
        outage_ratio = float(outage_count) / (float(active_count) + epsilon)
        self._mu = float(
            np.clip(
                self._mu + self.learning_rate * (outage_ratio - self.outage_target),
                0.0,
                self.mu_max,
            )
        )
        if not np.isfinite(self._mu):
            raise FloatingPointError("reliability dual update produced a non-finite value")
        return self._mu
