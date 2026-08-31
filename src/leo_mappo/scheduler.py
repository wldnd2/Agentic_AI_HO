from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LegRequest:
    ue_id: int
    cell_id: int
    spectral_efficiency: float
    priority: float
    demand_bps: float
    is_secondary: bool


@dataclass
class ScheduleResult:
    rates_bps: dict[tuple[int, int], float]
    fractions: dict[tuple[int, int], float]
    utilization: np.ndarray
    pressure: np.ndarray
    available_fraction: np.ndarray
    request_saturation_count: int


class QoSAwareFairScheduler:
    """Fixed weighted-fair scheduler; it is intentionally not controlled by MAPPO."""

    def __init__(self, request_epsilon: float = 1e-8):
        self.request_epsilon = float(request_epsilon)

    def allocate(
        self,
        requests: list[LegRequest],
        bandwidth_hz: np.ndarray,
        capacity_scale: np.ndarray,
        available_fraction: np.ndarray | None = None,
    ) -> ScheduleResult:
        num_cells = len(bandwidth_hz)
        rates: dict[tuple[int, int], float] = {}
        fractions: dict[tuple[int, int], float] = {}
        utilization = np.zeros(num_cells, dtype=np.float64)
        pressure = np.zeros(num_cells, dtype=np.float64)
        request_saturation_count = 0
        available = np.ones(num_cells, dtype=np.float64) if available_fraction is None else np.asarray(
            available_fraction, dtype=np.float64
        )
        if available.shape != (num_cells,) or np.any(available < -1e-12) or np.any(available > 1.0 + 1e-12):
            raise ValueError("available_fraction must be a per-cell vector in [0, 1]")
        for cell_id in range(num_cells):
            cell_requests = [request for request in requests if request.cell_id == cell_id]
            if not cell_requests:
                continue
            requested_fractions = []
            for request in cell_requests:
                unit_capacity = (
                    bandwidth_hz[cell_id]
                    * max(0.0, request.spectral_efficiency)
                    * max(0.0, capacity_scale[cell_id])
                )
                raw_fraction = request.demand_bps / (unit_capacity + self.request_epsilon)
                request_saturation_count += int(raw_fraction > 1.0)
                requested_fractions.append(min(1.0, max(0.0, raw_fraction)))
            pressure[cell_id] = float(sum(requested_fractions))
            weights = np.array(
                [
                    max(1e-6, fraction * request.priority)
                    for fraction, request in zip(requested_fractions, cell_requests)
                ],
                dtype=np.float64,
            )
            if weights.sum() <= available[cell_id]:
                shares = weights
            else:
                shares = available[cell_id] * weights / weights.sum()
            utilization[cell_id] = float(shares.sum())
            for request, share in zip(cell_requests, shares):
                key = (request.ue_id, cell_id)
                rate = share * bandwidth_hz[cell_id] * request.spectral_efficiency * capacity_scale[cell_id]
                rates[key] = float(rate)
                fractions[key] = float(share)
        return ScheduleResult(
            rates,
            fractions,
            utilization,
            pressure,
            available,
            request_saturation_count,
        )
