from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EventEffects:
    traffic_multiplier: np.ndarray
    cell_capacity_multiplier: np.ndarray
    link_extra_loss_db: np.ndarray


class EventGenerator:
    """Composable events; labels are internal and never exposed in observations."""

    def __init__(self, event_specs: list[dict[str, Any]], num_ues: int, num_cells: int, rng: np.random.Generator):
        self.event_specs = event_specs
        self.num_ues = num_ues
        self.num_cells = num_cells
        self.rng = rng
        self.ue_targets: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self.ue_targets.clear()
        for index, spec in enumerate(self.event_specs):
            fraction = float(spec.get("ue_fraction", 1.0))
            count = max(1, int(round(self.num_ues * fraction)))
            self.ue_targets[index] = self.rng.choice(self.num_ues, size=min(count, self.num_ues), replace=False)

    def effects(self, step: int) -> EventEffects:
        traffic = np.ones(self.num_ues, dtype=np.float64)
        capacity = np.ones(self.num_cells, dtype=np.float64)
        loss = np.zeros((self.num_ues, self.num_cells), dtype=np.float64)
        for index, spec in enumerate(self.event_specs):
            start = int(spec.get("start_step", 0))
            duration = int(spec.get("duration_steps", 0))
            if not (start <= step < start + duration):
                continue
            event_type = spec.get("type")
            severity = float(spec.get("severity", 1.0))
            targets = self.ue_targets.get(index, np.arange(self.num_ues))
            if event_type == "traffic_burst":
                traffic[targets] *= severity
            elif event_type == "capacity_drop":
                cell_id = int(spec.get("satellite_id", 0)) % self.num_cells
                capacity[cell_id] *= np.clip(severity, 0.05, 1.0)
            elif event_type == "blockage":
                cell_id = spec.get("satellite_id")
                if cell_id is None:
                    loss[targets, :] += severity
                else:
                    loss[targets, int(cell_id) % self.num_cells] += severity
        return EventEffects(traffic, capacity, loss)

