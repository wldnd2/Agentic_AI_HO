from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import numpy as np


@dataclass
class EventEffects:
    traffic_multiplier: np.ndarray
    cell_capacity_multiplier: np.ndarray
    link_extra_loss_db: np.ndarray


class EventGenerator:
    """Composable events; labels are internal and never exposed in observations."""

    def __init__(
        self,
        event_specs: list[dict[str, Any]],
        num_ues: int,
        num_cells: int,
        rng: np.random.Generator,
        randomized: bool = False,
    ):
        self.event_specs = event_specs
        self.num_ues = num_ues
        self.num_cells = num_cells
        self.rng = rng
        self.randomized = bool(randomized)
        self.ue_targets: dict[int, np.ndarray] = {}
        self.realized_specs: list[dict[str, Any]] = []

    def _sample_range(self, spec: dict[str, Any], name: str, integer: bool) -> Any:
        bounds = spec.get(f"{name}_range")
        if bounds is None:
            return spec.get(name)
        low, high = bounds
        if integer:
            return int(self.rng.integers(int(low), int(high) + 1))
        return float(self.rng.uniform(float(low), float(high)))

    def reset(self) -> None:
        self.ue_targets.clear()
        self.realized_specs = []
        for index, template in enumerate(self.event_specs):
            spec = deepcopy(template)
            if self.randomized:
                for name, integer in (
                    ("start_step", True),
                    ("duration_steps", True),
                    ("severity", False),
                    ("ue_fraction", False),
                ):
                    sampled = self._sample_range(template, name, integer)
                    if sampled is not None:
                        spec[name] = sampled
                choices = template.get("satellite_choices")
                if choices is not None:
                    spec["satellite_id"] = int(self.rng.choice(np.asarray(choices, dtype=np.int64)))
            self.realized_specs.append(spec)
            fraction = float(spec.get("ue_fraction", 1.0))
            count = max(1, int(round(self.num_ues * fraction)))
            self.ue_targets[index] = self.rng.choice(self.num_ues, size=min(count, self.num_ues), replace=False)

    def resolve_dynamic_targets(self, step: int, primary_cells: list[int | None]) -> None:
        """Bind events with a `satellite_selector` to a satellite when they start.

        serving_random: uniformly among satellites currently serving >= 1 UE.
        busiest:        the satellite currently serving the most UEs.
        Satellite indices of a visibility-filtered Walker slice change with the
        geometry, so fixed `satellite_id`/`satellite_choices` may hit a satellite
        that is not visible at that time.
        """
        serving = [int(c) for c in primary_cells if c is not None]
        if not serving:
            return
        for spec in self.realized_specs:
            selector = spec.get("satellite_selector")
            if selector is None or spec.get("_resolved") or step < int(spec.get("start_step", 0)):
                continue
            cells, counts = np.unique(np.asarray(serving), return_counts=True)
            if selector == "busiest":
                spec["satellite_id"] = int(cells[np.argmax(counts)])
            elif selector == "serving_random":
                spec["satellite_id"] = int(self.rng.choice(cells))
            else:
                raise ValueError(f"unknown satellite_selector {selector!r}")
            spec["_resolved"] = True

    def effects(self, step: int) -> EventEffects:
        traffic = np.ones(self.num_ues, dtype=np.float64)
        capacity = np.ones(self.num_cells, dtype=np.float64)
        loss = np.zeros((self.num_ues, self.num_cells), dtype=np.float64)
        for index, spec in enumerate(self.realized_specs or self.event_specs):
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
                if spec.get("satellite_selector") is not None and not spec.get("_resolved"):
                    continue
                cell_id = int(spec.get("satellite_id", 0)) % self.num_cells
                capacity[cell_id] *= np.clip(severity, 0.05, 1.0)
            elif event_type == "blockage":
                if spec.get("satellite_selector") not in (None, "all") and not spec.get("_resolved"):
                    continue
                cell_id = spec.get("satellite_id")
                if cell_id is None:
                    loss[targets, :] += severity
                else:
                    loss[targets, int(cell_id) % self.num_cells] += severity
        return EventEffects(traffic, capacity, loss)

    def realization_summary(self) -> list[dict[str, Any]]:
        """Return JSON-friendly realized schedules without exposing them to actors."""
        summary: list[dict[str, Any]] = []
        for index, spec in enumerate(self.realized_specs or self.event_specs):
            summary.append(
                {
                    "type": str(spec.get("type", "unknown")),
                    "start_step": int(spec.get("start_step", 0)),
                    "duration_steps": int(spec.get("duration_steps", 0)),
                    "severity": float(spec.get("severity", 1.0)),
                    "satellite_id": (
                        None if spec.get("satellite_id") is None else int(spec["satellite_id"])
                    ),
                    "ue_targets": [int(value) for value in self.ue_targets.get(index, [])],
                }
            )
        return summary
