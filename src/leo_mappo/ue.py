from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .satellite import ground_position_ecef


class MobilityState(IntEnum):
    SINGLE = 0
    PREPARED = 1
    DUPLICATING = 2
    SWITCHING = 3
    COOLDOWN = 4


@dataclass
class PacketBatch:
    packets: int
    age_s: float = 0.0


@dataclass
class UE:
    ue_id: int
    latitude_rad: float
    longitude_rad: float
    mean_arrival_mbps: float
    packet_size_bits: int
    deadline_s: float
    reliability_target: float
    qos_priority: float
    max_queue_packets: int
    primary_cell: int | None = None
    secondary_cell: int | None = None
    state: MobilityState = MobilityState.SINGLE
    protocol_timer: int = 0
    last_primary_cell: int | None = None
    last_switch_step: int = -10_000
    queue: deque[PacketBatch] = field(default_factory=deque)
    generated_packets: int = 0
    delivered_packets: int = 0
    missed_packets: int = 0
    overflow_packets: int = 0
    # Traffic class (mixed-criticality scenario); the defaults reproduce the
    # single-class model.
    traffic_class: str = "default"
    critical: bool = False
    reward_weight: float = 1.0

    @property
    def position(self) -> np.ndarray:
        return ground_position_ecef(self.latitude_rad, self.longitude_rad)

    @property
    def queue_packets(self) -> int:
        return sum(batch.packets for batch in self.queue)

    @property
    def head_of_line_delay_s(self) -> float:
        return self.queue[0].age_s if self.queue else 0.0

    def reset_runtime(self) -> None:
        self.secondary_cell = None
        self.state = MobilityState.SINGLE
        self.protocol_timer = 0
        self.last_primary_cell = None
        self.last_switch_step = -10_000
        self.queue.clear()
        self.generated_packets = 0
        self.delivered_packets = 0
        self.missed_packets = 0
        self.overflow_packets = 0

    def generate_traffic(self, interval_s: float, multiplier: float, rng: np.random.Generator) -> int:
        mean_packets = self.mean_arrival_mbps * 1e6 * interval_s * multiplier / self.packet_size_bits
        arrivals = int(rng.poisson(max(0.0, mean_packets)))
        self.generated_packets += arrivals
        available = max(0, self.max_queue_packets - self.queue_packets)
        admitted = min(available, arrivals)
        dropped = arrivals - admitted
        if admitted:
            self.queue.append(PacketBatch(admitted))
        self.overflow_packets += dropped
        return arrivals

    def age_and_drop_expired(self, interval_s: float) -> int:
        missed = 0
        for batch in self.queue:
            batch.age_s += interval_s
        while self.queue and self.queue[0].age_s > self.deadline_s:
            missed += self.queue.popleft().packets
        self.missed_packets += missed
        return missed

    def deliver(self, packet_budget: int, success_probability: float, rng: np.random.Generator) -> int:
        attempted = min(self.queue_packets, max(0, packet_budget))
        successes = int(rng.binomial(attempted, np.clip(success_probability, 0.0, 1.0)))
        remaining = successes
        while remaining and self.queue:
            take = min(remaining, self.queue[0].packets)
            self.queue[0].packets -= take
            remaining -= take
            if self.queue[0].packets == 0:
                self.queue.popleft()
        self.delivered_packets += successes
        return successes

