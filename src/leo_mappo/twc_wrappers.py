"""Component wrapper that gives a policy-gradient learner the EG-CMADQN components.

The memoryless CTDE-MAPPO baseline (M1) learns from the team reward returned by
the environment, so C2-C4 are attached around the environment instead of inside
MAPPO:

C2  team potential-based shaping with Phi(t) = mean_i(-min_a F_i(a)), i.e. the
    mean of the per-UE ephemeris potentials of EG-CMADQN (a potential of the
    team state, so the shaping keeps the potential-based form);
C3  outage and signalling penalties with the same dual variables (updated once
    per iteration from rollout averages);
C4  local congestion prices appended to every observation.

C1 (ephemeris-guided exploration) needs an off-policy learner and is not
applicable to on-policy MAPPO.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .coordinator import SatellitePriceCoordinator
from .egcmadqn import Components, DualVariable, TrainerConfig
from .environment import LEOMobilityEnv
from .reasoning import LookaheadReasoner


class ComponentEnvWrapper:
    """Drop-in replacement for LEOMobilityEnv in RolloutRunner (memoryless MAPPO)."""

    def __init__(self, env: LEOMobilityEnv, components: Components, trainer_config: TrainerConfig | None = None,
                 gamma: float = 0.99):
        if components.guided_exploration:
            raise ValueError("C1 (guided exploration) is not applicable to on-policy MAPPO")
        self.env = env
        self.c = components
        self.tc = trainer_config or TrainerConfig()
        self.gamma = float(gamma)
        cfg = env.config
        self.coordinator = (
            SatellitePriceCoordinator(env.num_cells, cfg.get("coordination", {})) if self.c.price_input else None
        )
        self.price_dim = env.candidate_count + 1 if self.c.price_input else 0
        self.obs_dim = env.obs_dim + self.price_dim
        self.reasoner = (
            LookaheadReasoner(cfg.get("reasoning", {}), env.reward_cfg, cfg.get("coordination", {}),
                              use_coordination=self.c.price_input, use_constraint=False)
            if self.c.potential_shaping else None
        )
        self.dual_out = DualVariable(self.tc.outage_target, self.tc.dual_learning_rate_out, self.tc.dual_max)
        self.dual_sig = DualVariable(self.tc.signaling_target, self.tc.dual_learning_rate_sig, self.tc.dual_max)
        self._phi = 0.0
        self._masks: np.ndarray | None = None
        self._reset_accumulators()

    # Everything not overridden (ues, candidates, action_masks, num_agents, ...)
    # is served by the wrapped environment.
    def __getattr__(self, name):
        # Guard so that unpickling (before `env` exists) does not recurse.
        if name == "env" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    # ------------------------------------------------------------- helpers
    def _augment(self, observations: np.ndarray) -> np.ndarray:
        if self.coordinator is None:
            return observations
        price = self.coordinator.local_features([u.primary_cell for u in self.env.ues], self.env.candidates,
                                                self.env.candidate_count)
        return np.concatenate([observations, price], axis=1)

    def _team_potential(self, masks: np.ndarray) -> float:
        if self.reasoner is None:
            return 0.0
        prices = self.coordinator.prices if self.coordinator is not None else np.zeros(self.env.num_cells)
        costs = self.reasoner.analyze(self.env, masks, prices, 0.0).raw_costs
        return float(np.mean(-np.min(np.where(masks > 0.0, costs, np.inf), axis=1)))

    def _reset_accumulators(self) -> None:
        self._acc = {"common_reward": 0.0, "training_reward": 0.0, "outage": 0.0, "signaling": 0.0,
                     "handover_started": 0.0, "shaping": 0.0, "pingpong": 0.0, "handover_success": 0.0}
        self._steps = 0

    # ---------------------------------------------------------- environment
    def reset(self, seed: int | None = None):
        observations, global_state, masks = self.env.reset(seed=seed)
        if self.coordinator is not None:
            self.coordinator.reset()
        self._masks = masks
        self._phi = self._team_potential(masks)
        return self._augment(observations), global_state, masks

    def step(self, actions: np.ndarray):
        out = self.env.step(actions)
        if self.coordinator is not None:
            self.coordinator.update(out.requested_pressure)
        base = float(np.mean(out.rewards))
        outage, signaling = float(out.info["outage_ratio"]), float(out.info["signaling_cost"])
        reward = base
        if self.c.dual_constraints:
            reward -= self.dual_out.value * (outage - self.tc.outage_target)
            reward -= self.dual_sig.value * (signaling - self.tc.signaling_target)
        phi_next = 0.0 if out.done else self._team_potential(out.action_masks)
        shaping = self.tc.shaping_scale * (self.gamma * phi_next - self._phi) if self.c.potential_shaping else 0.0
        self._phi = phi_next
        rewards = np.full_like(out.rewards, reward + shaping)
        info = dict(out.info)
        info.update(common_reward=base, shaping=shaping, wrapped_training_reward=reward + shaping)
        acc = self._acc
        acc["common_reward"] += base
        acc["training_reward"] += reward + shaping
        acc["outage"] += outage
        acc["signaling"] += signaling
        acc["shaping"] += shaping
        for key in ("handover_started", "pingpong", "handover_success"):
            acc[key] += float(out.info.get(key, 0.0))
        self._steps += 1
        return dataclasses.replace(out, observations=self._augment(out.observations), rewards=rewards, info=info)

    def end_iteration(self) -> dict[str, float]:
        """Per-iteration averages; updates the dual variables (C3)."""
        n = max(self._steps, 1)
        metrics = {k: v / n for k, v in self._acc.items() if k not in ("pingpong", "handover_success")}
        metrics["pingpong_ratio"] = self._acc["pingpong"] / max(self._acc["handover_success"], 1.0)
        if self.c.dual_constraints:
            metrics["lambda_out"] = self.dual_out.update(metrics["outage"])
            metrics["lambda_sig"] = self.dual_sig.update(metrics["signaling"])
        self._reset_accumulators()
        return metrics
