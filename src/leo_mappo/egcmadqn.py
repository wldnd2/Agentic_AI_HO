"""Ephemeris-guided constrained multi-agent DQN (EG-CMADQN).

Extends the parameter-shared fingerprint Double DQN (``madqn.MADQN``) with

C1  ephemeris-guided exploration: an exploring agent samples, with probability
    beta_k, from softmax(-F_i(a)/T_g) over its feasible actions instead of the
    uniform distribution. F_i(a) is the planner cost of the LookaheadReasoner.
    DQN is off-policy, so the behaviour distribution does not bias the target.
C2  ephemeris potential-based shaping:
        Phi_i(t) = -min_a F_i(a),   r_i <- r_i + kappa (gamma Phi_i(t+1) - Phi_i(t)),
    with Phi = 0 at episode ends.
C3  dual constraints on the outage and the preparation-signalling cost:
        r <- r - lambda_out (C_out - C_out_thr) - lambda_sig (C_sig - C_sig_thr),
    with projected dual ascent after every iteration. The normalised
    multipliers extend the training fingerprint. Two details matter for an
    off-policy learner:
    * reward relabelling - the replay buffer stores the reward components
      (base reward, shaping, outage and signalling costs) and the Lagrangian
      reward is recomputed with the *current* multipliers when a batch is
      sampled, so that replayed transitions do not carry stale multipliers;
    * target-policy constraint estimates - the multipliers are updated from the
      decisions of agents that acted greedily in that epoch (exploratory
      actions are excluded), because the constraint concerns the learned
      policy, not the exploration noise.
C4  satellite congestion prices (serving + candidates) appended to the input.

The planner is used only during training (C1, C2); execution needs one forward
pass of the Q-network plus the price features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coordinator import SatellitePriceCoordinator
from .environment import LEOMobilityEnv
from .madqn import MADQN, ReplayBuffer
from .reasoning import LookaheadReasoner


@dataclass(frozen=True)
class Components:
    guided_exploration: bool = True   # C1
    potential_shaping: bool = True    # C2
    dual_constraints: bool = True     # C3
    price_input: bool = True          # C4

    @property
    def needs_planner(self) -> bool:
        return self.guided_exploration or self.potential_shaping


VARIANTS: dict[str, Components] = {
    "proposed": Components(),
    "no_c1": Components(guided_exploration=False),
    "no_c2": Components(potential_shaping=False),
    "no_c3": Components(dual_constraints=False),
    "no_c4": Components(price_input=False),
    "madqn": Components(False, False, False, False),
    # Constrained MADQN without the planner (C3 only / C3 + prices).
    "c3_only": Components(guided_exploration=False, potential_shaping=False, dual_constraints=True, price_input=False),
    "c3_c4": Components(guided_exploration=False, potential_shaping=False, dual_constraints=True, price_input=True),
}


class DualVariable:
    """Projected dual ascent on a rollout-average cost: lambda <- [lambda + eta (c - thr)]_0^max."""

    def __init__(self, target: float, learning_rate: float, maximum: float, initial: float = 0.0):
        if maximum <= 0.0 or learning_rate < 0.0:
            raise ValueError("invalid dual variable configuration")
        self.target, self.learning_rate, self.maximum = float(target), float(learning_rate), float(maximum)
        self.value = float(np.clip(initial, 0.0, self.maximum))

    def update(self, average_cost: float) -> float:
        self.value = float(np.clip(self.value + self.learning_rate * (average_cost - self.target), 0.0, self.maximum))
        return self.value

    @property
    def normalized(self) -> float:
        return self.value / self.maximum


class RelabelingReplayBuffer(ReplayBuffer):
    """Replay buffer that recomputes the Lagrangian reward at sampling time.

    reward = base + shaping - lambda_out (out - out_thr) - lambda_sig (sig - sig_thr),
    with (lambda_out, lambda_sig, out_thr, sig_thr) supplied by `weights()`.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, rng: np.random.Generator, weights):
        super().__init__(capacity, obs_dim, action_dim, rng)
        self.weights = weights
        self.base = np.zeros(self.capacity, dtype=np.float32)
        self.shaping = np.zeros(self.capacity, dtype=np.float32)
        self.out_cost = np.zeros(self.capacity, dtype=np.float32)
        self.sig_cost = np.zeros(self.capacity, dtype=np.float32)

    def add_components(self, obs, actions, base, shaping, out_cost, sig_cost, next_obs, next_masks, dones) -> None:
        n = len(actions)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.base[idx] = base
        self.shaping[idx] = shaping
        self.out_cost[idx] = out_cost
        self.sig_cost[idx] = sig_cost
        super().add_batch(obs, actions, np.zeros(n), next_obs, next_masks, dones)

    def relabeled(self, idx: np.ndarray) -> np.ndarray:
        lam_out, lam_sig, out_thr, sig_thr = self.weights()
        return (self.base[idx] + self.shaping[idx] - lam_out * (self.out_cost[idx] - out_thr)
                - lam_sig * (self.sig_cost[idx] - sig_thr)).astype(np.float64)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return (self.obs[idx].astype(np.float64), self.actions[idx], self.relabeled(idx),
                self.next_obs[idx].astype(np.float64), self.next_masks[idx].astype(np.float64),
                self.dones[idx].astype(np.float64))


class EGCMADQN(MADQN):
    """MADQN whose fingerprint optionally carries the normalised multipliers."""

    def __init__(self, obs_dim: int, action_dim: int, config: dict, seed: int, constraint_fingerprint: bool):
        self.constraint_fingerprint = bool(constraint_fingerprint)
        self.FINGERPRINT_DIM = 4 if self.constraint_fingerprint else 2
        super().__init__(obs_dim, action_dim, config, seed)
        self.multipliers = np.zeros(2)
        self.fingerprint = self._compose(1.0, self.eps_end)
        self.last_explore = np.zeros(0, dtype=bool)

    def use_relabeling(self, weights) -> None:
        """Switch to a relabelling replay buffer (before any transition is stored)."""
        self.buffer = RelabelingReplayBuffer(self.buffer.capacity, self.obs_dim + self.FINGERPRINT_DIM,
                                             self.action_dim, self.rng, weights)

    def store_components(self, obs, actions, base, shaping, out_cost, sig_cost, next_obs, next_masks, done) -> None:
        n = len(actions)
        self.buffer.add_components(self.augment(obs), actions, np.full(n, base), np.asarray(shaping),
                                   np.full(n, out_cost), np.full(n, sig_cost), self.augment(next_obs),
                                   next_masks, np.full(n, float(done)))

    def _compose(self, progress: float, epsilon: float) -> np.ndarray:
        base = [progress, epsilon]
        if self.constraint_fingerprint:
            base += list(self.multipliers)
        return np.asarray(base, dtype=np.float64)

    def set_multipliers(self, normalized_out: float, normalized_sig: float) -> None:
        self.multipliers = np.asarray([normalized_out, normalized_sig], dtype=np.float64)
        self.fingerprint = self._compose(float(self.fingerprint[0]), float(self.fingerprint[1]))

    def set_iteration(self, iteration: int, total: int) -> tuple[float, float]:
        eps, lr = super().set_iteration(iteration, total)  # epsilon + annealed learning rate
        self.fingerprint = self._compose(iteration / max(1, total), eps)
        return eps, lr

    def act_guided(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        epsilon: float,
        guidance: float,
        planner_costs: np.ndarray | None,
        temperature: float,
    ) -> np.ndarray:
        """epsilon-greedy whose exploratory draw follows the planner with prob. `guidance`."""
        actions = self.greedy(observations, masks)
        explore = self.rng.random(len(actions)) < epsilon
        self.last_explore = explore
        for i in np.flatnonzero(explore):
            feasible = np.flatnonzero(masks[i] > 0.0)
            if planner_costs is not None and self.rng.random() < guidance:
                logits = -planner_costs[i, feasible] / max(temperature, 1e-6)
                p = np.exp(logits - logits.max())
                actions[i] = self.rng.choice(feasible, p=p / p.sum())
            else:
                actions[i] = self.rng.choice(feasible)
        return actions

    def store_vector(self, obs, actions, rewards: np.ndarray, next_obs, next_masks, done) -> None:
        """Store per-agent transitions with per-agent (shaped) rewards."""
        n = len(actions)
        self.buffer.add_batch(self.augment(obs), actions, np.asarray(rewards, dtype=np.float64),
                              self.augment(next_obs), next_masks, np.full(n, float(done)))

    @classmethod
    def from_checkpoint(cls, path, seed: int, config: dict | None = None) -> "EGCMADQN":
        with np.load(path, allow_pickle=False) as data:
            obs_dim, action_dim = int(data["obs_dim"]), int(data["action_dim"])
            hidden = [int(h) for h in np.asarray(data["hidden_dims"]).ravel()]
            constrained = bool(int(data["constraint_fingerprint"])) if "constraint_fingerprint" in data else False
        cfg = dict(config or {})
        cfg.update(hidden_dims=hidden, buffer_capacity=1)
        agent = cls(obs_dim, action_dim, cfg, seed, constraint_fingerprint=constrained)
        agent.load(path)
        return agent

    def save(self, path) -> None:
        super().save(path)
        with np.load(path, allow_pickle=False) as data:
            payload = dict(data)
        payload["model_type"] = np.asarray("EGCMADQN")
        payload["constraint_fingerprint"] = np.asarray(int(self.constraint_fingerprint))
        np.savez(path, **payload)


@dataclass
class TrainerConfig:
    rollout_steps: int = 128
    guidance_temperature: float = 0.5        # T_g
    guidance_decay_fraction: float = 0.4     # beta: 1 -> 0 over this fraction of training
    shaping_scale: float = 0.05              # kappa
    outage_target: float = 0.20              # attainable level (validation: MADQN 0.21, tuned rules 0.31-0.41)
    signaling_target: float = 0.16           # ~ tuned A3 CHO signalling per UE and epoch
    relabel_rewards: bool = True             # recompute the Lagrangian reward with current multipliers
    fixed_lambda_out: float | None = None    # fixed-penalty baseline: constant multipliers, no dual ascent
    fixed_lambda_sig: float | None = None
    greedy_constraint_estimates: bool = True # update multipliers from non-exploring agents only
    dual_learning_rate_out: float = 0.005
    dual_learning_rate_sig: float = 0.005
    dual_max: float = 10.0
    extra: dict = field(default_factory=dict)


class EGCTrainer:
    """Rollout collection and updates for EG-CMADQN and its ablations."""

    def __init__(self, env: LEOMobilityEnv, components: Components, agent_config: dict, seed: int,
                 trainer_config: TrainerConfig | None = None):
        self.env = env
        self.c = components
        self.tc = trainer_config or TrainerConfig()
        cfg = env.config
        self.coordinator = (
            SatellitePriceCoordinator(env.num_cells, cfg.get("coordination", {})) if self.c.price_input else None
        )
        self.price_dim = env.candidate_count + 1 if self.c.price_input else 0
        self.agent = EGCMADQN(env.obs_dim + self.price_dim, env.action_dim, agent_config, seed,
                              constraint_fingerprint=self.c.dual_constraints)
        self.reasoner = (
            LookaheadReasoner(cfg.get("reasoning", {}), env.reward_cfg, cfg.get("coordination", {}),
                              use_coordination=self.c.price_input, use_constraint=False)
            if self.c.needs_planner else None
        )
        fixed = self.tc.fixed_lambda_out is not None or self.tc.fixed_lambda_sig is not None
        self.dual_out = DualVariable(self.tc.outage_target, 0.0 if fixed else self.tc.dual_learning_rate_out,
                                     self.tc.dual_max, initial=self.tc.fixed_lambda_out or 0.0)
        self.dual_sig = DualVariable(self.tc.signaling_target, 0.0 if fixed else self.tc.dual_learning_rate_sig,
                                     self.tc.dual_max, initial=self.tc.fixed_lambda_sig or 0.0)
        self.relabel = self.tc.relabel_rewards
        if self.relabel:
            # A bound method (not a lambda) so that the trainer can be pickled for resuming.
            self.agent.use_relabeling(self.lagrange_weights)
        self.obs, _, self.masks = env.reset()
        if self.coordinator is not None:
            self.coordinator.reset()
        self.costs = self._planner_costs()

    def lagrange_weights(self) -> tuple[float, float, float, float]:
        """(lambda_out, lambda_sig, out_thr, sig_thr) used when replayed rewards are relabelled."""
        constrained = self.c.dual_constraints
        return (self.dual_out.value if constrained else 0.0, self.dual_sig.value if constrained else 0.0,
                self.tc.outage_target, self.tc.signaling_target)

    # ------------------------------------------------------------------ inputs
    def _prices(self) -> np.ndarray:
        if self.coordinator is None:
            return np.zeros(self.env.num_cells)
        return self.coordinator.prices

    def inputs(self, observations: np.ndarray) -> np.ndarray:
        if self.coordinator is None:
            return observations
        price = self.coordinator.local_features([u.primary_cell for u in self.env.ues], self.env.candidates,
                                                self.env.candidate_count)
        return np.concatenate([observations, price], axis=1)

    def _planner_costs(self) -> np.ndarray | None:
        if self.reasoner is None:
            return None
        return self.reasoner.analyze(self.env, self.masks, self._prices(), 0.0).raw_costs

    def _potential(self, costs: np.ndarray | None, masks: np.ndarray) -> np.ndarray:
        if costs is None:
            return np.zeros(self.env.num_agents)
        feasible = np.where(masks > 0.0, costs, np.inf)
        return -np.min(feasible, axis=1)

    def guidance(self, iteration: int, total: int) -> float:
        if not self.c.guided_exploration:
            return 0.0
        return float(max(0.0, 1.0 - iteration / max(1.0, self.tc.guidance_decay_fraction * total)))

    # --------------------------------------------------------------- training
    def run_iteration(self, iteration: int, total: int) -> dict[str, float]:
        eps, _ = self.agent.set_iteration(iteration, total)
        if self.c.dual_constraints:
            self.agent.set_multipliers(self.dual_out.normalized, self.dual_sig.normalized)
        beta = self.guidance(iteration, total)
        gamma = self.agent.gamma
        sums = {"common_reward": 0.0, "training_reward": 0.0, "outage": 0.0, "signaling": 0.0,
                "handover_started": 0.0, "pingpong": 0.0, "handover_success": 0.0, "shaping": 0.0}
        greedy = {"outage": 0.0, "active": 0.0, "requests": 0.0}
        prep_lo, prep_hi = self.env.prepare_start, self.env.prepare_start + self.env.candidate_count
        for _ in range(self.tc.rollout_steps):
            z = self.inputs(self.obs)
            actions = self.agent.act_guided(z, self.masks, eps, beta, self.costs, self.tc.guidance_temperature)
            phi = self._potential(self.costs, self.masks)
            step = self.env.step(actions)
            if self.coordinator is not None:
                self.coordinator.update(step.requested_pressure)
            base = float(np.mean(step.rewards))
            outage = float(step.info["outage_ratio"])
            signaling = float(step.info["signaling_cost"])
            reward = base
            if self.c.dual_constraints:
                reward -= self.dual_out.value * (outage - self.tc.outage_target)
                reward -= self.dual_sig.value * (signaling - self.tc.signaling_target)
            next_obs, next_masks = step.observations, step.action_masks
            z_next = self.inputs(next_obs)
            self.masks = next_masks
            if step.done:
                next_costs, phi_next = None, np.zeros(self.env.num_agents)
            else:
                next_costs = self._planner_costs()
                phi_next = self._potential(next_costs, next_masks)
            shaping = (self.tc.shaping_scale * (gamma * phi_next - phi)
                       if self.c.potential_shaping else np.zeros(self.env.num_agents))
            if self.relabel:
                self.agent.store_components(z, actions, base, shaping, outage, signaling, z_next, next_masks, step.done)
            else:
                self.agent.store_vector(z, actions, reward + shaping, z_next, next_masks, step.done)
            # Target-policy constraint statistics (agents that did not explore).
            active = (step.agent_active > 0.0) if step.agent_active is not None else np.ones(len(actions), bool)
            chosen = active & ~self.agent.last_explore
            greedy["active"] += float(chosen.sum())
            greedy["requests"] += float(np.sum(chosen & (actions >= prep_lo) & (actions < prep_hi)))
            if step.agent_outcomes is not None:
                greedy["outage"] += float(step.agent_outcomes[chosen, 1].sum())
            sums["common_reward"] += base
            sums["training_reward"] += reward
            sums["outage"] += outage
            sums["signaling"] += signaling
            sums["shaping"] += float(np.mean(shaping))
            for key in ("handover_started", "pingpong", "handover_success"):
                sums[key] += float(step.info.get(key, 0.0))
            if step.done:
                self.obs, _, self.masks = self.env.reset()
                if self.coordinator is not None:
                    self.coordinator.reset()
                self.costs = self._planner_costs()
            else:
                self.obs, self.costs = next_obs, next_costs
        learn = self.agent.update()
        n = float(self.tc.rollout_steps)
        metrics = {k: v / n for k, v in sums.items() if k not in ("pingpong", "handover_success")}
        metrics["pingpong_ratio"] = sums["pingpong"] / max(sums["handover_success"], 1.0)
        if greedy["active"] > 0.0:
            metrics["greedy_outage"] = greedy["outage"] / greedy["active"]
            metrics["greedy_signaling"] = greedy["requests"] / greedy["active"]
        else:
            metrics["greedy_outage"], metrics["greedy_signaling"] = metrics["outage"], metrics["signaling"]
        if self.c.dual_constraints:
            use_greedy = self.tc.greedy_constraint_estimates
            metrics["lambda_out"] = self.dual_out.update(metrics["greedy_outage"] if use_greedy else metrics["outage"])
            metrics["lambda_sig"] = self.dual_sig.update(
                metrics["greedy_signaling"] if use_greedy else metrics["signaling"])
        metrics.update({"epsilon": eps, "guidance": beta, **learn})
        return metrics

    # -------------------------------------------------------------- execution
    def greedy_actions(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return self.agent.greedy(self.inputs(observations), masks)


class EGCExecutor:
    """Decentralised execution of a trained (EG-C)MADQN: Q-network + prices, no planner."""

    def __init__(self, env: LEOMobilityEnv, agent: EGCMADQN, price_input: bool, rng: np.random.Generator):
        self.env, self.agent, self.rng = env, agent, rng
        self.coordinator = (
            SatellitePriceCoordinator(env.num_cells, env.config.get("coordination", {})) if price_input else None
        )

    def reset(self):
        obs, state, masks = self.env.reset()
        if self.coordinator is not None:
            self.coordinator.reset()
        return obs, masks

    def act(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        z = observations
        if self.coordinator is not None:
            price = self.coordinator.local_features([u.primary_cell for u in self.env.ues], self.env.candidates,
                                                    self.env.candidate_count)
            z = np.concatenate([observations, price], axis=1)
        return self.agent.greedy(z, masks, self.rng)

    def observe(self, step) -> None:
        if self.coordinator is not None:
            self.coordinator.update(step.requested_pressure)
