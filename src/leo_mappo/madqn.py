"""Multi-agent Double DQN (MADQN) baseline.

Parameter-shared independent Double DQN with
  * the same local observation, action set, action mask and team reward as M1,
  * a training fingerprint [iteration / I, epsilon] appended to the observation
    (Foerster et al., ICML 2017; used for LEO handover in MAF-DDQN), which
    mitigates the non-stationarity of independent learners,
  * experience replay, a hard-updated target network, the Huber loss and
    masked (feasible-only) greedy action selection and bootstrapping.
At execution the fingerprint is fixed to its final training value.
"""
from __future__ import annotations

import numpy as np

from .networks import MLP, Adam


class ReplayBuffer:
    """Uniform replay of per-agent transitions (o, b, a, r, o', b', done)."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_masks = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add_batch(self, obs, actions, rewards, next_obs, next_masks, dones) -> None:
        n = len(actions)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.next_masks[idx] = next_masks
        self.dones[idx] = dones
        self.pos = int((self.pos + n) % self.capacity)
        self.size = int(min(self.capacity, self.size + n))

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return (self.obs[idx].astype(np.float64), self.actions[idx], self.rewards[idx].astype(np.float64),
                self.next_obs[idx].astype(np.float64), self.next_masks[idx].astype(np.float64),
                self.dones[idx].astype(np.float64))


class MADQN:
    FINGERPRINT_DIM = 2

    def __init__(self, obs_dim: int, action_dim: int, config: dict, seed: int):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = dict(config)
        self.rng = np.random.default_rng(int(seed))
        self.hidden = list(config.get("hidden_dims", [128, 128]))
        self.gamma = float(config.get("gamma", 0.99))
        self.initial_lr = float(config.get("learning_rate", 3e-4))
        self.final_lr = float(config.get("final_learning_rate", 0.1 * self.initial_lr))
        self.batch_size = int(config.get("batch_size", 256))
        self.updates_per_iteration = int(config.get("updates_per_iteration", 32))
        self.target_update_interval = int(config.get("target_update_interval", 250))
        self.warmup = int(config.get("warmup_transitions", 10_000))
        self.eps_start = float(config.get("epsilon_start", 1.0))
        self.eps_end = float(config.get("epsilon_end", 0.05))
        self.eps_fraction = float(config.get("epsilon_decay_fraction", 0.4))
        self.max_grad_norm = float(config.get("max_grad_norm", 10.0))
        self.huber_delta = float(config.get("huber_delta", 1.0))
        self.hysteretic_beta = float(config.get("hysteretic_beta", 1.0))
        in_dim = self.obs_dim + self.FINGERPRINT_DIM
        self.q = MLP(in_dim, self.hidden, self.action_dim, self.rng)
        self.target = MLP(in_dim, self.hidden, self.action_dim, self.rng)
        self._sync_target()
        self.optimizer = Adam(self.q.parameters, self.initial_lr)
        self.buffer = ReplayBuffer(int(config.get("buffer_capacity", 200_000)), in_dim,
                                   self.action_dim, self.rng)
        self.updates = 0
        self.fingerprint = np.array([1.0, self.eps_end])  # execution value

    # ------------------------------------------------------------ schedule --
    def epsilon(self, iteration: int, total: int) -> float:
        frac = min(1.0, iteration / max(1.0, self.eps_fraction * total))
        return float(self.eps_start + frac * (self.eps_end - self.eps_start))

    def set_iteration(self, iteration: int, total: int) -> tuple[float, float]:
        eps = self.epsilon(iteration, total)
        lr = self.initial_lr + (iteration / max(1, total - 1)) * (self.final_lr - self.initial_lr)
        self.optimizer.learning_rate = float(lr)
        self.fingerprint = np.array([iteration / max(1, total), eps])
        return eps, float(lr)

    # --------------------------------------------------------------- acting --
    def augment(self, observations: np.ndarray) -> np.ndarray:
        fp = np.broadcast_to(self.fingerprint, (len(observations), self.FINGERPRINT_DIM))
        return np.concatenate([observations, fp], axis=1)

    @staticmethod
    def _masked(q: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.where(masks > 0.0, q, -1e9)

    def greedy(self, observations: np.ndarray, masks: np.ndarray,
               rng: np.random.Generator | None = None) -> np.ndarray:
        q, _ = self.q.forward(self.augment(observations))
        qm = self._masked(q, masks)
        rng = self.rng if rng is None else rng
        # random tie-breaking among maximal feasible actions
        best = qm >= (qm.max(axis=1, keepdims=True) - 1e-12)
        return np.array([rng.choice(np.flatnonzero(row)) for row in best], dtype=np.int64)

    def act(self, observations: np.ndarray, masks: np.ndarray, epsilon: float) -> np.ndarray:
        actions = self.greedy(observations, masks)
        explore = self.rng.random(len(actions)) < epsilon
        for i in np.flatnonzero(explore):
            actions[i] = self.rng.choice(np.flatnonzero(masks[i] > 0.0))
        return actions

    # ------------------------------------------------------------- learning --
    def store(self, obs, actions, reward, next_obs, next_masks, done) -> None:
        n = len(actions)
        self.buffer.add_batch(self.augment(obs), actions, np.full(n, reward),
                              self.augment(next_obs), next_masks, np.full(n, float(done)))

    def update(self) -> dict[str, float]:
        if self.buffer.size < max(self.warmup, self.batch_size):
            return {"td_loss": 0.0, "q_mean": 0.0, "updates": float(self.updates)}
        losses, qmeans = [], []
        for _ in range(self.updates_per_iteration):
            o, a, r, o2, m2, d = self.buffer.sample(self.batch_size)
            q_next_online, _ = self.q.forward(o2)
            a_star = self._masked(q_next_online, m2).argmax(axis=1)          # Double DQN
            q_next_target, _ = self.target.forward(o2)
            y = r + self.gamma * (1.0 - d) * q_next_target[np.arange(len(a)), a_star]
            q, cache = self.q.forward(o)
            q_sa = q[np.arange(len(a)), a]
            err = q_sa - y
            grad_err = np.clip(err, -self.huber_delta, self.huber_delta) / len(a)
            if self.hysteretic_beta != 1.0:
                # Hysteretic Q-learning: smaller step when the TD error is negative
                # (y < Q), i.e. when a value would be decreased.
                grad_err = np.where(err > 0.0, self.hysteretic_beta * grad_err, grad_err)
            grad_out = np.zeros_like(q)
            grad_out[np.arange(len(a)), a] = grad_err
            self.optimizer.step(self.q.backward(grad_out, cache), self.max_grad_norm)
            self.updates += 1
            if self.updates % self.target_update_interval == 0:
                self._sync_target()
            absd = np.abs(err)
            huber = np.where(absd <= self.huber_delta, 0.5 * err**2,
                             self.huber_delta * (absd - 0.5 * self.huber_delta))
            losses.append(float(huber.mean()))
            qmeans.append(float(q_sa.mean()))
        return {"td_loss": float(np.mean(losses)), "q_mean": float(np.mean(qmeans)),
                "updates": float(self.updates)}

    def _sync_target(self) -> None:
        for name, value in self.q.parameters.items():
            self.target.parameters[name][...] = value

    @classmethod
    def from_checkpoint(cls, path, seed: int, config: dict | None = None) -> "MADQN":
        with np.load(path, allow_pickle=False) as data:
            obs_dim, action_dim = int(data["obs_dim"]), int(data["action_dim"])
            hidden = [int(h) for h in np.asarray(data["hidden_dims"]).ravel()]
        cfg = dict(config or {})
        cfg["hidden_dims"] = hidden
        cfg["buffer_capacity"] = 1  # no replay needed for execution
        agent = cls(obs_dim, action_dim, cfg, seed)
        agent.load(path)
        return agent

    # -------------------------------------------------------------- storage --
    def save(self, path) -> None:
        data = {f"q_{k}": v for k, v in self.q.parameters.items()}
        data.update({"obs_dim": np.asarray(self.obs_dim), "action_dim": np.asarray(self.action_dim),
                     "hidden_dims": np.asarray(self.hidden), "fingerprint": self.fingerprint,
                     "model_type": np.asarray("MADQN")})
        np.savez(path, **data)

    def load(self, path) -> None:
        with np.load(path, allow_pickle=False) as data:
            if int(data["obs_dim"]) != self.obs_dim or int(data["action_dim"]) != self.action_dim:
                raise ValueError("MADQN checkpoint does not match the environment dimensions")
            for k in self.q.parameters:
                self.q.parameters[k][...] = data[f"q_{k}"]
            self.fingerprint = np.asarray(data["fingerprint"], dtype=np.float64)
        self._sync_target()
