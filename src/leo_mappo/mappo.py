from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .networks import Adam, GRU, MLP, masked_softmax


def _group_recurrent_chunks(
    chunks: list[tuple[int, int, int]],
    shuffled_indices: np.ndarray,
    target_transitions: int,
) -> list[list[int]]:
    """Pack whole TBPTT chunks until each group reaches the transition target."""
    if target_transitions <= 0:
        raise ValueError("recurrent actor minibatch size must be positive")
    groups: list[list[int]] = []
    current_group: list[int] = []
    current_transitions = 0
    for raw_index in shuffled_indices:
        chunk_index = int(raw_index)
        _, start, end = chunks[chunk_index]
        if end <= start:
            raise ValueError("recurrent TBPTT chunk must contain at least one transition")
        current_group.append(chunk_index)
        current_transitions += end - start
        if current_transitions >= target_transitions:
            groups.append(current_group)
            current_group = []
            current_transitions = 0
    if current_group:
        groups.append(current_group)
    return groups


class MAPPO:
    """Small dependency-free CTDE-MAPPO reference implementation.

    The actor is shared by all UE agents and consumes local observations. The
    centralized critic consumes the flattened global network state.
    """

    def __init__(
        self,
        obs_dim: int,
        global_state_dim: int,
        action_dim: int,
        config: dict[str, Any],
        seed: int = 0,
        num_satellites: int = 0,
        candidate_count: int = 0,
        eta_handover: float = 0.02,
    ):
        self.obs_dim = obs_dim
        self.base_global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.use_memory = bool(config.get("use_memory", False))
        self.use_reasoning = bool(config.get("use_reasoning", False))
        self.use_coordination = bool(config.get("use_coordination", False))
        self.use_constraint = bool(config.get("use_constraint", False))
        if self.use_reasoning and not self.use_memory:
            raise ValueError("use_reasoning=true requires use_memory=true in Step 2")
        if self.use_coordination and not (self.use_memory and self.use_reasoning):
            raise ValueError("use_coordination=true requires memory and reasoning")
        # The M1D ablation (plain MAPPO + reliability Lagrangian) is allowed;
        # recurrent models still need coordination before the constraint.
        if self.use_constraint and self.use_memory and not self.use_coordination:
            raise ValueError("use_constraint=true requires coordination")
        self.num_satellites = int(num_satellites)
        self.candidate_count = int(candidate_count)
        self.eta_handover = float(eta_handover)
        self.price_feature_dim = self.candidate_count + 1 if self.use_coordination else 0
        if self.use_coordination and (self.num_satellites <= 0 or self.candidate_count <= 0):
            raise ValueError("coordination mode requires num_satellites and candidate_count")
        self.global_state_dim = (
            self.base_global_state_dim
            + (self.num_satellites if self.use_coordination else 0)
            + (1 if self.use_constraint else 0)
        )
        self.outage_dual_mu = 0.0
        self.dual_state_loaded = False
        self.outcome_dim = 6
        self.recurrent_hidden_dim = int(config.get("recurrent_hidden_dim", 128))
        if self.recurrent_hidden_dim <= 0 or int(config.get("bptt_steps", 16)) <= 0:
            raise ValueError("recurrent_hidden_dim and bptt_steps must be positive")
        hidden = int(config.get("hidden_dim", 64))
        actor_hidden = config.get("actor_hidden_dims", [hidden])
        critic_hidden = config.get("critic_hidden_dims", [hidden])
        actor_lr = float(config["actor_lr"])
        critic_lr = float(config["critic_lr"])
        lr_anneal = config.get("lr_anneal", {})
        if not isinstance(lr_anneal, dict):
            raise ValueError("mappo.lr_anneal must be a mapping")
        self.lr_anneal_enabled = bool(lr_anneal.get("enabled", False))
        self.lr_anneal_schedule = str(lr_anneal.get("schedule", "linear"))
        self.actor_initial_lr = (
            float(lr_anneal.get("actor_initial_lr", actor_lr))
            if self.lr_anneal_enabled
            else actor_lr
        )
        self.critic_initial_lr = (
            float(lr_anneal.get("critic_initial_lr", critic_lr))
            if self.lr_anneal_enabled
            else critic_lr
        )
        self.actor_final_lr = float(
            lr_anneal.get("actor_final_lr", 0.1 * self.actor_initial_lr)
        )
        self.critic_final_lr = float(
            lr_anneal.get("critic_final_lr", 0.1 * self.critic_initial_lr)
        )
        if self.lr_anneal_enabled and self.lr_anneal_schedule != "linear":
            raise ValueError("only linear mappo learning-rate annealing is supported")
        learning_rates = (
            self.actor_initial_lr,
            self.critic_initial_lr,
            self.actor_final_lr,
            self.critic_final_lr,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in learning_rates):
            raise ValueError("MAPPO learning rates must be finite and non-negative")
        actor_input_dim = obs_dim
        if self.use_memory:
            self.gru = GRU(obs_dim + action_dim + self.outcome_dim, self.recurrent_hidden_dim, self.rng)
            actor_input_dim += self.recurrent_hidden_dim
            if self.use_reasoning:
                actor_input_dim += action_dim
            actor_input_dim += self.price_feature_dim
            if self.use_constraint:
                actor_input_dim += 1
        else:
            self.gru = None
        self.actor = MLP(actor_input_dim, actor_hidden, action_dim, self.rng)
        self.critic = MLP(self.global_state_dim, critic_hidden, 1, self.rng)
        self.actor_optimizer = Adam(self.actor.parameters, self.actor_initial_lr)
        self.gru_optimizer = (
            Adam(self.gru.parameters, self.actor_initial_lr) if self.gru is not None else None
        )
        self.critic_optimizer = Adam(self.critic.parameters, self.critic_initial_lr)

    def set_training_iteration(
        self, iteration: int, total_iterations: int
    ) -> tuple[float, float]:
        """Set optimizer rates for a zero-based training iteration."""
        if total_iterations <= 0 or not 0 <= iteration < total_iterations:
            raise ValueError("training iteration must be within total_iterations")
        if self.lr_anneal_enabled:
            progress = iteration / max(total_iterations - 1, 1)
            actor_lr = self.actor_initial_lr + progress * (
                self.actor_final_lr - self.actor_initial_lr
            )
            critic_lr = self.critic_initial_lr + progress * (
                self.critic_final_lr - self.critic_initial_lr
            )
            self.actor_optimizer.learning_rate = float(actor_lr)
            self.critic_optimizer.learning_rate = float(critic_lr)
            if self.gru_optimizer is not None:
                self.gru_optimizer.learning_rate = float(actor_lr)
        return self.learning_rates

    @property
    def learning_rates(self) -> tuple[float, float]:
        return (
            float(self.actor_optimizer.learning_rate),
            float(self.critic_optimizer.learning_rate),
        )

    def policy(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        logits, _ = self.actor.forward(observations)
        return masked_softmax(logits, masks)

    def act(self, observations: np.ndarray, masks: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if self.use_memory:
            raise RuntimeError("memory mode requires act_recurrent()")
        probabilities = self.policy(observations, masks)
        if deterministic:
            actions = probabilities.argmax(axis=-1)
        else:
            actions = np.array(
                [self.rng.choice(self.action_dim, p=row) for row in probabilities], dtype=np.int64
            )
        log_probabilities = np.log(np.maximum(probabilities[np.arange(len(actions)), actions], 1e-12))
        return actions, log_probabilities

    def act_recurrent(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        previous_actions: np.ndarray,
        previous_outcomes: np.ndarray,
        hidden_states: np.ndarray,
        reasoning_features: np.ndarray | None = None,
        price_features: np.ndarray | None = None,
        normalized_mu: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.gru is None:
            raise RuntimeError("act_recurrent() requires use_memory=true")
        action_one_hot = np.zeros((len(observations), self.action_dim), dtype=np.float64)
        valid_previous = (previous_actions >= 0) & (previous_actions < self.action_dim)
        rows = np.flatnonzero(valid_previous)
        action_one_hot[rows, previous_actions[rows]] = 1.0
        gru_inputs = np.concatenate([observations, action_one_hot, previous_outcomes], axis=1)
        new_hidden, _ = self.gru.forward_step(gru_inputs, hidden_states)
        actor_parts = [observations, new_hidden]
        if self.use_reasoning:
            if reasoning_features is None:
                raise ValueError("reasoning_features are required when use_reasoning=true")
            actor_parts.append(reasoning_features)
        if self.use_coordination:
            if price_features is None or price_features.shape != (
                len(observations), self.price_feature_dim
            ):
                raise ValueError("coordination mode requires local price features")
            actor_parts.append(price_features)
        if self.use_constraint:
            actor_parts.append(np.full((len(observations), 1), normalized_mu, dtype=np.float64))
        logits, _ = self.actor.forward(np.concatenate(actor_parts, axis=1))
        probabilities = masked_softmax(logits, masks)
        if deterministic:
            actions = probabilities.argmax(axis=-1)
        else:
            actions = np.asarray(
                [self.rng.choice(self.action_dim, p=row) for row in probabilities], dtype=np.int64
            )
        log_probabilities = np.log(
            np.maximum(probabilities[np.arange(len(actions)), actions], 1e-12)
        )
        return actions, log_probabilities, new_hidden

    def value(self, global_states: np.ndarray) -> np.ndarray:
        values, _ = self.critic.forward(np.atleast_2d(global_states))
        return values[:, 0]

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        if self.use_memory:
            return self._update_recurrent(batch)
        observations = batch["observations"].reshape(-1, self.obs_dim)
        masks = batch["masks"].reshape(-1, self.action_dim)
        actions = batch["actions"].reshape(-1)
        old_log_probs = batch["log_probs"].reshape(-1)
        time_advantages = batch["advantages"]
        num_agents = batch["observations"].shape[1]
        advantages = np.repeat(time_advantages, num_agents)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states = batch["global_states"]
        returns = batch["returns"]
        clip_ratio = float(self.config["clip_ratio"])
        entropy_coef = float(self.config["entropy_coef"])
        max_grad_norm = float(self.config["max_grad_norm"])
        minibatch = int(self.config["minibatch_size"])
        epochs = int(self.config["update_epochs"])
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []

        for _ in range(epochs):
            actor_indices = self.rng.permutation(len(observations))
            stop_early = False
            for start in range(0, len(actor_indices), minibatch):
                index = actor_indices[start : start + minibatch]
                logits, cache = self.actor.forward(observations[index])
                probabilities = masked_softmax(logits, masks[index])
                chosen = probabilities[np.arange(len(index)), actions[index]]
                new_log_probs = np.log(np.maximum(chosen, 1e-12))
                ratio = np.exp(new_log_probs - old_log_probs[index])
                advantage = advantages[index]
                active = np.where(advantage >= 0.0, ratio <= 1.0 + clip_ratio, ratio >= 1.0 - clip_ratio)
                coefficient = active.astype(np.float64) * advantage * ratio
                one_hot = np.eye(self.action_dim)[actions[index]]
                grad_logits = -coefficient[:, None] * (one_hot - probabilities) / len(index)
                log_all = np.log(np.maximum(probabilities, 1e-12))
                entropy = -np.sum(probabilities * log_all, axis=1)
                grad_logits += entropy_coef * probabilities * (log_all + entropy[:, None]) / len(index)
                gradients = self.actor.backward(grad_logits, cache)
                self.actor_optimizer.step(gradients, max_grad_norm)
                clipped = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                surrogate = np.minimum(ratio * advantage, clipped * advantage)
                actor_losses.append(float(-surrogate.mean()))
                entropies.append(float(entropy.mean()))
                approximate_kl = float((old_log_probs[index] - new_log_probs).mean())
                kls.append(approximate_kl)
                if approximate_kl > float(self.config["target_kl"]):
                    stop_early = True
                    break
            critic_indices = self.rng.permutation(len(states))
            critic_minibatch = max(8, minibatch // max(1, num_agents))
            for start in range(0, len(critic_indices), critic_minibatch):
                index = critic_indices[start : start + critic_minibatch]
                predictions, cache = self.critic.forward(states[index])
                errors = predictions[:, 0] - returns[index]
                grad_values = np.zeros_like(predictions)
                grad_values[:, 0] = float(self.config.get("value_coef", 0.5)) * errors / len(index)
                gradients = self.critic.backward(grad_values, cache)
                self.critic_optimizer.step(gradients, max_grad_norm)
                critic_losses.append(float(0.5 * np.mean(errors**2)))
            if stop_early:
                break
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
        }

    def _update_recurrent(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        if self.gru is None or self.gru_optimizer is None:
            raise RuntimeError("recurrent update requested without GRU parameters")
        observations = batch["observations"]
        masks = batch["masks"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        previous_actions = batch["previous_actions"]
        previous_outcomes = batch["previous_outcomes"]
        reasoning = batch["reasoning_features"]
        price_features = batch.get("price_features")
        mu_features = batch.get("mu_features")
        hidden_before = batch["hidden_states_before"]
        dones = batch["dones"]
        time_steps, num_agents = actions.shape
        advantages_time = batch["advantages"]
        advantages_time = (advantages_time - advantages_time.mean()) / (advantages_time.std() + 1e-8)
        clip_ratio = float(self.config["clip_ratio"])
        entropy_coef = float(self.config["entropy_coef"])
        max_grad_norm = float(self.config["max_grad_norm"])
        bptt_steps = int(self.config.get("bptt_steps", 16))
        minibatch = int(self.config["minibatch_size"])
        epochs = int(self.config["update_epochs"])
        chunks: list[tuple[int, int, int]] = []
        for agent_id in range(num_agents):
            start = 0
            while start < time_steps:
                end = min(time_steps, start + bptt_steps)
                boundaries = np.flatnonzero(dones[start:end] > 0.0)
                if len(boundaries):
                    end = start + int(boundaries[0]) + 1
                chunks.append((agent_id, start, end))
                start = end
        actor_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        recurrent_minibatch_transitions: list[int] = []
        actor_optimizer_steps = 0
        gru_optimizer_steps = 0
        stop_early = False

        def accumulate_gradients(
            accumulator: dict[str, np.ndarray],
            gradients: dict[str, np.ndarray],
            parameters: dict[str, np.ndarray],
            label: str,
        ) -> None:
            if gradients.keys() != parameters.keys():
                raise RuntimeError(f"{label} gradient keys do not match parameter keys")
            for key, parameter in parameters.items():
                gradient = gradients[key]
                if gradient.shape != parameter.shape:
                    raise RuntimeError(
                        f"{label} gradient shape mismatch for {key}: "
                        f"{gradient.shape} != {parameter.shape}"
                    )
                if not np.all(np.isfinite(gradient)):
                    raise FloatingPointError(f"{label} produced a non-finite gradient")
                accumulator[key] += gradient

        for _ in range(epochs):
            shuffled_chunks = self.rng.permutation(len(chunks))
            actor_minibatches = _group_recurrent_chunks(
                chunks, shuffled_chunks, minibatch
            )
            for chunk_group in actor_minibatches:
                batch_transitions = sum(
                    chunks[chunk_index][2] - chunks[chunk_index][1]
                    for chunk_index in chunk_group
                )
                if batch_transitions <= 0:
                    raise RuntimeError("recurrent actor minibatch has no transitions")
                actor_accumulator = {
                    key: np.zeros_like(value) for key, value in self.actor.parameters.items()
                }
                gru_accumulator = {
                    key: np.zeros_like(value) for key, value in self.gru.parameters.items()
                }
                surrogate_sum = 0.0
                entropy_sum = 0.0
                kl_sum = 0.0
                for chunk_index in chunk_group:
                    agent_id, start, end = chunks[chunk_index]
                    chunk_length = end - start
                    chunk_obs = observations[start:end, agent_id]
                    chunk_previous = previous_actions[start:end, agent_id]
                    one_hot = np.zeros((chunk_length, self.action_dim), dtype=np.float64)
                    valid = (chunk_previous >= 0) & (chunk_previous < self.action_dim)
                    rows = np.flatnonzero(valid)
                    one_hot[rows, chunk_previous[rows]] = 1.0
                    gru_inputs = np.concatenate(
                        [chunk_obs, one_hot, previous_outcomes[start:end, agent_id]], axis=1
                    )[:, None, :]
                    hidden_sequence, _, gru_cache = self.gru.forward_sequence(
                        gru_inputs, hidden_before[start, agent_id][None, :]
                    )
                    actor_parts = [chunk_obs, hidden_sequence[:, 0, :]]
                    if self.use_reasoning:
                        actor_parts.append(reasoning[start:end, agent_id])
                    if self.use_coordination:
                        if price_features is None:
                            raise RuntimeError("recurrent batch is missing price_features")
                        actor_parts.append(price_features[start:end, agent_id])
                    if self.use_constraint:
                        if mu_features is None:
                            raise RuntimeError("recurrent batch is missing mu_features")
                        actor_parts.append(mu_features[start:end, agent_id])
                    actor_inputs = np.concatenate(actor_parts, axis=1)
                    logits, actor_cache = self.actor.forward(actor_inputs)
                    probabilities = masked_softmax(logits, masks[start:end, agent_id])
                    chunk_actions = actions[start:end, agent_id]
                    chosen = probabilities[np.arange(chunk_length), chunk_actions]
                    new_log_probs = np.log(np.maximum(chosen, 1e-12))
                    old_chunk_log_probs = old_log_probs[start:end, agent_id]
                    ratio = np.exp(new_log_probs - old_chunk_log_probs)
                    advantage = advantages_time[start:end]
                    active_clip = np.where(
                        advantage >= 0.0,
                        ratio <= 1.0 + clip_ratio,
                        ratio >= 1.0 - clip_ratio,
                    )
                    coefficient = active_clip.astype(np.float64) * advantage * ratio
                    selected_one_hot = np.eye(self.action_dim)[chunk_actions]
                    grad_logits = (
                        -coefficient[:, None] * (selected_one_hot - probabilities)
                        / batch_transitions
                    )
                    log_all = np.log(np.maximum(probabilities, 1e-12))
                    entropy = -np.sum(probabilities * log_all, axis=1)
                    grad_logits += (
                        entropy_coef
                        * probabilities
                        * (log_all + entropy[:, None])
                        / batch_transitions
                    )
                    actor_gradients, grad_actor_inputs = self.actor.backward_with_input(
                        grad_logits, actor_cache
                    )
                    hidden_start = self.obs_dim
                    hidden_end = hidden_start + self.recurrent_hidden_dim
                    grad_hidden = grad_actor_inputs[:, hidden_start:hidden_end, None].transpose(
                        0, 2, 1
                    )
                    gru_gradients, _, _ = self.gru.backward_sequence(
                        grad_hidden, gru_cache
                    )
                    accumulate_gradients(
                        actor_accumulator,
                        actor_gradients,
                        self.actor.parameters,
                        "actor",
                    )
                    accumulate_gradients(
                        gru_accumulator,
                        gru_gradients,
                        self.gru.parameters,
                        "GRU",
                    )
                    clipped = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                    surrogate_sum += float(
                        np.sum(np.minimum(ratio * advantage, clipped * advantage))
                    )
                    entropy_sum += float(np.sum(entropy))
                    kl_sum += float(np.sum(old_chunk_log_probs - new_log_probs))
                if not all(np.all(np.isfinite(value)) for value in actor_accumulator.values()):
                    raise FloatingPointError("actor accumulated a non-finite gradient")
                if not all(np.all(np.isfinite(value)) for value in gru_accumulator.values()):
                    raise FloatingPointError("GRU accumulated a non-finite gradient")
                self.actor_optimizer.step(actor_accumulator, max_grad_norm)
                actor_optimizer_steps += 1
                self.gru_optimizer.step(gru_accumulator, max_grad_norm)
                gru_optimizer_steps += 1
                recurrent_minibatch_transitions.append(batch_transitions)
                actor_losses.append(float(-surrogate_sum / batch_transitions))
                entropies.append(float(entropy_sum / batch_transitions))
                approximate_kl = float(kl_sum / batch_transitions)
                kls.append(approximate_kl)
                if approximate_kl > float(self.config["target_kl"]):
                    stop_early = True
                    break
            if stop_early:
                break

        if actor_optimizer_steps != gru_optimizer_steps:
            raise RuntimeError("actor and GRU optimizer step counts diverged")

        states = batch["global_states"]
        returns = batch["returns"]
        critic_losses: list[float] = []
        for _ in range(epochs):
            critic_indices = self.rng.permutation(len(states))
            critic_minibatch = max(8, minibatch // max(1, num_agents))
            for start in range(0, len(critic_indices), critic_minibatch):
                index = critic_indices[start : start + critic_minibatch]
                predictions, cache = self.critic.forward(states[index])
                errors = predictions[:, 0] - returns[index]
                grad_values = np.zeros_like(predictions)
                grad_values[:, 0] = float(self.config.get("value_coef", 0.5)) * errors / len(index)
                gradients = self.critic.backward(grad_values, cache)
                self.critic_optimizer.step(gradients, max_grad_norm)
                critic_losses.append(float(0.5 * np.mean(errors**2)))
        diagnostic_weights = np.asarray(recurrent_minibatch_transitions, dtype=np.float64)
        return {
            "actor_loss": (
                float(np.average(actor_losses, weights=diagnostic_weights))
                if actor_losses else 0.0
            ),
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": (
                float(np.average(entropies, weights=diagnostic_weights))
                if entropies else 0.0
            ),
            "approx_kl": (
                float(np.average(kls, weights=diagnostic_weights)) if kls else 0.0
            ),
            "recurrent_chunk_count": float(len(chunks)),
            "recurrent_actor_minibatch_count": float(len(recurrent_minibatch_transitions)),
            "recurrent_actor_optimizer_steps": float(actor_optimizer_steps),
            "recurrent_mean_minibatch_transitions": (
                float(np.mean(diagnostic_weights)) if len(diagnostic_weights) else 0.0
            ),
            "recurrent_min_minibatch_transitions": (
                float(np.min(diagnostic_weights)) if len(diagnostic_weights) else 0.0
            ),
            "recurrent_max_minibatch_transitions": (
                float(np.max(diagnostic_weights)) if len(diagnostic_weights) else 0.0
            ),
        }

    def save(self, path: str | Path) -> None:
        arrays = {
            "model_use_memory": np.asarray(int(self.use_memory), dtype=np.int64),
            "model_use_reasoning": np.asarray(int(self.use_reasoning), dtype=np.int64),
            "model_use_coordination": np.asarray(int(self.use_coordination), dtype=np.int64),
            "model_use_constraint": np.asarray(int(self.use_constraint), dtype=np.int64),
            "model_num_satellites": np.asarray(self.num_satellites, dtype=np.int64),
            "model_candidate_count": np.asarray(self.candidate_count, dtype=np.int64),
            "model_recurrent_hidden_dim": np.asarray(self.recurrent_hidden_dim, dtype=np.int64),
            "outage_dual_mu": np.asarray(self.outage_dual_mu, dtype=np.float64),
            "reward_eta_handover": np.asarray(self.eta_handover, dtype=np.float64),
        }
        arrays.update({f"actor_{key}": value for key, value in self.actor.parameters.items()})
        arrays.update({f"critic_{key}": value for key, value in self.critic.parameters.items()})
        if self.gru is not None:
            arrays.update({f"gru_{key}": value for key, value in self.gru.parameters.items()})
        np.savez(Path(path), **arrays)

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path))
        saved_memory = bool(int(data["model_use_memory"])) if "model_use_memory" in data.files else False
        saved_reasoning = bool(int(data["model_use_reasoning"])) if "model_use_reasoning" in data.files else False
        saved_coordination = bool(int(data["model_use_coordination"])) if "model_use_coordination" in data.files else False
        saved_constraint = bool(int(data["model_use_constraint"])) if "model_use_constraint" in data.files else False
        if (
            saved_memory != self.use_memory
            or saved_reasoning != self.use_reasoning
            or saved_coordination != self.use_coordination
            or saved_constraint != self.use_constraint
        ):
            raise ValueError(
                "model architecture mismatch: weights and configuration disagree on enabled capabilities"
            )
        if "model_num_satellites" in data.files and self.use_coordination:
            if int(data["model_num_satellites"]) != self.num_satellites or int(
                data["model_candidate_count"]
            ) != self.candidate_count:
                raise ValueError("model architecture mismatch: satellite/candidate dimensions differ")
        if "model_recurrent_hidden_dim" in data.files and int(
            data["model_recurrent_hidden_dim"]
        ) != self.recurrent_hidden_dim:
            raise ValueError("model architecture mismatch: recurrent hidden dimension differs")
        if "outage_dual_mu" in data.files:
            self.outage_dual_mu = float(data["outage_dual_mu"])
        self.dual_state_loaded = True
        for key in self.actor.parameters:
            saved = data[f"actor_{key}"]
            if saved.shape != self.actor.parameters[key].shape:
                raise ValueError(f"actor architecture mismatch for parameter {key}")
            self.actor.parameters[key][...] = saved
        for key in self.critic.parameters:
            saved = data[f"critic_{key}"]
            if saved.shape != self.critic.parameters[key].shape:
                raise ValueError(f"critic architecture mismatch for parameter {key}")
            self.critic.parameters[key][...] = saved
        if self.gru is not None:
            for key in self.gru.parameters:
                saved = data[f"gru_{key}"]
                if saved.shape != self.gru.parameters[key].shape:
                    raise ValueError(f"GRU architecture mismatch for parameter {key}")
                self.gru.parameters[key][...] = saved
