from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .coordinator import (
    ReliabilityDualController,
    SatellitePriceCoordinator,
    compute_training_reward,
)
from .environment import LEOMobilityEnv
from .mappo import MAPPO
from .reasoning import LookaheadReasoner, ReasoningResult


def summarize_metrics(totals: dict[str, float], steps: int, epsilon: float = 1e-8) -> dict[str, float]:
    """Average ordinary step metrics and compute evaluation KPIs as ratios of sums."""
    result = {key: value / max(1, steps) for key, value in totals.items()}
    result["outage_ratio"] = totals.get("outage_ue_count", 0.0) / (
        totals.get("active_ue_count", 0.0) + epsilon
    )
    result["strict_shortfall_ratio"] = totals.get("strict_shortfall_ue_count", 0.0) / (
        totals.get("active_ue_count", 0.0) + epsilon
    )
    result["service_ratio_mean"] = totals.get("service_ratio_sum", 0.0) / (
        totals.get("active_ue_count", 0.0) + epsilon
    )
    result["handover_failure_rate"] = totals.get("handover_failure", 0.0) / (
        totals.get("handover_started", 0.0) + epsilon
    )
    result["pingpong_rate"] = totals.get("pingpong", 0.0) / (
        totals.get("handover_success", 0.0) + epsilon
    )
    result["deadline_violation_rate"] = totals.get("deadline_misses", 0.0) / (
        totals.get("arrivals", 0.0) + epsilon
    )
    result["preparation_rejection_rate"] = totals.get("preparation_rejected", 0.0) / (
        totals.get("preparation_requested", 0.0) + epsilon
    )
    result["prepare_rate"] = result.get("action_prepare_fraction", 0.0)
    result["release_rate"] = result.get("action_release_fraction", 0.0)
    result["commit_rate"] = result.get("action_commit_fraction", 0.0)
    result["prepare_to_release_without_commit_rate"] = totals.get(
        "prepare_to_release_without_commit", 0.0
    ) / (totals.get("preparation_admitted", 0.0) + epsilon)
    result["prepare_to_commit_rate"] = totals.get("prepare_to_commit", 0.0) / (
        totals.get("preparation_admitted", 0.0) + epsilon
    )
    result["mean_prepared_duration"] = totals.get("prepared_duration_sum", 0.0) / (
        totals.get("prepared_duration_count", 0.0) + epsilon
    )
    return result


class RolloutRunner:
    def __init__(self, env: LEOMobilityEnv, agent: MAPPO):
        self.env = env
        self.agent = agent
        self.observations, self.global_state, self.masks = env.reset()
        self.reasoner = (
            LookaheadReasoner(
                env.config.get("reasoning", {}),
                env.reward_cfg,
                env.config.get("coordination", {}),
                agent.use_coordination,
                agent.use_constraint,
            )
            if agent.use_reasoning
            else None
        )
        self.price_coordinator = (
            SatellitePriceCoordinator(env.num_cells, env.config.get("coordination", {}))
            if agent.use_coordination
            else None
        )
        self.reliability_controller = ReliabilityDualController(
            env.config.get("constraint", {}),
            initial_override=agent.outage_dual_mu if agent.dual_state_loaded else None,
        )
        self.agent.outage_dual_mu = self.reliability_controller.mu
        self.last_outage_count = 0.0
        self.last_active_count = 0.0
        self._reset_actor_state()

    def _reset_actor_state(self) -> None:
        self.hidden_states = np.zeros(
            (self.env.num_agents, self.agent.recurrent_hidden_dim), dtype=np.float64
        )
        self.previous_actions = np.full(self.env.num_agents, -1, dtype=np.int64)
        self.previous_outcomes = np.zeros(
            (self.env.num_agents, self.agent.outcome_dim), dtype=np.float64
        )

    def _local_price_features(self) -> np.ndarray:
        if self.price_coordinator is None:
            return np.zeros((self.env.num_agents, 0), dtype=np.float64)
        return self.price_coordinator.local_features(
            [ue.primary_cell for ue in self.env.ues],
            self.env.candidates,
            self.env.candidate_count,
        )

    def _critic_state(self, global_state: np.ndarray) -> np.ndarray:
        parts = [global_state]
        if self.price_coordinator is not None:
            parts.append(self.price_coordinator.normalized_prices())
        if self.agent.use_constraint:
            parts.append(np.asarray([self.reliability_controller.normalized_mu]))
        return np.concatenate(parts).astype(np.float64)

    def update_constraint(self) -> float:
        if self.agent.use_constraint:
            self.reliability_controller.update(
                self.last_outage_count,
                self.last_active_count,
                float(self.env.reward_cfg["epsilon"]),
            )
            self.agent.outage_dual_mu = self.reliability_controller.mu
        return self.reliability_controller.mu

    def collect(self, steps: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        storage: dict[str, list[Any]] = defaultdict(list)
        metrics: dict[str, float] = defaultdict(float)
        rollout_mu = self.reliability_controller.mu if self.agent.use_constraint else 0.0
        normalized_rollout_mu = (
            self.reliability_controller.normalized_mu if self.agent.use_constraint else 0.0
        )
        for _ in range(steps):
            reasoning_result: ReasoningResult | None = None
            reasoning_features = np.zeros(
                (self.env.num_agents, self.env.action_dim), dtype=np.float64
            )
            predicted_deficits = np.zeros_like(reasoning_features)
            price_features = self._local_price_features()
            prices_before = (
                self.price_coordinator.prices
                if self.price_coordinator is not None
                else np.zeros(self.env.num_cells, dtype=np.float64)
            )
            hidden_before = self.hidden_states.copy()
            if self.agent.use_reasoning:
                if self.reasoner is None:
                    raise RuntimeError("reasoning mode has no LookaheadReasoner")
                reasoning_result = self.reasoner.analyze(
                    self.env, self.masks, prices_before, rollout_mu
                )
                reasoning_features = reasoning_result.features
                predicted_deficits = reasoning_result.predicted_deficits
            if self.agent.use_memory:
                storage["previous_actions"].append(self.previous_actions.copy())
                storage["previous_outcomes"].append(self.previous_outcomes.copy())
                storage["reasoning_features"].append(reasoning_features.copy())
                storage["reasoning_predicted_deficits"].append(predicted_deficits.copy())
                storage["hidden_states_before"].append(hidden_before)
                if self.agent.use_coordination:
                    storage["price_features"].append(price_features.copy())
                if self.agent.use_constraint:
                    storage["mu_features"].append(
                        np.full((self.env.num_agents, 1), normalized_rollout_mu, dtype=np.float64)
                    )
                actions, log_probs, new_hidden = self.agent.act_recurrent(
                    self.observations,
                    self.masks,
                    self.previous_actions,
                    self.previous_outcomes,
                    self.hidden_states,
                    reasoning_features,
                    price_features if self.agent.use_coordination else None,
                    normalized_rollout_mu,
                )
            else:
                actions, log_probs = self.agent.act(self.observations, self.masks)
                new_hidden = self.hidden_states
            critic_state = self._critic_state(self.global_state)
            value = self.agent.value(critic_state)[0]
            storage["observations"].append(self.observations.copy())
            storage["global_states"].append(critic_state.copy())
            storage["masks"].append(self.masks.copy())
            storage["actions"].append(actions.copy())
            storage["log_probs"].append(log_probs.copy())
            storage["values"].append(value)
            output = self.env.step(actions)
            if self.agent.use_constraint:
                reward, constraint_penalty = compute_training_reward(
                    output.info["service_deficit"],
                    output.info["resource_cost"],
                    output.info["outage_cost"],
                    float(self.env.reward_cfg["lambda_resource"]),
                    True,
                    rollout_mu,
                    self.reliability_controller.outage_target,
                )
                output.rewards.fill(reward)
                output.info["team_reward"] = reward
                output.info["constraint_penalty"] = constraint_penalty
                output.info["constraint_reward"] = reward
            reward = float(output.rewards.mean())
            storage["rewards"].append(reward)
            storage["dones"].append(float(output.done))
            for key, value_metric in output.info.items():
                metrics[key] += float(value_metric)
            metrics["resource_price_mean"] += float(prices_before.mean())
            metrics["resource_price_max"] += float(prices_before.max(initial=0.0))
            metrics["resource_price_std"] += float(prices_before.std())
            metrics["price_nonzero_fraction"] += float(np.mean(prices_before > 1e-8))
            metrics["outage_dual_mu"] += rollout_mu
            metrics["normalized_outage_dual_mu"] += normalized_rollout_mu
            metrics["outage_target"] += self.reliability_controller.outage_target
            hidden_norm = float(
                np.mean(np.linalg.norm(new_hidden, axis=1) / np.sqrt(self.agent.recurrent_hidden_dim))
            ) if self.agent.use_memory else 0.0
            metrics["hidden_state_norm"] += hidden_norm
            if reasoning_result is not None:
                metrics["reasoning_best_score"] += float(reasoning_result.best_scores.mean())
                metrics["reasoning_score_gap"] += float(reasoning_result.score_gaps.mean())
                ranks = []
                outage_spreads = []
                planning_risk_spreads = []
                for ue_id, action in enumerate(actions):
                    feasible = np.flatnonzero(self.masks[ue_id] > 0.0)
                    ordered = feasible[np.argsort(reasoning_result.raw_costs[ue_id, feasible])]
                    ranks.append(int(np.flatnonzero(ordered == action)[0]) + 1)
                    feasible_risks = reasoning_result.predicted_outages[ue_id, feasible]
                    outage_spreads.append(
                        float(feasible_risks.max() - feasible_risks.min())
                    )
                    feasible_planning_risks = (
                        reasoning_result.planning_reliability_risks[ue_id, feasible]
                    )
                    planning_risk_spreads.append(
                        float(
                            feasible_planning_risks.max()
                            - feasible_planning_risks.min()
                        )
                    )
                metrics["selected_action_reasoning_rank"] += float(np.mean(ranks))
                outage_action_spread = float(np.mean(outage_spreads))
                metrics["reasoning_outage_action_spread"] += outage_action_spread
                metrics["reasoning_outage_discriminating_fraction"] += float(
                    np.mean(np.asarray(outage_spreads) > 1e-6)
                )
                metrics["reasoning_mu_outage_spread"] += rollout_mu * outage_action_spread
                planning_risk_action_spread = float(np.mean(planning_risk_spreads))
                metrics["reasoning_planning_risk_action_spread"] += (
                    planning_risk_action_spread
                )
                metrics["reasoning_planning_risk_discriminating_fraction"] += float(
                    np.mean(np.asarray(planning_risk_spreads) > 1e-6)
                )
                metrics["reasoning_mu_planning_risk_spread"] += (
                    rollout_mu * planning_risk_action_spread
                )
                if output.agent_outcomes is None or output.agent_active is None:
                    raise RuntimeError("environment did not return per-agent Step 2 outcomes")
                selected_prediction = reasoning_result.predicted_deficits[
                    np.arange(self.env.num_agents), actions
                ]
                actual_deficit = output.agent_outcomes[:, 2]
                active = output.agent_active
                prediction_error = float(
                    np.sum(active * np.abs(selected_prediction - actual_deficit))
                    / (np.sum(active) + float(self.env.reward_cfg["epsilon"]))
                )
                metrics["reasoning_prediction_error"] += prediction_error
                selected_coordination_cost = float(
                    np.mean(
                        reasoning_result.coordination_costs[
                            np.arange(self.env.num_agents), actions
                        ]
                    )
                )
                metrics["reasoning_coordination_cost"] += selected_coordination_cost
                metrics["coordination_cost"] += selected_coordination_cost
                selected_rows = np.arange(self.env.num_agents)
                metrics["reasoning_service_cost"] += float(
                    np.mean(reasoning_result.service_costs[selected_rows, actions])
                )
                metrics["reasoning_reliability_cost"] += float(
                    np.mean(reasoning_result.reliability_costs[selected_rows, actions])
                )
                metrics["reasoning_total_cost"] += float(
                    np.mean(reasoning_result.total_costs[selected_rows, actions])
                )
                metrics["reasoning_selected_predicted_outage"] += float(
                    np.mean(reasoning_result.predicted_outages[selected_rows, actions])
                )
                metrics["reasoning_selected_planning_risk"] += float(
                    np.mean(
                        reasoning_result.planning_reliability_risks[
                            selected_rows, actions
                        ]
                    )
                )
                metrics["reasoning_selected_control_cost"] += float(
                    np.mean(reasoning_result.predicted_control_costs[selected_rows, actions])
                )
                metrics["reasoning_selected_handover_cost"] += float(
                    np.mean(reasoning_result.predicted_handover_costs[selected_rows, actions])
                )
                metrics["reasoning_selected_preparation_cost"] += float(
                    np.mean(reasoning_result.predicted_preparation_costs[selected_rows, actions])
                )
            else:
                metrics["reasoning_best_score"] += 0.0
                metrics["reasoning_score_gap"] += 0.0
                metrics["reasoning_prediction_error"] += 0.0
                metrics["selected_action_reasoning_rank"] += 0.0
                metrics["reasoning_coordination_cost"] += 0.0
                metrics["coordination_cost"] += 0.0
                metrics["reasoning_service_cost"] += 0.0
                metrics["reasoning_reliability_cost"] += 0.0
                metrics["reasoning_total_cost"] += 0.0
                metrics["reasoning_selected_predicted_outage"] += 0.0
                metrics["reasoning_selected_control_cost"] += 0.0
                metrics["reasoning_selected_handover_cost"] += 0.0
                metrics["reasoning_selected_preparation_cost"] += 0.0
                metrics["reasoning_outage_action_spread"] += 0.0
                metrics["reasoning_outage_discriminating_fraction"] += 0.0
                metrics["reasoning_mu_outage_spread"] += 0.0
                metrics["reasoning_selected_planning_risk"] += 0.0
                metrics["reasoning_planning_risk_action_spread"] += 0.0
                metrics["reasoning_planning_risk_discriminating_fraction"] += 0.0
                metrics["reasoning_mu_planning_risk_spread"] += 0.0
            if self.agent.use_memory:
                if output.agent_outcomes is None:
                    raise RuntimeError("environment did not return per-agent memory outcomes")
                self.hidden_states = new_hidden
                self.previous_actions = actions.copy()
                self.previous_outcomes = output.agent_outcomes.copy()
            if self.price_coordinator is not None:
                if output.requested_pressure is None:
                    raise RuntimeError("environment did not return requested pressure")
                self.price_coordinator.update(output.requested_pressure)
            self.observations = output.observations
            self.global_state = output.global_state
            self.masks = output.action_masks
            if output.done:
                self.observations, self.global_state, self.masks = self.env.reset()
                self._reset_actor_state()
                if self.price_coordinator is not None:
                    self.price_coordinator.reset()
        bootstrap = self.agent.value(self._critic_state(self.global_state))[0]
        values = np.asarray(storage["values"], dtype=np.float64)
        rewards = np.asarray(storage["rewards"], dtype=np.float64)
        dones = np.asarray(storage["dones"], dtype=np.float64)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        gamma = float(self.agent.config["gamma"])
        gae_lambda = float(self.agent.config["gae_lambda"])
        for t in reversed(range(steps)):
            next_value = bootstrap if t == steps - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae
        batch = {key: np.asarray(value) for key, value in storage.items()}
        batch["advantages"] = advantages
        batch["returns"] = advantages + values
        summary = summarize_metrics(metrics, steps)
        self.last_outage_count = metrics.get("outage_ue_count", 0.0)
        self.last_active_count = metrics.get("active_ue_count", 0.0)
        gap = summary["outage_ratio"] - self.reliability_controller.outage_target
        summary["outage_constraint_gap"] = gap
        summary["constraint_satisfied"] = float(gap <= 0.0)
        summary["outage_constraint_satisfied"] = summary["constraint_satisfied"]
        return batch, summary


def _evaluate_policy_impl(
    env: LEOMobilityEnv,
    agent: MAPPO,
    episodes: int = 5,
    deterministic: bool = True,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    total_steps = 0
    reasoner = (
        LookaheadReasoner(
            env.config.get("reasoning", {}),
            env.reward_cfg,
            env.config.get("coordination", {}),
            agent.use_coordination,
            agent.use_constraint,
        )
        if agent.use_reasoning
        else None
    )
    price_coordinator = (
        SatellitePriceCoordinator(env.num_cells, env.config.get("coordination", {}))
        if agent.use_coordination
        else None
    )
    reliability_controller = ReliabilityDualController(
        env.config.get("constraint", {}), initial_override=agent.outage_dual_mu
    )
    for episode in range(episodes):
        observations, _, masks = env.reset(seed=10_000 + episode)
        if price_coordinator is not None:
            price_coordinator.reset()
        hidden_states = np.zeros(
            (env.num_agents, agent.recurrent_hidden_dim), dtype=np.float64
        )
        previous_actions = np.full(env.num_agents, -1, dtype=np.int64)
        previous_outcomes = np.zeros((env.num_agents, agent.outcome_dim), dtype=np.float64)
        done = False
        while not done:
            prices_before = (
                price_coordinator.prices
                if price_coordinator is not None
                else np.zeros(env.num_cells, dtype=np.float64)
            )
            price_features = (
                price_coordinator.local_features(
                    [ue.primary_cell for ue in env.ues], env.candidates, env.candidate_count
                )
                if price_coordinator is not None
                else np.zeros((env.num_agents, 0), dtype=np.float64)
            )
            current_mu = reliability_controller.mu if agent.use_constraint else 0.0
            normalized_mu = reliability_controller.normalized_mu if agent.use_constraint else 0.0
            reasoning_result = (
                reasoner.analyze(env, masks, prices_before, current_mu)
                if reasoner is not None
                else None
            )
            reasoning_features = (
                reasoning_result.features
                if reasoning_result is not None
                else np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
            )
            if agent.use_memory:
                actions, _, new_hidden = agent.act_recurrent(
                    observations,
                    masks,
                    previous_actions,
                    previous_outcomes,
                    hidden_states,
                    reasoning_features,
                    price_features if agent.use_coordination else None,
                    normalized_mu,
                    deterministic=deterministic,
                )
            else:
                actions, _ = agent.act(observations, masks, deterministic=deterministic)
                new_hidden = hidden_states
            decision_masks = masks
            output = env.step(actions)
            if agent.use_constraint:
                reward, constraint_penalty = compute_training_reward(
                    output.info["service_deficit"],
                    output.info["resource_cost"],
                    output.info["outage_cost"],
                    float(env.reward_cfg["lambda_resource"]),
                    True,
                    current_mu,
                    reliability_controller.outage_target,
                )
                output.rewards.fill(reward)
                output.info["team_reward"] = reward
                output.info["constraint_penalty"] = constraint_penalty
                output.info["constraint_reward"] = reward
            for key, value in output.info.items():
                totals[key] += float(value)
            totals["resource_price_mean"] += float(prices_before.mean())
            totals["resource_price_max"] += float(prices_before.max(initial=0.0))
            totals["resource_price_std"] += float(prices_before.std())
            totals["price_nonzero_fraction"] += float(np.mean(prices_before > 1e-8))
            totals["outage_dual_mu"] += current_mu
            totals["normalized_outage_dual_mu"] += normalized_mu
            totals["outage_target"] += reliability_controller.outage_target
            totals["hidden_state_norm"] += (
                float(np.mean(np.linalg.norm(new_hidden, axis=1) / np.sqrt(agent.recurrent_hidden_dim)))
                if agent.use_memory
                else 0.0
            )
            if reasoning_result is not None:
                totals["reasoning_best_score"] += float(reasoning_result.best_scores.mean())
                totals["reasoning_score_gap"] += float(reasoning_result.score_gaps.mean())
                ranks = []
                outage_spreads = []
                planning_risk_spreads = []
                for ue_id, action in enumerate(actions):
                    feasible = np.flatnonzero(decision_masks[ue_id] > 0.0)
                    ordered = feasible[np.argsort(reasoning_result.raw_costs[ue_id, feasible])]
                    ranks.append(int(np.flatnonzero(ordered == action)[0]) + 1)
                    feasible_risks = reasoning_result.predicted_outages[ue_id, feasible]
                    outage_spreads.append(
                        float(feasible_risks.max() - feasible_risks.min())
                    )
                    feasible_planning_risks = (
                        reasoning_result.planning_reliability_risks[ue_id, feasible]
                    )
                    planning_risk_spreads.append(
                        float(
                            feasible_planning_risks.max()
                            - feasible_planning_risks.min()
                        )
                    )
                totals["selected_action_reasoning_rank"] += float(np.mean(ranks))
                outage_action_spread = float(np.mean(outage_spreads))
                totals["reasoning_outage_action_spread"] += outage_action_spread
                totals["reasoning_outage_discriminating_fraction"] += float(
                    np.mean(np.asarray(outage_spreads) > 1e-6)
                )
                totals["reasoning_mu_outage_spread"] += current_mu * outage_action_spread
                planning_risk_action_spread = float(np.mean(planning_risk_spreads))
                totals["reasoning_planning_risk_action_spread"] += (
                    planning_risk_action_spread
                )
                totals["reasoning_planning_risk_discriminating_fraction"] += float(
                    np.mean(np.asarray(planning_risk_spreads) > 1e-6)
                )
                totals["reasoning_mu_planning_risk_spread"] += (
                    current_mu * planning_risk_action_spread
                )
                if output.agent_outcomes is None or output.agent_active is None:
                    raise RuntimeError("environment did not return per-agent Step 2 outcomes")
                predicted = reasoning_result.predicted_deficits[np.arange(env.num_agents), actions]
                active = output.agent_active
                totals["reasoning_prediction_error"] += float(
                    np.sum(active * np.abs(predicted - output.agent_outcomes[:, 2]))
                    / (np.sum(active) + float(env.reward_cfg["epsilon"]))
                )
                selected_coordination_cost = float(
                    np.mean(
                        reasoning_result.coordination_costs[
                            np.arange(env.num_agents), actions
                        ]
                    )
                )
                totals["reasoning_coordination_cost"] += selected_coordination_cost
                totals["coordination_cost"] += selected_coordination_cost
                selected_rows = np.arange(env.num_agents)
                totals["reasoning_service_cost"] += float(
                    np.mean(reasoning_result.service_costs[selected_rows, actions])
                )
                totals["reasoning_reliability_cost"] += float(
                    np.mean(reasoning_result.reliability_costs[selected_rows, actions])
                )
                totals["reasoning_total_cost"] += float(
                    np.mean(reasoning_result.total_costs[selected_rows, actions])
                )
                totals["reasoning_selected_predicted_outage"] += float(
                    np.mean(reasoning_result.predicted_outages[selected_rows, actions])
                )
                totals["reasoning_selected_planning_risk"] += float(
                    np.mean(
                        reasoning_result.planning_reliability_risks[
                            selected_rows, actions
                        ]
                    )
                )
                totals["reasoning_selected_control_cost"] += float(
                    np.mean(reasoning_result.predicted_control_costs[selected_rows, actions])
                )
                totals["reasoning_selected_handover_cost"] += float(
                    np.mean(reasoning_result.predicted_handover_costs[selected_rows, actions])
                )
                totals["reasoning_selected_preparation_cost"] += float(
                    np.mean(reasoning_result.predicted_preparation_costs[selected_rows, actions])
                )
            else:
                totals["reasoning_best_score"] += 0.0
                totals["reasoning_score_gap"] += 0.0
                totals["reasoning_prediction_error"] += 0.0
                totals["selected_action_reasoning_rank"] += 0.0
                totals["reasoning_coordination_cost"] += 0.0
                totals["coordination_cost"] += 0.0
                totals["reasoning_service_cost"] += 0.0
                totals["reasoning_reliability_cost"] += 0.0
                totals["reasoning_total_cost"] += 0.0
                totals["reasoning_selected_predicted_outage"] += 0.0
                totals["reasoning_selected_control_cost"] += 0.0
                totals["reasoning_selected_handover_cost"] += 0.0
                totals["reasoning_selected_preparation_cost"] += 0.0
                totals["reasoning_outage_action_spread"] += 0.0
                totals["reasoning_outage_discriminating_fraction"] += 0.0
                totals["reasoning_mu_outage_spread"] += 0.0
                totals["reasoning_selected_planning_risk"] += 0.0
                totals["reasoning_planning_risk_action_spread"] += 0.0
                totals["reasoning_planning_risk_discriminating_fraction"] += 0.0
                totals["reasoning_mu_planning_risk_spread"] += 0.0
            if agent.use_memory:
                if output.agent_outcomes is None:
                    raise RuntimeError("environment did not return per-agent memory outcomes")
                hidden_states = new_hidden
                previous_actions = actions.copy()
                previous_outcomes = output.agent_outcomes.copy()
            if price_coordinator is not None:
                if output.requested_pressure is None:
                    raise RuntimeError("environment did not return requested pressure")
                price_coordinator.update(output.requested_pressure)
            observations, masks, done = output.observations, output.action_masks, output.done
            total_steps += 1
    result = summarize_metrics(totals, total_steps)
    gap = result["outage_ratio"] - reliability_controller.outage_target
    result["outage_constraint_gap"] = gap
    result["constraint_satisfied"] = float(gap <= 0.0)
    result["outage_constraint_satisfied"] = result["constraint_satisfied"]
    return result


def evaluate_policy(
    env: LEOMobilityEnv,
    agent: MAPPO,
    episodes: int = 5,
    deterministic: bool = True,
    evaluation_action_seed: int | None = None,
) -> dict[str, float]:
    """Evaluate with an optional action RNG independent of initialization/training.

    None deliberately preserves legacy RNG consumption. A supplied seed installs
    a temporary generator and restores the exact original generator afterward.
    """
    if evaluation_action_seed is None:
        return _evaluate_policy_impl(env, agent, episodes, deterministic)
    original_rng = agent.rng
    agent.rng = np.random.default_rng(int(evaluation_action_seed))
    try:
        return _evaluate_policy_impl(env, agent, episodes, deterministic)
    finally:
        agent.rng = original_rng
