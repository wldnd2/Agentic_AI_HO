from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from diagnose_v3_outage_controllability import (  # noqa: E402
    plan_potential_ratio,
)
from diagnose_v3_release_decision import (  # noqa: E402
    recurrent_policy,
    sample_actions,
)
from leo_mappo import LEOMobilityEnv, MAPPO, load_config  # noqa: E402
from leo_mappo.coordinator import SatellitePriceCoordinator  # noqa: E402
from leo_mappo.reasoning import LookaheadReasoner  # noqa: E402
from leo_mappo.ue import MobilityState  # noqa: E402


ENVIRONMENT_SEED = 7
AGENT_SEED = 7
EVALUATION_ACTION_SEED = 7007
EPISODE_SEED_BASE = 10_000
HORIZONS = (1, 2, 4, 6)
PAIR_THRESHOLDS = (1e-6, 1e-4, 1e-3, 1e-2)
CHECKPOINT = ROOT / "outputs/v3_validation_420/m5_marginal_headroom/mappo_weights.npz"


def load_experiment() -> tuple[dict[str, Any], LEOMobilityEnv, MAPPO]:
    config = load_config(
        ROOT / "configs/default.yaml",
        [
            ROOT / "configs/validation_agentic_challenge_v3.yaml",
            ROOT / "configs/eta_handover_005.yaml",
            ROOT / "configs/validation_dwell_15.yaml",
            ROOT / "configs/full_agentic.yaml",
            ROOT / "configs/dual_mid_0005.yaml",
        ],
    )
    config["environment_seed"] = ENVIRONMENT_SEED
    config["agent_seed"] = AGENT_SEED
    env = LEOMobilityEnv(config)
    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        config["mappo"],
        seed=AGENT_SEED,
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )
    agent.load(CHECKPOINT)
    if not (
        agent.use_memory
        and agent.use_reasoning
        and agent.use_coordination
        and agent.use_constraint
    ):
        raise RuntimeError("checkpoint/config combination is not M5")
    return config, env, agent


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return correlation(rankdata(left), rankdata(right))


def kendall(left: np.ndarray, right: np.ndarray) -> float | None:
    concordant = 0
    discordant = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            product = (left[first] - left[second]) * (right[first] - right[second])
            if product > 0.0:
                concordant += 1
            elif product < 0.0:
                discordant += 1
    denominator = concordant + discordant
    return float((concordant - discordant) / denominator) if denominator else None


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float(
        (np.sum(ranks[labels]) - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    positives = int(np.sum(labels))
    if positives == 0:
        return None
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered = labels[order]
    cumulative_positive = np.cumsum(ordered)
    group_ends = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    precisions = cumulative_positive[group_ends] / (group_ends + 1)
    recalls = cumulative_positive[group_ends] / positives
    recall_increments = np.diff(np.r_[0.0, recalls])
    return float(np.sum(recall_increments * precisions))


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def risk_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([row["risk"] for row in records], dtype=np.float64)
    result = distribution(values.tolist())
    result["threshold_fractions"] = {
        "lt_1e-12": float(np.mean(values < 1e-12)) if len(values) else 0.0,
        "lt_1e-9": float(np.mean(values < 1e-9)) if len(values) else 0.0,
        "lt_1e-6": float(np.mean(values < 1e-6)) if len(values) else 0.0,
        "lt_1e-3": float(np.mean(values < 1e-3)) if len(values) else 0.0,
        "gt_0.01": float(np.mean(values > 0.01)) if len(values) else 0.0,
        "gt_0.1": float(np.mean(values > 0.1)) if len(values) else 0.0,
    }
    return result


def alignment(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    scores = np.asarray([row["risk"] for row in records], dtype=np.float64)
    labels = np.asarray([row[label] for row in records], dtype=np.float64)
    return {
        "count": len(records),
        "positive_rate": float(np.mean(labels)) if len(labels) else 0.0,
        "roc_auc": roc_auc(scores, labels),
        "pr_auc": pr_auc(scores, labels),
        "brier": float(np.mean((scores - labels) ** 2)) if len(labels) else None,
        "pearson": correlation(scores, labels),
        "spearman": spearman(scores, labels),
    }


def protocol_group(state: MobilityState, timer: int) -> str:
    if state == MobilityState.SINGLE:
        return "SINGLE"
    if state == MobilityState.PREPARED:
        return "PREPARING_PREPARED_READY"
    if state == MobilityState.DUPLICATING:
        return "DUPLICATING"
    if state == MobilityState.SWITCHING:
        return "SWITCHING"
    return "COOLDOWN"


def event_flags(env: LEOMobilityEnv, ue_id: int) -> dict[str, bool]:
    ue = env.ues[ue_id]
    cells = [ue.primary_cell]
    if ue.secondary_cell is not None:
        cells.append(ue.secondary_cell)
    valid_cells = [int(cell) for cell in cells if cell is not None]
    return {
        "latent_degraded": any(
            env.latent_degradation_state[ue_id, cell] == 1 for cell in valid_cells
        ),
        "recovering": any(
            env.latent_degradation_state[ue_id, cell] == 2 for cell in valid_cells
        ),
        "random_blockage": bool(
            np.any(env._active_effects.link_extra_loss_db[ue_id] > 0.0)
        ),
        "capacity_drop": any(
            env._active_effects.cell_capacity_multiplier[cell] < 1.0
            for cell in valid_cells
        ),
        "traffic_burst": bool(env._active_effects.traffic_multiplier[ue_id] > 1.0),
        "switching_transition": ue.state == MobilityState.SWITCHING,
    }


def selected_potential_ratios(
    reasoner: LookaheadReasoner,
    env: LEOMobilityEnv,
    masks: np.ndarray,
    prices: np.ndarray,
    required_rates: np.ndarray,
    mu: float,
    actions: np.ndarray,
) -> np.ndarray:
    available = np.ones(env.num_cells, dtype=np.float64)
    ratios = np.zeros(env.num_agents, dtype=np.float64)
    for ue_id, action_value in enumerate(actions):
        action = int(action_value)
        action_plans = reasoner._plans(env, ue_id, masks[ue_id])[action]
        if len(action_plans) == 1:
            plan, target = action_plans[0]
        else:
            evaluations = [
                reasoner._evaluate_plan(
                    env,
                    ue_id,
                    candidate_plan,
                    candidate_target,
                    float(required_rates[ue_id]),
                    available,
                    prices,
                    mu,
                    0.0,
                    0.0,
                )
                for candidate_plan, candidate_target in action_plans
            ]
            position = min(
                range(len(evaluations)), key=lambda index: evaluations[index][0]
            )
            plan, target = action_plans[position]
        ratios[ue_id] = plan_potential_ratio(
            reasoner,
            env,
            ue_id,
            plan,
            target,
            float(required_rates[ue_id]),
            available,
        )
    return ratios


def add_future_labels(records: list[dict[str, Any]]) -> None:
    by_episode_ue: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_episode_ue[(row["episode"], row["ue_id"])][row["step"]] = row
    for row in records:
        lookup = by_episode_ue[(row["episode"], row["ue_id"])]
        row["Y1"] = bool(row["actual_outage"])
        for horizon, label in ((1, "Y2"), (3, "Y3"), (6, "Y6")):
            row[label] = any(
                lookup.get(row["step"] + offset, {}).get("actual_outage", False)
                for offset in range(horizon + 1)
            )
        row["pre_outage_1_3"] = not row["Y1"] and any(
            lookup.get(row["step"] + offset, {}).get("actual_outage", False)
            for offset in range(1, 4)
        )
        row["pre_outage_1_6"] = not row["Y1"] and any(
            lookup.get(row["step"] + offset, {}).get("actual_outage", False)
            for offset in range(1, 7)
        )


def decile_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = np.argsort(
        np.asarray([row["risk"] for row in records]), kind="mergesort"
    )
    output = []
    for index, positions in enumerate(np.array_split(order, 10), start=1):
        rows = [records[int(position)] for position in positions]
        output.append(
            {
                "decile": index,
                "count": len(rows),
                "mean_predicted_risk": float(np.mean([row["risk"] for row in rows])),
                "current_outage_rate": float(np.mean([row["Y1"] for row in rows])),
                "next_1_outage_rate": float(np.mean([row["Y2"] for row in rows])),
                "next_3_outage_rate": float(np.mean([row["Y3"] for row in rows])),
                "next_6_outage_rate": float(np.mean([row["Y6"] for row in rows])),
            }
        )
    return output


def choose_counterfactual_records(
    records: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(51_007)
    eligible = [row for row in records if len(row["legal_actions"]) >= 2]
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def take(pool: list[dict[str, Any]], amount: int) -> None:
        candidates = [row for row in pool if row["record_id"] not in selected_ids]
        if not candidates:
            return
        positions = rng.permutation(len(candidates))[:amount]
        for position in positions:
            row = candidates[int(position)]
            selected.append(row)
            selected_ids.add(row["record_id"])

    quota = max(1, count // 4)
    take([row for row in eligible if row["pre_outage_1_3"]], quota)
    take(
        [
            row
            for row in eligible
            if row["protocol_group"] == "PREPARING_PREPARED_READY"
        ],
        quota,
    )
    take([row for row in eligible if row["risk_spread"] > 1e-12], quota)
    take(eligible, count - len(selected))
    if len(selected) < min(count, len(eligible)):
        take(eligible, count - len(selected))
    return sorted(selected[:count], key=lambda row: row["record_id"])


def ready_available(env: LEOMobilityEnv, ue_id: int) -> bool:
    ue = env.ues[ue_id]
    return bool(
        ue.secondary_cell is not None
        and ue.state in (MobilityState.PREPARED, MobilityState.DUPLICATING)
        and ue.protocol_timer == 0
    )


def counterfactual_branch(
    snapshot: dict[str, Any],
    agent: MAPPO,
    config: dict[str, Any],
    first_action: int,
) -> dict[int, dict[str, float]]:
    env = deepcopy(snapshot["env"])
    coordinator = deepcopy(snapshot["coordinator"])
    reasoner = LookaheadReasoner(
        config["reasoning"],
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=True,
        use_constraint=True,
    )
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(snapshot["actor_rng_state"])
    actions = snapshot["sampled_actions"].copy()
    actions[snapshot["ue_id"]] = first_action
    observations = snapshot["observations"].copy()
    masks = snapshot["masks"].copy()
    hidden_states = snapshot["new_hidden"].copy()
    previous_actions = snapshot["previous_actions"].copy()
    previous_outcomes = snapshot["previous_outcomes"].copy()
    cumulative = defaultdict(float)
    results: dict[int, dict[str, float]] = {}

    for offset in range(max(HORIZONS)):
        if offset > 0:
            reasoning_result = reasoner.analyze(
                env, masks, coordinator.prices, snapshot["mu"]
            )
            price_features = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            _, probabilities, new_hidden = recurrent_policy(
                agent,
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                reasoning_result.features,
                price_features,
                snapshot["normalized_mu"],
            )
            actions = sample_actions(probabilities, rng)
        output = env.step(actions)
        if output.agent_outcomes is None or output.requested_pressure is None:
            raise RuntimeError("counterfactual branch omitted outcomes")
        cumulative["outage_ue_steps"] += float(output.info["outage_ue_count"])
        cumulative["target_outage_steps"] += float(
            output.agent_outcomes[snapshot["ue_id"], 1] > 0.5
        )
        cumulative["service_deficit"] += float(output.info["service_deficit"])
        cumulative["resource_cost"] += float(output.info["resource_cost"])
        cumulative["ready_steps"] += float(ready_available(env, snapshot["ue_id"]))
        coordinator.update(output.requested_pressure)
        hidden_states = snapshot["new_hidden"].copy() if offset == 0 else new_hidden
        previous_actions = actions.copy()
        previous_outcomes = output.agent_outcomes.copy()
        observations = output.observations
        masks = output.action_masks
        step_count = offset + 1
        if step_count in HORIZONS:
            results[step_count] = dict(cumulative)
        if output.done:
            break
    return results


def replay_counterfactuals(
    config: dict[str, Any], selected: list[dict[str, Any]], episodes: int
) -> dict[int, dict[int, dict[int, dict[str, float]]]]:
    _, env, agent = load_experiment()
    reasoner = LookaheadReasoner(
        config["reasoning"],
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=True,
        use_constraint=True,
    )
    coordinator = SatellitePriceCoordinator(env.num_cells, config["coordination"])
    agent.rng = np.random.default_rng(EVALUATION_ACTION_SEED)
    mu = float(agent.outage_dual_mu)
    normalized_mu = mu / float(config["constraint"]["mu_max"])
    selected_by_step: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_step[(row["episode"], row["step"])].append(row)
    results: dict[int, dict[int, dict[int, dict[str, float]]]] = {}
    completed = 0

    for episode in range(episodes):
        observations, _, masks = env.reset(seed=EPISODE_SEED_BASE + episode)
        coordinator.reset()
        hidden_states = np.zeros(
            (env.num_agents, agent.recurrent_hidden_dim), dtype=np.float64
        )
        previous_actions = np.full(env.num_agents, -1, dtype=np.int64)
        previous_outcomes = np.zeros(
            (env.num_agents, agent.outcome_dim), dtype=np.float64
        )
        done = False
        step = 0
        while not done:
            reasoning_result = reasoner.analyze(
                env, masks, coordinator.prices, mu
            )
            price_features = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            _, probabilities, new_hidden = recurrent_policy(
                agent,
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                reasoning_result.features,
                price_features,
                normalized_mu,
            )
            actions = sample_actions(probabilities, agent.rng)
            selected_here = selected_by_step.get((episode, step), [])
            if selected_here:
                base_env = deepcopy(env)
                actor_rng_state = deepcopy(agent.rng.bit_generator.state)
                for row in selected_here:
                    ue_id = int(row["ue_id"])
                    if int(actions[ue_id]) != int(row["selected_action"]):
                        raise RuntimeError("counterfactual replay diverged")
                    snapshot = {
                        "env": deepcopy(base_env),
                        "coordinator": deepcopy(coordinator),
                        "ue_id": ue_id,
                        "sampled_actions": actions.copy(),
                        "new_hidden": new_hidden.copy(),
                        "observations": observations.copy(),
                        "masks": masks.copy(),
                        "previous_actions": previous_actions.copy(),
                        "previous_outcomes": previous_outcomes.copy(),
                        "actor_rng_state": deepcopy(actor_rng_state),
                        "mu": mu,
                        "normalized_mu": normalized_mu,
                    }
                    results[row["record_id"]] = {
                        action: counterfactual_branch(snapshot, agent, config, action)
                        for action in row["legal_actions"]
                    }
                    completed += 1
                    if completed % 25 == 0 or completed == len(selected):
                        print(
                            f"completed counterfactual state {completed}/{len(selected)}",
                            file=sys.stderr,
                            flush=True,
                        )
            output = env.step(actions)
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("counterfactual replay omitted outcomes")
            coordinator.update(output.requested_pressure)
            hidden_states = new_hidden
            previous_actions = actions.copy()
            previous_outcomes = output.agent_outcomes.copy()
            observations = output.observations
            masks = output.action_masks
            done = output.done
            step += 1
    return results


def regret_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def action_alignment_summary(
    selected: list[dict[str, Any]],
    counterfactuals: dict[int, dict[int, dict[int, dict[str, float]]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    horizon_output: dict[str, Any] = {}
    pair_output: dict[str, Any] = {}
    for horizon in HORIZONS:
        agreements_outage = []
        agreements_deficit = []
        risk_outage_correlations = []
        risk_deficit_correlations = []
        risk_outage_kendall = []
        risk_deficit_kendall = []
        outage_regrets = []
        deficit_regrets = []
        pair_counts = {
            threshold: defaultdict(int) for threshold in PAIR_THRESHOLDS
        }
        evaluated = 0
        for row in selected:
            legal = row["legal_actions"]
            branches = counterfactuals[row["record_id"]]
            if any(horizon not in branches[action] for action in legal):
                continue
            risks = np.asarray([row["action_risks"][str(action)] for action in legal])
            outages = np.asarray(
                [branches[action][horizon]["target_outage_steps"] for action in legal]
            )
            deficits = np.asarray(
                [branches[action][horizon]["service_deficit"] for action in legal]
            )
            risk_choice = int(np.argmin(risks))
            agreements_outage.append(outages[risk_choice] <= np.min(outages) + 1e-12)
            agreements_deficit.append(deficits[risk_choice] <= np.min(deficits) + 1e-12)
            outage_regrets.append(float(outages[risk_choice] - np.min(outages)))
            deficit_regrets.append(float(deficits[risk_choice] - np.min(deficits)))
            for destination, value in (
                (risk_outage_correlations, spearman(risks, outages)),
                (risk_deficit_correlations, spearman(risks, deficits)),
                (risk_outage_kendall, kendall(risks, outages)),
                (risk_deficit_kendall, kendall(risks, deficits)),
            ):
                if value is not None:
                    destination.append(value)
            for first in range(len(legal)):
                for second in range(first + 1, len(legal)):
                    difference = abs(float(risks[first] - risks[second]))
                    if risks[first] <= risks[second]:
                        lower, higher = first, second
                    else:
                        lower, higher = second, first
                    for threshold in PAIR_THRESHOLDS:
                        if difference <= threshold:
                            continue
                        counts = pair_counts[threshold]
                        counts["count"] += 1
                        counts["outage_agree"] += int(
                            outages[lower] <= outages[higher] + 1e-12
                        )
                        counts["outage_strict"] += int(
                            outages[lower] < outages[higher] - 1e-12
                        )
                        counts["deficit_agree"] += int(
                            deficits[lower] <= deficits[higher] + 1e-12
                        )
                        counts["deficit_strict"] += int(
                            deficits[lower] < deficits[higher] - 1e-12
                        )
            evaluated += 1
        horizon_output[str(horizon)] = {
            "state_count": evaluated,
            "lowest_risk_is_lowest_outage_fraction": float(np.mean(agreements_outage)),
            "lowest_risk_is_lowest_deficit_fraction": float(np.mean(agreements_deficit)),
            "mean_state_spearman_risk_vs_outage": float(
                np.mean(risk_outage_correlations)
            ) if risk_outage_correlations else None,
            "spearman_outage_valid_state_count": len(risk_outage_correlations),
            "mean_state_spearman_risk_vs_deficit": float(
                np.mean(risk_deficit_correlations)
            ) if risk_deficit_correlations else None,
            "spearman_deficit_valid_state_count": len(risk_deficit_correlations),
            "mean_state_kendall_risk_vs_outage": float(
                np.mean(risk_outage_kendall)
            ) if risk_outage_kendall else None,
            "mean_state_kendall_risk_vs_deficit": float(
                np.mean(risk_deficit_kendall)
            ) if risk_deficit_kendall else None,
            "outage_regret": regret_summary(outage_regrets),
            "deficit_regret": regret_summary(deficit_regrets),
        }
        pair_output[str(horizon)] = {
            f"risk_difference_gt_{threshold:g}": {
                "pair_count": int(pair_counts[threshold]["count"]),
                "outage_directional_agreement": (
                    pair_counts[threshold]["outage_agree"]
                    / max(pair_counts[threshold]["count"], 1)
                ),
                "outage_strict_improvement": (
                    pair_counts[threshold]["outage_strict"]
                    / max(pair_counts[threshold]["count"], 1)
                ),
                "deficit_directional_agreement": (
                    pair_counts[threshold]["deficit_agree"]
                    / max(pair_counts[threshold]["count"], 1)
                ),
                "deficit_strict_improvement": (
                    pair_counts[threshold]["deficit_strict"]
                    / max(pair_counts[threshold]["count"], 1)
                ),
            }
            for threshold in PAIR_THRESHOLDS
        }
    return horizon_output, pair_output


def mismatch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = np.asarray(
        [row["predicted_potential_ratio"] for row in records], dtype=np.float64
    )
    actual = np.asarray([row["actual_ratio"] for row in records], dtype=np.float64)
    return {
        "count": len(records),
        "predicted_mean": float(np.mean(predicted)) if len(records) else None,
        "predicted_median": float(np.median(predicted)) if len(records) else None,
        "actual_mean": float(np.mean(actual)) if len(records) else None,
        "actual_median": float(np.median(actual)) if len(records) else None,
        "prediction_minus_actual_mean": float(np.mean(predicted - actual))
        if len(records)
        else None,
        **{
            f"fraction_predicted_ge_{threshold:g}_actual_lt_0.8": float(
                np.mean((predicted >= threshold) & (actual < 0.8))
            ) if len(records) else 0.0
            for threshold in (1.2, 2.0, 4.0)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose M5 reliability alignment")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--counterfactual-states", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0 or args.counterfactual_states <= 0:
        raise ValueError("episodes and counterfactual-states must be positive")

    config, env, agent = load_experiment()
    reasoner = LookaheadReasoner(
        config["reasoning"],
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=True,
        use_constraint=True,
    )
    coordinator = SatellitePriceCoordinator(env.num_cells, config["coordination"])
    agent.rng = np.random.default_rng(EVALUATION_ACTION_SEED)
    mu = float(agent.outage_dual_mu)
    normalized_mu = mu / float(config["constraint"]["mu_max"])
    records: list[dict[str, Any]] = []
    capture: dict[str, Any] = {}
    original_schedule = env._schedule_and_transmit

    def captured_schedule(
        required_rates: np.ndarray, active: np.ndarray
    ) -> dict[str, float]:
        delivered_before = np.asarray(
            [ue.delivered_packets for ue in env.ues], dtype=np.int64
        )
        metrics = original_schedule(required_rates, active)
        delivered_after = np.asarray(
            [ue.delivered_packets for ue in env.ues], dtype=np.int64
        )
        realized_goodput = (
            (delivered_after - delivered_before)
            * int(env.traffic_cfg["packet_size_bits"])
            / max(env.interval_s, 1e-8)
        )
        capture["step"] = {
            "active": active.copy(),
            "required_rates": required_rates.copy(),
            "actual_ratio": np.where(
                active > 0.0,
                realized_goodput / (required_rates + env.reward_cfg["epsilon"]),
                0.0,
            ),
        }
        return metrics

    env._schedule_and_transmit = captured_schedule  # type: ignore[method-assign]

    for episode in range(args.episodes):
        observations, _, masks = env.reset(seed=EPISODE_SEED_BASE + episode)
        coordinator.reset()
        hidden_states = np.zeros(
            (env.num_agents, agent.recurrent_hidden_dim), dtype=np.float64
        )
        previous_actions = np.full(env.num_agents, -1, dtype=np.int64)
        previous_outcomes = np.zeros(
            (env.num_agents, agent.outcome_dim), dtype=np.float64
        )
        done = False
        step = 0
        while not done:
            prices = coordinator.prices
            _, observed_required_rates = env.observed_required_rates()
            reasoning_result = reasoner.analyze(env, masks, prices, mu)
            price_features = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            _, probabilities, new_hidden = recurrent_policy(
                agent,
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                reasoning_result.features,
                price_features,
                normalized_mu,
            )
            actions = sample_actions(probabilities, agent.rng)
            potential_ratios = selected_potential_ratios(
                reasoner,
                env,
                masks,
                prices,
                observed_required_rates,
                mu,
                actions,
            )
            pending = []
            for ue in env.ues:
                ue_id = ue.ue_id
                action = int(actions[ue_id])
                legal = [int(value) for value in np.flatnonzero(masks[ue_id] > 0.0)]
                action_risks = {
                    str(legal_action): float(
                        reasoning_result.planning_reliability_risks[
                            ue_id, legal_action
                        ]
                    )
                    for legal_action in legal
                }
                action_service_costs = {
                    str(legal_action): float(
                        reasoning_result.service_costs[ue_id, legal_action]
                    )
                    for legal_action in legal
                }
                action_coordination_costs = {
                    str(legal_action): float(
                        reasoning_result.coordination_costs[ue_id, legal_action]
                    )
                    for legal_action in legal
                }
                action_total_costs = {
                    str(legal_action): float(
                        reasoning_result.total_costs[ue_id, legal_action]
                    )
                    for legal_action in legal
                }
                pending.append(
                    {
                        "episode": episode,
                        "step": step,
                        "ue_id": ue_id,
                        "selected_action": action,
                        "legal_actions": legal,
                        "action_risks": action_risks,
                        "action_service_costs": action_service_costs,
                        "action_coordination_costs": action_coordination_costs,
                        "action_total_costs": action_total_costs,
                        "risk_spread": max(action_risks.values())
                        - min(action_risks.values()),
                        "risk": float(
                            reasoning_result.planning_reliability_risks[ue_id, action]
                        ),
                        "service_cost": float(
                            reasoning_result.service_costs[ue_id, action]
                        ),
                        "coordination_cost": float(
                            reasoning_result.coordination_costs[ue_id, action]
                        ),
                        "total_cost": float(
                            reasoning_result.total_costs[ue_id, action]
                        ),
                        "protocol_state": int(ue.state),
                        "protocol_timer": int(ue.protocol_timer),
                        "protocol_group": protocol_group(
                            MobilityState(ue.state), int(ue.protocol_timer)
                        ),
                        "observed_required_rate": float(
                            observed_required_rates[ue_id]
                        ),
                        "predicted_potential_ratio": float(potential_ratios[ue_id]),
                        "event_flags": event_flags(env, ue_id),
                    }
                )
            output = env.step(actions)
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("environment omitted diagnostic outcomes")
            true_step = capture["step"]
            for ue_id in np.flatnonzero(true_step["active"] > 0.0):
                row = pending[int(ue_id)]
                row.update(
                    {
                        "record_id": len(records),
                        "actual_required_rate": float(
                            true_step["required_rates"][ue_id]
                        ),
                        "actual_ratio": float(true_step["actual_ratio"][ue_id]),
                        "actual_outage": bool(
                            output.agent_outcomes[ue_id, 1] > 0.5
                        ),
                    }
                )
                records.append(row)
            coordinator.update(output.requested_pressure)
            hidden_states = new_hidden
            previous_actions = actions.copy()
            previous_outcomes = output.agent_outcomes.copy()
            observations = output.observations
            masks = output.action_masks
            done = output.done
            step += 1
        print(
            f"completed state-level episode {episode + 1}/{args.episodes}",
            file=sys.stderr,
            flush=True,
        )

    add_future_labels(records)
    selected = choose_counterfactual_records(records, args.counterfactual_states)
    counterfactuals = replay_counterfactuals(config, selected, args.episodes)
    action_summary, pair_summary = action_alignment_summary(
        selected, counterfactuals
    )

    current_outage = [row for row in records if row["Y1"]]
    pre_outage_1_3 = [row for row in records if row["pre_outage_1_3"]]
    pre_outage_1_6 = [row for row in records if row["pre_outage_1_6"]]
    non_outage = [row for row in records if not row["Y6"]]
    protocol_output = {}
    for group in (
        "SINGLE",
        "PREPARING_PREPARED_READY",
        "DUPLICATING",
        "SWITCHING",
        "COOLDOWN",
    ):
        subset = [row for row in records if row["protocol_group"] == group]
        protocol_output[group] = {
            "risk": risk_distribution(subset),
            "alignment": {
                label: alignment(subset, label) for label in ("Y1", "Y2", "Y3", "Y6")
            },
        }
    event_output = {}
    event_population = [row for row in records if row["Y1"] or row["pre_outage_1_6"]]
    for event_name in (
        "latent_degraded",
        "recovering",
        "random_blockage",
        "capacity_drop",
        "traffic_burst",
        "switching_transition",
    ):
        subset = [row for row in event_population if row["event_flags"][event_name]]
        event_output[event_name] = {
            "count": len(subset),
            "mean_predicted_risk": float(np.mean([row["risk"] for row in subset]))
            if subset
            else None,
            "current_outage_rate": float(np.mean([row["Y1"] for row in subset]))
            if subset
            else None,
            "future_Y6_outage_rate": float(np.mean([row["Y6"] for row in subset]))
            if subset
            else None,
        }

    report = {
        "configuration": {
            "episodes": args.episodes,
            "environment_seed": ENVIRONMENT_SEED,
            "agent_seed": AGENT_SEED,
            "evaluation_action_seed": EVALUATION_ACTION_SEED,
            "episode_seed_base": EPISODE_SEED_BASE,
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_mu": mu,
            "reasoning_horizon": int(config["reasoning"]["horizon_steps"]),
            "counterfactual_requested": args.counterfactual_states,
        },
        "counts": {
            "active_states": len(records),
            "counterfactual_states": len(selected),
            "counterfactual_pre_outage_1_3": sum(
                row["pre_outage_1_3"] for row in selected
            ),
            "counterfactual_preparing_ready": sum(
                row["protocol_group"] == "PREPARING_PREPARED_READY"
                for row in selected
            ),
            "counterfactual_nonzero_risk_spread": sum(
                row["risk_spread"] > 1e-12 for row in selected
            ),
        },
        "risk_saturation": {
            "all": risk_distribution(records),
            "current_outage": risk_distribution(current_outage),
            "pre_outage_1_3": risk_distribution(pre_outage_1_3),
            "pre_outage_1_6": risk_distribution(pre_outage_1_6),
            "non_outage_through_6": risk_distribution(non_outage),
        },
        "state_predictive_alignment": {
            label: alignment(records, label) for label in ("Y1", "Y2", "Y3", "Y6")
        },
        "risk_deciles": decile_table(records),
        "protocol_state_alignment": protocol_output,
        "action_ranking_alignment": action_summary,
        "pairwise_directional_accuracy": pair_summary,
        "potential_goodput_vs_realized": {
            "all": mismatch_summary(records),
            "current_outage": mismatch_summary(current_outage),
            "pre_outage_1_3": mismatch_summary(pre_outage_1_3),
            "pre_outage_1_6": mismatch_summary(pre_outage_1_6),
        },
        "event_conditional_mismatch": event_output,
        "label_semantics": {
            "Y1": "current-step outage",
            "Y2": "at least one outage from t through t+1",
            "Y3": "at least one outage from t through t+3",
            "Y6": "at least one outage from t through t+6",
        },
        "counterfactual_randomness_limitation": (
            "All legal-action branches start from identical environment and actor RNG "
            "states and keep other UEs' first actions fixed. Action-dependent environment "
            "RNG consumption can desynchronize later exogenous draws."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
