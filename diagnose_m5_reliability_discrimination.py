from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from leo_mappo import LEOMobilityEnv, MAPPO, load_config  # noqa: E402
from leo_mappo.coordinator import SatellitePriceCoordinator  # noqa: E402
from leo_mappo.reasoning import LookaheadReasoner  # noqa: E402


MU_VALUES = (0.0, 0.1, 0.2, 0.325419922, 0.5, 1.0)
ACTION_NAMES = ("STAY", "PREPARE", "DUP_ON", "DUP_OFF", "COMMIT", "RELEASE", "ROLLBACK")


def action_name(env: LEOMobilityEnv, action: int) -> str:
    if action == 0:
        return "STAY"
    if env.prepare_start <= action < env.prepare_start + env.candidate_count:
        return "PREPARE"
    mapping = {
        env.dup_on_action: "DUP_ON",
        env.dup_off_action: "DUP_OFF",
        env.commit_action: "COMMIT",
        env.release_action: "RELEASE",
        env.rollback_action: "ROLLBACK",
    }
    return mapping[action]


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def kendall_rank_correlation(reference: np.ndarray, current: np.ndarray) -> float:
    size = len(reference)
    if size < 2:
        return 1.0
    concordant = 0
    discordant = 0
    reference_ties = 0
    current_ties = 0
    for left in range(size):
        for right in range(left + 1, size):
            reference_difference = reference[left] - reference[right]
            current_difference = current[left] - current[right]
            product = reference_difference * current_difference
            if product > 0.0:
                concordant += 1
            elif product < 0.0:
                discordant += 1
            reference_ties += int(reference_difference == 0.0)
            current_ties += int(current_difference == 0.0)
    pairs = size * (size - 1) // 2
    denominator = np.sqrt(
        float((pairs - reference_ties) * (pairs - current_ties))
    )
    if denominator == 0.0:
        return 1.0 if np.array_equal(reference, current) else 0.0
    return float((concordant - discordant) / denominator)


def load_experiment() -> tuple[dict, LEOMobilityEnv, MAPPO]:
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
    config["environment_seed"] = 7
    config["agent_seed"] = 7
    env = LEOMobilityEnv(config)
    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        config["mappo"],
        seed=int(config["agent_seed"]),
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )
    agent.load(ROOT / "outputs/v3_final_1500/m5_seed7/mappo_weights.npz")
    return config, env, agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose M5 reliability action discrimination")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-seed-base", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    config, env, agent = load_experiment()
    if not (agent.use_memory and agent.use_reasoning and agent.use_coordination and agent.use_constraint):
        raise RuntimeError("checkpoint/config combination is not M5")
    reasoner = LookaheadReasoner(
        config.get("reasoning", {}),
        env.reward_cfg,
        config.get("coordination", {}),
        use_coordination=True,
        use_constraint=True,
    )
    coordinator = SatellitePriceCoordinator(env.num_cells, config.get("coordination", {}))
    trajectory_mu = float(agent.outage_dual_mu)
    normalized_trajectory_mu = trajectory_mu / float(config["constraint"]["mu_max"])

    canonical_risk_spreads: list[float] = []
    lower_risk_reductions: list[float] = []
    action_risks: dict[str, list[float]] = defaultdict(list)
    by_mu = {
        mu: {
            "risk_spread": [],
            "service_spread": [],
            "coordination_spread": [],
            "reliability_contribution_spread": [],
            "applied_reliability_cost_spread": [],
            "ratio_to_service": [],
            "ratio_to_coordination": [],
            "best_action_flips": 0,
            "top2_order_changes": 0,
            "rank_correlations": [],
        }
        for mu in MU_VALUES
    }
    evaluated_states = 0
    legal_action_evaluations_per_mu = 0
    epsilon = float(env.reward_cfg["epsilon"])

    for episode in range(args.episodes):
        observations, _, masks = env.reset(seed=args.episode_seed_base + episode)
        coordinator.reset()
        hidden_states = np.zeros(
            (env.num_agents, agent.recurrent_hidden_dim), dtype=np.float64
        )
        previous_actions = np.full(env.num_agents, -1, dtype=np.int64)
        previous_outcomes = np.zeros((env.num_agents, agent.outcome_dim), dtype=np.float64)
        done = False
        while not done:
            prices = coordinator.prices
            results = {
                mu: reasoner.analyze(env, masks, prices, mu) for mu in MU_VALUES
            }
            active, _ = env.observed_required_rates()
            for ue_id in np.flatnonzero(active > 0.0):
                legal = np.flatnonzero(masks[ue_id] > 0.0)
                if legal.size == 0:
                    continue
                evaluated_states += 1
                legal_action_evaluations_per_mu += int(legal.size)
                reference = results[0.0]
                reference_costs = reference.total_costs[ue_id, legal]
                reference_order = np.argsort(reference_costs, kind="stable")
                reference_best_position = int(reference_order[0])
                reference_best_risk = float(
                    reference.planning_reliability_risks[
                        ue_id, legal[reference_best_position]
                    ]
                )
                legal_reference_risks = reference.planning_reliability_risks[ue_id, legal]
                minimum_risk = float(np.min(legal_reference_risks))
                lower_risk_reductions.append(max(0.0, reference_best_risk - minimum_risk))
                canonical_risk_spreads.append(
                    float(np.max(legal_reference_risks) - np.min(legal_reference_risks))
                )
                for action, risk in zip(legal, legal_reference_risks):
                    action_risks[action_name(env, int(action))].append(float(risk))

                for mu, result in results.items():
                    risks = result.planning_reliability_risks[ue_id, legal]
                    services = result.service_costs[ue_id, legal]
                    coordination = result.coordination_costs[ue_id, legal]
                    reliability_costs = result.reliability_costs[ue_id, legal]
                    totals = result.total_costs[ue_id, legal]
                    risk_spread = float(np.max(risks) - np.min(risks))
                    service_spread = float(np.max(services) - np.min(services))
                    coordination_spread = float(
                        np.max(coordination) - np.min(coordination)
                    )
                    contribution_spread = float(mu * risk_spread)
                    bucket = by_mu[mu]
                    bucket["risk_spread"].append(risk_spread)
                    bucket["service_spread"].append(service_spread)
                    bucket["coordination_spread"].append(coordination_spread)
                    bucket["reliability_contribution_spread"].append(
                        contribution_spread
                    )
                    bucket["applied_reliability_cost_spread"].append(
                        float(np.max(reliability_costs) - np.min(reliability_costs))
                    )
                    bucket["ratio_to_service"].append(
                        contribution_spread / (service_spread + epsilon)
                    )
                    bucket["ratio_to_coordination"].append(
                        contribution_spread / (coordination_spread + epsilon)
                    )
                    order = np.argsort(totals, kind="stable")
                    bucket["best_action_flips"] += int(order[0] != reference_order[0])
                    bucket["top2_order_changes"] += int(
                        not np.array_equal(order[:2], reference_order[:2])
                    )
                    bucket["rank_correlations"].append(
                        kendall_rank_correlation(reference_costs, totals)
                    )

            trajectory_result = min(
                results.items(), key=lambda item: abs(item[0] - trajectory_mu)
            )[1]
            price_features = coordinator.local_features(
                [ue.primary_cell for ue in env.ues],
                env.candidates,
                env.candidate_count,
            )
            actions, _, new_hidden = agent.act_recurrent(
                observations,
                masks,
                previous_actions,
                previous_outcomes,
                hidden_states,
                trajectory_result.features,
                price_features,
                normalized_trajectory_mu,
                deterministic=False,
            )
            output = env.step(actions)
            if output.agent_outcomes is None or output.requested_pressure is None:
                raise RuntimeError("environment omitted M5 trajectory diagnostics")
            coordinator.update(output.requested_pressure)
            hidden_states = new_hidden
            previous_actions = actions.copy()
            previous_outcomes = output.agent_outcomes.copy()
            observations = output.observations
            masks = output.action_masks
            done = output.done
        print(
            f"completed episode {episode + 1}/{args.episodes}", file=sys.stderr, flush=True
        )

    risk_summary = distribution(canonical_risk_spreads)
    for threshold in (1e-4, 1e-3, 1e-2):
        risk_summary[f"fraction_gt_{threshold:g}"] = float(
            np.mean(np.asarray(canonical_risk_spreads) > threshold)
        )

    scale_summary = {}
    ranking_summary = {}
    for mu, bucket in by_mu.items():
        key = f"{mu:g}"
        reliability_mean = float(np.mean(bucket["reliability_contribution_spread"]))
        service_mean = float(np.mean(bucket["service_spread"]))
        coordination_mean = float(np.mean(bucket["coordination_spread"]))
        scale_summary[key] = {
            "mean_reliability_contribution_spread": reliability_mean,
            "mean_applied_reliability_cost_spread": float(
                np.mean(bucket["applied_reliability_cost_spread"])
            ),
            "mean_service_cost_spread": service_mean,
            "mean_coordination_cost_spread": coordination_mean,
            "mean_ratio_to_service": float(np.mean(bucket["ratio_to_service"])),
            "mean_ratio_to_coordination": float(
                np.mean(bucket["ratio_to_coordination"])
            ),
            "ratio_of_means_to_service": reliability_mean / (service_mean + epsilon),
            "ratio_of_means_to_coordination": reliability_mean
            / (coordination_mean + epsilon),
        }
        ranking_summary[key] = {
            "best_action_flip_fraction": float(
                bucket["best_action_flips"] / max(1, evaluated_states)
            ),
            "top2_order_change_fraction": float(
                bucket["top2_order_changes"] / max(1, evaluated_states)
            ),
            "mean_kendall_rank_correlation": float(
                np.mean(bucket["rank_correlations"])
            ),
        }

    reductions = np.asarray(lower_risk_reductions, dtype=np.float64)
    meaningful = reductions > 1e-4
    positive_reductions = reductions[meaningful]
    alternative_summary = {
        "meaningful_threshold": 1e-4,
        "fraction_with_lower_risk_legal_alternative": float(np.mean(meaningful)),
        "mean_achievable_risk_reduction_all_states": float(np.mean(reductions)),
        "median_achievable_risk_reduction_all_states": float(np.median(reductions)),
        "mean_reduction_when_available": float(
            np.mean(positive_reductions) if positive_reductions.size else 0.0
        ),
        "median_reduction_when_available": float(
            np.median(positive_reductions) if positive_reductions.size else 0.0
        ),
    }
    action_summary = {
        name: {
            "count": len(action_risks.get(name, [])),
            "mean_planning_risk": float(
                np.mean(action_risks[name]) if action_risks.get(name) else 0.0
            ),
            "median_planning_risk": float(
                np.median(action_risks[name]) if action_risks.get(name) else 0.0
            ),
        }
        for name in ACTION_NAMES
    }
    report = {
        "configuration": {
            "episodes": args.episodes,
            "environment_seed": 7,
            "agent_seed": 7,
            "episode_seed_base": args.episode_seed_base,
            "checkpoint_mu": trajectory_mu,
            "mu_sweep": list(MU_VALUES),
        },
        "counts": {
            "evaluated_active_ue_states": evaluated_states,
            "legal_action_evaluations_per_mu": legal_action_evaluations_per_mu,
            "total_mu_swept_legal_action_evaluations": (
                legal_action_evaluations_per_mu * len(MU_VALUES)
            ),
        },
        "reliability_risk_spread_at_mu_0": risk_summary,
        "relative_action_discrimination_scale": scale_summary,
        "action_ranking_changes_vs_mu_0": ranking_summary,
        "lower_risk_alternative": alternative_summary,
        "action_type_risk_at_mu_0": action_summary,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
