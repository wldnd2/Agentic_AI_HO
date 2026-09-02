from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_mappo import LEOMobilityEnv, load_config  # noqa: E402
from leo_mappo.coordinator import SatellitePriceCoordinator  # noqa: E402
from leo_mappo.runner import summarize_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Short, training-free V3 environment diagnostic")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=30_000)
    args = parser.parse_args()

    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    env = LEOMobilityEnv(config)
    action_rng = np.random.default_rng(args.seed + 991)
    coordinator = SatellitePriceCoordinator(env.num_cells, config.get("coordination", {}))
    totals: dict[str, float] = defaultdict(float)
    intended_sum = np.zeros(env.num_cells, dtype=np.float64)
    preparation_sum = np.zeros(env.num_cells, dtype=np.float64)
    duplication_sum = np.zeros(env.num_cells, dtype=np.float64)
    legal_counts: list[float] = []
    event_realizations = []
    price_means: list[float] = []
    price_maxima: list[float] = []
    preparation_requests_by_ue = np.zeros(env.num_agents, dtype=np.int64)
    preparation_admissions_by_ue = np.zeros(env.num_agents, dtype=np.int64)
    preparation_rejections_by_ue = np.zeros(env.num_agents, dtype=np.int64)
    steps = 0

    for episode in range(args.episodes):
        _, _, masks = env.reset(seed=args.seed + episode)
        coordinator.reset()
        event_realizations.append(env.events.realization_summary())
        done = False
        while not done:
            actions = np.zeros(env.num_agents, dtype=np.int64)
            for ue_id, mask in enumerate(masks):
                legal = np.flatnonzero(mask > 0.0)
                non_stay = legal[legal != 0]
                actions[ue_id] = (
                    int(action_rng.choice(non_stay))
                    if len(non_stay) and action_rng.random() < 0.65
                    else 0
                )
                legal_counts.append(float(len(legal)))
            output = env.step(actions)
            for key, value in output.info.items():
                totals[key] += float(value)
            intended_sum += env.last_coordination_demand
            preparation_sum += env.last_preparation_demand
            duplication_sum += env.last_duplication_demand
            coordinator.update(env.last_coordination_demand)
            prices = coordinator.prices
            price_means.append(float(prices.mean()))
            price_maxima.append(float(prices.max(initial=0.0)))
            masks = output.action_masks
            done = output.done
            steps += 1
        preparation_requests_by_ue += env.preparation_requests_by_ue
        preparation_admissions_by_ue += env.preparation_admissions_by_ue
        preparation_rejections_by_ue += env.preparation_rejections_by_ue

    summary = summarize_metrics(totals, steps)
    report = {
        "episodes": args.episodes,
        "steps": steps,
        "event_realizations": event_realizations,
        "telemetry_sinr_mae_db": summary.get("telemetry_sinr_mae_db", 0.0),
        "telemetry_load_mae": summary.get("telemetry_load_mae", 0.0),
        "latent_degraded_fraction": summary.get("latent_degraded_fraction", 0.0),
        "latent_recovering_fraction": summary.get("latent_recovering_fraction", 0.0),
        "preparation_requests": totals.get("preparation_requested", 0.0),
        "preparation_rejects": totals.get("preparation_rejected", 0.0),
        "preparation_rejection_rate": summary.get("preparation_rejection_rate", 0.0),
        "preparation_requests_by_ue": preparation_requests_by_ue.tolist(),
        "preparation_admissions_by_ue": preparation_admissions_by_ue.tolist(),
        "preparation_rejections_by_ue": preparation_rejections_by_ue.tolist(),
        "duplication_active_fraction": summary.get("duplication_active_time_fraction", 0.0),
        "useful_duplication_intervals": totals.get("useful_duplication_interval_count", 0.0),
        "per_satellite_aggregate_intended_demand": (intended_sum / max(1, steps)).tolist(),
        "per_satellite_preparation_demand": (preparation_sum / max(1, steps)).tolist(),
        "per_satellite_duplication_demand": (duplication_sum / max(1, steps)).tolist(),
        "overload_rate": summary.get("data_overload_event_rate", 0.0),
        "price_mean": float(np.mean(price_means)) if price_means else 0.0,
        "price_max": float(np.max(price_maxima)) if price_maxima else 0.0,
        "price_nonzero_fraction": float(
            np.mean(np.asarray(price_means) > 1e-8)
        ) if price_means else 0.0,
        "legal_action_count": {
            "min": float(np.min(legal_counts)),
            "mean": float(np.mean(legal_counts)),
            "max": float(np.max(legal_counts)),
        },
        "lifecycle": {
            key: summary.get(key, 0.0)
            for key in (
                "prepare_rate",
                "release_rate",
                "prepare_to_release_without_commit_rate",
                "prepare_to_commit_rate",
                "mean_prepared_duration",
                "duplication_activation_rate",
                "commit_rate",
                "rollback_rate",
            )
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
