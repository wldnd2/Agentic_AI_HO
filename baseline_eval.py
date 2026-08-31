from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.baselines import always_duplication_policy, max_sinr_policy, threshold_duplication_policy
from leo_mappo.runner import summarize_metrics


POLICIES = {
    "max_sinr": max_sinr_policy,
    "always_dup": always_duplication_policy,
    "threshold_dup": threshold_duplication_policy,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="threshold_dup")
    parser.add_argument("--output", default=None, help="Optional directory for evaluation JSON and CSV")
    args = parser.parse_args()
    env = LEOMobilityEnv(load_config(args.config, args.override))
    totals: dict[str, float] = defaultdict(float)
    total_steps = 0
    for episode in range(args.episodes):
        env.reset(seed=20_000 + episode)
        done = False
        while not done:
            output = env.step(POLICIES[args.policy](env))
            for key, value in output.info.items():
                totals[key] += float(value)
            done = output.done
            total_steps += 1
    result = summarize_metrics(totals, total_steps)
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        with (output_dir / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(result))
            writer.writeheader()
            writer.writerow(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
