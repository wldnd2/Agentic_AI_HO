"""Paired, evaluation-only M4 price ablation across five seeds."""

import csv
import math
import os
from statistics import mean, stdev

from leo_mappo import LEOMobilityEnv, MAPPO
from leo_mappo.runner import evaluate_policy

from eval_m4_price_ablation import AGENT_SEED, EPISODES, ROOT, WEIGHTS, make_config


EVALUATION_SEEDS = (7, 8, 9, 10, 11)
RAW_CSV = ROOT / "outputs" / "m4_price_ablation_multiseed_raw.csv"
T_95_DF4 = 2.7764451051977987
REQUIRED_METRICS = (
    "outage_ratio",
    "service_deficit",
    "resource_cost",
    "dro",
    "handover_started",
    "pingpong_rate",
    "duplication_active_time_fraction",
    "team_reward",
)
CSV_METADATA = (
    "evaluation_seed",
    "condition",
    "environment_seed",
    "action_seed",
    "episodes",
    "checkpoint",
)


def evaluate_seed(evaluation_seed: int, zero_price: bool) -> dict[str, float]:
    cfg = make_config(zero_price=zero_price)
    cfg["environment_seed"] = evaluation_seed
    env = LEOMobilityEnv(cfg)

    # evaluate_policy uses reset(seed=10_000 + episode). Give each evaluation
    # seed a disjoint episode range; seed 7 keeps the original single-seed run.
    original_reset = env.reset
    episode_offset = (evaluation_seed - EVALUATION_SEEDS[0]) * EPISODES
    started_episodes = 0
    condition = "price_zero" if zero_price else "normal"

    def paired_reset(seed=None):
        nonlocal started_episodes
        if started_episodes % 10 == 0:
            print(
                f"[seed={evaluation_seed} {condition}] "
                f"{started_episodes}/{EPISODES} episodes complete",
                flush=True,
            )
        reset_seed = None if seed is None else seed + episode_offset
        result = original_reset(seed=reset_seed)
        started_episodes += 1
        return result

    env.reset = paired_reset
    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        cfg["mappo"],
        AGENT_SEED,
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )
    agent.load(str(WEIGHTS))
    if not agent.use_coordination:
        raise RuntimeError("Loaded checkpoint is not an M4 coordination policy.")

    result = evaluate_policy(
        env,
        agent,
        episodes=EPISODES,
        deterministic=False,
        evaluation_action_seed=7000 + evaluation_seed,
    )
    print(
        f"[seed={evaluation_seed} {condition}] {EPISODES}/{EPISODES} episodes complete",
        flush=True,
    )
    return result


def summary(values: list[float]) -> tuple[float, float, float, float]:
    average = mean(values)
    standard_deviation = stdev(values)
    margin = T_95_DF4 * standard_deviation / math.sqrt(len(values))
    return average, standard_deviation, average - margin, average + margin


def load_completed() -> tuple[dict[tuple[int, str], dict[str, float]], list[str] | None]:
    if not RAW_CSV.exists():
        return {}, None

    results = {}
    with RAW_CSV.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if not set(CSV_METADATA).issubset(fieldnames):
            raise ValueError(f"Invalid resume CSV header: {RAW_CSV}")
        metric_keys = [key for key in fieldnames if key not in CSV_METADATA]
        if not set(REQUIRED_METRICS).issubset(metric_keys):
            raise ValueError(f"Required metrics are missing from {RAW_CSV}")

        for row in reader:
            # An interruption during the last write can leave a partial final row.
            if None in row or any(row.get(key) in (None, "") for key in fieldnames):
                print("Ignoring incomplete final CSV row", flush=True)
                break
            seed = int(row["evaluation_seed"])
            condition = row["condition"]
            if (
                seed not in EVALUATION_SEEDS
                or condition not in ("normal", "price_zero")
                or int(row["environment_seed"]) != seed
                or int(row["action_seed"]) != 7000 + seed
                or int(row["episodes"]) != EPISODES
                or row["checkpoint"] != str(WEIGHTS)
            ):
                raise ValueError(f"Resume CSV settings disagree with this run: {RAW_CSV}")
            key = (seed, condition)
            if key in results:
                raise ValueError(f"Duplicate completed evaluation in {RAW_CSV}: {key}")
            results[key] = {metric: float(row[metric]) for metric in metric_keys}
    return results, metric_keys


def save_completed(
    results: dict[tuple[int, str], dict[str, float]], metric_keys: list[str]
) -> None:
    temporary_csv = RAW_CSV.with_name(RAW_CSV.name + ".tmp")
    with temporary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*CSV_METADATA, *metric_keys])
        writer.writeheader()
        for seed in EVALUATION_SEEDS:
            for condition in ("normal", "price_zero"):
                key = (seed, condition)
                if key not in results:
                    continue
                writer.writerow(
                    {
                        "evaluation_seed": seed,
                        "condition": condition,
                        "environment_seed": seed,
                        "action_seed": 7000 + seed,
                        "episodes": EPISODES,
                        "checkpoint": str(WEIGHTS),
                        **results[key],
                    }
                )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_csv, RAW_CSV)


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    results, metric_keys = load_completed()
    for evaluation_seed in EVALUATION_SEEDS:
        for zero_price in (False, True):
            condition = "price_zero" if zero_price else "normal"
            key = (evaluation_seed, condition)
            if key in results:
                print(f"[seed={evaluation_seed} {condition}] complete; skipping", flush=True)
                continue
            metrics = evaluate_seed(evaluation_seed, zero_price)
            if metric_keys is None:
                metric_keys = sorted(metrics)
                missing = set(REQUIRED_METRICS) - set(metric_keys)
                if missing:
                    raise KeyError(f"Missing evaluation metrics: {sorted(missing)}")
            elif set(metrics) != set(metric_keys):
                raise ValueError("Evaluation metric keys changed between runs")
            results[key] = metrics
            save_completed(results, metric_keys)

    print(f"\nRaw metrics: {RAW_CSV}")
    print("95% CI uses the two-sided t interval with 4 degrees of freedom.")
    print("metric,condition,mean,std,ci95_low,ci95_high")
    for metric in metric_keys:
        normal = [results[seed, "normal"][metric] for seed in EVALUATION_SEEDS]
        zero = [results[seed, "price_zero"][metric] for seed in EVALUATION_SEEDS]
        for condition, values in (
            ("normal", normal),
            ("price_zero", zero),
            ("price_zero_minus_normal", [z - n for z, n in zip(zero, normal)]),
        ):
            average, standard_deviation, low, high = summary(values)
            print(
                f"{metric},{condition},{average:.9g},{standard_deviation:.9g},"
                f"{low:.9g},{high:.9g}"
            )


if __name__ == "__main__":
    main()
