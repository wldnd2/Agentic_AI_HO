from pathlib import Path
from collections import defaultdict
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.runner import summarize_metrics
from leo_mappo.ue import MobilityState


# ============================================================
# Settings
# ============================================================

ROOT = Path(".").resolve()

MODEL_ROOT = (
    ROOT
    / "outputs"
    / "v3_convergence_anneal_2000"
)

MODEL_DIRS = {
    "M1": MODEL_ROOT / "m1_seed7",
    "M2": MODEL_ROOT / "m2_seed7",
    "M3": MODEL_ROOT / "m3_seed7",
    "M4": MODEL_ROOT / "m4_seed7",
    "M5": MODEL_ROOT / "m5_dual_only_seed7",
}

OUTPUT = ROOT / "paper_stage1_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = ROOT / "configs" / "default.yaml"

ENV_OVERRIDES = [
    ROOT / "configs" / "validation_agentic_challenge_v3.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
]

EPISODES = 200
MOVING_AVERAGE_WINDOW = 100

ENV_SEED = 7

# Same environment realizations as MAPPO evaluation:
# evaluate_policy() uses reset(seed=10000 + episode)
EVALUATION_SEED_BASE = 10000


# ============================================================
# Config
# ============================================================

cfg = load_config(
    str(BASE_CONFIG),
    [str(p) for p in ENV_OVERRIDES],
)

cfg["environment_seed"] = ENV_SEED

LAMBDA_RESOURCE = float(
    cfg["reward"]["lambda_resource"]
)

print("=" * 80)
print("PAPER STAGE 1")
print("=" * 80)
print(f"lambda_resource = {LAMBDA_RESOURCE}")
print(f"heuristic episodes = {EPISODES}")
print()


# ============================================================
# Utilities
# ============================================================

def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return values

    result = np.empty_like(values)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start:i + 1])

    return result


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def common_reward(service_deficit, resource_cost):
    return (
        -float(service_deficit)
        - LAMBDA_RESOURCE * float(resource_cost)
    )


def mask_action(env, ue_id, desired_action):
    masks = env.action_masks()

    if (
        0 <= desired_action < env.action_dim
        and masks[ue_id, desired_action] > 0.0
    ):
        return desired_action

    return 0


# ============================================================
# PART A
# Common reward convergence M1-M5
# ============================================================

print("=" * 80)
print("PART A: COMMON REWARD CONVERGENCE")
print("=" * 80)

common_histories = {}
tail_rows = []


for model, directory in MODEL_DIRS.items():

    csv_path = directory / "training_metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{model}: missing {csv_path}"
        )

    rows = read_csv(csv_path)

    rewards = []

    for row in rows:

        reward = common_reward(
            row["service_deficit"],
            row["resource_cost"],
        )

        rewards.append(reward)

    common_histories[model] = rewards

    tail_row = {
        "model": model,
        "iterations": len(rewards),
    }

    for window in [100, 200, 500]:

        values = rewards[-window:]

        tail_row[
            f"common_reward_last{window}"
        ] = float(np.mean(values))

        tail_row[
            f"common_reward_std_last{window}"
        ] = float(np.std(values))

    tail_rows.append(tail_row)

    print(
        f"{model}: "
        f"last100={tail_row['common_reward_last100']:.6f} | "
        f"last200={tail_row['common_reward_last200']:.6f} | "
        f"last500={tail_row['common_reward_last500']:.6f}"
    )


# ------------------------------------------------------------
# Save tail CSV
# ------------------------------------------------------------

tail_csv = (
    OUTPUT
    / "common_reward_training_summary.csv"
)

with tail_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(tail_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(tail_rows)


# ------------------------------------------------------------
# Plot common reward
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5),
)

for model in ["M1", "M2", "M3", "M4", "M5"]:

    rewards = common_histories[model]

    smooth = moving_average(
        rewards,
        MOVING_AVERAGE_WINDOW,
    )

    x = np.arange(
        1,
        len(rewards) + 1,
    )

    ax.plot(
        x,
        smooth,
        label=model,
    )

ax.set_xlabel("Training iteration")
ax.set_ylabel(
    "Common reward "
    r"$(-C_{\mathrm{def}}-\lambda C_{\mathrm{ctrl}})$"
)

ax.legend()

fig.tight_layout()

common_fig = (
    OUTPUT
    / "figure_common_reward_convergence.png"
)

fig.savefig(
    common_fig,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print(f"Saved: {common_fig}")
print()


# ============================================================
# PART B
# Fair V3 heuristic baselines
#
# IMPORTANT:
# Use observed/delayed/noisy SINR rather than hidden true SINR.
# ============================================================

print("=" * 80)
print("PART B: HEURISTIC BASELINE EVALUATION")
print("=" * 80)


def max_observed_sinr_policy(env):

    actions = np.zeros(
        env.num_agents,
        dtype=np.int64,
    )

    for ue in env.ues:

        uid = ue.ue_id

        primary = ue.primary_cell

        if primary is None:
            continue

        primary_sinr = (
            env.observed_sinr[uid, primary]
        )

        if (
            ue.state == MobilityState.SINGLE
            and env.candidates[uid]
        ):

            slot = max(
                range(
                    len(env.candidates[uid])
                ),
                key=lambda index:
                    env.observed_sinr[
                        uid,
                        env.candidates[uid][index],
                    ],
            )

            target = env.candidates[uid][slot]

            target_sinr = (
                env.observed_sinr[
                    uid,
                    target,
                ]
            )

            if target_sinr > primary_sinr + 1.0:

                actions[uid] = (
                    env.prepare_start
                    + slot
                )

        elif (
            ue.state == MobilityState.PREPARED
            and ue.protocol_timer <= 0
        ):

            actions[uid] = env.commit_action

    for uid in range(env.num_agents):

        actions[uid] = mask_action(
            env,
            uid,
            int(actions[uid]),
        )

    return actions


def always_duplication_observed_policy(env):

    actions = np.zeros(
        env.num_agents,
        dtype=np.int64,
    )

    for ue in env.ues:

        uid = ue.ue_id

        if (
            ue.state == MobilityState.SINGLE
            and env.candidates[uid]
        ):

            slot = max(
                range(
                    len(env.candidates[uid])
                ),
                key=lambda index:
                    env.observed_sinr[
                        uid,
                        env.candidates[uid][index],
                    ],
            )

            actions[uid] = (
                env.prepare_start
                + slot
            )

        elif (
            ue.state == MobilityState.PREPARED
            and ue.protocol_timer <= 0
        ):

            actions[uid] = env.dup_on_action

        elif (
            ue.state == MobilityState.DUPLICATING
        ):

            actions[uid] = env.commit_action

    for uid in range(env.num_agents):

        actions[uid] = mask_action(
            env,
            uid,
            int(actions[uid]),
        )

    return actions


def threshold_duplication_observed_policy(
    env,
    threshold_db=4.0,
):

    actions = np.zeros(
        env.num_agents,
        dtype=np.int64,
    )

    for ue in env.ues:

        uid = ue.ue_id
        primary = ue.primary_cell

        if primary is None:
            continue

        primary_sinr = (
            env.observed_sinr[
                uid,
                primary,
            ]
        )

        if (
            ue.state == MobilityState.SINGLE
            and env.candidates[uid]
            and primary_sinr < threshold_db
        ):

            # Candidate 0 is V3's highest-ranked
            # observed-current-SINR candidate.
            actions[uid] = env.prepare_start

        elif (
            ue.state == MobilityState.PREPARED
            and ue.protocol_timer <= 0
        ):

            actions[uid] = env.dup_on_action

        elif (
            ue.state == MobilityState.DUPLICATING
        ):

            target = ue.secondary_cell

            if target is not None:

                target_sinr = (
                    env.observed_sinr[
                        uid,
                        target,
                    ]
                )

                if (
                    target_sinr
                    > primary_sinr + 1.0
                ):

                    actions[uid] = (
                        env.commit_action
                    )

    for uid in range(env.num_agents):

        actions[uid] = mask_action(
            env,
            uid,
            int(actions[uid]),
        )

    return actions


BASELINES = {
    "Max-Observed-SINR":
        max_observed_sinr_policy,

    "Always-Dup":
        always_duplication_observed_policy,

    "Threshold-Dup":
        threshold_duplication_observed_policy,
}


def evaluate_heuristic(
    name,
    policy,
):

    env = LEOMobilityEnv(cfg)

    totals = defaultdict(float)

    total_steps = 0

    start_time = time.time()

    print()
    print("-" * 80)
    print(f"START {name}")
    print("-" * 80)

    for episode in range(EPISODES):

        env.reset(
            seed=EVALUATION_SEED_BASE
            + episode
        )

        done = False

        while not done:

            actions = policy(env)

            output = env.step(actions)

            for key, value in output.info.items():

                totals[key] += float(value)

            done = output.done

            total_steps += 1

        if (
            (episode + 1) % 20 == 0
            or episode == 0
            or episode + 1 == EPISODES
        ):

            elapsed = time.time() - start_time

            print(
                f"[{name}] "
                f"episode {episode + 1:3d}/{EPISODES} "
                f"({100.0 * (episode + 1) / EPISODES:5.1f}%) "
                f"elapsed={elapsed / 60.0:.1f} min",
                flush=True,
            )

    result = summarize_metrics(
        totals,
        total_steps,
    )

    result["common_reward"] = (
        common_reward(
            result["service_deficit"],
            result["resource_cost"],
        )
    )

    result["method"] = name

    print()
    print(f"{name} RESULT")

    for key in [
        "outage_ratio",
        "service_deficit",
        "resource_cost",
        "handover_started",
        "pingpong_rate",
        "duplication_active_time_fraction",
        "deadline_violation_rate",
        "common_reward",
    ]:

        if key in result:
            print(
                f"  {key:36s}"
                f" = {result[key]}"
            )

    return result


baseline_results = []

for name, policy in BASELINES.items():

    baseline_results.append(
        evaluate_heuristic(
            name,
            policy,
        )
    )


# ------------------------------------------------------------
# Save heuristic CSV + JSON
# ------------------------------------------------------------

baseline_csv = (
    OUTPUT
    / "heuristic_baselines_200ep.csv"
)

all_baseline_keys = sorted(
    set().union(
        *[
            result.keys()
            for result
            in baseline_results
        ]
    )
)

with baseline_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=all_baseline_keys,
    )

    writer.writeheader()
    writer.writerows(
        baseline_results
    )


baseline_json = (
    OUTPUT
    / "heuristic_baselines_200ep.json"
)

baseline_json.write_text(
    json.dumps(
        baseline_results,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PART C
# Combine heuristic baselines with M1-M5 stochastic results
# ============================================================

print()
print("=" * 80)
print("PART C: COMBINED MODEL + HEURISTIC TABLE")
print("=" * 80)

combined = []


for model, directory in MODEL_DIRS.items():

    path = (
        directory
        / "evaluation_stochastic.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"{model}: missing {path}"
        )

    result = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    row = {
        "method": model,
        **result,
    }

    row["common_reward"] = (
        common_reward(
            row["service_deficit"],
            row["resource_cost"],
        )
    )

    combined.append(row)


combined.extend(
    baseline_results
)


IMPORTANT_METRICS = [
    "method",
    "common_reward",
    "outage_ratio",
    "service_deficit",
    "resource_cost",
    "handover_started",
    "handover_failure_rate",
    "pingpong_rate",
    "duplication_active_time_fraction",
    "deadline_violation_rate",
    "preparation_rejection_rate",
    "prepare_to_commit_rate",
]


combined_csv = (
    OUTPUT
    / "main_model_baseline_comparison.csv"
)

with combined_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=IMPORTANT_METRICS,
    )

    writer.writeheader()

    for row in combined:

        writer.writerow({
            key: row.get(
                key,
                "",
            )
            for key in IMPORTANT_METRICS
        })


# ============================================================
# PART D
# KPI bar figures
# ============================================================

methods = [
    row["method"]
    for row in combined
]


def plot_metric(
    key,
    ylabel,
    filename,
):

    values = [
        float(row.get(key, np.nan))
        for row in combined
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.bar(
        methods,
        values,
    )

    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


plot_metric(
    "common_reward",
    "Common reward",
    "figure_common_reward_final.png",
)

plot_metric(
    "outage_ratio",
    "Outage ratio",
    "figure_all_methods_outage.png",
)

plot_metric(
    "service_deficit",
    "Service deficit",
    "figure_all_methods_service_deficit.png",
)

plot_metric(
    "resource_cost",
    "Resource cost",
    "figure_all_methods_resource.png",
)

plot_metric(
    "handover_started",
    "Handover started",
    "figure_all_methods_handover.png",
)

plot_metric(
    "pingpong_rate",
    "Ping-pong rate",
    "figure_all_methods_pingpong.png",
)

plot_metric(
    "duplication_active_time_fraction",
    "Duplication active time fraction",
    "figure_all_methods_duplication.png",
)


# ============================================================
# Final print
# ============================================================

print()
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

header = (
    f"{'Method':22s}"
    f"{'Outage':>12s}"
    f"{'Service':>12s}"
    f"{'Resource':>12s}"
    f"{'HO':>12s}"
    f"{'PingPong':>12s}"
    f"{'CommonR':>12s}"
)

print(header)
print("-" * len(header))

for row in combined:

    print(
        f"{row['method']:22s}"
        f"{float(row['outage_ratio']):12.5f}"
        f"{float(row['service_deficit']):12.5f}"
        f"{float(row['resource_cost']):12.5f}"
        f"{float(row['handover_started']):12.5f}"
        f"{float(row['pingpong_rate']):12.5f}"
        f"{float(row['common_reward']):12.5f}"
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print("Generated directory:")
print(OUTPUT)

print()
print("Important files:")

for filename in [
    "figure_common_reward_convergence.png",
    "common_reward_training_summary.csv",
    "heuristic_baselines_200ep.csv",
    "main_model_baseline_comparison.csv",
    "figure_common_reward_final.png",
    "figure_all_methods_outage.png",
    "figure_all_methods_service_deficit.png",
    "figure_all_methods_resource.png",
    "figure_all_methods_handover.png",
    "figure_all_methods_pingpong.png",
    "figure_all_methods_duplication.png",
]:

    print(
        "  ",
        OUTPUT / filename,
    )
