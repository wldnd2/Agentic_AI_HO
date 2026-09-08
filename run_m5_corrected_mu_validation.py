from pathlib import Path
import csv
import json
import threading
import time

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import evaluate_policy


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

WEIGHTS = (
    ROOT
    / "outputs"
    / "v3_convergence_anneal_2000"
    / "m5_dual_only_seed7"
    / "mappo_weights.npz"
)

OUTDIR = WEIGHTS.parent

BASE_CONFIG = ROOT / "configs" / "default.yaml"

OVERRIDES = [
    ROOT / "configs" / "validation_agentic_challenge_v3.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
    ROOT / "configs" / "full_agentic.yaml",
    ROOT / "configs" / "dual_mid_0005.yaml",
]


# ============================================================
# Evaluation settings
# ============================================================

ENV_SEED = 7
AGENT_SEED = 7
ACTION_SEED = 7007

DET_EPISODES = 20
STOCH_EPISODES = 200

HEARTBEAT_SECONDS = 15


# ============================================================
# Helpers
# ============================================================

def make_env_agent():
    cfg = load_config(
        str(BASE_CONFIG),
        [str(p) for p in OVERRIDES],
    )

    cfg["environment_seed"] = ENV_SEED
    cfg["agent_seed"] = AGENT_SEED

    env = LEOMobilityEnv(cfg)

    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        cfg["mappo"],
        AGENT_SEED,
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(
            env.reward_cfg["eta_handover"]
        ),
    )

    agent.load(str(WEIGHTS))

    if not agent.use_constraint:
        raise RuntimeError(
            "ERROR: loaded checkpoint is not an M5 constrained policy."
        )

    return env, agent


def run_with_progress(label, func):
    """
    Run one evaluation while printing a heartbeat every
    HEARTBEAT_SECONDS seconds.
    """

    start = time.time()
    stop_event = threading.Event()

    print()
    print("=" * 90)
    print(f"[START] {label}")
    print("=" * 90)
    print(
        f"[PROGRESS] {label} started. "
        f"Heartbeat every {HEARTBEAT_SECONDS}s.",
        flush=True,
    )

    def heartbeat():
        while not stop_event.wait(HEARTBEAT_SECONDS):
            elapsed = time.time() - start

            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            print(
                f"[PROGRESS] {label} still running... "
                f"elapsed={minutes:02d}m {seconds:02d}s",
                flush=True,
            )

    thread = threading.Thread(
        target=heartbeat,
        daemon=True,
    )
    thread.start()

    try:
        result = func()
    finally:
        stop_event.set()
        thread.join(timeout=1.0)

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print(
        f"[DONE] {label} "
        f"elapsed={minutes:02d}m {seconds:05.2f}s",
        flush=True,
    )

    return result


def print_metrics(result):
    keys = [
        "outage_ratio",
        "service_deficit",
        "service_ratio_mean",
        "resource_cost",
        "dro",
        "handover_started",
        "handover_failure_rate",
        "pingpong_rate",
        "deadline_violation_rate",
        "preparation_rejection_rate",
        "prepare_to_release_without_commit_rate",
        "prepare_to_commit_rate",
        "duplication_active_time_fraction",
        "resource_price_mean",
        "reasoning_service_cost",
        "reasoning_coordination_cost",
        "reasoning_reliability_cost",
        "reasoning_selected_planning_risk",
        "reasoning_score_gap",
        "team_reward",
        "outage_dual_mu",
        "normalized_outage_dual_mu",
        "outage_constraint_gap",
        "constraint_satisfied",
    ]

    for key in keys:
        if key in result:
            print(
                f"{key:46s} = {result[key]}",
                flush=True,
            )


def compare(new, reference, title):
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)

    keys = [
        "outage_ratio",
        "service_deficit",
        "resource_cost",
        "dro",
        "handover_started",
        "pingpong_rate",
        "prepare_to_commit_rate",
        "duplication_active_time_fraction",
    ]

    for key in keys:
        if key not in new or key not in reference:
            continue

        new_value = float(new[key])
        ref_value = float(reference[key])

        diff = new_value - ref_value

        relative = (
            diff
            / max(abs(ref_value), 1e-12)
            * 100.0
        )

        print(
            f"{key:38s} "
            f"new={new_value:.9f} "
            f"reference={ref_value:.9f} "
            f"delta={diff:+.9f} "
            f"rel={relative:+.3f}%"
        )


# ============================================================
# Main
# ============================================================

overall_start = time.time()

print()
print("=" * 90)
print("CORRECTED M5 MU VALIDATION")
print("=" * 90)

print(f"ROOT       = {ROOT}")
print(f"WEIGHTS    = {WEIGHTS}")
print(f"OUTDIR     = {OUTDIR}")

print()
print("CONFIG OVERRIDES:")

for p in OVERRIDES:
    print(f"  - {p.name}")

print()
print(f"ENV_SEED          = {ENV_SEED}")
print(f"AGENT_SEED        = {AGENT_SEED}")
print(f"ACTION_SEED       = {ACTION_SEED}")
print(f"DET_EPISODES      = {DET_EPISODES}")
print(f"STOCH_EPISODES    = {STOCH_EPISODES}")
print(f"HEARTBEAT_SECONDS = {HEARTBEAT_SECONDS}")


# ============================================================
# Check checkpoint
# ============================================================

if not WEIGHTS.exists():
    raise SystemExit(
        f"ERROR: checkpoint does not exist:\n{WEIGHTS}"
    )

raw = np.load(WEIGHTS)

if "outage_dual_mu" not in raw.files:
    raise SystemExit(
        "ERROR: outage_dual_mu is missing from checkpoint."
    )

SAVED_MU = float(raw["outage_dual_mu"])

print()
print("=" * 90)
print("CHECKPOINT")
print("=" * 90)
print(f"SAVED RAW MU        = {SAVED_MU:.12f}")
print(f"SAVED NORMALIZED MU = {SAVED_MU / 10.0:.12f}")


# ============================================================
# Evaluation jobs
# ============================================================

MU_VALUES = [
    0.0,
    0.10,
    0.20,
    0.30,
    SAVED_MU,
    0.45,
]

TOTAL_JOBS = 1 + len(MU_VALUES)

print()
print("=" * 90)
print("EVALUATION PLAN")
print("=" * 90)

print(
    f"Total evaluation jobs = {TOTAL_JOBS}"
)

print(
    f"  Job 1: deterministic reproduction "
    f"({DET_EPISODES} episodes)"
)

for i, mu in enumerate(MU_VALUES, start=2):
    print(
        f"  Job {i}: stochastic mu={mu:.12f} "
        f"({STOCH_EPISODES} episodes)"
    )


# ============================================================
# Job 1: deterministic reproduction
# ============================================================

def deterministic_job():
    env, agent = make_env_agent()

    agent.outage_dual_mu = SAVED_MU

    return evaluate_policy(
        env,
        agent,
        episodes=DET_EPISODES,
        deterministic=True,
        evaluation_action_seed=ACTION_SEED,
    )


det_result = run_with_progress(
    f"[1/{TOTAL_JOBS}] deterministic "
    f"saved_mu={SAVED_MU:.6f}",
    deterministic_job,
)

print()
print("DETERMINISTIC RESULT")
print_metrics(det_result)


reference_det_path = (
    OUTDIR / "evaluation.json"
)

reference_det = None

if reference_det_path.exists():
    reference_det = json.loads(
        reference_det_path.read_text(
            encoding="utf-8"
        )
    )

    compare(
        det_result,
        reference_det,
        "NEW DETERMINISTIC vs EXISTING evaluation.json",
    )
else:
    print()
    print(
        "[WARNING] Existing evaluation.json not found."
    )


# ============================================================
# Jobs 2-7: mu sensitivity
# ============================================================

results = []

for job_number, mu in enumerate(
    MU_VALUES,
    start=2,
):
    def stochastic_job(mu_value=mu):
        env, agent = make_env_agent()

        agent.outage_dual_mu = float(mu_value)

        return evaluate_policy(
            env,
            agent,
            episodes=STOCH_EPISODES,
            deterministic=False,
            evaluation_action_seed=ACTION_SEED,
        )

    result = run_with_progress(
        f"[{job_number}/{TOTAL_JOBS}] "
        f"stochastic mu={mu:.6f}",
        stochastic_job,
    )

    print()
    print(
        f"RESULT FOR FIXED MU = {mu:.12f}"
    )
    print_metrics(result)

    row = {
        "mu": float(mu),
    }

    for key, value in result.items():
        try:
            row[key] = float(value)
        except (TypeError, ValueError):
            pass

    results.append(row)


# ============================================================
# Saved-mu stochastic reproduction check
# ============================================================

saved_mu_row = min(
    results,
    key=lambda row: abs(
        row["mu"] - SAVED_MU
    ),
)

reference_stoch_path = (
    OUTDIR / "evaluation_stochastic.json"
)

reference_stoch = None

if reference_stoch_path.exists():
    reference_stoch = json.loads(
        reference_stoch_path.read_text(
            encoding="utf-8"
        )
    )

    compare(
        saved_mu_row,
        reference_stoch,
        "CORRECTED SAVED-MU vs EXISTING evaluation_stochastic.json",
    )
else:
    print()
    print(
        "[WARNING] Existing evaluation_stochastic.json not found."
    )


# ============================================================
# Delta from saved mu
# ============================================================

print()
print("=" * 90)
print("DELTA RELATIVE TO TRAINED/SAVED MU")
print("=" * 90)

delta_keys = [
    "outage_ratio",
    "service_deficit",
    "resource_cost",
    "dro",
    "handover_started",
    "pingpong_rate",
    "prepare_to_commit_rate",
    "duplication_active_time_fraction",
]

for row in results:
    print()
    print(
        f"MU = {row['mu']:.12f}"
    )

    for key in delta_keys:
        if (
            key not in row
            or key not in saved_mu_row
        ):
            continue

        diff = (
            row[key]
            - saved_mu_row[key]
        )

        relative = (
            diff
            / max(
                abs(saved_mu_row[key]),
                1e-12,
            )
            * 100.0
        )

        print(
            f"  {key:38s} "
            f"delta={diff:+.9f} "
            f"rel={relative:+.3f}%"
        )


# ============================================================
# Compact summary
# ============================================================

print()
print("=" * 90)
print("COMPACT MU SENSITIVITY SUMMARY")
print("=" * 90)

summary_keys = [
    "mu",
    "outage_ratio",
    "service_deficit",
    "resource_cost",
    "dro",
    "handover_started",
    "pingpong_rate",
    "prepare_to_commit_rate",
    "duplication_active_time_fraction",
    "reasoning_selected_planning_risk",
    "team_reward",
]

print(",".join(summary_keys))

for row in results:
    values = []

    for key in summary_keys:
        value = row.get(
            key,
            float("nan"),
        )

        values.append(
            f"{value:.12f}"
        )

    print(
        ",".join(values)
    )


# ============================================================
# Save CSV
# ============================================================

csv_path = (
    ROOT
    / "m5_corrected_mu_sensitivity.csv"
)

all_keys = sorted(
    set().union(
        *(row.keys() for row in results)
    )
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=all_keys,
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# Save JSON
# ============================================================

json_path = (
    ROOT
    / "m5_corrected_mu_sensitivity.json"
)

payload = {
    "checkpoint": str(WEIGHTS),
    "saved_mu": SAVED_MU,
    "environment_seed": ENV_SEED,
    "agent_seed": AGENT_SEED,
    "action_seed": ACTION_SEED,
    "deterministic_episodes": DET_EPISODES,
    "stochastic_episodes": STOCH_EPISODES,
    "config_overrides": [
        str(p)
        for p in OVERRIDES
    ],
    "deterministic_reproduction": det_result,
    "existing_deterministic_reference": reference_det,
    "existing_stochastic_reference": reference_stoch,
    "mu_sensitivity": results,
}

json_path.write_text(
    json.dumps(
        payload,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# Finish
# ============================================================

overall_elapsed = (
    time.time()
    - overall_start
)

overall_minutes = int(
    overall_elapsed // 60
)

overall_seconds = (
    overall_elapsed % 60
)

print()
print("=" * 90)
print("ALL EVALUATIONS COMPLETE")
print("=" * 90)

print(
    f"Total elapsed = "
    f"{overall_minutes:02d}m "
    f"{overall_seconds:05.2f}s"
)

print()
print("Generated files:")
print(f"  {csv_path}")
print(f"  {json_path}")

print()
print(
    "Next: upload "
    "m5_corrected_mu_sensitivity.json "
    "and m5_corrected_mu_validation.txt"
)
