from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import evaluate_policy

ITERATIONS = 1000
EVAL_EPISODES = 200

# seed 7 = 기존에 이미 학습 완료한 checkpoint
ALL_SEEDS = [7, 8, 9]
NEW_TRAIN_SEEDS = [8, 9]

OUT_ROOT = ROOT / "outputs" / "multiseed_789"
SEED_CFG_DIR = OUT_ROOT / "seed_configs"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
SEED_CFG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG = ROOT / "configs" / "default.yaml"

COMMON_OVERRIDES = [
    ROOT / "configs" / "validation_agentic_balanced_v2.yaml",
    ROOT / "configs" / "validation_reliability_stress_055.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
]

MODEL_OVERRIDES = {
    "M4": [
        ROOT / "configs" / "m4_coordination.yaml",
    ],
    "M5": [
        ROOT / "configs" / "full_agentic.yaml",
        ROOT / "configs" / "dual_mid_0005.yaml",
    ],
}

# 기존 seed=7 결과
EXISTING_SEED7 = {
    "M4": ROOT / "outputs" / "m4_dense_risk_1000",
    "M5": ROOT / "outputs" / "m5_dense_risk_dual0005_1000",
}

METRIC_KEYS = [
    "outage_ratio",
    "service_deficit",
    "service_ratio_mean",
    "resource_cost",
    "dro",
    "handover_started",
    "pingpong_rate",
    "handover_failure_rate",
    "deadline_violation_rate",
    "action_stay_fraction",
    "action_prepare_fraction",
    "action_dup_on_fraction",
    "action_commit_fraction",
    "action_release_fraction",
    "outage_dual_mu",
]


def require_files():
    paths = [DEFAULT_CFG, *COMMON_OVERRIDES]

    for vals in MODEL_OVERRIDES.values():
        paths.extend(vals)

    missing = [str(p) for p in paths if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
        )


def seed_override(seed):
    path = SEED_CFG_DIR / f"seed_{seed}.yaml"

    path.write_text(
        f"seed: {seed}\n",
        encoding="utf-8",
    )

    return path


def config_overrides(model, seed):
    return [
        *MODEL_OVERRIDES[model],
        *COMMON_OVERRIDES,
        seed_override(seed),
    ]


def run_output_dir(model, seed):

    if seed == 7:
        return EXISTING_SEED7[model]

    return (
        OUT_ROOT
        / f"{model.lower()}_seed{seed}_1000"
    )


def fmt_seconds(seconds):

    seconds = max(0, int(round(seconds)))

    hour, remain = divmod(seconds, 3600)
    minute, second = divmod(remain, 60)

    if hour:
        return f"{hour}h {minute:02d}m"

    if minute:
        return f"{minute}m {second:02d}s"

    return f"{second}s"


def train(model, seed, out_dir):

    checkpoint = out_dir / "mappo_weights.npz"

    if checkpoint.exists():
        print(
            f"[{model} seed={seed}] "
            "checkpoint exists -> training skipped",
            flush=True,
        )
        return

    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"Partial output directory exists:\n"
            f"{out_dir}\n"
            "Delete or rename it before rerunning."
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "train",
        "--config",
        str(DEFAULT_CFG),
    ]

    for override in config_overrides(
        model,
        seed,
    ):
        cmd += [
            "--override",
            str(override),
        ]

    cmd += [
        "--iterations",
        str(ITERATIONS),

        "--eval-episodes",
        "20",

        "--stochastic-eval-episodes",
        "50",

        "--output",
        str(out_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)

    log_path = out_dir / "run.log"

    iteration_pattern = re.compile(
        r"iteration=(\d+)"
    )

    # 100, 200, ..., 1000
    milestones = [
        ITERATIONS * p // 10
        for p in range(1, 11)
    ]

    next_milestone = 0

    start = time.monotonic()

    tail = deque(maxlen=30)

    print()
    print(
        f"[{model} seed={seed}] "
        f"TRAIN START ({ITERATIONS} iterations)",
        flush=True,
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None

        for line in proc.stdout:

            # 전체 원본 로그는 파일에 저장
            log_file.write(line)
            log_file.flush()

            tail.append(
                line.rstrip()
            )

            match = iteration_pattern.search(
                line
            )

            if not match:
                continue

            iteration = int(
                match.group(1)
            )

            while (
                next_milestone
                < len(milestones)
                and iteration
                >= milestones[next_milestone]
            ):

                reached = milestones[
                    next_milestone
                ]

                percent = (
                    next_milestone + 1
                ) * 10

                elapsed = (
                    time.monotonic()
                    - start
                )

                fraction = (
                    reached
                    / ITERATIONS
                )

                eta = (
                    elapsed
                    * (1.0 - fraction)
                    / max(
                        fraction,
                        1e-12,
                    )
                )

                print(
                    f"[{model} seed={seed}] "
                    f"{percent:3d}% "
                    f"({reached}/{ITERATIONS}) | "
                    f"elapsed "
                    f"{fmt_seconds(elapsed)} | "
                    f"ETA "
                    f"{fmt_seconds(eta)}",
                    flush=True,
                )

                if percent == 100:
                    print(
                        f"[{model} seed={seed}] "
                        "training iterations complete; "
                        "evaluation/save finishing...",
                        flush=True,
                    )

                next_milestone += 1

        return_code = proc.wait()

    if return_code != 0:

        print(
            "\n".join(tail),
            file=sys.stderr,
        )

        raise RuntimeError(
            f"Training failed: "
            f"{model} seed={seed}"
        )

    elapsed = (
        time.monotonic()
        - start
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint missing:\n"
            f"{checkpoint}"
        )

    print(
        f"[{model} seed={seed}] "
        f"TRAIN DONE | "
        f"total {fmt_seconds(elapsed)}",
        flush=True,
    )


def evaluate_200(
    model,
    seed,
    out_dir,
):

    checkpoint = (
        out_dir
        / "mappo_weights.npz"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    overrides = [
        str(p)
        for p in config_overrides(
            model,
            seed,
        )
    ]

    cfg = load_config(
        str(DEFAULT_CFG),
        overrides,
    )

    env = LEOMobilityEnv(cfg)

    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        cfg["mappo"],
        int(cfg.get("agent_seed", cfg["seed"])),
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(
            env.reward_cfg[
                "eta_handover"
            ]
        ),
    )

    agent.load(
        str(checkpoint)
    )

    print(
        f"[{model} seed={seed}] "
        "200-episode stochastic "
        "evaluation START",
        flush=True,
    )

    result = evaluate_policy(
        env,
        agent,
        episodes=EVAL_EPISODES,
        deterministic=False,
    )

    metrics = {
        key: float(result[key])
        for key in METRIC_KEYS
        if key in result
    }

    metrics[
        "constraint_satisfied"
    ] = float(
        metrics["outage_ratio"]
        <= 0.10
    )

    eval_path = (
        OUT_ROOT
        / f"eval200_"
          f"{model.lower()}_"
          f"seed{seed}.json"
    )

    eval_path.write_text(
        json.dumps(
            {
                "model": model,
                "seed": seed,
                "episodes":
                    EVAL_EPISODES,
                "output_dir":
                    str(out_dir),
                "metrics":
                    metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[{model} seed={seed}] "
        f"EVAL DONE | "
        f"OR="
        f"{metrics['outage_ratio']:.6f} | "
        f"service="
        f"{metrics['service_ratio_mean']:.6f} | "
        f"control="
        f"{metrics['resource_cost']:.6f} | "
        f"constraint="
        f"{'PASS' if metrics['constraint_satisfied'] else 'FAIL'}",
        flush=True,
    )

    return metrics


def save_summary(rows):

    csv_path = (
        OUT_ROOT
        / "multiseed_summary.csv"
    )

    fieldnames = [
        "model",
        "seed",
        *METRIC_KEYS,
        "constraint_satisfied",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    json_path = (
        OUT_ROOT
        / "multiseed_summary.json"
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "========== "
        "MULTI-SEED SUMMARY "
        "=========="
    )

    for model in [
        "M4",
        "M5",
    ]:

        vals = [
            float(
                row["outage_ratio"]
            )
            for row in rows
            if row["model"]
            == model
        ]

        passes = sum(
            int(
                float(
                    row[
                        "constraint_satisfied"
                    ]
                )
            )
            for row in rows
            if row["model"]
            == model
        )

        mean = statistics.mean(
            vals
        )

        std = (
            statistics.stdev(vals)
            if len(vals) > 1
            else 0.0
        )

        print(
            f"{model}: "
            f"OR mean="
            f"{mean:.6f}, "
            f"std="
            f"{std:.6f}, "
            f"constraint_pass="
            f"{passes}/{len(vals)}"
        )

    print(
        f"CSV:  {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )


def main():

    require_files()

    # 기존 seed 7 checkpoint 확인
    for model in [
        "M4",
        "M5",
    ]:

        checkpoint = (
            EXISTING_SEED7[model]
            / "mappo_weights.npz"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                "Existing seed-7 "
                "checkpoint not found:\n"
                f"{checkpoint}"
            )

    # 새로 학습할 것은:
    # seed 8: M4 + M5
    # seed 9: M4 + M5
    total_new_runs = (
        len(NEW_TRAIN_SEEDS)
        * 2
    )

    completed = 0

    for seed in NEW_TRAIN_SEEDS:

        for model in [
            "M4",
            "M5",
        ]:

            train(
                model,
                seed,
                run_output_dir(
                    model,
                    seed,
                ),
            )

            completed += 1

            print(
                "[OVERALL NEW TRAINING] "
                f"{completed}/"
                f"{total_new_runs} "
                "complete "
                f"("
                f"{100.0 * completed / total_new_runs:.0f}"
                f"%)",
                flush=True,
            )

    # seed 7/8/9 모두
    # 동일하게 200 episode 평가
    rows = []

    for seed in ALL_SEEDS:

        for model in [
            "M4",
            "M5",
        ]:

            metrics = evaluate_200(
                model,
                seed,
                run_output_dir(
                    model,
                    seed,
                ),
            )

            rows.append(
                {
                    "model":
                        model,
                    "seed":
                        seed,
                    **metrics,
                }
            )

    save_summary(rows)


if __name__ == "__main__":
    main()
