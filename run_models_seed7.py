from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

COMMON_OVERRIDES = [
    ROOT / "configs" / "validation_agentic_balanced_v2.yaml",
    ROOT / "configs" / "validation_reliability_stress_055.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
]

MODEL_OVERRIDES = {
    "M1": [],
    "M2": [
        ROOT / "configs" / "memory_only.yaml",
    ],
    "M3": [
        ROOT / "configs" / "memory_reasoning.yaml",
    ],
    "M4": [
        ROOT / "configs" / "m4_coordination.yaml",
    ],
    "M5": [
        ROOT / "configs" / "full_agentic.yaml",
    ],
}

MODEL_FINAL_OVERRIDES = {
    "M1": [],
    "M2": [],
    "M3": [],
    "M4": [],
    "M5": [
        ROOT / "configs" / "dual_mid_0005.yaml",
    ],
}


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remain = divmod(seconds, 3600)
    minutes, seconds = divmod(remain, 60)

    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def mean(rows, key):
    values = [
        float(row[key])
        for row in rows
        if key in row and row[key] not in ("", None)
    ]
    return sum(values) / len(values) if values else None


def slope_per_100(rows, key):
    values = [
        float(row[key])
        for row in rows
        if key in row and row[key] not in ("", None)
    ]

    n = len(values)
    if n < 2:
        return None

    xmean = (n - 1) / 2.0
    ymean = sum(values) / n

    numerator = sum(
        (i - xmean) * (y - ymean)
        for i, y in enumerate(values)
    )

    denominator = sum(
        (i - xmean) ** 2
        for i in range(n)
    )

    return 100.0 * numerator / denominator


def print_convergence(csv_path: Path):
    with csv_path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    print()
    print("========== CONVERGENCE SUMMARY ==========")

    if not rows:
        print("No training metrics available.")
        print("=========================================")
        print()
        return

    if "team_reward" in rows[0]:
        for window in (100, 200, 500):
            value = mean(rows[-window:], "team_reward") if len(rows) >= window else None
            rendered = f"{value:.6f}" if value is not None else "n/a"
            print(f"last_{window}_reward_mean={rendered}")
        slope = (
            slope_per_100(rows[-500:], "team_reward")
            if len(rows) >= 500
            else None
        )
        rendered_slope = f"{slope:.6f}" if slope is not None else "n/a"
        print(f"reward_slope_per_100_last_500={rendered_slope}")

    if len(rows) < 420:
        print("Not enough iterations for 210-vs-210 comparison.")
        print("=========================================")
        print()
        return

    prev210 = rows[-420:-210]
    last210 = rows[-210:]
    last105 = rows[-105:]

    keys = [
        "team_reward",
        "outage_ratio",
        "service_deficit",
        "resource_cost",
        "outage_dual_mu",
        "entropy",
        "approx_kl",
        "critic_loss",
    ]

    for key in keys:
        if key not in rows[0]:
            continue

        p = mean(prev210, key)
        c = mean(last210, key)
        h = mean(last105, key)
        s = slope_per_100(last210, key)

        print(
            f"{key:28s} "
            f"prev210={p: .6f} | "
            f"last210={c: .6f} | "
            f"last105={h: .6f} | "
            f"slope/100={s: .6f}"
        )

    print("=========================================")
    print()


def run_model(
    model: str,
    iterations: int,
    output_root: Path,
    seed_config: Path,
    evaluation_action_seed: int | None,
):
    output_dir = output_root / model.lower()
    checkpoint = output_dir / "mappo_weights.npz"
    csv_path = output_dir / "training_metrics.csv"

    if checkpoint.exists() and csv_path.exists():
        print(
            f"[{model}] existing completed run found -> skip training"
        )
        print_convergence(csv_path)
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Partial output already exists: {output_dir}\n"
            "Delete or rename that directory before rerunning."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        *MODEL_OVERRIDES[model],
        *COMMON_OVERRIDES,
        seed_config,
        *MODEL_FINAL_OVERRIDES[model],
    ]

    required = [
        DEFAULT_CONFIG,
        *overrides,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing config files:\n"
            + "\n".join(missing)
        )

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "train",
        "--config",
        str(DEFAULT_CONFIG),
    ]

    for override in overrides:
        cmd += [
            "--override",
            str(override),
        ]

    cmd += [
        "--iterations",
        str(iterations),
        "--eval-episodes",
        "20",
        "--stochastic-eval-episodes",
        "50",
        "--output",
        str(output_dir),
    ]
    if evaluation_action_seed is not None:
        cmd += ["--evaluation-action-seed", str(evaluation_action_seed)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)

    log_path = output_dir / "run.log"

    pattern = re.compile(r"iteration=(\d+)")

    milestones = [
        iterations * p // 10
        for p in range(1, 11)
    ]

    next_milestone = 0
    start = time.monotonic()

    print()
    print(
        f"========== {model} START =========="
    )
    print(
        f"seed=7 | iterations={iterations}"
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

        tail = []

        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()

            tail.append(line.rstrip())
            if len(tail) > 30:
                tail.pop(0)

            match = pattern.search(line)
            if match is None:
                continue

            iteration = int(match.group(1))

            while (
                next_milestone < len(milestones)
                and iteration >= milestones[next_milestone]
            ):
                reached = milestones[next_milestone]
                percent = (next_milestone + 1) * 10

                elapsed = time.monotonic() - start
                fraction = reached / iterations

                eta = (
                    elapsed
                    * (1.0 - fraction)
                    / max(fraction, 1e-12)
                )

                print(
                    f"[{model}] "
                    f"{percent:3d}% "
                    f"({reached}/{iterations}) | "
                    f"elapsed={fmt_time(elapsed)} | "
                    f"ETA={fmt_time(eta)}",
                    flush=True,
                )

                next_milestone += 1

        return_code = proc.wait()

    if return_code != 0:
        print("\n".join(tail))
        raise RuntimeError(
            f"{model} training failed."
        )

    elapsed = time.monotonic() - start

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not created: {checkpoint}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"training_metrics.csv not created: {csv_path}"
        )

    print(
        f"[{model}] DONE | total={fmt_time(elapsed)}"
    )

    print_convergence(csv_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["M1", "M2", "M3", "M4", "M5"],
        required=True,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--evaluation-action-seed",
        type=int,
        default=None,
        help="Optional shared stochastic-evaluation action seed across models",
    )

    args = parser.parse_args()

    output_root = (
        ROOT
        / "outputs"
        / f"seed7_compare_{args.iterations}"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Environment + agent seed를 모두 확실하게 7로 고정.
    seed_config = output_root / "seed7.yaml"
    seed_config.write_text(
        "seed: 7\n",
        encoding="utf-8",
    )

    print(
        f"Output root: {output_root}"
    )
    print(
        "Fixed global seed: 7"
    )

    for model in args.models:
        run_model(
            model,
            args.iterations,
            output_root,
            seed_config,
            args.evaluation_action_seed,
        )


if __name__ == "__main__":
    main()
