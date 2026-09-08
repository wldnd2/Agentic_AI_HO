from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

BASE = ROOT / "configs" / "default.yaml"

COMMON = [
    ROOT / "configs" / "validation_agentic_challenge_v3.yaml",
    ROOT / "configs" / "eta_handover_005.yaml",
    ROOT / "configs" / "validation_dwell_15.yaml",
    ROOT / "configs" / "lr_anneal_linear.yaml",
]

MODEL_CONFIG = {
    "M2": [
        ROOT / "configs" / "memory_only.yaml",
    ],
    "M4": [
        ROOT / "configs" / "m4_coordination.yaml",
    ],
    "M5": [
        ROOT / "configs" / "full_agentic.yaml",
        ROOT / "configs" / "dual_mid_0005.yaml",
    ],
}


def completed(outdir: Path) -> bool:
    required = [
        outdir / "mappo_weights.npz",
        outdir / "training_metrics.csv",
        outdir / "evaluation_stochastic.json",
    ]
    return all(p.exists() for p in required)


def check_partial(outdir: Path):
    if not outdir.exists():
        return

    if completed(outdir):
        return

    contents = list(outdir.iterdir())

    if contents:
        raise RuntimeError(
            f"\nPartial run already exists:\n"
            f"  {outdir}\n\n"
            f"Rename/delete it before rerunning."
        )


def make_ue_config(config_dir: Path, ue: int) -> Path:
    path = config_dir / f"ue_{ue}.yaml"

    path.write_text(
        "environment:\n"
        f"  num_ues: {ue}\n"
        "\n"
        "mappo:\n"
        "  checkpoint_interval: 500\n",
        encoding="utf-8",
    )

    return path


def run_one(
    ue: int,
    model: str,
    iterations: int,
    eval_episodes: int,
    stochastic_eval_episodes: int,
    output_root: Path,
    config_dir: Path,
):
    outdir = output_root / f"ue_{ue}" / model.lower()

    if completed(outdir):
        print()
        print("=" * 78)
        print(f"SKIP: UE={ue}, {model} already completed")
        print(f"      {outdir}")
        print("=" * 78)
        return

    check_partial(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    ue_config = make_ue_config(config_dir, ue)

    # IMPORTANT:
    # challenge_v3 sets num_ues=12.
    # Therefore UE config is deliberately applied LAST
    # so that num_ues becomes 10/20/30/40/50.
    overrides = [
        *COMMON,
        *MODEL_CONFIG[model],
        ue_config,
    ]

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "train",
        "--config",
        str(BASE),
    ]

    for override in overrides:
        cmd += [
            "--override",
            str(override),
        ]

    cmd += [
        "--environment-seed",
        "7",
        "--agent-seed",
        "7",
        "--evaluation-action-seed",
        "7007",
        "--iterations",
        str(iterations),
        "--eval-episodes",
        str(eval_episodes),
        "--stochastic-eval-episodes",
        str(stochastic_eval_episodes),
        "--output",
        str(outdir),
    ]

    command_string = " ".join(str(x) for x in cmd)

    (outdir / "run_command.txt").write_text(
        command_string + "\n",
        encoding="utf-8",
    )

    print()
    print("#" * 78)
    print(
        f"START | UE={ue:2d} | {model} | "
        f"iterations={iterations}"
    )
    print(f"OUTPUT: {outdir}")
    print("#" * 78)
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)

    log_path = outdir / "run.log"

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

        last_lines = []

        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()

            last_lines.append(line.rstrip())

            if len(last_lines) > 50:
                last_lines.pop(0)

            # Print only useful progress lines to terminal.
            if (
                line.startswith("iteration=")
                or line.startswith("checkpoint=")
            ):
                if line.startswith("iteration="):
                    try:
                        number = int(
                            line.split()[0]
                            .split("=")[1]
                        )
                    except Exception:
                        number = -1

                    if (
                        number == 1
                        or number % 100 == 0
                        or number == iterations
                    ):
                        print(
                            f"[UE={ue:2d} {model}] "
                            + line.rstrip(),
                            flush=True,
                        )
                else:
                    print(
                        f"[UE={ue:2d} {model}] "
                        + line.rstrip(),
                        flush=True,
                    )

        return_code = proc.wait()

    if return_code != 0:
        print()
        print("LAST LOG LINES:")
        print("\n".join(last_lines))

        raise RuntimeError(
            f"FAILED: UE={ue}, model={model}"
        )

    if not completed(outdir):
        raise RuntimeError(
            f"Run terminated but required output is missing: "
            f"{outdir}"
        )

    # main.py writes two identical final checkpoints.
    legacy = outdir / "mappo_weights.npz"
    final = outdir / "mappo_weights_final.npz"

    if legacy.exists() and final.exists():
        same = subprocess.run(
            ["cmp", "-s", str(legacy), str(final)]
        ).returncode == 0

        if same:
            final.unlink()
            print(
                f"[cleanup] removed duplicate "
                f"{final.name}"
            )

    print()
    print(
        f"DONE | UE={ue:2d} | {model}"
    )
    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ues",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["M2", "M4", "M5"],
        default=["M2", "M4", "M5"],
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--stochastic-eval-episodes",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--output-root",
        default="outputs/paper_ue_scalability_seed7",
    )

    args = parser.parse_args()

    output_root = ROOT / args.output_root

    config_dir = (
        output_root
        / "_configs"
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print("UE SCALABILITY EXPERIMENT")
    print("=" * 78)

    print(f"UEs       : {args.ues}")
    print(f"Models    : {args.models}")
    print(f"Iterations: {args.iterations}")
    print("Seed      : 7")
    print("Action seed: 7007")
    print(
        f"Stochastic evaluation: "
        f"{args.stochastic_eval_episodes} episodes"
    )
    print(f"Output    : {output_root}")

    print()
    print("Fixed:")
    print("  satellites = 6")
    print("  candidates = Top-3")
    print("  V3 challenge environment")
    print("  traffic = 2 Mbps / UE")
    print("  LR annealing = linear")
    print()

    total = (
        len(args.ues)
        * len(args.models)
    )

    current = 0

    for ue in args.ues:
        for model in args.models:

            current += 1

            print()
            print(
                f">>> RUN {current}/{total}"
            )

            run_one(
                ue=ue,
                model=model,
                iterations=args.iterations,
                eval_episodes=args.eval_episodes,
                stochastic_eval_episodes=(
                    args.stochastic_eval_episodes
                ),
                output_root=output_root,
                config_dir=config_dir,
            )

    print()
    print("=" * 78)
    print("ALL UE SCALABILITY RUNS COMPLETED")
    print("=" * 78)
    print(output_root)


if __name__ == "__main__":
    main()
