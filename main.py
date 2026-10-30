from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.runner import RolloutRunner, evaluate_policy


def _rolling_mean(values: list[float], window: int = 50) -> list[float]:
    """Return a trailing mean with min_periods=1 semantics."""
    if window <= 0:
        raise ValueError("rolling-mean window must be positive")
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += float(value)
        if index >= window:
            running_sum -= float(values[index - window])
        result.append(running_sum / min(index + 1, window))
    return result


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.override)
    if args.environment_seed is not None:
        config["environment_seed"] = args.environment_seed
    if args.agent_seed is not None:
        config["agent_seed"] = args.agent_seed
    if args.iterations is not None:
        config["mappo"]["training_iterations"] = args.iterations
    env = LEOMobilityEnv(config)
    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        config["mappo"],
        int(config.get("agent_seed", config["seed"])),
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )
    runner = RolloutRunner(env, agent)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = int(config["mappo"].get("checkpoint_interval", 100))
    if checkpoint_interval <= 0:
        raise ValueError("mappo.checkpoint_interval must be positive")
    total_iterations = int(config["mappo"]["training_iterations"])
    history = []
    for iteration in range(total_iterations):
        actor_lr, critic_lr = agent.set_training_iteration(iteration, total_iterations)
        batch, rollout_metrics = runner.collect(int(config["mappo"]["rollout_steps"]))
        losses = agent.update(batch)
        row = {
            "iteration": iteration + 1,
            **rollout_metrics,
            **losses,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
        }
        history.append(row)
        runner.update_constraint()
        print(
            f"iteration={iteration + 1:04d} reward={rollout_metrics.get('team_reward', 0.0):8.3f} "
            f"miss={rollout_metrics.get('deadline_misses', 0.0):6.3f} "
            f"prep_cost={rollout_metrics.get('preparation_train_cost', 0.0):6.3f} "
            f"actor_loss={losses['actor_loss']:7.4f}"
        )
        completed_iteration = iteration + 1
        if completed_iteration % checkpoint_interval == 0:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = (
                checkpoint_dir / f"mappo_weights_iter_{completed_iteration:04d}.npz"
            )
            agent.save(checkpoint_path)
            smoothed_reward = _rolling_mean(
                [float(item.get("team_reward", 0.0)) for item in history], 50
            )[-1]
            print(
                f"checkpoint={checkpoint_path} "
                f"smoothed_reward={smoothed_reward:.6f} "
                f"outage_ratio={rollout_metrics.get('outage_ratio', 0.0):.6f}"
            )
    final_checkpoint = output_dir / "mappo_weights_final.npz"
    agent.save(final_checkpoint)
    print(f"checkpoint={final_checkpoint} final=true")
    legacy_checkpoint = output_dir / "mappo_weights.npz"
    agent.save(legacy_checkpoint)
    print(f"checkpoint={legacy_checkpoint} final=true legacy=true")
    fieldnames = sorted({key for row in history for key in row})
    with (output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    evaluation = evaluate_policy(
        env,
        agent,
        episodes=args.eval_episodes,
        evaluation_action_seed=args.evaluation_action_seed,
    )
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2)
    with (output_dir / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(evaluation))
        writer.writeheader()
        writer.writerow(evaluation)
    if args.stochastic_eval_episodes > 0:
        stochastic_evaluation = evaluate_policy(
            env,
            agent,
            episodes=args.stochastic_eval_episodes,
            deterministic=False,
            evaluation_action_seed=args.evaluation_action_seed,
        )
        with (output_dir / "evaluation_stochastic.json").open("w", encoding="utf-8") as handle:
            json.dump(stochastic_evaluation, handle, indent=2)
        with (output_dir / "evaluation_stochastic_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(stochastic_evaluation))
            writer.writeheader()
            writer.writerow(stochastic_evaluation)
    _save_training_plots(history, output_dir)
    print(json.dumps(evaluation, indent=2))


def _save_training_plots(history: list[dict[str, float]], output_dir: Path) -> None:
    """Write plots after durable artifacts; plotting must not endanger CSV/weights."""
    if not history:
        return
    try:
        x = [row["iteration"] for row in history]
        def series(name: str) -> list[float]:
            return [float(row.get(name, 0.0)) for row in history]

        reward = series("team_reward")
        reward_window = 50
        smoothed_reward = _rolling_mean(reward, reward_window)
        fig, axis = plt.subplots()
        axis.plot(x, reward, alpha=0.25, linewidth=1.0, label="raw reward")
        axis.plot(
            x,
            smoothed_reward,
            linewidth=2.5,
            label=f"smoothed reward (window={reward_window})",
        )
        axis.set(xlabel="iteration", ylabel="reward")
        axis.legend()
        fig.tight_layout(); fig.savefig(output_dir / "reward.png"); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        for axis, name in zip(axes.flat, ("actor_loss", "critic_loss", "entropy", "approx_kl")):
            axis.plot(x, series(name)); axis.set_title(name); axis.set_xlabel("iteration")
        fig.tight_layout(); fig.savefig(output_dir / "losses.png"); plt.close(fig)

        fig, axis = plt.subplots()
        axis.plot(x, series("service_deficit"), label="service_deficit")
        axis.plot(x, series("outage_ratio"), label="outage_ratio")
        axis.plot(x, series("strict_shortfall_ratio"), label="strict_shortfall_ratio")
        axis.set(xlabel="iteration", ylabel="ratio"); axis.legend()
        fig.tight_layout(); fig.savefig(output_dir / "core_kpi.png"); plt.close(fig)

        action_names = ("stay", "prepare", "commit", "release")
        fig, axis = plt.subplots(figsize=(9, 5))
        for name in action_names:
            axis.plot(x, series(f"action_{name}_fraction"), label=name.upper())
        axis.set(xlabel="iteration", ylabel="action fraction"); axis.legend(ncol=2)
        fig.tight_layout(); fig.savefig(output_dir / "action_distribution.png"); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        diagnostics = (
            "hidden_state_norm",
            "reasoning_score_gap",
            "reasoning_prediction_error",
            "selected_action_reasoning_rank",
        )
        for axis, name in zip(axes.flat, diagnostics):
            axis.plot(x, series(name)); axis.set_title(name); axis.set_xlabel("iteration")
        fig.tight_layout(); fig.savefig(output_dir / "memory_reasoning.png"); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(x, series("resource_price_mean"), label="resource_price_mean")
        axes[0].plot(x, series("resource_price_max"), label="resource_price_max")
        axes[0].legend(); axes[0].set_ylabel("price")
        axes[1].plot(x, series("requested_pressure_mean"), label="requested_pressure_mean")
        axes[1].plot(x, series("requested_pressure_max"), label="requested_pressure_max")
        axes[1].legend(); axes[1].set(xlabel="iteration", ylabel="requested pressure")
        fig.tight_layout(); fig.savefig(output_dir / "coordination.png"); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(x, series("outage_ratio"), label="outage_ratio")
        axes[0].plot(x, series("outage_target"), label="outage_target")
        axes[0].legend(); axes[0].set_ylabel("outage")
        axes[1].plot(x, series("outage_dual_mu"), label="outage_dual_mu")
        axes[1].plot(x, series("outage_constraint_gap"), label="outage_constraint_gap")
        axes[1].legend(); axes[1].set(xlabel="iteration", ylabel="dual / gap")
        fig.tight_layout(); fig.savefig(output_dir / "constraint.png"); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        mobility_metrics = (
            "handover_started",
            "pingpong_rate",
            "preparation_rejection_rate",
            "handover_control_cost",
        )
        for axis, name in zip(axes.flat, mobility_metrics):
            axis.plot(x, series(name)); axis.set_title(name); axis.set_xlabel("iteration")
        fig.tight_layout(); fig.savefig(output_dir / "mobility_control.png"); plt.close(fig)
    except Exception as error:
        print(f"warning: training plots were not generated: {error}")


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.override)
    if args.environment_seed is not None:
        config["environment_seed"] = args.environment_seed
    if args.agent_seed is not None:
        config["agent_seed"] = args.agent_seed
    env = LEOMobilityEnv(config)
    agent = MAPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        config["mappo"],
        int(config.get("agent_seed", config["seed"])),
        num_satellites=env.num_cells,
        candidate_count=env.candidate_count,
        eta_handover=float(env.reward_cfg["eta_handover"]),
    )
    agent.load(args.weights)
    print(
        json.dumps(
            evaluate_policy(
                env,
                agent,
                episodes=args.eval_episodes,
                evaluation_action_seed=args.evaluation_action_seed,
            ),
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LEO CTDE-MAPPO reference simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default="configs/default.yaml")
        sub.add_argument(
            "--override",
            action="append",
            default=None,
            help="Optional scenario YAML; repeat to deep-merge files in command-line order",
        )
        sub.add_argument("--eval-episodes", type=int, default=3)
        sub.add_argument("--environment-seed", type=int, default=None)
        sub.add_argument("--agent-seed", type=int, default=None)
        sub.add_argument("--evaluation-action-seed", type=int, default=None)
    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--iterations", type=int, default=None)
    train_parser.add_argument("--output", default="outputs/default_run")
    train_parser.add_argument("--stochastic-eval-episodes", type=int, default=0)
    subparsers.choices["evaluate"].add_argument("--weights", required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        train(arguments)
    else:
        evaluate(arguments)
