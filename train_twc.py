"""Train EG-CMADQN (and ablations / plain MADQN) in the TWC environment.

Examples
  PYTHONPATH=src python train_twc.py --variant proposed --seed 7
  PYTHONPATH=src python train_twc.py --variant madqn --seed 7 --iterations 2000
  PYTHONPATH=src python train_twc.py --variant proposed --seed 7 --iterations 3 --smoke

Variants: proposed (C1-C4), no_c1, no_c2, no_c3, no_c4, madqn (none),
          mappo (memoryless CTDE-MAPPO = JSAC M1), mappo_c234 (MAPPO + C2-C4 via
          ComponentEnvWrapper; C1 needs an off-policy learner).
Outputs (outputs/twc/<variant>_seed<seed>/):
  training_metrics.csv, egcmadqn_weights.npz or mappo_weights.npz, duals.json, run_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.egcmadqn import VARIANTS, Components, EGCTrainer, TrainerConfig
from leo_mappo.runner import RolloutRunner
from leo_mappo.twc_wrappers import ComponentEnvWrapper

MAPPO_VARIANTS = {
    "mappo": Components(False, False, False, False),
    "mappo_c234": Components(guided_exploration=False, potential_shaping=True, dual_constraints=True, price_input=True),
}

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs"
TWC_STACK = [
    CFG / "validation_agentic_challenge_v3.yaml",
    CFG / "eta_handover_005.yaml",
    CFG / "validation_dwell_15.yaml",
    CFG / "lr_anneal_linear.yaml",
    CFG / "twc_walker_p3.yaml",
    CFG / "twc_3gpp_set1.yaml",
    CFG / "twc_load_24ue.yaml", CFG / "twc_events.yaml",
]
# Same agent hyper-parameters as the JSAC MADQN baseline (learning rate 1e-3
# selected on validation seeds).
AGENT_CFG = {
    "hidden_dims": [128, 128], "gamma": 0.99, "learning_rate": 1e-3, "final_learning_rate": 1e-4,
    "batch_size": 256, "updates_per_iteration": 32, "target_update_interval": 250,
    "warmup_transitions": 10_000, "epsilon_start": 1.0, "epsilon_end": 0.05,
    "epsilon_decay_fraction": 0.4, "max_grad_norm": 10.0, "huber_delta": 1.0, "buffer_capacity": 200_000,
}


def build_env(seed: int, smoke: bool) -> LEOMobilityEnv:
    cfg = load_config(CFG / "default.yaml", [p for p in TWC_STACK if p.exists()])
    cfg["environment_seed"] = seed
    if smoke:
        cfg["environment"]["episode_steps"] = 40
    return LEOMobilityEnv(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS) + sorted(MAPPO_VARIANTS), default="proposed")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--kappa", type=float, default=TrainerConfig.shaping_scale)
    parser.add_argument("--temperature", type=float, default=TrainerConfig.guidance_temperature)
    parser.add_argument("--outage-target", type=float, default=TrainerConfig.outage_target)
    parser.add_argument("--signaling-target", type=float, default=TrainerConfig.signaling_target)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--smoke", action="store_true", help="short episodes and a small rollout")
    args = parser.parse_args()

    is_mappo = args.variant in MAPPO_VARIANTS
    components = MAPPO_VARIANTS[args.variant] if is_mappo else VARIANTS[args.variant]
    tc = TrainerConfig(
        rollout_steps=16 if args.smoke else args.rollout,
        shaping_scale=args.kappa,
        guidance_temperature=args.temperature,
        outage_target=args.outage_target,
        signaling_target=args.signaling_target,
    )
    agent_cfg = dict(AGENT_CFG)
    if args.smoke:
        agent_cfg.update(warmup_transitions=200, batch_size=64, updates_per_iteration=2)
    env = build_env(args.seed, args.smoke)
    if is_mappo:
        mappo_cfg = dict(env.config["mappo"])
        wrapper = ComponentEnvWrapper(env, components, tc, gamma=float(mappo_cfg.get("gamma", 0.99)))
        agent = MAPPO(wrapper.obs_dim, env.global_state_dim, env.action_dim, mappo_cfg, args.seed,
                      num_satellites=env.num_cells, candidate_count=env.candidate_count,
                      eta_handover=float(env.reward_cfg["eta_handover"]))
        runner = RolloutRunner(wrapper, agent)
        agent_cfg = mappo_cfg
    else:
        trainer = EGCTrainer(env, components, agent_cfg, args.seed, tc)

    out = Path(args.outdir) if args.outdir else ROOT / "outputs" / "twc" / f"{args.variant}_seed{args.seed}"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({
        "variant": args.variant, "learner": "mappo" if is_mappo else "egcmadqn",
        "components": asdict(components), "seed": args.seed,
        "iterations": args.iterations, "trainer": asdict(tc), "agent": agent_cfg,
        "config_stack": [p.name for p in TWC_STACK if p.exists()],
    }, indent=2))

    fields = None
    t_start = time.time()
    with (out / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        for it in range(args.iterations):
            t0 = time.time()
            if is_mappo:
                agent.set_training_iteration(it, args.iterations)
                batch, _ = runner.collect(tc.rollout_steps)
                losses = agent.update(batch)
                metrics = {"iteration": it, **wrapper.end_iteration(), "epsilon": 0.0, "guidance": 0.0,
                           "actor_loss": losses.get("actor_loss", 0.0), "critic_loss": losses.get("critic_loss", 0.0),
                           "entropy": losses.get("entropy", 0.0)}
            else:
                metrics = {"iteration": it, **trainer.run_iteration(it, args.iterations)}
            metrics["seconds"] = time.time() - t0
            if fields is None:
                fields = list(metrics) + [k for k in ("lambda_out", "lambda_sig") if k not in metrics]
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
            writer.writerow(metrics)
            handle.flush()
            if (it + 1) % args.checkpoint_every == 0:
                if is_mappo:
                    agent.save(out / "checkpoints" / f"mappo_weights_iter_{it + 1:04d}.npz")
                else:
                    trainer.agent.save(out / "checkpoints" / f"egcmadqn_weights_iter_{it + 1:04d}.npz")
            if it % 50 == 0 or it == args.iterations - 1:
                print(f"[{args.variant} s{args.seed}] it {it:4d} common {metrics['common_reward']:.4f} "
                      f"outage {metrics['outage']:.3f} signal {metrics['signaling']:.3f} "
                      f"eps {metrics['epsilon']:.2f} beta {metrics['guidance']:.2f} "
                      f"{metrics['seconds']:.1f}s", flush=True)
    if is_mappo:
        agent.save(out / "mappo_weights.npz")
        duals = {"lambda_out": wrapper.dual_out.value, "lambda_sig": wrapper.dual_sig.value}
    else:
        trainer.agent.save(out / "egcmadqn_weights.npz")
        duals = {"lambda_out": trainer.dual_out.value, "lambda_sig": trainer.dual_sig.value,
                 "fingerprint": trainer.agent.fingerprint.tolist()}
    (out / "duals.json").write_text(json.dumps(duals, indent=2))
    print(f"done in {(time.time() - t_start) / 3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()
