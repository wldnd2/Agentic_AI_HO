"""Train and tune the MADQN baseline (parameter-shared Double DQN + fingerprints).

Same environment, observation, action mask, team reward r and training budget
(2000 iterations x 128 epochs) as the MAPPO-based variants M1-M5.

  tune   train one policy per learning rate on seed 7 (shorter budget) and pick
         the rate with the best common reward on the CHO validation seeds 20-22
         (episodes disjoint from the held-out evaluation streams)
  train  train the final policies for the given seeds (learning rate from the
         tuning result unless --lr is given)

Usage (project root):
  PYTHONPATH=src python train_madqn.py tune
  PYTHONPATH=src python train_madqn.py train --seeds 7 8 9 10 11
Finished runs are skipped, so re-running resumes.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
CFG = ROOT / "configs"
BASE = CFG / "default.yaml"
COMMON = [CFG / "validation_agentic_challenge_v3.yaml", CFG / "eta_handover_005.yaml",
          CFG / "validation_dwell_15.yaml"]
FINAL_ROOT = ROOT / "outputs" / "v3_anneal_2000_multiseed"
TUNE_ROOT = ROOT / "outputs" / "madqn_tuning"
STREAM_LENGTH = 200
W_CTRL = 0.25
DEFAULT_CFG = {
    "hidden_dims": [128, 128], "gamma": 0.99, "learning_rate": 3e-4,
    "batch_size": 256, "updates_per_iteration": 32, "target_update_interval": 250,
    "warmup_transitions": 10_000, "buffer_capacity": 200_000,
    "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_fraction": 0.4,
    "max_grad_norm": 10.0, "huber_delta": 1.0,
}


def make_cfg(seed: int, condition: str = "nominal"):
    from leo_mappo import load_config
    paths = list(COMMON)
    if condition == "stress":
        paths.append(CFG / "compound_eval.yaml")
    cfg = load_config(str(BASE), [str(p) for p in paths])
    cfg["environment_seed"] = int(seed)
    cfg["agent_seed"] = int(seed)
    return cfg


def train_one(seed: int, iterations: int, lr: float, out_dir: str, rollout: int = 128) -> dict:
    from leo_mappo import LEOMobilityEnv
    from leo_mappo.madqn import MADQN
    from leo_mappo.runner import summarize_metrics
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "madqn_weights.npz").is_file():
        return {"seed": seed, "lr": lr, "out": str(out), "skipped": True}
    t0 = time.monotonic()
    cfg = make_cfg(seed)
    env = LEOMobilityEnv(cfg)
    agent = MADQN(env.obs_dim, env.action_dim, {**DEFAULT_CFG, "learning_rate": lr}, seed)
    obs, _, masks = env.reset()
    history = []
    log = (out / "train.log").open("w", encoding="utf-8")
    for it in range(iterations):
        eps, cur_lr = agent.set_iteration(it, iterations)
        metrics: dict[str, float] = defaultdict(float)
        for _ in range(rollout):
            actions = agent.act(obs, masks, eps)
            step = env.step(actions)
            reward = float(step.rewards.mean())
            agent.store(obs, actions, reward, step.observations, step.action_masks, step.done)
            for k, v in step.info.items():
                metrics[k] += float(v)
            obs, masks = step.observations, step.action_masks
            if step.done:
                obs, _, masks = env.reset()
        losses = agent.update()
        row = {"iteration": it + 1, **summarize_metrics(metrics, rollout), **losses,
               "epsilon": eps, "learning_rate": cur_lr}
        history.append(row)
        if (it + 1) % 10 == 0 or it == 0:
            log.write(f"iteration={it + 1:04d} reward={row.get('team_reward', 0.0):.4f} "
                      f"outage={row.get('outage_ratio', 0.0):.4f} eps={eps:.3f} "
                      f"td={losses['td_loss']:.5f}\n")
            log.flush()
        if (it + 1) % 100 == 0:
            (out / "checkpoints").mkdir(exist_ok=True)
            agent.save(out / "checkpoints" / f"madqn_weights_iter_{it + 1:04d}.npz")
    agent.set_iteration(iterations, iterations)      # final fingerprint (1, eps_end)
    agent.save(out / "madqn_weights.npz")
    fields = sorted({k for r in history for k in r})
    with (out / "training_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(history)
    (out / "madqn_config.json").write_text(json.dumps({**DEFAULT_CFG, "learning_rate": lr,
                                                       "iterations": iterations, "seed": seed},
                                                      indent=1), encoding="utf-8")
    log.close()
    return {"seed": seed, "lr": lr, "out": str(out), "minutes": (time.monotonic() - t0) / 60}


def evaluate_greedy(weights: str, train_seed: int, condition: str, eval_seed: int,
                    episodes: int) -> dict:
    """Greedy evaluation on the episode stream of `eval_seed` (same stream layout
    as eval_unified.py / cho_sweep.py)."""
    from leo_mappo import LEOMobilityEnv
    from leo_mappo.madqn import MADQN
    from leo_mappo.runner import summarize_metrics
    cfg = make_cfg(eval_seed, condition)
    cfg["agent_seed"] = int(train_seed)
    env = LEOMobilityEnv(cfg)
    agent = MADQN.from_checkpoint(weights, train_seed)
    rng = np.random.default_rng(7000 + int(eval_seed))
    offset = (int(eval_seed) - 7) * STREAM_LENGTH
    totals: dict[str, float] = defaultdict(float)
    steps = 0
    for ep in range(episodes):
        obs, _, masks = env.reset(seed=10_000 + ep + offset)
        done = False
        while not done:
            step = env.step(agent.greedy(obs, masks, rng))
            for k, v in step.info.items():
                totals[k] += float(v)
            obs, masks, done = step.observations, step.action_masks, step.done
            steps += 1
    return summarize_metrics(totals, steps)


def _val_job(args):
    lr, weights, seed, val_seed, episodes = args
    m = evaluate_greedy(weights, seed, "nominal", val_seed, episodes)
    return lr, val_seed, -m["service_deficit"] - W_CTRL * m["resource_cost"], m["outage_ratio"]


def tune(args) -> None:
    TUNE_ROOT.mkdir(parents=True, exist_ok=True)
    result_file = TUNE_ROOT / "tuning.json"
    if result_file.is_file():
        print(f"[skip] tuning result exists: {json.loads(result_file.read_text())['best_lr']}")
        return
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.lrs))) as pool:
        futs = [pool.submit(train_one, args.seed, args.iterations, lr,
                            str(TUNE_ROOT / f"lr_{lr:g}")) for lr in args.lrs]
        for f in as_completed(futs):
            r = f.result()
            print(f"[done] tuning train lr={r['lr']:g} " + ("(skipped)" if r.get("skipped")
                  else f"in {r['minutes']:.1f} min"), flush=True)
    jobs = [(lr, str(TUNE_ROOT / f"lr_{lr:g}" / "madqn_weights.npz"), args.seed, vs, args.episodes)
            for lr in args.lrs for vs in args.val_seeds]
    scores = defaultdict(list)
    outages = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for f in as_completed([pool.submit(_val_job, j) for j in jobs]):
            lr, vs, cr, out = f.result()
            scores[lr].append(cr)
            outages[lr].append(out)
            print(f"[done] validation lr={lr:g} seed={vs} common_reward={cr:.5f} outage={out:.4f}",
                  flush=True)
    table = [{"lr": lr, "common_reward": float(np.mean(scores[lr])),
              "outage_ratio": float(np.mean(outages[lr]))} for lr in args.lrs]
    best = max(table, key=lambda r: r["common_reward"])
    result_file.write_text(json.dumps({"best_lr": best["lr"], "results": table,
                                       "tuning_seed": args.seed, "iterations": args.iterations,
                                       "validation_seeds": args.val_seeds,
                                       "episodes_per_seed": args.episodes}, indent=1),
                           encoding="utf-8")
    print(f"[done] tuning: best learning rate {best['lr']:g} -> {result_file}", flush=True)


def train(args) -> None:
    lr = args.lr
    if lr is None:
        tf = TUNE_ROOT / "tuning.json"
        lr = json.loads(tf.read_text())["best_lr"] if tf.is_file() else DEFAULT_CFG["learning_rate"]
    print(f"MADQN final training: lr={lr:g}, seeds={args.seeds}, iterations={args.iterations}")
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.seeds))) as pool:
        futs = [pool.submit(train_one, s, args.iterations, lr, str(FINAL_ROOT / f"madqn_seed{s}"))
                for s in args.seeds]
        for f in as_completed(futs):
            r = f.result()
            print(f"[done] MADQN seed={r['seed']} " + ("(skipped)" if r.get("skipped")
                  else f"in {r['minutes']:.1f} min"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--iterations", type=int, default=800)
    t.add_argument("--val-seeds", nargs="+", type=int, default=[20, 21, 22])
    t.add_argument("--episodes", type=int, default=20)
    t.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r = sub.add_parser("train")
    r.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9, 10, 11])
    r.add_argument("--iterations", type=int, default=2000)
    r.add_argument("--lr", type=float, default=None)
    r.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    tune(args) if args.mode == "tune" else train(args)


if __name__ == "__main__":
    main()
