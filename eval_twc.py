"""Evaluate trained runs and rule baselines on common TWC episodes.

Examples
  # rule baselines on the validation seeds (nominal)
  PYTHONPATH=src python eval_twc.py --methods a3 t1 d2 maxsinr --seeds 20 21 22 --episodes 4
  # trained runs (directories written by train_twc.py), both conditions
  PYTHONPATH=src python eval_twc.py --runs outputs/twc/proposed_seed7 outputs/twc/madqn_seed7 \
      --methods a3 t1 --seeds 20 21 22 --episodes 4 --conditions nominal stress

Rows are written to outputs/twc/eval/<tag>.jsonl and a summary table is printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from cho_baselines import A3ConditionalHandover, CriticalDuplicationA3, LocationBasedCHO, TimeBasedCHO  # noqa: E402
from leo_mappo import LEOMobilityEnv, MAPPO, load_config  # noqa: E402
from leo_mappo.egcmadqn import Components  # noqa: E402
from leo_mappo.twc_wrappers import ComponentEnvWrapper  # noqa: E402
from leo_mappo.baselines import max_sinr_policy  # noqa: E402
from leo_mappo.egcmadqn import EGCExecutor, EGCMADQN  # noqa: E402
from train_twc import CFG, SCENARIOS, TWC_STACK  # noqa: E402

# Rule baselines tuned on validation seeds 20-22 (outputs/twc/cho_sweep_*.csv).
RULES = {
    "a3": lambda: A3ConditionalHandover(prep_offset_db=1.0, exec_offset_db=1.0, ttt_steps=1),
    "t1": lambda: TimeBasedCHO(service_elevation_deg=75.0, lead_s=15.0, target_rule="geometry"),
    "d2": lambda: LocationBasedCHO(thresh1_km=400.0, thresh2_km=1500.0, target_rule="load"),
    "maxsinr": lambda: max_sinr_policy,
    # PDCP-duplication rules (mixed-criticality scenario).
    "a3critdup": lambda: CriticalDuplicationA3(prep_offset_db=1.0, exec_offset_db=1.0, ttt_steps=1),
    "a3dup": lambda: A3ConditionalHandover(prep_offset_db=1.0, exec_offset_db=1.0, ttt_steps=1, dup_threshold_db=100.0),
}
# Compound-stress disturbances are active in epochs [60, 130) (twc_compound_eval.yaml).
STRESS_WINDOW = (60, 130)
WINDOW_KEYS = ("outage_ratio", "service_deficit", "signaling_cost", "handover_started")
SUM_KEYS = (
    "outage_ratio", "service_deficit", "signaling_cost", "resource_cost", "handover_started",
    "handover_failure", "handover_success", "pingpong", "deadline_misses", "arrivals",
    "duplication_active_time_fraction", "preparation_requested", "preparation_rejected",
)
# Per-class metrics (only present in the mixed-criticality scenario).
CLASS_KEYS = (
    "critical_outage_ratio", "noncritical_outage_ratio", "critical_duplication_fraction",
    "noncritical_duplication_fraction", "critical_deficit", "noncritical_deficit",
    "critical_deadline_misses", "critical_arrivals", "noncritical_deadline_misses", "noncritical_arrivals",
)


TRAIN_UES, TRAIN_RATE = 24, 1.2


def build_env(condition: str, seed: int, ues: int | None = None, rate: float | None = None,
              scenario: str = "default") -> LEOMobilityEnv:
    stack = [p for p in TWC_STACK if p.exists()] + SCENARIOS[scenario]
    if condition == "stress":
        stack.append(CFG / "twc_compound_eval.yaml")
    cfg = load_config(CFG / "default.yaml", stack)
    cfg["environment_seed"] = seed
    if ues is not None:
        cfg["environment"]["num_ues"] = int(ues)
    if rate is not None:
        cfg["traffic"]["mean_arrival_mbps"] = float(rate)
    return LEOMobilityEnv(cfg)


class MAPPOExecutor:
    """Memoryless MAPPO execution with the price input (C4) if it was trained with it."""

    def __init__(self, env, run_dir: Path, run_config: dict, seed: int, deterministic: bool):
        self.wrapper = ComponentEnvWrapper(env, Components(price_input=run_config["components"]["price_input"],
                                                           guided_exploration=False, potential_shaping=False,
                                                           dual_constraints=False))
        self.coordinator = self.wrapper.coordinator
        # The critic input depends on the training population; only the actor is used here,
        # so the networks are built with the training dimensions and run on any population.
        pop = run_config.get("population", {"num_ues": TRAIN_UES, "rate_mbps": TRAIN_RATE})
        ref = env if env.num_agents == pop["num_ues"] else build_env(
            "nominal", seed, pop["num_ues"], pop["rate_mbps"], run_config.get("scenario", "default"))
        ref_wrapper = ComponentEnvWrapper(ref, self.wrapper.c)
        self.agent = MAPPO(ref_wrapper.obs_dim, ref.global_state_dim, ref.action_dim, ref.config["mappo"], seed,
                           num_satellites=ref.num_cells, candidate_count=ref.candidate_count,
                           eta_handover=float(ref.reward_cfg["eta_handover"]))
        self.agent.load(run_dir / "mappo_weights.npz")
        self.deterministic = deterministic

    def act(self, observations, masks):
        return self.agent.act(self.wrapper._augment(observations), masks, deterministic=self.deterministic)[0]

    def observe(self, step) -> None:
        if self.coordinator is not None:
            self.coordinator.update(step.requested_pressure)


def run_job(job: tuple) -> dict:
    method, run_dir, condition, seed, episodes, mappo_deterministic = job[:6]
    ues, rate = (job[6], job[7]) if len(job) > 6 else (None, None)
    scenario = job[8] if len(job) > 8 else "default"
    env = build_env(condition, seed, ues, rate, scenario)
    steps = int(env.env_cfg["episode_steps"])
    executor = None
    if run_dir is not None:
        run_config = json.loads((Path(run_dir) / "run_config.json").read_text())
        if run_config.get("learner") == "mappo":
            executor = MAPPOExecutor(env, Path(run_dir), run_config, seed, mappo_deterministic)
        else:
            agent = EGCMADQN.from_checkpoint(Path(run_dir) / "egcmadqn_weights.npz", seed=seed)
            executor = EGCExecutor(env, agent, run_config["components"]["price_input"], np.random.default_rng(seed))
    else:
        policy = RULES[method]()
    sums = {k: 0.0 for k in SUM_KEYS}
    class_sums: dict[str, float] = {}
    window = {k: 0.0 for k in WINDOW_KEYS}
    window_steps = 0
    reward = 0.0
    for episode in range(episodes):
        obs, _, masks = env.reset(seed=10_000 * seed + episode)
        if executor is not None and executor.coordinator is not None:
            executor.coordinator.reset()
        for _ in range(steps):
            actions = executor.act(obs, masks) if executor is not None else policy(env)
            out = env.step(actions)
            if executor is not None:
                executor.observe(out)
            obs, masks = out.observations, out.action_masks
            reward += float(np.mean(out.rewards))
            for k in SUM_KEYS:
                sums[k] += float(out.info.get(k, 0.0))
            for k in CLASS_KEYS:
                if k in out.info:
                    class_sums[k] = class_sums.get(k, 0.0) + float(out.info[k])
            if STRESS_WINDOW[0] <= env.step_index - 1 < STRESS_WINDOW[1]:
                window_steps += 1
                for k in WINDOW_KEYS:
                    window[k] += float(out.info.get(k, 0.0))
    n = episodes * steps
    return {
        "method": method, "run": None if run_dir is None else str(run_dir), "condition": condition, "seed": seed,
        "num_ues": env.num_agents, "rate_mbps": float(env.config["traffic"]["mean_arrival_mbps"]),
        "common_reward": reward / n, "outage": sums["outage_ratio"] / n, "deficit": sums["service_deficit"] / n,
        "signaling": sums["signaling_cost"] / n, "resource": sums["resource_cost"] / n,
        "ho_per_epoch": sums["handover_started"] / n, "ho_failure_per_epoch": sums["handover_failure"] / n,
        "pingpong_ratio": sums["pingpong"] / max(sums["handover_success"], 1.0),
        "deadline_violation": sums["deadline_misses"] / max(sums["arrivals"], 1.0),
        "duplication_fraction": sums["duplication_active_time_fraction"] / n,
        "prep_rejection": sums["preparation_rejected"] / max(sums["preparation_requested"], 1.0),
        # Same epochs in both conditions, so nominal vs stress is a paired comparison.
        "window_outage": window["outage_ratio"] / max(window_steps, 1),
        "window_deficit": window["service_deficit"] / max(window_steps, 1),
        "window_signaling": window["signaling_cost"] / max(window_steps, 1),
        "window_ho_per_epoch": window["handover_started"] / max(window_steps, 1),
        "scenario": scenario,
        **({
            "critical_outage": class_sums["critical_outage_ratio"] / n,
            "noncritical_outage": class_sums["noncritical_outage_ratio"] / n,
            "critical_dup": class_sums["critical_duplication_fraction"] / n,
            "noncritical_dup": class_sums["noncritical_duplication_fraction"] / n,
            "critical_deficit": class_sums["critical_deficit"] / n,
            "critical_deadline_violation": class_sums["critical_deadline_misses"] / max(class_sums["critical_arrivals"], 1.0),
            "noncritical_deadline_violation": (class_sums["noncritical_deadline_misses"]
                                               / max(class_sums["noncritical_arrivals"], 1.0)),
        } if class_sums else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=[], choices=sorted(RULES))
    parser.add_argument("--runs", nargs="*", default=[], help="train_twc.py output directories")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20, 21, 22])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--conditions", nargs="+", default=["nominal"], choices=["nominal", "stress"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--ues", nargs="+", type=int, default=[None],
                        help="UE populations to evaluate (default: the training population, 24)")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="default")
    parser.add_argument("--load", choices=["per-ue", "total"], default="per-ue",
                        help="per-ue: keep 1.2 Mb/s per UE; total: keep the training total load (24 x 1.2 Mb/s)")
    parser.add_argument("--mappo-deterministic", action="store_true",
                        help="argmax MAPPO actions (default: sampled, as in the JSAC evaluation)")
    args = parser.parse_args()

    det = args.mappo_deterministic
    pops = []
    for n in args.ues:
        rate = None if n is None or args.load == "per-ue" else TRAIN_UES * TRAIN_RATE / n
        pops.append((n, rate))
    sc = args.scenario
    jobs = [(m, None, c, s, args.episodes, det, n, r, sc) for m in args.methods for c in args.conditions
            for s in args.seeds for n, r in pops]
    jobs += [(Path(x).name, x, c, s, args.episodes, det, n, r, sc) for x in args.runs for c in args.conditions
             for s in args.seeds for n, r in pops]
    # Resumable: every finished job is appended to <tag>.partial.jsonl at once, and
    # a re-run with the same arguments skips the jobs that are already there.
    out = ROOT / "outputs" / "twc" / "eval" / f"{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.jsonl")

    def key(job) -> tuple:
        method, run_dir, condition, seed, episodes, deterministic, n, rate, scenario = job
        return (method, None if run_dir is None else str(Path(run_dir)), condition, int(seed), int(episodes),
                bool(deterministic), n, None if rate is None else round(float(rate), 6), scenario)

    done: dict[tuple, dict] = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut by the interruption
            done[tuple(record["_job"])] = record
    keys = [key(job) for job in jobs]
    todo = [k for k, job_key in enumerate(keys) if job_key not in done]
    if done:
        print(f"resuming: {len(jobs) - len(todo)} of {len(jobs)} jobs already finished", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, partial.open("a", encoding="utf-8") as handle:
        futures = {pool.submit(run_job, jobs[k]): k for k in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            k = futures[future]
            record = dict(future.result())
            record["_job"] = list(keys[k])
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            done[keys[k]] = record
            job = jobs[k]
            print(f"  [{len(jobs) - len(todo) + count}/{len(jobs)}] {job[0]} {job[2]} seed {job[3]}"
                  + (f" UEs {job[6]}" if job[6] else ""), flush=True)
    rows = [{k: v for k, v in done[job_key].items() if k != "_job"} for job_key in keys]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    keys = ("common_reward", "outage", "deficit", "signaling", "ho_per_epoch", "pingpong_ratio",
            "prep_rejection", "window_outage")
    if args.scenario == "mixed":
        keys = ("common_reward", "signaling", "critical_outage", "noncritical_outage", "critical_dup",
                "noncritical_dup", "critical_deadline_violation", "noncritical_deadline_violation")
    print(f"{'condition':9s} {'UEs':>4s} {'method':28s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
    for condition in args.conditions:
        for n in dict.fromkeys(r["num_ues"] for r in rows):
            for name in dict.fromkeys(r["method"] for r in rows):
                sel = [r for r in rows if r["method"] == name and r["condition"] == condition and r["num_ues"] == n]
                if sel:
                    print(f"{condition:9s} {n:4d} {name:28s} " + " ".join(f"{np.mean([r[k] for r in sel]):12.4f}" for k in keys))
    print(f"rows -> {out}")


if __name__ == "__main__":
    main()
