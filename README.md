# CTDE-MAPPO for PDCP-Duplication-Assisted LEO Mobility

The opt-in partially observable Agentic Challenge V3 scenario is documented in
[`V3_DESIGN.md`](V3_DESIGN.md) and configured by
`configs/validation_agentic_challenge_v3.yaml`. Existing default/V2 scenarios
remain the strongly observed compatibility benchmark.

This project is an executable research scaffold for the following problem:

> In a shared-capacity multi-user LEO NTN, jointly control target preparation,
> PDCP duplication activation/deactivation, and primary-cell switching while
> reducing deadline violations, duplicated radio resources, and unnecessary
> handovers.

The implementation is intentionally dependency-light. MAPPO, the shared actor,
the centralized critic, GAE, PPO clipping, and Adam are implemented in NumPy so
the full smoke test can run without PyTorch.

## Model scope

- Downlink, multiple UE agents, multiple LEO NTN cells.
- One primary leg and at most one prepared secondary leg per UE.
- Geometry-based local multi-plane constellation slice, satellite visibility,
  and remaining-visibility prediction.
- Link budget with free-space loss, elevation loss, shadowing, interference,
  smooth CQI/BLER abstraction, and correlated shadowing.
- Packet arrivals, FIFO queues, deadlines, drops, and QoS priorities.
- A fixed weighted-fair scheduler creates shared-capacity coupling.
- PDCP duplication sends the same payload over both legs. Goodput is not summed;
  delivery succeeds when at least one copy succeeds.
- Protocol states: `Single -> Prepared -> Duplicating -> Switching -> Cooldown`.
- State-aware action masks and a feasibility layer prevent invalid transitions.
- Composable traffic-burst, capacity-drop, and blockage events are internal to
  the simulator and are not exposed as Normal/Congestion/Recovery labels.

This is a research abstraction, not a standards-complete 3GPP protocol stack.
The model assumes that a dual-leg, PDCP-duplication-capable deployment is
available. RLC/PHY retransmission details, packet-level path scheduling, new
reassembly logic, uplink power control, and RB-level learning are out of scope.

## Project layout

```text
configs/default.yaml             Base Core-MAPPO experiment
configs/compound_eval.yaml       Overlapping events for generalization tests
main.py                          Train/evaluate entry point
baseline_eval.py                 Heuristic baseline entry point
src/leo_mappo/satellite.py       Orbit, ECEF geometry, visibility
src/leo_mappo/channel.py         Link budget and packet success abstraction
src/leo_mappo/ue.py              UE traffic, queue, QoS, mobility state
src/leo_mappo/pdcp.py            Duplication reliability model
src/leo_mappo/scheduler.py       Fixed QoS-aware shared-resource scheduler
src/leo_mappo/events.py          Composable hidden event generator
src/leo_mappo/environment.py     Multi-agent environment and reward
src/leo_mappo/networks.py        NumPy MLP and Adam
src/leo_mappo/mappo.py           Shared actor + centralized critic MAPPO
src/leo_mappo/runner.py          Rollout, GAE, training and evaluation
src/leo_mappo/baselines.py       Max-SINR and threshold-duplication policies
tests/                            Environment and learner smoke tests
```

## Run

From the project directory:

```bash
export PYTHONPATH=src
python main.py train --config configs/default.yaml --output outputs/core_run
python main.py evaluate \
  --config configs/default.yaml \
  --weights outputs/core_run/mappo_weights.npz
```

Short smoke training:

```bash
PYTHONPATH=src python main.py train --iterations 2 --eval-episodes 1 --output outputs/smoke
```

Heuristic baseline:

```bash
PYTHONPATH=src python baseline_eval.py --policy threshold_dup --episodes 3
```

Compound-event evaluation uses the same environment and observation/action
interface:

```bash
PYTHONPATH=src python main.py evaluate \
  --config configs/default.yaml \
  --override configs/compound_eval.yaml \
  --weights outputs/core_run/mappo_weights.npz
```

Run tests without pytest:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

If `pytest` is available, use `PYTHONPATH=src pytest -q`.

## Observation and action

Each shared UE actor sees only local information: queue and deadline state,
primary-link quality, top-K candidate quality and selected-target flag, remaining visibility, broadcast
cell pressure, current protocol state, duplication state, and protocol timer.
The centralized critic sees all UE observations plus all cell utilization and
pressure values during training only.

For `K` candidates, actions are:

```text
Stay, Prepare-1..K, Dup-ON, Dup-OFF, Commit, Release, Rollback
```

## Reward interpretation

The formal research objective should be written as constrained optimization:
minimize mobility-induced extra resource cost subject to deadline reliability,
rate, visibility, capacity, and valid protocol-transition constraints. The RL
implementation uses a normalized team penalty for deadline misses, queue
overflow, duplicated resources, target preparation, handover failure,
ping-pong, cell overload, and throughput deficit. Reward weights are in YAML
and should be reported with sensitivity analysis in a paper.

## Recommended next refinements

1. Replace the smooth BLER abstraction with the exact selected NR MCS/CQI table.
2. Calibrate orbit, beam footprint, antenna pattern, traffic, and timers against
   the final 3GPP deployment assumptions.
3. Add explicit packet sub-slots if deadlines shorter than the mobility epoch
   are required.
4. Add telemetry delay/noise and event-correlation models.
5. Compare Max-SINR, Always-Dup, threshold duplication, load-aware heuristic,
   independent PPO, and CTDE-MAPPO across multiple seeds.
6. Only after Core-MAPPO is stable, compare rule-based selection with an
   Agentic diagnose-plan-monitor-reflect loop on unseen compound events.
