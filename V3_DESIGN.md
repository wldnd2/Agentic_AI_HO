# Agentic Challenge V3 design

V3 is opt-in through `environment.challenge_version: v3`. Missing flags and
`v2` retain the original strongly observed benchmark exactly.

## Information boundaries

V2 actor observations expose exact remaining visibility, requested pressure,
and effective capacity, while candidate ordering combines SINR, remaining
visibility, and pressure. Its event parameters are fixed by the scenario.

V3 keeps observation/network dimensions unchanged but redefines those slots.
Actors receive delayed, temporally correlated noisy SINR/efficiency and local
load telemetry, current elevation, and an observed SINR trend. They do not
receive exact remaining visibility, capacity multiplier, aggregate demand,
event labels/severity/remaining duration, or hidden link-regime labels.
Candidate discovery ranks currently visible cells only by observed current
SINR. The centralized critic may consume true shared state during training;
execution actors do not.

The V3 hard action mask enforces state-machine and current physical validity.
It deliberately does not decide whether a candidate will remain visible long
enough. The analytical M3 reasoner retains deterministic ephemeris lookahead,
but receives no future randomized event or latent-degradation oracle.

## Temporal uncertainty and reliability diversity

Each episode samples event start, duration, severity, satellite, and applicable
UE subset from configured ranges using the environment RNG. Every UE-satellite
path also has a hidden `GOOD -> DEGRADED -> RECOVERING -> GOOD` regime. These
periods persist for multiple steps and independently alter true link loss.
Rewards, scheduling, delivery, and outage use true physical links; only
imperfect telemetry reaches actors.

PDCP duplication reuses the two-leg scheduler and joint-delivery model. Path
diversity can rescue a degraded primary, while both legs consume shared data
resources and incur the existing duplication cost. No event-specific action
rule is used.

## Contention and coordination

V3 adds a control-plane maximum concurrent preparation count per target
satellite. Admission follows simultaneous requests and current occupancy; the
existing fractional data-capacity reservation remains separate.

The environment records serving, preparation, and duplication intended demand
per satellite. In V3 their configured weighted sum feeds the existing M4 price
coordinator after a step. The next decentralized decision therefore sees a
one-step-lagged satellite price, whereas M1-M3 see imperfect local load only.
The coordinator never allocates actions centrally.

## Incremental roles

- M1: current/local mobility and resource control.
- M2: recurrent inference of persistent hidden telemetry/link regimes.
- M3: protocol-aware analytical future geometry reasoning.
- M4: aggregate multi-UE resource-demand pricing.
- M5: unchanged realized-outage constraint and dual controller.

V3 does not modify PPO, GAE, GRU/TBPTT, dense-risk temperature, outage
semantics, the additive M5 reward, or model-specific reward coefficients.

## Seeds and evaluation

`environment_seed` and `agent_seed` independently fall back to legacy `seed`.
`evaluate_policy(..., evaluation_action_seed=N)` temporarily uses a separate
reproducible stochastic-action generator and restores the original training
RNG. Passing `None` preserves legacy behavior.

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_v3_environment.py --episodes 3
```
