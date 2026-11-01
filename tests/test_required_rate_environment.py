from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.demand import ARRIVAL_CAPPED, DEADLINE_AWARE
from leo_mappo.ue import PacketBatch


ROOT = Path(__file__).resolve().parents[1]


def _v3_config(required_rate_model: str | None) -> dict:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["num_ues"] = 1
    config["traffic"]["arrival_spread"] = 0.0
    config["events"] = []
    if required_rate_model is None:
        config["traffic"].pop("required_rate_model", None)
    else:
        config["traffic"]["required_rate_model"] = required_rate_model
    return config


def _environment(required_rate_model: str | None) -> LEOMobilityEnv:
    return LEOMobilityEnv(_v3_config(required_rate_model))


def _set_queue(
    env: LEOMobilityEnv,
    *,
    packets: int,
    age_s: float,
    mean_arrival_mbps: float = 1.0,
    deadline_s: float = 2.0,
) -> None:
    ue = env.ues[0]
    ue.queue.clear()
    if packets:
        ue.queue.append(PacketBatch(packets, age_s=age_s))
    ue.mean_arrival_mbps = mean_arrival_mbps
    ue.deadline_s = deadline_s


def test_omitted_required_rate_model_is_exact_explicit_legacy_behavior() -> None:
    omitted = _environment(None)
    explicit = _environment(ARRIVAL_CAPPED)
    for env in (omitted, explicit):
        _set_queue(env, packets=137, age_s=1.25, mean_arrival_mbps=1.3)
        env._active_effects.traffic_multiplier[:] = 1.7

    omitted_active, omitted_rates = omitted.current_required_rates()
    explicit_active, explicit_rates = explicit.current_required_rates()

    assert omitted.required_rate_model == ARRIVAL_CAPPED
    np.testing.assert_array_equal(omitted_active, explicit_active)
    np.testing.assert_array_equal(omitted_rates, explicit_rates)


def test_deadline_aware_demand_increases_only_under_backlog_pressure() -> None:
    env = _environment(DEADLINE_AWARE)
    _set_queue(env, packets=100, age_s=0.0)
    relaxed = env.observed_required_rates()[1][0]
    env.ues[0].queue[0].age_s = env.ues[0].deadline_s
    pressured = env.observed_required_rates()[1][0]

    assert relaxed == 1e6
    assert pressured > relaxed
    assert pressured <= 100 * env.ues[0].packet_size_bits / env.interval_s


def test_current_required_rates_use_physical_traffic_multiplier() -> None:
    env = _environment(DEADLINE_AWARE)
    _set_queue(
        env,
        packets=400,
        age_s=0.0,
        mean_arrival_mbps=1.0,
        deadline_s=100.0,
    )
    env._active_effects.traffic_multiplier[:] = 1.0
    baseline = env.current_required_rates()[1][0]
    env._active_effects.traffic_multiplier[:] = 3.0
    multiplied = env.current_required_rates()[1][0]

    assert baseline == 1e6
    assert multiplied == 3e6


def test_v3_observed_demand_does_not_read_hidden_traffic_multiplier() -> None:
    env = _environment(DEADLINE_AWARE)
    _set_queue(
        env,
        packets=400,
        age_s=0.0,
        mean_arrival_mbps=1.0,
        deadline_s=100.0,
    )
    env._active_effects.traffic_multiplier[:] = 1.0
    observed_before = env.observed_required_rates()[1].copy()
    physical_before = env.current_required_rates()[1].copy()
    env._active_effects.traffic_multiplier[:] = 4.0
    observed_after = env.observed_required_rates()[1]
    physical_after = env.current_required_rates()[1]

    np.testing.assert_array_equal(observed_before, observed_after)
    assert not np.array_equal(physical_before, physical_after)


def test_observed_deadline_aware_demand_uses_queue_and_hol_state() -> None:
    env = _environment(DEADLINE_AWARE)
    _set_queue(env, packets=100, age_s=0.0)
    relaxed = env.observed_required_rates()[1][0]
    env.ues[0].queue[0].age_s = 2.0
    urgent = env.observed_required_rates()[1][0]
    env.ues[0].queue[0].packets = 200
    larger_urgent_queue = env.observed_required_rates()[1][0]

    assert urgent > relaxed
    assert larger_urgent_queue > urgent


def test_required_rate_queries_do_not_mutate_environment_or_rng() -> None:
    env = _environment(DEADLINE_AWARE)
    _set_queue(env, packets=100, age_s=1.25)
    before = {
        "queues": tuple(
            tuple((batch.packets, batch.age_s) for batch in ue.queue)
            for ue in env.ues
        ),
        "protocol": tuple(
            (
                ue.primary_cell,
                ue.secondary_cell,
                ue.state,
                ue.protocol_timer,
            )
            for ue in env.ues
        ),
        "rng": deepcopy(env.rng.bit_generator.state),
    }

    env.current_required_rates()
    env.observed_required_rates()

    after_queues = tuple(
        tuple((batch.packets, batch.age_s) for batch in ue.queue)
        for ue in env.ues
    )
    after_protocol = tuple(
        (
            ue.primary_cell,
            ue.secondary_cell,
            ue.state,
            ue.protocol_timer,
        )
        for ue in env.ues
    )
    assert after_queues == before["queues"]
    assert after_protocol == before["protocol"]
    assert repr(env.rng.bit_generator.state) == repr(before["rng"])


def test_deadline_aware_overlay_composes_with_planner_and_hybrid_overlays() -> None:
    config = load_config(
        ROOT / "configs/default.yaml",
        [
            ROOT / "configs/validation_agentic_challenge_v3.yaml",
            ROOT / "configs/counterfactual_planner.yaml",
            ROOT / "configs/hybrid_world_model.yaml",
            ROOT / "configs/deadline_aware_demand.yaml",
        ],
    )
    assert config["traffic"]["required_rate_model"] == DEADLINE_AWARE
    assert config["reasoning"]["planner_objective"] == "counterfactual"
    assert config["reasoning"]["world_model"] == "hybrid"
