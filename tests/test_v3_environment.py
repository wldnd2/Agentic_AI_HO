from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from leo_mappo import LEOMobilityEnv, MAPPO, load_config
from leo_mappo.coordinator import ReliabilityDualController, compute_training_reward
from leo_mappo.reasoning import (
    LookaheadReasoner,
    _planning_reliability_risk,
    _predicted_goodputs,
)
from leo_mappo.runner import evaluate_policy
from leo_mappo.ue import MobilityState, PacketBatch


ROOT = Path(__file__).resolve().parents[1]


def v3_config() -> dict:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["episode_steps"] = 12
    return config


def run_four_ue_contention(seed: int) -> tuple[set[int], dict[str, int]]:
    config = v3_config()
    config["environment"]["num_ues"] = 4
    env = LEOMobilityEnv(config)
    env.reset(seed=seed)
    target = 0
    for ue in env.ues:
        ue.state = MobilityState.SINGLE
        ue.secondary_cell = None
        env.candidates[ue.ue_id] = [target]
    actions = np.full(env.num_agents, env.prepare_start, dtype=np.int64)
    masks = np.zeros((env.num_agents, env.action_dim), dtype=np.float64)
    masks[:, env.prepare_start] = 1.0
    counters = env._apply_actions(actions, masks)
    admitted = {
        ue.ue_id for ue in env.ues if ue.state == MobilityState.PREPARED
    }
    return admitted, counters


def make_reasoning_case(v3: bool = True) -> tuple[LEOMobilityEnv, LookaheadReasoner]:
    config = (
        v3_config()
        if v3
        else load_config(ROOT / "configs/default.yaml")
    )
    env = LEOMobilityEnv(config)
    env.reset(seed=707)
    for ue in env.ues:
        ue.queue.append(PacketBatch(1_000))
    reasoner = LookaheadReasoner(config["reasoning"], env.reward_cfg)
    return env, reasoner


class V3EnvironmentTest(unittest.TestCase):
    def test_service_clipping_is_separate_from_reliability_headroom(self) -> None:
        required_rate = 100.0
        service_goodputs = []
        reliability_risks = []
        for potential_rate in (required_rate, 1.5 * required_rate):
            service_rate = min(required_rate, potential_rate)
            service_goodput, reliability_goodput = _predicted_goodputs(
                service_rate,
                potential_rate,
                delivery_probability=1.0,
                preserve_reliability_headroom=True,
            )
            service_goodputs.append(service_goodput)
            reliability_risks.append(
                _planning_reliability_risk(
                    reliability_goodput,
                    required_rate,
                    outage_service_ratio=0.80,
                    temperature=0.10,
                    epsilon=1e-8,
                )
            )
        deficits = [
            max(0.0, required_rate - goodput) / required_rate
            for goodput in service_goodputs
        ]
        self.assertEqual(deficits, [0.0, 0.0])
        self.assertLess(reliability_risks[1], reliability_risks[0])

    def test_planning_risk_at_required_rate_matches_sigmoid_minus_two(self) -> None:
        risk = _planning_reliability_risk(
            expected_goodput_bps=100.0,
            required_rate_bps=100.0,
            outage_service_ratio=0.80,
            temperature=0.10,
            epsilon=1e-8,
        )
        self.assertAlmostEqual(risk, 1.0 / (1.0 + np.exp(2.0)), places=9)

    def test_planning_risk_strictly_decreases_with_potential_headroom(self) -> None:
        required_rate = 100.0
        risks = []
        for multiplier in (0.8, 1.0, 1.2, 1.5):
            potential_rate = multiplier * required_rate
            service_rate = min(required_rate, potential_rate)
            _, reliability_goodput = _predicted_goodputs(
                service_rate,
                potential_rate,
                delivery_probability=1.0,
                preserve_reliability_headroom=True,
            )
            risks.append(
                _planning_reliability_risk(
                    reliability_goodput,
                    required_rate,
                    outage_service_ratio=0.80,
                    temperature=0.10,
                    epsilon=1e-8,
                )
            )
        self.assertTrue(all(left > right for left, right in zip(risks, risks[1:])))

    def test_ready_prepared_mask_allows_stay_commit_and_release(self) -> None:
        env = LEOMobilityEnv(v3_config())
        env.reset(seed=41)
        ue = next(ue for ue in env.ues if env.candidates[ue.ue_id])
        ue.secondary_cell = env.candidates[ue.ue_id][0]
        ue.state = MobilityState.PREPARED
        ue.protocol_timer = 0
        mask = env.action_masks()[ue.ue_id]
        expected = {0, env.commit_action, env.release_action}
        self.assertEqual(set(np.flatnonzero(mask)), expected)

    def test_v2_explicit_flag_is_legacy_equivalent(self) -> None:
        legacy = load_config(ROOT / "configs/default.yaml")
        explicit = deepcopy(legacy)
        explicit["environment"]["challenge_version"] = "v2"
        env_a, env_b = LEOMobilityEnv(legacy), LEOMobilityEnv(explicit)
        state_a = env_a.reset(seed=123)
        state_b = env_b.reset(seed=123)
        for left, right in zip(state_a, state_b):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(env_a.obs_dim, env_b.obs_dim)
        self.assertEqual(env_a.action_dim, env_b.action_dim)
        for _ in range(3):
            output_a = env_a.step(np.zeros(env_a.num_agents, dtype=np.int64))
            output_b = env_b.step(np.zeros(env_b.num_agents, dtype=np.int64))
            np.testing.assert_array_equal(output_a.observations, output_b.observations)
            self.assertEqual(output_a.info["team_reward"], output_b.info["team_reward"])

    def test_randomized_event_realization_is_seed_reproducible(self) -> None:
        env = LEOMobilityEnv(v3_config())
        env.reset(seed=321)
        first = env.events.realization_summary()
        env.reset(seed=321)
        self.assertEqual(first, env.events.realization_summary())
        env.reset(seed=322)
        self.assertNotEqual(first, env.events.realization_summary())

    def test_v3_candidate_order_ignores_future_visibility_and_pressure(self) -> None:
        env = LEOMobilityEnv(v3_config())
        env.reset(seed=2)
        ue = env.ues[0]
        ue.primary_cell = 0
        for link in env.links[0]:
            link.visible = True
        env.observed_sinr[0] = np.arange(env.num_cells, dtype=np.float64)
        first = env._select_candidates(ue)
        env.remaining_visibility[0] = np.arange(env.num_cells, dtype=np.float64)[::-1] * 1e6
        env.last_pressure[:] = np.arange(env.num_cells, dtype=np.float64) * 1e6
        second = env._select_candidates(ue)
        self.assertEqual(first, second)
        self.assertEqual(first, [5, 4, 3])

    def test_actor_observation_does_not_encode_hidden_or_future_state(self) -> None:
        env = LEOMobilityEnv(v3_config())
        env.reset(seed=8)
        ue = env.ues[0]
        before = env._observation(ue)
        env.latent_degradation_state[:] = 2
        env.latent_degradation_timer[:] = 999
        env.remaining_visibility[:] = 999_999.0
        for satellite in env.satellites:
            satellite.capacity_multiplier = 0.05
        if env.events.realized_specs:
            env.events.realized_specs[0]["severity"] = 999.0
            env.events.realized_specs[0]["duration_steps"] = 999
        np.testing.assert_array_equal(before, env._observation(ue))

    def test_preparation_control_slots_reject_excess_concurrent_requests(self) -> None:
        admitted, counters = run_four_ue_contention(11)
        self.assertEqual(len(admitted), 2)
        self.assertEqual(counters["preparation_admitted"], 2)
        self.assertEqual(counters["preparation_rejected"], 2)

    def test_preparation_batch_arbitration_is_seed_reproducible(self) -> None:
        first, _ = run_four_ue_contention(1234)
        second, _ = run_four_ue_contention(1234)
        self.assertEqual(first, second)

    def test_preparation_batch_arbitration_has_no_fixed_low_index_priority(self) -> None:
        admitted_sets = {frozenset(run_four_ue_contention(seed)[0]) for seed in range(20, 32)}
        self.assertGreater(len(admitted_sets), 1)
        self.assertTrue(any(admitted != frozenset({0, 1}) for admitted in admitted_sets))

    def test_prepared_target_reserves_resources_but_carries_no_user_data(self) -> None:
        config = v3_config()
        config["environment"]["num_ues"] = 1
        config["environment"]["num_satellites"] = 3
        config["environment"]["candidate_count"] = 2
        env = LEOMobilityEnv(config)
        env.reset(seed=41)
        ue = env.ues[0]
        primary = int(ue.primary_cell)
        secondary = next(cell for cell in range(env.num_cells) if cell != primary)
        ue.secondary_cell = secondary
        ue.state = MobilityState.PREPARED
        env.last_preparation_intent_count[secondary] = 1.0
        ue.queue.append(PacketBatch(100))
        for cell, probability in ((primary, 0.0), (secondary, 1.0)):
            link = env.links[0][cell]
            link.visible = True
            link.spectral_efficiency = 3.0
            link.packet_success_probability = probability
        active = np.ones(1, dtype=np.float64)
        required = np.asarray([120_000.0])
        metrics = env._schedule_and_transmit(required, active)
        rho = float(env.protocol_cfg["preparation_resource_fraction"])
        self.assertEqual(metrics["delivered_packets"], 0.0)
        self.assertAlmostEqual(env.last_utilization[secondary], rho)
        expected_coordination = (
            env.last_serving_demand + env.last_preparation_demand
        )
        np.testing.assert_allclose(env.last_coordination_demand, expected_coordination)

    def test_evaluation_action_seed_is_reproducible_and_restores_rng(self) -> None:
        config = v3_config()
        config["environment"]["episode_steps"] = 4
        env = LEOMobilityEnv(config)
        agent = MAPPO(
            env.obs_dim,
            env.global_state_dim,
            env.action_dim,
            config["mappo"],
            seed=13,
            num_satellites=env.num_cells,
            candidate_count=env.candidate_count,
        )
        original_rng = agent.rng
        state_before = repr(agent.rng.bit_generator.state)
        first = evaluate_policy(
            env, agent, episodes=2, deterministic=False, evaluation_action_seed=77
        )
        second = evaluate_policy(
            env, agent, episodes=2, deterministic=False, evaluation_action_seed=77
        )
        self.assertEqual(first, second)
        self.assertIs(agent.rng, original_rng)
        self.assertEqual(state_before, repr(agent.rng.bit_generator.state))

    def test_environment_and_agent_seeds_are_separable(self) -> None:
        first = v3_config()
        second = deepcopy(first)
        first.update(environment_seed=7, agent_seed=7)
        second.update(environment_seed=7, agent_seed=8)
        env_a, env_b = LEOMobilityEnv(first), LEOMobilityEnv(second)
        population_a = [(ue.latitude_rad, ue.longitude_rad, ue.mean_arrival_mbps) for ue in env_a.ues]
        population_b = [(ue.latitude_rad, ue.longitude_rad, ue.mean_arrival_mbps) for ue in env_b.ues]
        self.assertEqual(population_a, population_b)
        agent_a = MAPPO(env_a.obs_dim, env_a.global_state_dim, env_a.action_dim, first["mappo"], first["agent_seed"])
        agent_b = MAPPO(env_b.obs_dim, env_b.global_state_dim, env_b.action_dim, second["mappo"], second["agent_seed"])
        self.assertFalse(np.array_equal(agent_a.actor.parameters["w1"], agent_b.actor.parameters["w1"]))

    def test_v3_reasoner_does_not_read_true_traffic_multiplier(self) -> None:
        env, reasoner = make_reasoning_case()
        observed_before = env.observed_required_rates()[1]
        first = reasoner.analyze(env)
        true_before = env.current_required_rates()[1]
        env._active_effects.traffic_multiplier[:] = 4.0
        true_after = env.current_required_rates()[1]
        observed_after = env.observed_required_rates()[1]
        second = reasoner.analyze(env)
        self.assertFalse(np.array_equal(true_before, true_after))
        np.testing.assert_array_equal(observed_before, observed_after)
        np.testing.assert_array_equal(first.raw_costs, second.raw_costs)

    def test_v3_reasoner_does_not_read_true_capacity_multiplier(self) -> None:
        env, reasoner = make_reasoning_case()
        first = reasoner.analyze(env)
        true_before = np.asarray(
            [satellite.effective_capacity_scale for satellite in env.satellites]
        )
        for satellite in env.satellites:
            satellite.capacity_multiplier = 0.10
        true_after = np.asarray(
            [satellite.effective_capacity_scale for satellite in env.satellites]
        )
        second = reasoner.analyze(env)
        self.assertFalse(np.array_equal(true_before, true_after))
        np.testing.assert_array_equal(first.raw_costs, second.raw_costs)

    def test_v3_m5_reliability_prediction_ignores_hidden_future_events(self) -> None:
        env, _ = make_reasoning_case()
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
            use_constraint=True,
        )
        prices = np.full(env.num_cells, 0.25, dtype=np.float64)
        first = reasoner.analyze(env, prices=prices, outage_dual_mu=0.5)
        env._active_effects.traffic_multiplier[:] = 4.0
        env.latent_degradation_state[:] = 2
        env.latent_degradation_timer[:] = 999
        for satellite in env.satellites:
            satellite.capacity_multiplier = 0.05
        for spec in env.events.realized_specs:
            spec["severity"] = 999.0
            spec["duration_steps"] = 999
        second = reasoner.analyze(env, prices=prices, outage_dual_mu=0.5)
        np.testing.assert_array_equal(
            first.planning_reliability_risks,
            second.planning_reliability_risks,
        )
        np.testing.assert_array_equal(first.total_costs, second.total_costs)

    def test_v3_reasoner_remains_sensitive_to_future_geometry(self) -> None:
        env, reasoner = make_reasoning_case()
        ue_id = next(index for index, candidates in enumerate(env.candidates) if candidates)
        target = env.candidates[ue_id][0]
        action = env.prepare_start
        satellite = env.satellites[target]
        satellite.angular_rate_rad_s = 0.0
        static_geometry = reasoner.analyze(env).raw_costs[ue_id, action]
        satellite.angular_rate_rad_s = 0.25
        changing_geometry = reasoner.analyze(env).raw_costs[ue_id, action]
        self.assertNotEqual(static_geometry, changing_geometry)

    def test_v3_m3_reasoning_total_is_service_only(self) -> None:
        env, reasoner = make_reasoning_case()
        result = reasoner.analyze(env)
        masks = env.action_masks()
        feasible = masks > 0.0
        self.assertGreater(float(result.predicted_control_costs[feasible].max()), 0.0)
        np.testing.assert_allclose(
            result.total_costs[feasible], result.service_costs[feasible], atol=1e-12
        )
        np.testing.assert_array_equal(
            result.coordination_costs[feasible], np.zeros(np.count_nonzero(feasible))
        )
        np.testing.assert_array_equal(
            result.reliability_costs[feasible], np.zeros(np.count_nonzero(feasible))
        )

    def test_v3_m3_target_scores_ignore_other_ue_dynamic_demand(self) -> None:
        env, reasoner = make_reasoning_case()
        target_ue = 0
        target_observation = env._observation(env.ues[target_ue]).copy()
        target_candidates = list(env.candidates[target_ue])
        first = reasoner.analyze(env).raw_costs[target_ue].copy()
        for other in env.ues[1:]:
            other.queue.clear()
            other.mean_arrival_mbps *= 100.0
        second = reasoner.analyze(env).raw_costs[target_ue]
        np.testing.assert_array_equal(target_observation, env._observation(env.ues[target_ue]))
        self.assertEqual(target_candidates, env.candidates[target_ue])
        np.testing.assert_array_equal(first, second)

    def test_v3_m4_target_scores_retain_price_dependency(self) -> None:
        env, _ = make_reasoning_case()
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
        )
        target_ue = next(
            index for index, candidates in enumerate(env.candidates) if candidates
        )
        target_cell = env.candidates[target_ue][0]
        action = env.prepare_start
        zero_prices = np.zeros(env.num_cells, dtype=np.float64)
        priced = zero_prices.copy()
        priced[target_cell] = 1.0
        without_price = reasoner.analyze(env, prices=zero_prices)
        with_price = reasoner.analyze(env, prices=priced)
        self.assertNotEqual(
            without_price.coordination_costs[target_ue, action],
            with_price.coordination_costs[target_ue, action],
        )
        self.assertNotEqual(
            without_price.raw_costs[target_ue, action],
            with_price.raw_costs[target_ue, action],
        )
        self.assertEqual(with_price.reliability_costs[target_ue, action], 0.0)
        self.assertAlmostEqual(
            with_price.total_costs[target_ue, action],
            with_price.service_costs[target_ue, action]
            + with_price.coordination_costs[target_ue, action],
        )

    def test_v3_ready_stay_and_release_receive_no_price_rebate(self) -> None:
        env, _ = make_reasoning_case()
        ue_id = next(index for index, candidates in enumerate(env.candidates) if candidates)
        ue = env.ues[ue_id]
        ue.secondary_cell = env.candidates[ue_id][0]
        ue.state = MobilityState.PREPARED
        ue.protocol_timer = 0
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
        )
        result = reasoner.analyze(env, prices=np.ones(env.num_cells))
        self.assertAlmostEqual(result.coordination_costs[ue_id, 0], 0.0)
        self.assertAlmostEqual(
            result.coordination_costs[ue_id, env.release_action], 0.0
        )

    def test_v3_prepare_has_positive_price_scaled_incremental_demand(self) -> None:
        env, _ = make_reasoning_case()
        ue_id = next(index for index, candidates in enumerate(env.candidates) if candidates)
        target = env.candidates[ue_id][0]
        action = env.prepare_start
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
        )
        low_prices = np.zeros(env.num_cells)
        low_prices[target] = 0.5
        high_prices = low_prices.copy()
        high_prices[target] = 2.0
        low = reasoner.analyze(env, prices=low_prices)
        high = reasoner.analyze(env, prices=high_prices)
        self.assertGreater(low.coordination_costs[ue_id, action], 0.0)
        self.assertGreater(
            high.coordination_costs[ue_id, action],
            low.coordination_costs[ue_id, action],
        )

    def test_v3_control_cost_contains_only_preparation_and_handover_terms(self) -> None:
        env, _ = make_reasoning_case()
        ue_id = next(index for index, candidates in enumerate(env.candidates) if candidates)
        ue = env.ues[ue_id]
        target = env.candidates[ue_id][0]
        ue.secondary_cell = target
        ue.state = MobilityState.PREPARED
        ue.protocol_timer = 0
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
        )
        result = reasoner.analyze(env)
        for action in np.flatnonzero(env.action_masks()[ue_id] > 0.0):
            expected = (
                reasoner.eta_preparation
                * result.predicted_preparation_costs[ue_id, action]
                + reasoner.eta_handover
                * result.predicted_handover_costs[ue_id, action]
            )
            self.assertAlmostEqual(
                result.predicted_control_costs[ue_id, action], expected
            )

    def test_v3_all_legal_actions_have_no_negative_coordination_credit(self) -> None:
        env, _ = make_reasoning_case()
        ue_id = next(index for index, candidates in enumerate(env.candidates) if candidates)
        ue = env.ues[ue_id]
        ue.secondary_cell = env.candidates[ue_id][0]
        ue.state = MobilityState.PREPARED
        ue.protocol_timer = 0
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
        )
        result = reasoner.analyze(env, prices=np.ones(env.num_cells))
        legal = env.action_masks()[ue_id] > 0.0
        self.assertTrue(np.all(result.coordination_costs[ue_id, legal] >= -1e-12))
        self.assertAlmostEqual(
            result.coordination_costs[ue_id, env.release_action], 0.0
        )

    def test_v3_commit_prices_only_positive_target_increment(self) -> None:
        env, _ = make_reasoning_case()
        ue_id = next(index for index, candidates in enumerate(env.candidates) if candidates)
        ue = env.ues[ue_id]
        target = env.candidates[ue_id][0]
        ue.secondary_cell = target
        ue.state = MobilityState.PREPARED
        ue.protocol_timer = 0
        prices = np.zeros(env.num_cells)
        prices[ue.primary_cell] = 5.0
        prices[target] = 1.0
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
        )
        result = reasoner.analyze(env, prices=prices)
        self.assertGreater(result.coordination_costs[ue_id, env.commit_action], 0.0)

    def test_v3_price_still_ranks_new_demand_between_candidates(self) -> None:
        env, _ = make_reasoning_case()
        ue_id = next(
            index for index, candidates in enumerate(env.candidates)
            if len(candidates) >= 2
        )
        first_target, second_target = env.candidates[ue_id][:2]
        env.remaining_visibility[ue_id, [first_target, second_target]] = 1e9
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
        )
        prices = np.zeros(env.num_cells)
        prices[first_target] = 0.25
        prices[second_target] = 2.0
        with patch.object(reasoner, "_future_link", return_value=(1.0, 1.0)):
            result = reasoner.analyze(env, prices=prices)
        self.assertLess(
            result.coordination_costs[ue_id, env.prepare_start],
            result.coordination_costs[ue_id, env.prepare_start + 1],
        )

    def test_v3_m4_and_m5_reasoner_rankings_are_identical(self) -> None:
        env, _ = make_reasoning_case()
        prices = np.linspace(0.1, 0.6, env.num_cells)
        m4 = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
        ).analyze(env, prices=prices)
        m5 = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
            use_constraint=True,
        ).analyze(env, prices=prices, outage_dual_mu=1.0)
        feasible = env.action_masks() > 0.0
        np.testing.assert_allclose(
            m5.total_costs[feasible], m4.total_costs[feasible], atol=1e-12
        )
        np.testing.assert_allclose(
            m5.raw_costs[feasible], m4.raw_costs[feasible], atol=1e-12
        )
        np.testing.assert_allclose(
            m5.features[feasible], m4.features[feasible], atol=1e-12
        )
        np.testing.assert_array_equal(
            np.argsort(m5.raw_costs, axis=1), np.argsort(m4.raw_costs, axis=1)
        )
        np.testing.assert_array_equal(
            m5.reliability_costs[feasible], np.zeros(np.count_nonzero(feasible))
        )

    def test_v2_absolute_coordination_semantics_are_unchanged(self) -> None:
        env, _ = make_reasoning_case(v3=False)
        ue_id = next(index for index, candidates in enumerate(env.candidates) if candidates)
        ue = env.ues[ue_id]
        ue.secondary_cell = env.candidates[ue_id][0]
        ue.state = MobilityState.PREPARED
        ue.protocol_timer = 0
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
        )
        result = reasoner.analyze(env, prices=np.ones(env.num_cells))
        self.assertGreater(
            result.coordination_costs[ue_id, 0],
            result.coordination_costs[ue_id, env.release_action],
        )

    def test_v3_m5_reasoner_ranking_is_mu_invariant(self) -> None:
        env, _ = make_reasoning_case()
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
            use_constraint=True,
        )
        prices = np.full(env.num_cells, 0.5, dtype=np.float64)
        results = [
            reasoner.analyze(env, prices=prices, outage_dual_mu=mu)
            for mu in (0.0, 0.1, 1.0)
        ]
        feasible = env.action_masks() > 0.0
        for result in results:
            np.testing.assert_allclose(
                result.total_costs[feasible],
                result.service_costs[feasible] + result.coordination_costs[feasible],
                atol=1e-12,
            )
            np.testing.assert_array_equal(
                result.reliability_costs[feasible],
                np.zeros(np.count_nonzero(feasible)),
            )
        for result in results[1:]:
            np.testing.assert_array_equal(result.total_costs, results[0].total_costs)
            np.testing.assert_array_equal(result.raw_costs, results[0].raw_costs)
            np.testing.assert_array_equal(result.features, results[0].features)
            np.testing.assert_array_equal(
                np.argsort(result.raw_costs, axis=1),
                np.argsort(results[0].raw_costs, axis=1),
            )

    def test_v3_m5_analytical_risk_does_not_leak_into_actor_reasoning_feature(self) -> None:
        env, _ = make_reasoning_case()
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
            use_constraint=True,
        )
        prices = np.linspace(0.1, 0.6, env.num_cells)
        with patch("leo_mappo.reasoning._planning_reliability_risk", return_value=0.0):
            zero_risk = reasoner.analyze(env, prices=prices, outage_dual_mu=1.0)
        with patch("leo_mappo.reasoning._planning_reliability_risk", return_value=1.0):
            unit_risk = reasoner.analyze(env, prices=prices, outage_dual_mu=1.0)
        np.testing.assert_array_equal(zero_risk.features, unit_risk.features)
        np.testing.assert_array_equal(zero_risk.raw_costs, unit_risk.raw_costs)
        np.testing.assert_array_equal(zero_risk.total_costs, unit_risk.total_costs)
        self.assertFalse(
            np.array_equal(
                zero_risk.planning_reliability_risks,
                unit_risk.planning_reliability_risks,
            )
        )

    def test_v3_m5_actor_still_receives_normalized_mu(self) -> None:
        config = load_config(
            ROOT / "configs/default.yaml",
            ROOT / "configs/validation_agentic_challenge_v3.yaml",
        )
        config["mappo"].update(
            use_memory=True,
            use_reasoning=True,
            use_coordination=True,
            use_constraint=True,
        )
        env = LEOMobilityEnv(config)
        observations, _, masks = env.reset(seed=909)
        agent = MAPPO(
            env.obs_dim,
            env.global_state_dim,
            env.action_dim,
            config["mappo"],
            seed=17,
            num_satellites=env.num_cells,
            candidate_count=env.candidate_count,
        )
        captured_inputs: list[np.ndarray] = []
        original_forward = agent.actor.forward

        def capture_forward(values: np.ndarray):
            captured_inputs.append(values.copy())
            return original_forward(values)

        common = (
            observations,
            masks,
            np.full(env.num_agents, -1, dtype=np.int64),
            np.zeros((env.num_agents, agent.outcome_dim)),
            np.zeros((env.num_agents, agent.recurrent_hidden_dim)),
            np.zeros((env.num_agents, env.action_dim)),
            np.zeros((env.num_agents, agent.price_feature_dim)),
        )
        with patch.object(agent.actor, "forward", side_effect=capture_forward):
            agent.act_recurrent(*common, normalized_mu=0.0, deterministic=True)
            agent.act_recurrent(*common, normalized_mu=0.25, deterministic=True)
        np.testing.assert_array_equal(captured_inputs[0][:, -1], 0.0)
        np.testing.assert_array_equal(captured_inputs[1][:, -1], 0.25)
        np.testing.assert_array_equal(
            captured_inputs[0][:, :-1], captured_inputs[1][:, :-1]
        )

    def test_m5_constrained_reward_and_dual_update_remain_mu_dependent(self) -> None:
        base, base_penalty = compute_training_reward(
            0.2, 0.1, 0.4, 0.25, True, 0.0, 0.1
        )
        constrained, constrained_penalty = compute_training_reward(
            0.2, 0.1, 0.4, 0.25, True, 0.5, 0.1
        )
        self.assertEqual(base_penalty, 0.0)
        self.assertGreater(constrained_penalty, 0.0)
        self.assertLess(constrained, base)

        controller = ReliabilityDualController(
            {
                "outage_target": 0.1,
                "dual_learning_rate": 0.005,
                "mu_initial": 0.5,
                "mu_max": 1.0,
            }
        )
        increased = controller.update(outage_count=2.0, active_count=10.0)
        decreased = controller.update(outage_count=0.0, active_count=10.0)
        self.assertGreater(increased, 0.5)
        self.assertLess(decreased, increased)
        lower = ReliabilityDualController(
            {
                "outage_target": 0.1,
                "dual_learning_rate": 1.0,
                "mu_initial": 0.0,
                "mu_max": 1.0,
            }
        )
        upper = ReliabilityDualController(
            {
                "outage_target": 0.1,
                "dual_learning_rate": 1.0,
                "mu_initial": 1.0,
                "mu_max": 1.0,
            }
        )
        self.assertEqual(lower.update(0.0, 10.0), 0.0)
        self.assertEqual(upper.update(10.0, 10.0), 1.0)

    def test_v2_constrained_reasoner_keeps_legacy_reliability_term(self) -> None:
        env, _ = make_reasoning_case(v3=False)
        reasoner = LookaheadReasoner(
            env.config["reasoning"],
            env.reward_cfg,
            env.config.get("coordination", {}),
            use_coordination=True,
            use_constraint=True,
        )
        prices = np.zeros(env.num_cells)
        without_mu = reasoner.analyze(env, prices=prices, outage_dual_mu=0.0)
        with_mu = reasoner.analyze(env, prices=prices, outage_dual_mu=1.0)
        feasible = env.action_masks() > 0.0
        self.assertGreater(float(with_mu.reliability_costs[feasible].max()), 0.0)
        self.assertTrue(
            np.any(
                with_mu.total_costs[feasible]
                != without_mu.total_costs[feasible]
            )
        )

    def test_v2_reasoner_required_rate_interface_preserves_true_behavior(self) -> None:
        env, reasoner = make_reasoning_case(v3=False)
        active_true, required_true = env.current_required_rates()
        active_reasoner, required_reasoner = env.observed_required_rates()
        np.testing.assert_array_equal(active_true, active_reasoner)
        np.testing.assert_array_equal(required_true, required_reasoner)
        result = reasoner.analyze(env)
        self.assertTrue(np.isfinite(result.raw_costs).all())
        feasible_with_control = (
            (env.action_masks() > 0.0) & (result.predicted_control_costs > 0.0)
        )
        self.assertTrue(np.any(feasible_with_control))
        self.assertTrue(
            np.all(
                result.total_costs[feasible_with_control]
                > result.service_costs[feasible_with_control]
            )
        )


if __name__ == "__main__":
    unittest.main()
