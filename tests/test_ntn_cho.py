from pathlib import Path

import numpy as np

from cho_baselines import LocationBasedCHO, TimeBasedCHO
from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.ue import MobilityState

CFG = Path(__file__).resolve().parents[1] / "configs"
TWC = [
    CFG / "validation_agentic_challenge_v3.yaml", CFG / "eta_handover_005.yaml", CFG / "validation_dwell_15.yaml",
    CFG / "twc_walker_p3.yaml", CFG / "twc_3gpp_set1.yaml", CFG / "twc_load_24ue.yaml", CFG / "twc_events.yaml",
]


def _env():
    env = LEOMobilityEnv(load_config(CFG / "default.yaml", TWC))
    env.reset(seed=11)
    return env


def test_policies_only_emit_valid_actions():
    for policy in (TimeBasedCHO(), TimeBasedCHO(target_rule="load"), LocationBasedCHO(), LocationBasedCHO(target_rule="load")):
        env = _env()
        for _ in range(40):
            actions = policy(env)
            masks = env.action_masks()
            assert np.all(masks[np.arange(env.num_agents), actions] > 0)
            env.step(actions)


def test_t1_commits_only_inside_the_time_window():
    env = _env()
    policy = TimeBasedCHO(service_elevation_deg=30.0, lead_s=15.0)
    for _ in range(300):
        actions = policy(env)
        for ue in env.ues:
            if actions[ue.ue_id] == env.commit_action:
                t_end = policy._coverage_end_s(env, ue.primary_cell)
                assert t_end - policy.lead_s < env.time_s <= t_end
        env.step(actions)


def test_d2_entering_condition_matches_ts38331():
    env = _env()
    policy = LocationBasedCHO(thresh1_km=500.0, thresh2_km=900.0, hys_km=10.0, ttt_steps=1)
    for _ in range(300):
        actions = policy(env)
        for ue in env.ues:
            if actions[ue.ue_id] == env.commit_action and ue.state == MobilityState.PREPARED:
                ml1 = policy._distance(env, ue.ue_id, ue.primary_cell)
                ml2 = policy._distance(env, ue.ue_id, ue.secondary_cell)
                assert ml1 - policy.hys > policy.thresh1 and ml2 + policy.hys < policy.thresh2
        env.step(actions)
