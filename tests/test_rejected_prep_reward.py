from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from leo_mappo import LEOMobilityEnv, load_config
from leo_mappo.ue import MobilityState, PacketBatch


ROOT = Path(__file__).resolve().parents[1]


def _config(*, charge_rejected: bool, admission_capacity: int = 0) -> dict:
    config = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/validation_agentic_challenge_v3.yaml",
    )
    config["environment"]["episode_steps"] = 3
    config["v3"]["preparation_admission"][
        "max_concurrent_per_satellite"
    ] = admission_capacity
    config["reward"]["charge_rejected_preparation_signaling"] = charge_rejected
    return config


def _forced_rejected_request(config: dict, seed: int = 71):
    env = LEOMobilityEnv(deepcopy(config))
    env.reset(seed=seed)
    for ue in env.ues:
        ue.queue.append(PacketBatch(1_000))
    requester = next(ue for ue in env.ues if env.candidates[ue.ue_id])
    requester.state = MobilityState.SINGLE
    requester.secondary_cell = None
    actions = np.zeros(env.num_agents, dtype=np.int64)
    actions[requester.ue_id] = env.prepare_start
    output = env.step(actions)
    return env, requester.ue_id, output


def test_old_ablation_preserves_rejected_prep_reward_behavior() -> None:
    env, ue_id, output = _forced_rejected_request(_config(charge_rejected=False))

    assert output.info["preparation_rejected"] == 1.0
    assert env.ues[ue_id].state == MobilityState.SINGLE
    assert output.info["rejected_preparation_signaling_cost"] > 0.0
    assert output.info["charged_rejected_preparation_signaling_cost"] == 0.0
    assert output.info["rejected_preparation_control_cost"] == 0.0
    assert output.info["preparation_control_cost"] == output.info[
        "preparation_reservation_control_cost"
    ]


def test_corrected_reward_charges_rejected_prep_with_existing_coefficients() -> None:
    old_env, _, old = _forced_rejected_request(_config(charge_rejected=False))
    new_env, ue_id, new = _forced_rejected_request(_config(charge_rejected=True))

    rejected_signaling = new.info["rejected_preparation_signaling_cost"]
    expected_control_delta = (
        float(new_env.reward_cfg["eta_preparation"]) * rejected_signaling
    )
    expected_reward_delta = (
        float(new_env.reward_cfg["lambda_resource"]) * expected_control_delta
    )

    assert new.info["preparation_rejected"] == old.info["preparation_rejected"] == 1.0
    assert new_env.ues[ue_id].state == MobilityState.SINGLE
    assert new.info["signaling_cost"] == old.info["signaling_cost"]
    assert new.info["preparation_train_cost"] == old.info["preparation_train_cost"]
    assert new.info["handover_control_cost"] == old.info["handover_control_cost"]
    assert np.isclose(new.info["rejected_preparation_control_cost"], expected_control_delta)
    assert np.isclose(new.info["control_cost"] - old.info["control_cost"], expected_control_delta)
    assert np.isclose(old.info["team_reward"] - new.info["team_reward"], expected_reward_delta)
    assert old_env.reward_cfg["charge_rejected_preparation_signaling"] is False


def test_admitted_prep_is_charged_only_for_reservation_occupancy() -> None:
    env, ue_id, output = _forced_rejected_request(
        _config(charge_rejected=True, admission_capacity=1)
    )

    assert output.info["preparation_admitted"] == 1.0
    assert output.info["preparation_rejected"] == 0.0
    assert env.ues[ue_id].state == MobilityState.PREPARED
    assert output.info["signaling_cost"] > 0.0
    assert output.info["rejected_preparation_signaling_cost"] == 0.0
    assert output.info["rejected_preparation_control_cost"] == 0.0
    assert output.info["preparation_reservation_control_cost"] > 0.0
    assert output.info["preparation_control_cost"] == output.info[
        "preparation_reservation_control_cost"
    ]


def test_corrected_ablation_changes_only_the_reward_switch() -> None:
    old = load_config(ROOT / "configs/default.yaml")
    corrected = load_config(
        ROOT / "configs/default.yaml",
        ROOT / "configs/rejected_prep_signaling_reward.yaml",
    )

    expected = deepcopy(old)
    expected["reward"]["charge_rejected_preparation_signaling"] = True
    assert corrected == expected
