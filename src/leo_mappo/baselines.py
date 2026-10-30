from __future__ import annotations

import numpy as np

from .environment import LEOMobilityEnv
from .ue import MobilityState


def _mask_aware_actions(env: LEOMobilityEnv, desired: np.ndarray) -> np.ndarray:
    masks = env.action_masks()
    actions = np.zeros(env.num_agents, dtype=np.int64)
    for ue_id, action in enumerate(desired):
        if 0 <= action < env.action_dim and masks[ue_id, action] > 0.0:
            actions[ue_id] = action
    return actions


def max_sinr_policy(env: LEOMobilityEnv) -> np.ndarray:
    """Prepare the best candidate and commit when preparation completes."""
    actions = np.zeros(env.num_agents, dtype=np.int64)
    for ue in env.ues:
        if ue.state == MobilityState.SINGLE and env.candidates[ue.ue_id]:
            primary_sinr = env.links[ue.ue_id][ue.primary_cell].sinr_db
            slot = max(
                range(len(env.candidates[ue.ue_id])),
                key=lambda index: env.links[ue.ue_id][env.candidates[ue.ue_id][index]].sinr_db,
            )
            target = env.candidates[ue.ue_id][slot]
            if env.links[ue.ue_id][target].sinr_db > primary_sinr + 1.0:
                actions[ue.ue_id] = env.prepare_start + slot
        elif ue.state == MobilityState.PREPARED and ue.protocol_timer <= 0:
            actions[ue.ue_id] = env.commit_action
    return _mask_aware_actions(env, actions)
