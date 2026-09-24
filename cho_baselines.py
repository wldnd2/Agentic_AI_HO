"""3GPP-style Conditional Handover (CHO) baselines for the V3 environment.

Mapping of Rel-16/17 CHO onto the simulator's protocol state machine:
  * Preparation  : measurement-report trigger. When the best observed candidate
                   exceeds the serving cell by `prep_offset_db`, prepare it
                   (Single -> Prepared). This is the early, low-offset A3 report.
  * Execution    : CHO execution condition (A3 event). When the prepared target
                   exceeds the serving cell by `exec_offset_db` for `ttt_steps`
                   consecutive control epochs (time-to-trigger), Commit.
  * Cancellation : if the prepared target falls below serving + `release_offset_db`
                   for `release_steps` epochs, Release (or Dup-OFF when duplicating).
  * Duplication  : optional (`dup_threshold_db`). While a target is prepared and the
                   serving observed SINR drops below the threshold, activate PDCP
                   duplication until execution or cancellation.

Like the other V3-fair heuristics, the policies only use actor-visible telemetry
(delayed, noisy `env.observed_sinr`) and the V3 candidate list (ranked by observed
current SINR). Invalid actions fall back to Stay through the environment mask.
Control epochs are 1 s, so TTT is expressed in epochs rather than milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from leo_mappo.ue import MobilityState


class A3ConditionalHandover:
    def __init__(
        self,
        prep_offset_db: float = 1.0,
        exec_offset_db: float = 3.0,
        ttt_steps: int = 2,
        release_offset_db: float = -1.0,
        release_steps: int = 3,
        dup_threshold_db: float | None = None,
    ) -> None:
        self.prep_offset_db = float(prep_offset_db)
        self.exec_offset_db = float(exec_offset_db)
        self.ttt_steps = int(ttt_steps)
        self.release_offset_db = float(release_offset_db)
        self.release_steps = int(release_steps)
        self.dup_threshold_db = None if dup_threshold_db is None else float(dup_threshold_db)
        self._exec_count: np.ndarray | None = None
        self._release_count: np.ndarray | None = None
        self._last_step = -1

    def _maybe_reset(self, env) -> None:
        # New episode (step counter went back) or first call: clear TTT timers.
        if (
            self._exec_count is None
            or len(self._exec_count) != env.num_agents
            or env.step_index <= self._last_step
        ):
            self._exec_count = np.zeros(env.num_agents, dtype=np.int64)
            self._release_count = np.zeros(env.num_agents, dtype=np.int64)
        self._last_step = env.step_index

    def __call__(self, env) -> np.ndarray:
        self._maybe_reset(env)
        obs = env.observed_sinr
        actions = np.zeros(env.num_agents, dtype=np.int64)

        for ue in env.ues:
            uid = ue.ue_id
            primary = ue.primary_cell
            if primary is None:
                continue
            serving = float(obs[uid, primary])

            if ue.state == MobilityState.SINGLE:
                self._exec_count[uid] = 0
                self._release_count[uid] = 0
                candidates = env.candidates[uid]
                if candidates:
                    best = candidates[0]
                    if best != primary and float(obs[uid, best]) > serving + self.prep_offset_db:
                        actions[uid] = env.prepare_start  # slot 0 = best observed candidate

            elif ue.state in (MobilityState.PREPARED, MobilityState.DUPLICATING):
                target = ue.secondary_cell
                if target is None:
                    continue
                target_sinr = float(obs[uid, target])
                self._exec_count[uid] = (
                    self._exec_count[uid] + 1
                    if target_sinr > serving + self.exec_offset_db else 0
                )
                self._release_count[uid] = (
                    self._release_count[uid] + 1
                    if target_sinr < serving + self.release_offset_db else 0
                )

                if self._exec_count[uid] >= self.ttt_steps:
                    actions[uid] = env.commit_action
                elif self._release_count[uid] >= self.release_steps:
                    actions[uid] = (
                        env.dup_off_action
                        if ue.state == MobilityState.DUPLICATING
                        else env.release_action
                    )
                elif (
                    self.dup_threshold_db is not None
                    and ue.state == MobilityState.PREPARED
                    and serving < self.dup_threshold_db
                ):
                    actions[uid] = env.dup_on_action
            else:
                # Switching / Cooldown: protocol timers run, nothing to decide.
                self._exec_count[uid] = 0
                self._release_count[uid] = 0

        masks = env.action_masks()
        for uid in range(env.num_agents):
            if masks[uid, int(actions[uid])] <= 0.0:
                actions[uid] = 0
        return actions


PARAMS_FILE = Path(__file__).resolve().parent / "cho_params.json"
DEFAULT_PARAMS = {
    "prep_offset_db": 1.0,
    "exec_offset_db": 3.0,
    "ttt_steps": 2,
    "release_offset_db": -1.0,
    "release_steps": 3,
    "dup_threshold_db": 4.0,
}


def load_cho_params() -> dict:
    """Tuned parameters from cho_sweep.py (cho_params.json) if present, else defaults."""
    params = dict(DEFAULT_PARAMS)
    if PARAMS_FILE.is_file():
        stored = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        params.update({k: v for k, v in stored.items() if k in DEFAULT_PARAMS})
    return params


def make_cho_policies(params: dict | None = None) -> dict:
    """Fresh (stateful) policy instances; call once per evaluation job."""
    p = load_cho_params() if params is None else {**DEFAULT_PARAMS, **params}
    a3 = {k: p[k] for k in ("prep_offset_db", "exec_offset_db", "ttt_steps",
                            "release_offset_db", "release_steps")}
    return {
        "CHO-A3-TTT": A3ConditionalHandover(**a3),
        "CHO-A3-TTT+Dup": A3ConditionalHandover(**a3, dup_threshold_db=p["dup_threshold_db"]),
    }
