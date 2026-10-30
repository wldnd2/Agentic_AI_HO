"""3GPP-style Conditional Handover (CHO) baselines for the V3 environment.

Mapping of Rel-16/17 CHO onto the simulator's protocol state machine:
  * Preparation  : measurement-report trigger. When the best observed candidate
                   exceeds the serving cell by `prep_offset_db`, prepare it
                   (Single -> Prepared). This is the early, low-offset A3 report.
  * Execution    : CHO execution condition (A3 event). When the prepared target
                   exceeds the serving cell by `exec_offset_db` for `ttt_steps`
                   consecutive control epochs (time-to-trigger), Commit.
  * Cancellation : if the prepared target falls below serving + `release_offset_db`
                   for `release_steps` epochs, Release.

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
    ) -> None:
        self.prep_offset_db = float(prep_offset_db)
        self.exec_offset_db = float(exec_offset_db)
        self.ttt_steps = int(ttt_steps)
        self.release_offset_db = float(release_offset_db)
        self.release_steps = int(release_steps)
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

            elif ue.state == MobilityState.PREPARED:
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
                    actions[uid] = env.release_action
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
    }


# ---------------------------------------------------------------------------
# NTN conditional handover (TS 38.331 V19.1.0)
# ---------------------------------------------------------------------------
# CondEvent T1 (Sec. 5.5.4.16): entering Mt > Thresh1, leaving Mt > Thresh1 + Duration.
# CondEvent D2 (Sec. 5.5.4.15a, Rel-18): entering Ml1 - Hys > Thresh1 and
#   Ml2 + Hys < Thresh2, leaving Ml1 + Hys < Thresh1 or Ml2 - Hys > Thresh2,
#   with moving reference locations (here: the sub-satellite points).
# CondEvent D1 (Sec. 5.5.4.15) uses fixed reference locations of the serving and
# target cells. With one earth-fixed cell served by every satellite, both
# reference locations coincide, so D1 is not applicable in this model.

def _sub_satellite_point(position: np.ndarray) -> np.ndarray:
    from leo_mappo.satellite import EARTH_RADIUS_M

    return EARTH_RADIUS_M * position / np.linalg.norm(position)


def _ground_distance_m(a: np.ndarray, b: np.ndarray) -> float:
    from leo_mappo.satellite import EARTH_RADIUS_M

    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return EARTH_RADIUS_M * float(np.arccos(np.clip(cosine, -1.0, 1.0)))


class TimeBasedCHO:
    """Rel-17 CHO with condEventT1 for a quasi-earth-fixed cell.

    The network plans, from the ephemeris, the time t_end at which the serving
    satellite's elevation at the cell centre falls below the service elevation
    service_elevation_deg (satellite-cell assignment plan), and configures
    t1-Threshold = t_end - lead_s and duration = lead_s. A preparation is
    requested prep_lead_s before the window opens, towards the candidate whose
    planned service lasts longest.
    """

    def __init__(
        self,
        service_elevation_deg: float = 30.0,
        lead_s: float = 15.0,
        prep_lead_s: float = 5.0,
        a3_offset_db: float | None = None,
        target_rule: str = "geometry",
        min_service_s: float = 60.0,
    ) -> None:
        self.target_rule = target_rule
        self.min_service_s = float(min_service_s)
        self.service_elevation_rad = np.radians(float(service_elevation_deg))
        self.lead_s = float(lead_s)
        self.prep_lead_s = float(prep_lead_s)
        self.a3_offset_db = None if a3_offset_db is None else float(a3_offset_db)
        self._cache_step = -1
        self._coverage_end: dict[int, float] = {}

    def _coverage_end_s(self, env, cell_id: int) -> float:
        from leo_mappo.satellite import estimate_remaining_visibility, ground_position_ecef

        if env.step_index != self._cache_step:
            self._cache_step, self._coverage_end = env.step_index, {}
        if cell_id not in self._coverage_end:
            centre = ground_position_ecef(0.0, 0.0)
            remaining = estimate_remaining_visibility(
                env.satellites[cell_id], centre, env.time_s, max(env.min_elevation_rad, self.service_elevation_rad)
            )
            self._coverage_end[cell_id] = env.time_s + float(remaining)
        return self._coverage_end[cell_id]

    def __call__(self, env) -> np.ndarray:
        actions = np.zeros(env.num_agents, dtype=np.int64)
        assigned: dict[int, int] = {}
        for ue in env.ues:
            uid, primary = ue.ue_id, ue.primary_cell
            if primary is None:
                continue
            t_end = self._coverage_end_s(env, primary)
            thresh1, leave = t_end - self.lead_s, t_end  # leaving: Mt > Thresh1 + Duration
            if ue.state == MobilityState.SINGLE:
                candidates = list(env.candidates[uid])
                if candidates and env.time_s >= thresh1 - self.prep_lead_s:
                    best = _pick_target(
                        env, candidates, lambda c: self._coverage_end_s(env, c),
                        lambda c: self._coverage_end_s(env, c) >= env.time_s + self.min_service_s,
                        self.target_rule, assigned)
                    assigned[candidates[best]] = assigned.get(candidates[best], 0) + 1
                    actions[uid] = env.prepare_start + best
            elif ue.state == MobilityState.PREPARED:
                target = ue.secondary_cell
                if target is None:
                    continue
                entering = thresh1 < env.time_s <= leave
                if entering and self.a3_offset_db is not None:
                    obs = env.observed_sinr
                    entering = float(obs[uid, target]) > float(obs[uid, primary]) + self.a3_offset_db
                if entering:
                    actions[uid] = env.commit_action
                elif env.time_s > leave:
                    actions[uid] = env.release_action
        return _mask_invalid(env, actions)


class LocationBasedCHO:
    """Rel-18 CHO with condEventD2, using the sub-satellite points of the
    serving and target satellites as moving reference locations."""

    def __init__(
        self,
        thresh1_km: float = 1000.0,
        thresh2_km: float = 700.0,
        hys_km: float = 10.0,
        ttt_steps: int = 1,
        prep_margin_km: float = 100.0,
        release_steps: int = 3,
        target_rule: str = "geometry",
    ) -> None:
        self.target_rule = target_rule
        self.thresh1 = float(thresh1_km) * 1e3
        self.thresh2 = float(thresh2_km) * 1e3
        self.hys = float(hys_km) * 1e3
        self.ttt_steps = int(ttt_steps)
        self.prep_margin = float(prep_margin_km) * 1e3
        self.release_steps = int(release_steps)
        self._enter: np.ndarray | None = None
        self._leave: np.ndarray | None = None
        self._last_step = -1

    def _distance(self, env, uid: int, cell_id: int) -> float:
        position = env.satellites[cell_id].position_ecef(env.time_s)
        return _ground_distance_m(env.ues[uid].position, _sub_satellite_point(position))

    def __call__(self, env) -> np.ndarray:
        if self._enter is None or len(self._enter) != env.num_agents or env.step_index <= self._last_step:
            self._enter = np.zeros(env.num_agents, dtype=np.int64)
            self._leave = np.zeros(env.num_agents, dtype=np.int64)
        self._last_step = env.step_index
        actions = np.zeros(env.num_agents, dtype=np.int64)
        assigned: dict[int, int] = {}
        for ue in env.ues:
            uid, primary = ue.ue_id, ue.primary_cell
            if primary is None:
                continue
            ml1 = self._distance(env, uid, primary)
            if ue.state == MobilityState.SINGLE:
                self._enter[uid] = self._leave[uid] = 0
                candidates = list(env.candidates[uid])
                if candidates and ml1 - self.hys > self.thresh1 - self.prep_margin:
                    best = _pick_target(
                        env, candidates, lambda c: -self._distance(env, uid, c),
                        lambda c: self._distance(env, uid, c) + self.hys < self.thresh2,
                        self.target_rule, assigned)
                    assigned[candidates[best]] = assigned.get(candidates[best], 0) + 1
                    actions[uid] = env.prepare_start + best
            elif ue.state == MobilityState.PREPARED:
                target = ue.secondary_cell
                if target is None:
                    continue
                ml2 = self._distance(env, uid, target)
                entering = (ml1 - self.hys > self.thresh1) and (ml2 + self.hys < self.thresh2)
                leaving = (ml1 + self.hys < self.thresh1) or (ml2 - self.hys > self.thresh2)
                self._enter[uid] = self._enter[uid] + 1 if entering else 0
                self._leave[uid] = self._leave[uid] + 1 if leaving else 0
                if self._enter[uid] >= self.ttt_steps:
                    actions[uid] = env.commit_action
                elif self._leave[uid] >= self.release_steps and ml1 + self.hys < self.thresh1 - self.prep_margin:
                    actions[uid] = env.release_action
            else:
                self._enter[uid] = self._leave[uid] = 0
        return _mask_invalid(env, actions)


def _pick_target(env, candidates: list[int], score, eligible, target_rule: str, assigned: dict[int, int]) -> int:
    """Index (within candidates) of the CHO target.

    geometry: best geometric score; every UE of the cell gets the same target.
    load:     network-side load-aware assignment. Among candidates that satisfy
              the event's geometric condition (eligible), choose the satellite with
              the lowest load known to the network (last realised pressure plus
              the share of UEs already assigned to it in this epoch). Falls back
              to the geometric choice when no candidate is eligible.
    """
    scores = [score(c) for c in candidates]
    best = int(np.argmax(scores))
    if target_rule == "geometry" or len(candidates) == 1:
        return best
    usable = [k for k, c in enumerate(candidates) if eligible(c)]
    if not usable:
        return best

    def load(k: int) -> float:
        c = candidates[k]
        return float(env.last_requested_pressure[c]) + assigned.get(c, 0) / max(env.num_agents, 1)

    return min(usable, key=load)


def _mask_invalid(env, actions: np.ndarray) -> np.ndarray:
    masks = env.action_masks()
    for uid in range(env.num_agents):
        if masks[uid, int(actions[uid])] <= 0.0:
            actions[uid] = 0
    return actions
