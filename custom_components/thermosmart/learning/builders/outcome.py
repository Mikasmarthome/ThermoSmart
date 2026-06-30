"""OutcomeEpisodeBuilder (Phase 6, pure Python).

Lifecycle of a control decision whose result is later scored. The score itself
is NOT computed or stored here; only the raw boundaries, capped trajectory and
labels are produced. May span heating and afterheat, since it judges the whole
decision.
"""
from __future__ import annotations

from typing import Optional

from ..contracts import Regime
from ..episode_schemas import ControllerKind, EpisodeReason, EpisodeType, OutcomeEpisode
from .base import (
    BuilderAction,
    BuilderInput,
    BuilderResult,
    ConfounderCode,
    EpisodeBuilder,
    OutcomeReason,
    ReasonCode,
)

# The primary episode reason expresses the actual close cause directly.
_EPISODE_REASON = {
    OutcomeReason.REACHED: EpisodeReason.REACHED,
    OutcomeReason.TIMEOUT: EpisodeReason.TIMEOUT,
    OutcomeReason.INTERRUPTED: EpisodeReason.INTERRUPTED,
    OutcomeReason.MANUAL_CORRECTION: EpisodeReason.MANUAL_CORRECTION,
    OutcomeReason.SUPERSEDED: EpisodeReason.SUPERSEDED,
    OutcomeReason.MODE_CHANGE: EpisodeReason.MODE_CHANGE,
}


class OutcomeEpisodeBuilder(EpisodeBuilder):
    episode_type = EpisodeType.OUTCOME

    def __init__(self, learning_zone_id, params=None) -> None:
        super().__init__(learning_zone_id, params)
        self.trajectory_cap = self._params.outcome_trajectory_cap

    def _step(self, inp: BuilderInput) -> BuilderResult:
        if not self._active:
            if inp.decision_start is not None:
                self._open(inp, self.trajectory_cap)
                d = inp.decision_start
                self._extra.update({
                    "decision_id": d.decision_id, "o_target": d.target,
                    "tolerance": d.comfort_tolerance_at_start,
                    "controller": d.controller.value,
                    "expected": d.expected_minutes,
                })
                if d.start_temp is not None and self._start_temp is None:
                    self._start_temp = d.start_temp
                return BuilderResult(BuilderAction.OPENED, open_episode_id=self._episode_id,
                                     reason_codes=(ReasonCode.OPENED.value,))
            return BuilderResult(BuilderAction.NO_CHANGE)

        # active outcome
        if inp.heating_failure or (inp.regime is not None
                                   and inp.regime.regime is Regime.DISTURBED):
            self._confounders.add(ConfounderCode.HEATING_FAILURE if inp.heating_failure
                                  else ConfounderCode.WINDOW_OPEN)
            return self._abort_outcome(OutcomeReason.DISTURBED)

        self._accumulate(inp)

        if inp.decision_superseded:
            return self._close(OutcomeReason.SUPERSEDED, inp)
        if inp.manual_correction:
            self._confounders.add(ConfounderCode.MANUAL_CORRECTION)
            return self._close(OutcomeReason.MANUAL_CORRECTION, inp)
        if inp.mode_change:
            self._confounders.add(ConfounderCode.MODE_CHANGE)
            return self._close(OutcomeReason.MODE_CHANGE, inp)

        # reached: within tolerance band, held for the stability window
        target = self._extra["o_target"]
        tol = self._extra["tolerance"]
        if inp.indoor_temp is not None and inp.indoor_temp >= target - tol:
            if self._extra.get("reached_since") is None:
                self._extra["reached_since"] = inp.ts.isoformat()
            else:
                from datetime import datetime
                since = datetime.fromisoformat(self._extra["reached_since"])
                if (inp.ts - since).total_seconds() >= self._params.outcome_stability_window_s:
                    return self._close(OutcomeReason.REACHED, inp)
        else:
            self._extra["reached_since"] = None

        if self._episode_duration_s(inp.ts) > self._params.outcome_timeout_s:
            return self._close(OutcomeReason.TIMEOUT, inp)

        return BuilderResult(BuilderAction.CONTINUED, open_episode_id=self._episode_id,
                             reason_codes=(ReasonCode.CONTINUED.value,))

    def _abort_outcome(self, reason: OutcomeReason) -> BuilderResult:
        confounders = tuple(sorted(c.value for c in self._confounders))
        self._reset_episode()
        return BuilderResult(BuilderAction.ABORTED, reason_codes=(reason.value,),
                             confounder_flags=confounders)

    def _close(self, reason: OutcomeReason, inp: BuilderInput,
               extra: tuple[str, ...] = ()) -> BuilderResult:
        start_temp = self._resolved_start_temp()
        end_temp = self._traj.last_valid() if self._traj else None
        if start_temp is None or end_temp is None:
            confounders = tuple(sorted(c.value for c in self._confounders))
            self._reset_episode()
            return BuilderResult(BuilderAction.DISCARDED,
                                 reason_codes=(OutcomeReason.INSUFFICIENT_DATA.value,),
                                 confounder_flags=confounders)
        from .base import BUILDER_VERSION
        episode = OutcomeEpisode(
            episode_id=self._episode_id, learning_zone_id=self._zone,
            decision_id=self._extra["decision_id"],
            episode_schema_version=1, builder_version=BUILDER_VERSION,
            classifier_version=inp.regime.classifier_version if inp.regime else 1,
            start_ts=self._start_ts, end_ts=inp.ts, regime=Regime.ACTIVE_HEATING,
            reliability=self._episode_reliability(),
            start_temp=start_temp, end_temp=end_temp, target=self._extra["o_target"],
            comfort_tolerance_at_start=self._extra["tolerance"],
            reason=_EPISODE_REASON[reason],
            controller=ControllerKind(self._extra["controller"]),
            trajectory=self._traj.build(self.trajectory_cap),
        )
        eid = self._episode_id
        reliability = self._episode_reliability()
        confounders = tuple(sorted(c.value for c in self._confounders))
        capped = self._traj.capped if self._traj else False
        self._reset_episode()
        codes = (reason.value,) + extra
        if capped:
            codes = codes + (ReasonCode.TRAJECTORY_CAPPED.value,)
        return BuilderResult(BuilderAction.CLOSED, episode=episode, open_episode_id=eid,
                             reliability=reliability, reason_codes=codes,
                             confounder_flags=confounders)
