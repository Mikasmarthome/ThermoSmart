"""AfterheatEpisodeBuilder (Phase 6, pure Python)."""
from __future__ import annotations

from ..contracts import Regime
from ..episode_schemas import AfterheatEpisode, EpisodeType
from .base import (
    BuilderAction,
    BuilderInput,
    BuilderResult,
    ConfounderCode,
    EpisodeBuilder,
    ReasonCode,
)


class AfterheatEpisodeBuilder(EpisodeBuilder):
    episode_type = EpisodeType.AFTERHEAT

    def __init__(self, learning_zone_id, params=None) -> None:
        super().__init__(learning_zone_id, params)
        self.trajectory_cap = self._params.afterheat_trajectory_cap

    def _step(self, inp: BuilderInput) -> BuilderResult:
        regime = inp.regime.regime if inp.regime else None

        if self._active:
            aborted = self._maybe_abort(inp)
            if aborted is not None:
                return aborted
            # renewed active heating ends afterheat; try graceful close first so that
            # morning-boost episodes (which peak and then cool until TPI restarts for
            # maintenance) are not simply discarded.  The confounder is added ONLY if
            # the close fails (episode too short) — a successful close means the heating
            # happened AFTER the episode, not during, so it is not a contamination event.
            if regime is Regime.ACTIVE_HEATING or inp.controller_demand is True \
                    or _valve_open(inp):
                candidate = self._close(ReasonCode.CLOSED_PEAK_COOLING, inp)
                if candidate.action is BuilderAction.CLOSED:
                    return candidate
                self._confounders.add(ConfounderCode.REHEATING)
                return self._abort(ReasonCode.ABORTED_REHEATING)
            # peak reached and cooling begins -> close
            if regime in (Regime.PASSIVE_COOLING, Regime.STABLE):
                self._accumulate(inp)
                return self._close(ReasonCode.CLOSED_PEAK_COOLING, inp)
            if regime is Regime.UNKNOWN:
                if self._unknown_since is None:
                    self._unknown_since = inp.ts
                elif (inp.ts - self._unknown_since).total_seconds() > \
                        self._params.max_unknown_duration_s:
                    self._accumulate(inp)
                    return self._close(ReasonCode.UNKNOWN_TIMEOUT, inp)
                self._accumulate(inp)
                return BuilderResult(BuilderAction.CONTINUED, open_episode_id=self._episode_id,
                                     reason_codes=(ReasonCode.UNKNOWN_PAUSE.value,))
            self._unknown_since = None
            self._accumulate(inp)
            if self._episode_duration_s(inp.ts) > self._params.max_episode_duration_s:
                return self._close(ReasonCode.CLOSED_MAX_DURATION, inp)
            return BuilderResult(BuilderAction.CONTINUED, open_episode_id=self._episode_id,
                                 reason_codes=(ReasonCode.CONTINUED.value,))

        if (regime is Regime.AFTERHEAT and inp.regime is not None
                and inp.regime.reliability >= self._params.min_open_reliability
                and self._is_disturbance(inp) is None):
            self._open(inp, self.trajectory_cap)
            # Prefer the captured boost setpoint (from before coordinator reduced to comfort)
            # so that setpoint_before > setpoint_after enables drive-end inference.
            self._extra["setpoint_before"] = (
                inp.drive_end_setpoint_c if inp.drive_end_setpoint_c is not None
                else inp.trv_setpoint
            )
            return BuilderResult(BuilderAction.OPENED, open_episode_id=self._episode_id,
                                 reason_codes=(ReasonCode.OPENED.value,))
        return BuilderResult(BuilderAction.NO_CHANGE)

    def _close(self, reason: ReasonCode, inp: BuilderInput) -> BuilderResult:
        bad = self._too_short_or_sparse(inp.ts)
        if bad is not None:
            return self._discard(bad)
        close_temp = self._resolved_start_temp()
        target = self._extra.get("target", inp.target)
        if close_temp is None or target is None:
            return self._discard(ReasonCode.DISCARDED_INSUFFICIENT)
        from .base import BUILDER_VERSION
        episode = AfterheatEpisode(
            episode_id=self._episode_id, learning_zone_id=self._zone,
            episode_schema_version=1, builder_version=BUILDER_VERSION,
            classifier_version=inp.regime.classifier_version if inp.regime else 1,
            start_ts=self._start_ts, end_ts=inp.ts, regime=Regime.AFTERHEAT,
            reliability=self._episode_reliability(),
            indoor_temp_at_close=close_temp, target=target,
            trv_setpoint_before=self._extra.get("setpoint_before") or 0.0,
            trv_setpoint_after=inp.trv_setpoint if inp.trv_setpoint is not None else 0.0,
            trajectory=self._traj.build(self.trajectory_cap),
            trv_binding_id=inp.trv_binding_id,
        )
        return self._finalize(episode, (reason,))


def _valve_open(inp: BuilderInput) -> bool:
    m = inp.valve_opening
    from ..contracts import DataQuality
    return (m is not None and m.value is not None and m.value > 0
            and m.quality in (DataQuality.OK, DataQuality.STALE))
