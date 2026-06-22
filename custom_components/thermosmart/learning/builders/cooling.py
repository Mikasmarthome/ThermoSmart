"""PassiveCoolingEpisodeBuilder (Phase 6, pure Python)."""
from __future__ import annotations

from ..contracts import Regime
from ..episode_schemas import EpisodeType, PassiveCoolingEpisode
from .base import (
    BuilderAction,
    BuilderInput,
    BuilderResult,
    EpisodeBuilder,
    ReasonCode,
)


class PassiveCoolingEpisodeBuilder(EpisodeBuilder):
    episode_type = EpisodeType.PASSIVE_COOLING

    def __init__(self, learning_zone_id, params=None) -> None:
        super().__init__(learning_zone_id, params)
        self.trajectory_cap = self._params.cooling_trajectory_cap

    def _step(self, inp: BuilderInput) -> BuilderResult:
        regime = inp.regime.regime if inp.regime else None

        if self._active:
            aborted = self._maybe_abort(inp)
            if aborted is not None:
                return aborted
            if regime in (Regime.STABLE, Regime.ACTIVE_HEATING, Regime.AFTERHEAT):
                self._accumulate(inp)
                return self._close(ReasonCode.CLOSED_TRANSITION, inp)
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

        if (regime is Regime.PASSIVE_COOLING and inp.regime is not None
                and inp.regime.reliability >= self._params.min_open_reliability
                and self._is_disturbance(inp) is None):
            self._open(inp, self.trajectory_cap)
            self._extra["relative_only"] = _no_outdoor(inp)
            reasons = (ReasonCode.OPENED.value,)
            if self._extra["relative_only"]:
                reasons = reasons + (ReasonCode.RELATIVE_COOLING_ONLY.value,)
            return BuilderResult(BuilderAction.OPENED, open_episode_id=self._episode_id,
                                 reason_codes=reasons)
        return BuilderResult(BuilderAction.NO_CHANGE)

    def _close(self, reason: ReasonCode, inp: BuilderInput) -> BuilderResult:
        bad = self._too_short_or_sparse(inp.ts)
        if bad is not None:
            return self._discard(bad)
        start_temp = self._resolved_start_temp()
        end_temp = self._traj.last_valid() if self._traj else None
        if start_temp is None or end_temp is None:
            return self._discard(ReasonCode.DISCARDED_INSUFFICIENT)
        from .base import BUILDER_VERSION
        episode = PassiveCoolingEpisode(
            episode_id=self._episode_id, learning_zone_id=self._zone,
            episode_schema_version=1, builder_version=BUILDER_VERSION,
            classifier_version=inp.regime.classifier_version if inp.regime else 1,
            start_ts=self._start_ts, end_ts=inp.ts, regime=Regime.PASSIVE_COOLING,
            reliability=self._episode_reliability(),
            start_temp=start_temp, end_temp=end_temp,
            trajectory=self._traj.build(self.trajectory_cap),
        )
        reasons = (reason,)
        if self._extra.get("relative_only"):
            reasons = reasons + (ReasonCode.RELATIVE_COOLING_ONLY,)
        return self._finalize(episode, reasons)


def _no_outdoor(inp: BuilderInput) -> bool:
    from ..contracts import DataQuality
    m = inp.outdoor_temp
    return not (m is not None and m.value is not None
                and m.quality in (DataQuality.OK, DataQuality.STALE))
