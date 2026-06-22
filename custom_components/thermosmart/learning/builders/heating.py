"""HeatingEpisodeBuilder (Phase 6, pure Python)."""
from __future__ import annotations

from typing import Optional

from ..contracts import Regime
from ..episode_schemas import EpisodeType, HeatingEpisode
from .base import (
    BuilderAction,
    BuilderInput,
    BuilderResult,
    EpisodeBuilder,
    ReasonCode,
)


class HeatingEpisodeBuilder(EpisodeBuilder):
    episode_type = EpisodeType.HEATING

    def __init__(self, learning_zone_id, params=None) -> None:
        super().__init__(learning_zone_id, params)
        self.trajectory_cap = self._params.heating_trajectory_cap

    def _step(self, inp: BuilderInput) -> BuilderResult:
        regime = inp.regime.regime if inp.regime else None

        if self._active:
            aborted = self._maybe_abort(inp)
            if aborted is not None:
                return aborted

            # close on an explicit drive-end event
            if inp.drive_end_event is not None:
                self._accumulate(inp)
                return self._close(ReasonCode.CLOSED_DRIVE_END, inp)

            # transition out of active heating -> close
            if regime in (Regime.AFTERHEAT, Regime.PASSIVE_COOLING, Regime.STABLE):
                self._accumulate(inp)
                return self._close(ReasonCode.CLOSED_TRANSITION, inp)

            # short UNKNOWN tolerance, then close
            if regime is Regime.UNKNOWN:
                if self._unknown_since is None:
                    self._unknown_since = inp.ts
                elif (inp.ts - self._unknown_since).total_seconds() > \
                        self._params.max_unknown_duration_s:
                    self._accumulate(inp)
                    return self._close(ReasonCode.UNKNOWN_TIMEOUT, inp)
                self._accumulate(inp)
                return BuilderResult(BuilderAction.CONTINUED,
                                     open_episode_id=self._episode_id,
                                     reason_codes=(ReasonCode.UNKNOWN_PAUSE.value,))

            self._unknown_since = None
            self._accumulate(inp)

            if self._episode_duration_s(inp.ts) > self._params.max_episode_duration_s:
                return self._close(ReasonCode.CLOSED_MAX_DURATION, inp)
            return BuilderResult(BuilderAction.CONTINUED, open_episode_id=self._episode_id,
                                 reason_codes=(ReasonCode.CONTINUED.value,))

        # not active: open only on a reliable ACTIVE_HEATING classification
        if (regime is Regime.ACTIVE_HEATING and inp.regime is not None
                and inp.regime.reliability >= self._params.min_open_reliability
                and self._is_disturbance(inp) is None):
            self._open(inp, self.trajectory_cap)
            return BuilderResult(BuilderAction.OPENED, open_episode_id=self._episode_id,
                                 reason_codes=(ReasonCode.OPENED.value,))
        return BuilderResult(BuilderAction.NO_CHANGE)

    def _close(self, reason: ReasonCode, inp: BuilderInput) -> BuilderResult:
        bad = self._too_short_or_sparse(inp.ts)
        if bad is not None:
            return self._discard(bad)
        start_temp = self._resolved_start_temp()
        target = inp.target if inp.target is not None else self._extra.get("target")
        if start_temp is None or target is None:
            return self._discard(ReasonCode.DISCARDED_INSUFFICIENT)
        episode = HeatingEpisode(
            episode_id=self._episode_id, learning_zone_id=self._zone,
            episode_schema_version=1, builder_version=self._builder_version(),
            classifier_version=self._classifier_version(inp),
            start_ts=self._start_ts, end_ts=inp.ts, regime=Regime.ACTIVE_HEATING,
            reliability=self._episode_reliability(),
            start_temp=start_temp, target=target,
            trajectory=self._traj.build(self.trajectory_cap),
            controller=None, trv_binding_id=inp.trv_binding_id,
        )
        return self._finalize(episode, (reason,))

    def _builder_version(self) -> int:
        from .base import BUILDER_VERSION
        return BUILDER_VERSION

    def _classifier_version(self, inp: BuilderInput) -> int:
        return inp.regime.classifier_version if inp.regime else 1
