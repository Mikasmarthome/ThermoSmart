"""WindowCoolingEpisodeBuilder (Phase 6, pure Python).

Driven by window-open state, not by regime. Without a window sensor it simply
stays inactive — it never invents a window-cooling episode and never blocks a
TRV-only setup.
"""
from __future__ import annotations

from ..contracts import Regime
from ..episode_schemas import EpisodeType, WindowCoolingEpisode
from .base import (
    BuilderAction,
    BuilderInput,
    BuilderResult,
    EpisodeBuilder,
    ReasonCode,
)


class WindowCoolingEpisodeBuilder(EpisodeBuilder):
    episode_type = EpisodeType.WINDOW_COOLING

    def __init__(self, learning_zone_id, params=None) -> None:
        super().__init__(learning_zone_id, params)
        self.trajectory_cap = self._params.window_trajectory_cap

    def _step(self, inp: BuilderInput) -> BuilderResult:
        if self._active:
            if inp.heating_failure:
                return self._abort(ReasonCode.ABORTED_HEATING_FAILURE)
            if inp.window_open is False:
                self._accumulate(inp)
                return self._close(inp)
            if self._episode_duration_s(inp.ts) > self._params.max_episode_duration_s:
                # no close event within the limit -> aborted, not invented
                confounders = tuple(sorted(c.value for c in self._confounders))
                self._reset_episode()
                return BuilderResult(BuilderAction.ABORTED,
                                     reason_codes=(ReasonCode.ABORTED_NO_CLOSE.value,),
                                     confounder_flags=confounders)
            self._accumulate(inp)
            return BuilderResult(BuilderAction.CONTINUED, open_episode_id=self._episode_id,
                                 reason_codes=(ReasonCode.CONTINUED.value,))

        # open only on a real window-open state (sensor required)
        if inp.window_open is True:
            self._open(inp, self.trajectory_cap)
            return BuilderResult(BuilderAction.OPENED, open_episode_id=self._episode_id,
                                 reason_codes=(ReasonCode.OPENED.value,))
        return BuilderResult(BuilderAction.NO_CHANGE)

    def _close(self, inp: BuilderInput) -> BuilderResult:
        temp_at_open = self._resolved_start_temp()
        temp_at_close = self._traj.last_valid() if self._traj else None
        if temp_at_open is None or temp_at_close is None:
            return self._discard(ReasonCode.DISCARDED_INSUFFICIENT)
        from .base import BUILDER_VERSION
        episode = WindowCoolingEpisode(
            episode_id=self._episode_id, learning_zone_id=self._zone,
            episode_schema_version=1, builder_version=BUILDER_VERSION,
            classifier_version=inp.regime.classifier_version if inp.regime else 1,
            start_ts=self._start_ts, end_ts=inp.ts, regime=Regime.DISTURBED,
            reliability=self._episode_reliability(),
            temp_at_open=temp_at_open, temp_at_close=temp_at_close,
            trajectory=self._traj.build(self.trajectory_cap),
        )
        return self._finalize(episode, (ReasonCode.CLOSED_WINDOW,))
