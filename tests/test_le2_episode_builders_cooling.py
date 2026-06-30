"""Phase 6 tests for PassiveCoolingEpisodeBuilder and WindowCoolingEpisodeBuilder."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement, Regime
from custom_components.thermosmart.learning.episode_schemas import (
    PassiveCoolingEpisode,
    WindowCoolingEpisode,
)
from custom_components.thermosmart.learning.builders import (
    BuilderAction,
    BuilderInput,
    PassiveCoolingEpisodeBuilder,
    ReasonCode,
    WindowCoolingEpisodeBuilder,
)
from tests.helpers_builders import at, regime


def cool():
    return PassiveCoolingEpisodeBuilder("lz_1")


def cstep(builder, minute, kind, temp, **kw):
    return builder.process(BuilderInput(ts=at(minute), regime=regime(kind), indoor_temp=temp,
                                        target=21.0, controller_demand=False, **kw))


class TestPassiveCooling:
    def test_open_close(self):
        builder = cool()
        assert cstep(builder, 0, Regime.PASSIVE_COOLING, 20.0).action is BuilderAction.OPENED
        cstep(builder, 5, Regime.PASSIVE_COOLING, 19.8)
        r = cstep(builder, 10, Regime.STABLE, 19.6)
        assert r.action is BuilderAction.CLOSED
        assert isinstance(r.episode, PassiveCoolingEpisode)

    def test_relative_without_outdoor(self):
        builder = cool()
        r = cstep(builder, 0, Regime.PASSIVE_COOLING, 20.0)
        assert ReasonCode.RELATIVE_COOLING_ONLY.value in r.reason_codes

    def test_window_open_aborts(self):
        builder = cool()
        cstep(builder, 0, Regime.PASSIVE_COOLING, 20.0)
        r = builder.process(BuilderInput(ts=at(5), regime=regime(Regime.DISTURBED),
                                         indoor_temp=19.8, window_open=True))
        assert r.action is BuilderAction.ABORTED

    def test_heating_transition_closes(self):
        builder = cool()
        cstep(builder, 0, Regime.PASSIVE_COOLING, 20.0)
        cstep(builder, 5, Regime.PASSIVE_COOLING, 19.8)
        r = cstep(builder, 10, Regime.ACTIVE_HEATING, 19.9)
        assert r.action is BuilderAction.CLOSED

    def test_short_unknown_pauses(self):
        builder = cool()
        cstep(builder, 0, Regime.PASSIVE_COOLING, 20.0)
        r = cstep(builder, 5, Regime.UNKNOWN, 19.9)
        assert r.action is BuilderAction.CONTINUED

    def test_disturbed_jump_aborts(self):
        builder = cool()
        cstep(builder, 0, Regime.PASSIVE_COOLING, 20.0)
        r = builder.process(BuilderInput(
            ts=at(5),
            regime=regime(Regime.DISTURBED, disturbance_reasons=("implausible_jump",)),
            indoor_temp=25.0))
        assert r.action is BuilderAction.ABORTED


class TestWindowCooling:
    def _b(self):
        return WindowCoolingEpisodeBuilder("lz_1")

    def test_open_on_window_open_close_on_close(self):
        builder = self._b()
        r0 = builder.process(BuilderInput(ts=at(0), indoor_temp=21.0, window_open=True))
        assert r0.action is BuilderAction.OPENED
        builder.process(BuilderInput(ts=at(2), indoor_temp=20.2, window_open=True))
        r = builder.process(BuilderInput(ts=at(5), indoor_temp=19.4, window_open=False))
        assert r.action is BuilderAction.CLOSED
        assert isinstance(r.episode, WindowCoolingEpisode)
        # no derived values stored
        assert not hasattr(r.episode, "temp_drop")
        assert not hasattr(r.episode, "cooling_rate_per_min")

    def test_no_sensor_inactive(self):
        builder = self._b()
        r = builder.process(BuilderInput(ts=at(0), indoor_temp=21.0))  # window_open None
        assert r.action is BuilderAction.NO_CHANGE

    def test_missing_close_times_out(self):
        builder = self._b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=21.0, window_open=True))
        r = builder.process(BuilderInput(ts=at(200), indoor_temp=18.0, window_open=True))
        assert r.action is BuilderAction.ABORTED
        assert ReasonCode.ABORTED_NO_CLOSE.value in r.reason_codes

    def test_heating_failure_aborts(self):
        builder = self._b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=21.0, window_open=True))
        r = builder.process(BuilderInput(ts=at(2), indoor_temp=20.0, window_open=True,
                                         heating_failure=True))
        assert r.action is BuilderAction.ABORTED

    def test_temp_missing_at_close_discarded(self):
        builder = self._b()
        builder.process(BuilderInput(ts=at(0), window_open=True))  # no temp at open
        r = builder.process(BuilderInput(ts=at(5), window_open=False))  # no temp at close
        assert r.action is BuilderAction.DISCARDED
