"""Phase 6 tests for AfterheatEpisodeBuilder."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement, Regime
from custom_components.thermosmart.learning.episode_schemas import AfterheatEpisode
from custom_components.thermosmart.learning.regime import ReasonCode as RR
from custom_components.thermosmart.learning.builders import (
    AfterheatEpisodeBuilder,
    BuilderAction,
    BuilderInput,
    ConfounderCode,
)
from tests.helpers_builders import at, regime


def b():
    return AfterheatEpisodeBuilder("lz_1")


def step(builder, minute, kind, temp, reliability=0.8, **kw):
    return builder.process(BuilderInput(ts=at(minute), regime=regime(kind, reliability),
                                        indoor_temp=temp, target=21.0,
                                        trv_setpoint=16.0, **kw))


class TestAfterheatLifecycle:
    def test_open_and_close_on_cooling(self):
        builder = b()
        assert step(builder, 0, Regime.AFTERHEAT, 20.6).action is BuilderAction.OPENED
        assert step(builder, 5, Regime.AFTERHEAT, 20.8).action is BuilderAction.CONTINUED
        r = step(builder, 10, Regime.PASSIVE_COOLING, 20.7)
        assert r.action is BuilderAction.CLOSED
        assert isinstance(r.episode, AfterheatEpisode)

    def test_without_valve_hvac_still_opens(self):
        builder = b()
        r = step(builder, 0, Regime.AFTERHEAT, 20.6)
        assert r.action is BuilderAction.OPENED
        r2 = step(builder, 5, Regime.AFTERHEAT, 20.8)
        r3 = step(builder, 10, Regime.STABLE, 20.7)
        assert r3.action is BuilderAction.CLOSED
        assert ConfounderCode.MISSING_OPTIONAL_EVIDENCE.value in r3.confounder_flags

    def test_reheating_aborts(self):
        builder = b()
        step(builder, 0, Regime.AFTERHEAT, 20.6)
        r = step(builder, 5, Regime.ACTIVE_HEATING, 20.9)
        assert r.action is BuilderAction.ABORTED
        assert ConfounderCode.REHEATING.value in r.confounder_flags

    def test_valve_reopen_aborts(self):
        builder = b()
        step(builder, 0, Regime.AFTERHEAT, 20.6)
        r = step(builder, 5, Regime.AFTERHEAT, 20.7,
                 valve_opening=Measurement(40.0, DataQuality.OK))
        assert r.action is BuilderAction.ABORTED

    def test_window_open_aborts(self):
        builder = b()
        step(builder, 0, Regime.AFTERHEAT, 20.6)
        r = builder.process(BuilderInput(ts=at(5), regime=regime(Regime.DISTURBED),
                                         indoor_temp=20.7, window_open=True))
        assert r.action is BuilderAction.ABORTED

    def test_solar_confounder_flagged(self):
        builder = b()
        step(builder, 0, Regime.AFTERHEAT, 20.6,
             reliability=0.6)
        r = builder.process(BuilderInput(
            ts=at(5),
            regime=regime(Regime.AFTERHEAT, 0.6, reason_codes=(RR.SOLAR_CONFOUNDER.value,)),
            indoor_temp=20.7, target=21.0, solar_radiation=Measurement(600.0, DataQuality.OK)))
        # continuing, but solar confounder must be recorded
        rc = step(builder, 10, Regime.STABLE, 20.6)
        assert ConfounderCode.SOLAR_GAIN.value in rc.confounder_flags

    def test_timeout_closes(self):
        builder = b()
        step(builder, 0, Regime.AFTERHEAT, 20.6)
        r = step(builder, 100, Regime.AFTERHEAT, 20.7)  # > 90 min
        assert r.action in (BuilderAction.CLOSED, BuilderAction.DISCARDED)

    def test_too_short_discarded(self):
        builder = b()
        step(builder, 0, Regime.AFTERHEAT, 20.6)
        r = step(builder, 1, Regime.STABLE, 20.6)
        assert r.action is BuilderAction.DISCARDED

    def test_high_gap_discarded(self):
        builder = b()
        step(builder, 0, Regime.AFTERHEAT, 20.6)
        # feed gaps (no indoor temp) then close
        builder.process(BuilderInput(ts=at(5), regime=regime(Regime.AFTERHEAT), target=21.0))
        builder.process(BuilderInput(ts=at(7), regime=regime(Regime.AFTERHEAT), target=21.0))
        r = builder.process(BuilderInput(ts=at(10), regime=regime(Regime.STABLE), target=21.0))
        assert r.action is BuilderAction.DISCARDED


class TestAfterheatReliability:
    def test_lower_regime_reliability_lowers_episode(self):
        hi = b()
        step(hi, 0, Regime.AFTERHEAT, 20.6, 0.9)
        step(hi, 5, Regime.AFTERHEAT, 20.8, 0.9)
        r_hi = step(hi, 10, Regime.STABLE, 20.7, 0.9)
        lo = b()
        step(lo, 0, Regime.AFTERHEAT, 20.6, 0.5)
        step(lo, 5, Regime.AFTERHEAT, 20.8, 0.5)
        r_lo = step(lo, 10, Regime.STABLE, 20.7, 0.5)
        assert r_lo.reliability < r_hi.reliability
        assert 0.0 <= r_lo.reliability <= 1.0


class TestSnapshotRestore:
    def test_restore_continues(self):
        a = b()
        step(a, 0, Regime.AFTERHEAT, 20.6)
        step(a, 5, Regime.AFTERHEAT, 20.8)
        snap = a.snapshot_state()
        restored = AfterheatEpisodeBuilder("lz_1")
        restored.restore_state(snap)
        r = step(restored, 10, Regime.STABLE, 20.7)
        assert r.action is BuilderAction.CLOSED
