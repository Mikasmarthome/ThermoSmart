"""Phase 6 tests for HeatingEpisodeBuilder + drive-end detection + snapshot."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement, Regime
from custom_components.thermosmart.learning.episode_schemas import HeatingEpisode
from custom_components.thermosmart.learning.raw_schemas import DriveEndSource, HeatingDriveEndEvent
from custom_components.thermosmart.learning.builders import (
    BuilderAction,
    BuilderInput,
    BuilderStateError,
    HeatingEpisodeBuilder,
)
from custom_components.thermosmart.learning.builders.drive_end import (
    DriveEndSignals,
    detect_drive_end,
)
from tests.helpers_builders import at, regime


def b():
    return HeatingEpisodeBuilder("lz_1")


def step(builder, minute, kind, temp, **kw):
    return builder.process(BuilderInput(ts=at(minute), regime=regime(kind), indoor_temp=temp,
                                        target=21.0, **kw))


class TestHeatingLifecycle:
    def test_open_continue_close(self):
        builder = b()
        assert step(builder, 0, Regime.ACTIVE_HEATING, 18.0).action is BuilderAction.OPENED
        assert step(builder, 5, Regime.ACTIVE_HEATING, 18.3).action is BuilderAction.CONTINUED
        r = step(builder, 10, Regime.STABLE, 18.6)
        assert r.action is BuilderAction.CLOSED
        assert isinstance(r.episode, HeatingEpisode)
        assert 0.0 <= r.reliability <= 1.0

    def test_trv_only_open(self):
        builder = b()
        r = builder.process(BuilderInput(ts=at(0), regime=regime(Regime.ACTIVE_HEATING),
                                         indoor_temp=18.0, target=21.0,
                                         trv_setpoint=22.0, controller_demand=True))
        assert r.action is BuilderAction.OPENED

    def test_close_on_drive_end(self):
        builder = b()
        step(builder, 0, Regime.ACTIVE_HEATING, 18.0)
        step(builder, 5, Regime.ACTIVE_HEATING, 18.5)
        ev = HeatingDriveEndEvent(learning_zone_id="lz_1", ts=at(10), event_id="de1",
                                  source=DriveEndSource.VALVE_CLOSING, source_confidence=0.9)
        r = builder.process(BuilderInput(ts=at(10), regime=regime(Regime.ACTIVE_HEATING),
                                         indoor_temp=18.7, target=21.0, drive_end_event=ev))
        assert r.action is BuilderAction.CLOSED

    def test_disturbed_aborts(self):
        builder = b()
        step(builder, 0, Regime.ACTIVE_HEATING, 18.0)
        r = builder.process(BuilderInput(ts=at(5), regime=regime(Regime.DISTURBED),
                                         indoor_temp=18.2, heating_failure=True))
        assert r.action is BuilderAction.ABORTED
        assert "heating_failure" in r.confounder_flags

    def test_short_unknown_pauses(self):
        builder = b()
        step(builder, 0, Regime.ACTIVE_HEATING, 18.0)
        r = step(builder, 5, Regime.UNKNOWN, 18.1)
        assert r.action is BuilderAction.CONTINUED

    def test_long_unknown_closes(self):
        builder = b()
        step(builder, 0, Regime.ACTIVE_HEATING, 18.0)
        step(builder, 5, Regime.UNKNOWN, 18.1)
        r = step(builder, 20, Regime.UNKNOWN, 18.1)  # > max_unknown 600s
        assert r.action in (BuilderAction.CLOSED, BuilderAction.DISCARDED)

    def test_too_short_discarded(self):
        builder = b()
        step(builder, 0, Regime.ACTIVE_HEATING, 18.0)
        r = step(builder, 1, Regime.STABLE, 18.1)  # < min_duration 180s
        assert r.action is BuilderAction.DISCARDED

    def test_duplicate_event_no_double_open(self):
        builder = b()
        ev = BuilderInput(ts=at(0), regime=regime(Regime.ACTIVE_HEATING), indoor_temp=18.0,
                          target=21.0, event_id="e1")
        assert builder.process(ev).action is BuilderAction.OPENED
        r = builder.process(ev)  # same event id replayed
        assert r.action is BuilderAction.NO_CHANGE
        assert "duplicate_event" in r.reason_codes

    def test_late_event_rejected(self):
        builder = b()
        step(builder, 5, Regime.ACTIVE_HEATING, 18.0)
        r = step(builder, 1, Regime.ACTIVE_HEATING, 18.1)  # earlier ts
        assert r.action is BuilderAction.NO_CHANGE
        assert "late_event" in r.reason_codes


class TestMultiTrv:
    def test_multiple_trvs_feed_one_zone_episode(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), regime=regime(Regime.ACTIVE_HEATING),
                                     indoor_temp=18.0, target=21.0, trv_binding_id="trv_a"))
        builder.process(BuilderInput(ts=at(5), regime=regime(Regime.ACTIVE_HEATING),
                                     indoor_temp=18.3, target=21.0, trv_binding_id="trv_b"))
        r = builder.process(BuilderInput(ts=at(10), regime=regime(Regime.STABLE),
                                         indoor_temp=18.6, target=21.0, trv_binding_id="trv_a"))
        # one zone episode, not one per TRV
        assert r.action is BuilderAction.CLOSED
        assert r.episode.learning_zone_id == "lz_1"


class TestSnapshotRestore:
    def test_snapshot_restore_continues_identically(self):
        a = b()
        step(a, 0, Regime.ACTIVE_HEATING, 18.0)
        step(a, 5, Regime.ACTIVE_HEATING, 18.3)
        snap = a.snapshot_state()

        restored = HeatingEpisodeBuilder("lz_1")
        restored.restore_state(snap)
        r_a = step(a, 10, Regime.STABLE, 18.6)
        r_b = step(restored, 10, Regime.STABLE, 18.6)
        assert r_a.action is r_b.action is BuilderAction.CLOSED
        assert r_a.episode.episode_id == r_b.episode.episode_id  # no duplicate id

    def test_version_mismatch_rejected(self):
        builder = b()
        snap = builder.snapshot_state()
        snap["builder_version"] = 999
        with pytest.raises(BuilderStateError):
            HeatingEpisodeBuilder("lz_1").restore_state(snap)

    def test_corrupted_snapshot_rejected(self):
        with pytest.raises(BuilderStateError):
            HeatingEpisodeBuilder("lz_1").restore_state({"builder_version": 1})

    def test_zone_isolation(self):
        builder = b()
        snap = builder.snapshot_state()
        with pytest.raises(BuilderStateError):
            HeatingEpisodeBuilder("lz_OTHER").restore_state(snap)


class TestDriveEndDetection:
    def test_valve_closing_direct(self):
        ev = detect_drive_end(DriveEndSignals(
            ts=at(0), learning_zone_id="lz_1", event_id="d",
            valve_before=Measurement(60.0, DataQuality.OK),
            valve_after=Measurement(0.0, DataQuality.OK)))
        assert ev.source is DriveEndSource.VALVE_CLOSING and ev.source_confidence > 0.8

    def test_setpoint_reduction_trv_only(self):
        ev = detect_drive_end(DriveEndSignals(
            ts=at(0), learning_zone_id="lz_1", event_id="d",
            setpoint_before=22.0, setpoint_after=16.0))
        assert ev.source is DriveEndSource.SETPOINT_REDUCTION

    def test_inferred_demand_off(self):
        ev = detect_drive_end(DriveEndSignals(
            ts=at(0), learning_zone_id="lz_1", event_id="d",
            demand_before=True, demand_after=False))
        assert ev.source is DriveEndSource.INFERRED_HEAT_INPUT_END
        assert ev.source_confidence < 0.5

    def test_direct_beats_inferred_when_merged(self):
        ev = detect_drive_end(DriveEndSignals(
            ts=at(0), learning_zone_id="lz_1", event_id="d",
            valve_before=Measurement(60.0, DataQuality.OK),
            valve_after=Measurement(0.0, DataQuality.OK),
            demand_before=True, demand_after=False))
        assert ev.source is DriveEndSource.VALVE_CLOSING

    def test_pure_temperature_change_no_drive_end(self):
        assert detect_drive_end(DriveEndSignals(
            ts=at(0), learning_zone_id="lz_1", event_id="d")) is None
