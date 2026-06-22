"""Phase 6 tests for OutcomeEpisodeBuilder."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.episode_schemas import (
    ControllerKind,
    EpisodeReason,
    OutcomeEpisode,
)
from custom_components.thermosmart.learning.builders import (
    BuilderAction,
    BuilderInput,
    BuilderParameters,
    ConfounderCode,
    DecisionContext,
    OutcomeEpisodeBuilder,
    OutcomeReason,
)
from tests.helpers_builders import at, regime


def decision(start_temp=18.0):
    return DecisionContext(decision_id="d1", target=21.0, comfort_tolerance_at_start=0.5,
                           controller=ControllerKind.THERMOSMART, expected_minutes=40,
                           start_temp=start_temp)


def b(params=None):
    return OutcomeEpisodeBuilder("lz_1", params)


def cont(builder, minute, temp, **kw):
    return builder.process(BuilderInput(ts=at(minute), regime=regime(Regime.ACTIVE_HEATING),
                                        indoor_temp=temp, target=21.0, **kw))


class TestOutcomeLifecycle:
    def test_open_on_decision(self):
        builder = b()
        r = builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        assert r.action is BuilderAction.OPENED

    def test_reached_after_stability_window(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        cont(builder, 5, 20.0)
        cont(builder, 10, 21.0)   # reached -> stability timer starts
        r = cont(builder, 21, 21.0)  # > 600s later, still in band
        assert r.action is BuilderAction.CLOSED
        assert isinstance(r.episode, OutcomeEpisode)
        assert r.episode.reason is EpisodeReason.REACHED

    def test_timeout(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        r = cont(builder, 100, 19.0)  # never reaches, > 90 min
        assert r.action is BuilderAction.CLOSED
        assert r.episode.reason is EpisodeReason.TIMEOUT

    def test_mode_change_precise_reason(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        r = cont(builder, 5, 19.0, mode_change=True)
        assert r.action is BuilderAction.CLOSED
        assert r.episode.reason is EpisodeReason.MODE_CHANGE  # primary reason, not generic
        assert "mode_change" in r.reason_codes

    def test_manual_correction(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        r = cont(builder, 5, 19.0, manual_correction=True)
        assert r.action is BuilderAction.CLOSED
        assert r.episode.reason is EpisodeReason.MANUAL_CORRECTION
        assert ConfounderCode.MANUAL_CORRECTION.value in r.confounder_flags

    def test_superseded(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        r = cont(builder, 5, 19.0, decision_superseded=True)
        assert r.action is BuilderAction.CLOSED
        assert r.episode.reason is EpisodeReason.SUPERSEDED
        assert "superseded" in r.reason_codes

    def test_disturbed_aborts(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        r = builder.process(BuilderInput(ts=at(5), indoor_temp=19.0, heating_failure=True))
        assert r.action is BuilderAction.ABORTED
        assert OutcomeReason.DISTURBED.value in r.reason_codes

    def test_insufficient_data_without_temps(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), decision_start=decision(start_temp=None)))
        r = builder.process(BuilderInput(ts=at(5), mode_change=True))  # close, no temps
        assert r.action is BuilderAction.DISCARDED
        assert OutcomeReason.INSUFFICIENT_DATA.value in r.reason_codes

    def test_no_weather_valve_window_required(self):
        builder = b()
        builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        r = cont(builder, 100, 19.0)
        assert r.action is BuilderAction.CLOSED  # works without any optional context

    def test_trajectory_cap(self):
        params = BuilderParameters(outcome_trajectory_cap=3, outcome_timeout_s=999999)
        builder = b(params)
        builder.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        for i in range(1, 8):
            cont(builder, i, 18.0 + i * 0.1)
        r = cont(builder, 9, 19.0, mode_change=True)
        assert r.action is BuilderAction.CLOSED
        assert len(r.episode.trajectory.points) <= 3
        assert "trajectory_capped" in r.reason_codes


class TestSnapshotRestore:
    def test_restore_continues(self):
        a = b()
        a.process(BuilderInput(ts=at(0), indoor_temp=18.0, decision_start=decision()))
        cont(a, 5, 20.0)
        snap = a.snapshot_state()
        restored = OutcomeEpisodeBuilder("lz_1")
        restored.restore_state(snap)
        r = cont(restored, 100, 19.0)
        assert r.action is BuilderAction.CLOSED
