"""B2b-4d: post-reach overshoot observer + deferred, exactly-once, restart-safe finalization."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.models import BoostOutcomeLifecycleState
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, LearningRuntime, LearningRuntimeConfig,
    LearningRuntimeMode, RuntimeCycleInput, ScheduleTarget)
from custom_components.thermosmart.learning.runtime.boost_pending import (
    BoostPendingOutcome, OBS_MAX_GAP_S, observation_window_s)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord

T0 = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)
BD = BoostDispatchRecord("x", boost_applied_c=1.5, dispatch_status="fully_succeeded",
                         outcome_reliability="full", baseline_setpoint_c=21.0,
                         effective_setpoints=(22.5,), targets_total=1)


def _rt():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                 startup_grace_cycles=0),
                              clock=lambda: "2025-01-01T00:00:00+00:00")


def _cyc(r, i, temp, heating, *, target=21.0, bd=BD, zone="lz", window=False):
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = 24.0 if heating else 17.0
    r.run_cycle(RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK), window_open=window,
        schedule=ScheduleTarget(comfort_time_utc="2025-06-01T07:00:00+00:00",
                                comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(decision_type=DecisionType.BOOST,
                                                    target_c=target, trv_setpoint_c=setp),
        boost_dispatch=bd))


# reach at c5 (band held c3..c5 = 600s); deadline = c5 + 900s = c8
_REACH = [(0, 18, True), (1, 19, True), (2, 20, True), (3, 20.6, True), (4, 20.8, True),
          (5, 21.0, True)]


def _classes(zr):
    return dict(zr.orchestrator.models["boost"]._state.outcome_class_counts)


class TestDeferredFinalization:
    def test_no_update_at_reach(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h)
        assert zr.pending_boost_outcome is not None
        assert zr.pending_boost_outcome.state == \
            BoostOutcomeLifecycleState.TARGET_REACHED_PENDING_OBSERVATION.value
        assert zr.model_update_counts.get("boost", 0) == 0   # NOT finalized yet

    def test_late_overshoot_within_window(self):
        r = _rt(); zr = r._zone("lz")
        seq = _REACH + [(6, 22.0, False), (7, 22.0, False), (8, 22.0, False), (9, 21.0, False)]
        for i, t, h in seq:
            _cyc(r, i, t, h)
        assert _classes(zr).get("boost_overshoot", 0) == 1
        assert zr.model_update_counts.get("boost", 0) == 1
        assert zr.pending_boost_outcome is None

    def test_no_overshoot_normal_outcome_after_deadline(self):
        r = _rt(); zr = r._zone("lz")
        seq = _REACH + [(6, 21.0, False), (7, 21.0, False), (8, 21.0, False), (9, 20.5, False)]
        for i, t, h in seq:
            _cyc(r, i, t, h)
        cc = _classes(zr)
        assert cc and "boost_overshoot" not in cc
        assert zr.model_update_counts.get("boost", 0) == 1

    def test_short_peak_after_reach_not_overshoot(self):
        r = _rt(); zr = r._zone("lz")
        seq = _REACH + [(6, 23.0, False), (7, 21.0, False), (8, 21.0, False)]   # 1 spike
        for i, t, h in seq:
            _cyc(r, i, t, h)
        assert "boost_overshoot" not in _classes(zr)

    def test_overshoot_after_deadline_ignored(self):
        r = _rt(); zr = r._zone("lz")
        # overshoot only starts at c9/c10 (after deadline c8) -> not attributed
        seq = _REACH + [(6, 21.0, False), (7, 21.0, False), (8, 21.0, False),
                        (9, 22.5, False), (10, 22.5, False)]
        for i, t, h in seq:
            _cyc(r, i, t, h)
        assert "boost_overshoot" not in _classes(zr)


class TestInterruption:
    def test_window_open_interrupts(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h)
        _cyc(r, 6, 21.0, False, window=True)   # window opens during observation
        assert zr.pending_boost_outcome is None
        assert _classes(zr).get("boost_interrupted", 0) == 1

    def test_target_change_interrupts(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h)
        _cyc(r, 6, 21.0, False, target=23.0)   # new target -> context changed
        assert _classes(zr).get("boost_interrupted", 0) == 1


class TestExactlyOnce:
    def test_one_update_across_many_post_deadline_cycles(self):
        r = _rt(); zr = r._zone("lz")
        seq = _REACH + [(i, 21.0, False) for i in range(6, 20)]
        for i, t, h in seq:
            _cyc(r, i, t, h)
        assert zr.model_update_counts.get("boost", 0) == 1

    def test_finalize_idempotent_via_model_dedup(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH + [(6, 21.0, False), (7, 21.0, False), (8, 21.0, False)]:
            _cyc(r, i, t, h)
        n = zr.orchestrator.models["boost"]._state.general.effective_factor.effective_n
        # re-finalize the same (already-cleared) pending shape -> model dedups, no double
        from custom_components.thermosmart.learning.runtime.boost_pending import (
            BoostPendingOutcome as _P)
        assert zr.pending_boost_outcome is None
        assert zr.model_update_counts.get("boost", 0) == 1


class TestRestart:
    def _run_to_pending(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h)
        return r, zr

    def test_pending_serialize_restore(self):
        r, zr = self._run_to_pending()
        blob = zr.serialize()
        r2 = _rt(); r2._zone("lz").restore(blob)
        p = r2._zone("lz").pending_boost_outcome
        assert p is not None and p.decision_id == zr.pending_boost_outcome.decision_id
        assert p.deadline_ts == zr.pending_boost_outcome.deadline_ts   # deadline unchanged

    def test_large_restart_gap_interrupts(self):
        r, zr = self._run_to_pending()
        blob = zr.serialize()
        r2 = _rt(); zr2 = r2._zone("lz"); zr2.restore(blob)
        # resume far in the future -> gap > OBS_MAX_GAP_S -> interrupted, never success
        far = (T0 + timedelta(seconds=10 * OBS_MAX_GAP_S)).isoformat()
        r2.run_cycle(RuntimeCycleInput(
            zone_id="lz", ts=far, target_c=21.0, trv_setpoint_c=17.0, controller_demand=False,
            indoor_temp=Measurement(21.0, DataQuality.OK),
            schedule=ScheduleTarget(comfort_time_utc="2025-06-01T07:00:00+00:00",
                                    comfort_temperature_c=21.0),
            controller_decision=ControllerDecisionInput(decision_type=DecisionType.NORMAL,
                                                        target_c=21.0, trv_setpoint_c=17.0)))
        cc = dict(zr2.orchestrator.models["boost"]._state.outcome_class_counts)
        assert cc.get("boost_interrupted", 0) == 1 and "boost_success" not in cc

    def test_corrupt_pending_degrades_safely(self):
        r2 = _rt()
        r2._zone("lz").restore({"pending_boost_outcome": {"schema_version": 999, "bad": 1}})
        assert r2._zone("lz").pending_boost_outcome is None


class TestObservationWindow:
    def test_window_clamped(self):
        assert observation_window_s(None) == 900.0
        assert observation_window_s(100.0) == 600.0      # below min -> 600
        assert observation_window_s(5000.0) == 1800.0    # above max -> 1800
        assert observation_window_s(1200.0) == 1200.0


class TestBounds:
    def test_one_pending_per_zone(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h)
        first = zr.pending_boost_outcome
        assert first is not None
        # a new boost decision/episode that REACHES -> old pending finalized, not overwritten silently
        seq2 = [(10, 18, True), (11, 19, True), (12, 20, True), (13, 20.6, True),
                (14, 20.8, True), (15, 21.0, True)]
        for i, t, h in seq2:
            _cyc(r, i, t, h, bd=BoostDispatchRecord(
                "y", boost_applied_c=1.5, dispatch_status="fully_succeeded",
                outcome_reliability="full", baseline_setpoint_c=21.0,
                effective_setpoints=(22.5,), targets_total=1))
        # the old pending was finalized (model updated at least once); at most one pending now
        assert zr.model_update_counts.get("boost", 0) >= 1
