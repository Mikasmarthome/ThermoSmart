"""B2b-1: lifecycle fan-out — a dispatched boost's OutcomeEpisode updates BoostModel.

Proves the previously-dead path lifecycle.apply_update -> BoostModel.update now fires,
gated by dispatch status, with the context bound by decision_id.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, LearningRuntime, LearningRuntimeConfig,
    LearningRuntimeMode, RuntimeCycleInput, ScheduleTarget)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord

T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
COMFORT = "2025-01-01T07:00:00+00:00"


def _rt():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                 startup_grace_cycles=0),
                              clock=lambda: "2025-01-01T00:00:00+00:00")


def _cycle(rt, i, temp, *, heating, boost_dispatch=None, target=21.0, zone="lz"):
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = (target + 3.0) if heating else (target - 4.0)
    inp = RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK),
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(
            decision_type=DecisionType.BOOST if boost_dispatch else DecisionType.NORMAL,
            target_c=target, trv_setpoint_c=setp),
        boost_dispatch=boost_dispatch)
    return rt.run_cycle(inp)


def _session(rt, boost_dispatch):
    """A heating ramp to target then settle — opens and closes one OutcomeEpisode."""
    temp = 18.0
    i = 0
    for _ in range(7):
        _cycle(rt, i, round(temp, 3), heating=True, boost_dispatch=boost_dispatch)
        temp = min(21.0, temp + 0.5)
        i += 1
    for _ in range(6):
        _cycle(rt, i, round(temp, 3), heating=False)
        temp = max(18.0, temp - 0.4)
        i += 1
    return rt._zone("lz")


class TestBoostFanOut:
    def test_full_dispatch_updates_boost_model(self):
        rt = _rt()
        bd = BoostDispatchRecord("ignored", boost_applied_c=1.5,
                                 dispatch_status="fully_succeeded", outcome_reliability="full")
        zr = _session(rt, bd)
        assert zr.model_update_counts.get("boost", 0) >= 1

    def test_failed_dispatch_no_boost_update(self):
        rt = _rt()
        bd = BoostDispatchRecord("ignored", boost_applied_c=1.5, dispatch_status="failed")
        zr = _session(rt, bd)
        assert zr.model_update_counts.get("boost", 0) == 0

    def test_not_attempted_no_boost_update(self):
        rt = _rt()
        bd = BoostDispatchRecord("ignored", dispatch_status="not_attempted")
        zr = _session(rt, bd)
        assert zr.model_update_counts.get("boost", 0) == 0

    def test_no_boost_dispatch_no_boost_update(self):
        rt = _rt()
        zr = _session(rt, None)   # plain heating, no boost applied
        assert zr.model_update_counts.get("boost", 0) == 0

    def test_pending_contexts_bounded(self):
        rt = _rt()
        zr = rt._zone("lz")
        # many distinct boost decisions without closing -> bounded FIFO
        for i in range(200):
            bd = BoostDispatchRecord("x", boost_applied_c=1.0,
                                     dispatch_status="fully_succeeded")
            _cycle(rt, i, 19.0, heating=True, boost_dispatch=bd, target=21.0 + (i % 3) * 0.0)
        assert len(zr.pending_boost_contexts) <= zr._pending_boost_cap

    def test_outcome_model_still_updates(self):
        # the existing outcome model update must not regress when boost fan-out added
        rt = _rt()
        zr = _session(rt, BoostDispatchRecord("x", boost_applied_c=1.5,
                                              dispatch_status="fully_succeeded"))
        assert zr.model_update_counts.get("outcome", 0) >= 1
