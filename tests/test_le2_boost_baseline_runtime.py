"""B2b-3: runtime wiring — non-boost outcomes create baseline samples; boost episodes
get a comparison; expected_non_boost_duration_s reaches the B2b-1 context path."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, LearningRuntime, LearningRuntimeConfig,
    LearningRuntimeMode, RuntimeCycleInput, ScheduleTarget)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from tests.helpers_baseline import sample

T0 = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)


def _rt():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                 startup_grace_cycles=0))


def _session(rt, target, base, *, boost, zone="lz"):
    bd = BoostDispatchRecord("x", boost_applied_c=(1.5 if boost else 0.0),
                             dispatch_status="fully_succeeded", outcome_reliability="full")
    t = target - 3.0
    i = 0
    dtype = DecisionType.BOOST if boost else DecisionType.NORMAL
    for heating, setp in [(True, target + 3)] * 7 + [(False, target - 4)] * 6:
        ts = (T0 + timedelta(minutes=base * 120 + i * 5)).isoformat()
        rt.run_cycle(RuntimeCycleInput(
            zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
            indoor_temp=Measurement(round(t, 3), DataQuality.OK),
            schedule=ScheduleTarget(comfort_time_utc="2025-06-01T07:00:00+00:00",
                                    comfort_temperature_c=target),
            controller_decision=ControllerDecisionInput(decision_type=dtype, target_c=target,
                                                        trv_setpoint_c=setp),
            boost_dispatch=bd))
        t = min(target, t + 0.5) if heating else max(target - 3, t - 0.4)
        i += 1


class TestBaselineRuntimeWiring:
    def test_non_boost_outcome_creates_baseline_sample(self):
        rt = _rt()
        for k in range(4):
            _session(rt, 20.5 + 0.05 * k, k, boost=False)
        assert rt._zone("lz").baseline_store.size >= 3

    def test_boost_episode_queries_baseline(self):
        rt = _rt()
        zr = rt._zone("lz")
        # pre-seed comparable historical non-boost samples completed in the past
        for i in range(5):
            zr.baseline_store.add(sample(f"h{i}", 1800, zone="lz", target=20.6, deficit=3.0,
                                         reached=f"2025-05-2{i}T00:00:00+00:00"))
        _session(rt, 20.6, 5, boost=True)
        est = zr.last_baseline_estimate
        assert est is not None and est.expected_non_boost_duration_s is not None
        assert est.source == "historical_only"

    def test_insufficient_baseline_without_samples(self):
        rt = _rt()
        _session(rt, 20.6, 0, boost=True)   # no historical samples
        est = rt._zone("lz").last_baseline_estimate
        assert est is None or est.expected_non_boost_duration_s is None

    def test_boost_episode_not_stored_as_baseline(self):
        rt = _rt()
        _session(rt, 20.6, 0, boost=True)
        assert rt._zone("lz").baseline_store.size == 0

    def test_baseline_store_persists(self):
        rt = _rt()
        zr = rt._zone("lz")
        for i in range(4):
            zr.baseline_store.add(sample(f"h{i}", 1800, zone="lz",
                                         reached=f"2025-05-2{i}T00:00:00+00:00"))
        blob = zr.serialize()
        rt2 = _rt()
        rt2._zone("lz").restore(blob)
        assert rt2._zone("lz").baseline_store.size == 4

    def test_control_path_stays_disabled(self):
        rt = _rt()
        _session(rt, 20.6, 0, boost=False)
        assert not rt.control_enabled

    def test_cross_zone_isolation(self):
        rt = _rt()
        for k in range(4):
            _session(rt, 20.5, k, boost=False, zone="lz")
        # other zone has no baseline samples
        assert rt._zone("other").baseline_store.size == 0
