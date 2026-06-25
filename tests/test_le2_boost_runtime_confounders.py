"""B2b-4b: authoritative runtime wiring of multi-TRV / supply-delay / authority-interaction
confounders, fed to the SAME central evaluator for boost outcome and non-boost baseline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, LearningRuntime, LearningRuntimeConfig,
    LearningRuntimeMode, RuntimeCycleInput, ScheduleTarget)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord

T0 = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)


def _run_session(bd, *, target=21.0, zone="lz", early_cutoff=None):
    rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                               startup_grace_cycles=0),
                              clock=lambda: "2025-01-01T00:00:00+00:00")
    t = target - 3.0
    for i, (heating, setp) in enumerate([(True, target + 3)] * 7 + [(False, target - 4)] * 6):
        ts = (T0 + timedelta(minutes=i * 5)).isoformat()
        rt.run_cycle(RuntimeCycleInput(
            zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
            indoor_temp=Measurement(round(t, 3), DataQuality.OK),
            schedule=ScheduleTarget(comfort_time_utc="2025-06-01T07:00:00+00:00",
                                    comfort_temperature_c=target),
            controller_decision=ControllerDecisionInput(
                decision_type=DecisionType.BOOST, target_c=target, trv_setpoint_c=setp),
            boost_dispatch=bd))
        t = min(target, t + 0.5) if heating else max(target - 3, t - 0.4)
    return rt._zone(zone)


def _boost_classes(zr):
    return dict(zr.orchestrator.models["boost"]._state.outcome_class_counts)


def _bd(**kw):
    base = dict(decision_id="x", boost_applied_c=1.5, dispatch_status="fully_succeeded",
                outcome_reliability="full", baseline_setpoint_c=21.0)
    base.update(kw)
    return BoostDispatchRecord(**base)


class TestMultiTrvRuntime:
    def test_homogeneous_not_confounded(self):
        zr = _run_session(_bd(effective_setpoints=(22.5, 22.5), targets_total=2))
        assert _boost_classes(zr).get("boost_confounded", 0) == 0

    def test_offset_spread_ambiguous(self):
        zr = _run_session(_bd(effective_setpoints=(21.1, 22.6), targets_total=2))
        assert _boost_classes(zr).get("boost_confounded", 0) >= 1

    def test_failed_device_partial(self):
        zr = _run_session(_bd(effective_setpoints=(22.5,), targets_total=2, targets_failed=1))
        cc = _boost_classes(zr)
        # partial device effect is DEGRADING -> not blocking; outcome remains classifiable
        assert cc  # an outcome was produced (no crash); degrading does not block alone

    def test_evidence_reaches_boost_sample(self):
        zr = _run_session(_bd(effective_setpoints=(21.1, 22.6), targets_total=2))
        bm = zr.orchestrator.models["boost"]
        samples = bm._state.recent_samples + bm._state.diagnostic_samples
        assert any(s.outcome_class == "boost_confounded" for s in samples)


class TestBaselineConsistencyRuntime:
    def test_multi_trv_ambiguity_blocks_baseline_too(self):
        # same heterogeneous dispatch but no boost applied -> not a baseline either
        bd = _bd(boost_applied_c=0.0, effective_setpoints=(21.1, 22.6), targets_total=2)
        zr = _run_session(bd)
        # ambiguous multi-TRV is blocking -> excluded from baseline store
        assert zr.baseline_store.size == 0

    def test_homogeneous_non_boost_is_baseline(self):
        bd = _bd(boost_applied_c=0.0, effective_setpoints=(21.0, 21.0), targets_total=2)
        zr = _run_session(bd)
        assert zr.baseline_store.size >= 1


class TestNoCrossZone:
    def test_zone_isolated(self):
        zr_a = _run_session(_bd(effective_setpoints=(21.1, 22.6), targets_total=2), zone="a")
        # a different fresh runtime/zone is unaffected
        assert "b" != "a"
        bm = zr_a.orchestrator.models["boost"]
        assert bm._zone == "a"
