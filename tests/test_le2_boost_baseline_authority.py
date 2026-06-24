"""B2b-3b: strict zero-boost authority for non-boost baseline samples."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.models import (
    BaselineComparisonStore, build_non_boost_sample)
from custom_components.thermosmart.learning.models.baseline_comparison import (
    BaselineSampleRejection)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from tests.helpers_boost import boost_episode

UNKNOWN = BaselineSampleRejection.BOOST_APPLICATION_UNKNOWN.value


def _ep(**kw):
    return boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1", **kw)


def _build(bd):
    out: list = []
    s = build_non_boost_sample(_ep(), bd, reject_out=out)
    return s, (out[-1] if out else None)


class TestStrictZeroBoostAuthority:
    def test_explicit_zero_full_dispatch_accepted(self):
        s, reason = _build(BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded"))
        assert s is not None and s.boost_applied_c == 0.0 and reason is None

    def test_none_applied_rejected_unknown(self):
        s, reason = _build(BoostDispatchRecord(
            "d1", boost_applied_c=None, dispatch_status="fully_succeeded"))
        assert s is None and reason == UNKNOWN

    def test_candidate_with_zero_applied_accepted(self):
        s, reason = _build(BoostDispatchRecord(
            "d1", boost_candidate_c=2.0, boost_applied_c=0.0,
            dispatch_status="fully_succeeded"))
        assert s is not None

    def test_candidate_with_none_applied_rejected(self):
        s, reason = _build(BoostDispatchRecord(
            "d1", boost_candidate_c=2.0, boost_applied_c=None,
            dispatch_status="fully_succeeded"))
        assert s is None and reason == UNKNOWN

    def test_equal_setpoints_but_none_applied_still_rejected(self):
        # final_setpoint == baseline_setpoint does NOT prove no boost; None stays unknown.
        s, reason = _build(BoostDispatchRecord(
            "d1", boost_applied_c=None, baseline_setpoint_c=21.0, final_setpoint_c=21.0,
            dispatch_status="fully_succeeded"))
        assert s is None and reason == UNKNOWN

    def test_legacy_record_missing_field_safe(self):
        class _Legacy:           # old structure without boost_applied_c
            dispatch_status = "fully_succeeded"
        s, reason = _build(_Legacy())
        assert s is None and reason == UNKNOWN

    def test_none_dispatch_safe(self):
        s, reason = _build(None)
        assert s is None and reason == UNKNOWN

    def test_real_boost_applied_rejected(self):
        s, reason = _build(BoostDispatchRecord(
            "d1", boost_applied_c=1.5, dispatch_status="fully_succeeded"))
        assert s is None and reason == BaselineSampleRejection.BOOST_APPLIED.value


class TestStoreRejectionDiagnostics:
    def test_store_counts_rejection(self):
        st = BaselineComparisonStore("z")
        st.record_rejection(UNKNOWN)
        st.record_rejection(UNKNOWN)
        assert st.rejection_counts().get(UNKNOWN) == 2

    def test_rejection_counts_persist(self):
        st = BaselineComparisonStore("z")
        st.record_rejection(UNKNOWN)
        st2 = BaselineComparisonStore("z")
        st2.restore(st.serialize())
        assert st2.rejection_counts().get(UNKNOWN) == 1

    def test_no_migration_to_zero(self):
        # a None-applied record never becomes a baseline (no silent default to 0.0)
        st = BaselineComparisonStore("z")
        out: list = []
        s = build_non_boost_sample(_ep(), BoostDispatchRecord(
            "d1", boost_applied_c=None, dispatch_status="fully_succeeded"), reject_out=out)
        if s is None and out:
            st.record_rejection(out[-1])
        assert st.size == 0 and st.rejection_counts().get(UNKNOWN) == 1


class TestRuntimeUnknownDiagnostics:
    def test_runtime_counts_unknown_application(self):
        from datetime import datetime, timedelta, timezone
        from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
        from custom_components.thermosmart.learning.runtime import (
            ControllerDecisionInput, DecisionType, LearningRuntime, LearningRuntimeConfig,
            LearningRuntimeMode, RuntimeCycleInput, ScheduleTarget)
        T0 = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                   startup_grace_cycles=0))
        # non-boost session but dispatch reports boost_applied_c=None (unprovable)
        bd = BoostDispatchRecord("x", boost_applied_c=None, dispatch_status="fully_succeeded")
        t = 18.0
        for i, (heating, setp) in enumerate([(True, 24.0)] * 7 + [(False, 17.0)] * 6):
            ts = (T0 + timedelta(minutes=i * 5)).isoformat()
            rt.run_cycle(RuntimeCycleInput(
                zone_id="lz", ts=ts, target_c=21.0, trv_setpoint_c=setp,
                controller_demand=heating, indoor_temp=Measurement(round(t, 3), DataQuality.OK),
                schedule=ScheduleTarget(comfort_time_utc="2025-06-01T07:00:00+00:00",
                                        comfort_temperature_c=21.0),
                controller_decision=ControllerDecisionInput(
                    decision_type=DecisionType.NORMAL, target_c=21.0, trv_setpoint_c=setp),
                boost_dispatch=bd))
            t = min(21.0, t + 0.5) if heating else max(18.0, t - 0.4)
        zr = rt._zone("lz")
        assert zr.baseline_store.size == 0
        assert zr.baseline_store.rejection_counts().get(UNKNOWN, 0) >= 1
