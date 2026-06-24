"""B2b-1: authoritative LiveDecisionRecord -> BoostUpdateContext -> BoostModel.update.

Closes the dead-context gap: the boost outcome context must come from the bound
dispatch result, dispatch status must gate outcome formation, and the setpoint-target
heuristic must no longer be authority when a live record exists.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.models import BoostModel
from custom_components.thermosmart.learning.runtime.capture import (
    BoostDispatchRecord, CapturedSnapshot, DecisionType)
from custom_components.thermosmart.learning.runtime.evidence import EvidenceMaterializer
from custom_components.thermosmart.learning.runtime.ha_integration import (
    _boost_dispatch_from_live_record)
from tests.helpers_boost import boost_episode


def _snap(bd, *, decision_id="d1", setpoint=23.0, target=21.0,
          dtype=DecisionType.BOOST):
    return CapturedSnapshot(
        zone_id="lz", ts="2025-01-01T07:00:00+00:00", decision_id=decision_id,
        target_c=target, trv_setpoint_c=setpoint,
        indoor_temp=Measurement(19.0, DataQuality.OK), indoor_temp_quality=DataQuality.OK,
        decision_type=dtype, controller_demand=True, tpi_on=True, boost_active=True,
        preheat_active=False, fallback_active=False, window_open=False, heating_failure=False,
        schedule_comfort_time_utc=None, schedule_comfort_temperature_c=None,
        outdoor_temp=None, valve_opening=None, hvac_action=None, forecast_high=None,
        accepted_manual_corrections=(), rejected_echo_corrections=(), boost_dispatch=bd)


@dataclass
class _FakeLiveRecord:
    decision_id: Optional[str]
    boost_candidate_c: Optional[float] = 2.0
    boost_applied_c: Optional[float] = 2.0
    baseline_setpoint_c: Optional[float] = 21.0
    final_setpoint_c: Optional[float] = 23.0
    dispatch_effective_setpoint_min_c: Optional[float] = 23.0
    dispatch_effective_setpoint_max_c: Optional[float] = 23.0
    dispatch_status: str = "fully_succeeded"
    outcome_eligible: bool = True
    outcome_reliability: str = "full"


M = EvidenceMaterializer()


class TestDispatchGating:
    def test_full_dispatch_authoritative(self):
        c = M.materialize(_snap(BoostDispatchRecord(
            "d1", boost_applied_c=2.0, dispatch_status="fully_succeeded",
            outcome_reliability="full"))).boost_context
        assert c is not None and c.authority == "live_record"
        assert c.outcome_reliability == "full" and c.boost_applied_c == 2.0

    def test_partial_dispatch_reduced(self):
        c = M.materialize(_snap(BoostDispatchRecord(
            "d1", boost_applied_c=2.0, dispatch_status="partially_succeeded"))).boost_context
        assert c is not None and c.outcome_reliability == "partial"

    def test_failed_dispatch_no_outcome(self):
        e = M.materialize(_snap(BoostDispatchRecord(
            "d1", boost_applied_c=2.0, dispatch_status="failed")))
        assert e.boost_context is None and "boost_dispatch_failed" in e.notes

    def test_not_attempted_no_outcome(self):
        e = M.materialize(_snap(BoostDispatchRecord("d1", dispatch_status="not_attempted")))
        assert e.boost_context is None and "boost_not_attempted" in e.notes

    def test_zero_applied_distinct_from_not_dispatched(self):
        e = M.materialize(_snap(BoostDispatchRecord(
            "d1", boost_candidate_c=2.0, boost_applied_c=0.0,
            dispatch_status="fully_succeeded")))
        assert e.boost_context is None and "boost_zero_applied" in e.notes


class TestLegacyHeuristic:
    def test_legacy_fallback_only_without_record(self):
        e = M.materialize(_snap(None))   # no authoritative record
        assert e.boost_context is not None
        assert e.boost_context.authority == "legacy_fallback"
        assert "boost_legacy_fallback" in e.notes

    def test_record_overrides_heuristic(self):
        # setpoint-target would be 2.0, but the record says applied=1.0 -> record wins
        c = M.materialize(_snap(BoostDispatchRecord(
            "d1", boost_applied_c=1.0, dispatch_status="fully_succeeded"),
            setpoint=23.0, target=21.0)).boost_context
        assert c.authority == "live_record" and c.boost_applied_c == 1.0
        assert c.requested_offset_c == 1.0   # not the 2.0 heuristic

    def test_non_boost_decision_no_legacy_context(self):
        e = M.materialize(_snap(None, dtype=DecisionType.NORMAL))
        assert e.boost_context is None


class TestLiveRecordMapper:
    def test_maps_decision_and_fields(self):
        bd = _boost_dispatch_from_live_record(_FakeLiveRecord(decision_id="dX"))
        assert bd.decision_id == "dX" and bd.boost_applied_c == 2.0
        assert bd.dispatch_status == "fully_succeeded" and bd.outcome_reliability == "full"

    def test_none_record(self):
        assert _boost_dispatch_from_live_record(None) is None

    def test_missing_decision_id(self):
        assert _boost_dispatch_from_live_record(_FakeLiveRecord(decision_id=None)) is None


class TestContextPassthrough:
    def test_context_reaches_model_update(self):
        model = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        ctx = M.materialize(_snap(BoostDispatchRecord(
            "d1", boost_applied_c=1.5, dispatch_status="fully_succeeded",
            outcome_reliability="full"), decision_id="d1")).boost_context
        ctx = replace(ctx, source_episode_id=ep.episode_id)  # bound at close
        res = model.update(ep, ctx)
        assert res.accepted
        # no baseline => INSUFFICIENT_COMPARISON => diagnostic (non-adaptive) sample
        sample = (model._state.recent_samples + model._state.diagnostic_samples)[-1]
        assert sample.decision_id == "d1"
        assert sample.requested_offset_c == pytest.approx(1.5)

    def test_partial_lowers_attribution(self):
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        full = BoostModel("lz")
        cfull = replace(M.materialize(_snap(BoostDispatchRecord(
            "d1", boost_applied_c=1.5, dispatch_status="fully_succeeded"))).boost_context,
            source_episode_id=ep.episode_id)
        full.update(ep, cfull)
        partial = BoostModel("lz")
        cpart = replace(M.materialize(_snap(BoostDispatchRecord(
            "d1", boost_applied_c=1.5, dispatch_status="partially_succeeded"))).boost_context,
            source_episode_id=ep.episode_id)
        partial.update(ep, cpart)
        psamp = (partial._state.recent_samples + partial._state.diagnostic_samples)[-1]
        fsamp = (full._state.recent_samples + full._state.diagnostic_samples)[-1]
        assert psamp.attribution_reliability < fsamp.attribution_reliability

    def test_decision_binding_chain(self):
        snap = _snap(BoostDispatchRecord("d1", boost_applied_c=1.5,
                                         dispatch_status="fully_succeeded"), decision_id="d1")
        ctx = M.materialize(snap).boost_context
        assert snap.decision_id == ctx.decision_id == "d1"
        ep = boost_episode("e1", [18.0, 20.0, 21.0], decision_id=ctx.decision_id)
        BoostModel("lz").update(ep, ctx)  # no raise -> bound consistently
