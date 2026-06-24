"""B2b-3c: authoritative zero-boost runtime emission + zero-preservation through the chain."""
from __future__ import annotations

import dataclasses

import pytest

from custom_components.thermosmart.learning.decision.contracts import (
    BoostEvaluationStatus as BES, LiveDecisionRecord)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from custom_components.thermosmart.learning.runtime.ha_integration import (
    _boost_dispatch_from_live_record)
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


def _live(**kw):
    base = dict(decision_id="d1", zone_id="z", ts="2025-01-01T00:00:00+00:00", mode="shadow",
                baseline_setpoint_c=21.0, final_setpoint_c=21.0, boost_applied=False,
                boost_candidate_c=None, boost_applied_c=None, boost_rejected_reason=None,
                preheat_minutes=0.0, preheat_status=None, early_cutoff_state=None,
                heat_rate_c_per_h=None, heat_rate_confidence=None, heat_rate_source=None,
                onset_delay_min=None, onset_delay_source=None)
    base.update(kw)
    return LiveDecisionRecord(**base)


class TestEmissionShadow:
    async def test_shadow_emits_evaluated_not_applied_zero(self):
        coord = make_recording_coordinator(indoor="18.0"); coord._active_control = True
        sh = attach_shadow(coord); await sh.async_setup()
        z = (await coord._async_update_data())["zone"]
        assert z["_boost_evaluation_status"] == BES.EVALUATED_NOT_APPLIED.value
        assert z["_boost_applied_c"] == 0.0    # authoritative zero, not None

    async def test_no_target_is_not_evaluated(self):
        sh = attach_shadow(make_recording_coordinator())
        await sh.async_setup()
        rec = {"trv_setpoint": None}            # missing target/setpoint
        sh.emit_boost_authority_status_safe(rec)
        assert rec["_boost_evaluation_status"] == BES.NOT_EVALUATED.value
        assert rec["_boost_applied_c"] is None

    async def test_direct_valve_evaluated_not_applied(self):
        sh = attach_shadow(make_recording_coordinator())
        await sh.async_setup()
        rec = {"effective_target": 21.0, "trv_setpoint": 21.0, "tpi_valve_direct": True}
        sh.emit_boost_authority_status_safe(rec)
        assert rec["_boost_evaluation_status"] == BES.EVALUATED_NOT_APPLIED.value
        assert rec["_boost_applied_c"] == 0.0 and rec["_boost_rejection_reason"] == "direct_valve"

    async def test_disabled_le2_not_evaluated(self):
        sh = attach_shadow(make_recording_coordinator())
        await sh.async_setup()
        sh._enabled = False
        rec = {"effective_target": 21.0, "trv_setpoint": 21.0}
        sh.emit_boost_authority_status_safe(rec)
        assert rec["_boost_evaluation_status"] == BES.NOT_EVALUATED.value
        assert rec["_boost_applied_c"] is None

    async def test_applied_offset_status_applied(self):
        sh = attach_shadow(make_recording_coordinator())
        await sh.async_setup()
        rec = {"effective_target": 21.0, "trv_setpoint": 23.0, "_boost_applied_c": 2.0}
        sh.emit_boost_authority_status_safe(rec)
        assert rec["_boost_evaluation_status"] == BES.APPLIED.value
        assert rec["_boost_applied_c"] == 2.0


class TestZeroPreservation:
    def test_zero_survives_live_to_dispatch_record(self):
        rec = _live(boost_applied_c=0.0, boost_evaluation_status=BES.EVALUATED_NOT_APPLIED.value,
                    dispatch_status="fully_succeeded", dispatch_path="setpoint")
        bd = _boost_dispatch_from_live_record(rec)
        assert bd.boost_applied_c == 0.0 and bd.boost_applied_c is not None
        assert bd.boost_evaluation_status == BES.EVALUATED_NOT_APPLIED.value

    def test_none_stays_none_to_dispatch_record(self):
        rec = _live(boost_applied_c=None, boost_evaluation_status=BES.NOT_EVALUATED.value)
        bd = _boost_dispatch_from_live_record(rec)
        assert bd.boost_applied_c is None
        assert bd.boost_evaluation_status == BES.NOT_EVALUATED.value

    def test_zero_survives_dataclasses_replace(self):
        rec = _live(boost_applied_c=0.0)
        rec2 = dataclasses.replace(rec, dispatch_status="fully_succeeded")
        assert rec2.boost_applied_c == 0.0 and rec2.boost_applied_c is not None

    def test_applied_offset_preserved_exact(self):
        rec = _live(boost_applied_c=1.7, boost_evaluation_status=BES.APPLIED.value,
                    dispatch_path="setpoint")
        bd = _boost_dispatch_from_live_record(rec)
        assert bd.boost_applied_c == pytest.approx(1.7)

    def test_device_control_type_from_dispatch_path(self):
        assert _boost_dispatch_from_live_record(
            _live(boost_applied_c=0.0, dispatch_path="direct_valve")).device_control_type \
            == "direct_valve"
        assert _boost_dispatch_from_live_record(
            _live(boost_applied_c=0.0, dispatch_path="setpoint")).device_control_type \
            == "setpoint"


class TestBaselineEligibilityAfterDispatch:
    def test_zero_full_dispatch_reaches_builder_as_zero(self):
        # the dispatch record an outcome would consume carries authoritative 0.0
        bd = _boost_dispatch_from_live_record(
            _live(boost_applied_c=0.0, dispatch_status="fully_succeeded", dispatch_path="setpoint",
                  boost_evaluation_status=BES.EVALUATED_NOT_APPLIED.value))
        from custom_components.thermosmart.learning.models import build_non_boost_sample
        from tests.helpers_boost import boost_episode
        ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1")
        assert build_non_boost_sample(ep, bd) is not None

    def test_zero_not_attempted_no_baseline(self):
        bd = _boost_dispatch_from_live_record(
            _live(boost_applied_c=0.0, dispatch_status="not_attempted"))
        from custom_components.thermosmart.learning.models import build_non_boost_sample
        from tests.helpers_boost import boost_episode
        ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1")
        assert build_non_boost_sample(ep, bd) is None
