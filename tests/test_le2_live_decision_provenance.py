"""Phase B1: Authoritative Live Decision Provenance — golden tests.

Verifies that:
1. ``build_live_decision_pre_dispatch()`` captures the authoritative live decision
   from the recommendation dict without re-running the resolver.
2. ``compute_decision_trace_safe(live_record=...)`` derives the trace from the
   LiveDecisionRecord — not from a second resolver run.
3. Boost provenance fields (_boost_candidate_c, _boost_applied_c,
   _boost_rejection_reason) are correctly captured by adjust_recommendation_safe().
4. Dispatch fields are correctly set via dataclasses.replace() post-dispatch.
5. Consistency: trace.final_device_setpoint_c matches what the coordinator
   passed to the HA service (coordinator-level, pre-per-entity-clamp).
6. Same-cycle invariant: trace fields never diverge from live-path values.

Architecture:
  adjust_recommendation_safe()           → recommendation["_boost_*"] provenance keys
  build_live_decision_pre_dispatch()     → LiveDecisionRecord (pre-dispatch)
  dataclasses.replace(record, dispatch_*) → LiveDecisionRecord (post-dispatch)
  compute_decision_trace_safe(live_record=...)  → DecisionTrace (no resolver re-run)
"""
from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.decision.contracts import LiveDecisionRecord
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)
from tests.helpers import make_coordinator
from tests.helpers_ha_runtime import attach_shadow, FakeStore


# ── shared helpers ────────────────────────────────────────────────────────────

def _pred(values, *, prediction_type, fallback_used=False, confidence=0.9):
    p = MagicMock()
    p.prediction_type = prediction_type
    p.values = values
    p.fallback_used = fallback_used
    p.confidence = confidence
    return p


def _od(delay_min, *, conf=0.9, fallback=False):
    return _pred({"onset_delay_minutes": delay_min},
                 prediction_type=PredictionType.ONSET_DELAY,
                 fallback_used=fallback, confidence=conf)


def _shadow(od=None):
    coord = make_coordinator()
    sh = attach_shadow(coord, store=FakeStore())
    if od is not None:
        sh.runtime._zone(coord.zone_id).last_predictions = {
            PredictionType.ONSET_DELAY: od}
    return sh


def _base_recommendation(
    *,
    setpoint: float = 21.0,
    target: float = 21.0,
    duty: float = 50.0,
    boost_applied: bool = False,
    boost_candidate: float | None = None,
    boost_offset: float | None = None,
    rejection_reason: str | None = None,
    preheat_minutes: float = 0.0,
    preheat_status: str | None = None,
    early_cutoff_state: str | None = None,
    window_open: bool = False,
    heating_failure: bool = False,
    tpi_valve_direct: bool = False,
) -> dict:
    """Minimal recommendation dict as produced by the coordinator after adjust_recommendation_safe."""
    rec: dict = {
        "trv_setpoint": setpoint,
        "effective_target": target,
        "tpi_duty_cycle": duty,
        "tpi_coef_int": 1.2,
        "tpi_coef_ext": 0.5,
        "tpi_coef_source": "learned",
        "tpi_coef_diag": {
            "heat_rate_c_per_h": 1.8,
            "hr_confidence": 0.85,
            "hl_confidence": 0.70,
        },
        "tpi_hl_rate": 0.4,
        "preheat_minutes": preheat_minutes,
        "preheat_status": preheat_status,
        "early_cutoff_state": early_cutoff_state,
        "window_open": window_open,
        "heating_failure": heating_failure,
        "tpi_valve_direct": tpi_valve_direct,
        # Phase B1 provenance keys (normally written by adjust_recommendation_safe)
        "_boost_rejection_reason": rejection_reason,
        "_boost_candidate_c": boost_candidate,
        "_boost_applied_c": boost_offset,
        "le2_boost_adjusted": boost_applied,
    }
    if boost_applied and boost_offset is not None:
        # When boost was applied, tpi_baseline_setpoint is the pre-boost setpoint
        rec["tpi_baseline_setpoint"] = setpoint - boost_offset
        rec["trv_setpoint"] = setpoint  # setpoint already includes boost
    return rec


# ── Section 1: LiveDecisionRecord construction ────────────────────────────────

class TestLiveDecisionRecordConstruction:
    """build_live_decision_pre_dispatch() builds a correct pre-dispatch record."""

    def test_returns_live_decision_record_instance(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert isinstance(result, LiveDecisionRecord)

    def test_zone_id_is_correct(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.zone_id == sh._zone

    def test_mode_shadow_when_control_disabled(self):
        sh = _shadow()
        assert not sh.control_enabled
        rec = _base_recommendation(setpoint=21.0)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.mode == "shadow"

    def test_final_setpoint_from_recommendation(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=22.5)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.final_setpoint_c == pytest.approx(22.5)

    def test_baseline_setpoint_no_boost(self):
        """Without boost, baseline == final (both from trv_setpoint)."""
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, boost_applied=False)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.baseline_setpoint_c == pytest.approx(21.0)
        assert result.final_setpoint_c == pytest.approx(21.0)

    def test_baseline_setpoint_with_boost(self):
        """With boost applied: baseline = tpi_baseline_setpoint, final = trv_setpoint."""
        sh = _shadow()
        # trv_setpoint = 22.5 (21.0 + 1.5 boost)
        rec = _base_recommendation(
            setpoint=22.5, boost_applied=True,
            boost_offset=1.5, boost_candidate=1.5)
        # _base_recommendation sets tpi_baseline_setpoint = 22.5 - 1.5 = 21.0
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.baseline_setpoint_c == pytest.approx(21.0)
        assert result.final_setpoint_c == pytest.approx(22.5)

    def test_boost_applied_true(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=22.5, boost_applied=True,
                                   boost_offset=1.5, boost_candidate=1.5)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.boost_applied is True

    def test_boost_candidate_captured(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=22.5, boost_applied=True,
                                   boost_offset=1.5, boost_candidate=1.5)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.boost_candidate_c == pytest.approx(1.5)

    def test_boost_applied_c_captured(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=22.5, boost_applied=True,
                                   boost_offset=1.5, boost_candidate=1.5)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.boost_applied_c == pytest.approx(1.5)

    def test_boost_rejected_reason_captured(self):
        sh = _shadow()
        rec = _base_recommendation(
            setpoint=21.0, boost_applied=False,
            rejection_reason="window_open", window_open=True)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.boost_rejected_reason == "window_open"

    def test_preheat_minutes_captured(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, preheat_minutes=45.0,
                                   preheat_status="valid")
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.preheat_minutes == pytest.approx(45.0)
        assert result.preheat_status == "valid"

    def test_early_cutoff_state_captured(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, early_cutoff_state="cutoff_applied")
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.early_cutoff_state == "cutoff_applied"

    def test_heat_rate_from_tpi_diag(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        rec["tpi_coef_diag"] = {"heat_rate_c_per_h": 2.3, "hr_confidence": 0.92}
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.heat_rate_c_per_h == pytest.approx(2.3)
        assert result.heat_rate_confidence == pytest.approx(0.92)

    def test_onset_delay_from_predictions(self):
        """Onset delay is read from zone runtime predictions, not from recommendation."""
        sh = _shadow(od=_od(10.0))
        rec = _base_recommendation(setpoint=21.0)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.onset_delay_min == pytest.approx(10.0)
        assert result.onset_delay_source == "valid"

    def test_onset_delay_fallback_when_no_prediction(self):
        sh = _shadow()  # no onset delay prediction
        rec = _base_recommendation(setpoint=21.0)
        result = sh.build_live_decision_pre_dispatch(rec)
        # Should be None or prior — not raised
        assert isinstance(result, LiveDecisionRecord)

    def test_dispatch_fields_are_pre_dispatch_defaults(self):
        """Pre-dispatch record has dispatch_attempted=False."""
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result.dispatch_attempted is False
        assert result.dispatch_setpoint_c is None
        assert result.dispatch_succeeded is False

    def test_returns_none_when_disabled(self):
        sh = _shadow()
        sh._enabled = False
        rec = _base_recommendation(setpoint=21.0)
        result = sh.build_live_decision_pre_dispatch(rec)
        assert result is None


# ── Section 2: dispatch binding via dataclasses.replace() ─────────────────────

class TestDispatchBinding:
    """dataclasses.replace() correctly completes the record with dispatch fields."""

    def test_setpoint_dispatch_path(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.5)
        pre = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            pre,
            dispatch_attempted=True,
            dispatch_path="setpoint",
            dispatch_setpoint_c=21.5,
            dispatch_duty_pct=None,
            dispatch_succeeded=True,
            dispatch_ts="2026-01-01T12:00:00",
        )
        assert complete.dispatch_attempted is True
        assert complete.dispatch_path == "setpoint"
        assert complete.dispatch_setpoint_c == pytest.approx(21.5)
        assert complete.dispatch_succeeded is True
        assert complete.dispatch_ts == "2026-01-01T12:00:00"
        # Decision fields are unchanged
        assert complete.final_setpoint_c == pytest.approx(21.5)

    def test_direct_valve_dispatch_path(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, tpi_valve_direct=True, duty=65.0)
        pre = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            pre,
            dispatch_attempted=True,
            dispatch_path="direct_valve",
            dispatch_setpoint_c=21.0,
            dispatch_duty_pct=65.0,
            dispatch_succeeded=True,
        )
        assert complete.dispatch_path == "direct_valve"
        assert complete.dispatch_duty_pct == pytest.approx(65.0)

    def test_no_dispatch_when_inactive(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        pre = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(pre, dispatch_attempted=False, dispatch_path="none")
        assert complete.dispatch_attempted is False
        assert complete.dispatch_path == "none"
        assert complete.dispatch_setpoint_c is None

    def test_record_is_immutable(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        result = sh.build_live_decision_pre_dispatch(rec)
        with pytest.raises((TypeError, dataclasses.FrozenInstanceError)):
            result.dispatch_attempted = True  # type: ignore[misc]


# ── Section 3: trace derived from live record ─────────────────────────────────

class TestTraceFromLiveRecord:
    """compute_decision_trace_safe(live_record=...) produces correct trace."""

    def _run_trace(self, sh, rec, live_record):
        sh.compute_decision_trace_safe(rec, live_record=live_record)
        return sh.last_decision_trace

    def test_trace_not_none(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace is not None

    def test_final_setpoint_in_trace_matches_live_record(self):
        """trace.final_setpoint_c == live_record.final_setpoint_c (not resolver's value)."""
        sh = _shadow()
        rec = _base_recommendation(setpoint=22.3)
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["final_setpoint_c"] == pytest.approx(22.3)

    def test_baseline_setpoint_in_trace_matches_live_record(self):
        """trace.baseline_setpoint_c == live_record.baseline_setpoint_c."""
        sh = _shadow()
        # Boost applied: baseline=21.0, boosted final=22.5
        rec = _base_recommendation(setpoint=22.5, boost_applied=True,
                                   boost_offset=1.5, boost_candidate=1.5)
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["baseline_setpoint_c"] == pytest.approx(21.0)

    def test_final_device_setpoint_from_dispatch_setpoint(self):
        """final_device_setpoint_c = dispatch_setpoint_c when dispatch occurred."""
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.5)
        pre = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            pre, dispatch_attempted=True, dispatch_path="setpoint",
            dispatch_setpoint_c=21.5, dispatch_succeeded=True)
        trace = self._run_trace(sh, rec, complete)
        assert trace["final_device_setpoint_c"] == pytest.approx(21.5)

    def test_final_device_setpoint_falls_back_to_final_setpoint(self):
        """When no dispatch, final_device_setpoint_c = final_setpoint_c."""
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.5)
        pre = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(pre, dispatch_attempted=False, dispatch_path="none")
        trace = self._run_trace(sh, rec, complete)
        assert trace["final_device_setpoint_c"] == pytest.approx(21.5)

    def test_boost_applied_c_in_trace(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=22.5, boost_applied=True,
                                   boost_offset=1.5, boost_candidate=1.5)
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["boost_offset_c_applied"] == pytest.approx(1.5)

    def test_boost_candidate_in_trace(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=22.5, boost_applied=True,
                                   boost_offset=1.5, boost_candidate=1.8)
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["boost_offset_requested_c"] == pytest.approx(1.8)

    def test_boost_rejected_reason_in_trace(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, boost_applied=False,
                                   rejection_reason="window_open")
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["boost_rejected_reason"] == "window_open"

    def test_no_boost_applied(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, boost_applied=False)
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["applied_any"] is False

    def test_boost_applied_sets_applied_any(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=22.5, boost_applied=True,
                                   boost_offset=1.5, boost_candidate=1.5)
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["applied_any"] is True

    def test_preheat_in_trace(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, preheat_minutes=60.0,
                                   preheat_status="valid")
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["selected_preheat_min"] == pytest.approx(60.0)
        assert trace["preheat_status"] == "valid"

    def test_early_cutoff_in_trace(self):
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, early_cutoff_state="cutoff_applied")
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["early_cutoff_state"] == "cutoff_applied"
        assert trace["early_cutoff_hold_active"] is True

    def test_onset_delay_from_live_record(self):
        sh = _shadow(od=_od(10.0))
        rec = _base_recommendation(setpoint=21.0, preheat_minutes=90.0)
        pre = sh.build_live_decision_pre_dispatch(rec)
        trace = self._run_trace(sh, rec, pre)
        assert trace["effective_onset_delay_min"] == pytest.approx(10.0)
        assert trace["effective_room_heating_duration_min"] == pytest.approx(80.0)


# ── Section 4: adjust_recommendation_safe rejection tracking ──────────────────

class TestAdjustRecommendationRejectionTracking:
    """adjust_recommendation_safe() correctly sets provenance keys in recommendation."""

    def _make_shadow_control(self):
        """Shadow in CONTROL mode."""
        from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime.set_mode(LearningRuntimeMode.CONTROL)
        return sh

    def test_window_open_sets_rejection_reason(self):
        sh = self._make_shadow_control()
        rec = _base_recommendation(setpoint=21.0, window_open=True)
        sh.adjust_recommendation_safe(rec)
        assert rec["_boost_rejection_reason"] == "window_open"
        assert not rec.get("le2_boost_adjusted")

    def test_direct_valve_sets_rejection_reason(self):
        sh = self._make_shadow_control()
        rec = _base_recommendation(setpoint=21.0, tpi_valve_direct=True)
        sh.adjust_recommendation_safe(rec)
        assert rec["_boost_rejection_reason"] == "direct_valve"
        assert not rec.get("le2_boost_adjusted")

    def test_heating_failure_sets_rejection_reason(self):
        sh = self._make_shadow_control()
        rec = _base_recommendation(setpoint=21.0, heating_failure=True)
        sh.adjust_recommendation_safe(rec)
        assert rec["_boost_rejection_reason"] == "heating_failure"
        assert not rec.get("le2_boost_adjusted")

    def test_shadow_mode_no_rejection_reason(self):
        """Shadow mode (control_enabled=False): rejection reason stays None."""
        sh = _shadow()
        assert not sh.control_enabled
        rec = _base_recommendation(setpoint=21.0)
        sh.adjust_recommendation_safe(rec)
        # In shadow mode, function returns immediately without modifying reason
        assert rec.get("_boost_rejection_reason") is None
        assert not rec.get("le2_boost_adjusted")

    def test_no_prediction_sets_rejection_reason(self):
        sh = self._make_shadow_control()
        # No BOOST_FACTOR prediction in zone runtime → "no_prediction"
        rec = _base_recommendation(setpoint=21.0, target=20.0)
        sh.adjust_recommendation_safe(rec)
        # No BOOST_FACTOR prediction → no_prediction
        assert rec["_boost_rejection_reason"] == "no_prediction"

    def test_early_cutoff_sets_rejection_reason(self):
        sh = self._make_shadow_control()
        rec = _base_recommendation(setpoint=21.0, early_cutoff_state="cutoff_applied")
        sh.adjust_recommendation_safe(rec)
        assert rec["_boost_rejection_reason"] == "early_cutoff"

    def test_provenance_keys_always_initialized(self):
        """_boost_* keys are always written, even in shadow mode."""
        sh = _shadow()
        rec = {"trv_setpoint": 21.0, "effective_target": 21.0}
        sh.adjust_recommendation_safe(rec)
        assert "_boost_rejection_reason" in rec
        assert "_boost_candidate_c" in rec
        assert "_boost_applied_c" in rec


# ── Section 5: same-cycle consistency golden test ─────────────────────────────

class TestSameCycleConsistency:
    """Trace fields must never diverge from what the live coordinator decided."""

    def test_setpoint_equality_after_boost(self):
        """trace.final_setpoint_c == recommendation.trv_setpoint (after boost)."""
        sh = _shadow()
        original_sp = 21.0
        boost = 1.2
        final_sp = original_sp + boost
        rec = _base_recommendation(
            setpoint=final_sp, boost_applied=True,
            boost_offset=boost, boost_candidate=boost)
        pre = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            pre, dispatch_attempted=True, dispatch_path="setpoint",
            dispatch_setpoint_c=final_sp, dispatch_succeeded=True)
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert trace["final_setpoint_c"] == pytest.approx(rec["trv_setpoint"])
        assert trace["baseline_setpoint_c"] == pytest.approx(original_sp)
        assert trace["final_device_setpoint_c"] == pytest.approx(final_sp)

    def test_no_boost_setpoint_equality(self):
        """Without boost: baseline == final == dispatch setpoint."""
        sh = _shadow()
        sp = 20.5
        rec = _base_recommendation(setpoint=sp, boost_applied=False)
        pre = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            pre, dispatch_attempted=True, dispatch_path="setpoint",
            dispatch_setpoint_c=sp, dispatch_succeeded=True)
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert trace["final_setpoint_c"] == pytest.approx(sp)
        assert trace["baseline_setpoint_c"] == pytest.approx(sp)
        assert trace["final_device_setpoint_c"] == pytest.approx(sp)

    def test_dispatch_setpoint_is_coordinator_level_value(self):
        """dispatch_setpoint_c = coordinator's trv_setpoint (pre-per-entity-clamp).

        The per-entity device clamp in _apply_temperature() may reduce the
        effective setpoint per TRV. The trace captures the coordinator-level
        decision, not each entity's clamped value.
        """
        sh = _shadow()
        coordinator_sp = 22.0  # coordinator's decision
        rec = _base_recommendation(setpoint=coordinator_sp)
        pre = sh.build_live_decision_pre_dispatch(rec)
        # Simulate: per-entity clamp brings effective to 21.5, but dispatch_setpoint_c
        # records the coordinator-level value
        complete = dataclasses.replace(
            pre, dispatch_attempted=True, dispatch_path="setpoint",
            dispatch_setpoint_c=coordinator_sp,  # coordinator-level, not entity-level
            dispatch_succeeded=True)
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert trace["final_device_setpoint_c"] == pytest.approx(coordinator_sp)

    def test_direct_valve_trace_has_correct_rejection(self):
        """Direct-valve TRV: boost is rejected with 'direct_valve', duty is dispatch value."""
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0, tpi_valve_direct=True,
                                   duty=70.0, rejection_reason="direct_valve")
        pre = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            pre, dispatch_attempted=True, dispatch_path="direct_valve",
            dispatch_setpoint_c=21.0, dispatch_duty_pct=70.0, dispatch_succeeded=True)
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert trace["boost_rejected_reason"] == "direct_valve"
        assert trace["applied_any"] is False

    def test_no_dispatch_in_shadow_mode(self):
        """In shadow/observation mode: dispatch_attempted=False, trace still valid."""
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        pre = sh.build_live_decision_pre_dispatch(rec)
        # Simulate non-active-control: no dispatch attempted
        complete = dataclasses.replace(pre, dispatch_attempted=False, dispatch_path="none")
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert trace is not None
        # final_device_setpoint_c falls back to final_setpoint_c
        assert trace["final_device_setpoint_c"] == pytest.approx(21.0)


# ── Section 6: _last_live_decision lifecycle ──────────────────────────────────

class TestLastLiveDecisionLifecycle:
    """_last_live_decision must be reset each cycle; stale/cross-cycle/cross-zone leaks forbidden."""

    async def test_no_shadow_record_is_none(self):
        """No LE2 shadow → _last_live_decision stays None after cycle."""
        from tests.helpers_ha_runtime import make_recording_coordinator
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        await coord._async_update_data()
        assert coord._last_live_decision is None

    async def test_shadow_present_record_is_set(self):
        """With LE2 shadow → _last_live_decision is not None after cycle."""
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        await coord._async_update_data()
        assert coord._last_live_decision is not None

    async def test_record_reset_at_cycle_start(self):
        """_last_live_decision from cycle N must not survive into cycle N+1."""
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        await coord._async_update_data()
        record_cycle1 = coord._last_live_decision
        assert record_cycle1 is not None
        # Force build to fail on next cycle → record stays None
        original_build = sh.build_live_decision_pre_dispatch
        sh.build_live_decision_pre_dispatch = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("build fail"))
        await coord._async_update_data()
        # Must be None — previous record must NOT persist
        assert coord._last_live_decision is None
        sh.build_live_decision_pre_dispatch = original_build

    async def test_shadow_removed_no_stale_record(self):
        """Shadow removed between cycles → no stale record in coordinator."""
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        await coord._async_update_data()
        assert coord._last_live_decision is not None
        # Detach shadow
        coord._le2_shadow = None
        await coord._async_update_data()
        assert coord._last_live_decision is None

    async def test_summer_mode_no_dispatch_in_record(self):
        """Summer mode → dispatch_attempted=False, record still built."""
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        coord._is_summer = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        await coord._async_update_data()
        rec = coord._last_live_decision
        if rec is not None:  # record built even in summer
            assert rec.dispatch_attempted is False

    async def test_dispatch_exception_sets_failure_reason(self):
        """Dispatch exception → record has dispatch_succeeded=False, dispatch_failure_reason set."""
        from unittest.mock import AsyncMock
        from homeassistant.exceptions import HomeAssistantError
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._apply_temperature = AsyncMock(side_effect=RuntimeError("valve error"))
        with pytest.raises((HomeAssistantError, RuntimeError, Exception)):
            await coord._async_update_data()
        rec = coord._last_live_decision
        if rec is not None:  # record may be built even on dispatch failure
            assert rec.dispatch_succeeded is False
            assert rec.dispatch_failure_reason is not None

    async def test_dispatch_exception_clears_stale_record_on_retry(self):
        """After dispatch exception, _last_live_decision is reset at next cycle start."""
        from unittest.mock import AsyncMock
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        # Normal cycle
        await coord._async_update_data()
        assert coord._last_live_decision is not None
        # Force dispatch failure on next cycle
        original = coord._apply_temperature
        coord._apply_temperature = AsyncMock(side_effect=RuntimeError("boom"))
        try:
            await coord._async_update_data()
        except Exception:
            pass
        finally:
            coord._apply_temperature = original
        # After failure cycle: _last_live_decision must have been reset at start of that cycle.
        # The record may be set to failure state OR None; in either case it must not be
        # the successful record from the previous cycle.
        if coord._last_live_decision is not None:
            # If a record was built for the failed cycle, it must reflect the failure
            assert coord._last_live_decision.dispatch_succeeded is False


# ── Section 7: Production coordinator always provides live_record ──────────────

class TestProductionLiveRecordGuard:
    """Production coordinator must ALWAYS call compute_decision_trace_safe(live_record=...)."""

    async def test_production_live_record_always_provided(self):
        """Production coordinator passes live_record to compute_decision_trace_safe."""
        from unittest.mock import patch
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        captured_live_records = []
        original = sh.compute_decision_trace_safe
        def capturing(*a, **kw):
            captured_live_records.append(kw.get("live_record"))
            return original(*a, **kw)
        sh.compute_decision_trace_safe = capturing
        await coord._async_update_data()
        assert len(captured_live_records) == 1, "compute_decision_trace_safe called exactly once"
        # When called from coordinator, live_record must not be None
        assert captured_live_records[0] is not None, \
            "Production coordinator must always pass live_record — legacy resolver path must not run"

    async def test_no_live_record_no_trace_published(self):
        """When build_live_decision_pre_dispatch fails, clear_last_trace() is called."""
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        # First cycle: establish a valid trace
        await coord._async_update_data()
        assert sh.last_decision_trace is not None
        # Force build to fail on second cycle
        sh.build_live_decision_pre_dispatch = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("build fail"))
        await coord._async_update_data()
        # Stale trace from prior cycle must be cleared, not silently published
        assert sh.last_decision_trace is None, \
            "Stale trace from previous cycle must be cleared when no live record is available"

    async def test_legacy_path_not_triggered_in_coordinator(self):
        """compute_decision_trace_safe must NOT be called with live_record=None in coordinator."""
        from tests.helpers_ha_runtime import make_recording_coordinator, attach_shadow
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        legacy_calls = []
        original = sh.compute_decision_trace_safe
        def tracking(*a, **kw):
            if kw.get("live_record") is None:
                legacy_calls.append(kw)
            return original(*a, **kw)
        sh.compute_decision_trace_safe = tracking
        await coord._async_update_data()
        assert legacy_calls == [], \
            f"Legacy resolver path must NOT be triggered from coordinator; got {legacy_calls}"


# ── Section 8: decision_id chain ──────────────────────────────────────────────

class TestDecisionIdChain:
    """decision_id must be identical across the full provenance chain."""

    def test_live_record_carries_zone_runtime_decision_id(self):
        """LiveDecisionRecord.decision_id == zr.last_decision_id."""
        sh = _shadow()
        test_id = "zone::episode::2026-06-23T10:00:00Z::abc123"
        sh.runtime._zone(sh._zone).last_decision_id = test_id
        rec = _base_recommendation(setpoint=21.0)
        sh.adjust_recommendation_safe(rec)
        record = sh.build_live_decision_pre_dispatch(rec)
        assert record is not None
        assert record.decision_id == test_id

    def test_trace_carries_record_decision_id(self):
        """DecisionTrace.decision_id == LiveDecisionRecord.decision_id."""
        sh = _shadow()
        test_id = "zone::ep::2026-06-23T10:00:00Z::def456"
        sh.runtime._zone(sh._zone).last_decision_id = test_id
        rec = _base_recommendation(setpoint=21.0)
        sh.adjust_recommendation_safe(rec)
        record = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            record, dispatch_attempted=True, dispatch_path="setpoint",
            dispatch_setpoint_c=21.0, dispatch_succeeded=True)
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert trace is not None
        assert trace["decision_id"] == test_id

    def test_no_decision_id_yields_none_not_generated(self):
        """When zr.last_decision_id is None, no substitute ID is generated."""
        sh = _shadow()
        sh.runtime._zone(sh._zone).last_decision_id = None
        rec = _base_recommendation(setpoint=21.0)
        sh.adjust_recommendation_safe(rec)
        record = sh.build_live_decision_pre_dispatch(rec)
        assert record is not None
        assert record.decision_id is None  # not a random fallback

    def test_decision_id_stable_within_cycle(self):
        """decision_id in live record equals decision_id in derived trace — same cycle."""
        sh = _shadow()
        test_id = "stable::id::for::cycle"
        sh.runtime._zone(sh._zone).last_decision_id = test_id
        rec = _base_recommendation(setpoint=21.0)
        sh.adjust_recommendation_safe(rec)
        record = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            record, dispatch_attempted=True, dispatch_path="setpoint",
            dispatch_setpoint_c=21.0, dispatch_succeeded=True)
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert record.decision_id == trace["decision_id"] == test_id


# ── Section 9: privacy ────────────────────────────────────────────────────────

class TestPrivacy:
    """No entity IDs, zone IDs, or internal identifiers in the public trace dict."""

    def test_support_dict_no_zone_id(self):
        """DecisionTrace.support_dict() must not expose zone_id."""
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        sh.adjust_recommendation_safe(rec)
        record = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            record, dispatch_attempted=True, dispatch_path="setpoint",
            dispatch_setpoint_c=21.0, dispatch_succeeded=True)
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert "zone_id" not in trace, "zone_id must not appear in the published trace dict"

    def test_boost_candidate_and_applied_are_separate_fields(self):
        """boost_candidate_c (LE2 proposal) and boost_applied_c (guard outcome) are distinct.

        Reads directly from _base_recommendation — adjust_recommendation_safe is NOT called
        because it resets _boost_* keys. We test the LiveDecisionRecord field mapping here,
        not the adjustment logic (covered by TestAdjustRecommendationRejectionTracking).
        """
        sh = _shadow()
        candidate = 1.5
        applied = 1.2  # guard clamped to applied < candidate
        # _base_recommendation builds the dict with _boost_* keys already set.
        rec = _base_recommendation(setpoint=21.0 + applied, boost_applied=True,
                                   boost_candidate=candidate, boost_offset=applied)
        # Do NOT call adjust_recommendation_safe — it resets _boost_* at the start.
        record = sh.build_live_decision_pre_dispatch(rec)
        assert record is not None
        assert record.boost_candidate_c == pytest.approx(candidate)
        assert record.boost_applied_c == pytest.approx(applied)
        assert record.boost_candidate_c != record.boost_applied_c

    def test_dispatch_fields_not_in_support_dict(self):
        """Dispatch fields (HA-internal) must not appear in DecisionTrace.support_dict()."""
        from custom_components.thermosmart.learning.decision.contracts import DecisionTrace
        import dataclasses as _dc
        fields = {f.name for f in _dc.fields(DecisionTrace)}
        # These internal field names must not leak into support_dict keys
        sh = _shadow()
        rec = _base_recommendation(setpoint=21.0)
        sh.adjust_recommendation_safe(rec)
        record = sh.build_live_decision_pre_dispatch(rec)
        complete = dataclasses.replace(
            record, dispatch_attempted=True, dispatch_path="setpoint",
            dispatch_setpoint_c=21.0, dispatch_succeeded=True)
        sh.compute_decision_trace_safe(rec, live_record=complete)
        trace = sh.last_decision_trace
        assert "dispatch_attempted" not in trace
        assert "dispatch_path" not in trace
        assert "dispatch_succeeded" not in trace
        assert "dispatch_failure_reason" not in trace
