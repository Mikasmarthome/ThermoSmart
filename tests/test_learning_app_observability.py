"""Phase A — read-only learning APPLICATION observability, restore-error
reporting (D6) and the authoritative TPI coef_source diagnostic (D7).

All diagnostics are pure reads of already-computed coordinator/runtime state
(the recommendation zone data and the id-free LiveDecisionRecord). These tests
verify the exposed values against the source decisions, multi-zone isolation,
missing/unavailable state, privacy, repeated-download stability, and that
collection triggers neither a control write nor runtime-state creation.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.export import _learning_application_export
from custom_components.thermosmart.learning.decision.contracts import LiveDecisionRecord
from custom_components.thermosmart.learning.privacy import scan_payload
from custom_components.thermosmart.sensor import ThermoSmartTpiSensor
from tests.helpers import make_coordinator
from tests.helpers_runtime_scenarios import runtime as scenario_runtime


def _ldr(**kw):
    # All LiveDecisionRecord required fields, overridable per test.
    base = dict(decision_id="d1", zone_id="z", ts="2025-01-01T00:00:00+00:00", mode="control",
                baseline_setpoint_c=21.0, final_setpoint_c=21.0,
                boost_applied=False, boost_candidate_c=None, boost_applied_c=None,
                boost_rejected_reason=None, preheat_minutes=0.0, preheat_status=None,
                heat_rate_c_per_h=None, heat_rate_confidence=None, heat_rate_source=None,
                onset_delay_min=None, onset_delay_source=None)
    base.update(kw)
    return LiveDecisionRecord(**base)


def _coord(*, zone=None, ldr=None):
    c = make_coordinator()
    c.data = {"zone": zone or {}}
    c._last_live_decision = ldr
    return c


# ── D7: authoritative TPI coef_source ────────────────────────────────────────

def _tpi_attrs(coef_int, coef_ext, source):
    c = make_coordinator()
    c.data = {"zone": {"tpi_coef_int": coef_int, "tpi_coef_ext": coef_ext,
                       "tpi_coef_source": source}}
    return ThermoSmartTpiSensor(c, c.entry).extra_state_attributes


class TestD7CoefSource:
    def test_learned_coef_equal_to_defaults_reports_learned(self):
        # Old guess reported "default" here (coef == 0.6/0.01); authoritative wins.
        a = _tpi_attrs(0.6, 0.01, "valid_learning")
        assert a["coef_source"] == "learned"
        assert a["coef_source_detail"] == "valid_learning"

    def test_deterministic_coef_differing_from_defaults_reports_default(self):
        # Old guess reported "learned" here (coef != 0.6); authoritative wins.
        a = _tpi_attrs(0.73, 0.02, "deterministic_baseline")
        assert a["coef_source"] == "default"
        assert a["coef_source_detail"] == "deterministic_baseline"

    def test_missing_source_reports_default(self):
        a = _tpi_attrs(0.6, 0.01, None)
        assert a["coef_source"] == "default"
        assert a["coef_source_detail"] is None

    def test_backward_compatible_attributes_present(self):
        a = _tpi_attrs(0.6, 0.01, "valid_learning")
        assert {"coef_int", "coef_ext", "coef_source", "valve_direct", "mode"} <= set(a)


# ── Application export: PREHEAT ───────────────────────────────────────────────

class TestPreheatApplication:
    def test_learned_selection_and_active_window(self):
        zone = {"preheat_active": True, "current_temp": 19.0,
                "preheat_minutes": 42.0, "preheat_status": "valid"}
        ph = _learning_application_export(_coord(zone=zone, ldr=_ldr()))["preheat"]
        assert ph["source"] == "learned"
        assert ph["selected_minutes"] == 42.0
        assert ph["learned_candidate_minutes"] == 42.0
        # deterministic baseline recomputed for comparison (deficit 2°C @ 1.5°C/h + onset)
        assert ph["baseline_minutes"] is not None and ph["baseline_minutes"] > 0
        assert ph["calculated"] is True
        assert ph["active_window"] is True        # calculated vs active distinguished

    def test_deterministic_fallback_not_active(self):
        zone = {"preheat_active": False, "current_temp": 19.0,
                "preheat_minutes": 85.0, "preheat_status": "deterministic_baseline"}
        ph = _learning_application_export(_coord(zone=zone, ldr=_ldr()))["preheat"]
        assert ph["source"] == "deterministic_baseline"
        assert ph["fallback_reason"] == "deterministic_baseline"
        assert ph["learned_candidate_minutes"] is None   # no learned value selected
        assert ph["active_window"] is False

    def test_calculated_but_not_active(self):
        zone = {"preheat_active": False, "preheat_minutes": 30.0, "preheat_status": "valid"}
        ph = _learning_application_export(_coord(zone=zone, ldr=_ldr()))["preheat"]
        assert ph["calculated"] is True
        assert ph["active_window"] is False


# ── Application export: TPI ───────────────────────────────────────────────────

class TestTpiApplication:
    def test_authoritative_source_and_candidate_vs_selected(self):
        zone = {"tpi_coef_source": "valid_learning", "tpi_coef_int": 0.6,
                "tpi_coef_ext": 0.01, "tpi_duty_cycle": 40.0,
                "tpi_coef_diag": {"coef_int_candidate": 0.62, "coef_int_previous": 0.58,
                                  "coef_int_default": 0.6, "blend_weight": 0.5,
                                  "transition_applied": True, "transition_reason": "step_limited",
                                  "relative_only": False}}
        tpi = _learning_application_export(_coord(zone=zone))["tpi"]
        assert tpi["coef_source"] == "valid_learning"
        assert tpi["coef_int_selected"] == 0.6
        assert tpi["coef_int_candidate"] == 0.62
        assert tpi["coef_int_previous"] == 0.58
        assert tpi["transition_applied"] is True
        assert tpi["transition_reason"] == "step_limited"
        assert tpi["duty_percent"] == 40.0
        assert tpi["is_selection_only"] is True   # selection, not dispatch

    def test_deterministic_source_not_inferred_from_default_coef(self):
        zone = {"tpi_coef_source": "deterministic_baseline", "tpi_coef_int": 0.6,
                "tpi_coef_ext": 0.01, "tpi_coef_diag": {}}
        tpi = _learning_application_export(_coord(zone=zone))["tpi"]
        assert tpi["coef_source"] == "deterministic_baseline"


# ── Application export: BOOST ─────────────────────────────────────────────────

class TestBoostApplication:
    def test_applied_with_new_write(self):
        ldr = _ldr(boost_candidate_c=2.0, boost_applied_c=1.2, boost_applied=True,
                   final_setpoint_c=22.2, baseline_setpoint_c=21.0,
                   dispatch_status="fully_succeeded", dispatch_targets_total=1,
                   dispatch_targets_failed=0, dispatch_succeeded=True, outcome_eligible=True)
        b = _learning_application_export(_coord(ldr=ldr))["boost"]
        assert b["proposed_offset_c"] == 2.0
        assert b["applied_offset_c"] == 1.2
        assert b["requested_setpoint_c"] == 22.2
        assert b["baseline_setpoint_c"] == 21.0
        assert b["policy_approved"] is True
        assert b["lifecycle_approved"] is True
        assert b["new_write"] is True
        assert b["within_tolerance_no_write"] is False
        assert b["service_call_succeeded"] is True
        assert b["outcome_eligible"] is True
        assert b["heating_effect_confirmed"] is None   # never inferred from dispatch

    def test_within_tolerance_no_write_is_observable(self):
        # D5a symptom made observable: boost applied, but no new service call.
        ldr = _ldr(boost_applied=True, boost_applied_c=1.0, dispatch_status="not_attempted",
                   dispatch_targets_total=0, dispatch_succeeded=False)
        b = _learning_application_export(_coord(ldr=ldr))["boost"]
        assert b["lifecycle_approved"] is True
        assert b["new_write"] is False
        assert b["within_tolerance_no_write"] is True
        assert b["service_call_succeeded"] is False
        assert b["heating_effect_confirmed"] is None

    def test_guard_rejected(self):
        ldr = _ldr(boost_candidate_c=2.0, boost_applied=False, boost_rejected_reason="guard_rejected")
        b = _learning_application_export(_coord(ldr=ldr))["boost"]
        assert b["policy_approved"] is False
        assert b["lifecycle_approved"] is False
        assert b["block_reason"] == "guard_rejected"

    def test_lifecycle_blocked_is_policy_approved_only(self):
        ldr = _ldr(boost_applied=False, boost_rejected_reason="lifecycle_blocked")
        b = _learning_application_export(_coord(ldr=ldr))["boost"]
        assert b["policy_approved"] is True
        assert b["lifecycle_approved"] is False

    def test_no_decision_record_boost_unavailable(self):
        out = _learning_application_export(_coord(zone={"tpi_coef_source": "valid_learning"}, ldr=None))
        assert out["available"] is True
        assert out["boost"]["available"] is False
        assert out["boost"]["reason"] == "no_decision_record"
        assert out["tpi"]["coef_source"] == "valid_learning"   # TPI still available from zone


# ── isolation / privacy / stability / no side effects ─────────────────────────

class TestObservabilityProperties:
    def test_multi_zone_isolation(self):
        a = _coord(zone={"tpi_coef_source": "valid_learning"},
                   ldr=_ldr(boost_applied=True, boost_applied_c=1.0, dispatch_targets_total=1))
        b = _coord(zone={"tpi_coef_source": "deterministic_baseline"},
                   ldr=_ldr(boost_applied=False, boost_rejected_reason="guard_rejected"))
        oa = _learning_application_export(a)
        ob = _learning_application_export(b)
        assert oa["tpi"]["coef_source"] == "valid_learning"
        assert ob["tpi"]["coef_source"] == "deterministic_baseline"
        assert oa["boost"]["lifecycle_approved"] is True
        assert ob["boost"]["lifecycle_approved"] is False

    def test_missing_data_is_graceful(self):
        c = make_coordinator()
        c.data = None
        c._last_live_decision = None
        out = _learning_application_export(c)
        assert out["available"] is True
        assert out["boost"]["available"] is False
        assert out["preheat"]["active_window"] is False

    def test_privacy_clean(self):
        ldr = _ldr(boost_candidate_c=2.0, boost_applied_c=1.2, boost_applied=True,
                   dispatch_status="fully_succeeded", dispatch_targets_total=1)
        out = _learning_application_export(_coord(
            zone={"preheat_active": True, "current_temp": 19.0, "preheat_status": "valid",
                  "tpi_coef_source": "valid_learning", "tpi_coef_int": 0.6,
                  "tpi_coef_diag": {"coef_int_candidate": 0.6}},
            ldr=ldr))
        assert scan_payload(out) == []

    def test_repeated_downloads_identical(self):
        ldr = _ldr(boost_applied=True, boost_applied_c=1.0, dispatch_targets_total=1)
        c = _coord(zone={"preheat_active": True, "current_temp": 19.0}, ldr=ldr)
        assert _learning_application_export(c) == _learning_application_export(c)

    def test_no_control_write_and_no_runtime_state(self):
        ldr = _ldr(boost_applied=True, boost_applied_c=1.0, dispatch_targets_total=1)
        c = _coord(zone={"preheat_active": True, "current_temp": 19.0,
                         "tpi_coef_source": "valid_learning"}, ldr=ldr)
        before_zone = dict(c.data["zone"])
        _learning_application_export(c)
        _learning_application_export(c)
        assert c.hass.services.async_call.called is False   # no actuator write
        assert c.data["zone"] == before_zone                # zone data unchanged
        assert c._last_live_decision is ldr                 # LDR unchanged


# ── D6: restore-error observability ───────────────────────────────────────────

class TestD6RestoreErrors:
    def test_clean_restore_reports_zero(self):
        rt = scenario_runtime()
        rt._restore_payload({"runtime_schema_version": 1, "zones": {}})
        h = rt.health()
        assert h.restore_errors == 0
        assert h.restore_schema_mismatch is False
        assert h.restore_error_reasons == ()

    def test_schema_mismatch_surfaced_and_payload_discarded(self):
        rt = scenario_runtime()
        rt._restore_payload({"runtime_schema_version": 999, "zones": {"z": {"models": {}}}})
        h = rt.health()
        assert h.restore_schema_mismatch is True
        assert h.restore_errors >= 1
        assert "schema_version_mismatch" in h.restore_error_reasons
        assert "z" not in rt._zones          # semantics preserved: whole payload discarded

    def test_corrupt_segment_surfaced_but_isolated(self):
        rt = scenario_runtime()
        # Valid envelope; a corrupt per-model state is isolated and surfaced,
        # not silently presented as a healthy (empty) model.
        rt._restore_payload({"runtime_schema_version": 1,
                             "zones": {"z": {"models": {"heat_rate": "corrupt-state"}}}})
        h = rt.health()
        assert h.restore_errors >= 1
        assert any("heat_rate" in r for r in h.restore_error_reasons)
        assert "z" in rt._zones              # zone still created; other segments intact

    def test_restart_starts_with_clean_budget(self):
        rt = scenario_runtime()
        assert rt.health().restore_errors == 0
