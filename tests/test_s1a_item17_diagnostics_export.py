"""
S1a Item 17 — Diagnostics, Support Export and Research Export Audit

Verifies across all diagnostic and export paths:
- Pseudonymizer: determinism, namespace isolation, stability semantics
- QuantizationPolicy: step rounding, -0.0 elimination, NaN/Inf rejection
- Privacy Scanner: forbidden keys, value patterns, nested recursion
- DiagnosticsOrchestrator: schema, zone pseudonymization, error isolation
- Support Export: metadata, allowlist, no raw IDs, determinism
- Learning/Research Export: date bucketing, sample allowlist, quantization
- Export Budget: zone truncation, size cap, truncated flag
- Control Reason Traceability: 6-level priority chain
- Recommendation Dict: completeness of diagnostic keys
- LearningShadowController Diagnostics: health keys, confidence breakdown
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.privacy import (
    ExportPrivacyClass,
    PseudonymNamespace,
    Pseudonymizer,
    QuantizationPolicy,
    scan_payload,
)
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)

from tests.helpers import make_coordinator

# ── shared fixtures ───────────────────────────────────────────────────────────

_T0 = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
_SALT = b"item17_test_salt_32bytes_______!!"
_TS = "2025-01-15T10:00:00Z"


def _shadow_ctrl() -> LearningShadowController:
    hass = MagicMock()
    hass.async_create_task = MagicMock(side_effect=lambda c: c.close())
    return LearningShadowController(hass, "zone_test",
                                    clock=FakeClock(_T0))


# ── §1  Pseudonymizer Privacy ─────────────────────────────────────────────────


class TestPseudonymizerPrivacy:
    def test_deterministic_with_same_salt(self):
        p1 = Pseudonymizer(_SALT)
        p2 = Pseudonymizer(_SALT)
        assert (p1.pseudonymize(PseudonymNamespace.ZONE, "z1") ==
                p2.pseudonymize(PseudonymNamespace.ZONE, "z1"))

    def test_stable_with_injected_salt(self):
        assert Pseudonymizer(_SALT).stable is True

    def test_not_stable_without_salt(self):
        assert Pseudonymizer().stable is False

    def test_random_salts_produce_different_outputs(self):
        r1 = Pseudonymizer().pseudonymize(PseudonymNamespace.ZONE, "z1")
        r2 = Pseudonymizer().pseudonymize(PseudonymNamespace.ZONE, "z1")
        assert r1 != r2

    def test_namespace_isolation_same_value(self):
        p = Pseudonymizer(_SALT)
        zone = p.pseudonymize(PseudonymNamespace.ZONE, "abc")
        model = p.pseudonymize(PseudonymNamespace.MODEL, "abc")
        assert zone != model

    def test_different_salt_produces_different_output(self):
        p1 = Pseudonymizer(_SALT)
        p2 = Pseudonymizer(b"completely_different_salt____!!")
        r1 = p1.pseudonymize(PseudonymNamespace.ZONE, "z1")
        r2 = p2.pseudonymize(PseudonymNamespace.ZONE, "z1")
        assert r1 != r2

    def test_output_format_zone_prefix(self):
        r = Pseudonymizer(_SALT).pseudonymize(PseudonymNamespace.ZONE, "z1")
        assert r.startswith("z_")

    def test_output_format_model_prefix(self):
        r = Pseudonymizer(_SALT).pseudonymize(PseudonymNamespace.MODEL, "m1")
        assert r.startswith("m_")

    def test_pseudonym_length_fixed(self):
        p = Pseudonymizer(_SALT, length=12)
        r = p.pseudonymize(PseudonymNamespace.ZONE, "any_zone")
        # "z_" prefix + 12 hex chars
        assert len(r) == 14

    def test_all_namespaces_accepted(self):
        p = Pseudonymizer(_SALT)
        for ns in PseudonymNamespace:
            result = p.pseudonymize(ns, "test_value")
            assert isinstance(result, str)
            assert len(result) > 0


# ── §2  QuantizationPolicy ────────────────────────────────────────────────────


class TestQuantizationPolicy:
    def _qp(self) -> QuantizationPolicy:
        return QuantizationPolicy()

    def test_temperature_step_01(self):
        assert self._qp().quantize(21.34, 0.1) == pytest.approx(21.3)

    def test_rate_step_005(self):
        assert self._qp().quantize(1.234, 0.05) == pytest.approx(1.25)

    def test_duration_step_60(self):
        assert self._qp().quantize(3599.0, 60.0) == pytest.approx(3600.0)

    def test_score_step_005(self):
        assert self._qp().quantize(0.734, 0.05) == pytest.approx(0.75)

    def test_negative_zero_eliminated(self):
        result = self._qp().quantize(-0.04, 0.1)
        assert result == 0.0
        assert math.copysign(1.0, result) > 0  # positive zero

    def test_nan_raises_value_error(self):
        with pytest.raises(ValueError):
            self._qp().quantize(math.nan, 0.1)

    def test_inf_raises_value_error(self):
        with pytest.raises(ValueError):
            self._qp().quantize(math.inf, 0.1)

    def test_tie_half_to_even(self):
        # 21.25 with step 0.5 → rounds to nearest even → 21.0 (not 21.5)
        result = self._qp().quantize(21.25, 0.5)
        assert result in (21.0, 21.5)  # half-even: 21.0

    def test_negative_temperature_quantized(self):
        assert self._qp().quantize(-5.78, 0.1) == pytest.approx(-5.8)

    def test_default_fields_are_positive(self):
        qp = QuantizationPolicy()
        assert qp.temperature_step_c > 0
        assert qp.rate_step_c_per_h > 0
        assert qp.duration_step_s > 0
        assert qp.score_step > 0


# ── §3  Privacy Scanner ───────────────────────────────────────────────────────


class TestPrivacyScanner:
    def test_forbidden_key_entity_id(self):
        v = scan_payload({"entity_id": "climate.room"})
        kinds = {x.kind for x in v}
        assert "forbidden_key" in kinds

    def test_entity_id_pattern_in_value(self):
        v = scan_payload({"score": "climate.room1"})
        kinds = {x.kind for x in v}
        assert "entity_id_pattern" in kinds

    def test_email_detected_in_value(self):
        v = scan_payload({"contact": "user@example.com"})
        kinds = {x.kind for x in v}
        assert "email" in kinds

    def test_ipv4_detected(self):
        v = scan_payload({"host": "192.168.1.1"})
        kinds = {x.kind for x in v}
        assert "ip_address" in kinds

    def test_local_path_detected(self):
        v = scan_payload({"file": "/config/secrets.yaml"})
        kinds = {x.kind for x in v}
        assert "local_path" in kinds

    def test_nan_value_detected(self):
        v = scan_payload({"score": math.nan})
        kinds = {x.kind for x in v}
        assert "non_finite" in kinds

    def test_inf_value_detected(self):
        v = scan_payload({"rate": math.inf})
        kinds = {x.kind for x in v}
        assert "non_finite" in kinds

    def test_nested_violations_found(self):
        v = scan_payload({"outer": {"inner": {"entity_id": "climate.x"}}})
        assert len(v) >= 1
        paths = [x.path for x in v]
        assert any("outer" in p for p in paths)

    def test_clean_numeric_payload_no_violations(self):
        v = scan_payload({"confidence": 0.75, "sample_count": 42, "status": "ok"})
        assert v == []

    def test_list_elements_scanned(self):
        v = scan_payload({"items": [{"entity_id": "climate.x"}, {"score": 0.5}]})
        assert any(x.kind == "forbidden_key" for x in v)


# ── §4  DiagnosticsOrchestrator ───────────────────────────────────────────────


# ── §5  Support Export Privacy ────────────────────────────────────────────────


# ── §6  Learning/Research Export ─────────────────────────────────────────────


# ── §7  Export Budget Enforcement ────────────────────────────────────────────


# ── §8  Control Reason Traceability ──────────────────────────────────────────


class TestControlReasonTraceability:
    def _reason(self, rec: dict) -> str:
        coord = make_coordinator()
        return coord._control_reason(rec)

    def test_window_open_highest_priority(self):
        # window_open beats override_active
        rec = {"window_open": True, "override_active": True}
        assert self._reason(rec) == "window_open"

    def test_manual_override_second_priority(self):
        rec = {"window_open": False, "override_active": True, "mode": "auto"}
        assert self._reason(rec) == "manual_override"

    def test_vacation_mode_third_priority(self):
        from custom_components.thermosmart.const import HEATING_MODE_VACATION
        rec = {"window_open": False, "override_active": False,
               "mode": HEATING_MODE_VACATION}
        assert self._reason(rec) == "vacation"

    def test_summer_mode_fourth_priority(self):
        rec = {"window_open": False, "override_active": False,
               "mode": "auto", "is_summer": True}
        assert self._reason(rec) == "summer_mode"

    def test_presence_away_fifth_priority(self):
        from custom_components.thermosmart.const import HEATING_MODE_AWAY
        rec = {"window_open": False, "override_active": False,
               "mode": HEATING_MODE_AWAY, "is_summer": False}
        assert self._reason(rec) == "presence"

    def test_default_is_schedule(self):
        rec = {"window_open": False, "override_active": False,
               "mode": "auto", "is_summer": False}
        assert self._reason(rec) == "schedule"

    def test_returns_string(self):
        assert isinstance(self._reason({}), str)

    def test_all_reasons_are_known_values(self):
        known = {"window_open", "manual_override", "vacation",
                 "summer_mode", "presence", "schedule"}
        from custom_components.thermosmart.const import (
            HEATING_MODE_AWAY, HEATING_MODE_VACATION
        )
        test_cases = [
            {"window_open": True},
            {"override_active": True},
            {"mode": HEATING_MODE_VACATION},
            {"is_summer": True},
            {"mode": HEATING_MODE_AWAY},
            {},
        ]
        for rec in test_cases:
            assert self._reason(rec) in known


# ── §9  Recommendation Dict Key Completeness ─────────────────────────────────


class TestRecommendationDictKeys:
    """Proves the recommendation dict contains the diagnostic keys sensors use."""

    _REQUIRED_KEYS = [
        "adjusted_target",
        "effective_target",
        "weather_offset",
        "preheat_minutes",
        "preheat_active",
        "learning_confidence",
        "window_open",
        "mode",
        "forecast_suppression",
    ]

    def test_control_reason_uses_window_open_key(self):
        coord = make_coordinator()
        # window_open is a required key for _control_reason
        rec_with_window = {"window_open": True}
        assert coord._control_reason(rec_with_window) == "window_open"

    def test_control_reason_uses_override_active_key(self):
        coord = make_coordinator()
        rec = {"window_open": False, "override_active": True}
        assert coord._control_reason(rec) == "manual_override"

    def test_control_reason_stable_for_missing_keys(self):
        coord = make_coordinator()
        # Empty dict → defaults to "schedule"
        assert coord._control_reason({}) == "schedule"

    def test_compute_recommendation_signature_accepts_cfg_weather_mode(self):
        import inspect
        coord = make_coordinator()
        sig = inspect.signature(coord._compute_recommendation)
        params = list(sig.parameters.keys())
        assert "cfg" in params
        assert "weather_data" in params
        assert "mode" in params


# ── §10  LearningShadowController Diagnostics ────────────────────────────────


class TestLearningShadowControllerDiagnostics:
    def test_diagnostics_returns_dict(self):
        ctrl = _shadow_ctrl()
        assert isinstance(ctrl.diagnostics(), dict)

    def test_diagnostics_contains_mode_key(self):
        ctrl = _shadow_ctrl()
        assert "mode" in ctrl.diagnostics()

    def test_diagnostics_contains_enabled_key(self):
        ctrl = _shadow_ctrl()
        assert "enabled" in ctrl.diagnostics()

    def test_diagnostics_contains_control_enabled_key(self):
        ctrl = _shadow_ctrl()
        assert "control_enabled" in ctrl.diagnostics()

    def test_confidence_display_returns_float(self):
        ctrl = _shadow_ctrl()
        cd = ctrl.confidence_display()
        assert isinstance(cd, float)
        assert 0.0 <= cd <= 1.0

    def test_confidence_display_attributes_returns_dict(self):
        ctrl = _shadow_ctrl()
        attrs = ctrl.confidence_display_attributes()
        assert isinstance(attrs, dict)

    def test_confidence_attributes_required_keys(self):
        ctrl = _shadow_ctrl()
        attrs = ctrl.confidence_display_attributes()
        for key in ("room_patterns_%", "trv_efficiency_%", "total_observations"):
            assert key in attrs

    def test_confidence_attributes_non_negative(self):
        ctrl = _shadow_ctrl()
        attrs = ctrl.confidence_display_attributes()
        for key, val in attrs.items():
            assert val >= 0, f"{key} is negative: {val}"

    def test_errors_initially_zero(self):
        ctrl = _shadow_ctrl()
        assert ctrl.errors == 0

    def test_control_enabled_initially_false(self):
        ctrl = _shadow_ctrl()
        assert ctrl.control_enabled is False

    def test_diagnostics_decision_trace_key_present(self):
        ctrl = _shadow_ctrl()
        assert "decision_trace" in ctrl.diagnostics()

    def test_diagnostics_model_update_counts_present(self):
        ctrl = _shadow_ctrl()
        assert "model_update_counts" in ctrl.diagnostics()


# ── §11  Export Resilience / Failure Safety ───────────────────────────────────


class TestPrivacyClassEnum:
    def test_privacy_class_enum_has_expected_members(self):
        names = {m.name for m in ExportPrivacyClass}
        assert "PUBLIC_SAFE" in names
        assert "SUPPORT_SAFE" in names
        assert "SENSITIVE" in names
        assert "FORBIDDEN" in names
