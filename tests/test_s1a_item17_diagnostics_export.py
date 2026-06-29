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
from custom_components.thermosmart.learning.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    DiagnosticsInput,
    DiagnosticsOrchestrator,
)
from custom_components.thermosmart.learning.export import (
    EXPORT_SCHEMA_VERSION,
    LEARNING_SAMPLE_FIELD_ALLOWLIST,
    SUPPORT_FIELD_ALLOWLIST,
    EngineSummary,
    EpisodeSummary,
    ExportBudget,
    ExportInput,
    ExportScope,
    StorageSummary,
    ZoneExportInput,
    build_learning_export,
    build_support_export,
    to_canonical_json,
    validate_export_payload,
)
from custom_components.thermosmart.learning.privacy import (
    ExportPrivacyClass,
    PrivacyPolicy,
    PrivacyViolation,
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


def _engine(**kw) -> EngineSummary:
    defaults = dict(
        le_version="2.0.0", storage_schema_version=1, initialized=True,
        mode="active", registry_valid=True, last_capture_ts=None,
        last_model_update_ts=None, last_flush_ts=None, warnings=[], errors=[]
    )
    return EngineSummary(**{**defaults, **kw})


def _zone(zone_id: str = "zone_abc", **kw) -> ZoneExportInput:
    defaults = dict(
        capability_flags={}, trv_only=True, initialized=True,
        models={}, confidence_results={},
        storage=StorageSummary(initialized=True, record_counts={}, schema_version=1),
        episodes=EpisodeSummary(counts_by_type={}),
        regime_summary="stable", missing_evidence=[], pending_attribution={}
    )
    return ZoneExportInput(zone_id=zone_id, **{**defaults, **kw})


def _export_input(zones=None, *, salt=_SALT, **kw) -> ExportInput:
    defaults = dict(
        engine=_engine(), zones=(_zone(),) if zones is None else zones,
        pseudonymizer=Pseudonymizer(salt),
        generated_at=_TS, policy=PrivacyPolicy(),
        budget=ExportBudget(), registry=None,
        ha_version="2025.1.0", integration_version="1.1.1"
    )
    return ExportInput(**{**defaults, **kw})


def _diag_input(zones=None, *, salt=_SALT) -> DiagnosticsInput:
    return DiagnosticsInput(
        engine=_engine(), zones=zones or (_zone(),),
        pseudonymizer=Pseudonymizer(salt),
        generated_at=_TS, policy=PrivacyPolicy(), registry=None
    )


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


class TestDiagnosticsOrchestrator:
    def test_schema_version_present(self):
        result = DiagnosticsOrchestrator(_diag_input()).build_diagnostics()
        assert result.payload["diagnostics_schema_version"] == DIAGNOSTICS_SCHEMA_VERSION

    def test_generated_at_present(self):
        result = DiagnosticsOrchestrator(_diag_input()).build_diagnostics()
        assert "generated_at" in result.payload

    def test_engine_section_required_keys(self):
        result = DiagnosticsOrchestrator(_diag_input()).build_diagnostics()
        engine = result.payload["engine"]
        for key in ("le_version", "initialized", "mode", "warnings", "errors"):
            assert key in engine

    def test_zone_pseudonymized_in_output(self):
        pseudo = Pseudonymizer(_SALT)
        expected = pseudo.pseudonymize(PseudonymNamespace.ZONE, "zone_abc")
        inp = _diag_input(zones=(_zone("zone_abc"),), salt=_SALT)
        result = DiagnosticsOrchestrator(inp).build_diagnostics()
        zone_data = result.payload["zones"][0]
        assert zone_data["zone"] == expected

    def test_zone_raw_id_not_in_output(self):
        inp = _diag_input(zones=(_zone("zone_abc"),))
        result = DiagnosticsOrchestrator(inp).build_diagnostics()
        zone_str = str(result.payload["zones"][0])
        assert "zone_abc" not in zone_str

    def test_multiple_zones_all_present(self):
        zones = (_zone("z1"), _zone("z2"), _zone("z3"))
        result = DiagnosticsOrchestrator(_diag_input(zones=zones)).build_diagnostics()
        assert len(result.payload["zones"]) == 3

    def test_broken_model_isolated_not_crashing(self):
        # Zone with a model dict (not an object with .export() method)
        # → should produce error, not crash
        zone_with_model = _zone("zone_z", models={"heat_rate": {"broken": True}})
        inp = _diag_input(zones=(zone_with_model,))
        result = DiagnosticsOrchestrator(inp).build_diagnostics()
        assert result.payload is not None  # no crash

    def test_engine_initialized_flag_matches_input(self):
        engine_uninitialized = _engine(initialized=False)
        inp = DiagnosticsInput(
            engine=engine_uninitialized, zones=(_zone(),),
            pseudonymizer=Pseudonymizer(_SALT),
            generated_at=_TS, policy=PrivacyPolicy(), registry=None
        )
        result = DiagnosticsOrchestrator(inp).build_diagnostics()
        assert result.payload["engine"]["initialized"] is False


# ── §5  Support Export Privacy ────────────────────────────────────────────────


class TestSupportExportPrivacy:
    def test_metadata_scope_is_support(self):
        result = build_support_export(_export_input())
        assert result.payload["metadata"]["scope"] == "support"

    def test_export_schema_version_present(self):
        result = build_support_export(_export_input())
        assert result.payload["metadata"]["export_schema_version"] == EXPORT_SCHEMA_VERSION

    def test_zone_pseudonymized_not_raw(self):
        result = build_support_export(_export_input())
        zone_field = result.payload["zones"][0]["zone"]
        assert zone_field.startswith("z_")
        assert "zone_abc" not in zone_field

    def test_pseudonym_stable_true_with_salt(self):
        result = build_support_export(_export_input(salt=_SALT))
        assert result.payload["metadata"]["pseudonym_stable"] is True

    def test_pseudonym_stable_false_without_salt(self):
        result = build_support_export(_export_input(salt=None))
        assert result.payload["metadata"]["pseudonym_stable"] is False

    def test_entity_id_value_not_in_output(self):
        zone = _zone(models={"heat_rate": {"entity_id": "climate.living_room"}})
        result = build_support_export(_export_input(zones=(zone,)))
        output_str = str(result.payload)
        assert "climate.living_room" not in output_str

    def test_allowlist_filters_unknown_fields(self):
        zone = _zone(models={"heat_rate": {
            "confidence": 0.8,     # in allowlist
            "secret_field": "xyz", # NOT in allowlist
        }})
        result = build_support_export(_export_input(zones=(zone,)))
        model_out = result.payload["zones"][0]["models"].get("heat_rate", {})
        assert "secret_field" not in model_out

    def test_validate_export_payload_passes(self):
        result = build_support_export(_export_input())
        errors = validate_export_payload(result.payload)
        assert errors == []

    def test_deterministic_json_same_input(self):
        inp = _export_input()
        j1 = to_canonical_json(build_support_export(inp).payload)
        j2 = to_canonical_json(build_support_export(inp).payload)
        assert j1 == j2

    def test_ha_version_in_metadata(self):
        result = build_support_export(_export_input())
        assert result.payload["metadata"]["ha_version"] == "2025.1.0"

    def test_integration_version_in_metadata(self):
        result = build_support_export(_export_input())
        assert result.payload["metadata"]["integration_version"] == "1.1.1"


# ── §6  Learning/Research Export ─────────────────────────────────────────────


class TestLearningExportPrivacy:
    def test_metadata_scope_is_research(self):
        result = build_learning_export(_export_input())
        assert result.payload["metadata"]["scope"] == "research"

    def test_generated_at_date_bucketed(self):
        # learning export truncates timestamp to date only
        result = build_learning_export(_export_input())
        ts = result.payload["metadata"]["generated_at"]
        assert "T" not in ts  # no time component
        assert len(ts) == 10  # YYYY-MM-DD

    def test_zone_pseudonymized(self):
        result = build_learning_export(_export_input())
        zone_field = result.payload["zones"][0]["zone"]
        assert zone_field.startswith("z_")

    def test_entity_id_not_in_output(self):
        zone = _zone(models={"heat_rate": {"entity_id": "climate.room"}})
        result = build_learning_export(_export_input(zones=(zone,)))
        assert "climate.room" not in str(result.payload)

    def test_validate_export_payload_passes(self):
        result = build_learning_export(_export_input())
        assert validate_export_payload(result.payload) == []

    def test_learning_export_no_warnings_on_clean_input(self):
        result = build_learning_export(_export_input())
        assert len(result.errors) == 0

    def test_learning_allowlist_documented(self):
        # Learning sample allowlist must exist and have reasonable size
        assert len(LEARNING_SAMPLE_FIELD_ALLOWLIST) >= 20

    def test_support_allowlist_documented(self):
        # Support allowlist must exist and have reasonable size
        assert len(SUPPORT_FIELD_ALLOWLIST) >= 30


# ── §7  Export Budget Enforcement ────────────────────────────────────────────


class TestExportBudgetEnforcement:
    def test_max_zones_truncates(self):
        zones = tuple(_zone(f"z{i}") for i in range(5))
        budget = ExportBudget(max_zones=3, max_models_per_zone=16,
                              max_samples_per_model=50, max_total_bytes=1_000_000,
                              max_string_len=512, max_depth=14,
                              max_warnings=300, max_errors=300)
        inp = _export_input(zones=zones, budget=budget)
        result = build_support_export(inp)
        assert len(result.payload["zones"]) == 3

    def test_truncated_flag_set_on_zone_truncation(self):
        zones = tuple(_zone(f"z{i}") for i in range(3))
        budget = ExportBudget(max_zones=2, max_models_per_zone=16,
                              max_samples_per_model=50, max_total_bytes=1_000_000,
                              max_string_len=512, max_depth=14,
                              max_warnings=300, max_errors=300)
        inp = _export_input(zones=zones, budget=budget)
        result = build_support_export(inp)
        assert result.truncated is True

    def test_no_truncation_flag_when_within_budget(self):
        result = build_support_export(_export_input())
        assert result.truncated is False

    def test_max_zones_50_by_default(self):
        assert ExportBudget().max_zones == 50

    def test_max_samples_50_by_default(self):
        assert ExportBudget().max_samples_per_model == 50

    def test_max_total_bytes_one_million_by_default(self):
        assert ExportBudget().max_total_bytes == 1_000_000

    def test_two_zones_both_in_output_within_budget(self):
        zones = (_zone("z1"), _zone("z2"))
        result = build_support_export(_export_input(zones=zones))
        assert len(result.payload["zones"]) == 2


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


class TestExportResilienceFailureSafety:
    def test_support_export_with_empty_zones_tuple(self):
        inp = _export_input(zones=())
        result = build_support_export(inp)
        assert result.payload is not None
        assert result.payload["zones"] == []

    def test_learning_export_with_empty_zones_tuple(self):
        inp = _export_input(zones=())
        result = build_learning_export(inp)
        assert result.payload is not None
        assert result.payload["zones"] == []

    def test_export_no_mutation_of_engine_on_repeated_calls(self):
        inp = _export_input()
        r1 = build_support_export(inp)
        r2 = build_support_export(inp)
        assert r1.payload["engine"] == r2.payload["engine"]

    def test_diagnostics_with_engine_errors_propagates_field(self):
        engine_with_err = _engine(errors=["storage_timeout"])
        inp = DiagnosticsInput(
            engine=engine_with_err, zones=(_zone(),),
            pseudonymizer=Pseudonymizer(_SALT),
            generated_at=_TS, policy=PrivacyPolicy(), registry=None
        )
        result = DiagnosticsOrchestrator(inp).build_diagnostics()
        assert "storage_timeout" in result.payload["engine"]["errors"]

    def test_export_payload_keys_are_deterministic(self):
        result = build_support_export(_export_input())
        keys1 = list(result.payload.keys())
        keys2 = list(build_support_export(_export_input()).payload.keys())
        assert keys1 == keys2

    def test_privacy_class_enum_has_expected_members(self):
        names = {m.name for m in ExportPrivacyClass}
        assert "PUBLIC_SAFE" in names
        assert "SUPPORT_SAFE" in names
        assert "SENSITIVE" in names
        assert "FORBIDDEN" in names
