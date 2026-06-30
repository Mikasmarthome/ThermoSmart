"""S1a Item 10 - Section 10: Privacy scanner and export safety.

Verifies scan_payload detects forbidden keys (entity_id, email, IP, path),
Pseudonymizer is non-reversible, and QuantizationPolicy rounds correctly.
Runs on Windows (pure Python, no HA).
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.privacy import (
    PrivacyViolation,
    Pseudonymizer,
    PseudonymNamespace,
    QuantizationPolicy,
    scan_payload,
)


# ── 1. scan_payload detects forbidden keys ────────────────────────────────────


class TestScanPayloadForbiddenKeys:
    def test_entity_id_key_detected(self):
        violations = scan_payload({"entity_id": "climate.living_room"})
        assert any(v.kind == "forbidden_key" for v in violations)

    def test_device_id_key_detected(self):
        violations = scan_payload({"device_id": "abc123"})
        assert any(v.kind == "forbidden_key" for v in violations)

    def test_learning_zone_id_key_detected(self):
        violations = scan_payload({"learning_zone_id": "some-entry"})
        assert any(v.kind == "forbidden_key" for v in violations)

    def test_decision_id_key_detected(self):
        violations = scan_payload({"decision_id": "dec_abc"})
        assert any(v.kind == "forbidden_key" for v in violations)

    def test_episode_id_key_detected(self):
        violations = scan_payload({"episode_id": "ep_xyz"})
        assert any(v.kind == "forbidden_key" for v in violations)

    def test_nested_forbidden_key_detected(self):
        violations = scan_payload({"zone": {"entity_id": "sensor.temp"}})
        assert any("entity_id" in v.detail for v in violations)

    def test_safe_key_not_flagged(self):
        violations = scan_payload({"heat_rate_c_per_h": 1.5, "confidence": 0.8})
        assert violations == []


# ── 2. scan_payload detects forbidden value patterns ─────────────────────────


class TestScanPayloadForbiddenValues:
    def test_entity_id_pattern_in_value_detected(self):
        violations = scan_payload({"source": "climate.living_room"})
        assert any(v.kind == "entity_id_pattern" for v in violations)

    def test_email_in_value_detected(self):
        violations = scan_payload({"contact": "user@example.com"})
        assert any(v.kind == "email" for v in violations)

    def test_ipv4_in_value_detected(self):
        violations = scan_payload({"host": "192.168.1.100"})
        assert any(v.kind == "ip_address" for v in violations)

    def test_local_path_unix_detected(self):
        violations = scan_payload({"path": "/config/thermosmart.yaml"})
        assert any(v.kind == "local_path" for v in violations)

    def test_local_path_windows_detected(self):
        # Scanner regex requires double-backslash (as found in JSON-escaped paths)
        violations = scan_payload({"path": "C:\\\\Users\\\\user\\\\config.json"})
        assert any(v.kind == "local_path" for v in violations)

    def test_clean_numeric_values_not_flagged(self):
        violations = scan_payload({"temperature": 21.5, "confidence": 0.82, "count": 42})
        assert violations == []

    def test_non_finite_float_detected(self):
        import math
        violations = scan_payload({"value": float("inf")})
        assert any(v.kind == "non_finite" for v in violations)

    def test_nan_detected(self):
        import math
        violations = scan_payload({"value": float("nan")})
        assert any(v.kind == "non_finite" for v in violations)

    def test_list_value_scanned_recursively(self):
        violations = scan_payload({"items": ["climate.bedroom", "ok"]})
        assert any(v.kind == "entity_id_pattern" for v in violations)

    def test_nested_dict_scanned_recursively(self):
        violations = scan_payload({"meta": {"sub": {"contact": "a@b.com"}}})
        assert any(v.kind == "email" for v in violations)


# ── 3. Pseudonymizer ──────────────────────────────────────────────────────────


class TestPseudonymizer:
    def test_deterministic_with_same_salt(self):
        p = Pseudonymizer(salt=b"test-salt")
        a = p.pseudonymize(PseudonymNamespace.ZONE, "entry-123")
        b = p.pseudonymize(PseudonymNamespace.ZONE, "entry-123")
        assert a == b

    def test_different_values_different_pseudonyms(self):
        p = Pseudonymizer(salt=b"test-salt")
        a = p.pseudonymize(PseudonymNamespace.ZONE, "zone_a")
        b = p.pseudonymize(PseudonymNamespace.ZONE, "zone_b")
        assert a != b

    def test_different_namespaces_different_pseudonyms(self):
        p = Pseudonymizer(salt=b"test-salt")
        a = p.pseudonymize(PseudonymNamespace.ZONE, "same")
        b = p.pseudonymize(PseudonymNamespace.MODEL, "same")
        assert a != b

    def test_stable_true_when_salt_provided(self):
        p = Pseudonymizer(salt=b"salt")
        assert p.stable is True

    def test_stable_false_when_no_salt(self):
        p = Pseudonymizer()
        assert p.stable is False

    def test_pseudonym_not_equal_to_original(self):
        p = Pseudonymizer(salt=b"salt")
        original = "zone-entry-id"
        pseudo = p.pseudonymize(PseudonymNamespace.ZONE, original)
        assert original not in pseudo

    def test_none_input_raises(self):
        p = Pseudonymizer(salt=b"salt")
        with pytest.raises((ValueError, AttributeError, TypeError)):
            p.pseudonymize(PseudonymNamespace.ZONE, None)

    def test_length_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Pseudonymizer(salt=b"salt", length=3)
        with pytest.raises(ValueError):
            Pseudonymizer(salt=b"salt", length=50)

    def test_pseudonym_starts_with_namespace_prefix(self):
        p = Pseudonymizer(salt=b"salt")
        pseudo = p.pseudonymize(PseudonymNamespace.ZONE, "val")
        assert pseudo.startswith("z_")


# ── 4. QuantizationPolicy ────────────────────────────────────────────────────


class TestQuantizationPolicy:
    def test_temperature_quantized_to_step(self):
        q = QuantizationPolicy(temperature_step_c=0.1)
        result = q.quantize(20.17, q.temperature_step_c)
        expected = round(20.17 / 0.1) * 0.1
        assert abs(result - expected) < 1e-6

    def test_zero_step_returns_value_unchanged(self):
        q = QuantizationPolicy()
        result = q.quantize(20.17, 0.0)
        assert result == 20.17

    def test_non_finite_raises(self):
        q = QuantizationPolicy()
        import math
        with pytest.raises(ValueError):
            q.quantize(float("inf"), 0.1)

    def test_zero_stays_zero(self):
        q = QuantizationPolicy()
        result = q.quantize(0.0, 0.1)
        assert result == 0.0

    def test_negative_value_quantized(self):
        q = QuantizationPolicy()
        result = q.quantize(-5.23, 0.1)
        assert isinstance(result, float)
        assert result < 0
