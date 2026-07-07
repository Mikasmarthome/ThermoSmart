"""Phase 16 privacy: pseudonymizer, scanner, allowlist, classes."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.privacy import (
    PRIVACY_POLICY_VERSION,
    ExportPrivacyClass,
    Pseudonymizer,
    PseudonymNamespace,
    QuantizationPolicy,
    scan_payload,
)
from custom_components.thermosmart.learning.export import (
    build_learning_export,
    build_support_export,
    to_canonical_json,
)
from tests.helpers_orchestration import export_input, zone_input
from custom_components.thermosmart.learning.models import HeatRateModel


class TestPseudonymizer:
    def test_deterministic_with_salt(self):
        a = Pseudonymizer(salt=b"s")
        b = Pseudonymizer(salt=b"s")
        assert a.pseudonymize(PseudonymNamespace.ZONE, "lr") == \
            b.pseudonymize(PseudonymNamespace.ZONE, "lr")

    def test_different_salt_different_pseudonym(self):
        a = Pseudonymizer(salt=b"s1")
        b = Pseudonymizer(salt=b"s2")
        assert a.pseudonymize(PseudonymNamespace.ZONE, "lr") != \
            b.pseudonymize(PseudonymNamespace.ZONE, "lr")

    def test_namespaces_isolated(self):
        p = Pseudonymizer(salt=b"s")
        assert p.pseudonymize(PseudonymNamespace.ZONE, "x") != \
            p.pseudonymize(PseudonymNamespace.MODEL, "x")

    def test_not_reversible_no_plaintext(self):
        p = Pseudonymizer(salt=b"s")
        out = p.pseudonymize(PseudonymNamespace.ZONE, "living_room")
        assert "living_room" not in out

    def test_no_salt_random_but_stable_in_instance(self):
        p = Pseudonymizer()
        assert not p.stable
        assert p.pseudonymize(PseudonymNamespace.ZONE, "x") == \
            p.pseudonymize(PseudonymNamespace.ZONE, "x")

    def test_no_salt_differs_across_instances(self):
        assert Pseudonymizer().pseudonymize(PseudonymNamespace.ZONE, "x") != \
            Pseudonymizer().pseudonymize(PseudonymNamespace.ZONE, "x")


class TestScanner:
    def test_entity_id_in_value(self):
        v = scan_payload({"x": "climate.living_room"})
        assert any(s.kind == "entity_id_pattern" for s in v)

    def test_forbidden_key(self):
        v = scan_payload({"entity_id": "abc"})
        assert any(s.kind == "forbidden_key" for s in v)

    def test_decision_id_key(self):
        assert any(s.kind == "forbidden_key" for s in scan_payload({"decision_id": "d1"}))

    def test_email(self):
        assert any(s.kind == "email" for s in scan_payload({"x": "a@b.com"}))

    def test_ip(self):
        assert any(s.kind == "ip_address" for s in scan_payload({"x": "192.168.0.1"}))

    def test_path(self):
        assert any(s.kind == "local_path" for s in scan_payload({"x": "/config/secrets.yaml"}))

    def test_nested_forbidden(self):
        v = scan_payload({"a": {"b": [{"event_id": "e"}]}})
        assert any(s.kind == "forbidden_key" for s in v)

    def test_nan_flagged(self):
        assert any(s.kind == "non_finite" for s in scan_payload({"x": float("nan")}))

    def test_clean_payload_no_violations(self):
        assert scan_payload({"value": 0.5, "level": "high", "count": 3}) == []


class TestAllowlist:
    def test_unknown_field_dropped_and_warned(self):
        class _Model:
            def export(self, scope):
                return {"general_rate_c_per_h": 1.0, "secret_entity": "climate.x",
                        "sample_count": 5}

        z = zone_input(models={"heat_rate": _Model()})
        r = build_support_export(export_input(zones=(z,)))
        section = r.payload["zones"][0]["models"]["heat_rate"]
        assert "secret_entity" not in section
        assert "general_rate_c_per_h" in section
        assert any(w.code == "dropped_field" for w in r.warnings)

    def test_allowlist_beats_blacklist(self):
        # even if a value looks safe, a non-allowlisted key never appears
        class _Model:
            def export(self, scope):
                return {"weird_future_field": 1.0, "sample_count": 1}

        z = zone_input(models={"heat_rate": _Model()})
        r = build_support_export(export_input(zones=(z,)))
        assert "weird_future_field" not in r.payload["zones"][0]["models"]["heat_rate"]

    def test_privacy_class_values_stable(self):
        assert {c.value for c in ExportPrivacyClass} == {
            "public_safe", "support_safe", "learning_safe", "sensitive", "forbidden"}

    def test_registry_and_export_privacy_classes_unambiguous(self):
        # both types coexist and are importable without ambiguity
        from custom_components.thermosmart.learning.registry import PrivacyClass as RegPC
        from custom_components.thermosmart.learning.privacy import ExportPrivacyClass as ExpPC
        assert RegPC is not ExpPC
        assert {c.value for c in RegPC} == {"low", "behavioral"}
        assert ExpPC.FORBIDDEN.value == "forbidden"

    def test_policy_version(self):
        assert PRIVACY_POLICY_VERSION == 1


class TestQuantization:
    def test_zero_preserved(self):
        assert QuantizationPolicy().quantize(0.0, 0.1) == 0.0

    def test_negative_preserved(self):
        assert QuantizationPolicy().quantize(-0.34, 0.1) == pytest.approx(-0.3, abs=1e-9)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            QuantizationPolicy().quantize(float("nan"), 0.1)

    def test_grid(self):
        assert QuantizationPolicy().quantize(0.37, 0.1) == pytest.approx(0.4, abs=1e-9)


class TestPseudonymInExport:
    def test_zone_pseudonymized_in_payload(self):
        r = build_support_export(export_input())
        assert r.payload["zones"][0]["zone"].startswith("z_")

    def test_different_salt_changes_only_pseudonyms(self):
        a = build_support_export(export_input(salt=b"a")).payload
        b = build_support_export(export_input(salt=b"b")).payload
        assert a["zones"][0]["zone"] != b["zones"][0]["zone"]
        # non-pseudonym content (engine mode) stays identical
        assert a["engine"]["mode"] == b["engine"]["mode"]
