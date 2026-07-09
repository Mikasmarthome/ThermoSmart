"""Phase 16 Support Export tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.export import (
    EXPORT_SCHEMA_VERSION,
    SUPPORT_EXPORT_NAME,
    ExportComponentStatus,
    build_support_export,
    to_canonical_json,
    validate_export_payload,
)
from custom_components.thermosmart.learning.privacy import scan_payload
from tests.helpers_orchestration import all_models, export_input, zone_input


class _Broken:
    def export(self, scope):
        raise RuntimeError("boom")


class TestSupportExport:
    def test_metadata(self):
        r = build_support_export(export_input())
        meta = r.payload["metadata"]
        assert meta["export_schema_version"] == EXPORT_SCHEMA_VERSION
        assert meta["scope"] == "support" and meta["scope_name"] == SUPPORT_EXPORT_NAME
        assert meta["le_engine_version"] == "2.0.0"

    def test_full_zone(self):
        r = build_support_export(export_input())
        z = r.payload["zones"][0]
        assert set(z["models"]) == set(all_models())
        assert z["models"]["heat_rate"]["status"] in ("ok", "no_evidence")

    def test_trv_only(self):
        z = zone_input(trv_only=True)
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["trv_only"] is True

    def test_missing_forecast_not_configured(self):
        from custom_components.thermosmart.learning.models import HeatRateModel
        z = zone_input(models={"heat_rate": HeatRateModel("living_room"), "forecast": None})
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["models"]["forecast"]["status"] == \
            ExportComponentStatus.NOT_CONFIGURED.value

    def test_model_without_samples_no_evidence(self):
        r = build_support_export(export_input())
        assert r.payload["zones"][0]["models"]["outcome"]["status"] == \
            ExportComponentStatus.NO_EVIDENCE.value

    def test_broken_model_isolated(self):
        z = zone_input(models={"heat_rate": all_models()["heat_rate"], "broken": _Broken()})
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["models"]["broken"]["status"] == \
            ExportComponentStatus.FAILED.value
        assert r.payload["zones"][0]["models"]["heat_rate"]["status"] in ("ok", "no_evidence")

    def test_confidence_caps_gates_present(self):
        r = build_support_export(export_input())
        conf = r.payload["zones"][0]["confidence"]
        assert conf and "applied_caps" in conf[0] and "gate" in conf[0]

    def test_no_ids(self):
        r = build_support_export(export_input())
        blob = to_canonical_json(r.payload)
        assert "living_room" not in blob and "decision_id" not in blob

    def test_no_raw_trajectories(self):
        r = build_support_export(export_input())
        blob = to_canonical_json(r.payload)
        assert "trajectory" not in blob

    def test_no_privacy_violations(self):
        r = build_support_export(export_input())
        assert scan_payload(r.payload) == []

    def test_validates(self):
        r = build_support_export(export_input())
        assert validate_export_payload(r.payload) == []

    def test_deterministic(self):
        a = to_canonical_json(build_support_export(export_input()).payload)
        b = to_canonical_json(build_support_export(export_input()).payload)
        assert a == b

    def test_canonical_json_no_nan(self):
        r = build_support_export(export_input())
        blob = to_canonical_json(r.payload)
        assert "NaN" not in blob and "Infinity" not in blob
