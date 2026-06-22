"""Phase 17B: runtime data flows into the Phase-16 diagnostics/export orchestration."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.export import (
    EngineSummary,
    ExportInput,
    ZoneExportInput,
    build_learning_export,
    build_support_export,
    to_canonical_json,
    validate_export_payload,
)
from custom_components.thermosmart.learning.diagnostics import (
    DiagnosticsInput,
    build_diagnostics,
)
from custom_components.thermosmart.learning.privacy import Pseudonymizer, scan_payload
from tests.helpers_runtime_scenarios import heating_ramp_then_settle, runtime


def _export_input(rt, scope_salt=b"s"):
    zr = rt._zone("lz")
    result = zr.orchestrator.run()
    zone = ZoneExportInput(
        zone_id="lz", capability_flags={"has_forecast": False}, trv_only=True,
        models=dict(zr.orchestrator.models), confidence_results=result.confidence_results)
    return ExportInput(
        engine=EngineSummary(le_version="2.0.0", initialized=True, mode="shadow"),
        zones=(zone,), pseudonymizer=Pseudonymizer(salt=scope_salt),
        generated_at="2026-06-22T10:00:00+00:00")


class TestExportIntegration:
    def test_support_export_from_runtime(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        r = build_support_export(_export_input(rt))
        assert r.payload["zones"][0]["models"]["heat_rate"]["status"] in ("ok", "no_evidence")
        assert validate_export_payload(r.payload) == []

    def test_learning_export_from_runtime(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        r = build_learning_export(_export_input(rt))
        assert validate_export_payload(r.payload) == []

    def test_no_ids_leak(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        blob = to_canonical_json(build_support_export(_export_input(rt)).payload)
        assert "lz" not in blob or "z_" in blob  # zone pseudonymised
        assert "decision_id" not in blob

    def test_diagnostics_from_runtime(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        zr = rt._zone("lz")
        res = zr.orchestrator.run()
        d = build_diagnostics(DiagnosticsInput(
            engine=EngineSummary(le_version="2.0.0", initialized=True, mode="shadow"),
            zones=(ZoneExportInput(zone_id="lz", models=dict(zr.orchestrator.models),
                                   confidence_results=res.confidence_results),),
            pseudonymizer=Pseudonymizer(salt=b"s"),
            generated_at="2026-06-22T10:00:00+00:00"))
        names = [m["name"] for m in d.payload["zones"][0]["models"]]
        assert "heat_rate" in names
        assert scan_payload(d.payload) == []

    def test_export_privacy_clean(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        assert scan_payload(build_learning_export(_export_input(rt)).payload) == []
