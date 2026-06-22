"""Phase 16 diagnostics orchestration tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    DiagnosticsResult,
    ModelDiagnosticsStatus,
    build_diagnostics,
)
from custom_components.thermosmart.learning.export import RegistrySummary
from custom_components.thermosmart.learning.privacy import scan_payload
from tests.helpers_orchestration import (
    all_models,
    diagnostics_input,
    zone_input,
)


class _Broken:
    model_name = "broken"

    def diagnostics(self):
        raise RuntimeError("boom")

    def confidence(self):
        raise RuntimeError("boom")


class TestDiagnostics:
    def test_basic(self):
        r = build_diagnostics(diagnostics_input())
        assert isinstance(r, DiagnosticsResult)
        assert r.payload["diagnostics_schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
        assert r.payload["engine"]["mode"] == "advisory"

    def test_all_models_present(self):
        r = build_diagnostics(diagnostics_input())
        names = [m["name"] for m in r.payload["zones"][0]["models"]]
        assert set(names) == set(all_models())

    def test_no_models(self):
        z = zone_input(models={})
        r = build_diagnostics(diagnostics_input(zones=(z,)))
        assert r.payload["zones"][0]["models"] == []

    def test_only_heat_rate(self):
        from custom_components.thermosmart.learning.models import HeatRateModel
        z = zone_input(models={"heat_rate": HeatRateModel("living_room")})
        r = build_diagnostics(diagnostics_input(zones=(z,)))
        assert len(r.payload["zones"][0]["models"]) == 1

    def test_broken_model_isolated(self):
        z = zone_input(models={"heat_rate": all_models()["heat_rate"], "broken": _Broken()})
        r = build_diagnostics(diagnostics_input(zones=(z,)))
        entries = {m["name"]: m for m in r.payload["zones"][0]["models"]}
        assert entries["broken"]["status"] == ModelDiagnosticsStatus.FAILED.value
        assert entries["heat_rate"]["status"] in ("ok", "no_evidence")
        assert any("broken" in e for e in r.errors)

    def test_no_evidence_status(self):
        # fresh models have zero evidence -> NO_EVIDENCE (from confidence.evidence_count)
        r = build_diagnostics(diagnostics_input())
        statuses = {m["name"]: m["status"] for m in r.payload["zones"][0]["models"]}
        assert statuses["outcome"] == ModelDiagnosticsStatus.NO_EVIDENCE.value
        assert all(s in ("ok", "no_evidence") for s in statuses.values())

    def test_zone_pseudonymized(self):
        r = build_diagnostics(diagnostics_input())
        assert r.payload["zones"][0]["zone"].startswith("z_")
        assert "living_room" not in str(r.payload)

    def test_confidence_present(self):
        r = build_diagnostics(diagnostics_input())
        assert r.payload["zones"][0]["confidence"]
        assert r.payload["zones"][0]["confidence"][0]["purpose"] == "heat_rate"

    def test_registry_section(self):
        r = build_diagnostics(diagnostics_input(
            registry=RegistrySummary(models=("heat_rate",), graph_valid=True)))
        assert r.payload["registry"]["graph_valid"] is True

    def test_no_privacy_violations(self):
        r = build_diagnostics(diagnostics_input())
        assert scan_payload(r.payload) == []

    def test_deterministic(self):
        a = build_diagnostics(diagnostics_input()).payload
        b = build_diagnostics(diagnostics_input()).payload
        assert a == b

    def test_storage_and_episode_summaries(self):
        r = build_diagnostics(diagnostics_input())
        z = r.payload["zones"][0]
        assert z["storage"]["record_counts"] == {"room": 12}
        assert z["episodes"] == {"outcome": 3}
