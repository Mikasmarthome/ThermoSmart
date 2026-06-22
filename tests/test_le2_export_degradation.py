"""Phase 16 error-isolation, TRV-only, partial-engine, budget tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.export import (
    ExportBudget,
    ExportComponentStatus,
    build_learning_export,
    build_support_export,
    to_canonical_json,
    validate_export_payload,
)
from custom_components.thermosmart.learning.privacy import scan_payload
from tests.helpers_orchestration import all_models, export_input, zone_input
from custom_components.thermosmart.learning.models import HeatRateModel


class _Raises:
    def export(self, scope):
        raise RuntimeError("boom")


class _NaNModel:
    def export(self, scope):
        return {"general_rate_c_per_h": float("nan"), "sample_count": 5}


class _ForeignField:
    def export(self, scope):
        return {"entity_id": "climate.x", "sample_count": 1, "general_rate_c_per_h": 1.0}


class _BigSamples:
    def export(self, scope):
        if scope.value == "research":
            return {"samples": [{"physical": 0.5} for _ in range(500)], "model_version": 1}
        return {"sample_count": 500}


class TestErrorIsolation:
    def test_model_raises_isolated(self):
        z = zone_input(models={"heat_rate": HeatRateModel("living_room"), "bad": _Raises()})
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["models"]["bad"]["status"] == \
            ExportComponentStatus.FAILED.value
        assert validate_export_payload(r.payload) == []

    def test_model_nan_dropped(self):
        z = zone_input(models={"m": _NaNModel()})
        r = build_support_export(export_input(zones=(z,)))
        # non-finite value must not appear in the payload
        assert "NaN" not in to_canonical_json(r.payload)
        assert validate_export_payload(r.payload) == []

    def test_model_foreign_field_dropped(self):
        z = zone_input(models={"m": _ForeignField()})
        r = build_support_export(export_input(zones=(z,)))
        section = r.payload["zones"][0]["models"]["m"]
        assert "entity_id" not in section
        assert scan_payload(r.payload) == []

    def test_model_over_sample_budget_truncated(self):
        z = zone_input(models={"m": _BigSamples()})
        r = build_learning_export(export_input(zones=(z,),
                                               budget=ExportBudget(max_samples_per_model=10)))
        assert len(r.payload["zones"][0]["models"]["m"]["samples"]) == 10
        assert r.truncated

    def test_other_models_survive(self):
        z = zone_input(models={"heat_rate": HeatRateModel("living_room"), "bad": _Raises()})
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["models"]["heat_rate"]["status"] in ("ok", "no_evidence")


class TestTrvOnly:
    def test_trv_only_full_support(self):
        z = zone_input(models={"heat_rate": HeatRateModel("living_room"), "forecast": None},
                       trv_only=True)
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["models"]["forecast"]["status"] == \
            ExportComponentStatus.NOT_CONFIGURED.value
        assert validate_export_payload(r.payload) == []

    def test_trv_only_learning_export(self):
        z = zone_input(models={"heat_rate": HeatRateModel("living_room")}, trv_only=True)
        r = build_learning_export(export_input(zones=(z,)))
        assert validate_export_payload(r.payload) == []

    def test_missing_evidence_listed(self):
        z = zone_input(missing_evidence=("forecast", "outdoor_temp"))
        r = build_support_export(export_input(zones=(z,)))
        assert "forecast" in r.payload["zones"][0]["missing_evidence"]


class TestPartialEngine:
    def test_no_models(self):
        z = zone_input(models={})
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["models"] == {}
        assert validate_export_payload(r.payload) == []

    def test_uninitialized_zone(self):
        z = zone_input(initialized=False, storage=False, episodes=False)
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["initialized"] is False

    def test_no_zones(self):
        r = build_support_export(export_input(zones=()))
        assert r.payload["zones"] == []


class TestBudget:
    def test_zone_budget(self):
        zones = tuple(zone_input(f"z{i}") for i in range(5))
        r = build_support_export(export_input(zones=zones, budget=ExportBudget(max_zones=2)))
        assert len(r.payload["zones"]) == 2 and r.truncated

    def test_model_budget(self):
        z = zone_input(models=all_models())
        r = build_support_export(export_input(zones=(z,),
                                              budget=ExportBudget(max_models_per_zone=3)))
        assert len(r.payload["zones"][0]["models"]) == 3 and r.truncated

    def test_budget_truncation_deterministic(self):
        zones = tuple(zone_input(f"z{i}") for i in range(5))
        a = build_support_export(export_input(zones=zones, budget=ExportBudget(max_zones=2)))
        b = build_support_export(export_input(zones=zones, budget=ExportBudget(max_zones=2)))
        assert to_canonical_json(a.payload) == to_canonical_json(b.payload)
