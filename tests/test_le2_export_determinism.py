"""Phase 16 determinism, registry, confidence-orchestration, multi-zone tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.export import (
    RegistrySummary,
    build_support_export,
    to_canonical_json,
)
from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceComponent,
    ConfidenceInput,
    ConfidencePurpose,
)
from tests.helpers_orchestration import (
    all_models,
    export_input,
    heat_rate_confidence,
    zone_input,
)


class TestDeterminism:
    def test_byte_identical(self):
        a = to_canonical_json(build_support_export(export_input()).payload)
        b = to_canonical_json(build_support_export(export_input()).payload)
        assert a == b

    def test_input_order_independent(self):
        models = all_models()
        reordered = dict(reversed(list(models.items())))
        za = zone_input(models=models)
        zb = zone_input(models=reordered)
        a = to_canonical_json(build_support_export(export_input(zones=(za,))).payload)
        b = to_canonical_json(build_support_export(export_input(zones=(zb,))).payload)
        assert a == b

    def test_other_salt_only_pseudonyms_change(self):
        a = build_support_export(export_input(salt=b"x")).payload
        b = build_support_export(export_input(salt=b"y")).payload
        assert a["zones"][0]["zone"] != b["zones"][0]["zone"]
        assert a["zones"][0]["models"]["heat_rate"]["status"] == \
            b["zones"][0]["models"]["heat_rate"]["status"]


class TestMultiZone:
    def _two_zones(self):
        return (zone_input("living_room"), zone_input("bedroom"))

    def test_two_zones(self):
        r = build_support_export(export_input(zones=self._two_zones()))
        assert len(r.payload["zones"]) == 2

    def test_zones_sorted_deterministically(self):
        r1 = build_support_export(export_input(zones=self._two_zones()))
        r2 = build_support_export(export_input(zones=tuple(reversed(self._two_zones()))))
        assert [z["zone"] for z in r1.payload["zones"]] == \
            [z["zone"] for z in r2.payload["zones"]]

    def test_separate_pseudonyms(self):
        r = build_support_export(export_input(zones=self._two_zones()))
        zones = [z["zone"] for z in r.payload["zones"]]
        assert len(set(zones)) == 2

    def test_no_cross_zone_leak(self):
        r = build_support_export(export_input(zones=self._two_zones()))
        blob = to_canonical_json(r.payload)
        assert "living_room" not in blob and "bedroom" not in blob

    def test_same_salt_stable_pseudonyms(self):
        a = build_support_export(export_input(zones=self._two_zones(), salt=b"k")).payload
        b = build_support_export(export_input(zones=self._two_zones(), salt=b"k")).payload
        assert [z["zone"] for z in a["zones"]] == [z["zone"] for z in b["zones"]]


class TestRegistryOrchestration:
    def test_registry_section(self):
        r = build_support_export(export_input(
            registry=RegistrySummary(raw_tracks=("room",), models=("heat_rate",),
                                     graph_valid=True)))
        assert r.payload["registry"]["graph_valid"] is True

    def test_invalid_graph_flagged(self):
        r = build_support_export(export_input(
            registry=RegistrySummary(models=("heat_rate",), graph_valid=False,
                                     issues=("model_unknown_track",))))
        assert r.payload["registry"]["graph_valid"] is False
        assert "model_unknown_track" in r.payload["registry"]["issues"]

    def test_no_registry(self):
        r = build_support_export(export_input())
        assert "registry" not in r.payload


class TestConfidenceOrchestration:
    def test_purposes_separate(self):
        agg = ConfidenceAggregator()
        hr = all_models()["heat_rate"]
        c_hr = agg.aggregate(ConfidenceInput(ConfidencePurpose.HEAT_RATE, components=(
            ConfidenceComponent(kind=ComponentKind.HEAT_RATE, contribution=hr.confidence()),)))
        c_af = agg.aggregate(ConfidenceInput(ConfidencePurpose.AFTERHEAT, components=(
            ConfidenceComponent(kind=ComponentKind.AFTERHEAT,
                                contribution=all_models()["afterheat"].confidence()),)))
        z = zone_input(confidence_results={"heat_rate": c_hr, "afterheat": c_af})
        r = build_support_export(export_input(zones=(z,)))
        purposes = [c["purpose"] for c in r.payload["zones"][0]["confidence"]]
        assert "heat_rate" in purposes and "afterheat" in purposes

    def test_caps_and_gates_preserved(self):
        z = zone_input()
        r = build_support_export(export_input(zones=(z,)))
        conf = r.payload["zones"][0]["confidence"][0]
        assert "applied_caps" in conf and "gate" in conf and "reason_codes" in conf

    def test_missing_confidence_result(self):
        z = zone_input(confidence_results={"heat_rate": None})
        r = build_support_export(export_input(zones=(z,)))
        assert r.payload["zones"][0]["confidence"][0]["status"] == "missing"

    def test_no_source_references(self):
        # the serialised confidence carries no component source ids, only safe kinds
        r = build_support_export(export_input())
        blob = to_canonical_json(r.payload)
        assert "source_episode" not in blob and "decision_id" not in blob
