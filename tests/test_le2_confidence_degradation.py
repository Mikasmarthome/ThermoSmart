"""Phase 12 TRV-only, degradation, diagnostics/export tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceCap,
    ConfidenceInput,
    ConfidencePurpose,
    ConfidenceStatus,
    EvidenceDomain,
    breakdown_to_research_export,
    breakdown_to_support_export,
)
from tests.helpers_confidence import component


def A():
    return ConfidenceAggregator()


def inp(purpose, components=(), **kw):
    return ConfidenceInput(purpose=purpose, components=tuple(components), **kw)


def caps_of(r):
    return {c.cap for c in r.applied_caps}


class TestTrvOnly:
    def test_heat_rate_high_without_weather(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        assert r.value > 0.7  # local evidence is enough

    def test_relative_heat_loss_capped_but_usable(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_LOSS, [
            component(ComponentKind.HEAT_LOSS, 0.95, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        assert ConfidenceCap.TRV_ONLY_RELATIVE in caps_of(r)
        assert r.value > 0.0

    def test_afterheat_sensible(self):
        r = A().aggregate(inp(ConfidencePurpose.AFTERHEAT, [
            component(ComponentKind.AFTERHEAT, 0.85, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        assert r.value > 0.7

    def test_preheat_local_high_without_forecast(self):
        # strong local evidence, no forecast configured, local path -> high, uncapped
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.HEAT_LOSS, 0.85, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,))]))
        assert r.value > 0.7
        assert ConfidenceCap.FORECAST_UNAVAILABLE not in caps_of(r)
        assert ConfidenceCap.TRV_ONLY_RELATIVE not in caps_of(r)
        assert r.control_allowed

    def test_preheat_forecast_path_capped_when_unavailable(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, forecast_required=True, components=[
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,))]))
        assert ConfidenceCap.FORECAST_UNAVAILABLE in caps_of(r)
        assert r.value <= A()._params.forecast_unavailable_cap + 1e-9

    def test_missing_optional_visible(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50)]))
        assert "forecast" in r.missing_optional

    def test_no_crash_minimal(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.6, evidence_count=10)]))
        assert r.value >= 0.0

    def test_weather_present_lifts_cap(self):
        no_weather = A().aggregate(inp(ConfidencePurpose.HEAT_LOSS, [
            component(ComponentKind.HEAT_LOSS, 0.95, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        with_weather = A().aggregate(inp(ConfidencePurpose.HEAT_LOSS, [
            component(ComponentKind.HEAT_LOSS, 0.95, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.OUTDOOR, 0.9, domains=(EvidenceDomain.WEATHER,))]))
        assert ConfidenceCap.TRV_ONLY_RELATIVE not in caps_of(with_weather)
        assert with_weather.value >= no_weather.value


class TestDegradation:
    def _req(self, status):
        return inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50, status=status)])

    def test_unavailable_required(self):
        r = A().aggregate(self._req(ConfidenceStatus.UNAVAILABLE))
        assert "heat_rate" in r.missing_required
        assert ConfidenceCap.MISSING_REQUIRED in caps_of(r)

    def test_stale_required(self):
        r = A().aggregate(self._req(ConfidenceStatus.STALE))
        assert ConfidenceCap.STALE_DATA in caps_of(r)

    def test_invalid_required(self):
        r = A().aggregate(self._req(ConfidenceStatus.INVALID))
        assert ConfidenceCap.INVALID_DATA in caps_of(r)
        assert r.value <= A()._params.invalid_cap + 1e-9

    def test_not_configured_optional_no_penalty_cap(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50),
            component(ComponentKind.OUTDOOR, 0.0, status=ConfidenceStatus.NOT_CONFIGURED)]))
        assert "outdoor" in r.missing_optional
        # a not-configured optional is not a disturbance
        assert ConfidenceCap.MISSING_REQUIRED not in caps_of(r)

    def test_unavailable_optional(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50),
            component(ComponentKind.OUTDOOR, 0.0, status=ConfidenceStatus.UNAVAILABLE)]))
        assert "outdoor" in r.missing_optional

    def test_fallback_visible_but_functional(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, [
            component(ComponentKind.PREHEAT_PRIOR, 0.5, evidence_count=0,
                      status=ConfidenceStatus.FALLBACK)]))
        assert r.fallback_used and ConfidenceCap.FALLBACK_USED in caps_of(r)
        assert r.value > 0.0

    def test_recovery_after_input_returns(self):
        degraded = A().aggregate(self._req(ConfidenceStatus.UNAVAILABLE))
        healthy = A().aggregate(self._req(ConfidenceStatus.HEALTHY))
        assert healthy.value > degraded.value


class TestDiagnosticsExport:
    def _result(self):
        return A().aggregate(inp(ConfidencePurpose.PREHEAT, [
            component(ComponentKind.HEAT_RATE, 0.8, evidence_count=30,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,))]))

    def test_breakdown_complete(self):
        b = self._result().breakdown
        assert b.purpose is ConfidencePurpose.PREHEAT
        assert b.components and b.applied_caps is not None
        assert b.level is not None and b.gate is not None
        assert b.aggregator_version and b.parameter_version

    def test_support_export_no_ids(self):
        exp = breakdown_to_support_export(self._result())
        blob = json.dumps(exp)
        assert "value" in exp and "level" in exp and "applied_caps" in exp
        assert "decision" not in blob and "episode_id" not in blob

    def test_research_export_pseudonymized(self):
        exp = breakdown_to_research_export(self._result())
        assert "cap_frequencies" in exp and "evidence_domain_distribution" in exp
        assert "component_kinds" in exp
        assert "aggregator_version" in exp and "parameter_version" in exp

    def test_caps_and_reasons_visible(self):
        # a capped scenario: every applied cap carries a human-readable reason
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.9, evidence_count=2)]))
        assert r.applied_caps and all(c.reason for c in r.applied_caps)
        assert r.reason_codes
