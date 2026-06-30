"""Phase 12 aggregation principle / purpose-policy / correlation tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceInput,
    ConfidencePurpose,
    ConfidenceStatus,
    EvidenceDomain,
)
from tests.helpers_confidence import component


def A():
    return ConfidenceAggregator()


def inp(purpose, components=(), **kw):
    return ConfidenceInput(purpose=purpose, components=tuple(components), **kw)


class TestAggregationPrinciple:
    def test_required_present(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.8)]))
        assert "required_present" in r.reason_codes and not r.missing_required

    def test_missing_required_capped_hard(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, []))
        caps = [c.cap.value for c in r.applied_caps]
        assert "missing_required" in caps
        assert any(c.hard for c in r.applied_caps)

    def test_required_plus_optional_bonus(self):
        base = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                                 [component(ComponentKind.HEAT_RATE, 0.7,
                                            domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        boosted = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.7, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.DATA_QUALITY, 0.9, domains=(EvidenceDomain.CURRENT_STATE,)),
            component(ComponentKind.BUCKET_SUPPORT, 0.9, domains=(EvidenceDomain.OUTCOME_HISTORY,)),
        ]))
        assert boosted.value > base.value

    def test_no_naive_average(self):
        # a strong optional must NOT pull a mediocre required up to the mean
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.4, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.DATA_QUALITY, 1.0, domains=(EvidenceDomain.CURRENT_STATE,)),
        ]))
        # naive average would be ~0.7; bounded bonus keeps it near the core
        assert r.value < 0.6

    def test_high_optional_cannot_mask_weak_core(self):
        weak = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.3, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.DATA_QUALITY, 1.0)]))
        assert weak.value <= 0.3 + A()._params.max_optional_bonus + 1e-9

    def test_bonus_cannot_override_cap(self):
        # invalid required -> hard cap 0.1; optional bonus cannot exceed it
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, status=ConfidenceStatus.INVALID),
            component(ComponentKind.DATA_QUALITY, 1.0)]))
        assert r.value <= 0.1 + 1e-9

    def test_missing_optional_not_strongly_penalised(self):
        only_required = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                                          [component(ComponentKind.HEAT_RATE, 0.8)]))
        assert only_required.value >= 0.7  # still healthy, just missing optional noted
        assert only_required.missing_optional


class TestPurposePolicies:
    def test_heat_rate_isolated_from_unrelated(self):
        # adding heat-loss/outcome is not even in HEAT_RATE optional -> ignored, no inflation
        with_unrelated = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.7),
            component(ComponentKind.HEAT_LOSS, 1.0)]))
        baseline = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                                     [component(ComponentKind.HEAT_RATE, 0.7)]))
        assert with_unrelated.value == pytest.approx(baseline.value, abs=1e-9)

    def test_preheat_functional_without_forecast(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, [
            component(ComponentKind.HEAT_RATE, 0.8,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,))]))
        # local path: no forecast cap, forecast merely documented as missing_optional
        assert r.value > 0.0
        assert "forecast_unavailable" not in [c.cap.value for c in r.applied_caps]
        assert "forecast" in r.missing_optional

    def test_early_cutoff_requires_afterheat(self):
        r = A().aggregate(inp(ConfidencePurpose.EARLY_CUTOFF, []))
        assert "missing_required" in [c.cap.value for c in r.applied_caps]

    def test_early_cutoff_with_afterheat(self):
        r = A().aggregate(inp(ConfidencePurpose.EARLY_CUTOFF, [
            component(ComponentKind.AFTERHEAT, 0.8,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        assert not r.missing_required

    def test_outcome_advisory(self):
        r = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, [
            component(ComponentKind.OUTCOME, 0.7,
                      domains=(EvidenceDomain.OUTCOME_HISTORY,))]))
        assert r.advisory_allowed and not r.control_allowed

    def test_advisory_purpose_no_required(self):
        r = A().aggregate(inp(ConfidencePurpose.ADVISORY, [
            component(ComponentKind.OUTCOME, 0.6, domains=(EvidenceDomain.OUTCOME_HISTORY,))]))
        assert not r.missing_required and r.value > 0.0

    def test_control_decision_policy(self):
        r = A().aggregate(inp(ConfidencePurpose.CONTROL_DECISION, [
            component(ComponentKind.HEAT_RATE, 0.85, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,))]))
        assert r.control_allowed

    def test_forecast_and_boost_and_manual_prepared(self):
        for purpose, kind in [
            (ConfidencePurpose.FORECAST, ComponentKind.FORECAST),
            (ConfidencePurpose.BOOST, ComponentKind.HEAT_RATE),
            (ConfidencePurpose.MANUAL_CORRECTION, ComponentKind.MANUAL_CORRECTION)]:
            r = A().aggregate(inp(purpose, [component(kind, 0.7)]))
            assert 0.0 <= r.value <= 1.0


class TestPreheatForecastPolicy:
    """Forecast is optional precision for preheat; never a blanket cap."""

    def _local(self, **kw):
        return [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.HEAT_LOSS, 0.85, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.AFTERHEAT, 0.85, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,))]

    def _caps(self, r):
        return [c.cap.value for c in r.applied_caps]

    def test_no_forecast_strong_local_high(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local()))
        assert r.value > 0.7
        assert "forecast_unavailable" not in self._caps(r)

    def test_missing_forecast_only_in_missing_optional(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local()))
        assert "forecast" in r.missing_optional
        assert "forecast_unavailable" not in self._caps(r)

    def test_no_cap_on_local_path(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local()))
        assert "trv_only_relative" not in self._caps(r)

    def test_forecast_configured_unavailable_local_path_usable(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local() + [
            component(ComponentKind.FORECAST, 0.0, status=ConfidenceStatus.UNAVAILABLE)]))
        # local path (forecast_required not set) stays usable, no forecast cap
        assert r.value > 0.7 and "forecast_unavailable" not in self._caps(r)

    def test_forecast_dependent_path_capped_on_unavailable(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, forecast_required=True,
                              components=self._local() + [
                                  component(ComponentKind.FORECAST, 0.0,
                                            status=ConfidenceStatus.UNAVAILABLE)]))
        assert "forecast_unavailable" in self._caps(r)

    def test_stale_forecast_limits_only_dependent_path(self):
        local = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local() + [
            component(ComponentKind.FORECAST, 0.8, status=ConfidenceStatus.STALE)]))
        dependent = A().aggregate(inp(ConfidencePurpose.PREHEAT, forecast_required=True,
                                      components=self._local() + [
                                          component(ComponentKind.FORECAST, 0.8,
                                                    status=ConfidenceStatus.STALE)]))
        assert "stale_data" not in [c.cap.value for c in local.applied_caps]
        assert "stale_data" in [c.cap.value for c in dependent.applied_caps]

    def test_healthy_forecast_only_bounded_bonus(self):
        without = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local()))
        with_fc = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local() + [
            component(ComponentKind.FORECAST, 0.9, domains=(EvidenceDomain.FORECAST,))]))
        assert with_fc.value >= without.value
        assert with_fc.value - without.value <= A()._params.max_optional_bonus + 1e-9

    def test_forecast_bonus_cannot_save_weak_heat_rate(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, [
            component(ComponentKind.HEAT_RATE, 0.3, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.FORECAST, 1.0, domains=(EvidenceDomain.FORECAST,))]))
        assert r.value <= 0.3 + A()._params.max_optional_bonus + 1e-9

    def test_trv_only_preheat_allowed(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local()))
        assert r.control_allowed

    def test_control_gate_open_with_local_confidence(self):
        r = A().aggregate(inp(ConfidencePurpose.PREHEAT, self._local()))
        assert r.gate.allowed and not r.gate.safe_fallback_required


class TestCorrelation:
    def test_same_domain_not_linear(self):
        same = A().aggregate(inp(ConfidencePurpose.CONTROL_DECISION, [
            component(ComponentKind.HEAT_RATE, 0.9, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,)),
            component(ComponentKind.OUTCOME, 0.9, domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        assert "correlated_evidence_discounted" in same.reason_codes
        assert same.effective_evidence < same.breakdown.raw_evidence_count

    def test_independent_domains_contribute_more(self):
        correlated = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, [
            component(ComponentKind.OUTCOME, 0.6, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.DATA_QUALITY, 0.9, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
        ]))
        independent = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, [
            component(ComponentKind.OUTCOME, 0.6, domains=(EvidenceDomain.OUTCOME_HISTORY,)),
            component(ComponentKind.DATA_QUALITY, 0.9, domains=(EvidenceDomain.CURRENT_STATE,)),
        ]))
        assert independent.value >= correlated.value

    def test_outcome_and_heat_rate_shared_trajectory(self):
        r = A().aggregate(inp(ConfidencePurpose.CONTROL_DECISION, [
            component(ComponentKind.HEAT_RATE, 0.9, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.OUTCOME, 0.9, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,))]))
        assert "correlated_evidence_discounted" in r.reason_codes

    def test_extra_independent_controller_evidence_helps(self):
        without = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, [
            component(ComponentKind.OUTCOME, 0.7, domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        with_ctrl = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, [
            component(ComponentKind.OUTCOME, 0.7, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.BUCKET_SUPPORT, 0.9,
                      domains=(EvidenceDomain.CONTROLLER_EVENTS,))]))
        assert with_ctrl.value >= without.value
