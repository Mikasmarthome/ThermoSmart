"""Phase 12 caps, gates, disturbed/unknown, disagreement tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceCap,
    ConfidenceConflict,
    ConfidenceInput,
    ConfidencePurpose,
    ConfidenceStatus,
    ConfidenceStatusResult,
    EvidenceDomain,
)
from custom_components.thermosmart.learning.contracts import Regime
from tests.helpers_confidence import component


def A():
    return ConfidenceAggregator()


def inp(purpose, components=(), **kw):
    return ConfidenceInput(purpose=purpose, components=tuple(components), **kw)


def caps_of(result):
    return {c.cap for c in result.applied_caps}


class TestCaps:
    def test_cold_start(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.9, evidence_count=2)]))
        assert ConfidenceCap.COLD_START in caps_of(r)
        assert r.value <= A()._params.cold_start_cap + 1e-9

    def test_prior_dominant(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50,
                                         prior_fraction=0.9)]))
        assert ConfidenceCap.PRIOR_DOMINANT in caps_of(r)

    def test_low_evidence(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.9, evidence_count=1)]))
        assert ConfidenceCap.LOW_EVIDENCE in caps_of(r)

    def test_low_reliability(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.9, reliability=0.2,
                                         evidence_count=50)]))
        assert ConfidenceCap.LOW_RELIABILITY in caps_of(r)

    def test_missing_current_temperature(self):
        r = A().aggregate(inp(ConfidencePurpose.CONTROL_DECISION,
                              [component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50)]))
        assert ConfidenceCap.MISSING_CURRENT_TEMPERATURE in caps_of(r)
        assert r.value <= A()._params.missing_current_temperature_cap + 1e-9

    def test_bucket_unsupported(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50),
            component(ComponentKind.BUCKET_SUPPORT, 0.0,
                      status=ConfidenceStatus.NOT_COMPUTABLE)]))
        assert ConfidenceCap.BUCKET_UNSUPPORTED in caps_of(r)

    def test_stale(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50,
                      status=ConfidenceStatus.STALE)]))
        assert ConfidenceCap.STALE_DATA in caps_of(r)

    def test_invalid_hard(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, status=ConfidenceStatus.INVALID)]))
        assert ConfidenceCap.INVALID_DATA in caps_of(r)
        assert any(c.hard for c in r.applied_caps)

    def test_partial_outcome(self):
        r = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, [
            component(ComponentKind.OUTCOME, 0.8, evidence_count=50, reasons=("partial",))]))
        assert ConfidenceCap.PARTIAL_OUTCOME in caps_of(r)

    def test_multiple_caps_strictest_wins(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=2, reliability=0.2,
                      prior_fraction=0.9)]))
        # cold-start(0.35), low-reliability(0.5), prior(0.5), low-evidence(0.45) -> 0.35
        assert r.value <= 0.35 + 1e-9
        assert len(r.applied_caps) >= 2  # all stay visible

    def test_caps_visible_in_breakdown(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.9, evidence_count=2)]))
        assert r.breakdown.applied_caps == r.applied_caps


class TestDisturbedUnknown:
    def test_disturbed_control_hard_near_zero(self):
        r = A().aggregate(inp(ConfidencePurpose.CONTROL_DECISION, regime=Regime.DISTURBED,
                              components=[
                                  component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50),
                                  component(ComponentKind.CURRENT_TEMPERATURE, 0.9)]))
        assert ConfidenceCap.DISTURBED in caps_of(r)
        assert r.value <= A()._params.disturbed_cap + 1e-9
        assert not r.control_allowed
        assert r.status is ConfidenceStatusResult.BLOCKED

    def test_disturbed_diagnostics_still_produced(self):
        r = A().aggregate(inp(ConfidencePurpose.ADVISORY, regime=Regime.DISTURBED,
                              components=[component(ComponentKind.OUTCOME, 0.8,
                                                   domains=(EvidenceDomain.OUTCOME_HISTORY,))]))
        assert r.breakdown is not None  # structured result still exists

    def test_unknown_regime_caps_prediction(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, regime=Regime.UNKNOWN,
                              components=[component(ComponentKind.HEAT_RATE, 0.9,
                                                   evidence_count=50)]))
        assert ConfidenceCap.UNKNOWN_REGIME in caps_of(r)
        assert r.value <= A()._params.unknown_regime_cap + 1e-9

    def test_unknown_does_not_zero_independent_purpose(self):
        r = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, regime=Regime.UNKNOWN,
                              components=[component(ComponentKind.OUTCOME, 0.8,
                                                   evidence_count=50,
                                                   domains=(EvidenceDomain.OUTCOME_HISTORY,))]))
        assert r.value > 0.0


class TestDisagreement:
    def test_model_disagreement_caps(self):
        conflict = ConfidenceConflict(
            kind="afterheat_vs_outcome", components=(ComponentKind.AFTERHEAT, ComponentKind.OUTCOME),
            severity=1.0, reason="repeated overshoot")
        r = A().aggregate(inp(ConfidencePurpose.CONTROL_DECISION, conflicts=(conflict,),
                              components=[
                                  component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50),
                                  component(ComponentKind.CURRENT_TEMPERATURE, 0.9)]))
        assert ConfidenceCap.MODEL_DISAGREEMENT in caps_of(r)
        assert "model_disagreement" in r.reason_codes

    def test_no_conflict_no_cap(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50)]))
        assert ConfidenceCap.MODEL_DISAGREEMENT not in caps_of(r)


class TestGates:
    def test_control_allowed(self):
        r = A().aggregate(inp(ConfidencePurpose.CONTROL_DECISION, components=[
            component(ComponentKind.HEAT_RATE, 0.9, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9,
                      domains=(EvidenceDomain.CURRENT_STATE,))]))
        assert r.control_allowed and r.gate.allowed

    def test_control_not_allowed_low(self):
        r = A().aggregate(inp(ConfidencePurpose.CONTROL_DECISION, components=[
            component(ComponentKind.HEAT_RATE, 0.3, evidence_count=2),
            component(ComponentKind.CURRENT_TEMPERATURE, 0.9)]))
        assert not r.control_allowed
        assert r.gate.safe_fallback_required

    def test_advisory_allowed_when_control_not(self):
        r = A().aggregate(inp(ConfidencePurpose.EARLY_CUTOFF, components=[
            component(ComponentKind.AFTERHEAT, 0.4, evidence_count=5,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        assert not r.control_allowed and r.advisory_allowed

    def test_purpose_specific_minimum(self):
        r = A().aggregate(inp(ConfidencePurpose.EARLY_CUTOFF, components=[
            component(ComponentKind.AFTERHEAT, 0.9, evidence_count=50,
                      domains=(EvidenceDomain.THERMAL_TRAJECTORY,))]))
        assert r.gate.required_minimum == A()._params.control_gate(ConfidencePurpose.EARLY_CUTOFF)

    def test_advisory_purpose_never_control(self):
        r = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, components=[
            component(ComponentKind.OUTCOME, 0.95, evidence_count=50,
                      domains=(EvidenceDomain.OUTCOME_HISTORY,))]))
        assert not r.control_allowed  # not a control-relevant purpose
