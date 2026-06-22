"""Phase 12 ConfidenceAggregator basic contract / numeric-guard tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.confidence import (
    AGGREGATOR_VERSION,
    PARAMETER_VERSION,
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceComponent,
    ConfidenceInput,
    ConfidenceLevel,
    ConfidencePurpose,
    ConfidenceResult,
    ConfidenceStatus,
    EvidenceDomain,
)
from custom_components.thermosmart.learning.contracts import ConfidenceContribution, Regime
from tests.helpers_confidence import component, contribution


def A():
    return ConfidenceAggregator()


def inp(purpose=ConfidencePurpose.HEAT_RATE, components=(), **kw):
    return ConfidenceInput(purpose=purpose, components=tuple(components), **kw)


class TestBasic:
    def test_single_required(self):
        r = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 0.8)]))
        assert isinstance(r, ConfidenceResult)
        assert 0.0 <= r.value <= 1.0
        assert r.aggregator_version == AGGREGATOR_VERSION
        assert r.parameter_version == PARAMETER_VERSION

    def test_no_contributions(self):
        r = A().aggregate(inp(components=[]))
        assert r.value == 0.0
        assert "no_contributions" in r.reason_codes or r.missing_required

    def test_value_in_range(self):
        r = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 1.0)]))
        assert 0.0 <= r.value <= 1.0

    def test_exact_zero_contribution(self):
        r = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 0.0)]))
        assert r.value == 0.0

    def test_exact_one_contribution_capped_by_factors(self):
        r = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 1.0,
                                                   reliability=1.0, evidence_count=50)]))
        assert r.value <= 1.0

    def test_deterministic(self):
        c = [component(ComponentKind.HEAT_RATE, 0.7), component(ComponentKind.DATA_QUALITY, 0.8)]
        assert A().aggregate(inp(components=c)).value == A().aggregate(inp(components=c)).value


class TestInvalidContributions:
    def test_nan_value_rejected(self):
        bad = ConfidenceComponent(kind=ComponentKind.HEAT_RATE,
                                  contribution=ConfidenceContribution(value=float("nan")))
        with pytest.raises(ValueError):
            A().aggregate(inp(components=[bad]))

    def test_value_above_one_rejected_at_contract(self):
        with pytest.raises(ValueError):
            ConfidenceContribution(value=1.5)

    def test_negative_value_rejected_at_contract(self):
        with pytest.raises(ValueError):
            ConfidenceContribution(value=-0.1)

    def test_infinite_freshness_rejected(self):
        with pytest.raises(ValueError):
            ConfidenceComponent(kind=ComponentKind.HEAT_RATE,
                                contribution=ConfidenceContribution(value=0.5),
                                freshness=float("inf"))

    def test_duplicate_component_rejected(self):
        c = [component(ComponentKind.HEAT_RATE, 0.5), component(ComponentKind.HEAT_RATE, 0.6)]
        with pytest.raises(ValueError):
            A().aggregate(inp(components=c))

    def test_reliability_zero_and_one(self):
        lo = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 0.9, reliability=0.0)]))
        hi = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 0.9, reliability=1.0)]))
        assert lo.value < hi.value

    def test_empty_evidence_domains_ok(self):
        r = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 0.8, domains=())]))
        assert math.isfinite(r.value)


class TestLevels:
    @pytest.mark.parametrize("value,level", [
        (0.05, ConfidenceLevel.VERY_LOW), (0.3, ConfidenceLevel.LOW),
        (0.5, ConfidenceLevel.MEDIUM), (0.7, ConfidenceLevel.HIGH),
        (0.95, ConfidenceLevel.VERY_HIGH),
    ])
    def test_level_mapping(self, value, level):
        assert A()._level(value) is level

    def test_exact_threshold_boundaries(self):
        # thresholds default (0.2,0.4,0.6,0.8): exact value falls into the upper band
        assert A()._level(0.2) is ConfidenceLevel.LOW
        assert A()._level(0.4) is ConfidenceLevel.MEDIUM
        assert A()._level(0.8) is ConfidenceLevel.VERY_HIGH

    def test_just_below_boundary(self):
        assert A()._level(0.1999) is ConfidenceLevel.VERY_LOW


class TestReliabilitySeparation:
    def test_confidence_and_reliability_both_present(self):
        r = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 0.8, reliability=0.6)]))
        assert r.reliability == pytest.approx(0.6, abs=1e-6)
        assert r.value != r.reliability

    def test_low_reliability_caps_confidence(self):
        r = A().aggregate(inp(components=[component(ComponentKind.HEAT_RATE, 0.95,
                                                   reliability=0.2, evidence_count=50)]))
        assert any(c.cap.value == "low_reliability" for c in r.applied_caps)
