"""Phase 12 stateless/reproducible + ConfidenceContribution-contract tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceComponent,
    ConfidenceInput,
    ConfidencePurpose,
    ConfidenceStatus,
    EvidenceDomain,
)
from custom_components.thermosmart.learning.contracts import ConfidenceContribution
from tests.helpers_confidence import component


def A():
    return ConfidenceAggregator()


def inp(purpose, components=()):
    return ConfidenceInput(purpose=purpose, components=tuple(components))


class TestStateless:
    def test_inputs_not_mutated(self):
        comp = component(ComponentKind.HEAT_RATE, 0.8, domains=(EvidenceDomain.THERMAL_TRAJECTORY,))
        contribution_before = comp.contribution
        A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [comp]))
        assert comp.contribution is contribution_before
        assert comp.contribution.value == 0.8

    def test_repeated_calls_identical(self):
        agg = A()
        i = inp(ConfidencePurpose.HEAT_RATE, [component(ComponentKind.HEAT_RATE, 0.7)])
        assert agg.aggregate(i).value == agg.aggregate(i).value

    def test_no_persisted_state(self):
        agg = A()
        assert not hasattr(agg, "_state")

    def test_order_independent(self):
        a = [component(ComponentKind.HEAT_RATE, 0.8, domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
             component(ComponentKind.DATA_QUALITY, 0.9, domains=(EvidenceDomain.CURRENT_STATE,))]
        b = list(reversed(a))
        assert A().aggregate(inp(ConfidencePurpose.HEAT_RATE, a)).value == \
            A().aggregate(inp(ConfidencePurpose.HEAT_RATE, b)).value


class TestContributionContract:
    def _wrap(self, kind, contribution, **kw):
        return ConfidenceComponent(kind=kind, contribution=contribution, **kw)

    def test_zero_value_valid(self):
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            self._wrap(ComponentKind.HEAT_RATE, ConfidenceContribution(value=0.0,
                                                                       evidence_count=10))]))
        assert r.value == 0.0

    def test_minimal_contribution_accepted(self):
        # a model that only sets value/evidence_count/reasons still works
        c = ConfidenceContribution(value=0.6, evidence_count=20, reasons=("ok",))
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE,
                              [self._wrap(ComponentKind.HEAT_RATE, c)]))
        assert r.value > 0.0

    def test_nan_contribution_rejected_by_aggregator(self):
        c = ConfidenceComponent(kind=ComponentKind.HEAT_RATE,
                                contribution=ConfidenceContribution(value=float("nan")))
        with pytest.raises(ValueError):
            A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [c]))

    def test_duplicate_component_rejected(self):
        with pytest.raises(ValueError):
            A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
                component(ComponentKind.HEAT_RATE, 0.5),
                component(ComponentKind.HEAT_RATE, 0.6)]))


class TestRealModelContributions:
    """Existing models' confidence() output feeds the aggregator unchanged."""

    def test_heat_rate_model_contribution(self):
        from custom_components.thermosmart.learning.models import HeatRateModel
        contrib = HeatRateModel("lz").confidence()
        assert isinstance(contrib, ConfidenceContribution)
        r = A().aggregate(inp(ConfidencePurpose.HEAT_RATE, [
            ConfidenceComponent(kind=ComponentKind.HEAT_RATE, contribution=contrib)]))
        assert 0.0 <= r.value <= 1.0

    def test_outcome_model_contribution(self):
        from custom_components.thermosmart.learning.models import OutcomeModel
        contrib = OutcomeModel("lz").confidence()
        r = A().aggregate(inp(ConfidencePurpose.OUTCOME_QUALITY, [
            ConfidenceComponent(kind=ComponentKind.OUTCOME, contribution=contrib,
                                status=ConfidenceStatus.FALLBACK)]))
        assert 0.0 <= r.value <= 1.0

    def test_afterheat_and_heat_loss_contributions(self):
        from custom_components.thermosmart.learning.models import (
            AfterheatModel,
            HeatLossModel,
        )
        for model, kind, purpose in [
            (AfterheatModel("lz"), ComponentKind.AFTERHEAT, ConfidencePurpose.AFTERHEAT),
            (HeatLossModel("lz"), ComponentKind.HEAT_LOSS, ConfidencePurpose.HEAT_LOSS)]:
            contrib = model.confidence()
            r = A().aggregate(inp(purpose, [
                ConfidenceComponent(kind=kind, contribution=contrib)]))
            assert 0.0 <= r.value <= 1.0
