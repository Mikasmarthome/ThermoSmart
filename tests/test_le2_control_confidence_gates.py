"""Phase 18 confidence-gate authority tests (single source: ConfidenceResult)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import (
    ComponentKind, ConfidenceAggregator, ConfidenceComponent, ConfidenceInput,
    ConfidencePurpose, EvidenceDomain)
from custom_components.thermosmart.learning.contracts import ConfidenceContribution, Regime
from custom_components.thermosmart.learning.runtime.control import (
    ControlContext, ControlFeature, ControlRejection)
from tests.helpers_control import control_decision, strong_preheat

F = ControlFeature.PREHEAT_START
AGG = ConfidenceAggregator()


def _preheat(value, *, reliability=0.9, prior=0.0, fallback=False, regime=None):
    comps = (
        ConfidenceComponent(kind=ComponentKind.HEAT_RATE,
                            contribution=ConfidenceContribution(
                                value=value, evidence_count=50, reliability=reliability,
                                evidence_domains=("thermal_trajectory",), prior_fraction=prior,
                                fallback_used=fallback),
                            evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
        ConfidenceComponent(kind=ComponentKind.CURRENT_TEMPERATURE,
                            contribution=ConfidenceContribution(value=0.9, evidence_count=10,
                                                                reliability=0.9),
                            evidence_domains=(EvidenceDomain.CURRENT_STATE,)),
    )
    return AGG.aggregate(ConfidenceInput(ConfidencePurpose.PREHEAT, components=comps,
                                         regime=regime))


class TestConfidenceGates:
    def test_open_gate_applies(self):
        d = control_decision(F, 60.0, 66.0, _preheat(0.9), unit="minutes")
        assert d.applied

    def test_insufficient_confidence(self):
        d = control_decision(F, 60.0, 66.0, _preheat(0.5), unit="minutes")
        # value below policy min_control_confidence (0.55 after reliability)
        assert not d.applied

    def test_low_reliability(self):
        d = control_decision(F, 60.0, 66.0, _preheat(0.9, reliability=0.2), unit="minutes")
        assert not d.applied and d.rejection in (
            ControlRejection.LOW_RELIABILITY.value, ControlRejection.CONTROL_GATE_CLOSED.value,
            ControlRejection.INSUFFICIENT_CONFIDENCE.value)

    def test_prior_dominant(self):
        d = control_decision(F, 60.0, 66.0, _preheat(0.9, prior=0.9), unit="minutes")
        assert not d.applied

    def test_fallback_dominant(self):
        # a model in fallback reports a dominant prior fraction
        d = control_decision(F, 60.0, 66.0, _preheat(0.9, fallback=True, prior=1.0),
                             unit="minutes")
        assert not d.applied and d.rejection in (
            ControlRejection.FALLBACK_DOMINANT.value, ControlRejection.PRIOR_DOMINANT.value,
            ControlRejection.CONTROL_GATE_CLOSED.value)

    def test_disturbed_regime_hard(self):
        d = control_decision(F, 60.0, 66.0, _preheat(0.9, regime=Regime.DISTURBED),
                             unit="minutes")
        # the aggregator's hard DISTURBED cap closes the gate
        assert not d.applied

    def test_confidence_summary_attached(self):
        d = control_decision(F, 60.0, 66.0, _preheat(0.9), unit="minutes")
        assert d.confidence is not None and d.confidence.control_allowed

    def test_policy_does_not_recompute(self):
        # the same ConfidenceResult drives the decision; policy only reads it
        cr = strong_preheat()
        d = control_decision(F, 60.0, 66.0, cr, unit="minutes")
        assert d.confidence.value == round(cr.value, 6)
