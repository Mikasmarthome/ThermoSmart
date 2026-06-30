"""Shared builders for LE 2.0 control-policy tests."""
from __future__ import annotations

from typing import Optional

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceComponent,
    ConfidenceInput,
    ConfidencePurpose,
    EvidenceDomain,
)
from custom_components.thermosmart.learning.contracts import ConfidenceContribution, Regime
from custom_components.thermosmart.learning.runtime.control import (
    ControlledPrediction,
    ControlContext,
    ControlFeature,
    ControlPolicy,
)

_AGG = ConfidenceAggregator()


def _comp(kind, value, *, reliability=0.9, ev=50, domains=(EvidenceDomain.THERMAL_TRAJECTORY,),
          prior=0.0, fallback=False, reasons=()):
    return ConfidenceComponent(
        kind=kind,
        contribution=ConfidenceContribution(
            value=value, evidence_count=ev, reliability=reliability,
            evidence_domains=tuple(d.value for d in domains), prior_fraction=prior,
            fallback_used=fallback, reasons=tuple(reasons)),
        evidence_domains=domains)


def strong_preheat():
    """A ConfidenceResult whose PREHEAT control gate is OPEN."""
    return _AGG.aggregate(ConfidenceInput(ConfidencePurpose.PREHEAT, components=(
        _comp(ComponentKind.HEAT_RATE, 0.9),
        _comp(ComponentKind.CURRENT_TEMPERATURE, 0.9, domains=(EvidenceDomain.CURRENT_STATE,)),
    )))


def strong_purpose(purpose, kind):
    return _AGG.aggregate(ConfidenceInput(purpose, components=(_comp(kind, 0.9),)))


def strong_boost():
    """A ConfidenceResult whose BOOST control gate is OPEN (required HEAT_RATE)."""
    return _AGG.aggregate(ConfidenceInput(ConfidencePurpose.BOOST, components=(
        _comp(ComponentKind.HEAT_RATE, 0.9),
        _comp(ComponentKind.OUTCOME, 0.9, domains=(EvidenceDomain.OUTCOME_HISTORY,)),
    )))


def weak_purpose(purpose, kind):
    return _AGG.aggregate(ConfidenceInput(purpose, components=(
        _comp(kind, 0.4, reliability=0.2, ev=2, prior=0.9, fallback=True),)))


def control_decision(feature, baseline, proposed, confidence_result, *,
                     context=None, unit="", runtime_is_control=True, policy=None):
    pol = policy or ControlPolicy(runtime_is_control=runtime_is_control)
    pred = (ControlledPrediction(feature, proposed, unit, confidence_result,
                                 model_version=1, parameter_version=1)
            if proposed is not None else None)
    return pol.resolve(feature, baseline, pred, context=context or ControlContext(), unit=unit)
