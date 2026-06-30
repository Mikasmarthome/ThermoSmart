"""Outcome Adaptation Engine — passive trace foundation.

Public surface:
  contracts  — CandidateType, AdaptationLifecycle, AdaptationDirection,
               SituationContext, OutcomeSignal, AdaptationGateResult,
               AdaptationTrace, make_adaptation_id
  gates      — evaluate_gates()
  candidates — suggest_candidates()
  trace      — traces_for_export(), filter_by_lifecycle()

No HA imports. No runtime state. No control modifications.
"""
from .contracts import (
    AdaptationDirection,
    AdaptationGateResult,
    AdaptationLifecycle,
    AdaptationTrace,
    CandidateType,
    OutcomeSignal,
    SituationContext,
    make_adaptation_id,
)
from .gates import evaluate_gates
from .candidates import suggest_candidates
from .trace import filter_by_lifecycle, traces_for_export

__all__ = [
    "AdaptationDirection",
    "AdaptationGateResult",
    "AdaptationLifecycle",
    "AdaptationTrace",
    "CandidateType",
    "OutcomeSignal",
    "SituationContext",
    "make_adaptation_id",
    "evaluate_gates",
    "suggest_candidates",
    "filter_by_lifecycle",
    "traces_for_export",
]
