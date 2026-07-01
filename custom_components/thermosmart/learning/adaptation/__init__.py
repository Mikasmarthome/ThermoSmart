"""Outcome Adaptation Engine — passive trace foundation.

Public surface:
  contracts          — CandidateType, AdaptationLifecycle, AdaptationDirection,
                       SituationContext, OutcomeSignal, AdaptationGateResult,
                       AdaptationTrace, make_adaptation_id
  gates              — evaluate_gates()
  candidates         — suggest_candidates()
  trace              — traces_for_export(), filter_by_lifecycle()
  history            — CandidateIdentity, make_candidate_key,
                       candidate_identity_from_trace, CandidateHistoryEntry,
                       accumulate_into_history, PromotionReadiness,
                       PromotionGateResult, evaluate_promotion_readiness
  history_store_schema — AdaptationHistoryState, HISTORY_SCHEMA_VERSION,
                         serialize_history_state, deserialize_history_state,
                         prune_history_entries
  application        — ApplicationPolicy, ApplicationRuntimeContext,
                       ApplicationDecision, ApplicationPlan,
                       evaluate_application_readiness,
                       application_readiness_preview,
                       build_preheat_application_plan
  monitoring         — AppliedAdaptationRecord, MonitoringEvaluation,
                       MonitoringStatus, build_applied_adaptation_record,
                       evaluate_applied_adaptation_outcome,
                       should_rollback_adaptation

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
from .history import (
    CandidateHistoryEntry,
    CandidateIdentity,
    PromotionGateResult,
    PromotionReadiness,
    accumulate_into_history,
    candidate_identity_from_trace,
    evaluate_promotion_readiness,
    make_candidate_key,
    update_adaptation_history,
)
from .history_store_schema import (
    HISTORY_SCHEMA_VERSION,
    AdaptationHistoryState,
    adaptation_history_entry_for_research_export,
    deserialize_history_state,
    prune_history_entries,
    serialize_history_state,
)
from .application import (
    ApplicationDecision,
    ApplicationPlan,
    ApplicationPolicy,
    ApplicationRuntimeContext,
    application_readiness_preview,
    build_preheat_application_plan,
    evaluate_application_readiness,
)
from .monitoring import (
    AppliedAdaptationRecord,
    MonitoringEvaluation,
    MonitoringStatus,
    build_applied_adaptation_record,
    evaluate_applied_adaptation_outcome,
    should_rollback_adaptation,
)

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
    "CandidateHistoryEntry",
    "CandidateIdentity",
    "PromotionGateResult",
    "PromotionReadiness",
    "accumulate_into_history",
    "candidate_identity_from_trace",
    "evaluate_promotion_readiness",
    "make_candidate_key",
    "update_adaptation_history",
    "HISTORY_SCHEMA_VERSION",
    "AdaptationHistoryState",
    "adaptation_history_entry_for_research_export",
    "deserialize_history_state",
    "prune_history_entries",
    "serialize_history_state",
    "ApplicationDecision",
    "ApplicationPlan",
    "ApplicationPolicy",
    "ApplicationRuntimeContext",
    "application_readiness_preview",
    "build_preheat_application_plan",
    "evaluate_application_readiness",
    "AppliedAdaptationRecord",
    "MonitoringEvaluation",
    "MonitoringStatus",
    "build_applied_adaptation_record",
    "evaluate_applied_adaptation_outcome",
    "should_rollback_adaptation",
]
