"""Runtime-disabled adaptation application orchestrator.

Pure connective layer that wires the existing passive-adaptation building
blocks together into one traceable preview, without ever mutating storage
or control state:

    CandidateHistoryEntry
    → evaluate_promotion_readiness()
    → evaluate_application_readiness()
    → build_preheat_application_plan()
    → optional build_applied_adaptation_record() (preview only)
    → ApplicationLifecycleState awareness (existing entry / cooldown / terminal status)
    → ApplicationOrchestrationResult

No HA imports. No runtime state. No control modification. No storage mutation.
The kill-switch (ApplicationPolicy.application_enabled) defaults to False, so
``would_apply`` is never True in real runtime — only an explicit test policy
can make it True, and even then this module performs no application, no
service call, and no persistence of any kind.

All dataclasses are frozen; evaluate_application_orchestration() is pure and
never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .application import (
    ApplicationDecision,
    ApplicationPlan,
    ApplicationPolicy,
    ApplicationRuntimeContext,
    build_preheat_application_plan,
    evaluate_application_readiness,
)
from .application_state import ApplicationLifecycleState, is_cooldown_active
from .history import CandidateHistoryEntry, evaluate_promotion_readiness
from .monitoring import (
    AppliedAdaptationRecord,
    MonitoringStatus,
    build_applied_adaptation_record,
)

# Lifecycle statuses that still require monitoring attention — a second
# application must not be planned while an entry is in one of these states.
_ACTIVE_LIFECYCLE_STATUSES: frozenset[str] = frozenset({
    MonitoringStatus.PENDING.value,
    MonitoringStatus.IN_PROGRESS.value,
    MonitoringStatus.ADOPTION_CANDIDATE.value,
    MonitoringStatus.ROLLBACK_RECOMMENDED.value,
})


class ApplicationOrchestrationMode(Enum):
    """Contract identifier for the orchestration mode.

    Only PREVIEW_ONLY exists in this step — no real application mode has
    been built yet. Reserved so a future real Application Layer can add a
    distinct mode value without changing this one's meaning.
    """
    PREVIEW_ONLY = "preview_only"


@dataclass(frozen=True)
class ApplicationOrchestratorPolicy:
    """Composed policy for one orchestration preview run.

    Wraps the existing ApplicationPolicy kill-switch and limits. Changing
    ``application_policy.application_enabled`` here has no runtime effect —
    it only changes what this pure preview function reports as
    ``would_apply``. Real runtime always uses the default (False).
    """
    application_policy: ApplicationPolicy = ApplicationPolicy()


@dataclass(frozen=True)
class ApplicationOrchestrationResult:
    """Public-safe, read-only preview of one candidate's application chain.

    No control modification is performed or implied by this object. No
    storage is read or written by this object — it only reports on inputs
    given to evaluate_application_orchestration().
    """
    candidate_key: str
    candidate_type_value: str            # CandidateType.value
    direction_value: str                 # AdaptationDirection.value
    mode: ApplicationOrchestrationMode
    promotion_readiness: str             # PromotionReadiness.value
    application_enabled: bool            # mirrors policy kill-switch; always False in real runtime
    application_readiness: str           # "eligible" | "blocked" (ignores kill-switch)
    plan_status: str                     # "would_apply" | "blocked_by_kill_switch" | "blocked"
    would_apply: bool                    # real outcome given the supplied policy
    would_apply_if_enabled: bool         # theoretical outcome if kill-switch were on
    blocked_reasons: tuple[str, ...]
    safety_reasons: tuple[str, ...]
    bounded_delta_min: Optional[float]
    current_cumulative_delta_min: Optional[float]
    next_cumulative_delta_min: Optional[float]
    monitoring_required: bool
    rollback_supported: bool
    lifecycle_existing_status: Optional[str]   # MonitoringStatus.value of existing entry, or None
    cooldown_active: bool
    applied_record_preview: Optional[dict]     # preview only — never persisted by this object

    def to_dict(self) -> dict:
        """Public-safe, read-only serialization. No control fields."""
        return {
            "candidate_key":              self.candidate_key,
            "candidate_type":              self.candidate_type_value,
            "direction":                   self.direction_value,
            "mode":                        self.mode.value,
            "promotion_readiness":         self.promotion_readiness,
            "application_enabled":         self.application_enabled,
            "application_readiness":       self.application_readiness,
            "plan_status":                 self.plan_status,
            "would_apply":                 self.would_apply,
            "would_apply_if_enabled":      self.would_apply_if_enabled,
            "blocked_reasons":             list(self.blocked_reasons),
            "safety_reasons":              list(self.safety_reasons),
            "bounded_delta_min":           self.bounded_delta_min,
            "current_cumulative_delta_min": self.current_cumulative_delta_min,
            "next_cumulative_delta_min":   self.next_cumulative_delta_min,
            "monitoring_required":         self.monitoring_required,
            "rollback_supported":          self.rollback_supported,
            "lifecycle_existing_status":   self.lifecycle_existing_status,
            "cooldown_active":             self.cooldown_active,
            "applied_record_preview":      self.applied_record_preview,
        }


def _lifecycle_block_reasons(
    *,
    lifecycle_state: Optional[ApplicationLifecycleState],
    candidate_key: str,
    now_ts: Optional[str],
) -> tuple[list[str], Optional[str], bool]:
    """Return (block_reasons, existing_status_value, cooldown_active).

    lifecycle_state is None (storage/error path) → conservative hard block.
    Empty lifecycle_state (no entries yet) → no block, evaluation continues.
    """
    if lifecycle_state is None:
        return (["lifecycle_state_unavailable"], None, False)

    existing = lifecycle_state.entries.get(candidate_key)
    if existing is None:
        return ([], None, False)

    reasons: list[str] = []
    if existing.status.value in _ACTIVE_LIFECYCLE_STATUSES:
        reasons.append("lifecycle_entry_active")
    elif existing.status is MonitoringStatus.ADOPTED:
        reasons.append("lifecycle_already_adopted")
    elif existing.status is MonitoringStatus.ROLLED_BACK:
        reasons.append("lifecycle_rolled_back")
    elif existing.status is MonitoringStatus.WINDOW_EXPIRED:
        reasons.append("lifecycle_window_expired")

    cooldown_active = is_cooldown_active(existing, now_ts=now_ts)
    if cooldown_active:
        reasons.append("lifecycle_cooldown_active")

    return (reasons, existing.status.value, cooldown_active)


def evaluate_application_orchestration(
    *,
    history_entry: CandidateHistoryEntry,
    lifecycle_state: Optional[ApplicationLifecycleState],
    runtime_context: Optional[ApplicationRuntimeContext],
    span_days: float,
    confounder_ratio: float,
    now_ts: Optional[str] = None,
    policy: Optional[ApplicationOrchestratorPolicy] = None,
) -> ApplicationOrchestrationResult:
    """Build a full, traceable, non-mutating preview for one candidate.

    Pure function — no HA imports, no control modification, no storage
    mutation, no side effects. Never writes to ``lifecycle_state`` (the
    caller's object is only read from). Never raises.

    Caller contract: ``lifecycle_state`` (when not None) must belong to the same
    learning zone as ``history_entry`` — candidate_key lookup is a plain dict
    read with no independent zone cross-check, since CandidateHistoryEntry
    carries no zone_id field of its own (the zone is already folded into the
    opaque candidate_key hash). Callers own one ApplicationLifecycleState per
    zone (see LearningShadowController), so this holds naturally in practice.

    Args:
        history_entry:    accumulated candidate history entry.
        lifecycle_state:  current ApplicationLifecycleState for the SAME zone as
                          history_entry, or None when the storage layer could
                          not load one (conservative block).
        runtime_context:  live safety snapshot; None → conservative block via
                          evaluate_application_readiness's "unknown_context".
        span_days:        calendar days between first/last sighting (caller
                          computed, as required by evaluate_promotion_readiness).
        confounder_ratio: fraction of rejected outcome attempts due to confounders.
        now_ts:           ISO 8601 UTC reference clock, used for the lifecycle
                          cooldown check and as the (theoretical) applied_ts of
                          the optional AppliedAdaptationRecord preview.
        policy:           orchestrator policy; defaults to
                          ApplicationOrchestratorPolicy() (kill-switch False).

    Returns:
        ApplicationOrchestrationResult. Never raises.
    """
    _policy = policy or ApplicationOrchestratorPolicy()
    app_policy = _policy.application_policy

    promotion_result = evaluate_promotion_readiness(
        history_entry, span_days=span_days, confounder_ratio=confounder_ratio,
    )

    application_decision: ApplicationDecision = evaluate_application_readiness(
        history_entry=history_entry,
        promotion_result=promotion_result,
        policy=app_policy,
        runtime_context=runtime_context,
    )

    plan: ApplicationPlan = build_preheat_application_plan(
        history_entry=history_entry,
        promotion_result=promotion_result,
        application_decision=application_decision,
        policy=app_policy,
        runtime_context=runtime_context,
    )

    # plan.blocking_reasons == list(application_decision.blocking_reasons) + plan_reasons
    # (guaranteed by build_preheat_application_plan's construction) — slice off the
    # decision-level prefix to recover the plan-only reasons without recomputation.
    decision_reason_count = len(application_decision.blocking_reasons)
    plan_only_reasons = list(plan.blocking_reasons[decision_reason_count:])
    plan_level_ok = len(plan_only_reasons) == 0

    lifecycle_reasons, lifecycle_existing_status, cooldown_active = _lifecycle_block_reasons(
        lifecycle_state=lifecycle_state,
        candidate_key=history_entry.candidate_key,
        now_ts=now_ts,
    )
    lifecycle_blocked = len(lifecycle_reasons) > 0

    would_apply = plan.would_apply and not lifecycle_blocked
    would_apply_if_enabled = (
        application_decision.would_allow_if_enabled
        and plan_level_ok
        and not lifecycle_blocked
    )

    blocked_reasons = (
        list(application_decision.blocking_reasons)
        + plan_only_reasons
        + lifecycle_reasons
    )

    application_readiness = (
        "eligible" if application_decision.would_allow_if_enabled else "blocked"
    )

    if would_apply:
        plan_status = "would_apply"
    elif would_apply_if_enabled:
        plan_status = "blocked_by_kill_switch"
    else:
        plan_status = "blocked"

    # Preview-only AppliedAdaptationRecord — never persisted, never mutates
    # lifecycle_state. Only built when the (possibly test-only) policy would
    # actually apply, purely to show the shape of what would be recorded.
    applied_record_preview: Optional[dict] = None
    if would_apply:
        record: Optional[AppliedAdaptationRecord] = build_applied_adaptation_record(
            plan=plan, history_entry=history_entry, applied_ts=now_ts or "",
        )
        if record is not None:
            applied_record_preview = record.to_dict()

    return ApplicationOrchestrationResult(
        candidate_key=history_entry.candidate_key,
        candidate_type_value=application_decision.candidate_type_value,
        direction_value=application_decision.direction_value,
        mode=ApplicationOrchestrationMode.PREVIEW_ONLY,
        promotion_readiness=promotion_result.readiness.value,
        application_enabled=application_decision.application_enabled,
        application_readiness=application_readiness,
        plan_status=plan_status,
        would_apply=would_apply,
        would_apply_if_enabled=would_apply_if_enabled,
        blocked_reasons=tuple(blocked_reasons),
        safety_reasons=application_decision.safety_reasons,
        bounded_delta_min=plan.bounded_delta_min,
        current_cumulative_delta_min=plan.current_cumulative_delta_min,
        next_cumulative_delta_min=plan.next_cumulative_delta_min,
        monitoring_required=plan.monitoring_required,
        rollback_supported=plan.rollback_supported,
        lifecycle_existing_status=lifecycle_existing_status,
        cooldown_active=cooldown_active,
        applied_record_preview=applied_record_preview,
    )
