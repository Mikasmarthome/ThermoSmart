"""Pure application-readiness and application-plan architecture for passive
adaptation candidates.

Defines the gate structure and bounded planning layer for candidate application.
No control modification is performed — application_enabled is always False
in real runtime. The Application Layer is runtime-disabled by default.

No HA imports. No runtime state. No control modifications.
All dataclasses are frozen; all functions are pure and never raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import AdaptationDirection, AdaptationLifecycle, CandidateType
from .history import CandidateHistoryEntry, PromotionGateResult, PromotionReadiness

# ── Allowlist — types theoretically applicable in the future ──────────────────
# BOOST_BIAS, TPI_GAIN_BIAS, COMFORT_BIAS_C are reserved and permanently blocked.

_ALLOWED_CANDIDATE_TYPES: frozenset[CandidateType] = frozenset({
    CandidateType.PREHEAT_DELTA_MIN,
    CandidateType.EARLY_CUTOFF_DELTA_MIN,
})

# ── Per-type step and cumulative limits ───────────────────────────────────────

_TYPE_STEP_LIMITS: dict[CandidateType, float] = {
    CandidateType.PREHEAT_DELTA_MIN:      5.0,   # max single-application delta (minutes)
    CandidateType.EARLY_CUTOFF_DELTA_MIN: 3.0,
}

_TYPE_CUMULATIVE_LIMITS: dict[CandidateType, float] = {
    CandidateType.PREHEAT_DELTA_MIN:      15.0,  # max cumulative drift (minutes)
    CandidateType.EARLY_CUTOFF_DELTA_MIN: 10.0,
}

# Safety flag attribute names on ApplicationRuntimeContext (True = active block)
_SAFETY_FLAG_NAMES: tuple[str, ...] = (
    "window_open",
    "manual_override",
    "vacation_mode",
    "summer_mode",
    "heating_unavailable",
    "sensor_unreliable",
    "control_disabled",
    "learning_disabled",
    "active_control_disabled",
)


# ── Policy ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApplicationPolicy:
    """Limits and global kill-switch for candidate application.

    application_enabled is the global kill-switch. It is False by design in
    the current foundation — no Application Layer exists yet.
    """
    application_enabled: bool = False          # global kill-switch; False by design
    max_confounder_ratio: float = 0.15         # above this → block application
    min_cooldown_days: float = 7.0             # min days between applications of same key
    monitoring_window_hours: float = 48.0      # post-apply monitoring window length


# ── Runtime context ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApplicationRuntimeContext:
    """Snapshot of runtime safety state for one application-readiness check.

    None values are treated conservatively for gates that require the field:
    - proposed_delta=None      → blocks step-limit gate (unknown_proposed_delta)
    - current_cumulative=None  → blocks cumulative gate (unknown_cumulative_delta)
    - now_ts=None with last_applied_ts set → blocks cooldown gate
    - Safety flags: None = absent = not actively blocking (no false alarm)
    """
    # Proposed delta from the most recent candidate trace
    proposed_delta: Optional[float] = None
    # Existing cumulative drift for this candidate key (sum of prior applications)
    current_cumulative_delta: Optional[float] = None
    # ISO 8601 UTC timestamp of last application of same candidate_key; None = no prior
    last_applied_ts: Optional[str] = None
    # Reference clock (ISO 8601 UTC) for cooldown check; None → cooldown gate blocks
    now_ts: Optional[str] = None
    # Safety flags — True = active blocking condition; None = absent (not checked)
    window_open: Optional[bool] = None
    manual_override: Optional[bool] = None
    vacation_mode: Optional[bool] = None
    summer_mode: Optional[bool] = None
    heating_unavailable: Optional[bool] = None
    sensor_unreliable: Optional[bool] = None
    control_disabled: Optional[bool] = None
    learning_disabled: Optional[bool] = None
    active_control_disabled: Optional[bool] = None
    # Storage health
    storage_corrupt: Optional[bool] = None
    # Outcome model confounder ratio [0.0, 1.0]
    confounder_ratio: Optional[float] = None


# ── Decision ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApplicationDecision:
    """Pure result of an application-readiness evaluation.

    No control modification is performed by this object.
    application_enabled is always False in this foundation step.
    """
    candidate_key: str
    candidate_type_value: str       # CandidateType.value
    direction_value: str            # AdaptationDirection.value
    application_enabled: bool       # always False; kill-switch value from policy
    would_allow_if_enabled: bool    # True iff all gates except kill-switch pass
    candidate_type_allowed: bool    # True for PREHEAT_DELTA_MIN / EARLY_CUTOFF_DELTA_MIN
    max_step_delta: Optional[float]       # type-specific step limit; None if type blocked
    max_cumulative_delta: Optional[float] # type-specific cumulative limit
    blocking_reasons: tuple[str, ...]     # all blocking reason strings (ordered)
    safety_reasons: tuple[str, ...]       # subset: "safety_*" reason strings

    def to_dict(self) -> dict:
        """Public-safe, read-only serialization. No control fields."""
        return {
            "candidate_key":          self.candidate_key,
            "candidate_type":         self.candidate_type_value,
            "direction":              self.direction_value,
            "application_enabled":    self.application_enabled,
            "would_allow_if_enabled": self.would_allow_if_enabled,
            "candidate_type_allowed": self.candidate_type_allowed,
            "max_step_delta":         self.max_step_delta,
            "max_cumulative_delta":   self.max_cumulative_delta,
            "blocking_reasons":       list(self.blocking_reasons),
            "safety_reasons":         list(self.safety_reasons),
        }


# ── Static preview (no runtime context) ──────────────────────────────────────

def application_readiness_preview(entry: CandidateHistoryEntry) -> dict:
    """Return a static, public-safe application readiness preview.

    No runtime context required. Shows type-level limits and allowlist status.
    application_enabled and would_allow_if_enabled are always False
    (runtime context absent → conservative; kill-switch off by design).

    Suitable for research export where live runtime state is unavailable.
    """
    ctype = entry.candidate_type
    type_allowed = ctype in _ALLOWED_CANDIDATE_TYPES
    return {
        "application_enabled":    False,
        "would_allow_if_enabled": False,
        "blocking_reasons":       ["application_disabled"],
        "candidate_type_allowed": type_allowed,
        "max_step_delta":         _TYPE_STEP_LIMITS.get(ctype) if type_allowed else None,
        "max_cumulative_delta":   _TYPE_CUMULATIVE_LIMITS.get(ctype) if type_allowed else None,
    }


# ── Main gate evaluation ──────────────────────────────────────────────────────

def evaluate_application_readiness(
    *,
    history_entry: CandidateHistoryEntry,
    promotion_result: PromotionGateResult,
    policy: Optional[ApplicationPolicy] = None,
    runtime_context: Optional[ApplicationRuntimeContext] = None,
) -> ApplicationDecision:
    """Evaluate whether a SHADOW candidate could theoretically be applied.

    Pure function — no HA imports, no control modification, no side effects.
    All gates are evaluated (no short-circuit) so all blocking reasons appear.

    application_enabled is always False in this foundation — the result records
    what would happen if it were enabled, but nothing is ever applied.

    Args:
        history_entry:    accumulated candidate history entry (must be SHADOW).
        promotion_result: result of evaluate_promotion_readiness() for this entry.
        policy:           limits and kill-switch; defaults to ApplicationPolicy().
        runtime_context:  live safety snapshot; None → "unknown_context" blocks
                          all runtime-dependent gates conservatively.

    Returns:
        ApplicationDecision with per-reason analysis. Never raises.
    """
    _policy = policy or ApplicationPolicy()
    ctype = history_entry.candidate_type

    reasons: list[str] = []
    safety: list[str] = []

    # ── Gate 1: global kill-switch ────────────────────────────────────────────
    if not _policy.application_enabled:
        reasons.append("application_disabled")

    # ── Gate 2: promotion readiness must be ELIGIBLE ──────────────────────────
    if promotion_result.readiness is not PromotionReadiness.ELIGIBLE:
        reasons.append(f"not_eligible:{promotion_result.readiness.value}")

    # ── Gate 3: lifecycle must be SHADOW ──────────────────────────────────────
    if history_entry.last_lifecycle is not AdaptationLifecycle.SHADOW:
        reasons.append(f"lifecycle_not_shadow:{history_entry.last_lifecycle.value}")

    # ── Gate 4: candidate type allowlist ─────────────────────────────────────
    type_allowed = ctype in _ALLOWED_CANDIDATE_TYPES
    if not type_allowed:
        reasons.append(f"type_not_allowed:{ctype.value}")

    # ── Gates 5+: runtime-dependent ──────────────────────────────────────────
    if runtime_context is None:
        reasons.append("unknown_context")
    else:
        # Safety flags: block when True; None = absent = not blocking
        for flag in _SAFETY_FLAG_NAMES:
            if getattr(runtime_context, flag, None) is True:
                r = f"safety_{flag}"
                reasons.append(r)
                safety.append(r)

        # Storage corruption
        if runtime_context.storage_corrupt is True:
            r = "safety_storage_corrupt"
            reasons.append(r)
            safety.append(r)

        # Confounder ratio
        if (runtime_context.confounder_ratio is not None
                and runtime_context.confounder_ratio > _policy.max_confounder_ratio):
            reasons.append(
                f"high_confounder_ratio:{runtime_context.confounder_ratio:.2f}"
                f">{_policy.max_confounder_ratio:.2f}"
            )

        # Step delta (only for allowed types — not meaningful for blocked types)
        if type_allowed:
            max_step = _TYPE_STEP_LIMITS.get(ctype)
            if runtime_context.proposed_delta is None:
                reasons.append("unknown_proposed_delta")
            elif max_step is not None and abs(runtime_context.proposed_delta) > max_step:
                reasons.append(
                    f"step_exceeds_limit:"
                    f"{abs(runtime_context.proposed_delta):.1f}>{max_step:.1f}"
                )

            # Cumulative delta
            max_cum = _TYPE_CUMULATIVE_LIMITS.get(ctype)
            if runtime_context.current_cumulative_delta is None:
                reasons.append("unknown_cumulative_delta")
            elif (runtime_context.proposed_delta is not None
                  and max_cum is not None):
                new_cum = abs(
                    runtime_context.current_cumulative_delta
                    + runtime_context.proposed_delta
                )
                if new_cum > max_cum:
                    reasons.append(
                        f"cumulative_exceeds_limit:{new_cum:.1f}>{max_cum:.1f}"
                    )

        # Cooldown
        if runtime_context.last_applied_ts is not None:
            if runtime_context.now_ts is None:
                reasons.append("unknown_now_ts_for_cooldown")
            else:
                try:
                    from datetime import datetime
                    t0 = datetime.fromisoformat(
                        runtime_context.last_applied_ts.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(
                        runtime_context.now_ts.replace("Z", "+00:00"))
                    days_since = (t1 - t0).total_seconds() / 86400.0
                    if days_since < _policy.min_cooldown_days:
                        reasons.append(
                            f"cooldown_active:{days_since:.1f}d"
                            f"<{_policy.min_cooldown_days:.1f}d"
                        )
                except Exception:
                    reasons.append("cooldown_parse_error")

    # ── Derive output values ──────────────────────────────────────────────────
    max_step_out = _TYPE_STEP_LIMITS.get(ctype) if type_allowed else None
    max_cum_out  = _TYPE_CUMULATIVE_LIMITS.get(ctype) if type_allowed else None

    # would_allow_if_enabled: True iff all gates EXCEPT the kill-switch pass
    non_app_reasons = [r for r in reasons if r != "application_disabled"]
    would_allow = (len(non_app_reasons) == 0)

    return ApplicationDecision(
        candidate_key=history_entry.candidate_key,
        candidate_type_value=ctype.value,
        direction_value=history_entry.direction.value,
        application_enabled=_policy.application_enabled,
        would_allow_if_enabled=would_allow,
        candidate_type_allowed=type_allowed,
        max_step_delta=max_step_out,
        max_cumulative_delta=max_cum_out,
        blocking_reasons=tuple(reasons),
        safety_reasons=tuple(safety),
    )


# ── Application Plan ──────────────────────────────────────────────────────────

# Bounded planning constants for PREHEAT_DELTA_MIN
_PLAN_PREHEAT_MAX_STEP       = _TYPE_STEP_LIMITS[CandidateType.PREHEAT_DELTA_MIN]       # 5.0 min
_PLAN_PREHEAT_MAX_CUMULATIVE = _TYPE_CUMULATIVE_LIMITS[CandidateType.PREHEAT_DELTA_MIN] # 15.0 min

# Directions supported in the PREHEAT bounded plan
_PLAN_SUPPORTED_DIRECTIONS: frozenset[AdaptationDirection] = frozenset({
    AdaptationDirection.INCREASE,
    AdaptationDirection.DECREASE,
})


@dataclass(frozen=True)
class ApplicationPlan:
    """Pure, bounded application plan for a PREHEAT_DELTA_MIN candidate.

    Describes what would theoretically be applied, how it is bounded, and
    what monitoring would be required after application.

    No control modification is performed by this object.
    application_enabled mirrors the policy kill-switch — always False in
    real runtime.  would_apply is only True in tests with an explicit
    ApplicationPolicy(application_enabled=True).
    """
    candidate_key: str
    candidate_type_value: str               # CandidateType.value
    direction_value: str                    # AdaptationDirection.value
    requested_delta_min: Optional[float]    # raw proposed delta from runtime context
    bounded_delta_min: Optional[float]      # capped to ±max_step; preserves sign
    current_cumulative_delta_min: Optional[float]   # prior cumulative drift
    next_cumulative_delta_min: Optional[float]       # current + bounded (if known)
    would_apply: bool                       # True only when kill-switch enabled + all gates pass
    application_enabled: bool               # mirrors kill-switch; always False in real runtime
    blocking_reasons: tuple[str, ...]       # all reasons (decision-level + plan-level)
    monitoring_required: bool               # always True — contract for the Application Layer
    rollback_supported: bool                # always True — contract for the Application Layer
    monitoring_window_hours: float          # length of post-apply observation window
    rollback_reason: Optional[str]          # None in plan phase; populated post-apply

    def to_dict(self) -> dict:
        """Public-safe, read-only serialization. No control fields."""
        return {
            "candidate_key":               self.candidate_key,
            "candidate_type":              self.candidate_type_value,
            "direction":                   self.direction_value,
            "requested_delta_min":         self.requested_delta_min,
            "bounded_delta_min":           self.bounded_delta_min,
            "current_cumulative_delta_min": self.current_cumulative_delta_min,
            "next_cumulative_delta_min":   self.next_cumulative_delta_min,
            "would_apply":                 self.would_apply,
            "application_enabled":         self.application_enabled,
            "blocking_reasons":            list(self.blocking_reasons),
            "monitoring_required":         self.monitoring_required,
            "rollback_supported":          self.rollback_supported,
            "monitoring_window_hours":     self.monitoring_window_hours,
            "rollback_reason":             self.rollback_reason,
        }


def build_preheat_application_plan(
    *,
    history_entry: CandidateHistoryEntry,
    promotion_result: PromotionGateResult,
    application_decision: ApplicationDecision,
    policy: Optional[ApplicationPolicy] = None,
    runtime_context: Optional[ApplicationRuntimeContext] = None,
) -> ApplicationPlan:
    """Build a bounded application plan for a PREHEAT_DELTA_MIN candidate.

    Pure function — no HA imports, no control modification, no side effects.
    Only PREHEAT_DELTA_MIN is handled; all other types return a plan blocked
    with "plan_type_not_supported".

    The plan computes a bounded delta (capped to ±5 min) and a projected
    cumulative drift. application_enabled mirrors the kill-switch from the
    pre-evaluated ApplicationDecision — it is always False in real runtime.

    would_apply is True only when:
      - application_decision.application_enabled is True (tests only)
      - application_decision.would_allow_if_enabled is True
      - no plan-level gate blocks (direction, delta, cumulative)

    No control modification of any kind is performed or implied.
    monitoring_required and rollback_supported are
    always True as structural contracts for a future Application Layer.

    Args:
        history_entry:        accumulated candidate history (SHADOW lifecycle).
        promotion_result:     output of evaluate_promotion_readiness().
        application_decision: output of evaluate_application_readiness().
        policy:               limits; defaults to ApplicationPolicy().
        runtime_context:      live snapshot providing proposed_delta and
                              current_cumulative_delta; None → plan blocks
                              with "missing_requested_delta".

    Returns:
        ApplicationPlan describing what would be applied if enabled.
        Never raises.
    """
    _policy = policy or ApplicationPolicy()
    ctype = history_entry.candidate_type
    direction = history_entry.direction

    plan_reasons: list[str] = []

    # ── Plan gate 1: only PREHEAT_DELTA_MIN ──────────────────────────────────
    is_preheat = (ctype is CandidateType.PREHEAT_DELTA_MIN)
    if not is_preheat:
        plan_reasons.append(f"plan_type_not_supported:{ctype.value}")

    # ── Plan gate 2: direction must be INCREASE or DECREASE ──────────────────
    direction_ok = direction in _PLAN_SUPPORTED_DIRECTIONS
    if not direction_ok:
        plan_reasons.append(f"unsupported_direction:{direction.value}")

    # ── Plan gate 3: requested delta from runtime context ─────────────────────
    requested_delta: Optional[float] = None
    bounded_delta: Optional[float] = None

    if is_preheat:
        if runtime_context is None or runtime_context.proposed_delta is None:
            plan_reasons.append("missing_requested_delta")
        else:
            requested_delta = runtime_context.proposed_delta
            if direction_ok:
                # Validate sign-direction consistency
                if direction is AdaptationDirection.INCREASE and requested_delta < 0:
                    plan_reasons.append("direction_delta_mismatch:increase_but_negative")
                elif direction is AdaptationDirection.DECREASE and requested_delta > 0:
                    plan_reasons.append("direction_delta_mismatch:decrease_but_positive")
                else:
                    # Bound to ±max_step — preserves sign
                    sign = 1.0 if requested_delta >= 0 else -1.0
                    bounded_delta = sign * min(abs(requested_delta), _PLAN_PREHEAT_MAX_STEP)

    # ── Plan gate 4: cumulative limit check with bounded delta ─────────────────
    current_cumulative: Optional[float] = (
        runtime_context.current_cumulative_delta
        if runtime_context is not None else None
    )
    next_cumulative: Optional[float] = None

    if is_preheat and bounded_delta is not None and current_cumulative is not None:
        next_cumulative = current_cumulative + bounded_delta
        if abs(next_cumulative) > _PLAN_PREHEAT_MAX_CUMULATIVE:
            plan_reasons.append(
                f"plan_cumulative_exceeds_limit:"
                f"{abs(next_cumulative):.1f}>{_PLAN_PREHEAT_MAX_CUMULATIVE:.1f}"
            )

    # ── Combine: decision-level reasons + plan-level reasons ─────────────────
    all_reasons = list(application_decision.blocking_reasons) + plan_reasons

    # ── would_apply: kill-switch AND readiness AND no plan-level blocks ───────
    # In real runtime application_decision.application_enabled is always False.
    # Only an explicit ApplicationPolicy(application_enabled=True) in tests
    # can make would_apply True.
    would_apply = (
        application_decision.application_enabled
        and application_decision.would_allow_if_enabled
        and len(plan_reasons) == 0
    )

    return ApplicationPlan(
        candidate_key=history_entry.candidate_key,
        candidate_type_value=ctype.value,
        direction_value=direction.value,
        requested_delta_min=requested_delta,
        bounded_delta_min=bounded_delta,
        current_cumulative_delta_min=current_cumulative,
        next_cumulative_delta_min=next_cumulative,
        would_apply=would_apply,
        application_enabled=application_decision.application_enabled,
        blocking_reasons=tuple(all_reasons),
        monitoring_required=True,
        rollback_supported=True,
        monitoring_window_hours=_policy.monitoring_window_hours,
        rollback_reason=None,
    )
