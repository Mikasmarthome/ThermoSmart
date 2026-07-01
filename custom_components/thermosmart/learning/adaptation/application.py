"""Pure application-readiness architecture for passive adaptation candidates.

Defines the gate structure that would gate real candidate application.
No control modification is performed — application_enabled is always False
in this foundation step.

No HA imports. No runtime state. No control modifications.
All dataclasses are frozen; the main function is pure and never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import AdaptationLifecycle, CandidateType
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
