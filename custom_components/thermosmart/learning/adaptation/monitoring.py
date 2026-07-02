"""Pure monitoring and rollback contracts for applied adaptation records.

Defines the evaluation pipeline for post-application monitoring:

    ApplicationPlan
    → build_applied_adaptation_record()  →  AppliedAdaptationRecord
    → evaluate_applied_adaptation_outcome()  →  MonitoringEvaluation
    → keep / rollback / adopt / continue_monitoring

No real adoption is performed — adoption_ready is a readiness flag only.
No control modification of any kind is performed or implied.
No HA imports. No runtime state.
All dataclasses are frozen; all functions are pure and never raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .application import ApplicationPlan
from .history import CandidateHistoryEntry


# ── Monitoring status ─────────────────────────────────────────────────────────

class MonitoringStatus(Enum):
    """Lifecycle of a post-application monitoring record.

    ADOPTED and ROLLED_BACK are reserved for a future runtime Application Layer
    that performs the actual state change. This foundation only reaches
    ADOPTION_CANDIDATE and ROLLBACK_RECOMMENDED.
    """
    PENDING              = "pending"               # record created, no outcomes yet
    IN_PROGRESS          = "in_progress"           # outcomes being collected
    ADOPTION_CANDIDATE   = "adoption_candidate"    # all gates met — ready for adoption
    ROLLBACK_RECOMMENDED = "rollback_recommended"  # regression detected
    WINDOW_EXPIRED       = "window_expired"        # window elapsed without decision
    ADOPTED              = "adopted"               # reserved — Application Layer only
    ROLLED_BACK          = "rolled_back"           # reserved — Application Layer only


# ── Evaluation thresholds ─────────────────────────────────────────────────────

_ROLLBACK_QUALITY_THRESHOLD: float = -0.10   # quality delta below this → rollback
_ROLLBACK_RATE_THRESHOLD:    float = +0.10   # rate delta above this → rollback
_ADOPTION_QUALITY_THRESHOLD: float = +0.05   # minimum quality delta for adoption
_ADOPTION_MIN_OUTCOMES:      int   = 3       # minimum post-apply outcomes for adoption


# ── AppliedAdaptationRecord ───────────────────────────────────────────────────

@dataclass(frozen=True)
class AppliedAdaptationRecord:
    """Immutable snapshot of a plan that was (theoretically) applied.

    Built from a would_apply=True ApplicationPlan. Records the applied delta,
    cumulative state, monitoring window parameters, and pre-application baselines.

    No control modification is performed by this object.
    rollback_supported is always True as a structural contract.
    status starts as PENDING and is updated by evaluate_applied_adaptation_outcome().
    """
    candidate_key: str
    candidate_type_value: str
    direction_value: str
    applied_delta_min: float                    # == plan.bounded_delta_min
    previous_cumulative_delta_min: float        # cumulative before this application
    new_cumulative_delta_min: float             # cumulative after this application
    applied_ts: str                             # ISO 8601 UTC of (theoretical) application
    monitoring_window_hours: float
    monitoring_deadline_ts: Optional[str]       # applied_ts + monitoring_window_hours
    baseline_outcome_quality: Optional[float]   # from history_entry.avg_outcome_quality
    baseline_timeout_rate: Optional[float]      # from history_entry.avg_timeout_rate
    baseline_overshoot_rate: Optional[float]    # from history_entry.avg_overshoot_rate
    status: MonitoringStatus                    # starts PENDING
    rollback_supported: bool                    # always True

    def to_dict(self) -> dict:
        """Public-safe, read-only serialization. No control fields."""
        return {
            "candidate_key":                 self.candidate_key,
            "candidate_type":                self.candidate_type_value,
            "direction":                     self.direction_value,
            "applied_delta_min":             self.applied_delta_min,
            "previous_cumulative_delta_min": self.previous_cumulative_delta_min,
            "new_cumulative_delta_min":      self.new_cumulative_delta_min,
            "applied_ts":                    self.applied_ts,
            "monitoring_window_hours":       self.monitoring_window_hours,
            "monitoring_deadline_ts":        self.monitoring_deadline_ts,
            "baseline_outcome_quality":      self.baseline_outcome_quality,
            "baseline_timeout_rate":         self.baseline_timeout_rate,
            "baseline_overshoot_rate":       self.baseline_overshoot_rate,
            "status":                        self.status.value,
            "rollback_supported":            self.rollback_supported,
        }


# ── MonitoringEvaluation ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonitoringEvaluation:
    """Pure result of evaluating post-application outcomes against baseline.

    No control modification is performed by this object.
    adoption_ready=True is a readiness flag only — no real adoption occurs here.
    rollback_recommended=True is a recommendation only — no real rollback occurs here.
    """
    candidate_key: str
    status: MonitoringStatus
    monitored_outcome_count: int
    outcome_quality_delta: Optional[float]    # avg_monitored - baseline; None if missing
    timeout_rate_delta: Optional[float]
    overshoot_rate_delta: Optional[float]
    manual_override_detected: bool
    confounder_detected: bool
    safety_flag_detected: bool
    rollback_recommended: bool
    rollback_reasons: tuple[str, ...]         # sorted deterministically
    adoption_ready: bool                      # readiness flag only; no real adoption
    continue_monitoring: bool

    def to_dict(self) -> dict:
        """Public-safe, read-only serialization. No control fields."""
        return {
            "candidate_key":           self.candidate_key,
            "status":                  self.status.value,
            "monitored_outcome_count": self.monitored_outcome_count,
            "outcome_quality_delta":   self.outcome_quality_delta,
            "timeout_rate_delta":      self.timeout_rate_delta,
            "overshoot_rate_delta":    self.overshoot_rate_delta,
            "manual_override_detected": self.manual_override_detected,
            "confounder_detected":     self.confounder_detected,
            "safety_flag_detected":    self.safety_flag_detected,
            "rollback_recommended":    self.rollback_recommended,
            "rollback_reasons":        list(self.rollback_reasons),
            "adoption_ready":          self.adoption_ready,
            "continue_monitoring":     self.continue_monitoring,
        }


# ── build_applied_adaptation_record ──────────────────────────────────────────

def build_applied_adaptation_record(
    *,
    plan: ApplicationPlan,
    history_entry: CandidateHistoryEntry,
    applied_ts: str,
) -> Optional[AppliedAdaptationRecord]:
    """Build an AppliedAdaptationRecord from a would_apply=True plan.

    Pure function — no HA imports, no control modification, no side effects.

    Returns None when:
    - plan.would_apply is False (kill-switch or gate blocks)
    - plan.bounded_delta_min is None (direction/delta gate failed)
    - cumulative fields are missing from the plan

    Args:
        plan:          ApplicationPlan from build_preheat_application_plan().
        history_entry: the CandidateHistoryEntry the plan was built from;
                       provides pre-application baseline rates.
        applied_ts:    ISO 8601 UTC timestamp of (theoretical) application.

    Returns:
        AppliedAdaptationRecord with PENDING status, or None if plan not applicable.
        Never raises.
    """
    if not plan.would_apply:
        return None
    if plan.bounded_delta_min is None:
        return None
    if (plan.current_cumulative_delta_min is None
            or plan.next_cumulative_delta_min is None):
        return None

    deadline_ts: Optional[str] = None
    try:
        from datetime import datetime, timedelta
        t0 = datetime.fromisoformat(applied_ts.replace("Z", "+00:00"))
        deadline = t0 + timedelta(hours=plan.monitoring_window_hours)
        deadline_ts = deadline.isoformat()
    except Exception:
        pass

    return AppliedAdaptationRecord(
        candidate_key=plan.candidate_key,
        candidate_type_value=plan.candidate_type_value,
        direction_value=plan.direction_value,
        applied_delta_min=plan.bounded_delta_min,
        previous_cumulative_delta_min=plan.current_cumulative_delta_min,
        new_cumulative_delta_min=plan.next_cumulative_delta_min,
        applied_ts=applied_ts,
        monitoring_window_hours=plan.monitoring_window_hours,
        monitoring_deadline_ts=deadline_ts,
        baseline_outcome_quality=history_entry.avg_outcome_quality,
        baseline_timeout_rate=history_entry.avg_timeout_rate,
        baseline_overshoot_rate=history_entry.avg_overshoot_rate,
        status=MonitoringStatus.PENDING,
        rollback_supported=True,
    )


# ── evaluate_applied_adaptation_outcome ──────────────────────────────────────

def evaluate_applied_adaptation_outcome(
    *,
    record: AppliedAdaptationRecord,
    monitored_outcome_count: int,
    avg_monitored_outcome_quality: Optional[float] = None,
    avg_monitored_timeout_rate: Optional[float] = None,
    avg_monitored_overshoot_rate: Optional[float] = None,
    manual_override_detected: bool = False,
    confounder_detected: bool = False,
    safety_flag_detected: bool = False,
    monitoring_window_complete: bool = False,
) -> MonitoringEvaluation:
    """Evaluate post-application outcomes against the pre-application baseline.

    Pure function — no HA imports, no control modification, no side effects.
    Conservative by design: prefer continue_monitoring over adoption.

    Rollback is recommended when any of:
    - outcome_quality_delta < -0.10
    - timeout_rate_delta > +0.10
    - overshoot_rate_delta > +0.10
    - manual_override_detected is True
    - confounder_detected is True
    - safety_flag_detected is True
    - monitoring window expired and quality improvement < +0.05

    Adoption is ready only when ALL of:
    - monitored_outcome_count >= 3
    - outcome_quality_delta >= +0.05
    - timeout_rate_delta <= 0
    - overshoot_rate_delta <= 0
    - no manual override, confounder, or safety flag
    - monitoring_window_complete is True
    - no rollback triggers

    Args:
        record:                       AppliedAdaptationRecord to evaluate.
        monitored_outcome_count:      outcome cycles observed post-application.
        avg_monitored_outcome_quality: post-application quality average; None = unknown.
        avg_monitored_timeout_rate:   post-application timeout rate; None = unknown.
        avg_monitored_overshoot_rate: post-application overshoot rate; None = unknown.
        manual_override_detected:     user override during monitoring window.
        confounder_detected:          confounder rejection during monitoring window.
        safety_flag_detected:         safety flag active during monitoring window.
        monitoring_window_complete:   True when monitoring_window_hours have elapsed.

    Returns:
        MonitoringEvaluation with deterministic rollback_reasons and status.
        Never raises.
    """
    # ── Compute deltas ────────────────────────────────────────────────────────
    quality_delta: Optional[float] = None
    timeout_delta: Optional[float] = None
    overshoot_delta: Optional[float] = None

    if (avg_monitored_outcome_quality is not None
            and record.baseline_outcome_quality is not None):
        quality_delta = avg_monitored_outcome_quality - record.baseline_outcome_quality

    if (avg_monitored_timeout_rate is not None
            and record.baseline_timeout_rate is not None):
        timeout_delta = avg_monitored_timeout_rate - record.baseline_timeout_rate

    if (avg_monitored_overshoot_rate is not None
            and record.baseline_overshoot_rate is not None):
        overshoot_delta = avg_monitored_overshoot_rate - record.baseline_overshoot_rate

    missing_data = (
        avg_monitored_outcome_quality is None
        or avg_monitored_timeout_rate is None
        or avg_monitored_overshoot_rate is None
    )

    # ── Rollback evaluation (all triggers checked) ────────────────────────────
    rollback_reasons: list[str] = []

    if quality_delta is not None and quality_delta < _ROLLBACK_QUALITY_THRESHOLD:
        rollback_reasons.append(
            f"outcome_quality_degraded:{quality_delta:.3f}<{_ROLLBACK_QUALITY_THRESHOLD:.2f}"
        )
    if timeout_delta is not None and timeout_delta > _ROLLBACK_RATE_THRESHOLD:
        rollback_reasons.append(
            f"timeout_rate_increased:{timeout_delta:.3f}>{_ROLLBACK_RATE_THRESHOLD:.2f}"
        )
    if overshoot_delta is not None and overshoot_delta > _ROLLBACK_RATE_THRESHOLD:
        rollback_reasons.append(
            f"overshoot_rate_increased:{overshoot_delta:.3f}>{_ROLLBACK_RATE_THRESHOLD:.2f}"
        )
    if manual_override_detected:
        rollback_reasons.append("manual_override_detected")
    if confounder_detected:
        rollback_reasons.append("confounder_detected")
    if safety_flag_detected:
        rollback_reasons.append("safety_flag_detected")

    # Window-expiry rollback: no improvement enough to justify adoption
    if (monitoring_window_complete
            and not rollback_reasons
            and quality_delta is not None
            and quality_delta < _ADOPTION_QUALITY_THRESHOLD):
        rollback_reasons.append("window_expired_no_improvement")

    rollback_recommended = len(rollback_reasons) > 0

    # ── Adoption evaluation (conservative — all conditions required) ──────────
    adoption_ready = (
        not missing_data
        and not rollback_recommended
        and monitored_outcome_count >= _ADOPTION_MIN_OUTCOMES
        and quality_delta is not None
        and quality_delta >= _ADOPTION_QUALITY_THRESHOLD
        and timeout_delta is not None
        and timeout_delta <= 0.0
        and overshoot_delta is not None
        and overshoot_delta <= 0.0
        and not manual_override_detected
        and not confounder_detected
        and not safety_flag_detected
        and monitoring_window_complete
    )

    continue_monitoring = not rollback_recommended and not adoption_ready

    # ── Status derivation ─────────────────────────────────────────────────────
    if rollback_recommended:
        status = MonitoringStatus.ROLLBACK_RECOMMENDED
    elif adoption_ready:
        status = MonitoringStatus.ADOPTION_CANDIDATE
    elif monitoring_window_complete:
        status = MonitoringStatus.WINDOW_EXPIRED
    else:
        status = MonitoringStatus.IN_PROGRESS

    return MonitoringEvaluation(
        candidate_key=record.candidate_key,
        status=status,
        monitored_outcome_count=monitored_outcome_count,
        outcome_quality_delta=quality_delta,
        timeout_rate_delta=timeout_delta,
        overshoot_rate_delta=overshoot_delta,
        manual_override_detected=manual_override_detected,
        confounder_detected=confounder_detected,
        safety_flag_detected=safety_flag_detected,
        rollback_recommended=rollback_recommended,
        rollback_reasons=tuple(sorted(rollback_reasons)),
        adoption_ready=adoption_ready,
        continue_monitoring=continue_monitoring,
    )


# ── Convenience helper ────────────────────────────────────────────────────────

def should_rollback_adaptation(evaluation: MonitoringEvaluation) -> bool:
    """Return True if the MonitoringEvaluation recommends rollback."""
    return evaluation.rollback_recommended
