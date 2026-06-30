"""Contracts for the Outcome Adaptation Engine foundation (passive trace only).

No HA imports. No runtime state. No control modifications.
All dataclasses are frozen; all enums are stable identifiers for future
Application Layer wiring.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional


class CandidateType(Enum):
    """Stable identifiers for the bounded adaptation action types.

    Values are serialized in traces — must not be renamed once in use.
    """
    PREHEAT_DELTA_MIN      = "preheat_delta_min"
    EARLY_CUTOFF_DELTA_MIN = "early_cutoff_delta_min"
    BOOST_BIAS             = "boost_bias"
    TPI_GAIN_BIAS          = "tpi_gain_bias"
    COMFORT_BIAS_C         = "comfort_bias_c"


class AdaptationLifecycle(Enum):
    """Lifecycle states for a candidate or trace.

    Only CANDIDATE / SHADOW are produced by this foundation.
    The remaining states are reserved for the Application Layer.
    """
    CANDIDATE   = "candidate"    # evaluation produced a candidate — not yet approved
    SHADOW      = "shadow"       # approved for shadow observation, not yet applied
    ELIGIBLE    = "eligible"     # meets all gates for real application (pending review)
    APPLIED     = "applied"      # applied to control — being monitored
    MONITORING  = "monitoring"   # in post-apply observation window
    ADOPTED     = "adopted"      # confirmed better — kept
    ROLLED_BACK = "rolled_back"  # regression detected — reverted
    EXPIRED     = "expired"      # expired without application
    REJECTED    = "rejected"     # blocked by gate or supervisor


class AdaptationDirection(Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    NEUTRAL  = "neutral"


@dataclass(frozen=True)
class SituationContext:
    """Distilled snapshot of the heating situation at outcome time.

    Used for later similar-situation matching. All fields are Optional;
    missing fields reduce matching precision but never block trace generation.
    """
    controller_kind:       Optional[str]   = None
    outdoor_bucket:        Optional[str]   = None
    mode_context:          Optional[str]   = None
    time_of_day_bucket:    Optional[int]   = None   # hour 0-23 at episode start
    weekday:               Optional[int]   = None   # 0=Monday … 6=Sunday
    is_weekend:            Optional[bool]  = None
    preheat_was_active:    Optional[bool]  = None
    boost_was_active:      Optional[bool]  = None
    target_delta_c:        Optional[float] = None   # target - start_temp
    heat_loss_c_per_h:     Optional[float] = None
    preheat_minutes_used:  Optional[float] = None
    active_control:        Optional[bool]  = None   # zone in active control vs observation
    learning_enabled:      Optional[bool]  = None   # learning switch state
    context_time_source:   Optional[str]   = None   # "model_last_update" | "unavailable"

    def to_dict(self) -> dict:
        return {
            "controller_kind":      self.controller_kind,
            "outdoor_bucket":       self.outdoor_bucket,
            "mode_context":         self.mode_context,
            "time_of_day_bucket":   self.time_of_day_bucket,
            "weekday":              self.weekday,
            "is_weekend":           self.is_weekend,
            "preheat_was_active":   self.preheat_was_active,
            "boost_was_active":     self.boost_was_active,
            "target_delta_c":       self.target_delta_c,
            "heat_loss_c_per_h":    self.heat_loss_c_per_h,
            "preheat_minutes_used": self.preheat_minutes_used,
            "active_control":       self.active_control,
            "learning_enabled":     self.learning_enabled,
            "context_time_source":  self.context_time_source,
        }


@dataclass(frozen=True)
class OutcomeSignal:
    """Distilled quality signals from a zone's outcome history.

    Derived from OutcomeModel.diagnostics(). Primary input to gate
    evaluation and candidate generation.
    """
    sample_count:             int
    timeout_rate:             Optional[float]
    overshoot_rate:           Optional[float]
    reached_rate:             Optional[float]
    general_data_quality:     float
    aggregate_reliability:    float
    partial_ratio:            float   # partial_count / (full + partial)
    confounder_contamination: bool    # any high-weight confounders in recent samples

    def to_dict(self) -> dict:
        return {
            "sample_count":             self.sample_count,
            "timeout_rate":             self.timeout_rate,
            "overshoot_rate":           self.overshoot_rate,
            "reached_rate":             self.reached_rate,
            "general_data_quality":     round(self.general_data_quality, 3),
            "aggregate_reliability":    round(self.aggregate_reliability, 3),
            "partial_ratio":            round(self.partial_ratio, 3),
            "confounder_contamination": self.confounder_contamination,
        }


@dataclass(frozen=True)
class AdaptationGateResult:
    """Result of passive gate evaluation for one candidate."""
    passed:       bool
    gate_results: Mapping[str, bool]
    blocking_gate: Optional[str]     # first blocking gate name; None when passed
    reason_codes:  tuple             # human-readable reasons for blocked gates

    def to_dict(self) -> dict:
        return {
            "passed":        self.passed,
            "gate_results":  dict(self.gate_results),
            "blocking_gate": self.blocking_gate,
            "reason_codes":  list(self.reason_codes),
        }


@dataclass(frozen=True)
class AdaptationTrace:
    """Passive adaptation candidate trace — unit of research traceability.

    Produced only in SHADOW or REJECTED state by this foundation.
    The Application Layer may later promote traces through the full lifecycle.
    No control state is modified by this object.
    """
    adaptation_id:   str
    zone_hash:       str                 # anonymized zone identifier (for export)
    candidate_type:  CandidateType
    direction:       AdaptationDirection
    candidate_value: Optional[float]     # suggested delta; None = direction only
    lifecycle:       AdaptationLifecycle
    situation_context: SituationContext
    outcome_signal:    OutcomeSignal
    gate_result:       AdaptationGateResult
    reason_codes:      tuple
    created_at:        str               # ISO 8601 UTC
    applied_at:        Optional[str] = None   # always None in this foundation
    adopted_at:        Optional[str] = None
    rolled_back_at:    Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "adaptation_id":    self.adaptation_id,
            "zone_hash":        self.zone_hash,
            "candidate_type":   self.candidate_type.value,
            "direction":        self.direction.value,
            "candidate_value":  self.candidate_value,
            "lifecycle":        self.lifecycle.value,
            "situation_context": self.situation_context.to_dict(),
            "outcome_signal":    self.outcome_signal.to_dict(),
            "gate_result":       self.gate_result.to_dict(),
            "reason_codes":      list(self.reason_codes),
            "created_at":        self.created_at,
            "applied_at":        self.applied_at,
            "adopted_at":        self.adopted_at,
            "rolled_back_at":    self.rolled_back_at,
        }


def make_adaptation_id(zone_id: str, candidate_type: CandidateType, created_at: str) -> str:
    """Deterministic, non-reversible adaptation ID (16-char hex digest).

    Suitable for deduplication across restarts. Zone ID is hashed,
    so the output is not privacy-sensitive.
    """
    raw = f"{zone_id}:{candidate_type.value}:{created_at}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
