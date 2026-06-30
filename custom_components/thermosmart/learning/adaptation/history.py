"""Passive adaptation candidate history — pure foundation.

Provides stable candidate identity, history accumulation, and promotion
readiness evaluation.

No HA imports. No Storage writes. No Runtime binding. No Control modification.
All functions are pure; all dataclasses are frozen.

Key design split:
  candidate_key  — stable across exports; identifies "same candidate type
                   in similar situation for this zone". Used for deduplication
                   and history accumulation.
  adaptation_id  — unique per trace instance. Identifies a specific
                   AdaptationTrace at a specific point in time.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .contracts import (
    AdaptationDirection,
    AdaptationLifecycle,
    AdaptationTrace,
    CandidateType,
    SituationContext,
)

# ── Promotion thresholds (intentionally conservative for first version) ────────

_MIN_SEEN_COUNT         = 3     # candidate must be observed at least N times
_MIN_OUTCOME_SAMPLES    = 12    # model must have >= N accepted outcome samples
_MIN_DATA_QUALITY       = 0.45  # aggregate quality floor
_MAX_CONFOUNDER_RATIO   = 0.15  # < N fraction confounder rejections
_MIN_SPAN_DAYS          = 7.0   # signal must span at least N calendar days
_MIN_STABLE_REASON_RATIO = 0.75 # dominant reason must appear in >= N fraction of sightings


# ── Candidate identity ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CandidateIdentity:
    """Coarse situation fingerprint for candidate key derivation.

    Uses only discrete, stable features. Fine-grained values (target_delta_c,
    preheat_minutes_used, time_of_day_bucket) are excluded so that similar but
    not identical situations share the same key.
    """
    zone_id:        str
    candidate_type: CandidateType
    direction:      AdaptationDirection
    outdoor_bucket: Optional[str]  # coarse outdoor temp band ("b0".."b6")
    mode_context:   Optional[str]  # comfort / eco / night / boost / None
    preheat_active: Optional[bool] # preheat was commanded during episode


def make_candidate_key(identity: CandidateIdentity) -> str:
    """Deterministic 16-char hex key for candidate history lookup.

    Stable across exports as long as zone and coarse situation are the same.
    The zone_id is hashed (SHA-256) so the key is not privacy-sensitive.
    """
    zone_hash = hashlib.sha256(identity.zone_id.encode()).hexdigest()[:12]
    raw = (
        f"{zone_hash}"
        f":{identity.candidate_type.value}"
        f":{identity.direction.value}"
        f":{identity.outdoor_bucket or '_'}"
        f":{identity.mode_context or '_'}"
        f":{int(identity.preheat_active) if identity.preheat_active is not None else '_'}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def candidate_identity_from_trace(
    zone_id: str, trace: AdaptationTrace
) -> CandidateIdentity:
    """Extract a CandidateIdentity from a trace, using coarse situation fields."""
    sit = trace.situation_context
    return CandidateIdentity(
        zone_id=zone_id,
        candidate_type=trace.candidate_type,
        direction=trace.direction,
        outdoor_bucket=sit.outdoor_bucket,
        mode_context=sit.mode_context,
        preheat_active=sit.preheat_was_active,
    )


# ── History entry ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CandidateHistoryEntry:
    """Aggregated history record for a shadow candidate across multiple observations.

    Built by repeated calls to accumulate_into_history(). Caller owns storage.
    Only SHADOW traces are accumulated — REJECTED traces are counted but not
    tracked per-candidate (they feed the gate evaluator separately).

    supporting_outcome_count reflects the sample count at the MOST RECENT
    observation, not a cumulative sum (the model grows monotonically; latest
    is the most informative value for gate evaluation).
    """
    candidate_key:              str
    candidate_type:             CandidateType
    direction:                  AdaptationDirection
    first_seen_ts:              Optional[str]   # ISO 8601 UTC of first sighting
    last_seen_ts:               Optional[str]   # ISO 8601 UTC of most recent sighting
    seen_count:                 int             # how many times this candidate appeared
    supporting_outcome_count:   int             # outcome sample count at last sighting
    avg_outcome_quality:        float           # incremental mean of general_data_quality
    avg_timeout_rate:           Optional[float] # incremental mean of timeout_rate
    avg_overshoot_rate:         Optional[float] # incremental mean of overshoot_rate
    last_lifecycle:             AdaptationLifecycle
    dominant_reason:            Optional[str]   # most frequent reason_codes[0]
    dominant_reason_ratio:      float           # fraction of sightings with dominant_reason

    def to_dict(self) -> dict:
        return {
            "candidate_key":              self.candidate_key,
            "candidate_type":             self.candidate_type.value,
            "direction":                  self.direction.value,
            "first_seen_ts":              self.first_seen_ts,
            "last_seen_ts":               self.last_seen_ts,
            "seen_count":                 self.seen_count,
            "supporting_outcome_count":   self.supporting_outcome_count,
            "avg_outcome_quality":        round(self.avg_outcome_quality, 3),
            "avg_timeout_rate": (
                round(self.avg_timeout_rate, 3)
                if self.avg_timeout_rate is not None else None
            ),
            "avg_overshoot_rate": (
                round(self.avg_overshoot_rate, 3)
                if self.avg_overshoot_rate is not None else None
            ),
            "last_lifecycle":             self.last_lifecycle.value,
            "dominant_reason":            self.dominant_reason,
            "dominant_reason_ratio":      round(self.dominant_reason_ratio, 3),
        }


# ── Accumulation ──────────────────────────────────────────────────────────────

def accumulate_into_history(
    existing: Optional[CandidateHistoryEntry],
    trace: AdaptationTrace,
    candidate_key: str,
) -> CandidateHistoryEntry:
    """Merge a new SHADOW trace into an existing history entry, or create one.

    Pure function — always returns a new immutable CandidateHistoryEntry.
    The caller is responsible for keying entries by candidate_key and persisting
    the result. Only SHADOW-lifecycle traces should be accumulated; the caller
    should filter before calling.

    Args:
        existing: current history entry for this candidate_key, or None.
        trace: the new AdaptationTrace to merge.
        candidate_key: pre-computed key (from make_candidate_key).
    """
    sig      = trace.outcome_signal
    new_ts   = trace.created_at or None
    new_reason = trace.reason_codes[0] if trace.reason_codes else None

    if existing is None:
        return CandidateHistoryEntry(
            candidate_key=candidate_key,
            candidate_type=trace.candidate_type,
            direction=trace.direction,
            first_seen_ts=new_ts,
            last_seen_ts=new_ts,
            seen_count=1,
            supporting_outcome_count=sig.sample_count,
            avg_outcome_quality=round(sig.general_data_quality, 4),
            avg_timeout_rate=sig.timeout_rate,
            avg_overshoot_rate=sig.overshoot_rate,
            last_lifecycle=trace.lifecycle,
            dominant_reason=new_reason,
            dominant_reason_ratio=1.0,
        )

    n   = existing.seen_count + 1
    old = existing.seen_count

    # Incremental mean: Q_n = (Q_{n-1} * (n-1) + x_n) / n
    new_quality = (existing.avg_outcome_quality * old + sig.general_data_quality) / n
    new_t_rate  = (
        ((existing.avg_timeout_rate  or 0.0) * old + (sig.timeout_rate  or 0.0)) / n
    )
    new_o_rate  = (
        ((existing.avg_overshoot_rate or 0.0) * old + (sig.overshoot_rate or 0.0)) / n
    )

    # Dominant reason: approximate majority tracking (streaming).
    # dom_count = approximate count of sightings with the current dominant reason.
    dom_count = round(existing.dominant_reason_ratio * old)
    if new_reason == existing.dominant_reason:
        dom_count += 1
        dominant_reason = existing.dominant_reason
    else:
        # existing dominant still leads if it remains the majority
        if dom_count / n >= 0.5:
            dominant_reason = existing.dominant_reason
        else:
            # flip: new_reason is now dominant; its count is n - dom_count
            dom_count = n - dom_count
            dominant_reason = new_reason

    return CandidateHistoryEntry(
        candidate_key=existing.candidate_key,
        candidate_type=existing.candidate_type,
        direction=existing.direction,
        first_seen_ts=existing.first_seen_ts,
        last_seen_ts=new_ts if new_ts else existing.last_seen_ts,
        seen_count=n,
        supporting_outcome_count=sig.sample_count,  # latest, not cumulative
        avg_outcome_quality=round(new_quality, 4),
        avg_timeout_rate=round(new_t_rate, 4),
        avg_overshoot_rate=round(new_o_rate, 4),
        last_lifecycle=trace.lifecycle,
        dominant_reason=dominant_reason,
        dominant_reason_ratio=round(dom_count / n, 4),
    )


# ── Promotion readiness ────────────────────────────────────────────────────────

class PromotionReadiness(Enum):
    NOT_EVALUATED = "not_evaluated"
    ELIGIBLE      = "eligible"  # all gates pass — ready for promotion review
    BLOCKED       = "blocked"   # one or more gates fail


@dataclass(frozen=True)
class PromotionGateResult:
    """Result of promotion-readiness evaluation for one history entry."""
    readiness:        PromotionReadiness
    gate_results:     dict           # gate_name -> bool
    blocking_gates:   tuple          # names of failed gates
    blocking_reasons: tuple          # human-readable reasons

    def to_dict(self) -> dict:
        return {
            "readiness":        self.readiness.value,
            "gate_results":     dict(self.gate_results),
            "blocking_gates":   list(self.blocking_gates),
            "blocking_reasons": list(self.blocking_reasons),
        }


def update_adaptation_history(
    history: dict,
    zone_id: str,
    signal: "OutcomeSignal",
    situation: "SituationContext",
    created_at: str,
    *,
    now_ts: Optional[str] = None,
    max_entries: int = 50,
) -> dict:
    """Pure function: suggest candidates, accumulate SHADOW ones, prune, return new dict.

    REJECTED candidates are not accumulated.
    Returns a new dict — never mutates the input.
    """
    from .candidates import suggest_candidates
    from .contracts import AdaptationLifecycle
    from .history_store_schema import prune_history_entries

    traces = suggest_candidates(zone_id, signal, situation, created_at)
    new_history = dict(history)
    for trace in traces:
        if trace.lifecycle is AdaptationLifecycle.SHADOW:
            identity = candidate_identity_from_trace(zone_id, trace)
            key = make_candidate_key(identity)
            new_history[key] = accumulate_into_history(new_history.get(key), trace, key)
    return prune_history_entries(
        new_history, now_ts=now_ts or created_at, max_entries=max_entries,
    )


def evaluate_promotion_readiness(
    entry: CandidateHistoryEntry,
    *,
    span_days: float,
    confounder_ratio: float,
) -> PromotionGateResult:
    """Evaluate whether a history entry is ready for promotion to ELIGIBLE.

    Pure function — no control modification, no side effects.

    Args:
        entry: accumulated candidate history entry.
        span_days: calendar days between first_seen_ts and last_seen_ts.
            Caller must compute from entry.first_seen_ts / entry.last_seen_ts.
        confounder_ratio: fraction of all outcome attempts (accepted + rejected)
            that were rejected due to confounders. Caller computes from the
            latest OutcomeDiagnostics.rejection_counts.

    Returns:
        PromotionGateResult with per-gate pass/fail and blocking reasons.
    """
    gates: dict = {}
    reasons: list = []

    gates["min_seen_count"] = entry.seen_count >= _MIN_SEEN_COUNT
    if not gates["min_seen_count"]:
        reasons.append(
            f"insufficient_seen_count:{entry.seen_count}<{_MIN_SEEN_COUNT}"
        )

    gates["min_outcome_samples"] = (
        entry.supporting_outcome_count >= _MIN_OUTCOME_SAMPLES
    )
    if not gates["min_outcome_samples"]:
        reasons.append(
            f"insufficient_outcome_samples:"
            f"{entry.supporting_outcome_count}<{_MIN_OUTCOME_SAMPLES}"
        )

    gates["min_data_quality"] = entry.avg_outcome_quality >= _MIN_DATA_QUALITY
    if not gates["min_data_quality"]:
        reasons.append(
            f"low_data_quality:{entry.avg_outcome_quality:.2f}<{_MIN_DATA_QUALITY}"
        )

    gates["max_confounder_ratio"] = confounder_ratio <= _MAX_CONFOUNDER_RATIO
    if not gates["max_confounder_ratio"]:
        reasons.append(
            f"high_confounder_ratio:{confounder_ratio:.2f}>{_MAX_CONFOUNDER_RATIO}"
        )

    gates["min_span_days"] = span_days >= _MIN_SPAN_DAYS
    if not gates["min_span_days"]:
        reasons.append(
            f"insufficient_span_days:{span_days:.1f}<{_MIN_SPAN_DAYS}"
        )

    gates["stable_reason"] = (
        entry.dominant_reason_ratio >= _MIN_STABLE_REASON_RATIO
    )
    if not gates["stable_reason"]:
        reasons.append(
            f"unstable_reason:{entry.dominant_reason_ratio:.2f}<{_MIN_STABLE_REASON_RATIO}"
        )

    gates["last_lifecycle_shadow"] = (
        entry.last_lifecycle is AdaptationLifecycle.SHADOW
    )
    if not gates["last_lifecycle_shadow"]:
        reasons.append(f"lifecycle_not_shadow:{entry.last_lifecycle.value}")

    passed   = all(gates.values())
    blocking = tuple(k for k, v in gates.items() if not v)
    readiness = PromotionReadiness.ELIGIBLE if passed else PromotionReadiness.BLOCKED

    return PromotionGateResult(
        readiness=readiness,
        gate_results=gates,
        blocking_gates=blocking,
        blocking_reasons=tuple(reasons),
    )
