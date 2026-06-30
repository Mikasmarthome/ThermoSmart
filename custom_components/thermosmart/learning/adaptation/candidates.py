"""Passive outcome-to-candidate evaluation (shadow trace generation).

Converts outcome quality signals and situation context into AdaptationTrace
records (SHADOW or REJECTED lifecycle). No control modifications; no HA
imports; no runtime state. All functions are pure and deterministic.

Only PREHEAT_DELTA_MIN and EARLY_CUTOFF_DELTA_MIN are generated here.
BOOST_BIAS and TPI_GAIN_BIAS require explicit per-episode boost context that
is not yet consistently captured in OutcomeUpdateContext — reserved for a
later increment. COMFORT_BIAS_C requires a supervisor step before it can
be generated safely.
"""
from __future__ import annotations
import hashlib
from typing import Optional

from .contracts import (
    AdaptationDirection,
    AdaptationLifecycle,
    AdaptationTrace,
    CandidateType,
    OutcomeSignal,
    SituationContext,
    make_adaptation_id,
)
from .gates import evaluate_gates

_TIMEOUT_RATE_MODERATE = 0.25   # above → preheat candidate
_TIMEOUT_RATE_SEVERE   = 0.45   # above → strong preheat candidate (+10 min)
_OVERSHOOT_RATE_MODERATE = 0.25  # above → early-cutoff candidate
_OVERSHOOT_RATE_SEVERE   = 0.40  # above → strong early-cutoff candidate (+10 min)


def _anon_zone(zone_id: str) -> str:
    """Anonymized zone hash for export (12-char hex, non-reversible)."""
    return hashlib.sha256(zone_id.encode()).hexdigest()[:12]


def suggest_candidates(
    zone_id: str,
    signal: OutcomeSignal,
    situation: SituationContext,
    created_at: str,
) -> list[AdaptationTrace]:
    """Generate shadow adaptation candidates from outcome quality signals.

    Returns zero or more AdaptationTrace records at SHADOW or REJECTED
    lifecycle state. No actual control state is modified.

    A candidate is produced only when:
    - There is a clear directional signal in the outcome distribution.
    - The gate evaluation runs (it may pass or fail — both produce a trace
      so the full picture is preserved for research export).

    Confounded or low-quality outcomes receive REJECTED lifecycle via the gate
    result; they do not produce SHADOW candidates.
    """
    candidates: list[AdaptationTrace] = []
    zone_hash = _anon_zone(zone_id)

    # ── Candidate 1: increase preheat lead time (many timeouts) ───────────────
    timeout_rate = signal.timeout_rate or 0.0
    if timeout_rate >= _TIMEOUT_RATE_MODERATE:
        gate = evaluate_gates(signal, candidate_mitigates_instability=True)
        value = 10.0 if timeout_rate >= _TIMEOUT_RATE_SEVERE else 5.0
        reasons = (
            f"high_timeout_rate:{timeout_rate:.2f}",
            "shadow_candidate_only",
        )
        candidates.append(AdaptationTrace(
            adaptation_id=make_adaptation_id(
                zone_id, CandidateType.PREHEAT_DELTA_MIN, created_at,
            ),
            zone_hash=zone_hash,
            candidate_type=CandidateType.PREHEAT_DELTA_MIN,
            direction=AdaptationDirection.INCREASE,
            candidate_value=value,
            lifecycle=(
                AdaptationLifecycle.SHADOW
                if gate.passed else AdaptationLifecycle.REJECTED
            ),
            situation_context=situation,
            outcome_signal=signal,
            gate_result=gate,
            reason_codes=reasons,
            created_at=created_at,
        ))

    # ── Candidate 2: increase early-cutoff lead (many overshoots) ─────────────
    overshoot_rate = signal.overshoot_rate or 0.0
    if overshoot_rate >= _OVERSHOOT_RATE_MODERATE:
        gate = evaluate_gates(signal, candidate_mitigates_instability=True)
        value = 10.0 if overshoot_rate >= _OVERSHOOT_RATE_SEVERE else 5.0
        reasons = (
            f"high_overshoot_rate:{overshoot_rate:.2f}",
            "shadow_candidate_only",
        )
        candidates.append(AdaptationTrace(
            adaptation_id=make_adaptation_id(
                zone_id, CandidateType.EARLY_CUTOFF_DELTA_MIN, created_at,
            ),
            zone_hash=zone_hash,
            candidate_type=CandidateType.EARLY_CUTOFF_DELTA_MIN,
            direction=AdaptationDirection.INCREASE,
            candidate_value=value,
            lifecycle=(
                AdaptationLifecycle.SHADOW
                if gate.passed else AdaptationLifecycle.REJECTED
            ),
            situation_context=situation,
            outcome_signal=signal,
            gate_result=gate,
            reason_codes=reasons,
            created_at=created_at,
        ))

    return candidates
