"""B2b-7/8: the single authoritative boost ACTIVATION-READINESS contract (pure Python).

This is the one place the future control layer asks: *may the currently selected boost
factor, in this context, be used for real control at all?* It answers ONLY the LE2
learning eligibility — it is NOT user consent and NOT active control:

    Learning Readiness  ≠  Adaptive-Boost switch  ≠  Active Control

No activation, no HA switch, no setpoint change happens here. The contract is fully
reconstructable from the persisted authoritative model state; unknown/legacy/corrupt inputs
resolve to ``eligibility=False`` (never an implicit promotion to eligible).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .boost_degradation import BoostLearningReadiness

ACTIVATION_CONTRACT_VERSION = 1

# ── thresholds (single authority) ───────────────────────────────────────────
# Activation is STRICTER than learning ELIGIBLE: a scope may reach learning ELIGIBLE at the
# re-eligibility confidence (0.45) yet still not be control-usable. Confidence must reach the
# existing control minimum; reliability must cover at least half of full_confidence_samples.
MIN_ACTIVATION_CONFIDENCE = 0.55          # == ControlPolicy/BoostParameters.min_control_confidence
MIN_ACTIVATION_RELIABILITY = 0.5          # >= 50% of full_confidence_samples of authoritative evidence


class BoostEligibilityReason:
    """Single typed source of eligibility reasons (no free-string creation across layers)."""
    ELIGIBLE = "eligible"
    INSUFFICIENT_DATA = "insufficient_data"
    SHADOW_LEARNING = "shadow_learning"
    LOW_CONFIDENCE = "low_confidence"
    LOW_RELIABILITY = "low_reliability"
    UNSTABLE_FACTOR = "unstable_factor"
    BUCKET_DEGRADED = "bucket_degraded"
    GENERAL_DEGRADED = "general_degraded"
    ROLLED_BACK = "rolled_back"
    COOLDOWN = "cooldown"
    REHABILITATION_REQUIRED = "rehabilitation_required"
    NO_USABLE_FACTOR = "no_usable_factor"
    NO_ELIGIBLE_SCOPE = "no_eligible_scope"
    BLOCKING_CONFOUNDER = "blocking_confounder"
    UNSUPPORTED_CONTROL_TYPE = "unsupported_control_type"
    MISSING_CONTEXT = "missing_context"
    UNKNOWN = "unknown"

    ALL = (ELIGIBLE, INSUFFICIENT_DATA, SHADOW_LEARNING, LOW_CONFIDENCE, LOW_RELIABILITY,
           UNSTABLE_FACTOR, BUCKET_DEGRADED, GENERAL_DEGRADED, ROLLED_BACK, COOLDOWN,
           REHABILITATION_REQUIRED, NO_USABLE_FACTOR, NO_ELIGIBLE_SCOPE, BLOCKING_CONFOUNDER,
           UNSUPPORTED_CONTROL_TYPE, MISSING_CONTEXT, UNKNOWN)


@dataclass(frozen=True)
class BoostActivationReadiness:
    """The verbatim contract the control layer consumes. ``eligibility`` is an LE2 statement,
    NOT a device release."""
    learning_readiness: str
    factor_usable: bool
    selected_scope: str                       # bucket | general | general_fallback | prior
    selected_bucket_handle: Optional[str]     # hashed; never an entity id
    bucket_readiness: Optional[str]
    general_readiness: str
    fallback_used: bool
    confidence: float
    reliability: float
    factor_stable: bool
    known_good_available: bool
    rollback_active: bool
    cooldown_active: bool
    cooldown_remaining_s: Optional[float]
    rehabilitation_progress: float
    current_factor_c: float
    control_type: str
    eligibility: bool                         # the hard LE2 control-eligibility verdict
    eligibility_reason: str
    blocking_reasons: tuple = ()
    degrading_reasons: tuple = ()
    min_activation_confidence: float = MIN_ACTIVATION_CONFIDENCE
    min_activation_reliability: float = MIN_ACTIVATION_RELIABILITY
    component_version: int = ACTIVATION_CONTRACT_VERSION

    def to_dict(self) -> dict:
        return {
            "learning_readiness": self.learning_readiness, "factor_usable": self.factor_usable,
            "selected_scope": self.selected_scope,
            "selected_bucket_handle": self.selected_bucket_handle,
            "bucket_readiness": self.bucket_readiness, "general_readiness": self.general_readiness,
            "fallback_used": self.fallback_used, "confidence": round(self.confidence, 4),
            "reliability": round(self.reliability, 4), "factor_stable": self.factor_stable,
            "known_good_available": self.known_good_available,
            "rollback_active": self.rollback_active, "cooldown_active": self.cooldown_active,
            "cooldown_remaining_s": self.cooldown_remaining_s,
            "rehabilitation_progress": round(self.rehabilitation_progress, 4),
            "current_factor_c": round(self.current_factor_c, 4), "control_type": self.control_type,
            "eligibility": self.eligibility, "eligibility_reason": self.eligibility_reason,
            "blocking_reasons": list(self.blocking_reasons),
            "degrading_reasons": list(self.degrading_reasons),
            "min_activation_confidence": self.min_activation_confidence,
            "min_activation_reliability": self.min_activation_reliability,
            "component_version": self.component_version,
        }


def _not_eligible(reason: str, **kw) -> BoostActivationReadiness:
    base = dict(
        learning_readiness=kw.get("learning_readiness", BoostLearningReadiness.INSUFFICIENT_DATA),
        factor_usable=False, selected_scope=kw.get("selected_scope", "prior"),
        selected_bucket_handle=kw.get("selected_bucket_handle"),
        bucket_readiness=kw.get("bucket_readiness"),
        general_readiness=kw.get("general_readiness", BoostLearningReadiness.INSUFFICIENT_DATA),
        fallback_used=kw.get("fallback_used", False), confidence=kw.get("confidence", 0.0),
        reliability=kw.get("reliability", 0.0), factor_stable=kw.get("factor_stable", False),
        known_good_available=kw.get("known_good_available", False),
        rollback_active=kw.get("rollback_active", False),
        cooldown_active=kw.get("cooldown_active", False),
        cooldown_remaining_s=kw.get("cooldown_remaining_s"),
        rehabilitation_progress=kw.get("rehabilitation_progress", 0.0),
        current_factor_c=kw.get("current_factor_c", 0.0),
        control_type=kw.get("control_type", "unknown"), eligibility=False,
        eligibility_reason=reason, blocking_reasons=tuple(kw.get("blocking_reasons", ())),
        degrading_reasons=tuple(kw.get("degrading_reasons", ())))
    return BoostActivationReadiness(**base)


def build_activation_readiness(*, recommendation, factor_stable: bool, control_type: Optional[str],
                               has_context: bool, blocking_reasons=(), degrading_reasons=(),
                               general_readiness: str,
                               min_confidence: float = MIN_ACTIVATION_CONFIDENCE,
                               min_reliability: float = MIN_ACTIVATION_RELIABILITY
                               ) -> BoostActivationReadiness:
    """Pure: fold the (already scope-resolved) recommendation + control-type + runtime gates
    into the hard, typed activation-readiness verdict.

    Hard invariant (no implicit release via a positive factor or via confidence alone):
        learning_readiness == ELIGIBLE AND factor_usable AND not rollback AND not cooldown
        AND factor_stable AND confidence >= min AND reliability >= min AND no blocking reason
        -> eligibility = True ; else False, with a single typed reason (first failing gate).
    """
    rec = recommendation
    common = dict(
        learning_readiness=rec.readiness, selected_scope=rec.selected_scope,
        selected_bucket_handle=rec.bucket_handle, bucket_readiness=getattr(rec, "_bucket_readiness", None),
        general_readiness=general_readiness, fallback_used=rec.fallback_used,
        confidence=rec.confidence, reliability=rec.reliability, factor_stable=factor_stable,
        known_good_available=rec.known_good_available, rollback_active=rec.rollback_active,
        cooldown_active=rec.cooldown_active, cooldown_remaining_s=rec.cooldown_remaining_s,
        rehabilitation_progress=rec.rehab_progress, current_factor_c=rec.boost_offset_c,
        control_type=(control_type or "unknown"),
        blocking_reasons=tuple(blocking_reasons), degrading_reasons=tuple(degrading_reasons))

    R = BoostEligibilityReason
    # priority-ordered gates -> first failing reason
    if control_type not in ("setpoint",):
        reason = (R.UNSUPPORTED_CONTROL_TYPE if control_type in ("direct_valve",)
                  else R.MISSING_CONTEXT)
        return _not_eligible(reason, **common)
    if not has_context:
        return _not_eligible(R.MISSING_CONTEXT, **common)
    if blocking_reasons:
        return _not_eligible(R.BLOCKING_CONFOUNDER, **common)
    if rec.readiness == BoostLearningReadiness.INSUFFICIENT_DATA:
        return _not_eligible(R.INSUFFICIENT_DATA, **common)
    if rec.rollback_active and rec.cooldown_active:
        return _not_eligible(R.ROLLED_BACK, **common)
    if rec.cooldown_active:
        return _not_eligible(R.COOLDOWN, **common)
    if rec.readiness == BoostLearningReadiness.DEGRADED:
        scope = rec.selected_scope
        return _not_eligible(R.GENERAL_DEGRADED if scope in ("general", "general_fallback")
                             else R.BUCKET_DEGRADED, **common)
    if rec.readiness == BoostLearningReadiness.SHADOW_LEARNING:
        # shadow after a rollback means rehabilitation is still in progress
        reason = (R.REHABILITATION_REQUIRED if rec.known_good_available and rec.rehab_progress < 1.0
                  else R.SHADOW_LEARNING)
        return _not_eligible(reason, **common)
    if rec.readiness == BoostLearningReadiness.BOOTSTRAP:
        # B2b-bootstrap: initial controlled trial using device prior offset.
        # No evidence yet — stability/confidence/reliability gates do not apply.
        # factor_usable must still be True (set by predict_boost_factor bootstrap path).
        if not rec.factor_usable:
            return _not_eligible(R.NO_USABLE_FACTOR, **common)
        return BoostActivationReadiness(
            eligibility=True, eligibility_reason=R.ELIGIBLE, factor_usable=True, **common,
            min_activation_confidence=min_confidence, min_activation_reliability=min_reliability)
    if not rec.factor_usable:
        reason = (R.NO_ELIGIBLE_SCOPE if rec.selected_scope in ("prior", "unavailable")
                  else R.NO_USABLE_FACTOR)
        return _not_eligible(reason, **common)
    if not factor_stable:
        return _not_eligible(R.UNSTABLE_FACTOR, **common)
    if rec.confidence < min_confidence:
        return _not_eligible(R.LOW_CONFIDENCE, **common)
    if rec.reliability < min_reliability:
        return _not_eligible(R.LOW_RELIABILITY, **common)
    # all gates satisfied
    return BoostActivationReadiness(
        eligibility=True, eligibility_reason=R.ELIGIBLE, factor_usable=True, **common,
        min_activation_confidence=min_confidence, min_activation_reliability=min_reliability)
