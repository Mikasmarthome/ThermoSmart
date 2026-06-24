"""B2b-5: boost-learning readiness, degradation detection, bounded rollback + cooldown.

Pure Python, no Home Assistant. This is the *learning-readiness* authority for the boost
factor — it is NOT a boost release lifecycle and performs NO activation. It watches the
finalized, classified, reliability-weighted boost outcomes and, on repeated *reliable*
degradation, rolls the learned boost factor back to a proven ``last_known_good`` snapshot
(or a safe zero-adaptive prior) and opens a cooldown that requires fresh clean shadow
evidence before adaptive maturity may return.

Design (all thresholds are central, justified params in ``DegradationParams``):

* Evidence is kept as a few BOUNDED decayed scalars (no outcome list). On each observe the
  scalars decay by an exponential recency factor since the last evidence, then the new
  weighted evidence is added:  ``severity × outcome_reliability × attribution_reliability``.
* Safety (overshoot) evidence decays SLOWER than efficiency (no-effect/unnecessary) evidence
  and is never reduced by positive outcomes, so a SUCCESS cannot immediately neutralise a
  preceding strong overshoot and the factor cannot drift up through alternating
  SUCCESS/OVERSHOOT.
* Asymmetry: positive evidence reduces the degradation score slowly (``positive_scale``),
  reliable negative evidence faster, repeated safety degradation triggers rollback+cooldown.
* Non-degradation classes (confounded/interrupted/partial/failed/not_attempted/
  insufficient/legacy) contribute ZERO — they never create false rollback evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Optional

DEGRADATION_SCHEMA_VERSION = 1


class BoostLearningReadiness:
    """Learning-evidence readiness (distinct from any boost release lifecycle)."""
    INSUFFICIENT_DATA = "insufficient_data"
    SHADOW_LEARNING = "shadow_learning"
    ELIGIBLE = "eligible"
    DEGRADED = "degraded"
    ROLLED_BACK = "rolled_back"
    COOLDOWN = "cooldown"

    ALL = (INSUFFICIENT_DATA, SHADOW_LEARNING, ELIGIBLE, DEGRADED, ROLLED_BACK, COOLDOWN)


@dataclass(frozen=True)
class DegradationParams:
    # evidence window (decay implicitly bounds count/age; explicit caps documented in report)
    count_cap: int = 20
    age_cap_s: float = 2592000.0              # 30 d (beyond -> evidence fully decayed/dropped)
    recency_half_life_s: float = 604800.0     # 7 d efficiency-evidence half life
    safety_half_life_s: float = 1209600.0     # 14 d safety-evidence half life (slower)
    # per-class severity (safety heavier than efficiency)
    overshoot_severity: float = 1.0
    no_effect_severity: float = 0.4
    unnecessary_severity: float = 0.5
    success_gain: float = 1.0
    positive_scale: float = 0.5               # asymmetry: positive evidence counts half
    extreme_overshoot_c: float = 1.5          # single stable overshoot -> fast safety trigger
    high_overshoot_reliability: float = 0.6
    # readiness / rollback thresholds
    eligible_min_outcomes: int = 5
    eligible_min_reliability: float = 0.45    # above cold-start cap (0.35), below control-min
    degraded_score: float = 1.5
    rollback_score: float = 3.0
    rollback_min_negative: int = 3
    consecutive_degraded: int = 3
    fast_overshoot_count: int = 2
    fast_overshoot_window_s: float = 86400.0  # 1 d
    # known good
    kg_min_outcomes: int = 5
    kg_min_reliability: float = 0.45          # stable shadow confidence (see eligible_min)
    kg_safety_free_score: float = 0.5         # overshoot_evidence must be below this
    # cooldown (bounded)
    cooldown_base_s: float = 21600.0          # 6 h
    cooldown_min_s: float = 21600.0           # 6 h
    cooldown_max_s: float = 604800.0          # 7 d
    # rehabilitation (cooldown elapse alone is NOT enough)
    rehab_required_clean: int = 3
    rehab_required_weight: float = 2.0
    # ── B2b-5a: re-eligibility hysteresis + factor stability ──────────────
    recovery_score: float = 0.75              # DEGRADED clears only below this (< degraded 1.5)
    reeligible_min_confidence: float = 0.45
    factor_history_cap: int = 6
    factor_stability_max_change: float = 0.5  # max factor spread in the stability window
    # ── B2b-5a: bucket scoping + general escalation ───────────────────────
    bucket_state_cap: int = 12                # max active bucket degradation states / zone
    bucket_age_cap_s: float = 2592000.0       # 30 d -> prune stale bucket trackers
    escalate_distinct_buckets: int = 2        # N distinct degraded buckets -> general
    escalate_local_rollbacks: int = 2         # local rollbacks in window -> general
    escalate_window_s: float = 86400.0        # 1 d escalation window
    component_version: int = DEGRADATION_SCHEMA_VERSION


_DEGRADATION_CLASSES = {"boost_overshoot", "boost_no_effect", "boost_unnecessary"}
_POSITIVE_CLASSES = {"boost_success"}


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _decay(value: float, dt_s: float, half_life_s: float) -> float:
    if value == 0.0 or dt_s <= 0.0 or half_life_s <= 0.0:
        return value
    return value * math.pow(0.5, dt_s / half_life_s)


@dataclass(frozen=True)
class KnownGoodState:
    """Proven-safe boost-factor snapshot used as the bounded rollback target."""
    factor: float
    effective_n: float
    dispersion: float
    confidence: float
    sample_weight: float
    ts: str
    model_version: int
    component_version: int = DEGRADATION_SCHEMA_VERSION
    provenance: Mapping[str, Any] = field(default_factory=dict)   # privacy-safe aggregates only

    def to_dict(self) -> dict:
        return {"factor": self.factor, "effective_n": self.effective_n,
                "dispersion": self.dispersion, "confidence": self.confidence,
                "sample_weight": self.sample_weight, "ts": self.ts,
                "model_version": self.model_version,
                "component_version": self.component_version,
                "provenance": dict(self.provenance)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Optional["KnownGoodState"]:
        if d is None:
            return None
        if int(d.get("component_version", 0)) != DEGRADATION_SCHEMA_VERSION:
            return None
        try:
            return cls(factor=float(d["factor"]), effective_n=float(d["effective_n"]),
                       dispersion=float(d["dispersion"]), confidence=float(d["confidence"]),
                       sample_weight=float(d["sample_weight"]), ts=d["ts"],
                       model_version=int(d["model_version"]),
                       provenance=dict(d.get("provenance", {})))
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class BoostDegradationState:
    readiness: str = BoostLearningReadiness.INSUFFICIENT_DATA
    degradation_score: float = 0.0
    overshoot_evidence: float = 0.0
    no_effect_evidence: float = 0.0
    unnecessary_evidence: float = 0.0
    weighted_positive: float = 0.0
    weighted_negative: float = 0.0
    consecutive_negative_count: int = 0
    adaptive_outcomes: int = 0
    recent_overshoot_count: int = 0           # within fast_overshoot_window
    recent_overshoot_ts: tuple = ()           # bounded list of recent overshoot timestamps
    last_evidence_ts: Optional[str] = None
    last_negative_ts: Optional[str] = None
    degradation_started_at: Optional[str] = None
    known_good: Optional[KnownGoodState] = None
    rollback_count: int = 0
    last_rollback_ts: Optional[str] = None
    cooldown_start_ts: Optional[str] = None
    cooldown_until_ts: Optional[str] = None
    cooldown_reason: str = ""
    rollback_scope: str = ""
    rehab_clean_count: int = 0
    rehab_weight: float = 0.0
    factor_history: tuple = ()                 # B2b-5a bounded ring of recent adaptive factors
    reason: str = ""
    component_version: int = DEGRADATION_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "readiness": self.readiness, "degradation_score": self.degradation_score,
            "overshoot_evidence": self.overshoot_evidence,
            "no_effect_evidence": self.no_effect_evidence,
            "unnecessary_evidence": self.unnecessary_evidence,
            "weighted_positive": self.weighted_positive,
            "weighted_negative": self.weighted_negative,
            "consecutive_negative_count": self.consecutive_negative_count,
            "adaptive_outcomes": self.adaptive_outcomes,
            "recent_overshoot_count": self.recent_overshoot_count,
            "recent_overshoot_ts": list(self.recent_overshoot_ts),
            "last_evidence_ts": self.last_evidence_ts, "last_negative_ts": self.last_negative_ts,
            "degradation_started_at": self.degradation_started_at,
            "known_good": self.known_good.to_dict() if self.known_good else None,
            "rollback_count": self.rollback_count, "last_rollback_ts": self.last_rollback_ts,
            "cooldown_start_ts": self.cooldown_start_ts,
            "cooldown_until_ts": self.cooldown_until_ts, "cooldown_reason": self.cooldown_reason,
            "rollback_scope": self.rollback_scope, "rehab_clean_count": self.rehab_clean_count,
            "rehab_weight": self.rehab_weight, "factor_history": list(self.factor_history),
            "reason": self.reason, "component_version": self.component_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BoostDegradationState":
        if not d or int(d.get("component_version", 0)) != DEGRADATION_SCHEMA_VERSION:
            return cls()                          # unknown version -> safe fresh (degrade)
        try:
            return cls(
                readiness=d.get("readiness", BoostLearningReadiness.INSUFFICIENT_DATA),
                degradation_score=float(d.get("degradation_score", 0.0)),
                overshoot_evidence=float(d.get("overshoot_evidence", 0.0)),
                no_effect_evidence=float(d.get("no_effect_evidence", 0.0)),
                unnecessary_evidence=float(d.get("unnecessary_evidence", 0.0)),
                weighted_positive=float(d.get("weighted_positive", 0.0)),
                weighted_negative=float(d.get("weighted_negative", 0.0)),
                consecutive_negative_count=int(d.get("consecutive_negative_count", 0)),
                adaptive_outcomes=int(d.get("adaptive_outcomes", 0)),
                recent_overshoot_count=int(d.get("recent_overshoot_count", 0)),
                recent_overshoot_ts=tuple(d.get("recent_overshoot_ts", [])),
                last_evidence_ts=d.get("last_evidence_ts"),
                last_negative_ts=d.get("last_negative_ts"),
                degradation_started_at=d.get("degradation_started_at"),
                known_good=KnownGoodState.from_dict(d.get("known_good")),
                rollback_count=int(d.get("rollback_count", 0)),
                last_rollback_ts=d.get("last_rollback_ts"),
                cooldown_start_ts=d.get("cooldown_start_ts"),
                cooldown_until_ts=d.get("cooldown_until_ts"),
                cooldown_reason=d.get("cooldown_reason", ""),
                rollback_scope=d.get("rollback_scope", ""),
                rehab_clean_count=int(d.get("rehab_clean_count", 0)),
                rehab_weight=float(d.get("rehab_weight", 0.0)),
                factor_history=tuple(d.get("factor_history", [])),
                reason=d.get("reason", ""))
        except (KeyError, TypeError, ValueError):
            return cls()


@dataclass(frozen=True)
class RollbackAction:
    """A bounded rollback decision: roll the factor to known-good (or zero) + open cooldown."""
    triggered: bool
    scope: str
    reason: str
    target_factor: Optional[float]            # None -> zero-adaptive safe prior
    target_known_good: Optional[KnownGoodState]
    cooldown_until_ts: Optional[str]
    fast_safety: bool = False


def classify_evidence(outcome_class: str, overshoot_c: Optional[float], reliability: float,
                      attribution: float, *, params: DegradationParams):
    """Return (kind, weight, is_extreme). kind in {overshoot,no_effect,unnecessary,success,
    ignore}. Non-degradation classes -> ('ignore', 0, False)."""
    rel = max(0.0, min(1.0, reliability)) * max(0.0, min(1.0, attribution))
    if outcome_class == "boost_overshoot":
        extreme = (overshoot_c is not None and overshoot_c >= params.extreme_overshoot_c
                   and reliability >= params.high_overshoot_reliability)
        return "overshoot", params.overshoot_severity * rel, extreme
    if outcome_class == "boost_no_effect":
        return "no_effect", params.no_effect_severity * rel, False
    if outcome_class == "boost_unnecessary":
        return "unnecessary", params.unnecessary_severity * rel, False
    if outcome_class == "boost_success":
        return "success", params.success_gain * rel, False
    return "ignore", 0.0, False


def _in_cooldown(state: BoostDegradationState, now: datetime) -> bool:
    until = _parse(state.cooldown_until_ts)
    return until is not None and now < until


def _cooldown_duration_s(params: DegradationParams, rollback_count: int,
                         fast_safety: bool, overshoot_evidence: float) -> float:
    # bounded: base × (1 + rollback_count) × severity, clamped [min, max]
    severity = 1.0 + (1.0 if fast_safety else 0.0) + min(2.0, overshoot_evidence)
    raw = params.cooldown_base_s * (1 + max(0, rollback_count)) * severity
    return max(params.cooldown_min_s, min(params.cooldown_max_s, raw))


def observe(state: BoostDegradationState, *, outcome_class: str, overshoot_c: Optional[float],
            reliability: float, attribution: float, adaptive: bool, reliability_known: bool,
            now_ts: str, current_factor: float, current_dim: Mapping[str, float],
            confidence: float, params: Optional[DegradationParams] = None,
            ) -> tuple[BoostDegradationState, RollbackAction]:
    """Fold one finalized boost outcome into the degradation tracker.

    Returns the new state and a RollbackAction (``triggered=False`` when none). The caller
    applies the rollback to the model factor atomically and persists the whole zone state.
    """
    p = params or DegradationParams()
    no_action = RollbackAction(False, "", "", None, None, state.cooldown_until_ts)
    now = _parse(now_ts)
    if now is None:
        return state, no_action

    # decay existing bounded scalars to 'now' (separate half-lives for safety vs efficiency)
    prev = _parse(state.last_evidence_ts)
    dt = (now - prev).total_seconds() if prev is not None else 0.0
    deg = _decay(state.degradation_score, dt, p.recency_half_life_s)
    ovr = _decay(state.overshoot_evidence, dt, p.safety_half_life_s)
    noe = _decay(state.no_effect_evidence, dt, p.recency_half_life_s)
    unn = _decay(state.unnecessary_evidence, dt, p.recency_half_life_s)
    pos = _decay(state.weighted_positive, dt, p.recency_half_life_s)
    neg = _decay(state.weighted_negative, dt, p.recency_half_life_s)

    # only adaptive, reliability-known outcomes carry rollback evidence
    kind, weight, extreme = ("ignore", 0.0, False)
    if adaptive and reliability_known:
        kind, weight, extreme = classify_evidence(
            outcome_class, overshoot_c, reliability, attribution, params=p)

    consec = state.consecutive_negative_count
    adaptive_outcomes = state.adaptive_outcomes + (1 if (adaptive and reliability_known) else 0)
    last_neg = state.last_negative_ts
    deg_started = state.degradation_started_at
    rehab_clean = state.rehab_clean_count
    rehab_weight = state.rehab_weight
    recent_ovr = [t for t in state.recent_overshoot_ts
                  if (now - _parse(t)).total_seconds() <= p.fast_overshoot_window_s
                  if _parse(t) is not None]

    if kind == "overshoot":
        ovr += weight
        deg += weight
        neg += weight
        consec += 1
        last_neg = now_ts
        deg_started = deg_started or now_ts
        # only a sufficiently RELIABLE overshoot counts toward the fast safety trigger; a
        # weak-reliability overshoot still adds (small) weight but cannot force a fast rollback.
        if reliability >= p.high_overshoot_reliability:
            recent_ovr.append(now_ts)
        rehab_clean = 0                          # safety event breaks rehabilitation streak
    elif kind in ("no_effect", "unnecessary"):
        if kind == "no_effect":
            noe += weight
        else:
            unn += weight
        deg += weight
        neg += weight
        consec += 1
        last_neg = now_ts
        deg_started = deg_started or now_ts
        rehab_clean = 0
    elif kind == "success":
        # asymmetric, bounded positive relief; never reduces SAFETY (overshoot) evidence
        deg = max(0.0, deg - weight * p.positive_scale)
        pos += weight
        consec = 0
        rehab_clean += 1
        rehab_weight += weight
    # 'ignore' -> no evidence change, does not break consecutive streak counters either

    recent_ovr = recent_ovr[-p.count_cap:]
    new = replace(
        state, degradation_score=deg, overshoot_evidence=ovr, no_effect_evidence=noe,
        unnecessary_evidence=unn, weighted_positive=pos, weighted_negative=neg,
        consecutive_negative_count=consec, adaptive_outcomes=adaptive_outcomes,
        recent_overshoot_count=len(recent_ovr), recent_overshoot_ts=tuple(recent_ovr),
        last_evidence_ts=now_ts, last_negative_ts=last_neg,
        degradation_started_at=deg_started, rehab_clean_count=rehab_clean,
        rehab_weight=rehab_weight)

    # ── rollback triggers ────────────────────────────────────────────────
    fast = (extreme or len(recent_ovr) >= p.fast_overshoot_count)
    normal = (deg >= p.rollback_score and neg >= 0.0
              and (consec >= p.rollback_min_negative or new.adaptive_outcomes >= p.rollback_min_negative)
              and (deg > _decay(state.degradation_score, dt, p.recency_half_life_s) or kind != "success"))
    # do not re-trigger while already cooling down
    if _in_cooldown(state, now):
        # repeated degradation during cooldown extends/raises it (bounded)
        if kind == "overshoot" or fast:
            dur = _cooldown_duration_s(p, state.rollback_count, fast, ovr)
            extended = (now + _td(dur)).isoformat()
            new = replace(new, readiness=BoostLearningReadiness.COOLDOWN,
                          cooldown_until_ts=_max_ts(state.cooldown_until_ts, extended),
                          cooldown_reason="degradation_during_cooldown",
                          reason="cooldown_extended")
        else:
            new = replace(new, readiness=BoostLearningReadiness.COOLDOWN)
        return new, RollbackAction(False, new.rollback_scope, new.reason, None, None,
                                   new.cooldown_until_ts)

    if fast or normal:
        reason = ("fast_safety_overshoot" if fast else "degradation_threshold")
        dur = _cooldown_duration_s(p, state.rollback_count, fast, ovr)
        cooldown_until = (now + _td(dur)).isoformat()
        kg = new.known_good
        action = RollbackAction(
            triggered=True, scope="general", reason=reason,
            target_factor=(kg.factor if kg else None),
            target_known_good=kg, cooldown_until_ts=cooldown_until, fast_safety=fast)
        rolled = replace(
            new, readiness=BoostLearningReadiness.ROLLED_BACK,
            rollback_count=new.rollback_count + 1, last_rollback_ts=now_ts,
            cooldown_start_ts=now_ts, cooldown_until_ts=cooldown_until,
            cooldown_reason=reason, rollback_scope="general", reason=reason,
            rehab_clean_count=0, rehab_weight=0.0,
            # rollback does NOT forget the tracker; safety evidence is retained
            consecutive_negative_count=new.consecutive_negative_count)
        return rolled, action

    # ── readiness (no rollback) ──────────────────────────────────────────
    new = replace(new, readiness=_readiness_no_rollback(new, p), reason="")
    return new, no_action


def _td(seconds: float):
    from datetime import timedelta
    return timedelta(seconds=seconds)


def _max_ts(a: Optional[str], b: Optional[str]) -> Optional[str]:
    pa, pb = _parse(a), _parse(b)
    if pa is None:
        return b
    if pb is None:
        return a
    return a if pa >= pb else b


def _readiness_no_rollback(s: BoostDegradationState, p: DegradationParams) -> str:
    if s.degradation_score >= p.degraded_score \
            or s.consecutive_negative_count >= p.consecutive_degraded:
        return BoostLearningReadiness.DEGRADED
    if s.adaptive_outcomes < p.eligible_min_outcomes:
        return (BoostLearningReadiness.SHADOW_LEARNING if s.adaptive_outcomes > 0
                else BoostLearningReadiness.INSUFFICIENT_DATA)
    return BoostLearningReadiness.ELIGIBLE


def resolve_cooldown(state: BoostDegradationState, now_ts: str,
                     params: Optional[DegradationParams] = None) -> BoostDegradationState:
    """Re-evaluate readiness w.r.t. cooldown+rehabilitation at prediction/observe time.

    Cooldown time elapsing is NOT sufficient: SHADOW_LEARNING (then ELIGIBLE) only returns
    after enough fresh CLEAN shadow evidence AND no recent safety degradation AND reliability.
    Never goes ROLLED_BACK -> ELIGIBLE directly."""
    p = params or DegradationParams()
    now = _parse(now_ts)
    if now is None:
        return state
    if state.readiness not in (BoostLearningReadiness.ROLLED_BACK,
                               BoostLearningReadiness.COOLDOWN):
        return state
    if _in_cooldown(state, now):
        return replace(state, readiness=BoostLearningReadiness.COOLDOWN)
    # time elapsed -> require rehabilitation evidence
    ovr = _decay(state.overshoot_evidence,
                 (now - _parse(state.last_evidence_ts)).total_seconds()
                 if state.last_evidence_ts else 0.0, p.safety_half_life_s)
    if (state.rehab_clean_count >= p.rehab_required_clean
            and state.rehab_weight >= p.rehab_required_weight
            and ovr < p.kg_safety_free_score):
        return replace(state, readiness=BoostLearningReadiness.SHADOW_LEARNING,
                       reason="rehabilitated_to_shadow")
    return replace(state, readiness=BoostLearningReadiness.COOLDOWN,
                   reason="cooldown_elapsed_awaiting_rehab")


def maybe_capture_known_good(state: BoostDegradationState, *, factor: float,
                             effective_n: float, dispersion: float, confidence: float,
                             sample_weight: float, now_ts: str, model_version: int,
                             provenance: Mapping[str, Any],
                             params: Optional[DegradationParams] = None,
                             ) -> BoostDegradationState:
    """Capture a new last_known_good ONLY from a stable, reliable, safety-free ELIGIBLE state."""
    p = params or DegradationParams()
    if state.readiness != BoostLearningReadiness.ELIGIBLE:
        return state
    if state.adaptive_outcomes < p.kg_min_outcomes:
        return state
    if confidence < p.kg_min_reliability:
        return state
    if state.overshoot_evidence >= p.kg_safety_free_score:
        return state
    if state.degradation_score >= p.degraded_score:
        return state
    kg = KnownGoodState(
        factor=factor, effective_n=effective_n, dispersion=dispersion, confidence=confidence,
        sample_weight=sample_weight, ts=now_ts, model_version=model_version,
        provenance=dict(provenance))
    return replace(state, known_good=kg)


# ════════════════════════════════════════════════════════════════════════════
# B2b-5a: factor-stability gate, hysteresis re-eligibility, and bucket scoping
# ════════════════════════════════════════════════════════════════════════════

def _factor_stable(state: BoostDegradationState, params: DegradationParams) -> bool:
    """Bounded stability metric: the factor spread across the recent ring must be small.
    Too few samples -> not yet stable."""
    h = [float(x) for x in state.factor_history]
    if len(h) < min(3, params.factor_history_cap):
        return False
    return (max(h) - min(h)) <= params.factor_stability_max_change


def _push_factor(state: BoostDegradationState, factor: float,
                 params: DegradationParams) -> BoostDegradationState:
    hist = (tuple(state.factor_history) + (round(float(factor), 4),))[-params.factor_history_cap:]
    return replace(state, factor_history=hist)


def recompute_readiness(state: BoostDegradationState, *, confidence: float,
                        params: DegradationParams) -> BoostDegradationState:
    """Deterministic readiness with HYSTERESIS + factor-stability gate.

    A scope leaves DEGRADED only once the score drops below ``recovery_score`` (strictly
    below ``degraded_score``), preventing flapping. Promotion to ELIGIBLE requires enough
    adaptive outcomes, reliability/confidence, no recent stable overshoot, low score AND a
    stable factor. Rollback/cooldown states are owned by the cooldown resolver."""
    if state.readiness in (BoostLearningReadiness.ROLLED_BACK, BoostLearningReadiness.COOLDOWN):
        return state
    score = state.degradation_score
    was_degraded = state.readiness == BoostLearningReadiness.DEGRADED
    enter_degraded = (score >= params.degraded_score
                      or state.consecutive_negative_count >= params.consecutive_degraded)
    stay_degraded = was_degraded and score >= params.recovery_score   # hysteresis band
    if enter_degraded or stay_degraded:
        return replace(state, readiness=BoostLearningReadiness.DEGRADED)
    if state.adaptive_outcomes < params.eligible_min_outcomes:
        return replace(state, readiness=(BoostLearningReadiness.SHADOW_LEARNING
                                         if state.adaptive_outcomes > 0
                                         else BoostLearningReadiness.INSUFFICIENT_DATA))
    if (confidence >= params.reeligible_min_confidence
            and state.overshoot_evidence < params.kg_safety_free_score
            and _factor_stable(state, params)):
        return replace(state, readiness=BoostLearningReadiness.ELIGIBLE)
    return replace(state, readiness=BoostLearningReadiness.SHADOW_LEARNING)


def is_scope_eligible(state: BoostDegradationState, now_ts: str,
                      params: DegradationParams) -> bool:
    """The hard control-eligibility gate for one scope (bucket or general)."""
    now = _parse(now_ts)
    if now is not None and _in_cooldown(state, now):
        return False
    return (state.readiness == BoostLearningReadiness.ELIGIBLE
            and state.overshoot_evidence < params.kg_safety_free_score
            and state.degradation_score < params.recovery_score)


@dataclass(frozen=True)
class BoostScopedDegradation:
    """Per-context-bucket degradation trackers + a general tracker + escalation history.

    Prediction still uses the model's existing central bucket/general selection; this only
    decides per-scope readiness / rollback / cooldown. Strictly bounded (bucket_state_cap)."""
    general: BoostDegradationState = field(default_factory=BoostDegradationState)
    buckets: Mapping[str, BoostDegradationState] = field(default_factory=dict)
    local_rollback_ts: tuple = ()             # bounded recent local-rollback timestamps
    component_version: int = DEGRADATION_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {"scoped": True, "general": self.general.to_dict(),
                "buckets": {k: v.to_dict() for k, v in self.buckets.items()},
                "local_rollback_ts": list(self.local_rollback_ts),
                "component_version": self.component_version}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BoostScopedDegradation":
        if not d:
            return cls()
        # B2b-5 legacy single-state dict (no scoped marker) -> treat as the general scope
        if not d.get("scoped") and "buckets" not in d and "general" not in d:
            return cls(general=BoostDegradationState.from_dict(d))
        if int(d.get("component_version", 0)) != DEGRADATION_SCHEMA_VERSION:
            return cls()                          # unknown version -> safe fresh
        buckets = {}
        for k, v in (d.get("buckets", {}) or {}).items():
            # corrupt / unknown-version bucket is ISOLATED (dropped), never silently reset
            if not isinstance(v, dict) or int(v.get("component_version", 0)) \
                    != DEGRADATION_SCHEMA_VERSION:
                continue
            buckets[k] = BoostDegradationState.from_dict(v)
        return cls(general=BoostDegradationState.from_dict(d.get("general", {})),
                   buckets=buckets,
                   local_rollback_ts=tuple(d.get("local_rollback_ts", [])))


@dataclass(frozen=True)
class ScopedObserveResult:
    scoped: BoostScopedDegradation
    bucket_key: Optional[str]
    bucket_action: RollbackAction
    general_action: RollbackAction


def _prune_buckets(buckets: dict, now: datetime, params: DegradationParams) -> dict:
    """Bounded retention: drop stale (age_cap) trackers, then LRU by last_evidence_ts, but
    NEVER evict a bucket actively rolled-back / cooling-down / holding a known-good."""
    def _protected(st: BoostDegradationState) -> bool:
        return (st.readiness in (BoostLearningReadiness.ROLLED_BACK,
                                 BoostLearningReadiness.COOLDOWN)
                or st.known_good is not None)
    kept = {}
    for k, st in buckets.items():
        last = _parse(st.last_evidence_ts)
        if last is not None and (now - last).total_seconds() > params.bucket_age_cap_s \
                and not _protected(st):
            continue
        kept[k] = st
    if len(kept) <= params.bucket_state_cap:
        return kept
    evictable = sorted((k for k in kept if not _protected(kept[k])),
                       key=lambda k: kept[k].last_evidence_ts or "")
    for k in evictable[:len(kept) - params.bucket_state_cap]:
        kept.pop(k, None)
    return kept


def observe_scoped(scoped: BoostScopedDegradation, *, bucket_key: Optional[str],
                   outcome_class: str, overshoot_c: Optional[float], reliability: float,
                   attribution: float, adaptive: bool, reliability_known: bool, now_ts: str,
                   bucket_factor: float, bucket_confidence: float, general_factor: float,
                   general_confidence: float,
                   params: Optional[DegradationParams] = None) -> ScopedObserveResult:
    """Route ONE finalized outcome to its canonical bucket scope (local evidence), refine
    readiness with hysteresis+stability, then evaluate GENERAL ESCALATION (NOT a second
    independent accumulation of the same outcome) -> bounded local and/or general rollback."""
    p = params or DegradationParams()
    now = _parse(now_ts)
    no_action = RollbackAction(False, "", "", None, None, None)
    if now is None:
        return ScopedObserveResult(scoped, bucket_key, no_action, no_action)

    buckets = dict(scoped.buckets)
    local_rollbacks = list(scoped.local_rollback_ts)
    bucket_action = no_action

    if bucket_key is not None:
        bstate = buckets.get(bucket_key) or BoostDegradationState()
        bstate, bucket_action = observe(
            bstate, outcome_class=outcome_class, overshoot_c=overshoot_c,
            reliability=reliability, attribution=attribution, adaptive=adaptive,
            reliability_known=reliability_known, now_ts=now_ts, current_factor=bucket_factor,
            current_dim={"value": bucket_factor}, confidence=bucket_confidence, params=p)
        if adaptive and reliability_known and outcome_class in (
                _DEGRADATION_CLASSES | _POSITIVE_CLASSES):
            bstate = _push_factor(bstate, bucket_factor, p)
        if not bucket_action.triggered:
            bstate = recompute_readiness(bstate, confidence=bucket_confidence, params=p)
            bstate = maybe_capture_known_good(
                bstate, factor=bucket_factor, effective_n=0.0, dispersion=0.0,
                confidence=bucket_confidence, sample_weight=1.0, now_ts=now_ts,
                model_version=1, provenance={"scope": "bucket"}, params=p)
        else:
            local_rollbacks.append(now_ts)
        buckets[bucket_key] = bstate

    # general escalation (bridges from local evidence; does NOT re-accumulate the outcome)
    local_rollbacks = [t for t in local_rollbacks
                       if _parse(t) is not None
                       and (now - _parse(t)).total_seconds() <= p.escalate_window_s]
    distinct_degraded = sum(
        1 for st in buckets.values()
        if st.readiness in (BoostLearningReadiness.DEGRADED, BoostLearningReadiness.ROLLED_BACK,
                            BoostLearningReadiness.COOLDOWN))
    general = scoped.general
    general_action = no_action
    if not _in_cooldown(general, now):
        systemic = (distinct_degraded >= p.escalate_distinct_buckets
                    or len(local_rollbacks) >= p.escalate_local_rollbacks)
        if systemic:
            dur = _cooldown_duration_s(p, general.rollback_count, True,
                                       general.overshoot_evidence)
            cooldown_until = (now + _td(dur)).isoformat()
            kg = general.known_good
            general_action = RollbackAction(
                True, "general", "escalation_multi_bucket",
                (kg.factor if kg else None), kg, cooldown_until, fast_safety=True)
            general = replace(
                general, readiness=BoostLearningReadiness.ROLLED_BACK,
                rollback_count=general.rollback_count + 1, last_rollback_ts=now_ts,
                cooldown_start_ts=now_ts, cooldown_until_ts=cooldown_until,
                cooldown_reason="escalation_multi_bucket", rollback_scope="general",
                reason="escalation_multi_bucket", degradation_started_at=now_ts,
                rehab_clean_count=0, rehab_weight=0.0)
            local_rollbacks = []                  # consumed by escalation
        else:
            # general aggregates maturity across buckets and tracks its OWN factor stability.
            # A single degraded bucket does NOT degrade general (that would block the legitimate
            # general fallback); only systemic escalation (above) rolls general back.
            general = replace(general, adaptive_outcomes=max(
                general.adaptive_outcomes, sum(st.adaptive_outcomes for st in buckets.values())))
            if adaptive and reliability_known and outcome_class in (
                    _DEGRADATION_CLASSES | _POSITIVE_CLASSES):
                general = _push_factor(general, general_factor, p)
            general = recompute_readiness(general, confidence=general_confidence, params=p)

    buckets = _prune_buckets(buckets, now, p)
    new = BoostScopedDegradation(general=general, buckets=buckets,
                                 local_rollback_ts=tuple(local_rollbacks[-p.count_cap:]))
    return ScopedObserveResult(new, bucket_key, bucket_action, general_action)


def resolve_scoped_cooldown(scoped: BoostScopedDegradation, now_ts: str,
                            params: Optional[DegradationParams] = None) -> BoostScopedDegradation:
    """Re-evaluate cooldown/rehabilitation for every scope (time alone never restores)."""
    p = params or DegradationParams()
    general = resolve_cooldown(scoped.general, now_ts, p)
    buckets = {k: resolve_cooldown(v, now_ts, p) for k, v in scoped.buckets.items()}
    return replace(scoped, general=general, buckets=buckets)

