"""Controlled Runtime Switch for LE 2.0 (pure Python, Phase 18).

A baseline-first, fully reversible control policy: the existing ThermoSmart
result is the input and the safe fallback. An LE-2.0 prediction may only adjust a
control parameter when its release flag is on, its ConfidenceResult allows control
(no hard cap, gate open), the value is plausible, and the change is clamped /
rolled-in conservatively. Any doubt -> the baseline is returned unchanged.

This module performs NO dispatch and calls NO service: it returns a typed
ControlDecision that the existing controller resolves. The runtime default stays
SHADOW; CONTROL is reachable only via explicit (test/dev) RuntimeConfig.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ..contracts import Regime

CONTROL_POLICY_VERSION = 1


class ControlFeature(Enum):
    PREHEAT_START = "preheat_start"
    EARLY_CUTOFF = "early_cutoff"
    BOOST_OFFSET = "boost_offset"
    BOOST_DURATION = "boost_duration"
    FORECAST_ADJUSTMENT = "forecast_adjustment"
    MANUAL_PREFERENCE_BIAS = "manual_preference_bias"


class ControlDecisionStatus(Enum):
    APPLIED = "applied"
    FALLBACK = "fallback"
    ADVISORY = "advisory"


class ControlRejection(Enum):
    RUNTIME_NOT_CONTROL = "runtime_not_control"
    FEATURE_NOT_ENABLED = "feature_not_enabled"
    PREDICTION_MISSING = "prediction_missing"
    PREDICTION_INVALID = "prediction_invalid"
    WRONG_PREDICTION_TYPE = "wrong_prediction_type"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    CONTROL_GATE_CLOSED = "control_gate_closed"
    HARD_CAP_ACTIVE = "hard_cap_active"
    LOW_RELIABILITY = "low_reliability"
    PRIOR_DOMINANT = "prior_dominant"
    FALLBACK_DOMINANT = "fallback_dominant"
    MISSING_REQUIRED = "missing_required"
    DISTURBED = "disturbed"
    UNKNOWN_REGIME = "unknown_regime"
    STALE_INPUT = "stale_input"
    MISSING_CURRENT_TEMPERATURE = "missing_current_temperature"
    UNSAFE_MODE = "unsafe_mode"
    WINDOW_OPEN = "window_open"
    HEATING_FAILURE = "heating_failure"
    FROST_PROTECTION = "frost_protection"
    DEVICE_LIMIT = "device_limit"
    BASELINE_MISSING = "baseline_missing"
    EXCESSIVE_DEVIATION = "excessive_deviation"
    IMPLAUSIBLE_DURATION = "implausible_duration"
    VERSION_MISMATCH = "version_mismatch"
    PARAMETER_MISMATCH = "parameter_mismatch"
    BOOST_LIMIT_MISMATCH = "boost_limit_mismatch"
    POLICY_ERROR = "policy_error"


_ADVISORY_ONLY = frozenset({
    ControlFeature.FORECAST_ADJUSTMENT, ControlFeature.MANUAL_PREFERENCE_BIAS})


@dataclass(frozen=True)
class FeatureClamp:
    min_value: float
    max_value: float
    max_step: float                 # max deviation from baseline per application (roll-in)
    excessive_deviation: float      # raw deviation beyond this is rejected outright


@dataclass(frozen=True)
class ControlPolicyParameters:
    enabled_features: frozenset = field(default_factory=lambda: frozenset({
        ControlFeature.PREHEAT_START, ControlFeature.EARLY_CUTOFF,
        ControlFeature.BOOST_OFFSET, ControlFeature.BOOST_DURATION}))
    min_control_confidence: float = 0.55
    reject_on_fallback: bool = True
    reject_on_prior_dominant: bool = True
    prior_dominant_threshold: float = 0.6
    min_reliability: float = 0.3
    clamps: Mapping[ControlFeature, FeatureClamp] = field(default_factory=lambda: {
        ControlFeature.PREHEAT_START: FeatureClamp(0.0, 180.0, 15.0, 90.0),       # minutes
        ControlFeature.EARLY_CUTOFF: FeatureClamp(0.0, 60.0, 10.0, 45.0),         # minutes early
        ControlFeature.BOOST_OFFSET: FeatureClamp(0.0, 8.0, 0.5, 4.0),            # °C additive
        ControlFeature.BOOST_DURATION: FeatureClamp(300.0, 7200.0, 600.0, 3600.0),  # seconds
    })
    boost_offset_runtime_limit: float = 8.0   # must equal TPI_MAX_BOOST_CELSIUS
    policy_version: int = CONTROL_POLICY_VERSION

    def clamp_for(self, feature: ControlFeature) -> Optional[FeatureClamp]:
        return self.clamps.get(feature)


@dataclass(frozen=True)
class ConfidenceSummary:
    purpose: Optional[str]
    value: float
    level: Optional[str]
    reliability: float
    control_allowed: bool
    fallback_used: bool
    hard_cap: bool
    prior_fraction: float
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControlFallback:
    baseline_source: str
    baseline_value: Optional[float]
    unit: str
    reason: str
    le2_prediction: Optional[float]
    confidence: Optional[ConfidenceSummary]
    applied: bool = False
    safe: bool = True


@dataclass(frozen=True)
class ControlDecision:
    feature: str
    status: ControlDecisionStatus
    baseline_value: Optional[float]
    proposed_value: Optional[float]
    final_value: Optional[float]
    unit: str
    clamp_applied: float                 # how much the proposal was clamped
    rejection: Optional[str]
    fallback: Optional[ControlFallback]
    confidence: Optional[ConfidenceSummary]
    reason_codes: tuple[str, ...]
    policy_version: int = CONTROL_POLICY_VERSION
    model_version: Optional[int] = None
    parameter_version: Optional[int] = None

    @property
    def applied(self) -> bool:
        return self.status is ControlDecisionStatus.APPLIED

    @property
    def control_value(self) -> Optional[float]:
        """The value the controller should use (always finite; baseline on fallback)."""
        return self.final_value


@dataclass(frozen=True)
class ControlledPrediction:
    feature: ControlFeature
    value: float
    unit: str
    confidence_result: Any                # a ConfidenceResult (not recomputed)
    model_version: Optional[int] = None
    parameter_version: Optional[int] = None


@dataclass(frozen=True)
class ControlContext:
    regime: Optional[Regime] = None
    has_current_temperature: bool = True
    window_open: bool = False
    heating_failure: bool = False
    frost_protection: bool = False
    mode_safe: bool = True
    stale_input: bool = False
    device_min: Optional[float] = None
    device_max: Optional[float] = None
    boost_runtime_limit: Optional[float] = None   # TPI_MAX_BOOST_CELSIUS at call site


def _summary(cr: Any) -> Optional[ConfidenceSummary]:
    if cr is None:
        return None
    hard = any(getattr(c, "hard", False) for c in getattr(cr, "applied_caps", ()))
    return ConfidenceSummary(
        purpose=getattr(cr.purpose, "value", None), value=float(cr.value),
        level=getattr(cr.level, "value", None), reliability=float(cr.reliability),
        control_allowed=bool(cr.control_allowed), fallback_used=bool(cr.fallback_used),
        hard_cap=hard, prior_fraction=float(getattr(cr.breakdown, "prior_fraction", 0.0)),
        reason_codes=tuple(cr.reason_codes))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# -- policy -------------------------------------------------------------------

class ControlPolicy:
    """Pure baseline-first control resolver. No dispatch, no services."""

    def __init__(self, params: Optional[ControlPolicyParameters] = None, *,
                 runtime_is_control: bool = False) -> None:
        self._p = params or ControlPolicyParameters()
        self._runtime_is_control = runtime_is_control

    @property
    def params(self) -> ControlPolicyParameters:
        return self._p

    def _fallback(self, feature, baseline, unit, reason, prediction, conf) -> ControlDecision:
        fb = ControlFallback(baseline_source="existing_controller", baseline_value=baseline,
                             unit=unit, reason=reason, le2_prediction=prediction,
                             confidence=conf, applied=False, safe=True)
        status = (ControlDecisionStatus.ADVISORY if reason == ControlRejection.FEATURE_NOT_ENABLED.value
                  and conf is not None else ControlDecisionStatus.FALLBACK)
        return ControlDecision(
            feature=feature.value, status=ControlDecisionStatus.FALLBACK,
            baseline_value=baseline, proposed_value=prediction, final_value=baseline,
            unit=unit, clamp_applied=0.0, rejection=reason, fallback=fb, confidence=conf,
            reason_codes=(reason,))

    def resolve(self, feature: ControlFeature, baseline_value: Optional[float],
                prediction: Optional[ControlledPrediction], *,
                context: Optional[ControlContext] = None, unit: str = "") -> ControlDecision:
        ctx = context or ControlContext()
        conf = _summary(prediction.confidence_result) if prediction is not None else None

        # 1. runtime + feature gating
        if not self._runtime_is_control:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.RUNTIME_NOT_CONTROL.value, None, conf)
        if feature in _ADVISORY_ONLY or feature not in self._p.enabled_features:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.FEATURE_NOT_ENABLED.value,
                                  prediction.value if prediction else None, conf)
        # 2. baseline + prediction presence/validity
        if baseline_value is None or not math.isfinite(baseline_value):
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.BASELINE_MISSING.value, None, conf)
        if prediction is None:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.PREDICTION_MISSING.value, None, conf)
        if not math.isfinite(prediction.value):
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.PREDICTION_INVALID.value, None, conf)
        # 3. safety context
        for cond, reason in (
            (ctx.window_open, ControlRejection.WINDOW_OPEN),
            (ctx.heating_failure, ControlRejection.HEATING_FAILURE),
            (ctx.frost_protection, ControlRejection.FROST_PROTECTION),
            (not ctx.mode_safe, ControlRejection.UNSAFE_MODE),
            (ctx.stale_input, ControlRejection.STALE_INPUT),
            (ctx.regime is Regime.DISTURBED, ControlRejection.DISTURBED),
        ):
            if cond:
                return self._fallback(feature, baseline_value, unit, reason.value,
                                      prediction.value, conf)
        # 4. confidence authority (single source: the ConfidenceResult)
        if conf is None:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.CONTROL_GATE_CLOSED.value, prediction.value, None)
        if conf.hard_cap:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.HARD_CAP_ACTIVE.value, prediction.value, conf)
        if not conf.control_allowed:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.CONTROL_GATE_CLOSED.value, prediction.value, conf)
        if conf.value < self._p.min_control_confidence:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.INSUFFICIENT_CONFIDENCE.value,
                                  prediction.value, conf)
        if conf.reliability < self._p.min_reliability:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.LOW_RELIABILITY.value, prediction.value, conf)
        if self._p.reject_on_fallback and conf.fallback_used:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.FALLBACK_DOMINANT.value, prediction.value, conf)
        if self._p.reject_on_prior_dominant and conf.prior_fraction >= self._p.prior_dominant_threshold:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.PRIOR_DOMINANT.value, prediction.value, conf)
        # 5. boost runtime-limit guard
        if feature is ControlFeature.BOOST_OFFSET:
            limit = ctx.boost_runtime_limit
            if limit is not None and abs(self._p.boost_offset_runtime_limit - limit) > 1e-9:
                return self._fallback(feature, baseline_value, unit,
                                      ControlRejection.BOOST_LIMIT_MISMATCH.value,
                                      prediction.value, conf)
        # 6. clamp + conservative roll-in
        clamp = self._p.clamp_for(feature)
        if clamp is None:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.FEATURE_NOT_ENABLED.value, prediction.value, conf)
        raw_dev = prediction.value - baseline_value
        if abs(raw_dev) > clamp.excessive_deviation:
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.EXCESSIVE_DEVIATION.value, prediction.value, conf)
        stepped = baseline_value + _clamp(raw_dev, -clamp.max_step, clamp.max_step)
        final = _clamp(stepped, clamp.min_value, clamp.max_value)
        # device limits (for setpoint-like features)
        if ctx.device_min is not None:
            final = max(final, ctx.device_min)
        if ctx.device_max is not None:
            final = min(final, ctx.device_max)
        if not math.isfinite(final):
            return self._fallback(feature, baseline_value, unit,
                                  ControlRejection.IMPLAUSIBLE_DURATION.value, prediction.value, conf)
        clamp_applied = round(abs(prediction.value - final), 6)
        return ControlDecision(
            feature=feature.value, status=ControlDecisionStatus.APPLIED,
            baseline_value=baseline_value, proposed_value=prediction.value,
            final_value=round(final, 6), unit=unit, clamp_applied=clamp_applied,
            rejection=None, fallback=None, confidence=conf,
            reason_codes=("applied",) + (("clamped",) if clamp_applied > 1e-9 else ()),
            model_version=prediction.model_version,
            parameter_version=prediction.parameter_version)

    def advisory(self, feature: ControlFeature, prediction: Optional[ControlledPrediction]
                 ) -> ControlDecision:
        """Advisory-only resolution (never applied, e.g. manual preference bias)."""
        conf = _summary(prediction.confidence_result) if prediction is not None else None
        return ControlDecision(
            feature=feature.value, status=ControlDecisionStatus.ADVISORY,
            baseline_value=None, proposed_value=prediction.value if prediction else None,
            final_value=None, unit="", clamp_applied=0.0, rejection=None, fallback=None,
            confidence=conf, reason_codes=("advisory_only",))


# -- control application ledger -----------------------------------------------

@dataclass(frozen=True)
class ControlApplicationResult:
    decision: ControlDecision
    applied: bool
    final_value: Optional[float]


@dataclass(frozen=True)
class ControlLedgerEntry:
    zone: str
    decision_id: str
    feature: str
    baseline_value: Optional[float]
    proposed_value: Optional[float]
    final_value: Optional[float]
    applied: bool
    rejection: Optional[str]
    confidence_value: Optional[float]
    fallback_used: bool
    ts: str
    policy_version: int
    model_version: Optional[int]
    parameter_version: Optional[int]


class ControlApplicationLedger:
    def __init__(self, max_entries: int = 300) -> None:
        self._max = max_entries
        self._entries: list[ControlLedgerEntry] = []
        self._applied_counts: dict[str, int] = {}
        self._rejection_counts: dict[str, int] = {}
        self._fallback_count = 0
        self._clamp_total = 0.0
        self._clamp_n = 0

    def record(self, zone: str, decision_id: str, decision: ControlDecision, ts: str) -> None:
        self._entries.append(ControlLedgerEntry(
            zone=zone, decision_id=decision_id, feature=decision.feature,
            baseline_value=decision.baseline_value, proposed_value=decision.proposed_value,
            final_value=decision.final_value, applied=decision.applied,
            rejection=decision.rejection,
            confidence_value=decision.confidence.value if decision.confidence else None,
            fallback_used=decision.confidence.fallback_used if decision.confidence else False,
            ts=ts, policy_version=decision.policy_version, model_version=decision.model_version,
            parameter_version=decision.parameter_version))
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]
        if decision.applied:
            self._applied_counts[decision.feature] = self._applied_counts.get(decision.feature, 0) + 1
            self._clamp_total += decision.clamp_applied
            self._clamp_n += 1
        else:
            self._fallback_count += 1
            if decision.rejection:
                self._rejection_counts[decision.rejection] = \
                    self._rejection_counts.get(decision.rejection, 0) + 1

    @property
    def size(self) -> int:
        return len(self._entries)

    def summary(self) -> dict:
        return {
            "applied_counts": dict(self._applied_counts),
            "rejection_counts": dict(self._rejection_counts),
            "fallback_count": self._fallback_count,
            "mean_clamp": round(self._clamp_total / self._clamp_n, 4) if self._clamp_n else None,
            "size": len(self._entries),
        }

    def recent(self, n: int = 50) -> tuple[ControlLedgerEntry, ...]:
        return tuple(self._entries[-n:])

    def research_samples(self, n: int = 50) -> list:
        return [
            {"feature": e.feature, "baseline": e.baseline_value, "proposed": e.proposed_value,
             "applied": e.applied, "rejection": e.rejection,
             "confidence": e.confidence_value, "policy_version": e.policy_version}
            for e in self._entries[-n:]]
