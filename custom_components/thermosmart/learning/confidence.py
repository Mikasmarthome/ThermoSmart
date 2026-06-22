"""ConfidenceAggregator for LE 2.0 (pure Python).

The single authoritative place that combines per-model ``ConfidenceContribution``
objects into a purpose-bound, capped, gated confidence. Models only report their
own contribution; the aggregator never recomputes a model, never mutates inputs,
holds no learning state, persists nothing and is fully deterministic.

It imports no concrete model and no Home Assistant / storage / coordinator code —
it works purely on typed contributions, a purpose policy and optional regime /
data-quality / conflict signals. Freshness must be supplied already materialised;
if a clock were ever needed it would be injected explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping, Optional, Sequence

from .contracts import ConfidenceContribution, Regime

AGGREGATOR_VERSION = 1
PARAMETER_VERSION = 1


# -- enums --------------------------------------------------------------------

class ConfidencePurpose(Enum):
    HEAT_RATE = "heat_rate"
    PREHEAT = "preheat"
    HEAT_LOSS = "heat_loss"
    AFTERHEAT = "afterheat"
    EARLY_CUTOFF = "early_cutoff"
    OUTCOME_QUALITY = "outcome_quality"
    FORECAST = "forecast"
    BOOST = "boost"
    MANUAL_CORRECTION = "manual_correction"
    CONTROL_DECISION = "control_decision"
    ADVISORY = "advisory"
    ONSET_DELAY = "onset_delay"       # TRV command → measurable room temperature response


class ComponentKind(Enum):
    HEAT_RATE = "heat_rate"
    HEAT_LOSS = "heat_loss"
    AFTERHEAT = "afterheat"
    PREHEAT = "preheat"
    OUTCOME = "outcome"
    FORECAST = "forecast"
    BOOST = "boost"
    MANUAL_CORRECTION = "manual_correction"
    CURRENT_TEMPERATURE = "current_temperature"
    OUTDOOR = "outdoor"
    DATA_QUALITY = "data_quality"
    BUCKET_SUPPORT = "bucket_support"
    PREHEAT_PRIOR = "preheat_prior"
    ONSET_DELAY = "onset_delay"       # latency component: TRV→room


class EvidenceDomain(Enum):
    THERMAL_TRAJECTORY = "thermal_trajectory"
    CONTROLLER_EVENTS = "controller_events"
    WEATHER = "weather"
    FORECAST = "forecast"
    USER_CORRECTION = "user_correction"
    OUTCOME_HISTORY = "outcome_history"
    DEVICE_TELEMETRY = "device_telemetry"
    CURRENT_STATE = "current_state"


class ConfidenceStatus(Enum):
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    INVALID = "invalid"
    NOT_COMPUTABLE = "not_computable"
    FALLBACK = "fallback"
    HEALTHY = "healthy"


class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ConfidenceCap(Enum):
    COLD_START = "cold_start"
    PRIOR_DOMINANT = "prior_dominant"
    LOW_EVIDENCE = "low_evidence"
    LOW_RELIABILITY = "low_reliability"
    STALE_DATA = "stale_data"
    INVALID_DATA = "invalid_data"
    MISSING_REQUIRED = "missing_required"
    MISSING_CURRENT_TEMPERATURE = "missing_current_temperature"
    TRV_ONLY_RELATIVE = "trv_only_relative"
    BUCKET_UNSUPPORTED = "bucket_unsupported"
    MODEL_DISAGREEMENT = "model_disagreement"
    DISTURBED = "disturbed"
    UNKNOWN_REGIME = "unknown_regime"
    FALLBACK_USED = "fallback_used"
    PARTIAL_OUTCOME = "partial_outcome"
    FORECAST_UNAVAILABLE = "forecast_unavailable"
    CONTEXT_MISMATCH = "context_mismatch"
    VERSION_MISMATCH = "version_mismatch"


class ConfidenceReason(Enum):
    REQUIRED_PRESENT = "required_present"
    REQUIRED_MISSING = "required_missing"
    OPTIONAL_PRESENT = "optional_present"
    OPTIONAL_MISSING = "optional_missing"
    BONUS_APPLIED = "bonus_applied"
    CORRELATED_EVIDENCE_DISCOUNTED = "correlated_evidence_discounted"
    CAP_APPLIED = "cap_applied"
    FALLBACK_USED = "fallback_used"
    DISTURBED = "disturbed"
    UNKNOWN_REGIME = "unknown_regime"
    MODEL_DISAGREEMENT = "model_disagreement"
    NO_CONTRIBUTIONS = "no_contributions"


class ConfidenceStatusResult(Enum):
    """Overall status of the aggregated result."""
    OK = "ok"
    DEGRADED = "degraded"
    FALLBACK = "fallback"
    BLOCKED = "blocked"


class AggregationMode(Enum):
    PREDICTION = "prediction"
    CONTROL = "control"
    ADVISORY = "advisory"


# -- value types --------------------------------------------------------------

@dataclass(frozen=True)
class PurposePolicy:
    purpose: ConfidencePurpose
    required: tuple[ComponentKind, ...]
    optional: tuple[ComponentKind, ...] = ()
    requires_current_temperature: bool = False
    control_relevant: bool = False
    trv_only_cap: Optional[float] = None
    fallback_components: tuple[ComponentKind, ...] = ()


@dataclass(frozen=True)
class ConfidenceComponent:
    """One typed model contribution as seen by the aggregator."""
    kind: ComponentKind
    contribution: ConfidenceContribution
    status: ConfidenceStatus = ConfidenceStatus.HEALTHY
    is_required: Optional[bool] = None
    evidence_domains: tuple[EvidenceDomain, ...] = ()
    freshness: Optional[float] = None
    reliability: Optional[float] = None
    conflicts_with: tuple[ComponentKind, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.contribution, ConfidenceContribution):
            raise ValueError("component requires a ConfidenceContribution")
        for name in ("freshness", "reliability"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or not 0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be finite within [0,1]")

    @property
    def domains(self) -> tuple[EvidenceDomain, ...]:
        if self.evidence_domains:
            return self.evidence_domains
        return tuple(EvidenceDomain(d) for d in self.contribution.evidence_domains
                     if d in {e.value for e in EvidenceDomain})

    @property
    def reliability_value(self) -> float:
        if self.reliability is not None:
            return self.reliability
        if self.contribution.reliability is not None:
            return self.contribution.reliability
        return 1.0

    @property
    def freshness_value(self) -> float:
        if self.freshness is not None:
            return self.freshness
        if self.contribution.freshness is not None:
            return self.contribution.freshness
        return 1.0

    @property
    def prior_fraction(self) -> float:
        pf = self.contribution.prior_fraction
        if pf is not None:
            return pf
        return 1.0 if self.contribution.fallback_used else 0.0

    @property
    def present(self) -> bool:
        return self.status in (ConfidenceStatus.HEALTHY, ConfidenceStatus.FALLBACK)


@dataclass(frozen=True)
class ConfidenceConflict:
    kind: str
    components: tuple[ComponentKind, ...]
    severity: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.severity) or not 0.0 <= self.severity <= 1.0:
            raise ValueError("conflict severity must be finite within [0,1]")


@dataclass(frozen=True)
class ConfidenceInput:
    purpose: ConfidencePurpose
    components: tuple[ConfidenceComponent, ...] = ()
    regime: Optional[Regime] = None
    data_quality: Optional[float] = None
    conflicts: tuple[ConfidenceConflict, ...] = ()
    # whether the SELECTED concrete prediction path actually depends on the
    # forecast. Forecast caps (FORECAST_UNAVAILABLE / stale forecast) apply only
    # to a forecast-dependent path; a purely local path is never forecast-capped.
    forecast_required: bool = False
    prediction_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.data_quality is not None and (
                not math.isfinite(self.data_quality) or not 0.0 <= self.data_quality <= 1.0):
            raise ValueError("data_quality must be finite within [0,1]")


@dataclass(frozen=True)
class AppliedCap:
    cap: ConfidenceCap
    max_value: float
    reason: str
    source: str
    priority: int
    hard: bool
    purpose: Optional[ConfidencePurpose] = None


@dataclass(frozen=True)
class ComponentBreakdown:
    kind: ComponentKind
    is_required: bool
    confidence: float
    reliability: float
    freshness: float
    effective_weight: float
    evidence_domains: tuple[str, ...]
    prior_fraction: float
    status: str
    included: bool


@dataclass(frozen=True)
class ControlGateResult:
    allowed: bool
    reason_codes: tuple[str, ...]
    required_minimum: float
    actual_confidence: float
    safe_fallback_required: bool


@dataclass(frozen=True)
class ConfidenceBreakdown:
    purpose: ConfidencePurpose
    base_value: float
    components: tuple[ComponentBreakdown, ...]
    required_components: tuple[str, ...]
    optional_components: tuple[str, ...]
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]
    positive_bonus: float
    applied_caps: tuple[AppliedCap, ...]
    conflicts: tuple[str, ...]
    evidence_domains: tuple[str, ...]
    raw_evidence_count: int
    effective_evidence: float
    prior_fraction: float
    final_value: float
    final_reliability: float
    level: ConfidenceLevel
    gate: ControlGateResult
    aggregator_version: int
    parameter_version: int
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ConfidenceResult:
    purpose: ConfidencePurpose
    value: float
    level: ConfidenceLevel
    status: ConfidenceStatusResult
    breakdown: ConfidenceBreakdown
    required_components: tuple[str, ...]
    optional_components: tuple[str, ...]
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]
    applied_caps: tuple[AppliedCap, ...]
    reason_codes: tuple[str, ...]
    evidence_count: int
    effective_evidence: float
    reliability: float
    fallback_used: bool
    aggregator_version: int
    parameter_version: int
    control_allowed: bool
    advisory_allowed: bool
    gate: ControlGateResult


# -- parameters ---------------------------------------------------------------

def _default_policies() -> dict[ConfidencePurpose, PurposePolicy]:
    P = ConfidencePurpose
    K = ComponentKind
    return {
        P.HEAT_RATE: PurposePolicy(
            P.HEAT_RATE, required=(K.HEAT_RATE,),
            optional=(K.DATA_QUALITY, K.BUCKET_SUPPORT, K.OUTDOOR)),
        P.HEAT_LOSS: PurposePolicy(
            P.HEAT_LOSS, required=(K.HEAT_LOSS,),
            optional=(K.DATA_QUALITY, K.OUTDOOR), trv_only_cap=0.85),
        P.AFTERHEAT: PurposePolicy(
            P.AFTERHEAT, required=(K.AFTERHEAT,), optional=(K.DATA_QUALITY,)),
        P.EARLY_CUTOFF: PurposePolicy(
            P.EARLY_CUTOFF, required=(K.AFTERHEAT,),
            optional=(K.HEAT_RATE, K.DATA_QUALITY, K.CURRENT_TEMPERATURE),
            requires_current_temperature=False, control_relevant=True),
        P.PREHEAT: PurposePolicy(
            P.PREHEAT, required=(K.HEAT_RATE,),
            optional=(K.HEAT_LOSS, K.AFTERHEAT, K.FORECAST, K.OUTDOOR, K.OUTCOME,
                      K.CURRENT_TEMPERATURE, K.DATA_QUALITY),
            requires_current_temperature=False, control_relevant=True,
            fallback_components=(K.PREHEAT_PRIOR,)),
            # NOTE: no trv_only_cap — a local preheat from HeatRate/HeatLoss/
            # Afterheat + current deficit is an absolute statement; forecast is
            # optional precision only.
        P.OUTCOME_QUALITY: PurposePolicy(
            P.OUTCOME_QUALITY, required=(K.OUTCOME,),
            optional=(K.DATA_QUALITY, K.BUCKET_SUPPORT)),
        P.FORECAST: PurposePolicy(
            P.FORECAST, required=(K.FORECAST,), optional=(K.OUTDOOR, K.DATA_QUALITY)),
        P.BOOST: PurposePolicy(
            P.BOOST, required=(K.HEAT_RATE,), optional=(K.OUTCOME, K.DATA_QUALITY),
            control_relevant=True),
        P.MANUAL_CORRECTION: PurposePolicy(
            P.MANUAL_CORRECTION, required=(K.MANUAL_CORRECTION,), optional=(K.OUTCOME,)),
        P.CONTROL_DECISION: PurposePolicy(
            P.CONTROL_DECISION, required=(K.HEAT_RATE,),
            optional=(K.HEAT_LOSS, K.AFTERHEAT, K.OUTCOME, K.FORECAST,
                      K.CURRENT_TEMPERATURE, K.DATA_QUALITY),
            requires_current_temperature=True, control_relevant=True),
        P.ADVISORY: PurposePolicy(
            P.ADVISORY, required=(), optional=(K.OUTCOME, K.HEAT_RATE, K.HEAT_LOSS,
                                              K.AFTERHEAT, K.DATA_QUALITY)),
    }


def _default_control_gates() -> dict[ConfidencePurpose, float]:
    P = ConfidencePurpose
    return {
        P.EARLY_CUTOFF: 0.6, P.PREHEAT: 0.55, P.CONTROL_DECISION: 0.6,
        P.BOOST: 0.55,
    }


def _default_advisory_gates() -> dict[ConfidencePurpose, float]:
    return {p: 0.25 for p in ConfidencePurpose}


@dataclass(frozen=True)
class ConfidenceParameters:
    level_thresholds: tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)
    control_gates: Mapping[ConfidencePurpose, float] = field(default_factory=_default_control_gates)
    advisory_gates: Mapping[ConfidencePurpose, float] = field(default_factory=_default_advisory_gates)
    default_control_gate: float = 0.6
    cold_start_cap: float = 0.35
    cold_start_evidence: int = 3
    prior_dominant_cap: float = 0.5
    prior_dominant_threshold: float = 0.6
    low_evidence_cap: float = 0.45
    min_effective_evidence: float = 2.0
    low_reliability_cap: float = 0.5
    low_reliability_threshold: float = 0.4
    stale_cap: float = 0.5
    invalid_cap: float = 0.1
    missing_required_cap: float = 0.2
    missing_current_temperature_cap: float = 0.3
    bucket_unsupported_cap: float = 0.6
    disagreement_cap: float = 0.5
    disturbed_cap: float = 0.05
    unknown_regime_cap: float = 0.5
    partial_outcome_cap: float = 0.6
    forecast_unavailable_cap: float = 0.7
    fallback_cap: float = 0.5
    max_optional_bonus: float = 0.15
    bonus_per_component: float = 0.05
    correlation_discount: float = 0.4
    policies: Mapping[ConfidencePurpose, PurposePolicy] = field(default_factory=_default_policies)
    parameter_version: int = PARAMETER_VERSION

    def control_gate(self, purpose: ConfidencePurpose) -> float:
        return self.control_gates.get(purpose, self.default_control_gate)

    def advisory_gate(self, purpose: ConfidencePurpose) -> float:
        return self.advisory_gates.get(purpose, 0.25)


_TEMPERATURE_PURPOSES = frozenset({
    ConfidencePurpose.CONTROL_DECISION, ConfidencePurpose.EARLY_CUTOFF,
    ConfidencePurpose.PREHEAT,
})


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


# -- aggregator ---------------------------------------------------------------

class ConfidenceAggregator:
    aggregator_version = AGGREGATOR_VERSION
    parameter_version = PARAMETER_VERSION

    def __init__(self, params: Optional[ConfidenceParameters] = None) -> None:
        self._params = params or ConfidenceParameters()

    # convenience purpose-bound entrypoints
    def aggregate_for_prediction(self, input: ConfidenceInput) -> ConfidenceResult:
        return self.aggregate(input)

    def aggregate_for_control(self, input: ConfidenceInput) -> ConfidenceResult:
        return self.aggregate(input)

    def aggregate_for_advisory(self, input: ConfidenceInput) -> ConfidenceResult:
        return self.aggregate(input)

    def aggregate(self, input: ConfidenceInput) -> ConfidenceResult:
        p = self._params
        policy = p.policies.get(input.purpose) or PurposePolicy(input.purpose, required=())
        self._validate(input)

        by_kind: dict[ComponentKind, ConfidenceComponent] = {}
        for comp in input.components:
            if comp.kind in by_kind:
                raise ValueError(f"duplicate component {comp.kind.value}")
            by_kind[comp.kind] = comp

        required_kinds = [k for k in policy.required]
        optional_kinds = [k for k in policy.optional]
        # per-component is_required override
        for comp in input.components:
            if comp.is_required is True and comp.kind not in required_kinds:
                required_kinds.append(comp.kind)
            if comp.is_required is False and comp.kind in required_kinds:
                required_kinds.remove(comp.kind)
                optional_kinds.append(comp.kind)

        reasons: list[str] = []
        caps: list[AppliedCap] = []

        # --- required components -> core base ---
        present_required: list[ConfidenceComponent] = []
        missing_required: list[str] = []
        for kind in required_kinds:
            comp = by_kind.get(kind)
            if comp is not None and comp.present:
                present_required.append(comp)
            else:
                missing_required.append(kind.value)

        fallback_used = False
        # fallback path: no required present but a safe fallback component is
        if not present_required and missing_required:
            for fk in policy.fallback_components:
                fc = by_kind.get(fk)
                if fc is not None and fc.present:
                    present_required.append(fc)
                    fallback_used = True
                    reasons.append(ConfidenceReason.FALLBACK_USED.value)
                    caps.append(self._cap(ConfidenceCap.FALLBACK_USED, p.fallback_cap,
                                          "safe prior used", fk.value, 40, False, input.purpose))
                    break

        # invalid required -> hard cap
        for kind in required_kinds:
            comp = by_kind.get(kind)
            if comp is not None and comp.status is ConfidenceStatus.INVALID:
                caps.append(self._cap(ConfidenceCap.INVALID_DATA, p.invalid_cap,
                                      "required input invalid", kind.value, 100, True,
                                      input.purpose))

        if missing_required and not fallback_used:
            caps.append(self._cap(ConfidenceCap.MISSING_REQUIRED, p.missing_required_cap,
                                  f"missing required: {','.join(missing_required)}",
                                  "policy", 90, True, input.purpose))
            reasons.append(ConfidenceReason.REQUIRED_MISSING.value)

        # current-temperature requirement
        if policy.requires_current_temperature:
            ct = by_kind.get(ComponentKind.CURRENT_TEMPERATURE)
            if ct is None or not ct.present:
                caps.append(self._cap(
                    ConfidenceCap.MISSING_CURRENT_TEMPERATURE,
                    p.missing_current_temperature_cap, "no current temperature",
                    "policy", 85, True, input.purpose))

        # build the weighted core base with novelty (anti double-counting)
        seen_domains: set[EvidenceDomain] = set()
        component_breakdowns: list[ComponentBreakdown] = []
        core_num = 0.0
        core_den = 0.0
        raw_evidence = 0
        effective_evidence = 0.0
        prior_fracs: list[float] = []
        reliabilities: list[float] = []

        def novelty(domains: Sequence[EvidenceDomain]) -> float:
            if not domains:
                return 1.0
            return 1.0 if not seen_domains.intersection(domains) else p.correlation_discount

        for comp in sorted(present_required, key=lambda c: c.kind.value):
            dom = comp.domains
            nov = novelty(dom)
            conf = comp.contribution.value
            rel = comp.reliability_value
            fresh = comp.freshness_value
            if comp.status is ConfidenceStatus.STALE:
                fresh = min(fresh, 0.5)
            eff_w = 1.0 * nov
            eff_conf = conf * rel * fresh
            core_num += eff_conf * eff_w
            core_den += eff_w
            raw_evidence += comp.contribution.evidence_count
            effective_evidence += comp.contribution.evidence_count * nov
            prior_fracs.append(comp.prior_fraction)
            reliabilities.append(rel)
            if nov < 1.0:
                reasons.append(ConfidenceReason.CORRELATED_EVIDENCE_DISCOUNTED.value)
            seen_domains.update(dom)
            component_breakdowns.append(ComponentBreakdown(
                kind=comp.kind, is_required=True, confidence=conf, reliability=rel,
                freshness=fresh, effective_weight=eff_w,
                evidence_domains=tuple(d.value for d in dom),
                prior_fraction=comp.prior_fraction, status=comp.status.value, included=True))
            reasons.append(ConfidenceReason.REQUIRED_PRESENT.value)

        # advisory / no-required purposes: derive base from optional present
        base_from_optional = False
        if core_den == 0.0 and not required_kinds:
            for comp in sorted((by_kind.get(k) for k in optional_kinds), key=_kind_sort):
                if comp is None or not comp.present:
                    continue
                base_from_optional = True
                dom = comp.domains
                nov = novelty(dom)
                conf = comp.contribution.value
                rel = comp.reliability_value
                fresh = comp.freshness_value
                core_num += conf * rel * fresh * nov
                core_den += nov
                raw_evidence += comp.contribution.evidence_count
                effective_evidence += comp.contribution.evidence_count * nov
                reliabilities.append(rel)
                prior_fracs.append(comp.prior_fraction)
                seen_domains.update(dom)
                component_breakdowns.append(ComponentBreakdown(
                    kind=comp.kind, is_required=False, confidence=conf, reliability=rel,
                    freshness=fresh, effective_weight=nov,
                    evidence_domains=tuple(d.value for d in dom),
                    prior_fraction=comp.prior_fraction, status=comp.status.value, included=True))

        base_value = (core_num / core_den) if core_den > 0 else 0.0

        # --- optional bonus (bounded, correlation-aware) ---
        positive_bonus = 0.0
        missing_optional: list[str] = []
        if not base_from_optional:
            for kind in optional_kinds:
                comp = by_kind.get(kind)
                if comp is None or not comp.present:
                    missing_optional.append(kind.value)
                    continue
                dom = comp.domains
                nov = novelty(dom)
                if nov < 1.0:
                    reasons.append(ConfidenceReason.CORRELATED_EVIDENCE_DISCOUNTED.value)
                contrib_bonus = p.bonus_per_component * comp.contribution.value \
                    * comp.reliability_value * nov
                positive_bonus += contrib_bonus
                seen_domains.update(dom)
                raw_evidence += comp.contribution.evidence_count
                effective_evidence += comp.contribution.evidence_count * nov
                reasons.append(ConfidenceReason.OPTIONAL_PRESENT.value)
                component_breakdowns.append(ComponentBreakdown(
                    kind=comp.kind, is_required=False, confidence=comp.contribution.value,
                    reliability=comp.reliability_value, freshness=comp.freshness_value,
                    effective_weight=nov, evidence_domains=tuple(d.value for d in dom),
                    prior_fraction=comp.prior_fraction, status=comp.status.value, included=True))
            positive_bonus = min(positive_bonus, p.max_optional_bonus)
            if positive_bonus > 0:
                reasons.append(ConfidenceReason.BONUS_APPLIED.value)
        if missing_optional:
            reasons.append(ConfidenceReason.OPTIONAL_MISSING.value)

        # forecast optional missing -> documented soft cap on weather purposes
        if input.purpose is ConfidencePurpose.FORECAST and \
                ComponentKind.FORECAST.value in missing_required:
            pass  # handled by missing_required

        agg_reliability = min(reliabilities) if reliabilities else 0.0
        prior_fraction = max(prior_fracs) if prior_fracs else 1.0

        # --- generic caps ---
        if present_required or base_from_optional:
            if effective_evidence < p.min_effective_evidence:
                caps.append(self._cap(ConfidenceCap.LOW_EVIDENCE, p.low_evidence_cap,
                                      "insufficient effective evidence", "aggregate", 50, False,
                                      input.purpose))
            if raw_evidence > 0 and raw_evidence <= p.cold_start_evidence:
                caps.append(self._cap(ConfidenceCap.COLD_START, p.cold_start_cap,
                                      "cold start", "aggregate", 55, False, input.purpose))
            if prior_fraction >= p.prior_dominant_threshold:
                caps.append(self._cap(ConfidenceCap.PRIOR_DOMINANT, p.prior_dominant_cap,
                                      "prior dominant", "aggregate", 45, False, input.purpose))
            if agg_reliability < p.low_reliability_threshold:
                caps.append(self._cap(ConfidenceCap.LOW_RELIABILITY, p.low_reliability_cap,
                                      "low reliability", "aggregate", 48, False, input.purpose))
        elif not missing_required:
            # nothing present at all and nothing required -> cold start
            caps.append(self._cap(ConfidenceCap.COLD_START, p.cold_start_cap,
                                  "no evidence", "aggregate", 55, False, input.purpose))

        # status-derived caps from any component. A STALE input only caps the
        # aggregate when it is a required/core component or when the selected
        # path actually depends on it (forecast on a forecast-dependent path);
        # a stale OPTIONAL input is simply dropped from the bonus instead.
        for comp in input.components:
            if comp.status is ConfidenceStatus.STALE:
                forecast_dependent = (comp.kind is ComponentKind.FORECAST
                                      and input.forecast_required)
                if comp.kind in required_kinds or forecast_dependent:
                    caps.append(self._cap(ConfidenceCap.STALE_DATA, p.stale_cap, "stale input",
                                          comp.kind.value, 60, False, input.purpose))
            if comp.status is ConfidenceStatus.FALLBACK:
                caps.append(self._cap(ConfidenceCap.FALLBACK_USED, p.fallback_cap,
                                      "fallback", comp.kind.value, 40, False, input.purpose))
            if comp.kind is ComponentKind.BUCKET_SUPPORT and not comp.present:
                caps.append(self._cap(ConfidenceCap.BUCKET_UNSUPPORTED, p.bucket_unsupported_cap,
                                      "bucket unsupported", comp.kind.value, 35, False,
                                      input.purpose))
            if comp.kind is ComponentKind.OUTCOME and \
                    "partial" in comp.contribution.reasons:
                caps.append(self._cap(ConfidenceCap.PARTIAL_OUTCOME, p.partial_outcome_cap,
                                      "partial outcome", comp.kind.value, 35, False,
                                      input.purpose))

        # TRV-only relative cap (no weather / forecast evidence available). Only
        # purposes whose statement is genuinely relative without outdoor/forecast
        # carry a trv_only_cap (e.g. HEAT_LOSS); local preheat does not.
        if policy.trv_only_cap is not None and not _has_weather(input):
            caps.append(self._cap(ConfidenceCap.TRV_ONLY_RELATIVE, policy.trv_only_cap,
                                  "trv-only relative", "policy", 30, False, input.purpose))

        # Forecast cap is PATH-dependent, never derived from the purpose alone.
        # It applies only when the selected prediction path needs the forecast
        # but the forecast is unavailable / not configured. A stale forecast on a
        # forecast-dependent path is already handled above. A purely local path
        # (forecast_required=False) is never forecast-capped, so an unconfigured
        # forecast stays a documented missing_optional only.
        if input.forecast_required:
            fc = by_kind.get(ComponentKind.FORECAST)
            if fc is None or fc.status in (ConfidenceStatus.UNAVAILABLE,
                                           ConfidenceStatus.NOT_CONFIGURED,
                                           ConfidenceStatus.INVALID,
                                           ConfidenceStatus.NOT_COMPUTABLE):
                caps.append(self._cap(
                    ConfidenceCap.FORECAST_UNAVAILABLE, p.forecast_unavailable_cap,
                    "forecast required by selected path but unavailable", "path", 25, False,
                    input.purpose))

        # conflicts / disagreement
        for conflict in input.conflicts:
            cap_val = _clamp(1.0 - conflict.severity * (1.0 - p.disagreement_cap))
            caps.append(self._cap(ConfidenceCap.MODEL_DISAGREEMENT,
                                  min(cap_val, p.disagreement_cap) if conflict.severity >= 1.0
                                  else cap_val, conflict.reason or conflict.kind,
                                  "+".join(c.value for c in conflict.components), 65, False,
                                  input.purpose))
            reasons.append(ConfidenceReason.MODEL_DISAGREEMENT.value)

        # regime caps
        if input.regime is Regime.DISTURBED:
            hard = policy.control_relevant
            caps.append(self._cap(ConfidenceCap.DISTURBED, p.disturbed_cap, "disturbed regime",
                                  "regime", 110, hard, input.purpose))
            reasons.append(ConfidenceReason.DISTURBED.value)
        elif input.regime is Regime.UNKNOWN:
            caps.append(self._cap(ConfidenceCap.UNKNOWN_REGIME, p.unknown_regime_cap,
                                  "unknown regime", "regime", 70, False, input.purpose))
            reasons.append(ConfidenceReason.UNKNOWN_REGIME.value)

        # --- apply caps: strictest wins, all stay visible ---
        pre_cap = _clamp(base_value + positive_bonus)
        final_value = pre_cap
        for cap in caps:
            final_value = min(final_value, cap.max_value)
            reasons.append(ConfidenceReason.CAP_APPLIED.value)
        final_value = _clamp(final_value)

        if not input.components:
            reasons.append(ConfidenceReason.NO_CONTRIBUTIONS.value)

        level = self._level(final_value)
        gate = self._gate(input.purpose, policy, final_value, caps, missing_required,
                          fallback_used)
        result_status = self._status(caps, fallback_used, missing_required, input.components)

        breakdown = ConfidenceBreakdown(
            purpose=input.purpose, base_value=round(base_value, 6),
            components=tuple(component_breakdowns),
            required_components=tuple(k.value for k in required_kinds),
            optional_components=tuple(k.value for k in optional_kinds),
            missing_required=tuple(missing_required), missing_optional=tuple(missing_optional),
            positive_bonus=round(positive_bonus, 6), applied_caps=tuple(caps),
            conflicts=tuple(c.kind for c in input.conflicts),
            evidence_domains=tuple(sorted(d.value for d in seen_domains)),
            raw_evidence_count=raw_evidence, effective_evidence=round(effective_evidence, 4),
            prior_fraction=round(prior_fraction, 4), final_value=round(final_value, 6),
            final_reliability=round(agg_reliability, 6), level=level, gate=gate,
            aggregator_version=AGGREGATOR_VERSION, parameter_version=p.parameter_version,
            reason_codes=tuple(_dedup(reasons)))

        return ConfidenceResult(
            purpose=input.purpose, value=round(final_value, 6), level=level,
            status=result_status, breakdown=breakdown,
            required_components=breakdown.required_components,
            optional_components=breakdown.optional_components,
            missing_required=tuple(missing_required), missing_optional=tuple(missing_optional),
            applied_caps=tuple(caps), reason_codes=breakdown.reason_codes,
            evidence_count=raw_evidence, effective_evidence=round(effective_evidence, 4),
            reliability=round(agg_reliability, 6), fallback_used=fallback_used,
            aggregator_version=AGGREGATOR_VERSION, parameter_version=p.parameter_version,
            control_allowed=gate.allowed,
            advisory_allowed=final_value >= p.advisory_gate(input.purpose),
            gate=gate)

    # -- helpers --------------------------------------------------------

    def _validate(self, input: ConfidenceInput) -> None:
        for comp in input.components:
            v = comp.contribution.value
            if not math.isfinite(v):
                raise ValueError(f"non-finite contribution value for {comp.kind.value}")
            rel = comp.contribution.reliability
            if rel is not None and not math.isfinite(rel):
                raise ValueError(f"non-finite reliability for {comp.kind.value}")

    def _cap(self, cap: ConfidenceCap, max_value: float, reason: str, source: str,
             priority: int, hard: bool, purpose: ConfidencePurpose) -> AppliedCap:
        return AppliedCap(cap=cap, max_value=_clamp(max_value), reason=reason, source=source,
                          priority=priority, hard=hard, purpose=purpose)

    def _level(self, value: float) -> ConfidenceLevel:
        t = self._params.level_thresholds
        if value < t[0]:
            return ConfidenceLevel.VERY_LOW
        if value < t[1]:
            return ConfidenceLevel.LOW
        if value < t[2]:
            return ConfidenceLevel.MEDIUM
        if value < t[3]:
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.VERY_HIGH

    def _gate(self, purpose, policy, value, caps, missing_required, fallback_used):
        minimum = self._params.control_gate(purpose)
        hard_block = any(c.hard for c in caps)
        codes: list[str] = []
        allowed = policy.control_relevant and value >= minimum and not hard_block
        if not policy.control_relevant:
            codes.append("not_control_relevant")
        if hard_block:
            codes.append("hard_cap")
        if value < minimum:
            codes.append("below_minimum")
        safe_fallback = (not allowed) and policy.control_relevant
        if safe_fallback:
            codes.append("safe_fallback_required")
        return ControlGateResult(
            allowed=allowed, reason_codes=tuple(codes), required_minimum=minimum,
            actual_confidence=round(value, 6), safe_fallback_required=safe_fallback)

    def _status(self, caps, fallback_used, missing_required, components) -> ConfidenceStatusResult:
        if any(c.hard for c in caps):
            return ConfidenceStatusResult.BLOCKED
        if fallback_used:
            return ConfidenceStatusResult.FALLBACK
        if caps or missing_required:
            return ConfidenceStatusResult.DEGRADED
        return ConfidenceStatusResult.OK


def _kind_sort(comp: Optional[ConfidenceComponent]) -> str:
    return comp.kind.value if comp is not None else "~"


def _has_weather(input: ConfidenceInput) -> bool:
    for comp in input.components:
        if not comp.present:
            continue
        if comp.kind in (ComponentKind.OUTDOOR, ComponentKind.FORECAST):
            return True
        if EvidenceDomain.WEATHER in comp.domains or EvidenceDomain.FORECAST in comp.domains:
            return True
    return False


def _dedup(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -- diagnostics / export -----------------------------------------------------

def breakdown_to_support_export(result: ConfidenceResult) -> Mapping[str, object]:
    return {
        "purpose": result.purpose.value, "value": result.value, "level": result.level.value,
        "status": result.status.value,
        "components": [
            {"kind": c.kind.value, "required": c.is_required, "confidence": c.confidence,
             "reliability": c.reliability, "status": c.status}
            for c in result.breakdown.components],
        "applied_caps": [{"cap": c.cap.value, "max_value": c.max_value, "reason": c.reason,
                          "hard": c.hard, "priority": c.priority} for c in result.applied_caps],
        "missing_required": list(result.missing_required),
        "missing_optional": list(result.missing_optional),
        "gate": {"allowed": result.gate.allowed,
                 "required_minimum": result.gate.required_minimum,
                 "safe_fallback_required": result.gate.safe_fallback_required},
        "control_allowed": result.control_allowed, "advisory_allowed": result.advisory_allowed,
        "aggregator_version": result.aggregator_version,
        "parameter_version": result.parameter_version,
    }


def breakdown_to_research_export(result: ConfidenceResult) -> Mapping[str, object]:
    cap_counts: dict[str, int] = {}
    for c in result.applied_caps:
        cap_counts[c.cap.value] = cap_counts.get(c.cap.value, 0) + 1
    domain_counts: dict[str, int] = {}
    for d in result.breakdown.evidence_domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1
    return {
        "purpose": result.purpose.value, "level": result.level.value, "value": result.value,
        "component_kinds": sorted(c.kind.value for c in result.breakdown.components),
        "evidence_domain_distribution": domain_counts, "cap_frequencies": cap_counts,
        "effective_evidence": result.effective_evidence, "fallback_used": result.fallback_used,
        "aggregator_version": result.aggregator_version,
        "parameter_version": result.parameter_version,
    }
