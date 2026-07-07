"""ManualCorrectionModel for LE 2.0 (pure Python).

Learns from deliberate user corrections: direction (warmer/cooler), magnitude,
context and whether a prior ThermoSmart decision was systematically off. A single
correction is strong user evidence but NOT automatically a lasting comfort
preference — only recurring, consistent, well-attributed corrections produce a
(hard-capped) preference-bias recommendation. ThermoSmart's own commands, echoes
and device syncs are never learned as user corrections.

Consumes the existing ``ManualCorrectionEvent`` raw track unchanged (its
``event_id`` already provides stable identity). Richer intent / source-class
distinctions are derived in the model + typed context, not in the raw schema.
No HA / storage / coordinator dependency; imports no concrete model. Advisory:
no control effect, no autonomous config change.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ..contracts import (
    ConfidenceContribution,
    ExportScope,
    ModelRebuildResult,
    OutlierStatus,
    Prediction,
    PredictionType,
    Regime,
    RejectionReason,
    UpdateEligibilityResult,
)
from ..raw_schemas import CorrectionActionType, CorrectionSource, ManualCorrectionEvent, RawTrackName
from .base import clamp, freshness_from_age, is_finite, median, outlier_z, robust_ema_update

MODEL_VERSION = 1
PARAMETER_VERSION = 1
EVALUATION_VERSION = 1
MODEL_NAME = "manual_correction"


class ManualCorrectionRejection(Enum):
    WRONG_EVENT_TYPE = "wrong_event_type"
    WRONG_ZONE = "wrong_zone"
    DUPLICATE_EVENT = "duplicate_event"
    UNSUPPORTED_ACTION = "unsupported_action"
    INTERNAL_COMMAND_ECHO = "internal_command_echo"
    DEVICE_SYNC = "device_sync"
    INSUFFICIENT_ATTRIBUTION = "insufficient_attribution"
    ZERO_CORRECTION = "zero_correction"
    IMPLAUSIBLE_CORRECTION = "implausible_correction"
    NON_FINITE_VALUE = "non_finite_value"
    DECISION_MISMATCH = "decision_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    DISTURBED = "disturbed"
    MODE_CHANGE = "mode_change"
    TEMPORARY_EXCEPTION = "temporary_exception"
    VERSION_MISMATCH = "version_mismatch"
    STALE_CORRECTION = "stale_correction"
    MISSING_SOURCE = "missing_source"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    SEVERE_OUTLIER = "severe_outlier"


_TO_CONTRACT = {
    ManualCorrectionRejection.WRONG_EVENT_TYPE: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.WRONG_ZONE: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.DUPLICATE_EVENT: RejectionReason.DUPLICATE,
    ManualCorrectionRejection.UNSUPPORTED_ACTION: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.INTERNAL_COMMAND_ECHO: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.DEVICE_SYNC: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.INSUFFICIENT_ATTRIBUTION: RejectionReason.LOW_RELIABILITY,
    ManualCorrectionRejection.ZERO_CORRECTION: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.IMPLAUSIBLE_CORRECTION: RejectionReason.OUTLIER,
    ManualCorrectionRejection.NON_FINITE_VALUE: RejectionReason.OUTLIER,
    ManualCorrectionRejection.DECISION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.CONTEXT_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.DISTURBED: RejectionReason.DISTURBED_REGIME,
    ManualCorrectionRejection.MODE_CHANGE: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.TEMPORARY_EXCEPTION: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.VERSION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.STALE_CORRECTION: RejectionReason.NOT_ELIGIBLE,
    ManualCorrectionRejection.MISSING_SOURCE: RejectionReason.MISSING_REQUIRED_DATA,
    ManualCorrectionRejection.CONFLICTING_EVIDENCE: RejectionReason.LOW_RELIABILITY,
    ManualCorrectionRejection.SEVERE_OUTLIER: RejectionReason.OUTLIER,
}


class CorrectionDirection(Enum):
    WARMER = "warmer"
    COOLER = "cooler"
    NEUTRAL = "neutral"


class CorrectionSourceClass(Enum):
    DIRECT_USER = "direct_user"
    PHYSICAL_TRV = "physical_trv"
    EXTERNAL_AUTOMATION = "external_automation"
    INTERNAL_ECHO = "internal_echo"
    DEVICE_SYNC = "device_sync"
    SCHEDULE = "schedule"
    UNKNOWN = "unknown"


_DIRECT_CLASSES = frozenset({CorrectionSourceClass.DIRECT_USER, CorrectionSourceClass.PHYSICAL_TRV})
_LEARNABLE_CLASSES = _DIRECT_CLASSES | {CorrectionSourceClass.EXTERNAL_AUTOMATION,
                                        CorrectionSourceClass.UNKNOWN}


class CorrectionIntent(Enum):
    WARMER_PREFERENCE = "warmer_preference"
    COOLER_PREFERENCE = "cooler_preference"
    TOO_EARLY = "too_early"
    TOO_LATE = "too_late"
    OVERSHOOT_CORRECTION = "overshoot_correction"
    UNDERSHOOT_CORRECTION = "undershoot_correction"
    BOOST_TOO_STRONG = "boost_too_strong"
    BOOST_TOO_WEAK = "boost_too_weak"
    TEMPORARY_OVERRIDE = "temporary_override"
    MODE_CHANGE = "mode_change"
    SCHEDULE_EXCEPTION = "schedule_exception"
    UNKNOWN = "unknown"


_CONTROLLER_ERROR_INTENTS = frozenset({
    CorrectionIntent.TOO_EARLY, CorrectionIntent.TOO_LATE,
    CorrectionIntent.OVERSHOOT_CORRECTION, CorrectionIntent.UNDERSHOOT_CORRECTION,
    CorrectionIntent.BOOST_TOO_STRONG, CorrectionIntent.BOOST_TOO_WEAK})


@dataclass(frozen=True)
class ManualCorrectionParameters:
    min_correction_c: float = 0.2
    max_correction_c: float = 8.0
    max_preference_pos_c: float = 1.0
    max_preference_neg_c: float = -1.0
    max_change_per_update_c: float = 0.25
    min_corrections_for_bias: int = 5
    min_distinct_days_for_bias: int = 3
    max_revert_rate: float = 0.5
    min_attribution_reliability: float = 0.3
    max_age_days: float = 60.0
    decision_attribution_window_s: float = 3600.0
    echo_dedup_window_s: float = 120.0
    quick_revert_window_s: float = 900.0
    max_sample_weight: float = 1.0
    automation_weight_factor: float = 0.3
    unknown_weight_factor: float = 0.5
    bucket_min_samples: int = 5
    bucket_seed_evidence: float = 1.0
    outlier_mad_k_mild: float = 3.0
    outlier_mad_k_severe: float = 6.0
    min_samples_for_outlier: int = 5
    dedup_max_ids: int = 500
    research_sample_cap: int = 50
    full_confidence_samples: float = 12.0
    cold_start_confidence_cap: float = 0.3
    parameter_version: int = PARAMETER_VERSION


# -- evaluation types ---------------------------------------------------------

@dataclass(frozen=True)
class CorrectionAttribution:
    source_class: CorrectionSourceClass
    reliability: float
    decision_bound: bool
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("attribution reliability must be within [0,1]")

    @property
    def is_direct_user(self) -> bool:
        return self.source_class in _DIRECT_CLASSES


@dataclass(frozen=True)
class ManualCorrectionPattern:
    persistence_likelihood: float
    reverted: bool
    recurring: bool
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.persistence_likelihood <= 1.0:
            raise ValueError("persistence_likelihood must be within [0,1]")


@dataclass(frozen=True)
class ManualCorrectionEvaluation:
    direction: CorrectionDirection
    magnitude_c: float
    signed_magnitude_c: float
    time_since_decision_s: Optional[float]
    intent: CorrectionIntent
    attribution: CorrectionAttribution
    pattern: ManualCorrectionPattern
    comfort_preference_evidence: float
    controller_error_evidence: float
    data_quality: float
    has_temperature: bool
    confounders: tuple[str, ...]
    evaluation_version: int = EVALUATION_VERSION
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("magnitude_c", "signed_magnitude_c", "comfort_preference_evidence",
                     "controller_error_evidence", "data_quality"):
            v = getattr(self, name)
            if not is_finite(v):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class CorrectionRecommendation:
    """Advisory controller-calibration signal (no model mutation, no control)."""
    intent: CorrectionIntent
    strength: float
    reliability: float
    decision_type: Optional[str]
    supporting_evidence: int
    reason_codes: tuple[str, ...]
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be within [0,1]")


@dataclass(frozen=True)
class ManualCorrectionUpdateContext:
    correction_event_id: str
    learning_zone_id: Optional[str] = None
    decision_id: Optional[str] = None
    outcome_episode_id: Optional[str] = None
    previous_thermosmart_target_c: Optional[float] = None
    time_since_decision_s: Optional[float] = None
    controller_kind: Optional[str] = None
    schedule_period: Optional[str] = None
    decision_type: Optional[str] = None        # normal / preheat / boost
    in_boost: bool = False
    in_preheat: bool = False
    current_temperature_c: Optional[float] = None
    target_temperature_c: Optional[float] = None
    outcome_overshoot_c: Optional[float] = None
    outcome_undershoot_c: Optional[float] = None
    expected_afterheat_rise_c: Optional[float] = None
    source_class_hint: Optional[str] = None
    is_internal_echo: bool = False
    is_device_sync: bool = False
    physical_trv: bool = False
    last_sent_setpoint_c: Optional[float] = None
    reverted_within_s: Optional[float] = None
    attribution_reliability: Optional[float] = None
    context_quality: float = 1.0
    source_refs: tuple[str, ...] = ()
    source_feature_version: int = 1

    def __post_init__(self) -> None:
        if not self.correction_event_id:
            raise ValueError("correction_event_id must be non-empty")
        for name in ("previous_thermosmart_target_c", "time_since_decision_s",
                     "current_temperature_c", "target_temperature_c", "outcome_overshoot_c",
                     "outcome_undershoot_c", "expected_afterheat_rise_c",
                     "last_sent_setpoint_c", "reverted_within_s", "attribution_reliability",
                     "context_quality"):
            v = getattr(self, name)
            if v is not None and not is_finite(v):
                raise ValueError(f"{name} must be finite (or None)")

    def to_dict(self) -> dict:
        return {
            "correction_event_id": self.correction_event_id,
            "learning_zone_id": self.learning_zone_id, "decision_id": self.decision_id,
            "outcome_episode_id": self.outcome_episode_id,
            "previous_thermosmart_target_c": self.previous_thermosmart_target_c,
            "time_since_decision_s": self.time_since_decision_s,
            "controller_kind": self.controller_kind, "schedule_period": self.schedule_period,
            "decision_type": self.decision_type, "in_boost": self.in_boost,
            "in_preheat": self.in_preheat, "current_temperature_c": self.current_temperature_c,
            "target_temperature_c": self.target_temperature_c,
            "outcome_overshoot_c": self.outcome_overshoot_c,
            "outcome_undershoot_c": self.outcome_undershoot_c,
            "expected_afterheat_rise_c": self.expected_afterheat_rise_c,
            "source_class_hint": self.source_class_hint, "is_internal_echo": self.is_internal_echo,
            "is_device_sync": self.is_device_sync, "physical_trv": self.physical_trv,
            "last_sent_setpoint_c": self.last_sent_setpoint_c,
            "reverted_within_s": self.reverted_within_s,
            "attribution_reliability": self.attribution_reliability,
            "context_quality": self.context_quality, "source_refs": list(self.source_refs),
            "source_feature_version": self.source_feature_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ManualCorrectionUpdateContext":
        return cls(
            correction_event_id=d["correction_event_id"],
            learning_zone_id=d.get("learning_zone_id"), decision_id=d.get("decision_id"),
            outcome_episode_id=d.get("outcome_episode_id"),
            previous_thermosmart_target_c=d.get("previous_thermosmart_target_c"),
            time_since_decision_s=d.get("time_since_decision_s"),
            controller_kind=d.get("controller_kind"), schedule_period=d.get("schedule_period"),
            decision_type=d.get("decision_type"), in_boost=d.get("in_boost", False),
            in_preheat=d.get("in_preheat", False),
            current_temperature_c=d.get("current_temperature_c"),
            target_temperature_c=d.get("target_temperature_c"),
            outcome_overshoot_c=d.get("outcome_overshoot_c"),
            outcome_undershoot_c=d.get("outcome_undershoot_c"),
            expected_afterheat_rise_c=d.get("expected_afterheat_rise_c"),
            source_class_hint=d.get("source_class_hint"),
            is_internal_echo=d.get("is_internal_echo", False),
            is_device_sync=d.get("is_device_sync", False),
            physical_trv=d.get("physical_trv", False),
            last_sent_setpoint_c=d.get("last_sent_setpoint_c"),
            reverted_within_s=d.get("reverted_within_s"),
            attribution_reliability=d.get("attribution_reliability"),
            context_quality=d.get("context_quality", 1.0),
            source_refs=tuple(d.get("source_refs", [])),
            source_feature_version=d.get("source_feature_version", 1))


@dataclass(frozen=True)
class ManualCorrectionRebuildItem:
    event: ManualCorrectionEvent
    context: Optional[ManualCorrectionUpdateContext] = None


# -- state --------------------------------------------------------------------

@dataclass(frozen=True)
class _Dim:
    value: float = 0.0
    effective_n: float = 0.0
    dispersion: float = 0.0
    sample_count: int = 0

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0 and self.effective_n > 0.0


def _apply_dim(dim: _Dim, sample: float, weight: float) -> _Dim:
    upd = robust_ema_update(dim.value, dim.effective_n, dim.dispersion, sample, weight)
    return _Dim(value=upd.value, effective_n=upd.effective_n, dispersion=upd.dispersion,
                sample_count=dim.sample_count + 1)


def _dim_dict(d: _Dim) -> dict:
    return {"value": d.value, "effective_n": d.effective_n, "dispersion": d.dispersion,
            "sample_count": d.sample_count}


def _dim_from(d: Mapping[str, Any]) -> _Dim:
    return _Dim(value=d["value"], effective_n=d["effective_n"], dispersion=d["dispersion"],
                sample_count=d["sample_count"])


@dataclass(frozen=True)
class ManualCorrectionBucket:
    key: str
    signed_magnitude: _Dim = field(default_factory=_Dim)
    warmer_count: int = 0
    cooler_count: int = 0
    sample_count: int = 0

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0


def _bucket_dict(b: ManualCorrectionBucket) -> dict:
    return {"key": b.key, "signed_magnitude": _dim_dict(b.signed_magnitude),
            "warmer_count": b.warmer_count, "cooler_count": b.cooler_count,
            "sample_count": b.sample_count}


def _bucket_from(d: Mapping[str, Any]) -> ManualCorrectionBucket:
    return ManualCorrectionBucket(
        key=d["key"], signed_magnitude=_dim_from(d["signed_magnitude"]),
        warmer_count=d["warmer_count"], cooler_count=d["cooler_count"],
        sample_count=d["sample_count"])


@dataclass(frozen=True)
class ManualCorrectionSample:
    correction_event_id: str
    decision_id: Optional[str]
    learning_zone_id: str
    direction: str
    signed_magnitude_c: float
    source_class: str
    intent: str
    attribution_reliability: float
    reverted: bool
    day: str
    effective_weight: float
    evaluation_version: int


@dataclass(frozen=True)
class ManualCorrectionState:
    learning_zone_id: str
    general: ManualCorrectionBucket
    buckets: Mapping[str, ManualCorrectionBucket] = field(default_factory=dict)
    preference_bias_c: float = 0.0
    processed_ids: tuple[str, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()
    recent_samples: tuple[ManualCorrectionSample, ...] = ()
    distinct_days: tuple[str, ...] = ()
    direct_count: int = 0
    automation_count: int = 0
    unknown_count: int = 0
    revert_count: int = 0
    learnable_count: int = 0
    intent_counts: Mapping[str, int] = field(default_factory=dict)
    last_update_ts: Optional[str] = None
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION
    evaluation_version: int = EVALUATION_VERSION


@dataclass(frozen=True)
class ManualCorrectionDiagnostics:
    preference_bias_c: float
    confidence: float
    warmer_count: int
    cooler_count: int
    direct_count: int
    automation_count: int
    unknown_count: int
    attribution_reliability: Optional[float]
    revert_rate: Optional[float]
    intent_distribution: Mapping[str, int]
    bucket_summaries: Mapping[str, float]
    distinct_days: int
    effective_evidence: float
    missing_evidence: tuple[str, ...]
    rejection_counts: Mapping[str, int]
    outlier_counts: Mapping[str, int]
    last_update_ts: Optional[str]
    model_version: int
    parameter_version: int
    evaluation_version: int


@dataclass(frozen=True)
class ManualCorrectionPredictionContext:
    decision_type: Optional[str] = None
    schedule_period: Optional[str] = None


@dataclass(frozen=True)
class CorrectionPrediction:
    preference_bias_c: float
    confidence: float
    reliability: float
    evidence_count: int
    effective_evidence: float
    direct_user_fraction: float
    revert_rate: float
    fallback_used: bool
    bucket: str
    missing_evidence: tuple[str, ...]
    reason_codes: tuple[str, ...]
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION

    def __post_init__(self) -> None:
        if not is_finite(self.preference_bias_c):
            raise ValueError("preference_bias_c must be finite")


def _bucket_key(intent: CorrectionIntent, decision_type: Optional[str],
                schedule_period: Optional[str]) -> Optional[str]:
    parts = [f"intent:{intent.value}"]
    if decision_type:
        parts.append(f"dt:{decision_type}")
    if schedule_period:
        parts.append(f"sch:{schedule_period}")
    return "|".join(parts)


# -- model --------------------------------------------------------------------

class ManualCorrectionModel:
    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    parameter_version = PARAMETER_VERSION
    consumed_raw_tracks = (RawTrackName.MANUAL_CORRECTION.value,)
    consumed_episode_types: tuple[str, ...] = ()
    required_features = ("target",)
    optional_features = ("indoor_temp", "outdoor_temp", "forecast")
    supported_prediction_types = (
        PredictionType.MANUAL_PREFERENCE_BIAS, PredictionType.CONTROLLER_CALIBRATION)
    min_tier = 0

    def __init__(self, learning_zone_id: str,
                 params: Optional[ManualCorrectionParameters] = None) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or ManualCorrectionParameters()
        self._state = self._empty_state()

    def cold_start_prior(self) -> Mapping[str, Any]:
        return {"preference_bias_c": 0.0}

    def _empty_state(self) -> ManualCorrectionState:
        return ManualCorrectionState(learning_zone_id=self._zone,
                                     general=ManualCorrectionBucket(key="general"))

    # -- deterministic evaluation (pure) --------------------------------

    def _classify_source(self, event: ManualCorrectionEvent,
                         context: Optional[ManualCorrectionUpdateContext]) -> CorrectionSourceClass:
        if context is not None:
            if context.is_internal_echo:
                return CorrectionSourceClass.INTERNAL_ECHO
            if context.is_device_sync:
                return CorrectionSourceClass.DEVICE_SYNC
            if context.source_class_hint:
                try:
                    return CorrectionSourceClass(context.source_class_hint)
                except ValueError:
                    pass
            # echo detection from the last sent setpoint
            if context.last_sent_setpoint_c is not None and is_finite(event.new_target) \
                    and abs(context.last_sent_setpoint_c - event.new_target) < 1e-6:
                return CorrectionSourceClass.INTERNAL_ECHO
        if event.source is CorrectionSource.AUTOMATION:
            return CorrectionSourceClass.EXTERNAL_AUTOMATION
        if event.source is CorrectionSource.EXTERNAL:
            return CorrectionSourceClass.EXTERNAL_AUTOMATION
        if event.source is CorrectionSource.UI:
            if context is not None and context.physical_trv:
                return CorrectionSourceClass.PHYSICAL_TRV
            return CorrectionSourceClass.DIRECT_USER
        return CorrectionSourceClass.UNKNOWN

    def _attribution(self, source_class, context, has_temp) -> CorrectionAttribution:
        base = {
            CorrectionSourceClass.DIRECT_USER: 0.9,
            CorrectionSourceClass.PHYSICAL_TRV: 0.85,
            CorrectionSourceClass.EXTERNAL_AUTOMATION: 0.4,
            CorrectionSourceClass.SCHEDULE: 0.4,
            CorrectionSourceClass.UNKNOWN: 0.3,
            CorrectionSourceClass.INTERNAL_ECHO: 0.0,
            CorrectionSourceClass.DEVICE_SYNC: 0.0,
        }[source_class]
        reasons = [source_class.value]
        decision_bound = context is not None and context.decision_id is not None
        if context is not None and context.attribution_reliability is not None:
            base = min(base, context.attribution_reliability)
            reasons.append("context_attribution")
        if decision_bound and context.time_since_decision_s is not None \
                and context.time_since_decision_s <= self._params.decision_attribution_window_s:
            base = min(1.0, base * 1.1)
            reasons.append("decision_bound")
        if context is not None:
            base *= clamp(context.context_quality, 0.0, 1.0)
        return CorrectionAttribution(source_class=source_class, reliability=clamp(base, 0.0, 1.0),
                                     decision_bound=decision_bound, reason_codes=tuple(reasons))

    def _classify_intent(self, event, context, direction, signed) -> CorrectionIntent:
        if event.action_type is CorrectionActionType.MODE_CHANGE:
            return CorrectionIntent.MODE_CHANGE
        if context is None:
            return (CorrectionIntent.WARMER_PREFERENCE if direction is CorrectionDirection.WARMER
                    else CorrectionIntent.COOLER_PREFERENCE)
        if context.reverted_within_s is not None \
                and context.reverted_within_s <= self._params.quick_revert_window_s:
            return CorrectionIntent.TEMPORARY_OVERRIDE
        if event.action_type is CorrectionActionType.OVERRIDE and context.schedule_period:
            return CorrectionIntent.SCHEDULE_EXCEPTION
        attributable = (context.time_since_decision_s is not None
                        and context.time_since_decision_s
                        <= self._params.decision_attribution_window_s)
        if context.in_boost:
            return (CorrectionIntent.BOOST_TOO_WEAK if direction is CorrectionDirection.WARMER
                    else CorrectionIntent.BOOST_TOO_STRONG)
        if direction is CorrectionDirection.COOLER and context.outcome_overshoot_c \
                and context.outcome_overshoot_c > 0:
            return CorrectionIntent.OVERSHOOT_CORRECTION
        if direction is CorrectionDirection.WARMER and context.outcome_undershoot_c \
                and context.outcome_undershoot_c > 0:
            return CorrectionIntent.UNDERSHOOT_CORRECTION
        if attributable and direction is CorrectionDirection.WARMER:
            if context.in_preheat:
                return CorrectionIntent.TOO_LATE
            return CorrectionIntent.WARMER_PREFERENCE
        if attributable and direction is CorrectionDirection.COOLER:
            return CorrectionIntent.COOLER_PREFERENCE
        # far from any decision -> weak attribution, treat as preference/override
        if direction is CorrectionDirection.WARMER:
            return CorrectionIntent.WARMER_PREFERENCE
        if direction is CorrectionDirection.COOLER:
            return CorrectionIntent.COOLER_PREFERENCE
        return CorrectionIntent.UNKNOWN

    def evaluate_correction(self, event: ManualCorrectionEvent,
                            context: Optional[ManualCorrectionUpdateContext]
                            ) -> ManualCorrectionEvaluation:
        p = self._params
        signed = event.new_target - event.prior_target
        magnitude = abs(signed)
        direction = (CorrectionDirection.WARMER if signed > 0 else
                     CorrectionDirection.COOLER if signed < 0 else CorrectionDirection.NEUTRAL)
        source_class = self._classify_source(event, context)
        has_temp = context is not None and context.current_temperature_c is not None
        attribution = self._attribution(source_class, context, has_temp)
        intent = self._classify_intent(event, context, direction, signed)

        reverted = (context is not None and context.reverted_within_s is not None
                    and context.reverted_within_s <= p.quick_revert_window_s)
        # persistence: single event is weak; recurrence is judged by the learner
        persistence = 0.0 if reverted else clamp(attribution.reliability * 0.5, 0.0, 0.5)
        pattern = ManualCorrectionPattern(
            persistence_likelihood=persistence, reverted=reverted, recurring=False,
            reason_codes=("reverted",) if reverted else ())

        comfort_pref = 0.0
        controller_err = 0.0
        if attribution.is_direct_user and not reverted \
                and intent in (CorrectionIntent.WARMER_PREFERENCE,
                               CorrectionIntent.COOLER_PREFERENCE):
            comfort_pref = attribution.reliability
        if intent in _CONTROLLER_ERROR_INTENTS:
            controller_err = attribution.reliability

        confounders = []
        if reverted:
            confounders.append("quick_revert")
        if not has_temp:
            confounders.append("no_indoor_temp")
        data_quality = attribution.reliability * (1.0 if has_temp else 0.7)

        return ManualCorrectionEvaluation(
            direction=direction, magnitude_c=magnitude, signed_magnitude_c=signed,
            time_since_decision_s=context.time_since_decision_s if context else None,
            intent=intent, attribution=attribution, pattern=pattern,
            comfort_preference_evidence=comfort_pref, controller_error_evidence=controller_err,
            data_quality=clamp(data_quality, 0.0, 1.0), has_temperature=has_temp,
            confounders=tuple(confounders),
            reason_codes=(direction.value, intent.value, source_class.value))

    # -- eligibility ----------------------------------------------------

    def update_eligibility(self, event: Any) -> UpdateEligibilityResult:
        return self._eligibility(event, None)

    def _eligibility(self, event, context) -> UpdateEligibilityResult:
        p = self._params
        if not isinstance(event, ManualCorrectionEvent):
            return self._reject(ManualCorrectionRejection.WRONG_EVENT_TYPE)
        if event.learning_zone_id != self._zone:
            return self._reject(ManualCorrectionRejection.WRONG_ZONE)
        if context is not None:
            if context.correction_event_id != event.event_id or (
                    context.learning_zone_id is not None
                    and context.learning_zone_id != self._zone):
                return self._reject(ManualCorrectionRejection.CONTEXT_MISMATCH)
        if not is_finite(event.prior_target) or not is_finite(event.new_target):
            return self._reject(ManualCorrectionRejection.NON_FINITE_VALUE)
        if event.event_id in self._state.processed_ids:
            return self._reject(ManualCorrectionRejection.DUPLICATE_EVENT)

        source_class = self._classify_source(event, context)
        if source_class is CorrectionSourceClass.INTERNAL_ECHO:
            return self._reject(ManualCorrectionRejection.INTERNAL_COMMAND_ECHO)
        if source_class is CorrectionSourceClass.DEVICE_SYNC:
            return self._reject(ManualCorrectionRejection.DEVICE_SYNC)

        signed = event.new_target - event.prior_target
        magnitude = abs(signed)
        if magnitude < p.min_correction_c:
            return self._reject(ManualCorrectionRejection.ZERO_CORRECTION)
        if magnitude > p.max_correction_c:
            return self._reject(ManualCorrectionRejection.IMPLAUSIBLE_CORRECTION)
        if event.action_type is CorrectionActionType.MODE_CHANGE:
            return self._reject(ManualCorrectionRejection.MODE_CHANGE)

        ev = self.evaluate_correction(event, context)
        if ev.pattern.reverted and ev.intent is CorrectionIntent.TEMPORARY_OVERRIDE:
            return self._reject(ManualCorrectionRejection.TEMPORARY_EXCEPTION)
        if ev.attribution.reliability < p.min_attribution_reliability:
            return self._reject(ManualCorrectionRejection.INSUFFICIENT_ATTRIBUTION)

        weight = clamp(ev.attribution.reliability, 0.0, p.max_sample_weight)
        if ev.attribution.source_class is CorrectionSourceClass.EXTERNAL_AUTOMATION:
            weight *= p.automation_weight_factor
        elif ev.attribution.source_class is CorrectionSourceClass.UNKNOWN:
            weight *= p.unknown_weight_factor
        return UpdateEligibilityResult(
            accepted=True, reliability=ev.attribution.reliability, weight=weight,
            outlier_status=OutlierStatus.NONE, regime=Regime.UNKNOWN,
            source_episode_id=event.event_id)

    def _reject(self, reason: ManualCorrectionRejection) -> UpdateEligibilityResult:
        return UpdateEligibilityResult(
            accepted=False, reliability=0.0, weight=0.0,
            rejection_reason=_TO_CONTRACT[reason], confounder_flags=(reason.value,))

    def _bump_rejection(self, code: str) -> None:
        rc = dict(self._state.rejection_counts)
        rc[code] = rc.get(code, 0) + 1
        self._state = replace(self._state, rejection_counts=rc)

    def _bump_outlier(self, kind: str) -> None:
        oc = dict(self._state.outlier_counts)
        oc[kind] = oc.get(kind, 0) + 1
        self._state = replace(self._state, outlier_counts=oc)

    # -- update ---------------------------------------------------------

    def update(self, event: Any,
               context: Optional[ManualCorrectionUpdateContext] = None) -> UpdateEligibilityResult:
        elig = self._eligibility(event, context)
        if not elig.accepted:
            code = elig.confounder_flags[0] if elig.confounder_flags else "unknown"
            self._bump_rejection(code)
            return elig

        ev = self.evaluate_correction(event, context)
        weight = elig.weight
        g = self._state.general
        if g.signed_magnitude.sample_count >= self._params.min_samples_for_outlier \
                and g.signed_magnitude.has_evidence:
            z = outlier_z(ev.signed_magnitude_c, g.signed_magnitude.value,
                          g.signed_magnitude.dispersion)
            if z > self._params.outlier_mad_k_severe:
                self._bump_outlier("severe")
                self._bump_rejection(ManualCorrectionRejection.SEVERE_OUTLIER.value)
                return self._reject(ManualCorrectionRejection.SEVERE_OUTLIER)
            if z > self._params.outlier_mad_k_mild:
                weight *= 0.25
                self._bump_outlier("mild")

        warmer = ev.direction is CorrectionDirection.WARMER
        cooler = ev.direction is CorrectionDirection.COOLER
        new_general = ManualCorrectionBucket(
            key="general", signed_magnitude=_apply_dim(g.signed_magnitude, ev.signed_magnitude_c, weight),
            warmer_count=g.warmer_count + (1 if warmer else 0),
            cooler_count=g.cooler_count + (1 if cooler else 0),
            sample_count=g.sample_count + 1)

        buckets = dict(self._state.buckets)
        dt = context.decision_type if context else None
        sch = context.schedule_period if context else None
        bkey = _bucket_key(ev.intent, dt, sch)
        b = buckets.get(bkey) or ManualCorrectionBucket(key=bkey)
        buckets[bkey] = ManualCorrectionBucket(
            key=bkey, signed_magnitude=_apply_dim(b.signed_magnitude, ev.signed_magnitude_c, weight),
            warmer_count=b.warmer_count + (1 if warmer else 0),
            cooler_count=b.cooler_count + (1 if cooler else 0),
            sample_count=b.sample_count + 1)

        is_direct = ev.attribution.is_direct_user
        is_auto = ev.attribution.source_class is CorrectionSourceClass.EXTERNAL_AUTOMATION
        is_unknown = ev.attribution.source_class is CorrectionSourceClass.UNKNOWN
        learnable = is_direct and not ev.pattern.reverted
        day = _day(event.ts)
        distinct_days = self._state.distinct_days
        if learnable and day not in distinct_days:
            distinct_days = (distinct_days + (day,))[-366:]

        intent_counts = dict(self._state.intent_counts)
        intent_counts[ev.intent.value] = intent_counts.get(ev.intent.value, 0) + 1

        sample = ManualCorrectionSample(
            correction_event_id=event.event_id,
            decision_id=context.decision_id if context else None, learning_zone_id=self._zone,
            direction=ev.direction.value, signed_magnitude_c=ev.signed_magnitude_c,
            source_class=ev.attribution.source_class.value, intent=ev.intent.value,
            attribution_reliability=ev.attribution.reliability, reverted=ev.pattern.reverted,
            day=day, effective_weight=weight, evaluation_version=EVALUATION_VERSION)

        new_revert = self._state.revert_count + (1 if ev.pattern.reverted else 0)
        new_learnable = self._state.learnable_count + (1 if learnable else 0)
        processed = (self._state.processed_ids + (event.event_id,))[-self._params.dedup_max_ids:]
        processed_dec = self._state.processed_decision_ids
        if context is not None and context.decision_id is not None:
            processed_dec = (processed_dec + (context.decision_id,))[-self._params.dedup_max_ids:]
        recent = (self._state.recent_samples + (sample,))[-self._params.research_sample_cap:]

        bias = self._compute_bias(new_general, new_learnable, distinct_days, new_revert,
                                  recent, self._state.preference_bias_c)

        self._state = replace(
            self._state, general=new_general, buckets=buckets, preference_bias_c=bias,
            processed_ids=processed, processed_decision_ids=processed_dec,
            recent_samples=recent, distinct_days=distinct_days,
            direct_count=self._state.direct_count + (1 if is_direct else 0),
            automation_count=self._state.automation_count + (1 if is_auto else 0),
            unknown_count=self._state.unknown_count + (1 if is_unknown else 0),
            revert_count=new_revert, learnable_count=new_learnable,
            intent_counts=intent_counts, last_update_ts=_iso(event.ts))
        return elig

    def _compute_bias(self, general, learnable_count, distinct_days, revert_count, recent,
                      prev_bias) -> float:
        p = self._params
        n = general.sample_count
        revert_rate = revert_count / n if n > 0 else 0.0
        if (learnable_count < p.min_corrections_for_bias
                or len(distinct_days) < p.min_distinct_days_for_bias
                or revert_rate > p.max_revert_rate):
            return prev_bias if abs(prev_bias) > 0 and learnable_count >= p.min_corrections_for_bias \
                else 0.0
        # robust target from direct, non-reverted samples
        direct = [s.signed_magnitude_c for s in recent
                  if s.source_class in (CorrectionSourceClass.DIRECT_USER.value,
                                        CorrectionSourceClass.PHYSICAL_TRV.value)
                  and not s.reverted]
        if len(direct) < p.min_corrections_for_bias:
            return 0.0
        target = clamp(median(direct), p.max_preference_neg_c, p.max_preference_pos_c)
        # limit per-update change
        delta = clamp(target - prev_bias, -p.max_change_per_update_c, p.max_change_per_update_c)
        return clamp(prev_bias + delta, p.max_preference_neg_c, p.max_preference_pos_c)

    # -- predictions ----------------------------------------------------

    def predict_correction(self, context: ManualCorrectionPredictionContext) -> CorrectionPrediction:
        p = self._params
        g = self._state.general
        n = g.sample_count
        revert_rate = self._state.revert_count / n if n > 0 else 0.0
        direct_fraction = self._state.direct_count / n if n > 0 else 0.0
        belastbar = (self._state.learnable_count >= p.min_corrections_for_bias
                     and len(self._state.distinct_days) >= p.min_distinct_days_for_bias
                     and revert_rate <= p.max_revert_rate)
        if not belastbar:
            return CorrectionPrediction(
                preference_bias_c=0.0,
                confidence=self._confidence_value(0.0, True),
                reliability=0.0, evidence_count=n, effective_evidence=g.signed_magnitude.effective_n,
                direct_user_fraction=direct_fraction, revert_rate=revert_rate,
                fallback_used=True, bucket="cold_start",
                missing_evidence=("insufficient_recurring_user_evidence",),
                reason_codes=("cold_start",))
        eff = g.signed_magnitude.effective_n
        return CorrectionPrediction(
            preference_bias_c=round(self._state.preference_bias_c, 4),
            confidence=self._confidence_value(eff, False),
            reliability=clamp(eff / p.full_confidence_samples, 0.0, 1.0),
            evidence_count=n, effective_evidence=eff, direct_user_fraction=direct_fraction,
            revert_rate=revert_rate, fallback_used=False, bucket="general",
            missing_evidence=(), reason_codes=("learned_preference",))

    def predict_preference_bias(self, context: ManualCorrectionPredictionContext) -> CorrectionPrediction:
        return self.predict_correction(context)

    def predict(self, context: Any) -> Prediction:
        ctx = context if isinstance(context, ManualCorrectionPredictionContext) \
            else ManualCorrectionPredictionContext()
        rec = self.predict_correction(ctx)
        return Prediction(
            prediction_type=PredictionType.MANUAL_PREFERENCE_BIAS,
            values={"preference_bias": rec.preference_bias_c},
            units={"preference_bias": "celsius"}, confidence=rec.confidence,
            reliability=rec.reliability, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION,
            prior_contribution=1.0 if rec.fallback_used else 0.0,
            learned_contribution=0.0 if rec.fallback_used else 1.0,
            fallback_used=rec.fallback_used, evidence_count=rec.evidence_count, bucket=rec.bucket,
            confidence_cap=self._params.cold_start_confidence_cap if rec.fallback_used else None,
            cap_reasons=("cold_start",) if rec.fallback_used else (),
            missing_evidence=rec.missing_evidence, reason_codes=rec.reason_codes)

    def controller_calibration_signal(self, context: Optional[ManualCorrectionPredictionContext]
                                      = None) -> Optional[CorrectionRecommendation]:
        """Advisory calibration hint from recurring controller-error intents."""
        p = self._params
        ce = {k: v for k, v in self._state.intent_counts.items()
              if k in {i.value for i in _CONTROLLER_ERROR_INTENTS}}
        if not ce:
            return None
        intent_value, count = max(ce.items(), key=lambda kv: kv[1])
        if count < p.min_corrections_for_bias:
            return None
        total = max(self._state.general.sample_count, 1)
        return CorrectionRecommendation(
            intent=CorrectionIntent(intent_value), strength=clamp(count / total, 0.0, 1.0),
            reliability=clamp(count / p.full_confidence_samples, 0.0, 1.0),
            decision_type=context.decision_type if context else None,
            supporting_evidence=count, reason_codes=("recurring_controller_error",))

    # -- confidence -----------------------------------------------------

    def _confidence_value(self, effective_evidence: float, fallback: bool) -> float:
        p = self._params
        if fallback or effective_evidence <= 0:
            return min(0.15, p.cold_start_confidence_cap)
        coverage = min(effective_evidence / p.full_confidence_samples, 1.0)
        n = self._state.general.sample_count
        consistency = abs(self._state.general.warmer_count - self._state.general.cooler_count) \
            / n if n > 0 else 0.0
        direct_fraction = self._state.direct_count / n if n > 0 else 0.0
        revert_rate = self._state.revert_count / n if n > 0 else 0.0
        days = min(len(self._state.distinct_days) / p.min_distinct_days_for_bias, 1.0)
        return clamp(coverage * (0.4 + 0.6 * consistency) * (0.4 + 0.6 * direct_fraction)
                     * (1.0 - 0.5 * revert_rate) * (0.5 + 0.5 * days), 0.0, 1.0)

    def confidence(self, *, now: Optional[str] = None) -> ConfidenceContribution:
        g = self._state.general
        n = g.sample_count
        reasons: list[str] = []
        if not g.has_evidence:
            reasons.append("cold_start")
        direct_fraction = self._state.direct_count / n if n > 0 else 0.0
        if n > 0 and direct_fraction < 0.5:
            reasons.append("mostly_automation_or_unknown")
        revert_rate = self._state.revert_count / n if n > 0 else 0.0
        if revert_rate > 0.3:
            reasons.append("high_revert_rate")
        fallback = not g.has_evidence
        value = self._confidence_value(g.signed_magnitude.effective_n, fallback)
        freshness, staleness = freshness_from_age(now, self._state.last_update_ts)
        status = "healthy" if g.has_evidence else "fallback"
        if g.has_evidence and staleness == "stale":
            status = "stale"
        return ConfidenceContribution(
            value=value, evidence_count=n, reasons=tuple(reasons),
            component="manual_correction",
            reliability=clamp(g.signed_magnitude.effective_n / self._params.full_confidence_samples,
                              0.0, 1.0),
            evidence_domains=("user_correction", "controller_events"),
            source_count=len(self._state.buckets),
            prior_fraction=1.0 if fallback else 0.0,
            learned_fraction=0.0 if fallback else 1.0, fallback_used=fallback, freshness=freshness,
            status=status)

    def _attribution_mean(self) -> Optional[float]:
        if not self._state.recent_samples:
            return None
        vals = [s.attribution_reliability for s in self._state.recent_samples]
        return sum(vals) / len(vals)

    # -- diagnostics / export -------------------------------------------

    def diagnostics(self) -> ManualCorrectionDiagnostics:
        s = self._state
        g = s.general
        n = g.sample_count
        return ManualCorrectionDiagnostics(
            preference_bias_c=s.preference_bias_c, confidence=self.confidence().value,
            warmer_count=g.warmer_count, cooler_count=g.cooler_count,
            direct_count=s.direct_count, automation_count=s.automation_count,
            unknown_count=s.unknown_count, attribution_reliability=self._attribution_mean(),
            revert_rate=(s.revert_count / n) if n > 0 else None,
            intent_distribution=dict(s.intent_counts),
            bucket_summaries={k: b.signed_magnitude.value for k, b in s.buckets.items()},
            distinct_days=len(s.distinct_days), effective_evidence=g.signed_magnitude.effective_n,
            missing_evidence=("recurring_user_evidence",)
            if s.learnable_count < self._params.min_corrections_for_bias else (),
            rejection_counts=dict(s.rejection_counts), outlier_counts=dict(s.outlier_counts),
            last_update_ts=s.last_update_ts, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION, evaluation_version=EVALUATION_VERSION)

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        s = self._state
        g = s.general
        n = g.sample_count
        if scope is ExportScope.SUPPORT:
            return {
                "preference_bias_c": s.preference_bias_c, "confidence": self.confidence().value,
                "warmer_count": g.warmer_count, "cooler_count": g.cooler_count,
                "direct_count": s.direct_count, "automation_count": s.automation_count,
                "unknown_count": s.unknown_count,
                "revert_rate": (s.revert_count / n) if n > 0 else None,
                "intent_distribution": dict(s.intent_counts), "sample_count": n,
                "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
                "evaluation_version": EVALUATION_VERSION,
            }
        if scope is ExportScope.RESEARCH:
            return {
                "samples": [
                    {"direction": x.direction, "signed_magnitude_c": x.signed_magnitude_c,
                     "source_class": x.source_class, "intent": x.intent,
                     "attribution_reliability": x.attribution_reliability,
                     "reverted": x.reverted, "evaluation_version": x.evaluation_version}
                    for x in s.recent_samples],
                "evaluation_version": EVALUATION_VERSION, "model_version": MODEL_VERSION,
            }
        return {"unsupported": "raw_export_not_provided_by_model"}

    # -- lifecycle ------------------------------------------------------

    def reset(self) -> None:
        self._state = self._empty_state()

    def rebuild(self, raw: Any = None,
                items: Optional[Sequence[Any]] = None) -> ModelRebuildResult:
        self.reset()
        pairs: list[tuple[Any, Optional[ManualCorrectionUpdateContext]]] = []
        for it in (items or []):
            if isinstance(it, ManualCorrectionRebuildItem):
                pairs.append((it.event, it.context))
            elif isinstance(it, ManualCorrectionEvent):
                pairs.append((it, None))
        pairs.sort(key=lambda pr: (_iso(getattr(pr[0], "ts", None)) or "",
                                   getattr(pr[0], "event_id", "")))
        accepted = rejected = duplicates = 0
        rejection_counts: dict[str, int] = {}
        for event, context in pairs:
            res = self.update(event, context)
            if res.accepted:
                accepted += 1
            else:
                rejected += 1
                code = res.confounder_flags[0] if res.confounder_flags else "unknown"
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
                if code == ManualCorrectionRejection.DUPLICATE_EVENT.value:
                    duplicates += 1
        return ModelRebuildResult(
            processed_count=len(pairs), accepted_count=accepted, rejected_count=rejected,
            duplicate_count=duplicates, error_count=0, rejection_counts=rejection_counts,
            started_from_prior=True, deterministic=True)

    # -- state ----------------------------------------------------------

    def serialize_state(self) -> Mapping[str, Any]:
        s = self._state
        return {
            "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
            "evaluation_version": EVALUATION_VERSION, "learning_zone_id": s.learning_zone_id,
            "general": _bucket_dict(s.general),
            "buckets": {k: _bucket_dict(b) for k, b in s.buckets.items()},
            "preference_bias_c": s.preference_bias_c, "processed_ids": list(s.processed_ids),
            "processed_decision_ids": list(s.processed_decision_ids),
            "recent_samples": [_sample_dict(x) for x in s.recent_samples],
            "distinct_days": list(s.distinct_days), "direct_count": s.direct_count,
            "automation_count": s.automation_count, "unknown_count": s.unknown_count,
            "revert_count": s.revert_count, "learnable_count": s.learnable_count,
            "intent_counts": dict(s.intent_counts), "last_update_ts": s.last_update_ts,
            "rejection_counts": dict(s.rejection_counts),
            "outlier_counts": dict(s.outlier_counts),
        }

    def validate_state(self, state: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        if not isinstance(state, dict):
            return ["state is not a mapping"]
        if state.get("model_version") != MODEL_VERSION:
            errors.append("model_version mismatch")
        if state.get("learning_zone_id") != self._zone:
            errors.append("learning_zone_id mismatch")
        bias = state.get("preference_bias_c")
        if bias is not None and not is_finite(bias):
            errors.append("non-finite preference_bias_c")
        g = state.get("general", {}).get("signed_magnitude", {})
        for k in ("value", "effective_n", "dispersion"):
            v = g.get(k)
            if v is not None and not is_finite(v):
                errors.append(f"non-finite general.signed_magnitude.{k}")
        return errors

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        errors = self.validate_state(state)
        if errors:
            raise ValueError(f"invalid manual-correction state: {errors}")
        self._state = ManualCorrectionState(
            learning_zone_id=state["learning_zone_id"], general=_bucket_from(state["general"]),
            buckets={k: _bucket_from(b) for k, b in state.get("buckets", {}).items()},
            preference_bias_c=state.get("preference_bias_c", 0.0),
            processed_ids=tuple(state.get("processed_ids", [])),
            processed_decision_ids=tuple(state.get("processed_decision_ids", [])),
            recent_samples=tuple(_sample_from(x) for x in state.get("recent_samples", [])),
            distinct_days=tuple(state.get("distinct_days", [])),
            direct_count=state.get("direct_count", 0),
            automation_count=state.get("automation_count", 0),
            unknown_count=state.get("unknown_count", 0),
            revert_count=state.get("revert_count", 0),
            learnable_count=state.get("learnable_count", 0),
            intent_counts=dict(state.get("intent_counts", {})),
            last_update_ts=state.get("last_update_ts"),
            rejection_counts=dict(state.get("rejection_counts", {})),
            outlier_counts=dict(state.get("outlier_counts", {})))

    def migrate_state(self, old_model_version: int, old_parameter_version: int,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        if old_model_version == MODEL_VERSION:
            return state
        raise ValueError(
            f"no migration path from model_version {old_model_version} to {MODEL_VERSION}")


# -- helpers ------------------------------------------------------------------

def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _day(ts: datetime) -> str:
    return ts.date().isoformat()


def _sample_dict(x: ManualCorrectionSample) -> dict:
    return {
        "correction_event_id": x.correction_event_id, "decision_id": x.decision_id,
        "learning_zone_id": x.learning_zone_id, "direction": x.direction,
        "signed_magnitude_c": x.signed_magnitude_c, "source_class": x.source_class,
        "intent": x.intent, "attribution_reliability": x.attribution_reliability,
        "reverted": x.reverted, "day": x.day, "effective_weight": x.effective_weight,
        "evaluation_version": x.evaluation_version,
    }


def _sample_from(d: Mapping[str, Any]) -> ManualCorrectionSample:
    return ManualCorrectionSample(
        correction_event_id=d["correction_event_id"], decision_id=d.get("decision_id"),
        learning_zone_id=d["learning_zone_id"], direction=d["direction"],
        signed_magnitude_c=d["signed_magnitude_c"], source_class=d["source_class"],
        intent=d["intent"], attribution_reliability=d["attribution_reliability"],
        reverted=d["reverted"], day=d["day"], effective_weight=d["effective_weight"],
        evaluation_version=d["evaluation_version"])


def manual_correction_model_definition():
    from ..registry import ModelDefinition

    return ModelDefinition(
        model_name=MODEL_NAME, model_factory=lambda: ManualCorrectionModel("__definition__"),
        model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
        consumed_raw_tracks=(RawTrackName.MANUAL_CORRECTION,), consumed_episode_types=(),
        supported_prediction_types=(
            PredictionType.MANUAL_PREFERENCE_BIAS, PredictionType.CONTROLLER_CALIBRATION),
        required_features=("target",),
        optional_features=("indoor_temp", "outdoor_temp", "forecast"),
        required_capability_flags=(), regime_dependent=False, advisory=True,
        control_relevant=False, rebuildable=True, can_be_not_available=True, min_trv_only=True)
