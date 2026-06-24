"""BoostModel for LE 2.0 (pure Python).

Learns whether — and how strongly — an extra heating impulse (a temporary TRV
setpoint offset above target) was worthwhile. It judges THERMAL effect and
controller quality, never real kWh. A boost is not "good" just because the target
was reached faster: overshoot, comfort, holding and attribution are scored
separately so a fast-but-overshooting boost is penalised.

Boost-factor semantics follow the existing ThermoSmart controller: an ADDITIVE
°C setpoint offset above the target (``trv_setpoint - target``), bounded by
``TPI_MAX_BOOST_CELSIUS`` (8.0 °C) and the 28 °C device cap (see
``trv_control.py`` / ``tpi.py`` / ``const.py``). ``1.0`` is NOT a multiplier here;
``0.0`` means no extra boost.

No HA / storage / coordinator dependency; imports no concrete model — only typed,
persistable snapshots via the context. No control effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ..contracts import (
    ConfidenceContribution,
    DataQuality,
    ExportScope,
    ModelRebuildResult,
    OutlierStatus,
    Prediction,
    PredictionType,
    Regime,
    RejectionReason,
    UpdateEligibilityResult,
)
from ..episode_schemas import ControllerKind, EpisodeReason, EpisodeType, OutcomeEpisode
from ..features import FeatureExtractor, FeatureName, FeatureStatus
from .base import clamp, is_finite, outlier_z, robust_ema_update
from .boost_degradation import (
    BoostDegradationState, BoostLearningReadiness, BoostScopedDegradation, DegradationParams,
    is_scope_eligible, maybe_capture_known_good, observe_scoped, resolve_cooldown,
    resolve_scoped_cooldown)

MODEL_VERSION = 1
PARAMETER_VERSION = 1
EVALUATION_VERSION = 1
MODEL_NAME = "boost"

# additive °C offset above target (mirrors TPI_MAX_BOOST_CELSIUS in const.py)
_MAX_BOOST_OFFSET_C = 8.0

_BLOCKING_REASONS = frozenset({EpisodeReason.DISTURBED})
_DIMENSIONS = ("heating_gain", "time_gain", "target_achievement", "comfort_impact",
               "overshoot_penalty", "controller_quality", "data_quality")
_CONFOUNDER_FLAGS = frozenset({
    "window_open", "solar_gain", "heating_failure", "mode_change", "reheating",
    "additional_radiator", "sensor_gap"})


class BoostRejection(Enum):
    WRONG_EPISODE_TYPE = "wrong_episode_type"
    MISSING_BOOST_DECISION = "missing_boost_decision"
    DECISION_MISMATCH = "decision_mismatch"
    WRONG_ZONE = "wrong_zone"
    DUPLICATE_DECISION = "duplicate_decision"
    INSUFFICIENT_DURATION = "insufficient_duration"
    INSUFFICIENT_TEMPERATURE_DATA = "insufficient_temperature_data"
    MISSING_TARGET = "missing_target"
    INVALID_TRAJECTORY = "invalid_trajectory"
    LOW_DATA_QUALITY = "low_data_quality"
    DISTURBED = "disturbed"
    WINDOW_CONFOUNDER = "window_confounder"
    HEATING_FAILURE = "heating_failure"
    SOLAR_CONFOUNDER = "solar_confounder"
    MANUAL_CORRECTION_CONTAMINATION = "manual_correction_contamination"
    MODE_CHANGE = "mode_change"
    SUPERSEDED = "superseded"
    ATTRIBUTION_UNAVAILABLE = "attribution_unavailable"
    NON_FINITE_EVALUATION = "non_finite_evaluation"
    VERSION_MISMATCH = "version_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    SEVERE_OUTLIER = "severe_outlier"


_TO_CONTRACT = {
    BoostRejection.WRONG_EPISODE_TYPE: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.MISSING_BOOST_DECISION: RejectionReason.MISSING_REQUIRED_DATA,
    BoostRejection.DECISION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.WRONG_ZONE: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.DUPLICATE_DECISION: RejectionReason.DUPLICATE,
    BoostRejection.INSUFFICIENT_DURATION: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.INSUFFICIENT_TEMPERATURE_DATA: RejectionReason.MISSING_REQUIRED_DATA,
    BoostRejection.MISSING_TARGET: RejectionReason.MISSING_REQUIRED_DATA,
    BoostRejection.INVALID_TRAJECTORY: RejectionReason.MISSING_REQUIRED_DATA,
    BoostRejection.LOW_DATA_QUALITY: RejectionReason.LOW_RELIABILITY,
    BoostRejection.DISTURBED: RejectionReason.DISTURBED_REGIME,
    BoostRejection.WINDOW_CONFOUNDER: RejectionReason.DISTURBED_REGIME,
    BoostRejection.HEATING_FAILURE: RejectionReason.DISTURBED_REGIME,
    BoostRejection.SOLAR_CONFOUNDER: RejectionReason.DISTURBED_REGIME,
    BoostRejection.MANUAL_CORRECTION_CONTAMINATION: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.MODE_CHANGE: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.SUPERSEDED: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.ATTRIBUTION_UNAVAILABLE: RejectionReason.LOW_RELIABILITY,
    BoostRejection.NON_FINITE_EVALUATION: RejectionReason.OUTLIER,
    BoostRejection.VERSION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.CONTEXT_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    BoostRejection.SEVERE_OUTLIER: RejectionReason.OUTLIER,
}


# ── B2b-2: discrete boost outcome classification ─────────────────────────────

class BoostOutcomeClass(Enum):
    """Primary discrete outcome of one boost episode. Exactly one per episode."""
    BOOST_SUCCESS = "boost_success"
    BOOST_NO_EFFECT = "boost_no_effect"
    BOOST_OVERSHOOT = "boost_overshoot"
    BOOST_UNNECESSARY = "boost_unnecessary"
    BOOST_CONFOUNDED = "boost_confounded"
    BOOST_PARTIAL_DISPATCH = "boost_partial_dispatch"
    BOOST_FAILED_DISPATCH = "boost_failed_dispatch"
    BOOST_NOT_ATTEMPTED = "boost_not_attempted"
    BOOST_INTERRUPTED = "boost_interrupted"
    INSUFFICIENT_COMPARISON = "insufficient_comparison"


# Confounders that mean the OBSERVATION is incomplete -> interrupted.
_INTERRUPT_CONFOUNDERS = frozenset({
    "sensor_gap", "restart_gap", "schedule_change", "target_change", "manual_override",
    "mode_change", "window_open", "heating_failure"})
# Confounders that mean the observation exists but CAUSE is ambiguous -> confounded.
_CONFOUND_CONFOUNDERS = frozenset({
    "solar_gain", "reheating", "additional_radiator", "external_heat", "supply_delay",
    "multi_trv_ambiguous"})


@dataclass(frozen=True)
class _OutcomeEffect:
    store: bool                 # persist a research sample
    factor_update: bool         # may move the learned effective_factor at all
    direction: str              # "positive" | "negative" | "neutral"
    confidence_increase: bool   # may evidence (effective_n) grow
    rollback_evidence: bool     # counts toward future rollback (B2b-5)
    cooldown_evidence: bool


_CLASS_EFFECT: Mapping[BoostOutcomeClass, _OutcomeEffect] = {
    BoostOutcomeClass.BOOST_SUCCESS:
        _OutcomeEffect(True, True, "positive", True, False, False),
    BoostOutcomeClass.BOOST_NO_EFFECT:
        _OutcomeEffect(True, True, "neutral", False, False, False),
    BoostOutcomeClass.BOOST_OVERSHOOT:
        _OutcomeEffect(True, True, "negative", False, True, True),
    BoostOutcomeClass.BOOST_UNNECESSARY:
        _OutcomeEffect(True, True, "negative", False, True, False),
    BoostOutcomeClass.BOOST_CONFOUNDED:
        _OutcomeEffect(True, False, "neutral", False, False, False),
    BoostOutcomeClass.BOOST_PARTIAL_DISPATCH:
        # B2b-2 correction #3: partial dispatch has no clean thermal attribution ->
        # store the sample + class, reduced reliability, but NO factor movement at all
        # (a weighted effect may arrive in B2b-3/B2b-4 with a measured executed share).
        _OutcomeEffect(True, False, "neutral", False, False, False),
    BoostOutcomeClass.BOOST_FAILED_DISPATCH:
        _OutcomeEffect(False, False, "neutral", False, False, True),
    BoostOutcomeClass.BOOST_NOT_ATTEMPTED:
        _OutcomeEffect(False, False, "neutral", False, False, False),
    BoostOutcomeClass.BOOST_INTERRUPTED:
        _OutcomeEffect(True, False, "neutral", False, False, False),
    BoostOutcomeClass.INSUFFICIENT_COMPARISON:
        _OutcomeEffect(True, False, "neutral", False, False, False),
}


def class_effect(oc: BoostOutcomeClass) -> _OutcomeEffect:
    return _CLASS_EFFECT[oc]


def classify_boost_outcome(*, dispatch_status: Optional[str], authority: str,
                           confounder_flags, attribution: Optional[float],
                           has_temp: bool, overshoot_c: Optional[float],
                           has_baseline: bool, relative_benefit: Optional[float],
                           target_reached: bool, params: "BoostParameters",
                           is_partial: bool = False,
                           overshoot_stable: Optional[bool] = None,
                           target_stable: Optional[bool] = None) -> BoostOutcomeClass:
    """Deterministic primary-class selection (single source of truth).

    Final priority (justified): failed > not_attempted > interrupted > partial >
    confounded > OVERSHOOT > insufficient_comparison > unnecessary > no_effect > success.

    Overshoot is raised ABOVE insufficient_comparison vs. the recommended order: a
    reliable negative *safety* finding must surface even when no baseline comparison
    exists — the absence of a comparison base must never mask an overshoot. All other
    positions follow the recommended order.
    """
    # B2b-4: a single authoritative confounder evaluator (shared with the baseline builder)
    # classifies interrupting vs. blocking by typed severity, with root-cause de-duplication.
    from .boost_reliability import evaluate_confounders
    conf = evaluate_confounders(list(confounder_flags or ()), partial_dispatch=is_partial)
    if dispatch_status == "failed":
        return BoostOutcomeClass.BOOST_FAILED_DISPATCH
    if dispatch_status == "not_attempted":
        return BoostOutcomeClass.BOOST_NOT_ATTEMPTED
    if conf.is_interrupting:
        return BoostOutcomeClass.BOOST_INTERRUPTED
    if is_partial or dispatch_status == "partially_succeeded":
        return BoostOutcomeClass.BOOST_PARTIAL_DISPATCH
    if conf.is_blocking:
        return BoostOutcomeClass.BOOST_CONFOUNDED
    if attribution is not None and attribution < params.min_attribution_reliability:
        return BoostOutcomeClass.BOOST_CONFOUNDED
    # B2b-4c: OVERSHOOT requires a STABLE overshoot (a short peak is not overshoot). When
    # stability evidence is supplied it is authoritative; otherwise fall back to the scalar.
    _overshoot = (overshoot_stable if overshoot_stable is not None
                  else (has_temp and overshoot_c is not None
                        and overshoot_c > params.overshoot_class_threshold_c))
    if _overshoot:
        return BoostOutcomeClass.BOOST_OVERSHOOT
    if not has_baseline or relative_benefit is None:
        return BoostOutcomeClass.INSUFFICIENT_COMPARISON
    # B2b-4c: SUCCESS / NO_EFFECT / UNNECESSARY require a STABLY reached target (not a single
    # in-band touch). If target reach is not stable, the comparison is not trustworthy.
    if target_stable is False:
        return BoostOutcomeClass.INSUFFICIENT_COMPARISON
    if relative_benefit <= params.unnecessary_relative_benefit:
        return (BoostOutcomeClass.BOOST_UNNECESSARY if target_reached
                else BoostOutcomeClass.BOOST_NO_EFFECT)
    if relative_benefit < params.success_relative_benefit:
        return BoostOutcomeClass.BOOST_NO_EFFECT
    return BoostOutcomeClass.BOOST_SUCCESS


# Rejected episodes never run the thermal classifier; map the eligibility rejection to a
# diagnostic class (counts only, no model effect). None => not a boost outcome at all.
_REJECTION_TO_CLASS: Mapping[BoostRejection, Optional[BoostOutcomeClass]] = {
    BoostRejection.WINDOW_CONFOUNDER: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.HEATING_FAILURE: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.DISTURBED: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.MODE_CHANGE: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.SUPERSEDED: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.MANUAL_CORRECTION_CONTAMINATION: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.INSUFFICIENT_TEMPERATURE_DATA: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.LOW_DATA_QUALITY: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.INVALID_TRAJECTORY: BoostOutcomeClass.BOOST_INTERRUPTED,
    BoostRejection.SOLAR_CONFOUNDER: BoostOutcomeClass.BOOST_CONFOUNDED,
    BoostRejection.ATTRIBUTION_UNAVAILABLE: BoostOutcomeClass.BOOST_CONFOUNDED,
    BoostRejection.SEVERE_OUTLIER: BoostOutcomeClass.BOOST_CONFOUNDED,
    # structural rejections are NOT boost outcomes (no count):
    BoostRejection.WRONG_EPISODE_TYPE: None,
    BoostRejection.MISSING_BOOST_DECISION: None,
    BoostRejection.DECISION_MISMATCH: None,
    BoostRejection.WRONG_ZONE: None,
    BoostRejection.DUPLICATE_DECISION: None,
    BoostRejection.MISSING_TARGET: None,
    BoostRejection.VERSION_MISMATCH: None,
    BoostRejection.CONTEXT_MISMATCH: None,
    BoostRejection.NON_FINITE_EVALUATION: None,
    BoostRejection.INSUFFICIENT_DURATION: None,
}


class DimensionStatus(Enum):
    OK = "ok"
    PARTIAL = "partial"
    NOT_COMPUTABLE = "not_computable"


class BoostSourceKind(Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    UNKNOWN = "unknown"


class BoostLifecycle(Enum):
    """Transient per-episode boost lifecycle state (not persisted)."""
    INACTIVE = "inactive"
    ELIGIBLE = "eligible"
    APPLIED = "applied"
    ACTIVE = "active"
    COMPLETED = "completed"
    RELEASED_TARGET_REACHED = "released_target_reached"
    RELEASED_TIMEOUT = "released_timeout"
    RELEASED_WINDOW_OPEN = "released_window_open"
    RELEASED_MANUAL_OVERRIDE = "released_manual_override"
    RELEASED_MODE_CHANGE = "released_mode_change"
    # Deescalation: boost released mid-episode because normal heating is sufficient
    RELEASED_TPI_SUFFICIENT = "released_tpi_sufficient"
    RELEASED_OVERSHOOT_RISK = "released_overshoot_risk"
    RELEASED_AFTERHEAT_SUFFICIENT = "released_afterheat_sufficient"
    FAILED_NO_RESPONSE = "failed_no_response"
    FAILED_OVERSHOOT = "failed_overshoot"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class BoostParameters:
    max_boost_offset_c: float = _MAX_BOOST_OFFSET_C
    min_boost_offset_c: float = 0.1
    min_duration_s: float = 120.0
    min_points: int = 3
    min_data_quality: float = 0.25
    min_attribution_reliability: float = 0.3
    max_sample_weight: float = 1.0
    # B2b-2 discrete classification thresholds
    overshoot_class_threshold_c: float = 0.5      # overshoot above target band => OVERSHOOT
    success_relative_benefit: float = 0.1         # min time-benefit vs baseline for SUCCESS
    unnecessary_relative_benefit: float = 0.0     # <= this => boost gave no time advantage
    bucket_min_samples: int = 5
    bucket_seed_evidence: float = 1.0
    start_deficit_edges_c: tuple[float, ...] = (1.0, 3.0)
    intensity_edges_c: tuple[float, ...] = (1.0, 3.0)
    overshoot_comfort_scale_c: float = 2.0
    outlier_mad_k_mild: float = 3.0
    outlier_mad_k_severe: float = 6.0
    min_samples_for_outlier: int = 5
    dedup_max_ids: int = 500
    research_sample_cap: int = 50
    full_confidence_samples: float = 20.0
    cold_start_confidence_cap: float = 0.35
    partial_weight_factor: float = 0.5
    default_boost_offset_c: float = 0.0    # neutral prior: no automatic boost without evidence
    default_boost_duration_s: float = 1800.0
    min_duration_pred_s: float = 300.0
    max_duration_pred_s: float = 7200.0
    # Adaptive control limits (separate from technical 8°C TPI maximum)
    # 3°C: conservative LE 2.0 cap — even a well-learned boost rarely exceeds this in residential
    adaptive_max_boost_c: float = 3.0
    # Mirrors ControlPolicy.min_control_confidence: used for adaptive cap tiering
    min_control_confidence: float = 0.55
    # Lifecycle durations — derived from domain logic, not arbitrary defaults:
    #   max_boost_duration_s = 3600 (1h):
    #     Hard lifecycle safety cap — NOT the expected boost duration. In normal operation
    #     the lifecycle releases via target_reached (current_temp >= target), which terminates
    #     boost exactly when the additional offset ceases to provide measurable benefit. The
    #     1h cap only fires when the target is never reached (severe under-heating, TRV fault)
    #     and prevents indefinite boost application in stuck/failed episodes. Slow rooms
    #     (floor heating, large post-setback deficits, onset delays up to 30 min) may
    #     legitimately need 45–60 min of heating; the cap accommodates that worst case.
    #     max_duration_pred_s = 7200 (model upper bound); 3600 is the hard lifecycle cap.
    #   failure_detection_window_s = 1200 (20 min):
    #     Starts AFTER effective onset. Slow rooms with ≥ 1°C/h heat rate may take
    #     20 min to show a detectable temperature rise above noise. 15 min was too
    #     aggressive for cold rooms with low heat rates.
    #   cooldown_duration_s = 300 (5 min): normal/external release — minimum guard against
    #     rapid re-application in the same scheduler cycle.
    #   failure_cooldown_duration_s = 900 (15 min): failure releases — TRV may need time to
    #     recover from overshoot or unresponsive state before another attempt is safe.
    max_boost_duration_s: float = 3600.0
    failure_detection_window_s: float = 1200.0
    cooldown_duration_s: float = 300.0
    failure_cooldown_duration_s: float = 900.0
    # Deescalation thresholds: when to release an active boost mid-episode.
    # released_tpi_sufficient: LE2 heat-rate projection shows TPI closes gap without boost.
    # High TPI duty alone proves high demand, NOT sufficiency; duty threshold is not used.
    deescalation_tpi_heat_rate_min_confidence: float = 0.35  # gate cold-start hr predictions
    deescalation_tpi_max_remaining_c: float = 1.0    # only release if deficit is small
    deescalation_tpi_safety_horizon_min: float = 20.0  # hr must close gap in ≤ this time
    # released_overshoot_risk: LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise) as primary,
    # or safety heuristic (fallback: remaining ≤ deficit_c AND slope > 0 AND age ≥ min_active).
    # BOOST_OUTCOME is NOT the authority here — it measures historical boost quality, not residual rise.
    deescalation_overshoot_min_confidence: float = 0.30   # gate for LE2 EXPECTED_OVERSHOOT path
    deescalation_overshoot_safety_buffer_c: float = 0.05  # remaining ≤ expected + buffer
    deescalation_overshoot_deficit_c: float = 0.3         # near-target threshold (LE2 + heuristic)
    deescalation_overshoot_min_active_s: float = 600.0    # heuristic age gate
    # released_afterheat_sufficient: requires LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise).
    # BOOST_OUTCOME must NOT be used — it is historical boost outcome quality, not physical afterheat.
    # Current slope is supporting evidence; AfterheatModel prediction is the authoritative source.
    deescalation_afterheat_min_confidence: float = 0.35   # prediction must be confident
    deescalation_afterheat_safety_margin_c: float = 0.1   # residual_rise ≥ remaining + margin
    deescalation_afterheat_slope_c_per_min: float = 0.05  # min slope supporting evidence
    # Soft deescalation step-down: reduces active boost offset by this amount per cycle
    # (rather than immediately zeroing). Prevents abrupt setpoint changes during soft release.
    deescalation_soft_step_c: float = 0.5
    # TPI sufficiency: when comfort target time is known, the safety margin ensures TPI
    # has enough headroom to actually reach target within the schedule window.
    deescalation_tpi_safety_margin_min: float = 5.0
    parameter_version: int = PARAMETER_VERSION


# -- effect / evaluation types ------------------------------------------------

@dataclass(frozen=True)
class BoostEffect:
    temperature_gain_c: Optional[float]
    effective_heating_rate_c_per_h: Optional[float]
    time_to_target_s: Optional[float]
    overshoot_c: Optional[float]
    holding_quality: Optional[float]
    boost_duration_s: float
    relative_benefit: Optional[float]
    attribution_reliability: float
    requested_offset_c: float

    def __post_init__(self) -> None:
        for name in ("temperature_gain_c", "effective_heating_rate_c_per_h",
                     "time_to_target_s", "overshoot_c", "holding_quality",
                     "relative_benefit"):
            v = getattr(self, name)
            if v is not None and not is_finite(v):
                raise ValueError(f"{name} must be finite or None")


@dataclass(frozen=True)
class BoostScore:
    name: str
    score: Optional[float]
    reliability: float
    status: DimensionStatus
    reason_codes: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be within [0,1]")
        if self.status is DimensionStatus.NOT_COMPUTABLE:
            if self.score is not None:
                raise ValueError("not-computable dimension must not carry a score")
        elif self.score is None or not is_finite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError(f"{self.name}: score must be finite within [0,1]")

    @property
    def computable(self) -> bool:
        return self.status is not DimensionStatus.NOT_COMPUTABLE


@dataclass(frozen=True)
class BoostEvaluation:
    effect: BoostEffect
    heating_gain: BoostScore
    time_gain: BoostScore
    target_achievement: BoostScore
    comfort_impact: BoostScore
    overshoot_penalty: BoostScore
    controller_quality: BoostScore
    data_quality: BoostScore
    reason: EpisodeReason
    partial: bool
    evaluation_version: int = EVALUATION_VERSION
    reason_codes: tuple[str, ...] = ()

    def get(self, name: str) -> BoostScore:
        return getattr(self, name)


@dataclass(frozen=True)
class BoostRecommendation:
    boost_offset_c: float
    recommended_duration_s: Optional[float]
    confidence: float
    reliability: float
    fallback_used: bool
    prior_contribution: float
    learned_contribution: float
    evidence_count: int
    bucket: str
    missing_evidence: tuple[str, ...]
    reason_codes: tuple[str, ...]
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION
    # Current LE 2.0 adaptive cap applied (separate from 8°C technical TPI max)
    adaptive_cap_c: float = 3.0
    # B2b-5 / B2b-5a: learning readiness (NOT a release lifecycle / not an activation signal)
    readiness: str = BoostLearningReadiness.INSUFFICIENT_DATA
    selected_scope: str = "prior"             # bucket | general | general_fallback | prior
    bucket_handle: Optional[str] = None       # hashed selected bucket key (privacy-safe)
    factor_usable: bool = False
    rollback_active: bool = False
    cooldown_active: bool = False
    cooldown_remaining_s: Optional[float] = None
    eligibility_reason: str = ""
    rehab_progress: float = 0.0
    known_good_available: bool = False

    def __post_init__(self) -> None:
        if not is_finite(self.boost_offset_c):
            raise ValueError("boost_offset_c must be finite")


@dataclass(frozen=True)
class BoostUpdateContext:
    source_episode_id: str
    decision_id: str
    learning_zone_id: Optional[str] = None
    boost_source: str = BoostSourceKind.UNKNOWN.value
    requested_offset_c: float = 0.0
    planned_duration_s: Optional[float] = None
    actual_duration_s: Optional[float] = None
    target_temperature_c: Optional[float] = None
    start_temperature_c: Optional[float] = None
    start_deficit_c: Optional[float] = None
    expected_non_boost_duration_s: Optional[float] = None
    expected_heat_rate_c_per_h: Optional[float] = None
    expected_afterheat_rise_c: Optional[float] = None
    expected_heat_loss_c_per_h: Optional[float] = None
    outcome_performance: Optional[float] = None
    outcome_comfort: Optional[float] = None
    controller_kind: Optional[str] = None
    outdoor_bucket: Optional[str] = None
    manual_correction_ref: Optional[str] = None
    context_quality: DataQuality = DataQuality.OK
    source_refs: tuple[str, ...] = ()
    source_feature_version: int = 1
    # ── B2b-1: authoritative dispatch binding (from LiveDecisionRecord) ──
    # ``authority`` distinguishes a context derived from the bound dispatch result
    # ("live_record") from a clearly-marked diagnostic legacy fallback
    # ("legacy_fallback"). ``boost_applied_c`` is the offset actually written
    # (0.0 ≠ "not dispatched"). ``dispatch_status``/``outcome_reliability`` gate the
    # thermal attribution (partial dispatch ⇒ reduced reliability).
    authority: str = "unknown"
    boost_applied_c: Optional[float] = None
    baseline_setpoint_c: Optional[float] = None
    final_setpoint_c: Optional[float] = None
    effective_setpoint_min_c: Optional[float] = None
    effective_setpoint_max_c: Optional[float] = None
    dispatch_status: Optional[str] = None
    outcome_reliability: Optional[str] = None     # "full"|"partial"|"none"
    # B2b-3a: reliability of the historical non-boost comparison baseline ([0,1]). When a
    # baseline is used (expected_non_boost_duration_s set), this scales the attribution so a
    # weak-but-admissible baseline yields a smaller adaptive update than a robust one.
    baseline_reliability: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.source_episode_id:
            raise ValueError("source_episode_id must be non-empty")
        if not self.decision_id:
            raise ValueError("decision_id must be non-empty")
        for name in ("requested_offset_c", "planned_duration_s", "actual_duration_s",
                     "target_temperature_c", "start_temperature_c", "start_deficit_c",
                     "expected_non_boost_duration_s", "expected_heat_rate_c_per_h",
                     "expected_afterheat_rise_c", "expected_heat_loss_c_per_h",
                     "outcome_performance", "outcome_comfort", "boost_applied_c",
                     "baseline_setpoint_c", "final_setpoint_c",
                     "effective_setpoint_min_c", "effective_setpoint_max_c"):
            v = getattr(self, name)
            if v is not None and not is_finite(v):
                raise ValueError(f"{name} must be finite (or None)")

    def to_dict(self) -> dict:
        return {
            "source_episode_id": self.source_episode_id, "decision_id": self.decision_id,
            "learning_zone_id": self.learning_zone_id, "boost_source": self.boost_source,
            "requested_offset_c": self.requested_offset_c,
            "planned_duration_s": self.planned_duration_s,
            "actual_duration_s": self.actual_duration_s,
            "target_temperature_c": self.target_temperature_c,
            "start_temperature_c": self.start_temperature_c,
            "start_deficit_c": self.start_deficit_c,
            "expected_non_boost_duration_s": self.expected_non_boost_duration_s,
            "expected_heat_rate_c_per_h": self.expected_heat_rate_c_per_h,
            "expected_afterheat_rise_c": self.expected_afterheat_rise_c,
            "expected_heat_loss_c_per_h": self.expected_heat_loss_c_per_h,
            "outcome_performance": self.outcome_performance,
            "outcome_comfort": self.outcome_comfort, "controller_kind": self.controller_kind,
            "outdoor_bucket": self.outdoor_bucket,
            "manual_correction_ref": self.manual_correction_ref,
            "context_quality": self.context_quality.value,
            "source_refs": list(self.source_refs),
            "source_feature_version": self.source_feature_version,
            "authority": self.authority,
            "boost_applied_c": self.boost_applied_c,
            "baseline_setpoint_c": self.baseline_setpoint_c,
            "final_setpoint_c": self.final_setpoint_c,
            "effective_setpoint_min_c": self.effective_setpoint_min_c,
            "effective_setpoint_max_c": self.effective_setpoint_max_c,
            "dispatch_status": self.dispatch_status,
            "outcome_reliability": self.outcome_reliability,
            "baseline_reliability": self.baseline_reliability,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BoostUpdateContext":
        return cls(
            source_episode_id=d["source_episode_id"], decision_id=d["decision_id"],
            learning_zone_id=d.get("learning_zone_id"),
            boost_source=d.get("boost_source", BoostSourceKind.UNKNOWN.value),
            requested_offset_c=d.get("requested_offset_c", 0.0),
            planned_duration_s=d.get("planned_duration_s"),
            actual_duration_s=d.get("actual_duration_s"),
            target_temperature_c=d.get("target_temperature_c"),
            start_temperature_c=d.get("start_temperature_c"),
            start_deficit_c=d.get("start_deficit_c"),
            expected_non_boost_duration_s=d.get("expected_non_boost_duration_s"),
            expected_heat_rate_c_per_h=d.get("expected_heat_rate_c_per_h"),
            expected_afterheat_rise_c=d.get("expected_afterheat_rise_c"),
            expected_heat_loss_c_per_h=d.get("expected_heat_loss_c_per_h"),
            outcome_performance=d.get("outcome_performance"),
            outcome_comfort=d.get("outcome_comfort"),
            controller_kind=d.get("controller_kind"), outdoor_bucket=d.get("outdoor_bucket"),
            manual_correction_ref=d.get("manual_correction_ref"),
            context_quality=DataQuality(d.get("context_quality", DataQuality.OK.value)),
            source_refs=tuple(d.get("source_refs", [])),
            source_feature_version=d.get("source_feature_version", 1),
            authority=d.get("authority", "unknown"),
            boost_applied_c=d.get("boost_applied_c"),
            baseline_setpoint_c=d.get("baseline_setpoint_c"),
            final_setpoint_c=d.get("final_setpoint_c"),
            effective_setpoint_min_c=d.get("effective_setpoint_min_c"),
            effective_setpoint_max_c=d.get("effective_setpoint_max_c"),
            dispatch_status=d.get("dispatch_status"),
            outcome_reliability=d.get("outcome_reliability"),
            baseline_reliability=d.get("baseline_reliability"))


@dataclass(frozen=True)
class BoostRebuildItem:
    episode: Any
    context: BoostUpdateContext


# -- learning state -----------------------------------------------------------

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
class BoostBucket:
    key: str
    gain: _Dim = field(default_factory=_Dim)
    overshoot: _Dim = field(default_factory=_Dim)
    comfort: _Dim = field(default_factory=_Dim)
    controller: _Dim = field(default_factory=_Dim)
    effective_factor: _Dim = field(default_factory=_Dim)
    duration: _Dim = field(default_factory=_Dim)
    sample_count: int = 0

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0


def _bucket_dict(b: BoostBucket) -> dict:
    return {"key": b.key, "gain": _dim_dict(b.gain), "overshoot": _dim_dict(b.overshoot),
            "comfort": _dim_dict(b.comfort), "controller": _dim_dict(b.controller),
            "effective_factor": _dim_dict(b.effective_factor),
            "duration": _dim_dict(b.duration), "sample_count": b.sample_count}


def _bucket_from(d: Mapping[str, Any]) -> BoostBucket:
    return BoostBucket(
        key=d["key"], gain=_dim_from(d["gain"]), overshoot=_dim_from(d["overshoot"]),
        comfort=_dim_from(d["comfort"]), controller=_dim_from(d["controller"]),
        effective_factor=_dim_from(d["effective_factor"]), duration=_dim_from(d["duration"]),
        sample_count=d["sample_count"])


@dataclass(frozen=True)
class BoostSample:
    source_episode_id: str
    decision_id: str
    learning_zone_id: str
    requested_offset_c: float
    effective_factor_c: float
    temperature_gain_c: Optional[float]
    overshoot_c: Optional[float]
    comfort_score: Optional[float]
    controller_score: Optional[float]
    boost_duration_s: float
    attribution_reliability: float
    effective_weight: float
    reason: str
    partial: bool
    evaluation_version: int
    # B2b-2: discrete primary outcome class (neutral default for legacy/old samples)
    outcome_class: str = BoostOutcomeClass.INSUFFICIENT_COMPARISON.value
    # B2b-2 corr.#2/#3: True only when this sample actually moved the learned model.
    # Non-adaptive samples (legacy_fallback, partial, confounded, interrupted,
    # insufficient_comparison) are stored for diagnostics but feed NO adaptive maturity
    # (confidence / reliability / counts). Old samples default False (safe/neutral).
    adaptive: bool = False
    # B2b-4 / B2b-4a: typed multi-component reliability snapshot (serialized dict) + overall.
    # Legacy samples without reliability metadata MUST degrade: reliability_known=False and
    # reliability_overall=0.0 (unknown quality != perfect quality) -> never adaptive evidence.
    reliability_known: bool = False
    reliability_overall: float = 0.0
    reliability_components: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoostState:
    learning_zone_id: str
    general: BoostBucket
    buckets: Mapping[str, BoostBucket] = field(default_factory=dict)
    processed_ids: tuple[str, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()
    recent_samples: tuple[BoostSample, ...] = ()       # adaptive samples (feed confidence)
    diagnostic_samples: tuple[BoostSample, ...] = ()   # B2b-2: non-adaptive, diagnostics only
    last_update_ts: Optional[str] = None
    full_count: int = 0
    partial_count: int = 0
    overshoot_count: int = 0
    manual_correction_count: int = 0
    reason_counts: Mapping[str, int] = field(default_factory=dict)
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    outcome_class_counts: Mapping[str, int] = field(default_factory=dict)  # B2b-2
    # B2b-5 / B2b-5a: scoped (per-bucket + general) readiness / degradation / rollback /
    # cooldown authority.
    degradation: BoostScopedDegradation = field(default_factory=BoostScopedDegradation)
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION
    evaluation_version: int = EVALUATION_VERSION
    # Runtime lifecycle (not persisted; reset on restore)
    lifecycle: BoostLifecycle = BoostLifecycle.INACTIVE
    current_episode_id: Optional[str] = None
    applied_offset_c: float = 0.0
    cooldown_until_ts: Optional[str] = None
    # Episode binding (runtime only, not persisted; reset on deserialize)
    lifecycle_start_ts: Optional[str] = None
    lifecycle_base_target_c: Optional[float] = None
    lifecycle_user_target_c: Optional[float] = None
    lifecycle_max_duration_s: float = 1800.0
    lifecycle_release_reason: Optional[str] = None
    # Failed-episode binding: ID of the last episode that ended in a failure reason.
    # Blocks re-application of the same episode even after cooldown expires.
    # Cleared on clean episode change (schedule/mode/target change).
    last_failed_episode_id: Optional[str] = None
    # Completed-episode binding: ID of the last successfully completed episode.
    # Blocks same episode from re-boosting in the next cycle (0s cooldown loophole).
    # Cleared on clean episode change (schedule/mode/target change).
    last_completed_episode_id: Optional[str] = None


@dataclass(frozen=True)
class BoostDiagnostics:
    general_recommended_offset_c: float
    general_recommended_duration_s: Optional[float]
    general_gain_c: float
    general_overshoot_c: float
    general_comfort: float
    general_controller: float
    bucket_offsets: Mapping[str, float]
    sample_counts: Mapping[str, int]
    full_partial: tuple[int, int]
    overshoot_rate: Optional[float]
    manual_correction_rate: Optional[float]
    confidence: float
    reason_counts: Mapping[str, int]
    rejection_counts: Mapping[str, int]
    outlier_counts: Mapping[str, int]
    missing_evidence: tuple[str, ...]
    last_update_ts: Optional[str]
    model_version: int
    parameter_version: int
    evaluation_version: int
    outcome_class_counts: Mapping[str, int] = field(default_factory=dict)  # B2b-2
    degradation: Mapping[str, Any] = field(default_factory=dict)           # B2b-5 privacy-safe


@dataclass(frozen=True)
class BoostPredictionContext:
    start_deficit_c: Optional[float] = None
    decision_type: Optional[str] = None
    intensity_c: Optional[float] = None
    device_prior_offset_c: Optional[float] = None
    outdoor_bucket: Optional[str] = None
    now_ts: Optional[str] = None              # B2b-5: clock for cooldown remaining/resolution
    control_type: Optional[str] = None        # B2b-7: "setpoint" (boostable) | "direct_valve"
    blocking_reasons: tuple = ()              # B2b-7: current runtime blocking confounders
    degrading_reasons: tuple = ()            # B2b-7: current runtime degrading confounders


def _band(value: Optional[float], edges: Sequence[float]) -> Optional[int]:
    if value is None or not is_finite(value):
        return None
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def _bucket_key(deficit: Optional[float], decision_type: Optional[str],
                intensity: Optional[float], p: BoostParameters) -> Optional[str]:
    parts = []
    db = _band(deficit, p.start_deficit_edges_c)
    if db is not None:
        parts.append(f"def{db}")
    if decision_type:
        parts.append(f"dt:{decision_type}")
    ib = _band(intensity, p.intensity_edges_c)
    if ib is not None:
        parts.append(f"int{ib}")
    return "|".join(parts) if parts else None


def boost_offset_c_to_compat_factor(offset_c: float,
                                    max_boost_c: float = _MAX_BOOST_OFFSET_C) -> float:
    """Convert internal additive °C offset to legacy 1.0-neutral multiplier for public attribute.

    The historical LE v1 boost_factor was a decorative multiplier (1.0 = neutral, range [1.0, 2.0])
    that was never applied to TRV control. This adapter keeps the public attribute backward-compatible:
    automations that relied on 1.0 = neutral continue to work correctly.

    Mapping: 0.0°C → 1.0 (neutral); max_boost_c → 2.0 (linear). Never returns 0.0.
    The formula is documented and intentional; it does NOT drive any control decision.
    """
    if offset_c <= 0.0:
        return 1.0
    return round(1.0 + clamp(offset_c / max(max_boost_c, 1.0), 0.0, 1.0), 4)


# -- model --------------------------------------------------------------------

class BoostModel:
    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    parameter_version = PARAMETER_VERSION
    consumed_raw_tracks: tuple[str, ...] = ()
    consumed_episode_types = (EpisodeType.OUTCOME.value,)
    required_features = ("indoor_temp", "target")
    optional_features = ("outdoor_temp", "valve_opening", "hvac_action", "forecast")
    supported_prediction_types = (
        PredictionType.BOOST_FACTOR, PredictionType.BOOST_DURATION,
        PredictionType.BOOST_OUTCOME)
    min_tier = 0

    def __init__(self, learning_zone_id: str, params: Optional[BoostParameters] = None, *,
                 extractor: Optional[FeatureExtractor] = None,
                 device_prior: Optional[Any] = None) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or BoostParameters()
        self._degradation_params = DegradationParams()   # B2b-5 central thresholds
        self._extractor = extractor or FeatureExtractor()
        self._device_prior = device_prior
        self._state = self._empty_state()

    def cold_start_prior(self) -> Mapping[str, Any]:
        return {"boost_offset_c": self._params.default_boost_offset_c,
                "boost_duration_s": self._params.default_boost_duration_s}

    def _empty_state(self) -> BoostState:
        return BoostState(learning_zone_id=self._zone, general=BoostBucket(key="general"))

    def _extractor_version(self) -> int:
        from ..features import FEATURE_EXTRACTOR_VERSION
        return FEATURE_EXTRACTOR_VERSION

    # -- deterministic evaluation (pure) --------------------------------

    def evaluate_boost(self, episode: OutcomeEpisode,
                       context: BoostUpdateContext) -> BoostEvaluation:
        p = self._params
        fs = self._extractor.extract_trajectory_features(episode.trajectory)
        comfort_fs = self._extractor.extract_comfort_features(
            episode.trajectory, target=episode.target,
            comfort_tolerance=episode.comfort_tolerance_at_start)
        valid = int(fs.get(FeatureName.TRAJ_VALID_POINTS).value or 0)
        gap_ratio = fs.get(FeatureName.TRAJ_GAP_RATIO).value or 0.0
        plausible = fs.get(FeatureName.TRAJ_PLAUSIBILITY).status is not FeatureStatus.INVALID
        has_temp = valid >= 1

        attribution = self._attribution_reliability(episode, context)
        duration_s = (context.actual_duration_s if context.actual_duration_s is not None
                      else (episode.end_ts - episode.start_ts).total_seconds())
        requested = clamp(context.requested_offset_c, 0.0, p.max_boost_offset_c)

        # raw effect quantities
        gain = ttp = overshoot = holding = rate = rel_benefit = None
        if has_temp:
            peak = fs.get(FeatureName.TRAJ_PEAK).value
            gain = max(0.0, peak - episode.start_temp)
            if duration_s > 0:
                rate = gain / (duration_s / 3600.0)
            ov = comfort_fs.get(FeatureName.OVERSHOOT)
            overshoot = ov.value if ov.is_present else 0.0
            ttp_f = comfort_fs.get(FeatureName.TIME_TO_FIRST_TARGET)
            ttp = ttp_f.value if ttp_f.is_present else None
            stab = comfort_fs.get(FeatureName.STABILITY_AFTER_TARGET)
            holding = clamp(1.0 - (stab.value / 2.0), 0.0, 1.0) if stab.is_present else 1.0
            # relative benefit only with a typed baseline AND sufficient attribution
            if (ttp is not None and context.expected_non_boost_duration_s
                    and attribution >= p.min_attribution_reliability):
                rel_benefit = clamp(
                    (context.expected_non_boost_duration_s - ttp)
                    / max(context.expected_non_boost_duration_s, 1.0), -1.0, 1.0)

        effect = BoostEffect(
            temperature_gain_c=gain, effective_heating_rate_c_per_h=rate,
            time_to_target_s=ttp, overshoot_c=overshoot, holding_quality=holding,
            boost_duration_s=duration_s, relative_benefit=rel_benefit,
            attribution_reliability=attribution, requested_offset_c=requested)

        data = self._score_data_quality(episode, valid, gap_ratio, plausible, attribution)
        heating_gain = self._score_heating_gain(effect, context, has_temp, attribution)
        time_gain = self._score_time_gain(effect, context, has_temp)
        target = self._score_target(episode, comfort_fs, has_temp)
        comfort = self._score_comfort(comfort_fs, has_temp)
        overshoot_pen = self._score_overshoot(comfort_fs, context, has_temp)
        controller = self._score_controller(episode, context, comfort_fs, duration_s)

        partial = not has_temp
        return BoostEvaluation(
            effect=effect, heating_gain=heating_gain, time_gain=time_gain,
            target_achievement=target, comfort_impact=comfort,
            overshoot_penalty=overshoot_pen, controller_quality=controller, data_quality=data,
            reason=episode.reason, partial=partial,
            reason_codes=(episode.reason.value,
                          f"attribution:{round(attribution, 2)}"))

    def _attribution_reliability(self, episode, context) -> float:
        rel = episode.reliability
        # B2b-4a (single confounder authority — Variant B): the confounder penalty is NOT
        # applied here. It is applied EXACTLY ONCE as ``confounder_reliability`` in
        # BoostOutcomeReliability (gate) and once to the adaptive update weight in update().
        if context.manual_correction_ref:
            rel *= 0.5
        # B2b-1: partial dispatch is not full thermal attribution -> reduced reliability
        if context is not None and getattr(context, "outcome_reliability", None) == "partial":
            rel *= 0.5
        # B2b-3a: when a historical comparison baseline drives the benefit, scale the
        # attribution by the baseline reliability — a weak (but admissible) baseline yields
        # a smaller adaptive update than a robust one. Applied ONCE (no double counting),
        # only when the baseline is actually used (expected_non_boost_duration_s present).
        if (context is not None and context.expected_non_boost_duration_s is not None
                and getattr(context, "baseline_reliability", None) is not None):
            rel *= clamp(float(context.baseline_reliability), 0.0, 1.0)
        # a clean typed baseline / expectations improve attribution
        if context.expected_heat_rate_c_per_h or context.expected_afterheat_rise_c:
            rel = min(1.0, rel * 1.1)
        return clamp(rel, 0.0, 1.0)

    def _score_data_quality(self, episode, valid, gap_ratio, plausible, attribution) -> BoostScore:
        coverage = clamp(valid / max(self._params.min_points, 1), 0.0, 1.0)
        score = episode.reliability * (1.0 - 0.5 * gap_ratio) * (1.0 if plausible else 0.3) \
            * (0.5 + 0.5 * coverage) * (0.5 + 0.5 * attribution)
        status = DimensionStatus.OK if valid >= 1 else DimensionStatus.PARTIAL
        return BoostScore("data_quality", clamp(score, 0.0, 1.0),
                          reliability=clamp(score, 0.0, 1.0), status=status,
                          reason_codes=() if plausible else ("implausible",),
                          missing_evidence=() if valid >= 1 else ("indoor_temp",))

    def _score_heating_gain(self, effect, context, has_temp, attribution) -> BoostScore:
        if not has_temp:
            return BoostScore("heating_gain", None, 0.0, DimensionStatus.NOT_COMPUTABLE,
                              reason_codes=("no_indoor_temp",), missing_evidence=("indoor_temp",))
        deficit = context.start_deficit_c
        if deficit and deficit > 0:
            score = clamp(effect.temperature_gain_c / deficit, 0.0, 1.0)
        else:
            score = clamp(effect.temperature_gain_c / 3.0, 0.0, 1.0)
        return BoostScore("heating_gain", round(score, 4), reliability=attribution,
                          status=DimensionStatus.OK,
                          reason_codes=("absolute_gain",) if effect.relative_benefit is None
                          else ("relative_gain",))

    def _score_time_gain(self, effect, context, has_temp) -> BoostScore:
        if not has_temp:
            return BoostScore("time_gain", None, 0.0, DimensionStatus.NOT_COMPUTABLE,
                              reason_codes=("no_indoor_temp",))
        if effect.relative_benefit is None:
            return BoostScore("time_gain", None, 0.3, DimensionStatus.NOT_COMPUTABLE,
                              reason_codes=("no_baseline",),
                              missing_evidence=("expected_non_boost_duration",))
        score = clamp(0.5 + 0.5 * effect.relative_benefit, 0.0, 1.0)
        return BoostScore("time_gain", round(score, 4), reliability=0.7,
                          status=DimensionStatus.OK, reason_codes=("baseline_relative",))

    def _score_target(self, episode, comfort_fs, has_temp) -> BoostScore:
        if not has_temp:
            return BoostScore("target_achievement", None, 0.0, DimensionStatus.NOT_COMPUTABLE,
                              reason_codes=("no_indoor_temp",))
        reached = (comfort_fs.get(FeatureName.TIME_TO_FIRST_TARGET).is_present
                   or episode.reason is EpisodeReason.REACHED)
        return BoostScore("target_achievement", 1.0 if reached else 0.2, reliability=0.8,
                          status=DimensionStatus.OK,
                          reason_codes=("reached",) if reached else ("not_reached",))

    def _score_comfort(self, comfort_fs, has_temp) -> BoostScore:
        if not has_temp:
            return BoostScore("comfort_impact", None, 0.0, DimensionStatus.NOT_COMPUTABLE,
                              reason_codes=("no_indoor_temp",))
        p = self._params
        ov = comfort_fs.get(FeatureName.OVERSHOOT)
        stab = comfort_fs.get(FeatureName.STABILITY_AFTER_TARGET)
        osc = comfort_fs.get(FeatureName.OSCILLATION)
        ov_term = clamp(1.0 - ov.value / p.overshoot_comfort_scale_c, 0.0, 1.0) \
            if ov.is_present else 1.0
        stab_term = clamp(1.0 - stab.value / 2.0, 0.0, 1.0) if stab.is_present else 1.0
        osc_term = clamp(1.0 - osc.value / 5.0, 0.0, 1.0) if osc.is_present else 1.0
        score = (ov_term + stab_term + osc_term) / 3.0
        return BoostScore("comfort_impact", round(score, 4), reliability=0.8,
                          status=DimensionStatus.OK, reason_codes=("comfort_evaluated",))

    def _score_overshoot(self, comfort_fs, context, has_temp) -> BoostScore:
        if not has_temp:
            return BoostScore("overshoot_penalty", None, 0.0, DimensionStatus.NOT_COMPUTABLE,
                              reason_codes=("no_indoor_temp",))
        ov = comfort_fs.get(FeatureName.OVERSHOOT)
        overshoot = ov.value if ov.is_present else 0.0
        # penalty: only overshoot beyond expected afterheat is the boost's fault
        expected = context.expected_afterheat_rise_c or 0.0
        avoidable = max(0.0, overshoot - expected)
        score = clamp(1.0 - avoidable / self._params.overshoot_comfort_scale_c, 0.0, 1.0)
        return BoostScore("overshoot_penalty", round(score, 4), reliability=0.7,
                          status=DimensionStatus.OK,
                          reason_codes=("avoidable_overshoot",) if avoidable > 0
                          else ("overshoot_within_physics",))

    def _score_controller(self, episode, context, comfort_fs, duration_s) -> BoostScore:
        reason = episode.reason
        reasons = [reason.value]
        if reason in (EpisodeReason.MODE_CHANGE, EpisodeReason.SUPERSEDED):
            return BoostScore("controller_quality", None, 0.0, DimensionStatus.NOT_COMPUTABLE,
                              reason_codes=tuple(reasons + ["not_attributable"]))
        base = 0.7
        if reason is EpisodeReason.REACHED:
            base = 0.85
        elif reason is EpisodeReason.TIMEOUT:
            base = 0.4
        elif reason is EpisodeReason.MANUAL_CORRECTION:
            base = 0.45
            reasons.append("user_disagreed")
        # over-long boost vs planned -> controller inefficiency
        if context.planned_duration_s and duration_s > context.planned_duration_s * 1.5:
            base *= 0.8
            reasons.append("over_long_boost")
        ov = comfort_fs.get(FeatureName.OVERSHOOT)
        if ov.is_present and ov.value > (context.expected_afterheat_rise_c or 0.0):
            base *= 0.85
            reasons.append("avoidable_overshoot")
        partial = not _has_temp(episode)
        return BoostScore("controller_quality", round(clamp(base, 0.0, 1.0), 4),
                          reliability=0.7 if not partial else 0.5,
                          status=DimensionStatus.PARTIAL if partial else DimensionStatus.OK,
                          reason_codes=tuple(reasons))

    # -- eligibility ----------------------------------------------------

    def update_eligibility(self, episode: Any) -> UpdateEligibilityResult:
        return self._eligibility(episode, None)

    def _eligibility(self, episode, context) -> UpdateEligibilityResult:
        p = self._params
        if not isinstance(episode, OutcomeEpisode):
            return self._reject(BoostRejection.WRONG_EPISODE_TYPE)
        if episode.learning_zone_id != self._zone:
            return self._reject(BoostRejection.WRONG_ZONE)
        if episode.episode_schema_version != 1:
            return self._reject(BoostRejection.VERSION_MISMATCH)
        if context is None:
            return self._reject(BoostRejection.MISSING_BOOST_DECISION)
        if context.source_episode_id != episode.episode_id or context.decision_id != \
                episode.decision_id:
            return self._reject(BoostRejection.DECISION_MISMATCH)
        if context.learning_zone_id is not None and context.learning_zone_id != self._zone:
            return self._reject(BoostRejection.CONTEXT_MISMATCH)
        if context.requested_offset_c <= 0.0:
            return self._reject(BoostRejection.MISSING_BOOST_DECISION)
        if episode.target is None:
            return self._reject(BoostRejection.MISSING_TARGET)
        if episode.reason in _BLOCKING_REASONS:
            return self._reject(BoostRejection.DISTURBED)
        confounders = set(episode.confounder_flags)
        if "window_open" in confounders:
            return self._reject(BoostRejection.WINDOW_CONFOUNDER)
        if "heating_failure" in confounders:
            return self._reject(BoostRejection.HEATING_FAILURE)
        if "solar_gain" in confounders:
            return self._reject(BoostRejection.SOLAR_CONFOUNDER)
        if episode.reason is EpisodeReason.MANUAL_CORRECTION:
            return self._reject(BoostRejection.MANUAL_CORRECTION_CONTAMINATION)
        if episode.reason is EpisodeReason.MODE_CHANGE:
            return self._reject(BoostRejection.MODE_CHANGE)
        if episode.reason is EpisodeReason.SUPERSEDED:
            return self._reject(BoostRejection.SUPERSEDED)
        if episode.episode_id in self._state.processed_ids \
                or episode.decision_id in self._state.processed_decision_ids:
            return self._reject(BoostRejection.DUPLICATE_DECISION)

        ev = self.evaluate_boost(episode, context)
        if ev.effect.attribution_reliability < p.min_attribution_reliability:
            return self._reject(BoostRejection.ATTRIBUTION_UNAVAILABLE)
        data_q = ev.data_quality.score or 0.0
        has_temp = ev.heating_gain.computable
        if not any(ev.get(n).computable for n in
                   ("heating_gain", "comfort_impact", "controller_quality")):
            return self._reject(BoostRejection.INSUFFICIENT_TEMPERATURE_DATA)
        if has_temp and data_q < p.min_data_quality:
            return self._reject(BoostRejection.LOW_DATA_QUALITY)
        for n in _DIMENSIONS:
            s = ev.get(n).score
            if s is not None and not is_finite(s):
                return self._reject(BoostRejection.NON_FINITE_EVALUATION)

        weight = clamp(episode.reliability * data_q * ev.effect.attribution_reliability,
                       0.0, p.max_sample_weight)
        if ev.partial:
            weight *= p.partial_weight_factor
        return UpdateEligibilityResult(
            accepted=True, reliability=episode.reliability, weight=weight,
            outlier_status=OutlierStatus.NONE, regime=Regime.UNKNOWN,
            source_episode_id=episode.episode_id,
            feature_extractor_version=self._extractor_version())

    def _reject(self, reason: BoostRejection) -> UpdateEligibilityResult:
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

    def update(self, episode: Any,
               context: Optional[BoostUpdateContext] = None) -> UpdateEligibilityResult:
        elig = self._eligibility(episode, context)
        if not elig.accepted:
            code = elig.confounder_flags[0] if elig.confounder_flags else "unknown"
            self._bump_rejection(code)
            # B2b-2: a rejected episode may still be a diagnostic boost outcome
            # (interrupted / confounded) — count it (no sample, no model effect).
            try:
                rej = BoostRejection(code)
                oc = _REJECTION_TO_CLASS.get(rej)
                if oc is not None:
                    occ = dict(self._state.outcome_class_counts)
                    occ[oc.value] = occ.get(oc.value, 0) + 1
                    self._state = replace(self._state, outcome_class_counts=occ)
            except ValueError:
                pass
            return elig

        ev = self.evaluate_boost(episode, context)
        weight = elig.weight
        # effective factor: requested offset scaled by realised comfort (overshoot
        # pulls the learned factor DOWN), so future recommendations stay conservative
        comfort = ev.comfort_impact.score if ev.comfort_impact.computable else 0.5
        effective_factor = clamp(ev.effect.requested_offset_c * comfort, 0.0,
                                 self._params.max_boost_offset_c)

        # B2b-4c: target/overshoot STABILITY drives the class (single in-band touch is not
        # "reached"; a short peak is not overshoot). Evaluated over the closed episode
        # trajectory — the REACHED stability dwell captures overshoot at/around target reach.
        from .boost_reliability import (evaluate_overshoot_stability, evaluate_target_stability)
        _pts = episode.trajectory.points
        _ahctx = bool(getattr(episode, "confounder_flags", ()) and
                      "afterheat_interaction" in episode.confounder_flags)
        _tstab = evaluate_target_stability(_pts, episode.target)
        _ostab = evaluate_overshoot_stability(_pts, episode.target, afterheat_context=_ahctx)
        # B2b-2: discrete primary class + its model-effect policy. Computed BEFORE the
        # outlier gate so neutral/diagnostic classes are still stored as observations.
        oc = classify_boost_outcome(
            dispatch_status=context.dispatch_status, authority=context.authority,
            confounder_flags=episode.confounder_flags,
            attribution=ev.effect.attribution_reliability, has_temp=ev.heating_gain.computable,
            overshoot_c=ev.effect.overshoot_c,
            has_baseline=context.expected_non_boost_duration_s is not None,
            relative_benefit=ev.effect.relative_benefit,
            target_reached=episode.reason is EpisodeReason.REACHED, params=self._params,
            is_partial=context.outcome_reliability == "partial",
            overshoot_stable=_ostab.stable,
            # target stability is the OutcomeEpisodeBuilder's authority (REACHED requires the
            # 600 s in-band dwell). The boost path consumes REACHED; _tstab is diagnostic.
            target_stable=episode.reason is EpisodeReason.REACHED)
        eff = class_effect(oc)
        # legacy fallback is diagnostic only: never an adaptive (factor) update.
        allow_factor = eff.factor_update and context.authority != "legacy_fallback"

        # B2b-4: typed multi-component reliability + class-dependent adaptive gate. A
        # blocking/interrupting confounder or below-class-minimum overall reliability blocks
        # the adaptive update entirely (positive SUCCESS demands more than negative OVERSHOOT).
        from .boost_reliability import (
            adaptive_allowed, compute_outcome_reliability, evaluate_confounders)
        _baseline_required = oc in (BoostOutcomeClass.BOOST_SUCCESS,
                                    BoostOutcomeClass.BOOST_NO_EFFECT,
                                    BoostOutcomeClass.BOOST_UNNECESSARY)
        _orel = context.outcome_reliability
        _conf_eval = evaluate_confounders(
            list(episode.confounder_flags or ()), partial_dispatch=(_orel == "partial"))
        reliability = compute_outcome_reliability(
            sensor_reliability=episode.reliability,
            trajectory_completeness=(ev.data_quality.score or 0.0),
            dispatch_reliability=(1.0 if _orel == "full" else 0.5 if _orel == "partial" else 1.0),
            attribution_reliability=ev.effect.attribution_reliability,
            baseline_reliability=(context.baseline_reliability
                                  if context.baseline_reliability is not None
                                  else (0.0 if _baseline_required else 1.0)),
            timing_reliability=(1.0 if ev.heating_gain.computable else 0.0),
            device_reliability=(0.5 if _orel == "partial" else 1.0),
            context_stability=1.0, confounders=_conf_eval,
            baseline_required=_baseline_required)
        if allow_factor and not adaptive_allowed(oc.value, reliability):
            allow_factor = False
        # B2b-4a: the confounder penalty enters the adaptive update weight EXACTLY ONCE
        # (single authority). One degrading root cause reduces the weight by 0.7; deduped.
        weight *= reliability.confounder_reliability

        g = self._state.general
        # neutral direction (no_effect / partial) must never INCREASE the learned factor,
        # and must never bootstrap it from nothing — only pull an existing value down.
        if allow_factor and eff.direction == "neutral":
            if g.effective_factor.has_evidence:
                effective_factor = min(effective_factor, g.effective_factor.value)
            else:
                allow_factor = False
        if allow_factor:
            if g.effective_factor.sample_count >= self._params.min_samples_for_outlier \
                    and g.effective_factor.has_evidence:
                z = outlier_z(effective_factor, g.effective_factor.value,
                              g.effective_factor.dispersion)
                if z > self._params.outlier_mad_k_severe:
                    self._bump_outlier("severe")
                    self._bump_rejection(BoostRejection.SEVERE_OUTLIER.value)
                    return self._reject(BoostRejection.SEVERE_OUTLIER)
                if z > self._params.outlier_mad_k_mild:
                    weight *= 0.25
                    self._bump_outlier("mild")
            new_general = self._apply(g, ev, effective_factor, weight)
            buckets = dict(self._state.buckets)
            bkey = _bucket_key(context.start_deficit_c, context.controller_kind,
                               ev.effect.requested_offset_c, self._params)
            if bkey is not None:
                b = buckets.get(bkey) or self._seed(bkey, new_general)
                buckets[bkey] = self._apply(b, ev, effective_factor, weight)
        else:
            # diagnostic / neutral class (confounded, interrupted, insufficient comparison,
            # legacy fallback): observation stored, learned dims untouched, no confidence gain.
            new_general = g
            buckets = dict(self._state.buckets)

        # adaptive == this sample actually moved the learned model. Non-adaptive outcomes
        # (legacy_fallback, partial, confounded, interrupted, insufficient) are stored for
        # diagnostics only and must NOT raise any adaptive maturity counter.
        adaptive = allow_factor
        overshoot_flag = (adaptive and ev.overshoot_penalty.computable
                          and ev.overshoot_penalty.score is not None
                          and ev.overshoot_penalty.score < 0.6)
        sample = BoostSample(
            source_episode_id=episode.episode_id, decision_id=episode.decision_id,
            learning_zone_id=self._zone, requested_offset_c=ev.effect.requested_offset_c,
            effective_factor_c=effective_factor,
            temperature_gain_c=ev.effect.temperature_gain_c, overshoot_c=ev.effect.overshoot_c,
            comfort_score=ev.comfort_impact.score, controller_score=ev.controller_quality.score,
            boost_duration_s=ev.effect.boost_duration_s,
            attribution_reliability=ev.effect.attribution_reliability, effective_weight=weight,
            reason=episode.reason.value, partial=ev.partial,
            evaluation_version=EVALUATION_VERSION, outcome_class=oc.value, adaptive=adaptive,
            reliability_known=True, reliability_overall=reliability.overall_reliability,
            reliability_components=reliability.to_dict())

        # B2b-5 / B2b-5a: fold the finalized outcome into the SCOPED degradation authority.
        # Order: normal bounded update (above) -> bucket tracker (local evidence) -> general
        # escalation (no double count) -> atomic per-scope rollback -> cooldown/readiness ->
        # per-scope known-good capture. The canonical bucket key is the SAME deterministic key
        # used for learning/prediction, frozen from the decision context.
        now_ts = _iso(episode.end_ts) or self._state.last_update_ts or ""
        deg_bkey = _bucket_key(context.start_deficit_c, context.controller_kind,
                               ev.effect.requested_offset_c, self._params)
        g_ef = new_general.effective_factor
        g_conf = self._confidence_value(g_ef.effective_n, not g_ef.has_evidence)
        b_bucket = buckets.get(deg_bkey) if deg_bkey is not None else None
        b_ef = b_bucket.effective_factor if b_bucket is not None else g_ef
        b_conf = self._confidence_value(b_ef.effective_n, not b_ef.has_evidence)
        scoped = observe_scoped(
            self._state.degradation, bucket_key=deg_bkey, outcome_class=oc.value,
            overshoot_c=ev.effect.overshoot_c, reliability=reliability.overall_reliability,
            attribution=ev.effect.attribution_reliability, adaptive=adaptive,
            reliability_known=True, now_ts=now_ts, bucket_factor=b_ef.value,
            bucket_confidence=b_conf, general_factor=g_ef.value, general_confidence=g_conf,
            params=self._degradation_params)
        deg_state = scoped.scoped
        if scoped.bucket_action.triggered and deg_bkey is not None and deg_bkey in buckets:
            buckets[deg_bkey] = self._apply_rollback(buckets[deg_bkey], scoped.bucket_action)
        if scoped.general_action.triggered:
            new_general = self._apply_rollback(new_general, scoped.general_action)
        # general known-good only from a stable, safety-free ELIGIBLE general scope
        if not scoped.general_action.triggered:
            ng = maybe_capture_known_good(
                deg_state.general, factor=new_general.effective_factor.value,
                effective_n=new_general.effective_factor.effective_n,
                dispersion=new_general.effective_factor.dispersion, confidence=g_conf,
                sample_weight=weight, now_ts=now_ts, model_version=MODEL_VERSION,
                provenance={"scope": "general",
                            "adaptive_outcomes": deg_state.general.adaptive_outcomes},
                params=self._degradation_params)
            deg_state = replace(deg_state, general=ng)

        processed = (self._state.processed_ids + (episode.episode_id,))[-self._params.dedup_max_ids:]
        processed_dec = (self._state.processed_decision_ids
                         + (episode.decision_id,))[-self._params.dedup_max_ids:]
        # adaptive samples feed confidence (recent_samples); non-adaptive ones are kept
        # in a SEPARATE bounded collection so they can never evict adaptive evidence.
        if adaptive:
            recent = (self._state.recent_samples + (sample,))[-self._params.research_sample_cap:]
            diagnostic = self._state.diagnostic_samples
        else:
            recent = self._state.recent_samples
            diagnostic = (self._state.diagnostic_samples
                          + (sample,))[-self._params.research_sample_cap:]
        rc = dict(self._state.reason_counts)
        rc[episode.reason.value] = rc.get(episode.reason.value, 0) + 1
        occ = dict(self._state.outcome_class_counts)
        occ[oc.value] = occ.get(oc.value, 0) + 1
        # adaptive-maturity counters / recency increment ONLY for adaptive samples
        self._state = replace(
            self._state, general=new_general, buckets=buckets, processed_ids=processed,
            processed_decision_ids=processed_dec, recent_samples=recent,
            diagnostic_samples=diagnostic, reason_counts=rc,
            outcome_class_counts=occ, degradation=deg_state,
            last_update_ts=_iso(episode.end_ts) if adaptive else self._state.last_update_ts,
            full_count=self._state.full_count + (1 if (adaptive and not ev.partial) else 0),
            partial_count=self._state.partial_count + (1 if (adaptive and ev.partial) else 0),
            overshoot_count=self._state.overshoot_count + (1 if overshoot_flag else 0),
            manual_correction_count=self._state.manual_correction_count
            + (1 if episode.reason is EpisodeReason.MANUAL_CORRECTION else 0))
        return elig

    def _apply_rollback(self, g: BoostBucket, rollback) -> BoostBucket:
        """B2b-5: roll the boost effective_factor back to the known-good snapshot, or — when
        no known good exists — to a safe ZERO-adaptive prior. Confidence (effective_n) is
        reduced; only the boost authority is touched (gain/overshoot/comfort/controller/
        duration dims are left as observed evidence; the TPI baseline and other models are
        untouched). No second factor authority is introduced."""
        kg = rollback.target_known_good
        if kg is not None:
            ef = _Dim(value=clamp(kg.factor, 0.0, self._params.max_boost_offset_c),
                      effective_n=min(g.effective_factor.effective_n, kg.effective_n),
                      dispersion=kg.dispersion, sample_count=g.effective_factor.sample_count)
        else:
            # no proven-good state -> deterministic zero-adaptive prior (never legacy values)
            ef = _Dim(value=0.0, effective_n=0.0,
                      dispersion=g.effective_factor.dispersion,
                      sample_count=g.effective_factor.sample_count)
        return replace(g, effective_factor=ef)

    def _apply(self, b: BoostBucket, ev: BoostEvaluation, effective_factor: float,
               weight: float) -> BoostBucket:
        def upd(dim, score):
            return _apply_dim(dim, score, weight) if score is not None else dim
        return BoostBucket(
            key=b.key, gain=upd(b.gain, ev.effect.temperature_gain_c),
            overshoot=upd(b.overshoot, ev.effect.overshoot_c),
            comfort=upd(b.comfort, ev.comfort_impact.score),
            controller=upd(b.controller, ev.controller_quality.score),
            effective_factor=_apply_dim(b.effective_factor, effective_factor, weight),
            duration=upd(b.duration, ev.effect.boost_duration_s),
            sample_count=b.sample_count + 1)

    def _seed(self, key: str, g: BoostBucket) -> BoostBucket:
        s = lambda d: _Dim(value=d.value, effective_n=self._params.bucket_seed_evidence,
                           dispersion=d.dispersion, sample_count=0)
        return BoostBucket(key=key, gain=s(g.gain), overshoot=s(g.overshoot),
                           comfort=s(g.comfort), controller=s(g.controller),
                           effective_factor=s(g.effective_factor), duration=s(g.duration),
                           sample_count=0)

    # -- predictions ----------------------------------------------------

    def _select(self, context: Optional[BoostPredictionContext]):
        p = self._params
        bkey = None
        if context is not None:
            bkey = _bucket_key(context.start_deficit_c, context.decision_type,
                               context.intensity_c, p)
        if bkey is not None:
            b = self._state.buckets.get(bkey)
            if b is not None and b.sample_count >= p.bucket_min_samples:
                return b, b.key, False, ("conditioned_bucket",)
        if self._state.general.has_evidence:
            return self._state.general, "general", False, ("general_state",)
        # device / profile prior
        if context is not None and context.device_prior_offset_c is not None:
            return None, "device_prior", True, ("device_prior",)
        return None, "safe_default", True, ("safe_default",)

    def predict_boost_factor(self, context: BoostPredictionContext) -> BoostRecommendation:
        p = self._params
        bucket, bkey, fallback, reasons = self._select(context)
        if bucket is not None:
            offset = clamp(bucket.effective_factor.value, 0.0, p.max_boost_offset_c)
            duration = bucket.duration.value if bucket.duration.has_evidence else None
            eff = bucket.effective_factor.effective_n
            learned = 1.0
        else:
            if bkey == "device_prior" and context is not None \
                    and context.device_prior_offset_c is not None:
                offset = clamp(context.device_prior_offset_c, 0.0, p.max_boost_offset_c)
            else:
                offset = p.default_boost_offset_c  # existing ThermoSmart safe default
            duration = p.default_boost_duration_s
            eff = 0.0
            learned = 0.0
        if duration is not None:
            duration = clamp(duration, p.min_duration_pred_s, p.max_duration_pred_s)
        conf = self._confidence_value(eff, fallback)
        adap_cap = self.adaptive_cap_c()
        # Adaptive cap applies to ALL predictions including device_prior.
        # device_prior is a cold-start hint (prior), NOT a user override command.
        # The safety hierarchy is: user_prior → adaptive_cap → confidence_gate → device_clamp.
        # Only the absolute device clamp (in DeviceAdapter) may override operator intent.
        offset = clamp(offset, 0.0, adap_cap)
        # B2b-5 / B2b-5a: SCOPE-AWARE readiness. The selected scope (bucket or general) must be
        # eligible for the factor to be usable; a non-eligible bucket falls back to an eligible
        # general state, else to the safe prior. Cooldown elapse alone never restores.
        now_ts = getattr(context, "now_ts", None)
        dp = self._degradation_params
        scoped = self._state.degradation
        if now_ts:
            scoped = resolve_scoped_cooldown(scoped, now_ts, dp)
        general = scoped.general
        sel_scope = "prior"
        scope_state = None
        if not fallback and bkey == "general":
            sel_scope, scope_state = "general", general
        elif not fallback and bkey not in (None, "device_prior", "safe_default"):
            scope_state = scoped.buckets.get(bkey)
            sel_scope = "bucket"
        bucket_readiness = scope_state.readiness if scope_state is not None else None
        # eligibility of the selected scope
        if sel_scope == "bucket":
            bucket_ok = scope_state is not None and is_scope_eligible(scope_state, now_ts or "", dp)
            general_ok = is_scope_eligible(general, now_ts or "", dp)
            if bucket_ok:
                usable, readiness, reason = True, scope_state.readiness, "bucket_eligible"
            elif general_ok:
                # B2b-5a: clear general fallback (do NOT use the impaired bucket factor); the
                # reported factor/confidence/evidence now reflect the GENERAL scope.
                usable, readiness, reason = True, general.readiness, "general_fallback"
                _gdim = self._state.general.effective_factor
                sel_scope = "general_fallback"
                offset = clamp(_gdim.value, 0.0, adap_cap)
                eff = _gdim.effective_n
                conf = self._confidence_value(eff, not _gdim.has_evidence)
            else:
                usable, readiness, reason = False, (
                    scope_state.readiness if scope_state else BoostLearningReadiness.SHADOW_LEARNING), "scope_not_eligible"
        elif sel_scope == "general":
            general_ok = is_scope_eligible(general, now_ts or "", dp)
            usable = general_ok
            readiness = general.readiness
            reason = "general_eligible" if general_ok else "scope_not_eligible"
        else:
            usable, readiness, reason = False, general.readiness, "prior"
        eff_state = scope_state if (sel_scope == "bucket" and scope_state) else general
        cooldown_active = readiness in (BoostLearningReadiness.ROLLED_BACK,
                                        BoostLearningReadiness.COOLDOWN)
        remaining = None
        if now_ts and eff_state.cooldown_until_ts:
            _u = _parse_iso(eff_state.cooldown_until_ts); _n = _parse_iso(now_ts)
            if _u is not None and _n is not None:
                remaining = max(0.0, (_u - _n).total_seconds())
        rehab = (clamp(eff_state.rehab_clean_count / dp.rehab_required_clean, 0.0, 1.0)
                 if dp.rehab_required_clean else 0.0)
        if cooldown_active or not usable:
            conf = min(conf, p.cold_start_confidence_cap)
        reasons = reasons + (sel_scope, readiness)
        return BoostRecommendation(
            boost_offset_c=round(offset, 4), recommended_duration_s=duration,
            confidence=conf, reliability=clamp(eff / p.full_confidence_samples, 0.0, 1.0),
            fallback_used=fallback or sel_scope in ("general_fallback", "prior"),
            prior_contribution=1.0 - learned,
            learned_contribution=learned, evidence_count=int(eff), bucket=bkey,
            missing_evidence=() if not fallback else ("boost_history",),
            reason_codes=reasons, adaptive_cap_c=adap_cap, selected_scope=sel_scope,
            bucket_handle=_hash_key(bkey), readiness=readiness, factor_usable=usable,
            rollback_active=eff_state.rollback_count > 0, cooldown_active=cooldown_active,
            cooldown_remaining_s=remaining, eligibility_reason=reason, rehab_progress=rehab,
            known_good_available=eff_state.known_good is not None)

    def predict_boost_duration(self, context: BoostPredictionContext) -> BoostRecommendation:
        return self.predict_boost_factor(context)

    def activation_readiness(self, context: Optional[BoostPredictionContext] = None):
        """B2b-7/8: the single authoritative LE2 boost ACTIVATION-READINESS contract.

        Pure-on-state: derived entirely from the persisted authoritative model state +
        runtime context. NOT activation, NOT user consent, NOT control. Unknown/legacy/missing
        inputs resolve to eligibility=False."""
        from .boost_activation import build_activation_readiness
        from .boost_degradation import _factor_stable
        if context is None:
            rec = self.predict_boost_factor(BoostPredictionContext())
            return build_activation_readiness(
                recommendation=rec, factor_stable=False, control_type=None, has_context=False,
                general_readiness=self._state.degradation.general.readiness)
        rec = self.predict_boost_factor(context)
        now_ts = getattr(context, "now_ts", None)
        scoped = self._state.degradation
        if now_ts:
            scoped = resolve_scoped_cooldown(scoped, now_ts, self._degradation_params)
        # effective scope state for the factor-stability gate
        if rec.selected_scope == "bucket":
            bkey = _bucket_key(context.start_deficit_c, context.decision_type,
                               context.intensity_c, self._params)
            scope_state = scoped.buckets.get(bkey)
        else:
            scope_state = scoped.general
        stable = (_factor_stable(scope_state, self._degradation_params)
                  if scope_state is not None else False)
        return build_activation_readiness(
            recommendation=rec, factor_stable=stable,
            control_type=getattr(context, "control_type", None), has_context=True,
            blocking_reasons=tuple(getattr(context, "blocking_reasons", ()) or ()),
            degrading_reasons=tuple(getattr(context, "degrading_reasons", ()) or ()),
            general_readiness=scoped.general.readiness)

    def predict(self, context: Any) -> Prediction:
        ctx = context if isinstance(context, BoostPredictionContext) else BoostPredictionContext()
        rec = self.predict_boost_factor(ctx)
        return Prediction(
            prediction_type=PredictionType.BOOST_FACTOR,
            values={"boost_factor": rec.boost_offset_c}, units={"boost_factor": "celsius_offset"},
            confidence=rec.confidence, reliability=rec.reliability, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION, prior_contribution=rec.prior_contribution,
            learned_contribution=rec.learned_contribution, fallback_used=rec.fallback_used,
            evidence_count=rec.evidence_count, bucket=rec.bucket,
            confidence_cap=self._params.cold_start_confidence_cap if rec.fallback_used else None,
            cap_reasons=("cold_start",) if rec.fallback_used else (),
            missing_evidence=rec.missing_evidence, reason_codes=rec.reason_codes)

    def predict_expected_outcome(self, context: BoostPredictionContext) -> Prediction:
        bucket, bkey, fallback, reasons = self._select(context)
        if bucket is not None:
            gain = max(0.0, bucket.gain.value)
            overshoot = max(0.0, bucket.overshoot.value)
            comfort = clamp(bucket.comfort.value, 0.0, 1.0)
            learned = 1.0
        else:
            gain, overshoot, comfort, learned = 0.5, 0.5, 0.5, 0.0
        return Prediction(
            prediction_type=PredictionType.BOOST_OUTCOME,
            values={"expected_gain": gain, "expected_overshoot": overshoot,
                    "expected_comfort": comfort},
            units={"expected_gain": "celsius", "expected_overshoot": "celsius",
                   "expected_comfort": "score"},
            confidence=self._confidence_value(
                bucket.effective_factor.effective_n if bucket else 0.0, fallback),
            reliability=0.5, model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
            prior_contribution=1.0 - learned, learned_contribution=learned,
            fallback_used=fallback, bucket=bkey,
            confidence_cap=self._params.cold_start_confidence_cap if fallback else None,
            cap_reasons=("cold_start",) if fallback else (), reason_codes=reasons)

    # -- confidence -----------------------------------------------------

    def _confidence_value(self, effective_evidence: float, fallback: bool) -> float:
        p = self._params
        if fallback or effective_evidence <= 0:
            return min(0.2, p.cold_start_confidence_cap)
        coverage = min(effective_evidence / p.full_confidence_samples, 1.0)
        disp = self._state.general.effective_factor.dispersion
        spread = clamp(1.0 - disp / max(p.max_boost_offset_c, 1.0), 0.3, 1.0)
        attr = self._mean_attribution() or 0.5
        return clamp(coverage * spread * attr, 0.0, 1.0)

    def _mean_attribution(self) -> Optional[float]:
        # Only adaptive samples contribute to adaptive confidence/attribution. Diagnostic
        # samples (legacy_fallback, partial, confounded, interrupted, insufficient) are
        # excluded so they can never raise adaptive maturity.
        vals = [s.attribution_reliability for s in self._state.recent_samples if s.adaptive]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def adaptive_cap_c(self) -> float:
        """Confidence- and outcome-adjusted LE 2.0 control cap, separate from technical 8°C limit.

        Tiers (derived from BoostParameters, mirroring ControlPolicy thresholds):
          < cold_start_confidence_cap (0.35): cap = 0.5°C  — cold start, very conservative
          < min_control_confidence   (0.55): cap = 1.0°C  — marginal confidence, cautious
          < 0.75:                            cap = 2.0°C  — moderate confidence
          >= 0.75:                           cap = adaptive_max_boost_c (3.0°C default)
        Overshoot penalty: if overshoot_rate > 0.3, cap *= 0.7 (outcomes mandate caution).
        Always bounded by [0, max_boost_offset_c] (technical TPI limit stays as backstop).
        """
        p = self._params
        conf = self._confidence_value(self._state.general.effective_factor.effective_n,
                                       not self._state.general.has_evidence)
        overshoot_rate = self._rate(self._state.overshoot_count) or 0.0
        if conf < p.cold_start_confidence_cap:
            cap = 0.5
        elif conf < p.min_control_confidence:
            cap = 1.0
        elif conf < 0.75:
            cap = min(2.0, p.adaptive_max_boost_c)
        else:
            cap = p.adaptive_max_boost_c
        if overshoot_rate > 0.3:
            cap *= 0.7
        return round(clamp(cap, 0.0, p.max_boost_offset_c), 4)

    # -- lifecycle (runtime only, not persisted) --------------------------------

    def apply_lifecycle(self, episode_id: str, applied_offset_c: float,
                        base_target_c: float, ts: str,
                        user_target_c: Optional[float] = None,
                        max_duration_s: Optional[float] = None) -> bool:
        """Transition lifecycle to APPLIED after boost is dispatched to TRV.

        Returns False (no-op) when episode binding blocks re-application of a failed episode.
        Returns True when the lifecycle was successfully applied.
        """
        # Episode binding: block same-episode retry after failure or successful completion.
        # Cleared only by clean episode change (schedule/mode/target change).
        if episode_id and episode_id == self._state.last_failed_episode_id:
            return False
        if episode_id and episode_id == self._state.last_completed_episode_id:
            return False
        max_dur = max_duration_s if max_duration_s is not None else self._params.max_boost_duration_s
        self._state = replace(
            self._state,
            lifecycle=BoostLifecycle.APPLIED,
            current_episode_id=episode_id,
            applied_offset_c=applied_offset_c,
            lifecycle_start_ts=ts,
            lifecycle_base_target_c=base_target_c,
            lifecycle_user_target_c=user_target_c,
            lifecycle_max_duration_s=max_dur,
            lifecycle_release_reason=None,
            cooldown_until_ts=None)
        return True

    def release_lifecycle(self, reason: str, ts: str) -> None:
        """Transition lifecycle to a terminal released/failed state and start cooldown.

        Cooldown is reason-specific:
        - target_reached / mode_change / schedule_change / early_cutoff → 0s
          (clean termination; new episode can boost immediately)
        - no_response / overshoot / heating_failure → failure_cooldown_duration_s (900s)
          (TRV may need time to recover; prevents rapid retry-pendling)
        - window_open / manual_override / timeout → cooldown_duration_s (300s)
          (external/neutral cause; short guard against same-cycle re-apply)
        """
        _release_map = {
            "target_reached": BoostLifecycle.RELEASED_TARGET_REACHED,
            "timeout": BoostLifecycle.RELEASED_TIMEOUT,
            "window_open": BoostLifecycle.RELEASED_WINDOW_OPEN,
            "manual_override": BoostLifecycle.RELEASED_MANUAL_OVERRIDE,
            "mode_change": BoostLifecycle.RELEASED_MODE_CHANGE,
            "schedule_change": BoostLifecycle.RELEASED_MODE_CHANGE,
            "early_cutoff": BoostLifecycle.RELEASED_MODE_CHANGE,
            "target_change": BoostLifecycle.RELEASED_MODE_CHANGE,
            "released_tpi_sufficient": BoostLifecycle.RELEASED_TPI_SUFFICIENT,
            "released_overshoot_risk": BoostLifecycle.RELEASED_OVERSHOOT_RISK,
            "released_afterheat_sufficient": BoostLifecycle.RELEASED_AFTERHEAT_SUFFICIENT,
        }
        _fail_map = {
            "no_response": BoostLifecycle.FAILED_NO_RESPONSE,
            "overshoot": BoostLifecycle.FAILED_OVERSHOOT,
            "heating_failure": BoostLifecycle.FAILED_NO_RESPONSE,
        }
        # 0s cooldown: clean termination, new episode can boost immediately.
        # "target_reached" is NOT in _CLEAN_CONTEXT_REASONS: it sets completed binding.
        _CLEAN_CONTEXT_REASONS = frozenset({
            "mode_change", "schedule_change", "early_cutoff", "target_change"})
        _NO_COOLDOWN_REASONS = frozenset({
            "target_reached", "mode_change", "schedule_change", "early_cutoff",
            "target_change"})
        _FAILURE_REASONS = frozenset({"no_response", "overshoot", "heating_failure"})
        # Deescalation reasons: successful mid-episode release (boost did its job).
        _DEESCALATION_REASONS = frozenset({
            "released_tpi_sufficient", "released_overshoot_risk", "released_afterheat_sufficient"})
        # Reasons that set completed-episode binding (block same episode from re-boosting).
        # manual_override: user intervened mid-episode; no automatic re-boost in same episode.
        _COMPLETED_REASONS = frozenset({"target_reached", "manual_override"}) | _DEESCALATION_REASONS

        lc = _release_map.get(reason) or _fail_map.get(reason) or BoostLifecycle.COMPLETED
        cooldown_until: Optional[str] = None
        try:
            from datetime import datetime, timedelta, timezone
            base_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if reason in _NO_COOLDOWN_REASONS or reason in _DEESCALATION_REASONS:
                cooldown_secs = 0.0
            elif reason in _FAILURE_REASONS:
                cooldown_secs = self._params.failure_cooldown_duration_s
            else:
                cooldown_secs = self._params.cooldown_duration_s
            if cooldown_secs > 0:
                cooldown_until = (base_ts + timedelta(seconds=cooldown_secs)).isoformat()
        except Exception:
            pass
        # Episode binding: track failed/completed episode IDs to block same-episode retry.
        # Clean context changes (schedule/mode/early-cutoff) clear both bindings.
        last_failed = self._state.last_failed_episode_id
        last_completed = self._state.last_completed_episode_id
        if reason in _FAILURE_REASONS:
            last_failed = self._state.current_episode_id
        elif reason in _CLEAN_CONTEXT_REASONS:
            last_failed = None
        if reason in _COMPLETED_REASONS:
            last_completed = self._state.current_episode_id
        elif reason in _CLEAN_CONTEXT_REASONS:
            last_completed = None
        self._state = replace(
            self._state, lifecycle=lc, lifecycle_release_reason=reason,
            cooldown_until_ts=cooldown_until,
            last_failed_episode_id=last_failed,
            last_completed_episode_id=last_completed)

    def reset_lifecycle(self) -> None:
        """Return lifecycle to INACTIVE (after cooldown expires or on clean restart)."""
        self._state = replace(
            self._state, lifecycle=BoostLifecycle.INACTIVE, current_episode_id=None,
            applied_offset_c=0.0, lifecycle_start_ts=None, lifecycle_base_target_c=None,
            lifecycle_user_target_c=None,
            lifecycle_max_duration_s=self._params.max_boost_duration_s,
            lifecycle_release_reason=None, cooldown_until_ts=None,
            last_failed_episode_id=None,
            last_completed_episode_id=None)

    def update_deescalation_offset(self, new_offset_c: float) -> None:
        """Reduce applied_offset_c in-place during soft step-down deescalation.

        Called exclusively by the soft deescalation path in adjust_recommendation_safe.
        Does NOT change lifecycle state — the lifecycle remains in RELEASED_xxx during
        step-down so episode binding continues to block new boosts.
        """
        self._state = replace(self._state, applied_offset_c=max(0.0, float(new_offset_c)))

    def cooldown_active(self, current_ts: str) -> bool:
        """True if the cooldown period following the last release is still running."""
        if not self._state.cooldown_until_ts:
            return False
        try:
            return current_ts < self._state.cooldown_until_ts
        except Exception:
            return False

    def confidence(self) -> ConfidenceContribution:
        g = self._state.general
        reasons: list[str] = []
        if not g.has_evidence:
            reasons.append("cold_start")
        attr = self._mean_attribution()
        if attr is not None and attr < 0.5:
            reasons.append("low_attribution")
        total = self._state.full_count + self._state.partial_count
        if total > 0 and self._state.partial_count / total > 0.5:
            reasons.append("mostly_partial")
        fallback = not g.has_evidence
        value = self._confidence_value(g.effective_factor.effective_n, fallback)
        return ConfidenceContribution(
            value=value, evidence_count=g.sample_count, reasons=tuple(reasons),
            component="boost",
            reliability=clamp(g.effective_factor.effective_n / self._params.full_confidence_samples,
                              0.0, 1.0),
            evidence_domains=("thermal_trajectory", "controller_events", "outcome_history"),
            source_count=len(self._state.buckets),
            prior_fraction=1.0 if fallback else 0.0,
            learned_fraction=0.0 if fallback else 1.0, fallback_used=fallback, freshness=1.0,
            status="healthy" if g.has_evidence else "fallback")

    # -- diagnostics / export -------------------------------------------

    def _rate(self, count: int) -> Optional[float]:
        n = self._state.general.sample_count
        return round(count / n, 4) if n > 0 else None

    def diagnostics(self) -> BoostDiagnostics:
        s = self._state
        g = s.general
        return BoostDiagnostics(
            general_recommended_offset_c=clamp(g.effective_factor.value, 0.0,
                                               self._params.max_boost_offset_c),
            general_recommended_duration_s=g.duration.value if g.duration.has_evidence else None,
            general_gain_c=g.gain.value, general_overshoot_c=g.overshoot.value,
            general_comfort=g.comfort.value, general_controller=g.controller.value,
            bucket_offsets={k: b.effective_factor.value for k, b in s.buckets.items()},
            sample_counts={"general": g.sample_count,
                           **{k: b.sample_count for k, b in s.buckets.items()}},
            full_partial=(s.full_count, s.partial_count),
            overshoot_rate=self._rate(s.overshoot_count),
            manual_correction_rate=self._rate(s.manual_correction_count),
            confidence=self.confidence().value, reason_counts=dict(s.reason_counts),
            rejection_counts=dict(s.rejection_counts), outlier_counts=dict(s.outlier_counts),
            missing_evidence=("conditioned_buckets",) if not s.buckets else (),
            last_update_ts=s.last_update_ts, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION, evaluation_version=EVALUATION_VERSION,
            outcome_class_counts=dict(s.outcome_class_counts),
            degradation=self.degradation_diagnostics())

    def degradation_diagnostics(self) -> Mapping[str, Any]:
        """B2b-5 / B2b-5a privacy-safe scoped readiness/degradation/rollback/cooldown summary
        (general scope + bounded per-bucket scopes; bucket keys are already non-entity)."""
        def _scope(d, factor):
            kg = d.known_good
            return {
                "readiness": d.readiness, "degradation_score": round(d.degradation_score, 4),
                "overshoot_evidence": round(d.overshoot_evidence, 4),
                "no_effect_evidence": round(d.no_effect_evidence, 4),
                "unnecessary_evidence": round(d.unnecessary_evidence, 4),
                "weighted_positive": round(d.weighted_positive, 4),
                "weighted_negative": round(d.weighted_negative, 4),
                "consecutive_negative_count": d.consecutive_negative_count,
                "adaptive_outcomes": d.adaptive_outcomes,
                "recent_overshoot_count": d.recent_overshoot_count,
                "rollback_count": d.rollback_count, "last_rollback_ts": d.last_rollback_ts,
                "rollback_scope": d.rollback_scope, "cooldown_start_ts": d.cooldown_start_ts,
                "cooldown_until_ts": d.cooldown_until_ts, "cooldown_reason": d.cooldown_reason,
                "rehab_clean_count": d.rehab_clean_count,
                "rehab_weight": round(d.rehab_weight, 4),
                "known_good_available": kg is not None,
                "known_good_factor": (kg.factor if kg else None),
                "current_factor": (round(factor, 4) if factor is not None else None),
                "reason": d.reason, "component_version": d.component_version,
            }
        s = self._state
        sc = s.degradation
        buckets = {h: _scope(st, (s.buckets[k].effective_factor.value
                                  if k in s.buckets else None))
                   for k, st in sc.buckets.items()
                   for h in (_hash_key(k),)}
        out = _scope(sc.general, clamp(s.general.effective_factor.value, 0.0,
                                       self._params.max_boost_offset_c))
        out["scope"] = "general"
        out["buckets"] = buckets
        out["degraded_bucket_count"] = sum(
            1 for st in sc.buckets.values()
            if st.readiness in (BoostLearningReadiness.DEGRADED,
                                BoostLearningReadiness.ROLLED_BACK,
                                BoostLearningReadiness.COOLDOWN))
        out["active_bucket_states"] = len(sc.buckets)
        return out

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        s = self._state
        g = s.general
        if scope is ExportScope.SUPPORT:
            return {
                "recommended_offset_c": clamp(g.effective_factor.value, 0.0,
                                              self._params.max_boost_offset_c),
                "recommended_duration_s": g.duration.value if g.duration.has_evidence else None,
                "gain_c": g.gain.value, "overshoot_c": g.overshoot.value,
                "comfort": g.comfort.value, "controller": g.controller.value,
                "overshoot_rate": self._rate(s.overshoot_count),
                "full_partial": [s.full_count, s.partial_count],
                "confidence": self.confidence().value, "sample_count": g.sample_count,
                "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
                "evaluation_version": EVALUATION_VERSION,
            }
        if scope is ExportScope.RESEARCH:
            return {
                "samples": [
                    {"requested_offset_c": x.requested_offset_c,
                     "effective_factor_c": x.effective_factor_c,
                     "temperature_gain_c": x.temperature_gain_c, "overshoot_c": x.overshoot_c,
                     "comfort": x.comfort_score, "controller": x.controller_score,
                     "boost_duration_s": x.boost_duration_s,
                     "attribution_reliability": x.attribution_reliability,
                     "partial": x.partial, "evaluation_version": x.evaluation_version}
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
        pairs: list[tuple[Any, BoostUpdateContext]] = []
        for it in (items or []):
            if isinstance(it, BoostRebuildItem):
                pairs.append((it.episode, it.context))
        pairs.sort(key=lambda pr: (_iso(getattr(pr[0], "start_ts", None)) or "",
                                   getattr(pr[0], "episode_id", "")))
        accepted = rejected = duplicates = 0
        rejection_counts: dict[str, int] = {}
        for episode, context in pairs:
            res = self.update(episode, context)
            if res.accepted:
                accepted += 1
            else:
                rejected += 1
                code = res.confounder_flags[0] if res.confounder_flags else "unknown"
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
                if code == BoostRejection.DUPLICATE_DECISION.value:
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
            "processed_ids": list(s.processed_ids),
            "processed_decision_ids": list(s.processed_decision_ids),
            "recent_samples": [_sample_dict(x) for x in s.recent_samples],
            "diagnostic_samples": [_sample_dict(x) for x in s.diagnostic_samples],
            "last_update_ts": s.last_update_ts, "full_count": s.full_count,
            "partial_count": s.partial_count, "overshoot_count": s.overshoot_count,
            "manual_correction_count": s.manual_correction_count,
            "reason_counts": dict(s.reason_counts), "rejection_counts": dict(s.rejection_counts),
            "outlier_counts": dict(s.outlier_counts),
            "outcome_class_counts": dict(s.outcome_class_counts),
            "degradation": s.degradation.to_dict(),       # B2b-5
        }

    def validate_state(self, state: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        if not isinstance(state, dict):
            return ["state is not a mapping"]
        if state.get("model_version") != MODEL_VERSION:
            errors.append("model_version mismatch")
        if state.get("learning_zone_id") != self._zone:
            errors.append("learning_zone_id mismatch")
        g = state.get("general", {})
        for dim in ("gain", "overshoot", "comfort", "controller", "effective_factor", "duration"):
            d = g.get(dim, {})
            for k in ("value", "effective_n", "dispersion"):
                v = d.get(k)
                if v is not None and not is_finite(v):
                    errors.append(f"non-finite general.{dim}.{k}")
        return errors

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        errors = self.validate_state(state)
        if errors:
            raise ValueError(f"invalid boost state: {errors}")
        self._state = BoostState(
            learning_zone_id=state["learning_zone_id"], general=_bucket_from(state["general"]),
            buckets={k: _bucket_from(b) for k, b in state.get("buckets", {}).items()},
            processed_ids=tuple(state.get("processed_ids", [])),
            processed_decision_ids=tuple(state.get("processed_decision_ids", [])),
            recent_samples=tuple(_sample_from(x) for x in state.get("recent_samples", [])),
            diagnostic_samples=tuple(
                _sample_from(x) for x in state.get("diagnostic_samples", [])),
            last_update_ts=state.get("last_update_ts"), full_count=state.get("full_count", 0),
            partial_count=state.get("partial_count", 0),
            overshoot_count=state.get("overshoot_count", 0),
            manual_correction_count=state.get("manual_correction_count", 0),
            reason_counts=dict(state.get("reason_counts", {})),
            rejection_counts=dict(state.get("rejection_counts", {})),
            outlier_counts=dict(state.get("outlier_counts", {})),
            outcome_class_counts=dict(state.get("outcome_class_counts", {})),
            degradation=BoostScopedDegradation.from_dict(state.get("degradation", {})))

    def migrate_state(self, old_model_version: int, old_parameter_version: int,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        if old_model_version == MODEL_VERSION:
            return state
        raise ValueError(
            f"no migration path from model_version {old_model_version} to {MODEL_VERSION}")


# -- helpers ------------------------------------------------------------------

def _has_temp(episode: OutcomeEpisode) -> bool:
    return any(not pt.gap and pt.value is not None for pt in episode.trajectory.points)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _hash_key(key: Optional[str]) -> Optional[str]:
    """Privacy-safe, stable, anonymised bucket-key handle for diagnostics/export."""
    if not key:
        return None
    import hashlib
    return hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:10]


def _sample_dict(x: BoostSample) -> dict:
    return {
        "source_episode_id": x.source_episode_id, "decision_id": x.decision_id,
        "learning_zone_id": x.learning_zone_id, "requested_offset_c": x.requested_offset_c,
        "effective_factor_c": x.effective_factor_c, "temperature_gain_c": x.temperature_gain_c,
        "overshoot_c": x.overshoot_c, "comfort_score": x.comfort_score,
        "controller_score": x.controller_score, "boost_duration_s": x.boost_duration_s,
        "attribution_reliability": x.attribution_reliability,
        "effective_weight": x.effective_weight, "reason": x.reason, "partial": x.partial,
        "evaluation_version": x.evaluation_version, "outcome_class": x.outcome_class,
        "adaptive": x.adaptive, "reliability_known": x.reliability_known,
        "reliability_overall": x.reliability_overall,
        "reliability_components": dict(x.reliability_components),
    }


def _sample_from(d: Mapping[str, Any]) -> BoostSample:
    return BoostSample(
        source_episode_id=d["source_episode_id"], decision_id=d["decision_id"],
        learning_zone_id=d["learning_zone_id"], requested_offset_c=d["requested_offset_c"],
        effective_factor_c=d["effective_factor_c"], temperature_gain_c=d.get("temperature_gain_c"),
        overshoot_c=d.get("overshoot_c"), comfort_score=d.get("comfort_score"),
        controller_score=d.get("controller_score"), boost_duration_s=d["boost_duration_s"],
        attribution_reliability=d["attribution_reliability"],
        effective_weight=d["effective_weight"], reason=d["reason"], partial=d["partial"],
        evaluation_version=d["evaluation_version"],
        outcome_class=d.get("outcome_class", BoostOutcomeClass.INSUFFICIENT_COMPARISON.value),
        adaptive=bool(d.get("adaptive", False)),
        # legacy sample missing the field -> unknown quality, conservative 0.0 (not 1.0).
        reliability_known=bool(d.get("reliability_known", False)),
        reliability_overall=float(d.get("reliability_overall", 0.0)),
        reliability_components=dict(d.get("reliability_components", {}) or {}))


def boost_model_definition():
    from ..registry import ModelDefinition

    return ModelDefinition(
        model_name=MODEL_NAME, model_factory=lambda: BoostModel("__definition__"),
        model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
        consumed_raw_tracks=(), consumed_episode_types=(EpisodeType.OUTCOME,),
        supported_prediction_types=(
            PredictionType.BOOST_FACTOR, PredictionType.BOOST_DURATION,
            PredictionType.BOOST_OUTCOME),
        required_features=("indoor_temp", "target"),
        optional_features=("outdoor_temp", "valve_opening", "hvac_action", "forecast"),
        required_capability_flags=(), regime_dependent=False, advisory=False,
        control_relevant=True, rebuildable=True, can_be_not_available=True, min_trv_only=True)
