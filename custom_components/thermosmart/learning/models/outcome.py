"""OutcomeModel for LE 2.0 (pure Python).

Evaluates heating decisions across four SEPARATE, reproducible dimensions —
physical target achievement, comfort quality, controller quality and data
quality — and learns their distributions over many outcomes. Data quality is
evidence quality, never blended in as a performance dimension. Scores are
recomputed from OutcomeEpisode + context + versioned parameters (never stored as
raw). Advisory: produces evaluation/calibration signals, no control output. No
HA / storage / coordinator dependency; imports no other model (only typed
prediction snapshots via the context).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

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
from .base import clamp, freshness_from_age, is_finite, outlier_z, robust_ema_update

MODEL_VERSION = 1
PARAMETER_VERSION = 1
SCORING_VERSION = 1
MODEL_NAME = "outcome"

_SUPPORTED_REASONS = frozenset(EpisodeReason)
_BLOCKING_REASONS = frozenset({EpisodeReason.DISTURBED})
_NOT_LEARNED_REASONS = frozenset({EpisodeReason.INSUFFICIENT_DATA})
_DIMENSIONS = ("physical_target", "comfort_quality", "controller_quality", "data_quality")
_PERFORMANCE_DIMS = ("physical_target", "comfort_quality", "controller_quality")


class OutcomeRejection(Enum):
    WRONG_EPISODE_TYPE = "wrong_episode_type"
    WRONG_ZONE = "wrong_zone"
    DISTURBED = "disturbed"
    INSUFFICIENT_DATA = "insufficient_data"
    MISSING_TARGET = "missing_target"
    MISSING_CONTEXT = "missing_context"
    INVALID_TRAJECTORY = "invalid_trajectory"
    LOW_DATA_QUALITY = "low_data_quality"
    UNSUPPORTED_REASON = "unsupported_reason"
    DECISION_MISMATCH = "decision_mismatch"
    DUPLICATE_EPISODE = "duplicate_episode"
    VERSION_MISMATCH = "version_mismatch"
    NON_FINITE_SCORE = "non_finite_score"
    INCONSISTENT_TIMESTAMPS = "inconsistent_timestamps"
    SEVERE_OUTLIER = "severe_outlier"
    CONTEXT_MISMATCH = "context_mismatch"


_TO_CONTRACT = {
    OutcomeRejection.WRONG_EPISODE_TYPE: RejectionReason.NOT_ELIGIBLE,
    OutcomeRejection.WRONG_ZONE: RejectionReason.NOT_ELIGIBLE,
    OutcomeRejection.DISTURBED: RejectionReason.DISTURBED_REGIME,
    OutcomeRejection.INSUFFICIENT_DATA: RejectionReason.MISSING_REQUIRED_DATA,
    OutcomeRejection.MISSING_TARGET: RejectionReason.MISSING_REQUIRED_DATA,
    OutcomeRejection.MISSING_CONTEXT: RejectionReason.MISSING_REQUIRED_DATA,
    OutcomeRejection.INVALID_TRAJECTORY: RejectionReason.MISSING_REQUIRED_DATA,
    OutcomeRejection.LOW_DATA_QUALITY: RejectionReason.LOW_RELIABILITY,
    OutcomeRejection.UNSUPPORTED_REASON: RejectionReason.NOT_ELIGIBLE,
    OutcomeRejection.DECISION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    OutcomeRejection.DUPLICATE_EPISODE: RejectionReason.DUPLICATE,
    OutcomeRejection.VERSION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    OutcomeRejection.NON_FINITE_SCORE: RejectionReason.OUTLIER,
    OutcomeRejection.INCONSISTENT_TIMESTAMPS: RejectionReason.NOT_ELIGIBLE,
    OutcomeRejection.SEVERE_OUTLIER: RejectionReason.OUTLIER,
    OutcomeRejection.CONTEXT_MISMATCH: RejectionReason.NOT_ELIGIBLE,
}


class DimensionStatus(Enum):
    OK = "ok"
    PARTIAL = "partial"
    NOT_COMPUTABLE = "not_computable"


@dataclass(frozen=True)
class OutcomeParameters:
    min_data_quality: float = 0.25
    min_points: int = 3
    max_gap_ratio: float = 0.6
    max_sample_weight: float = 1.0
    bucket_min_samples: int = 5
    bucket_seed_evidence: float = 1.0
    outlier_mad_k_mild: float = 3.0
    outlier_mad_k_severe: float = 6.0
    min_samples_for_outlier: int = 5
    dedup_max_ids: int = 500
    research_sample_cap: int = 50
    # performance weights (renormalised over computable dims)
    weight_physical: float = 0.45
    weight_comfort: float = 0.35
    weight_controller: float = 0.20
    duration_tolerance: float = 1.5      # real/expected ratio still "on time"
    overshoot_comfort_scale: float = 2.0
    full_confidence_samples: float = 20.0
    cold_start_confidence_cap: float = 0.4
    partial_weight_factor: float = 0.5
    parameter_version: int = PARAMETER_VERSION


# -- evaluation result types --------------------------------------------------

@dataclass(frozen=True)
class OutcomeScore:
    name: str
    score: Optional[float]
    reliability: float
    status: DimensionStatus
    reason_codes: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    caps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be within [0,1]")
        if self.status is DimensionStatus.NOT_COMPUTABLE:
            if self.score is not None:
                raise ValueError("not-computable dimension must not carry a score")
        else:
            if self.score is None or not is_finite(self.score) or not 0.0 <= self.score <= 1.0:
                raise ValueError(f"{self.name}: score must be finite within [0,1]")

    @property
    def computable(self) -> bool:
        return self.status is not DimensionStatus.NOT_COMPUTABLE


@dataclass(frozen=True)
class OutcomeDimensions:
    physical_target: OutcomeScore
    comfort_quality: OutcomeScore
    controller_quality: OutcomeScore
    data_quality: OutcomeScore

    def get(self, name: str) -> OutcomeScore:
        return getattr(self, name)


@dataclass(frozen=True)
class OutcomeEvaluation:
    dimensions: OutcomeDimensions
    performance_score: Optional[float]
    overall_reliability: float
    reason: EpisodeReason
    scoring_version: int = SCORING_VERSION
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.overall_reliability <= 1.0:
            raise ValueError("overall_reliability must be within [0,1]")
        if self.performance_score is not None and (
                not is_finite(self.performance_score)
                or not 0.0 <= self.performance_score <= 1.0):
            raise ValueError("performance_score must be finite within [0,1] or None")


@dataclass(frozen=True)
class OutcomeUpdateContext:
    source_episode_id: str
    learning_zone_id: Optional[str] = None
    decision_id: Optional[str] = None
    controller_kind: Optional[str] = None
    expected_duration_s: Optional[float] = None
    target_temperature_c: Optional[float] = None
    comfort_tolerance_c: Optional[float] = None
    start_temperature_c: Optional[float] = None
    heat_rate_c_per_h: Optional[float] = None
    heat_loss_c_per_h: Optional[float] = None
    afterheat_expected_overshoot_c: Optional[float] = None
    preheat_minutes: Optional[float] = None
    outdoor_bucket: Optional[str] = None
    mode_context: Optional[str] = None
    quality: DataQuality = DataQuality.OK
    source_feature_version: int = 1
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_episode_id:
            raise ValueError("source_episode_id must be non-empty")
        for name in ("expected_duration_s", "target_temperature_c", "comfort_tolerance_c",
                     "start_temperature_c", "heat_rate_c_per_h", "heat_loss_c_per_h",
                     "afterheat_expected_overshoot_c", "preheat_minutes"):
            v = getattr(self, name)
            if v is not None and not is_finite(v):
                raise ValueError(f"{name} must be finite (or None)")

    def to_dict(self) -> dict:
        return {
            "source_episode_id": self.source_episode_id,
            "learning_zone_id": self.learning_zone_id, "decision_id": self.decision_id,
            "controller_kind": self.controller_kind,
            "expected_duration_s": self.expected_duration_s,
            "target_temperature_c": self.target_temperature_c,
            "comfort_tolerance_c": self.comfort_tolerance_c,
            "start_temperature_c": self.start_temperature_c,
            "heat_rate_c_per_h": self.heat_rate_c_per_h,
            "heat_loss_c_per_h": self.heat_loss_c_per_h,
            "afterheat_expected_overshoot_c": self.afterheat_expected_overshoot_c,
            "preheat_minutes": self.preheat_minutes, "outdoor_bucket": self.outdoor_bucket,
            "mode_context": self.mode_context, "quality": self.quality.value,
            "source_feature_version": self.source_feature_version,
            "source_refs": list(self.source_refs),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OutcomeUpdateContext":
        return cls(
            source_episode_id=d["source_episode_id"],
            learning_zone_id=d.get("learning_zone_id"), decision_id=d.get("decision_id"),
            controller_kind=d.get("controller_kind"),
            expected_duration_s=d.get("expected_duration_s"),
            target_temperature_c=d.get("target_temperature_c"),
            comfort_tolerance_c=d.get("comfort_tolerance_c"),
            start_temperature_c=d.get("start_temperature_c"),
            heat_rate_c_per_h=d.get("heat_rate_c_per_h"),
            heat_loss_c_per_h=d.get("heat_loss_c_per_h"),
            afterheat_expected_overshoot_c=d.get("afterheat_expected_overshoot_c"),
            preheat_minutes=d.get("preheat_minutes"), outdoor_bucket=d.get("outdoor_bucket"),
            mode_context=d.get("mode_context"),
            quality=DataQuality(d.get("quality", DataQuality.OK.value)),
            source_feature_version=d.get("source_feature_version", 1),
            source_refs=tuple(d.get("source_refs", [])))


@dataclass(frozen=True)
class OutcomeRebuildItem:
    episode: Any
    context: Optional[OutcomeUpdateContext] = None


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


@dataclass(frozen=True)
class OutcomeBucket:
    key: str
    physical: _Dim = field(default_factory=_Dim)
    comfort: _Dim = field(default_factory=_Dim)
    controller: _Dim = field(default_factory=_Dim)
    data_quality: _Dim = field(default_factory=_Dim)
    sample_count: int = 0

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0


def _empty_bucket(key: str) -> OutcomeBucket:
    return OutcomeBucket(key=key)


@dataclass(frozen=True)
class OutcomeSample:
    source_episode_id: str
    decision_id: Optional[str]
    learning_zone_id: str
    physical_score: Optional[float]
    comfort_score: Optional[float]
    controller_score: Optional[float]
    data_quality_score: float
    performance_score: Optional[float]
    overall_reliability: float
    effective_weight: float
    reason: str
    controller_kind: Optional[str]
    partial: bool
    scoring_version: int
    feature_extractor_version: int
    builder_version: int
    classifier_version: int
    reason_codes: tuple[str, ...]
    confounder_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutcomeState:
    learning_zone_id: str
    general: OutcomeBucket
    buckets: Mapping[str, OutcomeBucket] = field(default_factory=dict)
    processed_ids: tuple[str, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()
    recent_samples: tuple[OutcomeSample, ...] = ()
    aggregate_reliability: float = 0.0
    last_update_ts: Optional[str] = None
    reason_counts: Mapping[str, int] = field(default_factory=dict)
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    reached_count: int = 0
    timeout_count: int = 0
    overshoot_count: int = 0
    manual_correction_count: int = 0
    full_count: int = 0
    partial_count: int = 0
    dedup_count: int = 0
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION
    scoring_version: int = SCORING_VERSION


@dataclass(frozen=True)
class OutcomeDiagnostics:
    general_physical: float
    general_comfort: float
    general_controller: float
    general_data_quality: float
    bucket_performance: Mapping[str, float]
    sample_counts: Mapping[str, int]
    reached_rate: Optional[float]
    timeout_rate: Optional[float]
    overshoot_rate: Optional[float]
    manual_correction_rate: Optional[float]
    full_partial: tuple[int, int]
    confidence: float
    reason_counts: Mapping[str, int]
    rejection_counts: Mapping[str, int]
    outlier_counts: Mapping[str, int]
    missing_optional_evidence: tuple[str, ...]
    last_update_ts: Optional[str]
    model_version: int
    parameter_version: int
    scoring_version: int


@dataclass(frozen=True)
class OutcomePredictionContext:
    controller_kind: Optional[str] = None
    start_deficit_c: Optional[float] = None
    outdoor_bucket: Optional[str] = None


def _bucket_key_for(controller_kind: Optional[str]) -> Optional[str]:
    if controller_kind:
        return f"controller:{controller_kind}"
    return None


# -- model --------------------------------------------------------------------

class OutcomeModel:
    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    parameter_version = PARAMETER_VERSION
    consumed_raw_tracks: tuple[str, ...] = ()
    consumed_episode_types = (EpisodeType.OUTCOME.value,)
    required_features = ("indoor_temp", "target")
    optional_features = ("outdoor_temp", "valve_opening", "hvac_action", "forecast",
                         "wind_speed", "solar_radiation")
    supported_prediction_types = (
        PredictionType.OUTCOME_QUALITY, PredictionType.COMFORT_QUALITY,
        PredictionType.CONTROLLER_QUALITY)
    min_tier = 0

    def __init__(self, learning_zone_id: str, params: Optional[OutcomeParameters] = None, *,
                 extractor: Optional[FeatureExtractor] = None) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or OutcomeParameters()
        self._extractor = extractor or FeatureExtractor()
        self._state = self._empty_state()

    def cold_start_prior(self) -> Mapping[str, Any]:
        return {"general_performance": None}

    def _empty_state(self) -> OutcomeState:
        return OutcomeState(learning_zone_id=self._zone, general=_empty_bucket("general"))

    def _extractor_version(self) -> int:
        from ..features import FEATURE_EXTRACTOR_VERSION
        return FEATURE_EXTRACTOR_VERSION

    # -- deterministic evaluation (pure, no state mutation) -------------

    def evaluate_outcome(self, episode: OutcomeEpisode,
                         context: Optional[OutcomeUpdateContext] = None) -> OutcomeEvaluation:
        p = self._params
        fs = self._extractor.extract_trajectory_features(episode.trajectory)
        comfort_fs = self._extractor.extract_comfort_features(
            episode.trajectory, target=episode.target,
            comfort_tolerance=episode.comfort_tolerance_at_start)
        valid = int(fs.get(FeatureName.TRAJ_VALID_POINTS).value or 0)
        gap_ratio = fs.get(FeatureName.TRAJ_GAP_RATIO).value or 0.0
        plausible = fs.get(FeatureName.TRAJ_PLAUSIBILITY).status is not FeatureStatus.INVALID
        has_temp = valid >= 1

        data = self._score_data_quality(episode, valid, gap_ratio, plausible, context)
        physical = self._score_physical(episode, fs, comfort_fs, context, has_temp)
        comfort = self._score_comfort(comfort_fs, has_temp)
        controller = self._score_controller(episode, comfort_fs, context)

        dims = OutcomeDimensions(physical_target=physical, comfort_quality=comfort,
                                 controller_quality=controller, data_quality=data)
        performance = self._performance(dims)
        overall_rel = clamp(data.score * _mean([d.reliability for d in
                                                (physical, comfort, controller) if d.computable]
                                               or [data.reliability]), 0.0, 1.0)
        return OutcomeEvaluation(dimensions=dims, performance_score=performance,
                                 overall_reliability=overall_rel, reason=episode.reason,
                                 reason_codes=(episode.reason.value,))

    def _score_data_quality(self, episode, valid, gap_ratio, plausible, context) -> OutcomeScore:
        p = self._params
        missing = []
        if context is None:
            missing.append("update_context")
        coverage = clamp(valid / max(p.min_points, 1), 0.0, 1.0)
        score = episode.reliability * (1.0 - 0.5 * gap_ratio) * (1.0 if plausible else 0.3) \
            * (0.5 + 0.5 * coverage)
        score = clamp(score, 0.0, 1.0)
        status = DimensionStatus.OK if valid >= 1 else DimensionStatus.PARTIAL
        reasons = [] if plausible else ["implausible"]
        if valid < 1:
            reasons.append("no_trajectory")
        return OutcomeScore("data_quality", score, reliability=clamp(score, 0.0, 1.0),
                            status=status, reason_codes=tuple(reasons),
                            features=("traj_valid_points", "traj_gap_ratio"),
                            missing_evidence=tuple(missing))

    def _score_physical(self, episode, fs, comfort_fs, context, has_temp) -> OutcomeScore:
        if not has_temp:
            return OutcomeScore("physical_target", None, reliability=0.0,
                                status=DimensionStatus.NOT_COMPUTABLE,
                                reason_codes=("no_indoor_temp",),
                                missing_evidence=("indoor_temp",))
        target = episode.target
        tol = episode.comfort_tolerance_at_start
        first_target = comfort_fs.get(FeatureName.TIME_TO_FIRST_TARGET)
        in_band = comfort_fs.get(FeatureName.TIME_IN_COMFORT_BAND)
        duration = fs.get(FeatureName.TRAJ_DURATION).value or 0.0
        reached = first_target.is_present or episode.reason is EpisodeReason.REACHED
        reasons: list[str] = []
        if not reached:
            # partial credit for progress toward target
            start = episode.start_temp
            peak = fs.get(FeatureName.TRAJ_PEAK).value
            denom = max(target - start, 0.1)
            progress = clamp((peak - start) / denom, 0.0, 1.0) if denom > 0 else 0.0
            reasons.append("target_not_reached")
            return OutcomeScore("physical_target", round(0.5 * progress, 4),
                                reliability=0.6, status=DimensionStatus.OK,
                                reason_codes=tuple(reasons),
                                features=("traj_peak",))
        # reached: combine timeliness + held
        timely = 1.0
        if context is not None and context.expected_duration_s:
            ratio = duration / max(context.expected_duration_s, 1.0)
            timely = 1.0 if ratio <= self._params.duration_tolerance else \
                clamp(1.0 - (ratio - self._params.duration_tolerance) * 0.5, 0.3, 1.0)
            reasons.append("timeliness_evaluated")
        else:
            reasons.append("no_expected_duration")
        end_dev = abs(episode.end_temp - target)
        held = clamp(1.0 - end_dev / 2.0, 0.0, 1.0)
        if in_band.is_present and duration > 0:
            held = max(held, clamp(in_band.value / duration, 0.0, 1.0))
        if episode.reason is EpisodeReason.REACHED:
            reasons.append("reached")
        score = clamp(0.5 + 0.25 * timely + 0.25 * held, 0.0, 1.0)
        return OutcomeScore("physical_target", round(score, 4), reliability=0.8,
                            status=DimensionStatus.OK, reason_codes=tuple(reasons),
                            features=("time_to_first_target", "time_in_comfort_band"))

    def _score_comfort(self, comfort_fs, has_temp) -> OutcomeScore:
        if not has_temp:
            return OutcomeScore("comfort_quality", None, reliability=0.0,
                                status=DimensionStatus.NOT_COMPUTABLE,
                                reason_codes=("no_indoor_temp",),
                                missing_evidence=("indoor_temp",))
        p = self._params
        # Comfort reflects behaviour around/after the target. The warm-up deficit
        # (UNDERSHOOT / time-below-band during the ramp) is expected and is NOT
        # penalised here; post-target shortfalls are captured by
        # STABILITY_AFTER_TARGET and LATE_DROP.
        overshoot = comfort_fs.get(FeatureName.OVERSHOOT)
        stability = comfort_fs.get(FeatureName.STABILITY_AFTER_TARGET)
        oscillation = comfort_fs.get(FeatureName.OSCILLATION)
        late_drop = comfort_fs.get(FeatureName.LATE_DROP)

        def term(feat, scale):
            return clamp(1.0 - (feat.value / scale), 0.0, 1.0) if feat.is_present else 1.0

        overshoot_term = term(overshoot, p.overshoot_comfort_scale)
        stability_term = term(stability, 2.0)
        late_term = term(late_drop, 2.0)
        osc_term = clamp(1.0 - (oscillation.value / 5.0), 0.0, 1.0) \
            if oscillation.is_present else 1.0
        score = _mean([overshoot_term, stability_term, late_term, osc_term])
        return OutcomeScore("comfort_quality", round(score, 4), reliability=0.8,
                            status=DimensionStatus.OK,
                            reason_codes=("comfort_evaluated",),
                            features=("overshoot", "stability_after_target",
                                      "oscillation", "late_drop"))

    def _score_controller(self, episode, comfort_fs, context) -> OutcomeScore:
        p = self._params
        reason = episode.reason
        reasons: list[str] = [reason.value]
        # reasons that are not attributable to the controller -> not computable
        if reason in (EpisodeReason.MODE_CHANGE, EpisodeReason.SUPERSEDED):
            return OutcomeScore("controller_quality", None, reliability=0.0,
                                status=DimensionStatus.NOT_COMPUTABLE,
                                reason_codes=tuple(reasons + ["not_attributable"]))
        base = 0.7
        if reason is EpisodeReason.REACHED:
            base = 0.85
        elif reason is EpisodeReason.TIMEOUT:
            base = 0.4
        elif reason is EpisodeReason.MANUAL_CORRECTION:
            base = 0.45
            reasons.append("user_disagreed")
        elif reason is EpisodeReason.INTERRUPTED:
            base = 0.6

        # duration vs expectation (controller decision quality, not building)
        duration = (episode.end_ts - episode.start_ts).total_seconds()
        if context is not None and context.expected_duration_s:
            ratio = duration / max(context.expected_duration_s, 1.0)
            if ratio > p.duration_tolerance:
                base *= clamp(1.0 - (ratio - p.duration_tolerance) * 0.2, 0.5, 1.0)
                reasons.append("longer_than_expected")

        # avoidable overshoot: only penalise overshoot beyond the expected afterheat
        overshoot = comfort_fs.get(FeatureName.OVERSHOOT)
        if overshoot.is_present and overshoot.value > 0:
            expected = context.afterheat_expected_overshoot_c if context is not None else None
            avoidable = overshoot.value - (expected or 0.0)
            if avoidable > 0:
                base *= clamp(1.0 - avoidable / 2.0, 0.4, 1.0)
                reasons.append("avoidable_overshoot")
            else:
                reasons.append("overshoot_within_physics")

        partial = not _has_temp_context(episode)
        status = DimensionStatus.PARTIAL if partial else DimensionStatus.OK
        return OutcomeScore("controller_quality", round(clamp(base, 0.0, 1.0), 4),
                            reliability=0.7 if not partial else 0.5, status=status,
                            reason_codes=tuple(reasons))

    def _performance(self, dims: OutcomeDimensions) -> Optional[float]:
        p = self._params
        weights = {"physical_target": p.weight_physical, "comfort_quality": p.weight_comfort,
                   "controller_quality": p.weight_controller}
        total_w = 0.0
        acc = 0.0
        for name in _PERFORMANCE_DIMS:
            d = dims.get(name)
            if d.computable and d.score is not None:
                total_w += weights[name]
                acc += weights[name] * d.score
        if total_w <= 0:
            return None
        return round(clamp(acc / total_w, 0.0, 1.0), 4)

    # -- eligibility / update -------------------------------------------

    def update_eligibility(self, episode: Any) -> UpdateEligibilityResult:
        return self._eligibility(episode, None)

    def _eligibility(self, episode, context) -> UpdateEligibilityResult:
        p = self._params
        if not isinstance(episode, OutcomeEpisode):
            return self._reject(OutcomeRejection.WRONG_EPISODE_TYPE)
        if episode.learning_zone_id != self._zone:
            return self._reject(OutcomeRejection.WRONG_ZONE)
        if episode.episode_schema_version != 1:
            return self._reject(OutcomeRejection.VERSION_MISMATCH)
        if episode.reason not in _SUPPORTED_REASONS:
            return self._reject(OutcomeRejection.UNSUPPORTED_REASON)
        if episode.reason in _BLOCKING_REASONS:
            return self._reject(OutcomeRejection.DISTURBED)
        if episode.reason in _NOT_LEARNED_REASONS:
            return self._reject(OutcomeRejection.INSUFFICIENT_DATA)
        if episode.target is None:
            return self._reject(OutcomeRejection.MISSING_TARGET)
        if episode.end_ts < episode.start_ts:
            return self._reject(OutcomeRejection.INCONSISTENT_TIMESTAMPS)
        # the typed context must agree with the episode's authoritative decision id
        if context is not None and context.decision_id is not None \
                and context.decision_id != episode.decision_id:
            return self._reject(OutcomeRejection.DECISION_MISMATCH)
        if episode.episode_id in self._state.processed_ids:
            return self._reject(OutcomeRejection.DUPLICATE_EPISODE)
        # a replay of the same controller decision must not learn twice
        if episode.decision_id in self._state.processed_decision_ids:
            return self._reject(OutcomeRejection.DUPLICATE_EPISODE)

        evaluation = self.evaluate_outcome(episode, context)
        data_q = evaluation.dimensions.data_quality.score or 0.0
        has_temp = evaluation.dimensions.physical_target.computable
        # at least one performance dimension must be computable
        if not any(evaluation.dimensions.get(n).computable for n in _PERFORMANCE_DIMS):
            return self._reject(OutcomeRejection.INSUFFICIENT_DATA)
        # The data-quality gate filters noisy TEMPERATURE-based outcomes. A
        # controller-only partial outcome (no indoor temp) is intentionally low
        # data quality yet still useful, so it is not rejected on that basis.
        if has_temp and data_q < p.min_data_quality:
            return self._reject(OutcomeRejection.LOW_DATA_QUALITY)
        # non-finite guard
        for n in _DIMENSIONS:
            s = evaluation.dimensions.get(n).score
            if s is not None and not is_finite(s):
                return self._reject(OutcomeRejection.NON_FINITE_SCORE)

        partial = evaluation.dimensions.physical_target.status is DimensionStatus.NOT_COMPUTABLE
        weight = clamp(episode.reliability * data_q, 0.0, p.max_sample_weight)
        if partial:
            weight *= p.partial_weight_factor
        return UpdateEligibilityResult(
            accepted=True, reliability=episode.reliability, weight=weight,
            outlier_status=OutlierStatus.NONE, regime=Regime.UNKNOWN,
            source_episode_id=episode.episode_id,
            feature_extractor_version=self._extractor_version())

    def _reject(self, reason: OutcomeRejection) -> UpdateEligibilityResult:
        return UpdateEligibilityResult(
            accepted=False, reliability=0.0, weight=0.0,
            rejection_reason=_TO_CONTRACT[reason], confounder_flags=(reason.value,))

    def update(self, episode: Any,
               context: Optional[OutcomeUpdateContext] = None) -> UpdateEligibilityResult:
        if context is not None:
            if context.source_episode_id != episode.episode_id or (
                    context.learning_zone_id is not None
                    and context.learning_zone_id != self._zone):
                self._bump_rejection(OutcomeRejection.CONTEXT_MISMATCH.value)
                return self._reject(OutcomeRejection.CONTEXT_MISMATCH)

        elig = self._eligibility(episode, context)
        if not elig.accepted:
            self._bump_rejection(elig.confounder_flags[0] if elig.confounder_flags else "unknown")
            if elig.confounder_flags and elig.confounder_flags[0] == \
                    OutcomeRejection.DUPLICATE_EPISODE.value:
                self._state = replace(self._state, dedup_count=self._state.dedup_count + 1)
            return elig

        ev = self.evaluate_outcome(episode, context)
        dims = ev.dimensions
        weight = elig.weight

        # severe outlier vs zone evidence on the performance score
        g = self._state.general
        if ev.performance_score is not None and g.physical.sample_count >= \
                self._params.min_samples_for_outlier and g.physical.has_evidence:
            z = outlier_z(ev.performance_score, g.physical.value, g.physical.dispersion)
            if z > self._params.outlier_mad_k_severe:
                self._bump_outlier("severe")
                return self._reject(OutcomeRejection.SEVERE_OUTLIER)
            if z > self._params.outlier_mad_k_mild:
                weight *= 0.25
                self._bump_outlier("mild")

        new_general = self._apply_bucket(g, dims, weight)
        new_buckets = dict(self._state.buckets)
        ck = context.controller_kind if context is not None else \
            (episode.controller.value if episode.controller else None)
        bkey = _bucket_key_for(ck)
        if bkey is not None:
            bkt = new_buckets.get(bkey) or self._seed_bucket(bkey, new_general)
            new_buckets[bkey] = self._apply_bucket(bkt, dims, weight)

        partial = dims.physical_target.status is DimensionStatus.NOT_COMPUTABLE
        overshoot_flag = (dims.comfort_quality.computable and dims.comfort_quality.score is not None
                          and dims.comfort_quality.score < 0.6)
        sample = OutcomeSample(
            source_episode_id=episode.episode_id,
            decision_id=episode.decision_id, learning_zone_id=self._zone,
            physical_score=dims.physical_target.score, comfort_score=dims.comfort_quality.score,
            controller_score=dims.controller_quality.score,
            data_quality_score=dims.data_quality.score, performance_score=ev.performance_score,
            overall_reliability=ev.overall_reliability, effective_weight=weight,
            reason=episode.reason.value,
            controller_kind=episode.controller.value if episode.controller else None,
            partial=partial, scoring_version=SCORING_VERSION,
            feature_extractor_version=self._extractor_version(),
            builder_version=episode.builder_version,
            classifier_version=episode.classifier_version, reason_codes=ev.reason_codes,
            confounder_flags=tuple(getattr(episode, "confounder_flags", ())))

        processed = (self._state.processed_ids + (episode.episode_id,))[-self._params.dedup_max_ids:]
        processed_dec = (self._state.processed_decision_ids
                         + (episode.decision_id,))[-self._params.dedup_max_ids:]
        recent = (self._state.recent_samples + (sample,))[-self._params.research_sample_cap:]
        n = new_general.sample_count
        agg = (self._state.aggregate_reliability * (n - 1) + ev.overall_reliability) / n
        rc = dict(self._state.reason_counts)
        rc[episode.reason.value] = rc.get(episode.reason.value, 0) + 1
        self._state = replace(
            self._state, general=new_general, buckets=new_buckets, processed_ids=processed,
            processed_decision_ids=processed_dec,
            recent_samples=recent, aggregate_reliability=agg, reason_counts=rc,
            last_update_ts=_iso(episode.end_ts),
            reached_count=self._state.reached_count + (1 if episode.reason is EpisodeReason.REACHED else 0),
            timeout_count=self._state.timeout_count + (1 if episode.reason is EpisodeReason.TIMEOUT else 0),
            overshoot_count=self._state.overshoot_count + (1 if overshoot_flag else 0),
            manual_correction_count=self._state.manual_correction_count
            + (1 if episode.reason is EpisodeReason.MANUAL_CORRECTION else 0),
            full_count=self._state.full_count + (0 if partial else 1),
            partial_count=self._state.partial_count + (1 if partial else 0))
        return elig

    def _apply_bucket(self, b: OutcomeBucket, dims: OutcomeDimensions, weight: float) -> OutcomeBucket:
        def upd(dim, score):
            return _apply_dim(dim, score, weight) if score is not None else dim
        return OutcomeBucket(
            key=b.key, physical=upd(b.physical, dims.physical_target.score),
            comfort=upd(b.comfort, dims.comfort_quality.score),
            controller=upd(b.controller, dims.controller_quality.score),
            data_quality=upd(b.data_quality, dims.data_quality.score),
            sample_count=b.sample_count + 1)

    def _seed_bucket(self, key: str, g: OutcomeBucket) -> OutcomeBucket:
        seed = lambda d: _Dim(value=d.value, effective_n=self._params.bucket_seed_evidence,
                              dispersion=d.dispersion, sample_count=0)
        return OutcomeBucket(key=key, physical=seed(g.physical), comfort=seed(g.comfort),
                             controller=seed(g.controller), data_quality=seed(g.data_quality),
                             sample_count=0)

    def _bump_rejection(self, code: str) -> None:
        rc = dict(self._state.rejection_counts)
        rc[code] = rc.get(code, 0) + 1
        self._state = replace(self._state, rejection_counts=rc)

    def _bump_outlier(self, kind: str) -> None:
        oc = dict(self._state.outlier_counts)
        oc[kind] = oc.get(kind, 0) + 1
        self._state = replace(self._state, outlier_counts=oc)

    # -- predictions (advisory) -----------------------------------------

    def predict(self, context: Any) -> Prediction:
        return self._quality_prediction(context, PredictionType.OUTCOME_QUALITY, "performance")

    def predict_controller_quality(self, context: OutcomePredictionContext) -> Prediction:
        return self._quality_prediction(context, PredictionType.CONTROLLER_QUALITY, "controller")

    def _quality_prediction(self, context, ptype, which) -> Prediction:
        p = self._params
        bucket = None
        bkey = _bucket_key_for(getattr(context, "controller_kind", None))
        if bkey is not None:
            cand = self._state.buckets.get(bkey)
            if cand is not None and cand.sample_count >= p.bucket_min_samples:
                bucket = cand
        src = bucket or (self._state.general if self._state.general.has_evidence else None)
        if src is None:
            value, learned, fallback, reasons, count = 0.6, 0.0, True, ("prior",), 0
        else:
            count = src.sample_count
            learned, fallback, reasons = 1.0, False, (("conditioned_bucket",) if bucket
                                                      else ("general_state",))
            if which == "controller":
                value = src.controller.value
            else:
                value = _mean([src.physical.value, src.comfort.value, src.controller.value])
        value = clamp(value, 0.0, 1.0)
        conf = self._confidence_value(count, fallback)
        key = "outcome_quality" if which == "performance" else "controller_quality"
        return Prediction(
            prediction_type=ptype, values={key: value}, units={key: "score"},
            confidence=conf, reliability=self._state.aggregate_reliability if not fallback else 0.3,
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
            prior_contribution=1.0 - learned, learned_contribution=learned,
            fallback_used=fallback, evidence_count=count,
            bucket=src.key if src else "prior",
            confidence_cap=p.cold_start_confidence_cap if fallback else None,
            cap_reasons=("cold_start",) if fallback else (),
            missing_evidence=(), reason_codes=reasons)

    # -- confidence ------------------------------------------------------

    def _confidence_value(self, sample_count: int, fallback: bool) -> float:
        if fallback or sample_count <= 0:
            return min(0.2, self._params.cold_start_confidence_cap)
        coverage = min(sample_count / self._params.full_confidence_samples, 1.0)
        disp = self._state.general.physical.dispersion
        spread = clamp(1.0 - disp, 0.3, 1.0)
        # many low-data-quality outcomes must not yield high confidence
        dq = max(self._state.general.data_quality.value, 0.1)
        return clamp(coverage * spread * dq, 0.0, 1.0)

    def confidence(self, *, now: Optional[str] = None) -> ConfidenceContribution:
        g = self._state.general
        reasons: list[str] = []
        if not g.has_evidence:
            reasons.append("cold_start")
        if g.data_quality.value < 0.5 and g.has_evidence:
            reasons.append("low_data_quality")
        total = self._state.full_count + self._state.partial_count
        if total > 0 and self._state.partial_count / total > 0.5:
            reasons.append("mostly_partial")
        value = self._confidence_value(g.sample_count, not g.has_evidence)
        freshness, status = freshness_from_age(now, self._state.last_update_ts)
        return ConfidenceContribution(value=value, evidence_count=g.sample_count,
                                      reasons=tuple(reasons), freshness=freshness, status=status)

    # -- diagnostics / export -------------------------------------------

    def _rate(self, count: int) -> Optional[float]:
        n = self._state.general.sample_count
        return round(count / n, 4) if n > 0 else None

    def diagnostics(self) -> OutcomeDiagnostics:
        s = self._state
        g = s.general
        return OutcomeDiagnostics(
            general_physical=g.physical.value, general_comfort=g.comfort.value,
            general_controller=g.controller.value, general_data_quality=g.data_quality.value,
            bucket_performance={k: _mean([b.physical.value, b.comfort.value, b.controller.value])
                                for k, b in s.buckets.items()},
            sample_counts={"general": g.sample_count,
                           **{k: b.sample_count for k, b in s.buckets.items()}},
            reached_rate=self._rate(s.reached_count), timeout_rate=self._rate(s.timeout_count),
            overshoot_rate=self._rate(s.overshoot_count),
            manual_correction_rate=self._rate(s.manual_correction_count),
            full_partial=(s.full_count, s.partial_count), confidence=self.confidence().value,
            reason_counts=dict(s.reason_counts), rejection_counts=dict(s.rejection_counts),
            outlier_counts=dict(s.outlier_counts),
            missing_optional_evidence=("controller_buckets",) if not s.buckets else (),
            last_update_ts=s.last_update_ts, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION, scoring_version=SCORING_VERSION)

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        s = self._state
        g = s.general
        if scope is ExportScope.SUPPORT:
            return {
                "physical": g.physical.value, "comfort": g.comfort.value,
                "controller": g.controller.value, "data_quality": g.data_quality.value,
                "reached_rate": self._rate(s.reached_count),
                "timeout_rate": self._rate(s.timeout_count),
                "overshoot_rate": self._rate(s.overshoot_count),
                "full_partial": [s.full_count, s.partial_count],
                "confidence": self.confidence().value, "sample_count": g.sample_count,
                "reason_counts": dict(s.reason_counts),
                "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
                "scoring_version": SCORING_VERSION,
            }
        if scope is ExportScope.RESEARCH:
            cap = self._params.research_sample_cap
            total = s.general.sample_count
            exported = len(s.recent_samples)
            return {
                "samples": [
                    {"physical": x.physical_score, "comfort": x.comfort_score,
                     "controller": x.controller_score, "data_quality": x.data_quality_score,
                     "performance": x.performance_score, "reason": x.reason,
                     "partial": x.partial, "scoring_version": x.scoring_version,
                     "confounder_codes": list(x.confounder_flags)}
                    for x in s.recent_samples
                ],
                "truncation_info": {
                    "max_samples": cap,
                    "total_samples": total,
                    "exported_samples": exported,
                    "truncated": total > cap,
                },
                "scoring_version": SCORING_VERSION, "model_version": MODEL_VERSION,
            }
        return {"unsupported": "raw_export_not_provided_by_model"}

    # -- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        self._state = self._empty_state()

    def rebuild(self, raw: Any = None,
                episodes: Optional[Sequence[Any]] = None) -> ModelRebuildResult:
        self.reset()
        raw_items = list(episodes or [])
        items: list[tuple[Any, Optional[OutcomeUpdateContext]]] = []
        for it in raw_items:
            if isinstance(it, OutcomeRebuildItem):
                items.append((it.episode, it.context))
            else:
                items.append((it, None))
        items.sort(key=lambda pair: (_iso(getattr(pair[0], "start_ts", None)) or "",
                                     getattr(pair[0], "episode_id", "")))
        accepted = rejected = duplicates = 0
        rejection_counts: dict[str, int] = {}
        for episode, context in items:
            res = self.update(episode, context)
            if res.accepted:
                accepted += 1
            else:
                rejected += 1
                code = res.confounder_flags[0] if res.confounder_flags else "unknown"
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
                if code == OutcomeRejection.DUPLICATE_EPISODE.value:
                    duplicates += 1
        return ModelRebuildResult(
            processed_count=len(items), accepted_count=accepted, rejected_count=rejected,
            duplicate_count=duplicates, error_count=0, rejection_counts=rejection_counts,
            started_from_prior=True, deterministic=True)

    # -- state -----------------------------------------------------------

    def serialize_state(self) -> Mapping[str, Any]:
        s = self._state
        return {
            "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
            "scoring_version": SCORING_VERSION, "learning_zone_id": s.learning_zone_id,
            "general": _bucket_dict(s.general),
            "buckets": {k: _bucket_dict(b) for k, b in s.buckets.items()},
            "processed_ids": list(s.processed_ids),
            "processed_decision_ids": list(s.processed_decision_ids),
            "aggregate_reliability": s.aggregate_reliability, "last_update_ts": s.last_update_ts,
            "reason_counts": dict(s.reason_counts), "rejection_counts": dict(s.rejection_counts),
            "outlier_counts": dict(s.outlier_counts), "reached_count": s.reached_count,
            "timeout_count": s.timeout_count, "overshoot_count": s.overshoot_count,
            "manual_correction_count": s.manual_correction_count, "full_count": s.full_count,
            "partial_count": s.partial_count, "dedup_count": s.dedup_count,
            "recent_samples": [_sample_dict(x) for x in s.recent_samples],
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
        for dim in ("physical", "comfort", "controller", "data_quality"):
            d = g.get(dim, {})
            for k in ("value", "effective_n", "dispersion"):
                v = d.get(k)
                if v is not None and not is_finite(v):
                    errors.append(f"non-finite general.{dim}.{k}")
        return errors

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        errors = self.validate_state(state)
        if errors:
            raise ValueError(f"invalid outcome state: {errors}")
        buckets = {k: _bucket_from(b) for k, b in state.get("buckets", {}).items()}
        recent = tuple(_sample_from(x) for x in state.get("recent_samples", []))
        self._state = OutcomeState(
            learning_zone_id=state["learning_zone_id"], general=_bucket_from(state["general"]),
            buckets=buckets, processed_ids=tuple(state.get("processed_ids", [])),
            processed_decision_ids=tuple(state.get("processed_decision_ids", [])),
            recent_samples=recent, aggregate_reliability=state.get("aggregate_reliability", 0.0),
            last_update_ts=state.get("last_update_ts"),
            reason_counts=dict(state.get("reason_counts", {})),
            rejection_counts=dict(state.get("rejection_counts", {})),
            outlier_counts=dict(state.get("outlier_counts", {})),
            reached_count=state.get("reached_count", 0),
            timeout_count=state.get("timeout_count", 0),
            overshoot_count=state.get("overshoot_count", 0),
            manual_correction_count=state.get("manual_correction_count", 0),
            full_count=state.get("full_count", 0), partial_count=state.get("partial_count", 0),
            dedup_count=state.get("dedup_count", 0))

    def migrate_state(self, old_model_version: int, old_parameter_version: int,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        if old_model_version == MODEL_VERSION:
            return state
        raise ValueError(
            f"no migration path from model_version {old_model_version} to {MODEL_VERSION}")


# -- helpers ------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _has_temp_context(episode: OutcomeEpisode) -> bool:
    return any(not pt.gap and pt.value is not None for pt in episode.trajectory.points)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _dim_dict(d: _Dim) -> dict:
    return {"value": d.value, "effective_n": d.effective_n, "dispersion": d.dispersion,
            "sample_count": d.sample_count}


def _dim_from(d: Mapping[str, Any]) -> _Dim:
    return _Dim(value=d["value"], effective_n=d["effective_n"], dispersion=d["dispersion"],
                sample_count=d["sample_count"])


def _bucket_dict(b: OutcomeBucket) -> dict:
    return {"key": b.key, "physical": _dim_dict(b.physical), "comfort": _dim_dict(b.comfort),
            "controller": _dim_dict(b.controller), "data_quality": _dim_dict(b.data_quality),
            "sample_count": b.sample_count}


def _bucket_from(d: Mapping[str, Any]) -> OutcomeBucket:
    return OutcomeBucket(key=d["key"], physical=_dim_from(d["physical"]),
                         comfort=_dim_from(d["comfort"]), controller=_dim_from(d["controller"]),
                         data_quality=_dim_from(d["data_quality"]), sample_count=d["sample_count"])


def _sample_dict(x: OutcomeSample) -> dict:
    return {
        "source_episode_id": x.source_episode_id, "decision_id": x.decision_id,
        "learning_zone_id": x.learning_zone_id, "physical_score": x.physical_score,
        "comfort_score": x.comfort_score, "controller_score": x.controller_score,
        "data_quality_score": x.data_quality_score, "performance_score": x.performance_score,
        "overall_reliability": x.overall_reliability, "effective_weight": x.effective_weight,
        "reason": x.reason, "controller_kind": x.controller_kind, "partial": x.partial,
        "scoring_version": x.scoring_version,
        "feature_extractor_version": x.feature_extractor_version,
        "builder_version": x.builder_version, "classifier_version": x.classifier_version,
        "reason_codes": list(x.reason_codes),
        "confounder_flags": list(x.confounder_flags),
    }


def _sample_from(d: Mapping[str, Any]) -> OutcomeSample:
    return OutcomeSample(
        source_episode_id=d["source_episode_id"], decision_id=d.get("decision_id"),
        learning_zone_id=d["learning_zone_id"], physical_score=d.get("physical_score"),
        comfort_score=d.get("comfort_score"), controller_score=d.get("controller_score"),
        data_quality_score=d["data_quality_score"], performance_score=d.get("performance_score"),
        overall_reliability=d["overall_reliability"], effective_weight=d["effective_weight"],
        reason=d["reason"], controller_kind=d.get("controller_kind"), partial=d["partial"],
        scoring_version=d["scoring_version"],
        feature_extractor_version=d["feature_extractor_version"],
        builder_version=d["builder_version"], classifier_version=d["classifier_version"],
        reason_codes=tuple(d.get("reason_codes", [])),
        confounder_flags=tuple(d.get("confounder_flags", [])))


def outcome_model_definition():
    from ..registry import ModelDefinition

    return ModelDefinition(
        model_name=MODEL_NAME, model_factory=lambda: OutcomeModel("__definition__"),
        model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
        consumed_raw_tracks=(), consumed_episode_types=(EpisodeType.OUTCOME,),
        supported_prediction_types=(
            PredictionType.OUTCOME_QUALITY, PredictionType.COMFORT_QUALITY,
            PredictionType.CONTROLLER_QUALITY),
        required_features=("indoor_temp", "target"),
        optional_features=("outdoor_temp", "valve_opening", "hvac_action", "forecast",
                           "wind_speed", "solar_radiation"),
        required_capability_flags=(), regime_dependent=False, advisory=True,
        control_relevant=False, rebuildable=True, can_be_not_available=True, min_trv_only=True)
