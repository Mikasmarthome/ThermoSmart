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


class DimensionStatus(Enum):
    OK = "ok"
    PARTIAL = "partial"
    NOT_COMPUTABLE = "not_computable"


class BoostSourceKind(Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoostParameters:
    max_boost_offset_c: float = _MAX_BOOST_OFFSET_C
    min_boost_offset_c: float = 0.1
    min_duration_s: float = 120.0
    min_points: int = 3
    min_data_quality: float = 0.25
    min_attribution_reliability: float = 0.3
    max_sample_weight: float = 1.0
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
    default_boost_offset_c: float = 1.5    # existing ThermoSmart safe default
    default_boost_duration_s: float = 1800.0
    min_duration_pred_s: float = 300.0
    max_duration_pred_s: float = 7200.0
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

    def __post_init__(self) -> None:
        if not self.source_episode_id:
            raise ValueError("source_episode_id must be non-empty")
        if not self.decision_id:
            raise ValueError("decision_id must be non-empty")
        for name in ("requested_offset_c", "planned_duration_s", "actual_duration_s",
                     "target_temperature_c", "start_temperature_c", "start_deficit_c",
                     "expected_non_boost_duration_s", "expected_heat_rate_c_per_h",
                     "expected_afterheat_rise_c", "expected_heat_loss_c_per_h",
                     "outcome_performance", "outcome_comfort"):
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
            source_feature_version=d.get("source_feature_version", 1))


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


@dataclass(frozen=True)
class BoostState:
    learning_zone_id: str
    general: BoostBucket
    buckets: Mapping[str, BoostBucket] = field(default_factory=dict)
    processed_ids: tuple[str, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()
    recent_samples: tuple[BoostSample, ...] = ()
    last_update_ts: Optional[str] = None
    full_count: int = 0
    partial_count: int = 0
    overshoot_count: int = 0
    manual_correction_count: int = 0
    reason_counts: Mapping[str, int] = field(default_factory=dict)
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION
    evaluation_version: int = EVALUATION_VERSION


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


@dataclass(frozen=True)
class BoostPredictionContext:
    start_deficit_c: Optional[float] = None
    decision_type: Optional[str] = None
    intensity_c: Optional[float] = None
    device_prior_offset_c: Optional[float] = None
    outdoor_bucket: Optional[str] = None


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
        confounders = set(episode.confounder_flags) & _CONFOUNDER_FLAGS
        rel *= 0.4 ** len(confounders)
        if context.manual_correction_ref:
            rel *= 0.5
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
            return elig

        ev = self.evaluate_boost(episode, context)
        weight = elig.weight
        # effective factor: requested offset scaled by realised comfort (overshoot
        # pulls the learned factor DOWN), so future recommendations stay conservative
        comfort = ev.comfort_impact.score if ev.comfort_impact.computable else 0.5
        effective_factor = clamp(ev.effect.requested_offset_c * comfort, 0.0,
                                 self._params.max_boost_offset_c)

        g = self._state.general
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

        overshoot_flag = (ev.overshoot_penalty.computable and ev.overshoot_penalty.score is not None
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
            evaluation_version=EVALUATION_VERSION)

        processed = (self._state.processed_ids + (episode.episode_id,))[-self._params.dedup_max_ids:]
        processed_dec = (self._state.processed_decision_ids
                         + (episode.decision_id,))[-self._params.dedup_max_ids:]
        recent = (self._state.recent_samples + (sample,))[-self._params.research_sample_cap:]
        rc = dict(self._state.reason_counts)
        rc[episode.reason.value] = rc.get(episode.reason.value, 0) + 1
        self._state = replace(
            self._state, general=new_general, buckets=buckets, processed_ids=processed,
            processed_decision_ids=processed_dec, recent_samples=recent, reason_counts=rc,
            last_update_ts=_iso(episode.end_ts),
            full_count=self._state.full_count + (0 if ev.partial else 1),
            partial_count=self._state.partial_count + (1 if ev.partial else 0),
            overshoot_count=self._state.overshoot_count + (1 if overshoot_flag else 0),
            manual_correction_count=self._state.manual_correction_count
            + (1 if episode.reason is EpisodeReason.MANUAL_CORRECTION else 0))
        return elig

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
        return BoostRecommendation(
            boost_offset_c=round(offset, 4), recommended_duration_s=duration,
            confidence=conf, reliability=clamp(eff / p.full_confidence_samples, 0.0, 1.0),
            fallback_used=fallback, prior_contribution=1.0 - learned,
            learned_contribution=learned, evidence_count=int(eff), bucket=bkey,
            missing_evidence=() if not fallback else ("boost_history",),
            reason_codes=reasons)

    def predict_boost_duration(self, context: BoostPredictionContext) -> BoostRecommendation:
        return self.predict_boost_factor(context)

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
        if not self._state.recent_samples:
            return None
        vals = [s.attribution_reliability for s in self._state.recent_samples]
        return sum(vals) / len(vals)

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
            parameter_version=PARAMETER_VERSION, evaluation_version=EVALUATION_VERSION)

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
            "last_update_ts": s.last_update_ts, "full_count": s.full_count,
            "partial_count": s.partial_count, "overshoot_count": s.overshoot_count,
            "manual_correction_count": s.manual_correction_count,
            "reason_counts": dict(s.reason_counts), "rejection_counts": dict(s.rejection_counts),
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
            last_update_ts=state.get("last_update_ts"), full_count=state.get("full_count", 0),
            partial_count=state.get("partial_count", 0),
            overshoot_count=state.get("overshoot_count", 0),
            manual_correction_count=state.get("manual_correction_count", 0),
            reason_counts=dict(state.get("reason_counts", {})),
            rejection_counts=dict(state.get("rejection_counts", {})),
            outlier_counts=dict(state.get("outlier_counts", {})))

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


def _sample_dict(x: BoostSample) -> dict:
    return {
        "source_episode_id": x.source_episode_id, "decision_id": x.decision_id,
        "learning_zone_id": x.learning_zone_id, "requested_offset_c": x.requested_offset_c,
        "effective_factor_c": x.effective_factor_c, "temperature_gain_c": x.temperature_gain_c,
        "overshoot_c": x.overshoot_c, "comfort_score": x.comfort_score,
        "controller_score": x.controller_score, "boost_duration_s": x.boost_duration_s,
        "attribution_reliability": x.attribution_reliability,
        "effective_weight": x.effective_weight, "reason": x.reason, "partial": x.partial,
        "evaluation_version": x.evaluation_version,
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
        evaluation_version=d["evaluation_version"])


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
