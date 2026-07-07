"""AfterheatModel for LE 2.0 (pure Python).

Learns the thermal after-effect once active heat input ends, from completed,
undisturbed AfterheatEpisodes: residual temperature rise (deg C), time to peak,
afterheat duration, and a simple robust decay indicator. The heating-drive-end
timestamp (episode start) is the authoritative reference. Near-zero afterheat is
a first-class, learnable outcome (a room with little residual heat must not stay
overestimated by a generic prior). Independent from HeatRate/HeatLoss; no HA /
storage / coordinator dependency; mutates no input; sends no command.
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
    Measurement,
    ModelRebuildResult,
    OutlierStatus,
    Prediction,
    PredictionType,
    Regime,
    RejectionReason,
    UpdateEligibilityResult,
)
from ..episode_schemas import AfterheatEpisode, EpisodeType
from ..features import FeatureExtractor, FeatureName, FeatureStatus
from .base import clamp, freshness_from_age, is_finite, outlier_z, robust_ema_update

MODEL_VERSION = 1
PARAMETER_VERSION = 1
MODEL_NAME = "afterheat"


class AfterheatRejection(Enum):
    WRONG_EPISODE_TYPE = "wrong_episode_type"
    WRONG_ZONE = "wrong_zone"
    MISSING_DRIVE_END = "missing_drive_end"
    INFERRED_DRIVE_END_TOO_WEAK = "inferred_drive_end_too_weak"
    DISTURBED = "disturbed"
    INSUFFICIENT_DURATION = "insufficient_duration"
    INSUFFICIENT_POINTS = "insufficient_points"
    MISSING_INDOOR_TEMP = "missing_indoor_temperature"
    INVALID_TRAJECTORY = "invalid_trajectory"
    NO_AFTERHEAT_EFFECT = "no_afterheat_effect"
    NEGATIVE_RESIDUAL_RISE = "negative_residual_rise"
    IMPLAUSIBLE_RESIDUAL_RISE = "implausible_residual_rise"
    IMPLAUSIBLE_TIME_TO_PEAK = "implausible_time_to_peak"
    LOW_RELIABILITY = "low_reliability"
    REHEATING_CONTAMINATION = "reheating_contamination"
    WINDOW_CONFOUNDER = "window_confounder"
    SOLAR_CONFOUNDER = "solar_confounder"
    HEATING_FAILURE = "heating_failure"
    EXCESSIVE_GAPS = "excessive_gaps"
    DUPLICATE_EPISODE = "duplicate_episode"
    VERSION_MISMATCH = "version_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    SEVERE_OUTLIER = "severe_outlier"


_TO_CONTRACT = {
    AfterheatRejection.WRONG_EPISODE_TYPE: RejectionReason.NOT_ELIGIBLE,
    AfterheatRejection.WRONG_ZONE: RejectionReason.NOT_ELIGIBLE,
    AfterheatRejection.MISSING_DRIVE_END: RejectionReason.MISSING_REQUIRED_DATA,
    AfterheatRejection.INFERRED_DRIVE_END_TOO_WEAK: RejectionReason.LOW_RELIABILITY,
    AfterheatRejection.DISTURBED: RejectionReason.DISTURBED_REGIME,
    AfterheatRejection.INSUFFICIENT_DURATION: RejectionReason.NOT_ELIGIBLE,
    AfterheatRejection.INSUFFICIENT_POINTS: RejectionReason.NOT_ELIGIBLE,
    AfterheatRejection.MISSING_INDOOR_TEMP: RejectionReason.MISSING_REQUIRED_DATA,
    AfterheatRejection.INVALID_TRAJECTORY: RejectionReason.MISSING_REQUIRED_DATA,
    AfterheatRejection.NO_AFTERHEAT_EFFECT: RejectionReason.NOT_ELIGIBLE,
    AfterheatRejection.NEGATIVE_RESIDUAL_RISE: RejectionReason.OUTLIER,
    AfterheatRejection.IMPLAUSIBLE_RESIDUAL_RISE: RejectionReason.OUTLIER,
    AfterheatRejection.IMPLAUSIBLE_TIME_TO_PEAK: RejectionReason.OUTLIER,
    AfterheatRejection.LOW_RELIABILITY: RejectionReason.LOW_RELIABILITY,
    AfterheatRejection.REHEATING_CONTAMINATION: RejectionReason.NOT_ELIGIBLE,
    AfterheatRejection.WINDOW_CONFOUNDER: RejectionReason.DISTURBED_REGIME,
    AfterheatRejection.SOLAR_CONFOUNDER: RejectionReason.DISTURBED_REGIME,
    AfterheatRejection.HEATING_FAILURE: RejectionReason.DISTURBED_REGIME,
    AfterheatRejection.EXCESSIVE_GAPS: RejectionReason.MISSING_REQUIRED_DATA,
    AfterheatRejection.DUPLICATE_EPISODE: RejectionReason.DUPLICATE,
    AfterheatRejection.VERSION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    AfterheatRejection.CONTEXT_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    AfterheatRejection.SEVERE_OUTLIER: RejectionReason.OUTLIER,
}


@dataclass(frozen=True)
class AfterheatParameters:
    min_episode_reliability: float = 0.4
    min_duration_s: float = 120.0
    min_points: int = 3
    max_gap_ratio: float = 0.5
    start_tolerance_ms: int = 120_000          # first valid point must be within this of drive end
    indirect_start_weight_factor: float = 0.7  # reduced weight when start reference is indirect
    min_drive_end_reliability: float = 0.3
    inferred_setpoint_reduction_reliability: float = 0.6
    inferred_weak_reliability: float = 0.4
    min_residual_rise_c: float = 0.15          # below this counts as near-zero
    negative_rise_tolerance_c: float = 0.2     # noise tolerance before rejecting
    max_residual_rise_c: float = 4.0
    max_time_to_peak_s: float = 3600.0
    max_sample_weight: float = 1.0
    bucket_min_samples: int = 5
    bucket_seed_evidence: float = 1.0
    outlier_mad_k_mild: float = 3.0
    outlier_mad_k_severe: float = 6.0
    min_samples_for_outlier: int = 5
    dedup_max_ids: int = 500
    research_sample_cap: int = 50
    heating_duration_buckets_s: tuple[float, ...] = (600.0, 1800.0, 3600.0)
    generic_prior_rise_c: float = 0.3
    generic_prior_time_to_peak_s: float = 600.0
    generic_prior_duration_s: float = 1200.0
    generic_prior_decay: float = -0.01
    early_cutoff_max_c: float = 3.0
    cold_start_confidence_cap: float = 0.4
    full_confidence_samples: float = 20.0
    parameter_version: int = PARAMETER_VERSION


# -- per-dimension robust state ----------------------------------------------

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
class AfterheatBucket:
    key: str
    rise: _Dim = field(default_factory=_Dim)
    time_to_peak: _Dim = field(default_factory=_Dim)
    duration: _Dim = field(default_factory=_Dim)
    decay: _Dim = field(default_factory=_Dim)
    sample_count: int = 0

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0 and self.rise.has_evidence


def _empty_bucket(key: str) -> AfterheatBucket:
    return AfterheatBucket(key=key)


@dataclass(frozen=True)
class AfterheatUpdateContext:
    source_episode_id: str
    learning_zone_id: Optional[str] = None
    drive_end_source: Optional[str] = None
    drive_end_source_reliability: Optional[float] = None
    heating_duration_s: Optional[float] = None
    heating_temp_rise_c: Optional[float] = None
    setpoint_excess_c: Optional[float] = None
    outdoor_temperature_c: Optional[float] = None
    device_profile_key: Optional[str] = None
    solar_bucket: Optional[str] = None
    source_feature_version: int = 1
    source_refs: tuple[str, ...] = ()
    quality: DataQuality = DataQuality.OK

    def __post_init__(self) -> None:
        if not self.source_episode_id:
            raise ValueError("source_episode_id must be non-empty")
        for name in ("drive_end_source_reliability", "heating_duration_s",
                     "heating_temp_rise_c", "setpoint_excess_c", "outdoor_temperature_c"):
            v = getattr(self, name)
            if v is not None and not is_finite(v):
                raise ValueError(f"{name} must be finite (or None)")

    def to_dict(self) -> dict:
        return {
            "source_episode_id": self.source_episode_id,
            "learning_zone_id": self.learning_zone_id,
            "drive_end_source": self.drive_end_source,
            "drive_end_source_reliability": self.drive_end_source_reliability,
            "heating_duration_s": self.heating_duration_s,
            "heating_temp_rise_c": self.heating_temp_rise_c,
            "setpoint_excess_c": self.setpoint_excess_c,
            "outdoor_temperature_c": self.outdoor_temperature_c,
            "device_profile_key": self.device_profile_key,
            "solar_bucket": self.solar_bucket,
            "source_feature_version": self.source_feature_version,
            "source_refs": list(self.source_refs), "quality": self.quality.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AfterheatUpdateContext":
        return cls(
            source_episode_id=d["source_episode_id"],
            learning_zone_id=d.get("learning_zone_id"),
            drive_end_source=d.get("drive_end_source"),
            drive_end_source_reliability=d.get("drive_end_source_reliability"),
            heating_duration_s=d.get("heating_duration_s"),
            heating_temp_rise_c=d.get("heating_temp_rise_c"),
            setpoint_excess_c=d.get("setpoint_excess_c"),
            outdoor_temperature_c=d.get("outdoor_temperature_c"),
            device_profile_key=d.get("device_profile_key"),
            solar_bucket=d.get("solar_bucket"),
            source_feature_version=d.get("source_feature_version", 1),
            source_refs=tuple(d.get("source_refs", [])),
            quality=DataQuality(d.get("quality", DataQuality.OK.value)))


@dataclass(frozen=True)
class AfterheatRebuildItem:
    episode: Any
    context: Optional[AfterheatUpdateContext] = None


@dataclass(frozen=True)
class AfterheatSample:
    source_episode_id: str
    learning_zone_id: str
    residual_rise_c: float
    time_to_peak_s: float
    afterheat_duration_s: float
    decay_indicator: float
    episode_reliability: float
    feature_reliability: float
    drive_end_reliability: float
    effective_weight: float
    bucket: str
    near_zero: bool
    feature_extractor_version: int
    builder_version: int
    classifier_version: int
    data_quality: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class AfterheatState:
    learning_zone_id: str
    general: AfterheatBucket
    buckets: Mapping[str, AfterheatBucket] = field(default_factory=dict)
    processed_ids: tuple[str, ...] = ()
    recent_samples: tuple[AfterheatSample, ...] = ()
    aggregate_reliability: float = 0.0
    last_update_ts: Optional[str] = None
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    dedup_count: int = 0
    direct_drive_end_count: int = 0
    inferred_drive_end_count: int = 0
    near_zero_count: int = 0
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION


@dataclass(frozen=True)
class AfterheatDiagnostics:
    general_rise_c: float
    general_time_to_peak_s: float
    general_duration_s: float
    general_decay: float
    bucket_rise: Mapping[str, float]
    sample_counts: Mapping[str, int]
    effective_sample_counts: Mapping[str, float]
    dispersion: Mapping[str, float]
    confidence: float
    direct_drive_end_count: int
    inferred_drive_end_count: int
    near_zero_count: int
    fallback: bool
    missing_optional_evidence: tuple[str, ...]
    rejection_counts: Mapping[str, int]
    outlier_counts: Mapping[str, int]
    last_update_ts: Optional[str]
    model_version: int
    parameter_version: int


@dataclass(frozen=True)
class AfterheatPredictionContext:
    current_temp: Optional[float] = None
    target: Optional[float] = None
    heating_duration_s: Optional[float] = None
    profile_id: Optional[str] = None


@dataclass(frozen=True)
class AfterheatPrediction:
    residual_rise_c: float
    time_to_peak_min: float
    afterheat_duration_min: float
    decay_indicator: float
    bucket: str
    confidence: float
    reliability: float
    prior_contribution: float
    learned_contribution: float
    fallback_used: bool
    evidence_count: int
    drive_end_evidence_level: float
    missing_evidence: tuple[str, ...]
    reason_codes: tuple[str, ...]
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION

    def __post_init__(self) -> None:
        for name in ("residual_rise_c", "time_to_peak_min", "afterheat_duration_min",
                     "decay_indicator"):
            v = getattr(self, name)
            if v is None or not is_finite(v):
                raise ValueError(f"{name} must be finite")


def _first_valid_point(trajectory) -> Optional[tuple[int, float]]:
    for pt in trajectory.points:
        if not pt.gap and pt.value is not None and is_finite(pt.value):
            return pt.offset_ms, float(pt.value)
    return None


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    idx = 0
    for edge in edges:
        if value >= edge:
            idx += 1
        else:
            break
    return idx


class AfterheatModel:
    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    parameter_version = PARAMETER_VERSION
    consumed_raw_tracks: tuple[str, ...] = ()
    consumed_episode_types = (EpisodeType.AFTERHEAT.value,)
    required_features = ("indoor_temp", "trv_setpoint")
    optional_features = ("valve_opening", "hvac_action", "outdoor_temp",
                         "solar_radiation", "radiator_profile")
    supported_prediction_types = (
        PredictionType.EXPECTED_OVERSHOOT, PredictionType.AFTERHEAT_TIME_TO_PEAK,
        PredictionType.AFTERHEAT_DURATION, PredictionType.RECOMMENDED_EARLY_CUTOFF)
    min_tier = 0

    def __init__(self, learning_zone_id: str, params: Optional[AfterheatParameters] = None, *,
                 device_prior: Optional[Callable[[Optional[str]], Optional[float]]] = None,
                 extractor: Optional[FeatureExtractor] = None) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or AfterheatParameters()
        self._device_prior = device_prior
        self._extractor = extractor or FeatureExtractor()
        self._state = self._empty_state()

    def cold_start_prior(self) -> Mapping[str, Any]:
        return {"general_rise_c": None, "generic_prior_rise_c": self._params.generic_prior_rise_c}

    def _empty_state(self) -> AfterheatState:
        return AfterheatState(learning_zone_id=self._zone, general=_empty_bucket("general"))

    def _extractor_version(self) -> int:
        from ..features import FEATURE_EXTRACTOR_VERSION
        return FEATURE_EXTRACTOR_VERSION

    # -- authoritative quantities ---------------------------------------

    def _compute(self, episode: AfterheatEpisode, context: Optional[AfterheatUpdateContext]):
        p = self._params
        fs = self._extractor.extract_trajectory_features(episode.trajectory)
        valid = fs.get(FeatureName.TRAJ_VALID_POINTS).value
        peak_r = fs.get(FeatureName.TRAJ_PEAK)
        ttp_r = fs.get(FeatureName.TRAJ_TIME_TO_PEAK)
        plaus = fs.get(FeatureName.TRAJ_PLAUSIBILITY)
        gap_r = fs.get(FeatureName.TRAJ_GAP_RATIO)
        sec_r = fs.get(FeatureName.TRAJ_SECOND_DERIVATIVE)
        slope_r = fs.get(FeatureName.TRAJ_LINEAR_SLOPE)
        feat_rel = slope_r.reliability if slope_r.is_present else 0.0

        if valid is None or valid < 1:
            return None, AfterheatRejection.MISSING_INDOOR_TEMP, feat_rel
        if not peak_r.is_present or not ttp_r.is_present:
            return None, AfterheatRejection.INVALID_TRAJECTORY, feat_rel
        if int(valid) < p.min_points:
            return None, AfterheatRejection.INSUFFICIENT_POINTS, feat_rel
        if gap_r.is_present and gap_r.value > p.max_gap_ratio:
            return None, AfterheatRejection.EXCESSIVE_GAPS, feat_rel
        if plaus.status is FeatureStatus.INVALID:
            return None, AfterheatRejection.IMPLAUSIBLE_RESIDUAL_RISE, feat_rel

        # Authoritative start reference = temperature at the heating drive end.
        # Peak and start come from the SAME quality-checked trajectory: use the
        # first valid trajectory point when it lies within the start tolerance of
        # the drive end (offset 0). Otherwise fall back to the recorded drive-end
        # temperature as an INDIRECT reference (reduced reliability). The episode
        # end temperature is never used as the start reference.
        first = _first_valid_point(episode.trajectory)
        if first is None:
            return None, AfterheatRejection.MISSING_INDOOR_TEMP, feat_rel
        first_offset, first_value = first
        indirect = False
        if first_offset <= p.start_tolerance_ms:
            temperature_at_drive_end = first_value
        elif episode.indoor_temp_at_close is not None \
                and is_finite(episode.indoor_temp_at_close):
            temperature_at_drive_end = episode.indoor_temp_at_close
            indirect = True
        else:
            return None, AfterheatRejection.MISSING_INDOOR_TEMP, feat_rel

        rise_raw = peak_r.value - temperature_at_drive_end
        if rise_raw < -p.negative_rise_tolerance_c:
            return None, AfterheatRejection.NEGATIVE_RESIDUAL_RISE, feat_rel
        rise = max(0.0, rise_raw)
        if rise > p.max_residual_rise_c:
            return None, AfterheatRejection.IMPLAUSIBLE_RESIDUAL_RISE, feat_rel
        ttp_s = ttp_r.value
        if not is_finite(ttp_s) or ttp_s < 0 or ttp_s > p.max_time_to_peak_s:
            return None, AfterheatRejection.IMPLAUSIBLE_TIME_TO_PEAK, feat_rel
        duration_s = (episode.end_ts - episode.start_ts).total_seconds()
        decay = sec_r.value if sec_r.is_present else 0.0
        if not is_finite(decay):
            decay = 0.0
        # near-zero evidence requires a reliable (direct) start reference
        near_zero = rise < p.min_residual_rise_c and not indirect
        return (rise, ttp_s, duration_s, decay, near_zero, indirect, feat_rel), None, feat_rel

    def _drive_end_reliability(self, episode: AfterheatEpisode,
                              context: Optional[AfterheatUpdateContext]) -> Optional[float]:
        p = self._params
        if context is not None and context.drive_end_source_reliability is not None:
            return context.drive_end_source_reliability
        # inferred from the episode's setpoint reduction (TRV-only drive end)
        if episode.trv_setpoint_before is not None and episode.trv_setpoint_after is not None \
                and episode.trv_setpoint_before > episode.trv_setpoint_after:
            return p.inferred_setpoint_reduction_reliability
        if context is not None and context.drive_end_source is not None:
            return p.inferred_weak_reliability
        return None  # no drive-end evidence at all

    # -- eligibility -----------------------------------------------------

    def update_eligibility(self, episode: Any) -> UpdateEligibilityResult:
        return self._eligibility(episode, None)

    def _eligibility(self, episode: Any,
                     context: Optional[AfterheatUpdateContext]) -> UpdateEligibilityResult:
        p = self._params
        if not isinstance(episode, AfterheatEpisode):
            return self._reject(AfterheatRejection.WRONG_EPISODE_TYPE)
        if episode.learning_zone_id != self._zone:
            return self._reject(AfterheatRejection.WRONG_ZONE)
        if episode.episode_schema_version != 1:
            return self._reject(AfterheatRejection.VERSION_MISMATCH)
        if episode.regime is not Regime.AFTERHEAT:
            return self._reject(AfterheatRejection.DISTURBED, regime=episode.regime)
        cf = set(episode.confounder_flags)
        if "reheating" in cf:
            return self._reject(AfterheatRejection.REHEATING_CONTAMINATION)
        if "window_open" in cf:
            return self._reject(AfterheatRejection.WINDOW_CONFOUNDER)
        if "solar_gain" in cf:
            return self._reject(AfterheatRejection.SOLAR_CONFOUNDER)
        if "heating_failure" in cf:
            return self._reject(AfterheatRejection.HEATING_FAILURE)
        if episode.episode_id in self._state.processed_ids:
            return self._reject(AfterheatRejection.DUPLICATE_EPISODE)
        if episode.reliability < p.min_episode_reliability:
            return self._reject(AfterheatRejection.LOW_RELIABILITY)
        if (episode.end_ts - episode.start_ts).total_seconds() < p.min_duration_s:
            return self._reject(AfterheatRejection.INSUFFICIENT_DURATION)

        del_rel = self._drive_end_reliability(episode, context)
        if del_rel is None:
            return self._reject(AfterheatRejection.MISSING_DRIVE_END)
        if del_rel < p.min_drive_end_reliability:
            return self._reject(AfterheatRejection.INFERRED_DRIVE_END_TOO_WEAK)

        computed, reject, feat_rel = self._compute(episode, context)
        if reject is not None:
            return self._reject(reject)
        indirect = computed[5]
        weight = clamp(episode.reliability * max(feat_rel, 0.1) * del_rel,
                       0.0, p.max_sample_weight)
        if indirect:
            weight *= p.indirect_start_weight_factor
        return UpdateEligibilityResult(
            accepted=True, reliability=episode.reliability, weight=weight,
            outlier_status=OutlierStatus.NONE, regime=Regime.AFTERHEAT,
            source_episode_id=episode.episode_id,
            feature_extractor_version=self._extractor_version())

    def _reject(self, reason: AfterheatRejection, *, regime=None) -> UpdateEligibilityResult:
        return UpdateEligibilityResult(
            accepted=False, reliability=0.0, weight=0.0,
            rejection_reason=_TO_CONTRACT[reason], confounder_flags=(reason.value,),
            regime=regime)

    # -- update ----------------------------------------------------------

    def update(self, episode: Any,
               context: Optional[AfterheatUpdateContext] = None) -> UpdateEligibilityResult:
        if context is not None:
            if context.source_episode_id != episode.episode_id or (
                    context.learning_zone_id is not None
                    and context.learning_zone_id != self._zone):
                self._bump_rejection(AfterheatRejection.CONTEXT_MISMATCH.value)
                return self._reject(AfterheatRejection.CONTEXT_MISMATCH)

        elig = self._eligibility(episode, context)
        if not elig.accepted:
            self._bump_rejection(elig.confounder_flags[0] if elig.confounder_flags else "unknown")
            if elig.confounder_flags and elig.confounder_flags[0] == \
                    AfterheatRejection.DUPLICATE_EPISODE.value:
                self._state = replace(self._state, dedup_count=self._state.dedup_count + 1)
            return elig

        computed, _, feat_rel = self._compute(episode, context)
        rise, ttp_s, duration_s, decay, near_zero, indirect, _ = computed
        del_rel = self._drive_end_reliability(episode, context)
        weight = elig.weight

        # multi-dimensional outlier scoring vs the zone's own evidence
        g = self._state.general
        outlier_status = OutlierStatus.NONE
        if g.rise.sample_count >= self._params.min_samples_for_outlier:
            z_rise = outlier_z(rise, g.rise.value, g.rise.dispersion)
            if z_rise > self._params.outlier_mad_k_severe:
                self._bump_outlier("severe")
                return self._reject(AfterheatRejection.SEVERE_OUTLIER)
            mild = z_rise > self._params.outlier_mad_k_mild
            for dim, val in ((g.time_to_peak, ttp_s), (g.duration, duration_s),
                             (g.decay, decay)):
                if dim.sample_count >= self._params.min_samples_for_outlier:
                    if outlier_z(val, dim.value, dim.dispersion) > self._params.outlier_mad_k_mild:
                        mild = True
            if mild:
                outlier_status = OutlierStatus.MILD
                weight *= 0.25
                self._bump_outlier("mild")

        new_general = _apply_bucket(g, rise, ttp_s, duration_s, decay, weight)

        new_buckets = dict(self._state.buckets)
        bucket_name = "general"
        heating_dur = context.heating_duration_s if context is not None else None
        if heating_dur is not None and is_finite(heating_dur) and heating_dur >= 0 \
                and (context is None or context.quality in (DataQuality.OK, DataQuality.STALE)):
            band = _bucket_index(heating_dur, self._params.heating_duration_buckets_s)
            key = f"heating_dur:{band}"
            if context.device_profile_key:
                key += f"|profile:{context.device_profile_key}"
            bkt = new_buckets.get(key) or _seed_bucket(key, new_general)
            new_buckets[key] = _apply_bucket(bkt, rise, ttp_s, duration_s, decay, weight)
            bucket_name = key

        direct = (context is not None and context.drive_end_source is not None) or \
            (del_rel is not None and del_rel >= self._params.inferred_setpoint_reduction_reliability
             and context is not None and context.drive_end_source_reliability is not None)
        sample = AfterheatSample(
            source_episode_id=episode.episode_id, learning_zone_id=self._zone,
            residual_rise_c=round(rise, 5), time_to_peak_s=ttp_s,
            afterheat_duration_s=duration_s, decay_indicator=round(decay, 6),
            episode_reliability=episode.reliability, feature_reliability=feat_rel,
            drive_end_reliability=del_rel, effective_weight=weight, bucket=bucket_name,
            near_zero=near_zero, feature_extractor_version=self._extractor_version(),
            builder_version=episode.builder_version,
            classifier_version=episode.classifier_version,
            data_quality=episode.time_quality.value,
            reason_codes=(("near_zero",) if near_zero else ())
            + (("indirect_start_reference",) if indirect else ())
            + (("mild_outlier",) if outlier_status is OutlierStatus.MILD else ()))

        processed = (self._state.processed_ids + (episode.episode_id,))[-self._params.dedup_max_ids:]
        recent = (self._state.recent_samples + (sample,))[-self._params.research_sample_cap:]
        n = new_general.sample_count
        agg = (self._state.aggregate_reliability * (n - 1) + episode.reliability) / n
        direct_inc = 1 if (context is not None and context.drive_end_source_reliability is not None
                           and context.drive_end_source is not None) else 0
        self._state = replace(
            self._state, general=new_general, buckets=new_buckets, processed_ids=processed,
            recent_samples=recent, aggregate_reliability=agg, last_update_ts=_iso(episode.end_ts),
            direct_drive_end_count=self._state.direct_drive_end_count + direct_inc,
            inferred_drive_end_count=self._state.inferred_drive_end_count + (1 - direct_inc),
            near_zero_count=self._state.near_zero_count + (1 if near_zero else 0))
        return elig

    def _bump_rejection(self, code: str) -> None:
        rc = dict(self._state.rejection_counts)
        rc[code] = rc.get(code, 0) + 1
        self._state = replace(self._state, rejection_counts=rc)

    def _bump_outlier(self, kind: str) -> None:
        oc = dict(self._state.outlier_counts)
        oc[kind] = oc.get(kind, 0) + 1
        self._state = replace(self._state, outlier_counts=oc)

    # -- predictions -----------------------------------------------------

    def predict(self, context: Any) -> Prediction:
        a = self.predict_afterheat(context)
        return Prediction(
            prediction_type=PredictionType.EXPECTED_OVERSHOOT,
            values={"expected_overshoot": a.residual_rise_c}, units={"expected_overshoot": "C"},
            confidence=a.confidence, reliability=a.reliability,
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
            prior_contribution=a.prior_contribution, learned_contribution=a.learned_contribution,
            fallback_used=a.fallback_used, evidence_count=a.evidence_count, bucket=a.bucket,
            confidence_cap=self._params.cold_start_confidence_cap if a.fallback_used else None,
            cap_reasons=("cold_start",) if a.fallback_used else (),
            missing_evidence=a.missing_evidence, reason_codes=a.reason_codes)

    def predict_afterheat(self, context: AfterheatPredictionContext) -> AfterheatPrediction:
        p = self._params
        missing: list[str] = []
        bucket = None
        if context.heating_duration_s is not None and is_finite(context.heating_duration_s):
            band = _bucket_index(context.heating_duration_s, p.heating_duration_buckets_s)
            key = f"heating_dur:{band}"
            if context.profile_id:
                key += f"|profile:{context.profile_id}"
            cand = self._state.buckets.get(key)
            if cand is not None and cand.sample_count >= p.bucket_min_samples:
                bucket = cand
        else:
            missing.append("heating_duration")

        if bucket is not None:
            return self._afterheat_prediction(bucket, learned=1.0, fallback=False,
                                              reasons=("conditioned_bucket",), missing=missing)
        if self._state.general.has_evidence:
            return self._afterheat_prediction(self._state.general, learned=1.0, fallback=False,
                                              reasons=("general_state",), missing=missing)
        prior = self._device_prior(context.profile_id) if self._device_prior else None
        rise = prior if (prior is not None and is_finite(prior) and prior >= 0) \
            else p.generic_prior_rise_c
        reason = "device_prior" if prior is not None else "generic_prior"
        return AfterheatPrediction(
            residual_rise_c=float(rise),
            time_to_peak_min=p.generic_prior_time_to_peak_s / 60.0,
            afterheat_duration_min=p.generic_prior_duration_s / 60.0,
            decay_indicator=p.generic_prior_decay, bucket=reason,
            confidence=min(0.2, p.cold_start_confidence_cap), reliability=0.3,
            prior_contribution=1.0, learned_contribution=0.0, fallback_used=True,
            evidence_count=0, drive_end_evidence_level=0.0,
            missing_evidence=tuple(missing), reason_codes=(reason,))

    def _afterheat_prediction(self, bucket: AfterheatBucket, *, learned, fallback, reasons,
                             missing) -> AfterheatPrediction:
        conf = self._confidence_value(bucket.sample_count, fallback)
        return AfterheatPrediction(
            residual_rise_c=max(0.0, bucket.rise.value),
            time_to_peak_min=max(0.0, bucket.time_to_peak.value) / 60.0,
            afterheat_duration_min=max(0.0, bucket.duration.value) / 60.0,
            decay_indicator=bucket.decay.value, bucket=bucket.key, confidence=conf,
            reliability=self._state.aggregate_reliability,
            prior_contribution=1.0 - learned, learned_contribution=learned,
            fallback_used=fallback, evidence_count=bucket.sample_count,
            drive_end_evidence_level=self._drive_end_level(),
            missing_evidence=tuple(missing), reason_codes=tuple(reasons))

    def _drive_end_level(self) -> float:
        total = self._state.direct_drive_end_count + self._state.inferred_drive_end_count
        if total == 0:
            return 0.0
        return self._state.direct_drive_end_count / total

    def predict_early_cutoff(self, context: AfterheatPredictionContext) -> Prediction:
        """Conservative, temperature-based cutoff: the residual rise (deg C) that
        active heating can stop short of the target by. Near-zero -> 0."""
        p = self._params
        a = self.predict_afterheat(context)
        residual = a.residual_rise_c
        reasons = list(a.reason_codes)
        if residual < p.min_residual_rise_c:
            residual = 0.0
            reasons.append("near_zero")
        residual = clamp(residual, 0.0, p.early_cutoff_max_c)
        # never exceed the remaining temperature delta, when known
        if context.current_temp is not None and context.target is not None \
                and is_finite(context.current_temp) and is_finite(context.target):
            remaining = max(0.0, context.target - context.current_temp)
            if residual > remaining:
                residual = remaining
                reasons.append("clamped_to_remaining_delta")
        if a.confidence < 0.3:
            reasons.append("low_confidence_cap")
        if not is_finite(residual):
            residual = 0.0
        return Prediction(
            prediction_type=PredictionType.RECOMMENDED_EARLY_CUTOFF,
            values={"expected_residual_rise": float(residual)},
            units={"expected_residual_rise": "C"},
            confidence=a.confidence, reliability=a.reliability if not a.fallback_used else 0.3,
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
            prior_contribution=1.0 if a.fallback_used else a.prior_contribution,
            learned_contribution=0.0 if a.fallback_used else a.learned_contribution,
            fallback_used=a.fallback_used, evidence_count=a.evidence_count, bucket=a.bucket,
            confidence_cap=p.cold_start_confidence_cap if a.fallback_used else None,
            cap_reasons=("cold_start",) if a.fallback_used else (),
            missing_evidence=a.missing_evidence, reason_codes=tuple(reasons))

    # -- confidence ------------------------------------------------------

    def _confidence_value(self, sample_count: int, fallback: bool) -> float:
        if fallback or sample_count <= 0:
            return min(0.2, self._params.cold_start_confidence_cap)
        coverage = min(sample_count / self._params.full_confidence_samples, 1.0)
        disp = self._state.general.rise.dispersion
        scale = max(abs(self._state.general.rise.value), 0.2)
        spread_penalty = clamp(1.0 - (disp / scale), 0.3, 1.0)
        rel = max(self._state.aggregate_reliability, 0.1)
        return clamp(coverage * spread_penalty * rel, 0.0, 1.0)

    def confidence(self, *, now: Optional[str] = None) -> ConfidenceContribution:
        g = self._state.general
        reasons: list[str] = []
        if not g.has_evidence:
            reasons.append("cold_start")
        if self._state.near_zero_count > 0 and self._state.near_zero_count == g.sample_count:
            reasons.append("near_zero_zone")
        if self._drive_end_level() < 0.5:
            reasons.append("mostly_inferred_drive_end")
        value = self._confidence_value(g.sample_count, not g.has_evidence)
        freshness, status = freshness_from_age(now, self._state.last_update_ts)
        return ConfidenceContribution(value=value, evidence_count=g.sample_count,
                                      reasons=tuple(reasons), freshness=freshness, status=status)

    # -- diagnostics / export -------------------------------------------

    def diagnostics(self) -> AfterheatDiagnostics:
        s = self._state
        g = s.general
        return AfterheatDiagnostics(
            general_rise_c=g.rise.value, general_time_to_peak_s=g.time_to_peak.value,
            general_duration_s=g.duration.value, general_decay=g.decay.value,
            bucket_rise={k: b.rise.value for k, b in s.buckets.items()},
            sample_counts={"general": g.sample_count,
                           **{k: b.sample_count for k, b in s.buckets.items()}},
            effective_sample_counts={"general": g.rise.effective_n,
                                     **{k: b.rise.effective_n for k, b in s.buckets.items()}},
            dispersion={"rise": g.rise.dispersion, "time_to_peak": g.time_to_peak.dispersion,
                        "duration": g.duration.dispersion, "decay": g.decay.dispersion},
            confidence=self.confidence().value,
            direct_drive_end_count=s.direct_drive_end_count,
            inferred_drive_end_count=s.inferred_drive_end_count,
            near_zero_count=s.near_zero_count, fallback=not g.has_evidence,
            missing_optional_evidence=("heating_duration",) if not s.buckets else (),
            rejection_counts=dict(s.rejection_counts), outlier_counts=dict(s.outlier_counts),
            last_update_ts=s.last_update_ts, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION)

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        s = self._state
        g = s.general
        if scope is ExportScope.SUPPORT:
            return {
                "general_rise_c": g.rise.value, "general_time_to_peak_s": g.time_to_peak.value,
                "general_duration_s": g.duration.value, "general_decay": g.decay.value,
                "buckets": {k: {"rise_c": b.rise.value, "samples": b.sample_count}
                            for k, b in s.buckets.items()},
                "confidence": self.confidence().value, "sample_count": g.sample_count,
                "direct_drive_end_count": s.direct_drive_end_count,
                "inferred_drive_end_count": s.inferred_drive_end_count,
                "near_zero_count": s.near_zero_count, "fallback": not g.has_evidence,
                "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
                "rejection_summary": dict(s.rejection_counts),
            }
        if scope is ExportScope.RESEARCH:
            return {
                "samples": [
                    {"residual_rise_c": x.residual_rise_c, "time_to_peak_s": x.time_to_peak_s,
                     "afterheat_duration_s": x.afterheat_duration_s,
                     "decay_indicator": x.decay_indicator, "near_zero": x.near_zero,
                     "bucket": x.bucket, "reliability": x.episode_reliability}
                    for x in s.recent_samples
                ],
                "model_version": MODEL_VERSION,
            }
        return {"unsupported": "raw_export_not_provided_by_model"}

    # -- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        self._state = self._empty_state()

    def rebuild(self, raw: Any = None,
                episodes: Optional[Sequence[Any]] = None) -> ModelRebuildResult:
        self.reset()
        raw_items = list(episodes or [])
        items: list[tuple[Any, Optional[AfterheatUpdateContext]]] = []
        for it in raw_items:
            if isinstance(it, AfterheatRebuildItem):
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
                if code == AfterheatRejection.DUPLICATE_EPISODE.value:
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
            "learning_zone_id": s.learning_zone_id, "general": _bucket_dict(s.general),
            "buckets": {k: _bucket_dict(b) for k, b in s.buckets.items()},
            "processed_ids": list(s.processed_ids),
            "aggregate_reliability": s.aggregate_reliability, "last_update_ts": s.last_update_ts,
            "rejection_counts": dict(s.rejection_counts), "outlier_counts": dict(s.outlier_counts),
            "dedup_count": s.dedup_count, "direct_drive_end_count": s.direct_drive_end_count,
            "inferred_drive_end_count": s.inferred_drive_end_count,
            "near_zero_count": s.near_zero_count,
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
        general = state.get("general", {})
        for dim in ("rise", "time_to_peak", "duration", "decay"):
            d = general.get(dim, {})
            for k in ("value", "effective_n", "dispersion"):
                v = d.get(k)
                if v is not None and not is_finite(v):
                    errors.append(f"non-finite general.{dim}.{k}")
        return errors

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        errors = self.validate_state(state)
        if errors:
            raise ValueError(f"invalid afterheat state: {errors}")
        buckets = {k: _bucket_from(b) for k, b in state.get("buckets", {}).items()}
        recent = tuple(_sample_from(x) for x in state.get("recent_samples", []))
        self._state = AfterheatState(
            learning_zone_id=state["learning_zone_id"], general=_bucket_from(state["general"]),
            buckets=buckets, processed_ids=tuple(state.get("processed_ids", [])),
            recent_samples=recent, aggregate_reliability=state.get("aggregate_reliability", 0.0),
            last_update_ts=state.get("last_update_ts"),
            rejection_counts=dict(state.get("rejection_counts", {})),
            outlier_counts=dict(state.get("outlier_counts", {})),
            dedup_count=state.get("dedup_count", 0),
            direct_drive_end_count=state.get("direct_drive_end_count", 0),
            inferred_drive_end_count=state.get("inferred_drive_end_count", 0),
            near_zero_count=state.get("near_zero_count", 0))

    def migrate_state(self, old_model_version: int, old_parameter_version: int,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        if old_model_version == MODEL_VERSION:
            return state
        raise ValueError(
            f"no migration path from model_version {old_model_version} to {MODEL_VERSION}")


# -- module helpers -----------------------------------------------------------

def _apply_bucket(b: AfterheatBucket, rise, ttp, duration, decay, weight) -> AfterheatBucket:
    return AfterheatBucket(
        key=b.key, rise=_apply_dim(b.rise, rise, weight),
        time_to_peak=_apply_dim(b.time_to_peak, ttp, weight),
        duration=_apply_dim(b.duration, duration, weight),
        decay=_apply_dim(b.decay, decay, weight), sample_count=b.sample_count + 1)


def _seed_bucket(key: str, general: AfterheatBucket) -> AfterheatBucket:
    seed = lambda d: _Dim(value=d.value, effective_n=1.0, dispersion=d.dispersion, sample_count=0)
    return AfterheatBucket(key=key, rise=seed(general.rise),
                           time_to_peak=seed(general.time_to_peak),
                           duration=seed(general.duration), decay=seed(general.decay),
                           sample_count=0)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _dim_dict(d: _Dim) -> dict:
    return {"value": d.value, "effective_n": d.effective_n, "dispersion": d.dispersion,
            "sample_count": d.sample_count}


def _dim_from(d: Mapping[str, Any]) -> _Dim:
    return _Dim(value=d["value"], effective_n=d["effective_n"], dispersion=d["dispersion"],
                sample_count=d["sample_count"])


def _bucket_dict(b: AfterheatBucket) -> dict:
    return {"key": b.key, "rise": _dim_dict(b.rise), "time_to_peak": _dim_dict(b.time_to_peak),
            "duration": _dim_dict(b.duration), "decay": _dim_dict(b.decay),
            "sample_count": b.sample_count}


def _bucket_from(d: Mapping[str, Any]) -> AfterheatBucket:
    return AfterheatBucket(key=d["key"], rise=_dim_from(d["rise"]),
                           time_to_peak=_dim_from(d["time_to_peak"]),
                           duration=_dim_from(d["duration"]), decay=_dim_from(d["decay"]),
                           sample_count=d["sample_count"])


def _sample_dict(x: AfterheatSample) -> dict:
    return {
        "source_episode_id": x.source_episode_id, "learning_zone_id": x.learning_zone_id,
        "residual_rise_c": x.residual_rise_c, "time_to_peak_s": x.time_to_peak_s,
        "afterheat_duration_s": x.afterheat_duration_s, "decay_indicator": x.decay_indicator,
        "episode_reliability": x.episode_reliability, "feature_reliability": x.feature_reliability,
        "drive_end_reliability": x.drive_end_reliability, "effective_weight": x.effective_weight,
        "bucket": x.bucket, "near_zero": x.near_zero,
        "feature_extractor_version": x.feature_extractor_version,
        "builder_version": x.builder_version, "classifier_version": x.classifier_version,
        "data_quality": x.data_quality, "reason_codes": list(x.reason_codes),
    }


def _sample_from(d: Mapping[str, Any]) -> AfterheatSample:
    return AfterheatSample(
        source_episode_id=d["source_episode_id"], learning_zone_id=d["learning_zone_id"],
        residual_rise_c=d["residual_rise_c"], time_to_peak_s=d["time_to_peak_s"],
        afterheat_duration_s=d["afterheat_duration_s"], decay_indicator=d["decay_indicator"],
        episode_reliability=d["episode_reliability"], feature_reliability=d["feature_reliability"],
        drive_end_reliability=d["drive_end_reliability"], effective_weight=d["effective_weight"],
        bucket=d["bucket"], near_zero=d["near_zero"],
        feature_extractor_version=d["feature_extractor_version"],
        builder_version=d["builder_version"], classifier_version=d["classifier_version"],
        data_quality=d["data_quality"], reason_codes=tuple(d.get("reason_codes", [])))


def afterheat_model_definition():
    from ..registry import ModelDefinition

    return ModelDefinition(
        model_name=MODEL_NAME, model_factory=lambda: AfterheatModel("__definition__"),
        model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
        consumed_raw_tracks=(), consumed_episode_types=(EpisodeType.AFTERHEAT,),
        supported_prediction_types=(
            PredictionType.EXPECTED_OVERSHOOT, PredictionType.AFTERHEAT_TIME_TO_PEAK,
            PredictionType.AFTERHEAT_DURATION, PredictionType.RECOMMENDED_EARLY_CUTOFF),
        required_features=("indoor_temp", "trv_setpoint"),
        optional_features=("valve_opening", "hvac_action", "outdoor_temp",
                           "solar_radiation", "radiator_profile"),
        required_capability_flags=(), regime_dependent=True, advisory=False,
        control_relevant=True, rebuildable=True, can_be_not_available=False, min_trv_only=True)
