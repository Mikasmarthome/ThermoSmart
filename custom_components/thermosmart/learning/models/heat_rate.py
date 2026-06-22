"""HeatRateModel for LE 2.0 (pure Python).

Learns a zone's heating rate (deg C per hour) from completed, undisturbed
ACTIVE_HEATING episodes. Two levels: a general zone rate (the TRV-only fallback)
and conditioned outdoor/profile buckets used only with enough evidence. No Home
Assistant / storage / coordinator dependency; mutates no input; calls no other
model; sends no command.

Heat-rate definition (authoritative, single source): rate = temperature_delta /
duration, in deg C/h, taken over the active heating phase via the Phase-4
FeatureExtractor. Seasonal comparability comes from *bucketing by outdoor band*,
not from a fragile ``rate / (target - outdoor)`` division (deliberately rejected:
it distorts the physical rate and divides by an artificial floor).
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
from ..episode_schemas import EpisodeType, HeatingEpisode
from ..features import FeatureExtractor, FeatureName, FeatureStatus
from .base import clamp, is_finite, outlier_z, robust_ema_update

MODEL_VERSION = 1
PARAMETER_VERSION = 1
MODEL_NAME = "heat_rate"


# -- rejection reasons --------------------------------------------------------

class HeatRateRejection(Enum):
    WRONG_EPISODE_TYPE = "wrong_episode_type"
    DISTURBED = "disturbed"
    UNKNOWN_DOMINANT = "unknown_dominant"
    INSUFFICIENT_DURATION = "insufficient_duration"
    INSUFFICIENT_POINTS = "insufficient_points"
    MISSING_INDOOR_TEMP = "missing_indoor_temperature"
    INVALID_TRAJECTORY = "invalid_trajectory"
    IMPLAUSIBLE_RATE = "implausible_rate"
    LOW_RELIABILITY = "low_reliability"
    AFTERHEAT_CONTAMINATION = "afterheat_contamination"
    WINDOW_CONFOUNDER = "window_confounder"
    HEATING_FAILURE = "heating_failure"
    DUPLICATE_EPISODE = "duplicate_episode"
    VERSION_MISMATCH = "version_mismatch"
    SEVERE_OUTLIER = "severe_outlier"
    WRONG_ZONE = "wrong_zone"
    CONTEXT_EPISODE_MISMATCH = "context_episode_mismatch"


_TO_CONTRACT = {
    HeatRateRejection.WRONG_EPISODE_TYPE: RejectionReason.NOT_ELIGIBLE,
    HeatRateRejection.DISTURBED: RejectionReason.DISTURBED_REGIME,
    HeatRateRejection.UNKNOWN_DOMINANT: RejectionReason.UNKNOWN_REGIME,
    HeatRateRejection.INSUFFICIENT_DURATION: RejectionReason.NOT_ELIGIBLE,
    HeatRateRejection.INSUFFICIENT_POINTS: RejectionReason.NOT_ELIGIBLE,
    HeatRateRejection.MISSING_INDOOR_TEMP: RejectionReason.MISSING_REQUIRED_DATA,
    HeatRateRejection.INVALID_TRAJECTORY: RejectionReason.MISSING_REQUIRED_DATA,
    HeatRateRejection.IMPLAUSIBLE_RATE: RejectionReason.OUTLIER,
    HeatRateRejection.LOW_RELIABILITY: RejectionReason.LOW_RELIABILITY,
    HeatRateRejection.AFTERHEAT_CONTAMINATION: RejectionReason.NOT_ELIGIBLE,
    HeatRateRejection.WINDOW_CONFOUNDER: RejectionReason.DISTURBED_REGIME,
    HeatRateRejection.HEATING_FAILURE: RejectionReason.DISTURBED_REGIME,
    HeatRateRejection.DUPLICATE_EPISODE: RejectionReason.DUPLICATE,
    HeatRateRejection.VERSION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    HeatRateRejection.SEVERE_OUTLIER: RejectionReason.OUTLIER,
    HeatRateRejection.WRONG_ZONE: RejectionReason.NOT_ELIGIBLE,
    HeatRateRejection.CONTEXT_EPISODE_MISMATCH: RejectionReason.NOT_ELIGIBLE,
}


# -- parameters ---------------------------------------------------------------

@dataclass(frozen=True)
class HeatRateParameters:
    """Versioned, conservative, provisional thresholds."""

    min_episode_reliability: float = 0.4
    min_duration_s: float = 180.0
    min_points: int = 3
    min_temp_delta_c: float = 0.2
    min_rate_c_per_h: float = 0.05
    max_rate_c_per_h: float = 12.0
    max_sample_weight: float = 1.0
    bucket_min_samples: int = 5
    bucket_seed_evidence: float = 1.0
    outlier_mad_k_mild: float = 3.0
    outlier_mad_k_severe: float = 6.0
    min_samples_for_outlier: int = 5
    dedup_max_ids: int = 500
    research_sample_cap: int = 50
    outdoor_buckets: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    generic_prior_rate_c_per_h: float = 2.0
    preheat_min_minutes: float = 0.0
    preheat_max_minutes: float = 180.0
    preheat_fallback_minutes: float = 60.0
    cold_start_confidence_cap: float = 0.4
    full_confidence_samples: float = 20.0
    parameter_version: int = PARAMETER_VERSION


# -- sample / bucket / state --------------------------------------------------

@dataclass(frozen=True)
class HeatRateSample:
    source_episode_id: str
    learning_zone_id: str
    rate_c_per_h: float
    duration_s: float
    temp_delta_c: float
    episode_reliability: float
    effective_weight: float
    bucket: str
    outdoor_band: Optional[int]
    profile_id: Optional[str]
    feature_extractor_version: int
    builder_version: int
    classifier_version: int
    data_quality: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class HeatRateBucket:
    key: str
    rate_c_per_h: float
    effective_n: float
    dispersion: float
    sample_count: int
    last_update_ts: Optional[str] = None

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0 and self.effective_n > 0.0


def _empty_bucket(key: str) -> HeatRateBucket:
    return HeatRateBucket(key=key, rate_c_per_h=0.0, effective_n=0.0,
                          dispersion=0.0, sample_count=0)


@dataclass(frozen=True)
class HeatRateState:
    learning_zone_id: str
    general: HeatRateBucket
    buckets: Mapping[str, HeatRateBucket] = field(default_factory=dict)
    processed_ids: tuple[str, ...] = ()
    recent_samples: tuple[HeatRateSample, ...] = ()
    aggregate_reliability: float = 0.0
    last_update_ts: Optional[str] = None
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    dedup_count: int = 0
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION


@dataclass(frozen=True)
class HeatRateDiagnostics:
    general_rate_c_per_h: float
    bucket_rates: Mapping[str, float]
    sample_counts: Mapping[str, int]
    effective_sample_counts: Mapping[str, float]
    dispersion: Mapping[str, float]
    confidence: float
    last_update_ts: Optional[str]
    rejection_counts: Mapping[str, int]
    outlier_counts: Mapping[str, int]
    dedup_count: int
    trv_only_fallback: bool
    missing_optional_evidence: tuple[str, ...]
    model_version: int
    parameter_version: int


@dataclass(frozen=True)
class HeatRatePredictionContext:
    current_temp: Optional[float] = None
    target: Optional[float] = None
    outdoor_temp: Optional[Measurement] = None
    profile_id: Optional[str] = None
    wind_speed: Optional[Measurement] = None
    solar_radiation: Optional[Measurement] = None


@dataclass(frozen=True)
class HeatRateUpdateContext:
    """Persistable, reproducible conditioning evidence bound to one episode.

    JSON-serializable, immutable, no HA objects, no visible entity/zone names.
    Conditioned buckets are derived deterministically from ``outdoor_temperature_c``
    so a rebuild from (episode + this context) reproduces the same bucket state.
    """

    source_episode_id: str
    outdoor_temperature_c: Optional[float] = None
    device_profile_key: Optional[str] = None
    wind_bucket: Optional[str] = None
    solar_bucket: Optional[str] = None
    source_feature_version: int = 1
    source_refs: tuple[str, ...] = ()
    quality: DataQuality = DataQuality.OK

    def __post_init__(self) -> None:
        if not self.source_episode_id:
            raise ValueError("source_episode_id must be non-empty")
        if self.outdoor_temperature_c is not None and not is_finite(self.outdoor_temperature_c):
            raise ValueError("outdoor_temperature_c must be finite (or None)")

    def to_dict(self) -> dict:
        return {
            "source_episode_id": self.source_episode_id,
            "outdoor_temperature_c": self.outdoor_temperature_c,
            "device_profile_key": self.device_profile_key,
            "wind_bucket": self.wind_bucket, "solar_bucket": self.solar_bucket,
            "source_feature_version": self.source_feature_version,
            "source_refs": list(self.source_refs), "quality": self.quality.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HeatRateUpdateContext":
        return cls(
            source_episode_id=d["source_episode_id"],
            outdoor_temperature_c=d.get("outdoor_temperature_c"),
            device_profile_key=d.get("device_profile_key"),
            wind_bucket=d.get("wind_bucket"), solar_bucket=d.get("solar_bucket"),
            source_feature_version=d.get("source_feature_version", 1),
            source_refs=tuple(d.get("source_refs", [])),
            quality=DataQuality(d.get("quality", DataQuality.OK.value)))


@dataclass(frozen=True)
class HeatRateRebuildItem:
    episode: Any
    context: Optional[HeatRateUpdateContext] = None


# -- helpers ------------------------------------------------------------------

def _usable(m: Optional[Measurement]) -> Optional[float]:
    if m is not None and m.value is not None and is_finite(m.value) \
            and m.quality in (DataQuality.OK, DataQuality.STALE):
        return m.value
    return None


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    idx = 0
    for edge in edges:
        if value >= edge:
            idx += 1
        else:
            break
    return idx


# -- model --------------------------------------------------------------------

class HeatRateModel:
    """Pure heating-rate model implementing the LE 2.0 Model contract."""

    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    parameter_version = PARAMETER_VERSION
    consumed_raw_tracks: tuple[str, ...] = ()
    consumed_episode_types = (EpisodeType.HEATING.value,)
    required_features = ("indoor_temp", "trv_setpoint", "target")
    optional_features = ("outdoor_temp", "wind_speed", "solar_radiation",
                         "valve_opening", "radiator_profile")
    supported_prediction_types = (PredictionType.HEAT_RATE, PredictionType.PREHEAT_MINUTES)
    min_tier = 0

    def __init__(
        self, learning_zone_id: str,
        params: Optional[HeatRateParameters] = None,
        *,
        device_prior: Optional[Callable[[Optional[str]], Optional[float]]] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or HeatRateParameters()
        self._device_prior = device_prior
        self._extractor = extractor or FeatureExtractor()
        self._state = self._empty_state()

    # -- declared cold-start prior --------------------------------------

    def cold_start_prior(self) -> Mapping[str, Any]:
        return {"general_rate_c_per_h": None,
                "generic_prior_rate_c_per_h": self._params.generic_prior_rate_c_per_h}

    def _empty_state(self) -> HeatRateState:
        return HeatRateState(learning_zone_id=self._zone, general=_empty_bucket("general"))

    # -- rate computation (authoritative) -------------------------------

    def _compute_rate(self, episode: HeatingEpisode):
        fs = self._extractor.extract_trajectory_features(episode.trajectory)
        valid = fs.get(FeatureName.TRAJ_VALID_POINTS).value
        delta_r = fs.get(FeatureName.TRAJ_DELTA)
        dur_r = fs.get(FeatureName.TRAJ_DURATION)
        plaus = fs.get(FeatureName.TRAJ_PLAUSIBILITY)
        slope_r = fs.get(FeatureName.TRAJ_LINEAR_SLOPE)
        feature_reliability = slope_r.reliability if slope_r.is_present else 0.0
        if valid is None or valid < 1:
            return None, HeatRateRejection.MISSING_INDOOR_TEMP, feature_reliability, 0
        if not delta_r.is_present or not dur_r.is_present:
            return None, HeatRateRejection.INVALID_TRAJECTORY, feature_reliability, int(valid)
        if int(valid) < self._params.min_points:
            return None, HeatRateRejection.INSUFFICIENT_POINTS, feature_reliability, int(valid)
        if plaus.status is FeatureStatus.INVALID:
            return None, HeatRateRejection.IMPLAUSIBLE_RATE, feature_reliability, int(valid)
        delta = delta_r.value
        duration_s = dur_r.value
        if duration_s <= 0 or not is_finite(delta) or not is_finite(duration_s):
            return None, HeatRateRejection.INVALID_TRAJECTORY, feature_reliability, int(valid)
        if delta < self._params.min_temp_delta_c:
            return None, HeatRateRejection.IMPLAUSIBLE_RATE, feature_reliability, int(valid)
        rate = delta / (duration_s / 3600.0)
        if not is_finite(rate) or rate < self._params.min_rate_c_per_h \
                or rate > self._params.max_rate_c_per_h:
            return None, HeatRateRejection.IMPLAUSIBLE_RATE, feature_reliability, int(valid)
        return (rate, delta, duration_s, feature_reliability, int(valid)), None, \
            feature_reliability, int(valid)

    # -- eligibility -----------------------------------------------------

    def update_eligibility(self, episode: Any) -> UpdateEligibilityResult:
        p = self._params
        if not isinstance(episode, HeatingEpisode):
            return self._reject(HeatRateRejection.WRONG_EPISODE_TYPE)
        if episode.learning_zone_id != self._zone:
            return self._reject(HeatRateRejection.WRONG_ZONE)
        if episode.episode_schema_version != 1:
            return self._reject(HeatRateRejection.VERSION_MISMATCH)
        if episode.regime is not Regime.ACTIVE_HEATING:
            return self._reject(HeatRateRejection.DISTURBED, regime=episode.regime)
        cf = set(episode.confounder_flags)
        if "window_open" in cf:
            return self._reject(HeatRateRejection.WINDOW_CONFOUNDER)
        if "heating_failure" in cf:
            return self._reject(HeatRateRejection.HEATING_FAILURE)
        if "reheating" in cf:
            return self._reject(HeatRateRejection.AFTERHEAT_CONTAMINATION)
        if episode.episode_id in self._state.processed_ids:
            return self._reject(HeatRateRejection.DUPLICATE_EPISODE)
        if episode.reliability < p.min_episode_reliability:
            return self._reject(HeatRateRejection.LOW_RELIABILITY)
        if (episode.end_ts - episode.start_ts).total_seconds() < p.min_duration_s:
            return self._reject(HeatRateRejection.INSUFFICIENT_DURATION)
        computed, reject, feat_rel, valid = self._compute_rate(episode)
        if reject is not None:
            return self._reject(reject)
        rate = computed[0]
        weight = clamp(episode.reliability * max(feat_rel, 0.1), 0.0, p.max_sample_weight)
        return UpdateEligibilityResult(
            accepted=True, reliability=episode.reliability, weight=weight,
            outlier_status=OutlierStatus.NONE, regime=Regime.ACTIVE_HEATING,
            source_episode_id=episode.episode_id,
            feature_extractor_version=self._extractor_version())

    def _reject(self, reason: HeatRateRejection, *, regime=None) -> UpdateEligibilityResult:
        return UpdateEligibilityResult(
            accepted=False, reliability=0.0, weight=0.0,
            rejection_reason=_TO_CONTRACT[reason], confounder_flags=(reason.value,),
            regime=regime)

    def _extractor_version(self) -> int:
        from ..features import FEATURE_EXTRACTOR_VERSION
        return FEATURE_EXTRACTOR_VERSION

    # -- update ----------------------------------------------------------

    def update(self, episode: Any,
               context: Optional["HeatRateUpdateContext"] = None) -> UpdateEligibilityResult:
        """Learn from one HeatingEpisode. ``context`` is a typed, persistable,
        episode-bound evidence record supplying the conditioning data for buckets
        (the episode itself stores no environmental data); without it only the
        general rate updates. A rebuild from (episode + context) is reproducible."""
        if context is not None and context.source_episode_id != episode.episode_id:
            self._bump_rejection(HeatRateRejection.CONTEXT_EPISODE_MISMATCH.value)
            return self._reject(HeatRateRejection.CONTEXT_EPISODE_MISMATCH)
        elig = self.update_eligibility(episode)
        if not elig.accepted:
            self._bump_rejection(elig.confounder_flags[0] if elig.confounder_flags else "unknown")
            if elig.confounder_flags and elig.confounder_flags[0] == \
                    HeatRateRejection.DUPLICATE_EPISODE.value:
                self._state = replace(self._state, dedup_count=self._state.dedup_count + 1)
            return elig

        computed, _, feat_rel, valid = self._compute_rate(episode)
        rate, delta, duration_s, _, _ = computed
        weight = elig.weight

        # relative outlier scoring vs the zone's own general evidence
        general = self._state.general
        outlier_status = OutlierStatus.NONE
        if general.sample_count >= self._params.min_samples_for_outlier:
            z = outlier_z(rate, general.rate_c_per_h, general.dispersion)
            if z > self._params.outlier_mad_k_severe:
                self._bump_outlier("severe")
                return self._reject(HeatRateRejection.SEVERE_OUTLIER)
            if z > self._params.outlier_mad_k_mild:
                outlier_status = OutlierStatus.MILD
                weight *= 0.25
                self._bump_outlier("mild")

        # general rate is the authoritative zone rate and TRV-only fallback
        new_general = _apply(general, rate, weight)

        new_buckets = dict(self._state.buckets)
        bucket_name = "general"
        outdoor_band: Optional[int] = None
        profile_id: Optional[str] = None
        if context is not None and context.outdoor_temperature_c is not None \
                and is_finite(context.outdoor_temperature_c) \
                and context.quality in (DataQuality.OK, DataQuality.STALE):
            ov = context.outdoor_temperature_c
            profile_id = context.device_profile_key
            key = self._bucket_key(ov, profile_id)  # deterministic, versioned banding
            outdoor_band = _bucket_index(ov, self._params.outdoor_buckets)
            bkt = new_buckets.get(key)
            if bkt is None:
                # new buckets start from the general rate, then blend in evidence
                bkt = HeatRateBucket(
                    key=key, rate_c_per_h=new_general.rate_c_per_h,
                    effective_n=self._params.bucket_seed_evidence,
                    dispersion=new_general.dispersion, sample_count=0)
            new_buckets[key] = _apply(bkt, rate, weight)
            bucket_name = key

        sample = HeatRateSample(
            source_episode_id=episode.episode_id, learning_zone_id=self._zone,
            rate_c_per_h=round(rate, 5), duration_s=duration_s, temp_delta_c=delta,
            episode_reliability=episode.reliability, effective_weight=weight,
            bucket=bucket_name, outdoor_band=outdoor_band, profile_id=profile_id,
            feature_extractor_version=self._extractor_version(),
            builder_version=episode.builder_version,
            classifier_version=episode.classifier_version,
            data_quality=episode.time_quality.value,
            reason_codes=("mild_outlier",) if outlier_status is OutlierStatus.MILD else ())

        processed = (self._state.processed_ids + (episode.episode_id,))[-self._params.dedup_max_ids:]
        recent = (self._state.recent_samples + (sample,))[-self._params.research_sample_cap:]
        n = new_general.sample_count
        agg = (self._state.aggregate_reliability * (n - 1) + episode.reliability) / n
        self._state = replace(
            self._state, general=new_general, buckets=new_buckets, processed_ids=processed,
            recent_samples=recent, aggregate_reliability=agg,
            last_update_ts=_iso(episode.end_ts))
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
        return self.predict_heat_rate(context)

    def predict_heat_rate(self, context: HeatRatePredictionContext) -> Prediction:
        p = self._params
        missing: list[str] = []
        outdoor = _usable(context.outdoor_temp)
        if outdoor is None:
            missing.append("outdoor_temp")

        bucket = None
        if outdoor is not None:
            key = self._bucket_key(outdoor, context.profile_id)
            cand = self._state.buckets.get(key)
            if cand is not None and cand.sample_count >= p.bucket_min_samples:
                bucket = cand

        if bucket is not None:
            return self._rate_prediction(bucket.rate_c_per_h, bucket.sample_count,
                                         learned=1.0, fallback=False, bucket=bucket.key,
                                         reasons=("conditioned_bucket",), missing=missing)
        if self._state.general.has_evidence:
            g = self._state.general
            return self._rate_prediction(g.rate_c_per_h, g.sample_count, learned=1.0,
                                         fallback=False, bucket="general",
                                         reasons=("general_rate",), missing=missing)
        prior = self._device_prior(context.profile_id) if self._device_prior else None
        if prior is not None and is_finite(prior) and prior > 0:
            return self._rate_prediction(prior, 0, learned=0.0, fallback=True,
                                         bucket="device_prior",
                                         reasons=("device_prior",), missing=missing,
                                         cap=p.cold_start_confidence_cap)
        return self._rate_prediction(p.generic_prior_rate_c_per_h, 0, learned=0.0,
                                     fallback=True, bucket="generic_prior",
                                     reasons=("generic_prior",), missing=missing,
                                     cap=p.cold_start_confidence_cap)

    def _rate_prediction(self, rate, sample_count, *, learned, fallback, bucket,
                         reasons, missing, cap=None) -> Prediction:
        rate = float(rate)
        if not is_finite(rate):
            rate = self._params.generic_prior_rate_c_per_h
            fallback = True
        conf = self._confidence_value(sample_count, fallback)
        if cap is not None:
            conf = min(conf, cap)
        return Prediction(
            prediction_type=PredictionType.HEAT_RATE,
            values={"heat_rate": rate}, units={"heat_rate": "C/h"},
            confidence=conf, reliability=self._state.aggregate_reliability if not fallback else 0.3,
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
            prior_contribution=1.0 - learned, learned_contribution=learned,
            fallback_used=fallback, evidence_count=sample_count, bucket=bucket,
            confidence_cap=cap, cap_reasons=("cold_start",) if cap is not None else (),
            missing_evidence=tuple(missing), reason_codes=tuple(reasons))

    def predict_preheat(self, context: HeatRatePredictionContext) -> Prediction:
        p = self._params
        rate_pred = self.predict_heat_rate(context)
        rate = rate_pred.control_value("heat_rate")
        current, target = context.current_temp, context.target
        reasons = list(rate_pred.reason_codes)
        missing = list(rate_pred.missing_evidence)

        if current is None or target is None or not is_finite(current) or not is_finite(target):
            reasons.append("missing_temps_fallback")
            return self._preheat_prediction(p.preheat_fallback_minutes, rate_pred,
                                            reasons, missing, guard_fallback=True)
        if target - current <= 0:
            return self._preheat_prediction(0.0, rate_pred, reasons + ["target_reached"],
                                            missing, guard_fallback=False)
        if rate <= 0 or not is_finite(rate):
            reasons.append("invalid_rate_fallback")
            return self._preheat_prediction(p.preheat_fallback_minutes, rate_pred,
                                            reasons, missing, guard_fallback=True)

        minutes = (target - current) / rate * 60.0
        if not is_finite(minutes):
            minutes = p.preheat_fallback_minutes
            reasons.append("non_finite_guard")
        minutes = clamp(minutes, p.preheat_min_minutes, p.preheat_max_minutes)
        if rate_pred.confidence < 0.3:
            minutes = min(minutes, p.preheat_max_minutes)
            reasons.append("low_confidence_cap")
        return self._preheat_prediction(minutes, rate_pred, reasons, missing,
                                        guard_fallback=False)

    def _preheat_prediction(self, minutes, rate_pred, reasons, missing, *,
                            guard_fallback: bool) -> Prediction:
        minutes = float(minutes)
        if not is_finite(minutes):
            minutes = self._params.preheat_fallback_minutes
            guard_fallback = True
        # A guard-based preheat (missing temps / invalid rate) is prior-based,
        # regardless of the rate's provenance; otherwise inherit the rate split.
        fallback = guard_fallback or rate_pred.fallback_used
        if fallback:
            prior_contribution, learned_contribution = 1.0, 0.0
        else:
            prior_contribution = rate_pred.prior_contribution
            learned_contribution = rate_pred.learned_contribution
        return Prediction(
            prediction_type=PredictionType.PREHEAT_MINUTES,
            values={"preheat_minutes": minutes}, units={"preheat_minutes": "min"},
            confidence=rate_pred.confidence,
            reliability=rate_pred.reliability if not fallback else 0.3,
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
            prior_contribution=prior_contribution, learned_contribution=learned_contribution,
            fallback_used=fallback, evidence_count=rate_pred.evidence_count,
            bucket=rate_pred.bucket, confidence_cap=rate_pred.confidence_cap,
            cap_reasons=rate_pred.cap_reasons, missing_evidence=tuple(missing),
            reason_codes=tuple(reasons))

    # -- confidence ------------------------------------------------------

    def _confidence_value(self, sample_count: int, fallback: bool) -> float:
        if fallback or sample_count <= 0:
            return min(0.2, self._params.cold_start_confidence_cap)
        coverage = min(sample_count / self._params.full_confidence_samples, 1.0)
        dispersion = self._state.general.dispersion
        rate = max(abs(self._state.general.rate_c_per_h), 1e-6)
        spread_penalty = clamp(1.0 - (dispersion / rate), 0.3, 1.0)
        rel = max(self._state.aggregate_reliability, 0.1)
        return clamp(coverage * spread_penalty * rel, 0.0, 1.0)

    def confidence(self) -> ConfidenceContribution:
        g = self._state.general
        reasons: list[str] = []
        if not g.has_evidence:
            reasons.append("cold_start")
        if g.sample_count < self._params.bucket_min_samples:
            reasons.append("few_samples")
        value = self._confidence_value(g.sample_count, not g.has_evidence)
        return ConfidenceContribution(value=value, evidence_count=g.sample_count,
                                      reasons=tuple(reasons))

    # -- diagnostics / export -------------------------------------------

    def diagnostics(self) -> HeatRateDiagnostics:
        s = self._state
        return HeatRateDiagnostics(
            general_rate_c_per_h=s.general.rate_c_per_h,
            bucket_rates={k: b.rate_c_per_h for k, b in s.buckets.items()},
            sample_counts={"general": s.general.sample_count,
                           **{k: b.sample_count for k, b in s.buckets.items()}},
            effective_sample_counts={"general": s.general.effective_n,
                                     **{k: b.effective_n for k, b in s.buckets.items()}},
            dispersion={"general": s.general.dispersion,
                        **{k: b.dispersion for k, b in s.buckets.items()}},
            confidence=self.confidence().value,
            last_update_ts=s.last_update_ts, rejection_counts=dict(s.rejection_counts),
            outlier_counts=dict(s.outlier_counts), dedup_count=s.dedup_count,
            trv_only_fallback=not s.general.has_evidence,
            missing_optional_evidence=("outdoor_temp",) if not s.buckets else (),
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION)

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        s = self._state
        if scope is ExportScope.SUPPORT:
            return {
                "general_rate_c_per_h": s.general.rate_c_per_h,
                "buckets": {k: {"rate_c_per_h": b.rate_c_per_h, "samples": b.sample_count}
                            for k, b in s.buckets.items()},
                "confidence": self.confidence().value,
                "sample_count": s.general.sample_count,
                "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
                "rejection_summary": dict(s.rejection_counts),
                "fallback": not s.general.has_evidence,
            }
        if scope is ExportScope.RESEARCH:
            return {
                "samples": [
                    {"rate_c_per_h": x.rate_c_per_h, "duration_s": x.duration_s,
                     "temp_delta_c": x.temp_delta_c, "bucket": x.bucket,
                     "reliability": x.episode_reliability}
                    for x in s.recent_samples
                ],  # no ids, no entity/zone names, no raw trajectories
                "model_version": MODEL_VERSION,
            }
        # RAW export is the export layer's job, not the model's.
        return {"unsupported": "raw_export_not_provided_by_model"}

    # -- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        self._state = self._empty_state()

    def rebuild(self, raw: Any = None,
                episodes: Optional[Sequence[Any]] = None) -> ModelRebuildResult:
        """Deterministically rebuild from a sequence of HeatingEpisode or
        HeatRateRebuildItem. Conditioned buckets are restored only when a typed
        context is supplied; result equals sequential updates of the same items."""
        self.reset()
        raw_items = list(episodes or [])
        items: list[tuple[Any, Optional[HeatRateUpdateContext]]] = []
        for it in raw_items:
            if isinstance(it, HeatRateRebuildItem):
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
                if code == HeatRateRejection.DUPLICATE_EPISODE.value:
                    duplicates += 1
        return ModelRebuildResult(
            processed_count=len(items), accepted_count=accepted, rejected_count=rejected,
            duplicate_count=duplicates, error_count=0, rejection_counts=rejection_counts,
            started_from_prior=True, deterministic=True)

    # -- state (de)serialization ----------------------------------------

    def serialize_state(self) -> Mapping[str, Any]:
        s = self._state
        return {
            "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
            "learning_zone_id": s.learning_zone_id,
            "general": _bucket_dict(s.general),
            "buckets": {k: _bucket_dict(b) for k, b in s.buckets.items()},
            "processed_ids": list(s.processed_ids),
            "aggregate_reliability": s.aggregate_reliability,
            "last_update_ts": s.last_update_ts,
            "rejection_counts": dict(s.rejection_counts),
            "outlier_counts": dict(s.outlier_counts),
            "dedup_count": s.dedup_count,
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
        for name in ("general", *list(state.get("buckets", {}).values())):
            bucket = state["general"] if name == "general" else name
            for k in ("rate_c_per_h", "effective_n", "dispersion"):
                v = bucket.get(k)
                if v is not None and not is_finite(v):
                    errors.append(f"non-finite {k}")
        return errors

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        errors = self.validate_state(state)
        if errors:
            raise ValueError(f"invalid heat-rate state: {errors}")
        buckets = {k: _bucket_from(b) for k, b in state.get("buckets", {}).items()}
        recent = tuple(_sample_from(x) for x in state.get("recent_samples", []))
        self._state = HeatRateState(
            learning_zone_id=state["learning_zone_id"],
            general=_bucket_from(state["general"]), buckets=buckets,
            processed_ids=tuple(state.get("processed_ids", [])),
            recent_samples=recent,
            aggregate_reliability=state.get("aggregate_reliability", 0.0),
            last_update_ts=state.get("last_update_ts"),
            rejection_counts=dict(state.get("rejection_counts", {})),
            outlier_counts=dict(state.get("outlier_counts", {})),
            dedup_count=state.get("dedup_count", 0))

    def migrate_state(self, old_model_version: int, old_parameter_version: int,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        if old_model_version == MODEL_VERSION:
            return state  # current version: identity, no v2+ migration needed
        raise ValueError(
            f"no migration path from model_version {old_model_version} to {MODEL_VERSION}")

    # -- internals -------------------------------------------------------

    def _bucket_key(self, outdoor: float, profile_id: Optional[str]) -> str:
        band = _bucket_index(outdoor, self._params.outdoor_buckets)
        key = f"outdoor:{band}"
        if profile_id:
            key += f"|profile:{profile_id}"
        return key


def _apply(bucket: HeatRateBucket, rate: float, weight: float) -> HeatRateBucket:
    upd = robust_ema_update(bucket.rate_c_per_h, bucket.effective_n, bucket.dispersion,
                            rate, weight)
    return HeatRateBucket(key=bucket.key, rate_c_per_h=upd.value, effective_n=upd.effective_n,
                          dispersion=upd.dispersion, sample_count=bucket.sample_count + 1)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _bucket_dict(b: HeatRateBucket) -> dict:
    return {"key": b.key, "rate_c_per_h": b.rate_c_per_h, "effective_n": b.effective_n,
            "dispersion": b.dispersion, "sample_count": b.sample_count,
            "last_update_ts": b.last_update_ts}


def _bucket_from(d: Mapping[str, Any]) -> HeatRateBucket:
    return HeatRateBucket(key=d["key"], rate_c_per_h=d["rate_c_per_h"],
                          effective_n=d["effective_n"], dispersion=d["dispersion"],
                          sample_count=d["sample_count"], last_update_ts=d.get("last_update_ts"))


def _sample_dict(x: HeatRateSample) -> dict:
    return {
        "source_episode_id": x.source_episode_id, "learning_zone_id": x.learning_zone_id,
        "rate_c_per_h": x.rate_c_per_h, "duration_s": x.duration_s,
        "temp_delta_c": x.temp_delta_c, "episode_reliability": x.episode_reliability,
        "effective_weight": x.effective_weight, "bucket": x.bucket,
        "outdoor_band": x.outdoor_band, "profile_id": x.profile_id,
        "feature_extractor_version": x.feature_extractor_version,
        "builder_version": x.builder_version, "classifier_version": x.classifier_version,
        "data_quality": x.data_quality, "reason_codes": list(x.reason_codes),
    }


def _sample_from(d: Mapping[str, Any]) -> HeatRateSample:
    return HeatRateSample(
        source_episode_id=d["source_episode_id"], learning_zone_id=d["learning_zone_id"],
        rate_c_per_h=d["rate_c_per_h"], duration_s=d["duration_s"],
        temp_delta_c=d["temp_delta_c"], episode_reliability=d["episode_reliability"],
        effective_weight=d["effective_weight"], bucket=d["bucket"],
        outdoor_band=d.get("outdoor_band"), profile_id=d.get("profile_id"),
        feature_extractor_version=d["feature_extractor_version"],
        builder_version=d["builder_version"], classifier_version=d["classifier_version"],
        data_quality=d["data_quality"], reason_codes=tuple(d.get("reason_codes", [])))


# -- registry definition ------------------------------------------------------

def heat_rate_model_definition():
    """Build the registry ModelDefinition for HeatRate (no auto-registration)."""
    from ..registry import ModelDefinition

    return ModelDefinition(
        model_name=MODEL_NAME, model_factory=lambda: HeatRateModel("__definition__"),
        model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
        consumed_raw_tracks=(), consumed_episode_types=(EpisodeType.HEATING,),
        supported_prediction_types=(PredictionType.HEAT_RATE, PredictionType.PREHEAT_MINUTES),
        required_features=("indoor_temp", "trv_setpoint", "target"),
        optional_features=("outdoor_temp", "wind_speed", "solar_radiation",
                           "valve_opening", "radiator_profile"),
        required_capability_flags=(), regime_dependent=True, advisory=False,
        control_relevant=True, rebuildable=True, can_be_not_available=False,
        min_trv_only=True)
