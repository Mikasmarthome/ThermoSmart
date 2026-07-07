"""HeatLossModel for LE 2.0 (pure Python).

Learns how fast a zone cools with no active heat input, from completed,
undisturbed PassiveCoolingEpisodes. Authoritative quantity: the relative cooling
rate in deg C/h as a POSITIVE magnitude (21.0 -> 20.5 over 1h => 0.5 C/h). With
outdoor temperature a separate normalized loss coefficient (1/h) may also be
learned, but it is never the only quantity and never updates on a small/negative
indoor-outdoor delta. No HA / storage / coordinator dependency; mutates no input;
calls no other model; sends no command. Independent from HeatRateModel.
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
from ..episode_schemas import EpisodeType, PassiveCoolingEpisode
from ..features import FeatureExtractor, FeatureName, FeatureStatus
from .base import clamp, freshness_from_age, is_finite, outlier_z, robust_ema_update

MODEL_VERSION = 1
PARAMETER_VERSION = 1
MODEL_NAME = "heat_loss"


class HeatLossRejection(Enum):
    WRONG_EPISODE_TYPE = "wrong_episode_type"
    WRONG_ZONE = "wrong_zone"
    DISTURBED = "disturbed"
    UNKNOWN_DOMINANT = "unknown_dominant"
    INSUFFICIENT_DURATION = "insufficient_duration"
    INSUFFICIENT_POINTS = "insufficient_points"
    MISSING_INDOOR_TEMP = "missing_indoor_temperature"
    INVALID_TRAJECTORY = "invalid_trajectory"
    NO_COOLING = "no_cooling"
    IMPLAUSIBLE_LOSS_RATE = "implausible_loss_rate"
    LOW_RELIABILITY = "low_reliability"
    AFTERHEAT_CONTAMINATION = "afterheat_contamination"
    ACTIVE_HEATING_CONTAMINATION = "active_heating_contamination"
    WINDOW_CONFOUNDER = "window_confounder"
    SOLAR_CONFOUNDER = "solar_confounder"
    DUPLICATE_EPISODE = "duplicate_episode"
    VERSION_MISMATCH = "version_mismatch"
    SEVERE_OUTLIER = "severe_outlier"
    CONTEXT_EPISODE_MISMATCH = "context_episode_mismatch"


_TO_CONTRACT = {
    HeatLossRejection.WRONG_EPISODE_TYPE: RejectionReason.NOT_ELIGIBLE,
    HeatLossRejection.WRONG_ZONE: RejectionReason.NOT_ELIGIBLE,
    HeatLossRejection.DISTURBED: RejectionReason.DISTURBED_REGIME,
    HeatLossRejection.UNKNOWN_DOMINANT: RejectionReason.UNKNOWN_REGIME,
    HeatLossRejection.INSUFFICIENT_DURATION: RejectionReason.NOT_ELIGIBLE,
    HeatLossRejection.INSUFFICIENT_POINTS: RejectionReason.NOT_ELIGIBLE,
    HeatLossRejection.MISSING_INDOOR_TEMP: RejectionReason.MISSING_REQUIRED_DATA,
    HeatLossRejection.INVALID_TRAJECTORY: RejectionReason.MISSING_REQUIRED_DATA,
    HeatLossRejection.NO_COOLING: RejectionReason.NOT_ELIGIBLE,
    HeatLossRejection.IMPLAUSIBLE_LOSS_RATE: RejectionReason.OUTLIER,
    HeatLossRejection.LOW_RELIABILITY: RejectionReason.LOW_RELIABILITY,
    HeatLossRejection.AFTERHEAT_CONTAMINATION: RejectionReason.NOT_ELIGIBLE,
    HeatLossRejection.ACTIVE_HEATING_CONTAMINATION: RejectionReason.NOT_ELIGIBLE,
    HeatLossRejection.WINDOW_CONFOUNDER: RejectionReason.DISTURBED_REGIME,
    HeatLossRejection.SOLAR_CONFOUNDER: RejectionReason.DISTURBED_REGIME,
    HeatLossRejection.DUPLICATE_EPISODE: RejectionReason.DUPLICATE,
    HeatLossRejection.VERSION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    HeatLossRejection.SEVERE_OUTLIER: RejectionReason.OUTLIER,
    HeatLossRejection.CONTEXT_EPISODE_MISMATCH: RejectionReason.NOT_ELIGIBLE,
}


@dataclass(frozen=True)
class HeatLossParameters:
    min_episode_reliability: float = 0.4
    min_duration_s: float = 300.0
    min_points: int = 3
    min_temp_drop_c: float = 0.2
    min_loss_rate_c_per_h: float = 0.02
    max_loss_rate_c_per_h: float = 6.0
    max_sample_weight: float = 1.0
    bucket_min_samples: int = 5
    bucket_seed_evidence: float = 1.0
    outlier_mad_k_mild: float = 3.0
    outlier_mad_k_severe: float = 6.0
    min_samples_for_outlier: int = 5
    dedup_max_ids: int = 500
    research_sample_cap: int = 50
    outdoor_buckets: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    min_io_delta_for_norm_c: float = 3.0
    generic_prior_loss_c_per_h: float = 0.4
    cooling_min_minutes: float = 0.0
    cooling_max_minutes: float = 1440.0
    cooling_fallback_minutes: float = 240.0
    cold_start_confidence_cap: float = 0.4
    full_confidence_samples: float = 20.0
    parameter_version: int = PARAMETER_VERSION


@dataclass(frozen=True)
class HeatLossUpdateContext:
    """Persistable, reproducible conditioning evidence bound to one episode."""

    source_episode_id: str
    outdoor_temperature_c: Optional[float] = None
    indoor_outdoor_delta_c: Optional[float] = None
    device_profile_key: Optional[str] = None
    wind_bucket: Optional[str] = None
    solar_bucket: Optional[str] = None
    source_feature_version: int = 1
    source_refs: tuple[str, ...] = ()
    quality: DataQuality = DataQuality.OK

    def __post_init__(self) -> None:
        if not self.source_episode_id:
            raise ValueError("source_episode_id must be non-empty")
        for name in ("outdoor_temperature_c", "indoor_outdoor_delta_c"):
            v = getattr(self, name)
            if v is not None and not is_finite(v):
                raise ValueError(f"{name} must be finite (or None)")

    def to_dict(self) -> dict:
        return {
            "source_episode_id": self.source_episode_id,
            "outdoor_temperature_c": self.outdoor_temperature_c,
            "indoor_outdoor_delta_c": self.indoor_outdoor_delta_c,
            "device_profile_key": self.device_profile_key,
            "wind_bucket": self.wind_bucket, "solar_bucket": self.solar_bucket,
            "source_feature_version": self.source_feature_version,
            "source_refs": list(self.source_refs), "quality": self.quality.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HeatLossUpdateContext":
        return cls(
            source_episode_id=d["source_episode_id"],
            outdoor_temperature_c=d.get("outdoor_temperature_c"),
            indoor_outdoor_delta_c=d.get("indoor_outdoor_delta_c"),
            device_profile_key=d.get("device_profile_key"),
            wind_bucket=d.get("wind_bucket"), solar_bucket=d.get("solar_bucket"),
            source_feature_version=d.get("source_feature_version", 1),
            source_refs=tuple(d.get("source_refs", [])),
            quality=DataQuality(d.get("quality", DataQuality.OK.value)))


@dataclass(frozen=True)
class HeatLossRebuildItem:
    episode: Any
    context: Optional[HeatLossUpdateContext] = None


@dataclass(frozen=True)
class HeatLossSample:
    source_episode_id: str
    learning_zone_id: str
    loss_rate_c_per_h: float
    normalized_coefficient_per_h: Optional[float]
    temp_drop_c: float
    duration_s: float
    episode_reliability: float
    effective_weight: float
    bucket: str
    outdoor_band: Optional[int]
    feature_extractor_version: int
    builder_version: int
    classifier_version: int
    data_quality: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class HeatLossBucket:
    key: str
    rate_c_per_h: float
    effective_n: float
    dispersion: float
    sample_count: int
    last_update_ts: Optional[str] = None

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0 and self.effective_n > 0.0


def _empty_bucket(key: str) -> HeatLossBucket:
    return HeatLossBucket(key=key, rate_c_per_h=0.0, effective_n=0.0,
                          dispersion=0.0, sample_count=0)


@dataclass(frozen=True)
class HeatLossState:
    learning_zone_id: str
    general: HeatLossBucket
    normalized: HeatLossBucket  # separate normalized coefficient (1/h) state
    buckets: Mapping[str, HeatLossBucket] = field(default_factory=dict)
    processed_ids: tuple[str, ...] = ()
    recent_samples: tuple[HeatLossSample, ...] = ()
    aggregate_reliability: float = 0.0
    last_update_ts: Optional[str] = None
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    dedup_count: int = 0
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION


@dataclass(frozen=True)
class HeatLossDiagnostics:
    general_loss_c_per_h: float
    normalized_coefficient_per_h: Optional[float]
    bucket_rates: Mapping[str, float]
    sample_counts: Mapping[str, int]
    effective_sample_counts: Mapping[str, float]
    dispersion: Mapping[str, float]
    confidence: float
    relative_only: bool
    last_update_ts: Optional[str]
    rejection_counts: Mapping[str, int]
    outlier_counts: Mapping[str, int]
    dedup_count: int
    missing_optional_evidence: tuple[str, ...]
    model_version: int
    parameter_version: int


@dataclass(frozen=True)
class HeatLossPredictionContext:
    current_temp: Optional[float] = None
    floor_temp: Optional[float] = None  # lower comfort / target bound
    outdoor_temp: Optional[Measurement] = None
    profile_id: Optional[str] = None
    wind_speed: Optional[Measurement] = None
    solar_radiation: Optional[Measurement] = None


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


class HeatLossModel:
    """Pure passive-cooling (heat-loss) model implementing the Model contract."""

    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    parameter_version = PARAMETER_VERSION
    consumed_raw_tracks: tuple[str, ...] = ()
    consumed_episode_types = (EpisodeType.PASSIVE_COOLING.value,)
    required_features = ("indoor_temp", "target")
    optional_features = ("outdoor_temp", "wind_speed", "solar_radiation", "radiator_profile")
    supported_prediction_types = (PredictionType.HEAT_LOSS_RATE, PredictionType.COOLING_MINUTES)
    min_tier = 0

    def __init__(
        self, learning_zone_id: str, params: Optional[HeatLossParameters] = None, *,
        device_prior: Optional[Callable[[Optional[str]], Optional[float]]] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or HeatLossParameters()
        self._device_prior = device_prior
        self._extractor = extractor or FeatureExtractor()
        self._state = self._empty_state()

    def cold_start_prior(self) -> Mapping[str, Any]:
        return {"general_loss_c_per_h": None,
                "generic_prior_loss_c_per_h": self._params.generic_prior_loss_c_per_h}

    def _empty_state(self) -> HeatLossState:
        return HeatLossState(learning_zone_id=self._zone, general=_empty_bucket("general"),
                             normalized=_empty_bucket("normalized"))

    def _extractor_version(self) -> int:
        from ..features import FEATURE_EXTRACTOR_VERSION
        return FEATURE_EXTRACTOR_VERSION

    # -- rate computation (authoritative positive cooling magnitude) ----

    def _compute_loss(self, episode: PassiveCoolingEpisode):
        fs = self._extractor.extract_trajectory_features(episode.trajectory)
        valid = fs.get(FeatureName.TRAJ_VALID_POINTS).value
        delta_r = fs.get(FeatureName.TRAJ_DELTA)
        dur_r = fs.get(FeatureName.TRAJ_DURATION)
        plaus = fs.get(FeatureName.TRAJ_PLAUSIBILITY)
        slope_r = fs.get(FeatureName.TRAJ_LINEAR_SLOPE)
        feat_rel = slope_r.reliability if slope_r.is_present else 0.0
        if valid is None or valid < 1:
            return None, HeatLossRejection.MISSING_INDOOR_TEMP, feat_rel
        if not delta_r.is_present or not dur_r.is_present:
            return None, HeatLossRejection.INVALID_TRAJECTORY, feat_rel
        if int(valid) < self._params.min_points:
            return None, HeatLossRejection.INSUFFICIENT_POINTS, feat_rel
        if plaus.status is FeatureStatus.INVALID:
            return None, HeatLossRejection.IMPLAUSIBLE_LOSS_RATE, feat_rel
        delta = delta_r.value          # negative while cooling
        duration_s = dur_r.value
        if duration_s <= 0 or not is_finite(delta) or not is_finite(duration_s):
            return None, HeatLossRejection.INVALID_TRAJECTORY, feat_rel
        drop = -delta                  # positive cooling magnitude
        if drop < self._params.min_temp_drop_c:
            return None, HeatLossRejection.NO_COOLING, feat_rel
        loss_rate = drop / (duration_s / 3600.0)
        if not is_finite(loss_rate) or loss_rate < self._params.min_loss_rate_c_per_h \
                or loss_rate > self._params.max_loss_rate_c_per_h:
            return None, HeatLossRejection.IMPLAUSIBLE_LOSS_RATE, feat_rel
        return (loss_rate, drop, duration_s, feat_rel), None, feat_rel

    # -- eligibility -----------------------------------------------------

    def update_eligibility(self, episode: Any) -> UpdateEligibilityResult:
        p = self._params
        if not isinstance(episode, PassiveCoolingEpisode):
            return self._reject(HeatLossRejection.WRONG_EPISODE_TYPE)
        if episode.learning_zone_id != self._zone:
            return self._reject(HeatLossRejection.WRONG_ZONE)
        if episode.episode_schema_version != 1:
            return self._reject(HeatLossRejection.VERSION_MISMATCH)
        if episode.regime is not Regime.PASSIVE_COOLING:
            return self._reject(HeatLossRejection.DISTURBED, regime=episode.regime)
        cf = set(episode.confounder_flags)
        if "window_open" in cf:
            return self._reject(HeatLossRejection.WINDOW_CONFOUNDER)
        if "heating_failure" in cf:
            return self._reject(HeatLossRejection.DISTURBED)
        if "reheating" in cf:
            return self._reject(HeatLossRejection.AFTERHEAT_CONTAMINATION)
        if "solar_gain" in cf:
            return self._reject(HeatLossRejection.SOLAR_CONFOUNDER)
        if episode.episode_id in self._state.processed_ids:
            return self._reject(HeatLossRejection.DUPLICATE_EPISODE)
        if episode.reliability < p.min_episode_reliability:
            return self._reject(HeatLossRejection.LOW_RELIABILITY)
        if (episode.end_ts - episode.start_ts).total_seconds() < p.min_duration_s:
            return self._reject(HeatLossRejection.INSUFFICIENT_DURATION)
        computed, reject, feat_rel = self._compute_loss(episode)
        if reject is not None:
            return self._reject(reject)
        weight = clamp(episode.reliability * max(feat_rel, 0.1), 0.0, p.max_sample_weight)
        return UpdateEligibilityResult(
            accepted=True, reliability=episode.reliability, weight=weight,
            outlier_status=OutlierStatus.NONE, regime=Regime.PASSIVE_COOLING,
            source_episode_id=episode.episode_id,
            feature_extractor_version=self._extractor_version())

    def _reject(self, reason: HeatLossRejection, *, regime=None) -> UpdateEligibilityResult:
        return UpdateEligibilityResult(
            accepted=False, reliability=0.0, weight=0.0,
            rejection_reason=_TO_CONTRACT[reason], confounder_flags=(reason.value,),
            regime=regime)

    # -- update ----------------------------------------------------------

    def update(self, episode: Any,
               context: Optional[HeatLossUpdateContext] = None) -> UpdateEligibilityResult:
        if context is not None and context.source_episode_id != episode.episode_id:
            self._bump_rejection(HeatLossRejection.CONTEXT_EPISODE_MISMATCH.value)
            return self._reject(HeatLossRejection.CONTEXT_EPISODE_MISMATCH)
        elig = self.update_eligibility(episode)
        if not elig.accepted:
            self._bump_rejection(elig.confounder_flags[0] if elig.confounder_flags else "unknown")
            if elig.confounder_flags and elig.confounder_flags[0] == \
                    HeatLossRejection.DUPLICATE_EPISODE.value:
                self._state = replace(self._state, dedup_count=self._state.dedup_count + 1)
            return elig

        computed, _, feat_rel = self._compute_loss(episode)
        loss_rate, drop, duration_s, _ = computed
        weight = elig.weight

        general = self._state.general
        outlier_status = OutlierStatus.NONE
        if general.sample_count >= self._params.min_samples_for_outlier:
            z = outlier_z(loss_rate, general.rate_c_per_h, general.dispersion)
            if z > self._params.outlier_mad_k_severe:
                self._bump_outlier("severe")
                return self._reject(HeatLossRejection.SEVERE_OUTLIER)
            if z > self._params.outlier_mad_k_mild:
                outlier_status = OutlierStatus.MILD
                weight *= 0.25
                self._bump_outlier("mild")

        new_general = _apply(general, loss_rate, weight)

        # optional normalized coefficient (1/h), only with a sufficient positive delta
        new_normalized = self._state.normalized
        norm_value: Optional[float] = None
        if context is not None and context.indoor_outdoor_delta_c is not None \
                and is_finite(context.indoor_outdoor_delta_c) \
                and context.indoor_outdoor_delta_c >= self._params.min_io_delta_for_norm_c \
                and context.quality in (DataQuality.OK, DataQuality.STALE):
            norm_value = loss_rate / context.indoor_outdoor_delta_c
            new_normalized = _apply(new_normalized, norm_value, weight)

        # conditioned relative buckets by outdoor band
        new_buckets = dict(self._state.buckets)
        bucket_name = "general"
        outdoor_band: Optional[int] = None
        if context is not None and context.outdoor_temperature_c is not None \
                and is_finite(context.outdoor_temperature_c) \
                and context.quality in (DataQuality.OK, DataQuality.STALE):
            ov = context.outdoor_temperature_c
            key = self._bucket_key(ov, context.device_profile_key)
            outdoor_band = _bucket_index(ov, self._params.outdoor_buckets)
            bkt = new_buckets.get(key)
            if bkt is None:
                bkt = HeatLossBucket(key=key, rate_c_per_h=new_general.rate_c_per_h,
                                     effective_n=self._params.bucket_seed_evidence,
                                     dispersion=new_general.dispersion, sample_count=0)
            new_buckets[key] = _apply(bkt, loss_rate, weight)
            bucket_name = key

        sample = HeatLossSample(
            source_episode_id=episode.episode_id, learning_zone_id=self._zone,
            loss_rate_c_per_h=round(loss_rate, 5),
            normalized_coefficient_per_h=round(norm_value, 6) if norm_value is not None else None,
            temp_drop_c=drop, duration_s=duration_s,
            episode_reliability=episode.reliability, effective_weight=weight,
            bucket=bucket_name, outdoor_band=outdoor_band,
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
            self._state, general=new_general, normalized=new_normalized, buckets=new_buckets,
            processed_ids=processed, recent_samples=recent, aggregate_reliability=agg,
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
        return self.predict_heat_loss(context)

    def predict_heat_loss(self, context: HeatLossPredictionContext) -> Prediction:
        p = self._params
        missing: list[str] = []
        outdoor = _usable(context.outdoor_temp)
        relative_only = outdoor is None
        if outdoor is None:
            missing.append("outdoor_temp")

        bucket = None
        if outdoor is not None:
            key = self._bucket_key(outdoor, context.profile_id)
            cand = self._state.buckets.get(key)
            if cand is not None and cand.sample_count >= p.bucket_min_samples:
                bucket = cand

        if bucket is not None:
            return self._loss_prediction(bucket.rate_c_per_h, bucket.sample_count, learned=1.0,
                                         fallback=False, bucket=bucket.key,
                                         reasons=("conditioned_bucket",), missing=missing,
                                         relative_only=False)
        if self._state.general.has_evidence:
            g = self._state.general
            reasons = ("general_rate",) + (("relative_only",) if relative_only else ())
            return self._loss_prediction(g.rate_c_per_h, g.sample_count, learned=1.0,
                                         fallback=False, bucket="general", reasons=reasons,
                                         missing=missing, relative_only=relative_only)
        prior = self._device_prior(context.profile_id) if self._device_prior else None
        if prior is not None and is_finite(prior) and prior > 0:
            return self._loss_prediction(prior, 0, learned=0.0, fallback=True,
                                         bucket="device_prior", reasons=("device_prior",),
                                         missing=missing, relative_only=relative_only,
                                         cap=p.cold_start_confidence_cap)
        return self._loss_prediction(p.generic_prior_loss_c_per_h, 0, learned=0.0,
                                     fallback=True, bucket="generic_prior",
                                     reasons=("generic_prior",), missing=missing,
                                     relative_only=relative_only,
                                     cap=p.cold_start_confidence_cap)

    def _loss_prediction(self, rate, sample_count, *, learned, fallback, bucket, reasons,
                         missing, relative_only, cap=None) -> Prediction:
        rate = float(rate)
        if not is_finite(rate):
            rate = self._params.generic_prior_loss_c_per_h
            fallback = True
        conf = self._confidence_value(sample_count, fallback)
        if relative_only:
            conf = min(conf, 0.6)  # relative-only cannot claim conditioned authority
        if cap is not None:
            conf = min(conf, cap)
        return Prediction(
            prediction_type=PredictionType.HEAT_LOSS_RATE,
            values={"heat_loss_rate": rate}, units={"heat_loss_rate": "C/h"},
            confidence=conf,
            reliability=self._state.aggregate_reliability if not fallback else 0.3,
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
            prior_contribution=1.0 - learned, learned_contribution=learned,
            fallback_used=fallback, evidence_count=sample_count, bucket=bucket,
            confidence_cap=cap, cap_reasons=("cold_start",) if cap is not None else (),
            missing_evidence=tuple(missing), reason_codes=tuple(reasons))

    def predict_cooling_duration(self, context: HeatLossPredictionContext) -> Prediction:
        p = self._params
        loss = self.predict_heat_loss(context)
        rate = loss.control_value("heat_loss_rate")
        current, floor = context.current_temp, context.floor_temp
        reasons = list(loss.reason_codes)
        missing = list(loss.missing_evidence)

        guard_fallback = False
        if current is None or floor is None or not is_finite(current) or not is_finite(floor):
            minutes = p.cooling_fallback_minutes
            reasons.append("missing_temps_fallback")
            guard_fallback = True
        elif current - floor <= 0:
            minutes = 0.0
            reasons.append("floor_reached")
        elif rate <= 0 or not is_finite(rate):
            minutes = p.cooling_fallback_minutes
            reasons.append("invalid_rate_fallback")
            guard_fallback = True
        else:
            minutes = (current - floor) / rate * 60.0
            if not is_finite(minutes):
                minutes = p.cooling_fallback_minutes
                guard_fallback = True
                reasons.append("non_finite_guard")
            minutes = clamp(minutes, p.cooling_min_minutes, p.cooling_max_minutes)

        fallback = guard_fallback or loss.fallback_used
        prior_c, learned_c = (1.0, 0.0) if fallback else (loss.prior_contribution,
                                                          loss.learned_contribution)
        return Prediction(
            prediction_type=PredictionType.COOLING_MINUTES,
            values={"cooling_minutes": float(minutes)}, units={"cooling_minutes": "min"},
            confidence=loss.confidence, reliability=loss.reliability if not fallback else 0.3,
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
            prior_contribution=prior_c, learned_contribution=learned_c, fallback_used=fallback,
            evidence_count=loss.evidence_count, bucket=loss.bucket,
            confidence_cap=loss.confidence_cap, cap_reasons=loss.cap_reasons,
            missing_evidence=tuple(missing), reason_codes=tuple(reasons))

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

    def confidence(self, *, now: Optional[str] = None) -> ConfidenceContribution:
        g = self._state.general
        reasons: list[str] = []
        if not g.has_evidence:
            reasons.append("cold_start")
        if not self._state.buckets:
            reasons.append("relative_only")
        value = self._confidence_value(g.sample_count, not g.has_evidence)
        freshness, status = freshness_from_age(now, self._state.last_update_ts)
        return ConfidenceContribution(value=value, evidence_count=g.sample_count,
                                      reasons=tuple(reasons), freshness=freshness, status=status)

    # -- diagnostics / export -------------------------------------------

    def diagnostics(self) -> HeatLossDiagnostics:
        s = self._state
        return HeatLossDiagnostics(
            general_loss_c_per_h=s.general.rate_c_per_h,
            normalized_coefficient_per_h=s.normalized.rate_c_per_h
            if s.normalized.has_evidence else None,
            bucket_rates={k: b.rate_c_per_h for k, b in s.buckets.items()},
            sample_counts={"general": s.general.sample_count,
                           **{k: b.sample_count for k, b in s.buckets.items()}},
            effective_sample_counts={"general": s.general.effective_n,
                                     **{k: b.effective_n for k, b in s.buckets.items()}},
            dispersion={"general": s.general.dispersion,
                        **{k: b.dispersion for k, b in s.buckets.items()}},
            confidence=self.confidence().value, relative_only=not s.buckets,
            last_update_ts=s.last_update_ts, rejection_counts=dict(s.rejection_counts),
            outlier_counts=dict(s.outlier_counts), dedup_count=s.dedup_count,
            missing_optional_evidence=("outdoor_temp",) if not s.buckets else (),
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION)

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        s = self._state
        if scope is ExportScope.SUPPORT:
            return {
                "general_loss_c_per_h": s.general.rate_c_per_h,
                "normalized_coefficient_per_h": s.normalized.rate_c_per_h
                if s.normalized.has_evidence else None,
                "buckets": {k: {"rate_c_per_h": b.rate_c_per_h, "samples": b.sample_count}
                            for k, b in s.buckets.items()},
                "confidence": self.confidence().value, "relative_only": not s.buckets,
                "sample_count": s.general.sample_count, "model_version": MODEL_VERSION,
                "parameter_version": PARAMETER_VERSION,
                "rejection_summary": dict(s.rejection_counts),
                "fallback": not s.general.has_evidence,
            }
        if scope is ExportScope.RESEARCH:
            return {
                "samples": [
                    {"loss_rate_c_per_h": x.loss_rate_c_per_h, "duration_s": x.duration_s,
                     "temp_drop_c": x.temp_drop_c, "bucket": x.bucket,
                     "normalized_coefficient_per_h": x.normalized_coefficient_per_h,
                     "reliability": x.episode_reliability}
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
        items: list[tuple[Any, Optional[HeatLossUpdateContext]]] = []
        for it in raw_items:
            if isinstance(it, HeatLossRebuildItem):
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
                if code == HeatLossRejection.DUPLICATE_EPISODE.value:
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
            "learning_zone_id": s.learning_zone_id,
            "general": _bucket_dict(s.general), "normalized": _bucket_dict(s.normalized),
            "buckets": {k: _bucket_dict(b) for k, b in s.buckets.items()},
            "processed_ids": list(s.processed_ids),
            "aggregate_reliability": s.aggregate_reliability,
            "last_update_ts": s.last_update_ts, "rejection_counts": dict(s.rejection_counts),
            "outlier_counts": dict(s.outlier_counts), "dedup_count": s.dedup_count,
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
        for bname in ("general", "normalized"):
            bucket = state.get(bname, {})
            for k in ("rate_c_per_h", "effective_n", "dispersion"):
                v = bucket.get(k)
                if v is not None and not is_finite(v):
                    errors.append(f"non-finite {bname}.{k}")
        return errors

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        errors = self.validate_state(state)
        if errors:
            raise ValueError(f"invalid heat-loss state: {errors}")
        buckets = {k: _bucket_from(b) for k, b in state.get("buckets", {}).items()}
        recent = tuple(_sample_from(x) for x in state.get("recent_samples", []))
        self._state = HeatLossState(
            learning_zone_id=state["learning_zone_id"], general=_bucket_from(state["general"]),
            normalized=_bucket_from(state["normalized"]), buckets=buckets,
            processed_ids=tuple(state.get("processed_ids", [])), recent_samples=recent,
            aggregate_reliability=state.get("aggregate_reliability", 0.0),
            last_update_ts=state.get("last_update_ts"),
            rejection_counts=dict(state.get("rejection_counts", {})),
            outlier_counts=dict(state.get("outlier_counts", {})),
            dedup_count=state.get("dedup_count", 0))

    def migrate_state(self, old_model_version: int, old_parameter_version: int,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        if old_model_version == MODEL_VERSION:
            return state
        raise ValueError(
            f"no migration path from model_version {old_model_version} to {MODEL_VERSION}")

    def _bucket_key(self, outdoor: float, profile_id: Optional[str]) -> str:
        band = _bucket_index(outdoor, self._params.outdoor_buckets)
        key = f"outdoor:{band}"
        if profile_id:
            key += f"|profile:{profile_id}"
        return key


def _apply(bucket: HeatLossBucket, rate: float, weight: float) -> HeatLossBucket:
    upd = robust_ema_update(bucket.rate_c_per_h, bucket.effective_n, bucket.dispersion,
                            rate, weight)
    return HeatLossBucket(key=bucket.key, rate_c_per_h=upd.value, effective_n=upd.effective_n,
                          dispersion=upd.dispersion, sample_count=bucket.sample_count + 1)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _bucket_dict(b: HeatLossBucket) -> dict:
    return {"key": b.key, "rate_c_per_h": b.rate_c_per_h, "effective_n": b.effective_n,
            "dispersion": b.dispersion, "sample_count": b.sample_count,
            "last_update_ts": b.last_update_ts}


def _bucket_from(d: Mapping[str, Any]) -> HeatLossBucket:
    return HeatLossBucket(key=d["key"], rate_c_per_h=d["rate_c_per_h"],
                          effective_n=d["effective_n"], dispersion=d["dispersion"],
                          sample_count=d["sample_count"], last_update_ts=d.get("last_update_ts"))


def _sample_dict(x: HeatLossSample) -> dict:
    return {
        "source_episode_id": x.source_episode_id, "learning_zone_id": x.learning_zone_id,
        "loss_rate_c_per_h": x.loss_rate_c_per_h,
        "normalized_coefficient_per_h": x.normalized_coefficient_per_h,
        "temp_drop_c": x.temp_drop_c, "duration_s": x.duration_s,
        "episode_reliability": x.episode_reliability, "effective_weight": x.effective_weight,
        "bucket": x.bucket, "outdoor_band": x.outdoor_band,
        "feature_extractor_version": x.feature_extractor_version,
        "builder_version": x.builder_version, "classifier_version": x.classifier_version,
        "data_quality": x.data_quality, "reason_codes": list(x.reason_codes),
    }


def _sample_from(d: Mapping[str, Any]) -> HeatLossSample:
    return HeatLossSample(
        source_episode_id=d["source_episode_id"], learning_zone_id=d["learning_zone_id"],
        loss_rate_c_per_h=d["loss_rate_c_per_h"],
        normalized_coefficient_per_h=d.get("normalized_coefficient_per_h"),
        temp_drop_c=d["temp_drop_c"], duration_s=d["duration_s"],
        episode_reliability=d["episode_reliability"], effective_weight=d["effective_weight"],
        bucket=d["bucket"], outdoor_band=d.get("outdoor_band"),
        feature_extractor_version=d["feature_extractor_version"],
        builder_version=d["builder_version"], classifier_version=d["classifier_version"],
        data_quality=d["data_quality"], reason_codes=tuple(d.get("reason_codes", [])))


def heat_loss_model_definition():
    from ..registry import ModelDefinition

    return ModelDefinition(
        model_name=MODEL_NAME, model_factory=lambda: HeatLossModel("__definition__"),
        model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
        consumed_raw_tracks=(), consumed_episode_types=(EpisodeType.PASSIVE_COOLING,),
        supported_prediction_types=(PredictionType.HEAT_LOSS_RATE, PredictionType.COOLING_MINUTES),
        required_features=("indoor_temp", "target"),
        optional_features=("outdoor_temp", "wind_speed", "solar_radiation", "radiator_profile"),
        required_capability_flags=(), regime_dependent=True, advisory=False,
        control_relevant=True, rebuildable=True, can_be_not_available=False,
        min_trv_only=True)
