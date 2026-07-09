"""ForecastModel for Learning (pure Python).

Learns how trustworthy past weather/temperature forecasts were for a zone — it
does NOT forecast. It matches a ``ForecastDecisionEvent`` (the forecast that was
acted on) to a later ``ForecastEvaluationEvent`` (the reality that arrived) via
``decision_id`` and learns signed bias, robust absolute error and a versioned
trust per forecast horizon (and optional source).

Forecast is strictly optional: with no forecast context the model stays empty and
returns a neutral/unavailable trust; it never blocks the local thermal models and
its confidence contribution only matters on forecast-dependent paths. No HA /
storage / coordinator dependency; imports no other model. No clock — all
timestamps are supplied as inputs.
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
from ..episode_schemas import EpisodeType
from ..raw_schemas import ForecastDecisionEvent, ForecastEvaluationEvent, RawTrackName
from .base import clamp, freshness_from_age, is_finite, outlier_z, robust_ema_update

MODEL_VERSION = 1
PARAMETER_VERSION = 1
MODEL_NAME = "forecast"


class ForecastRejection(Enum):
    MISSING_DECISION = "missing_decision"
    DECISION_MISMATCH = "decision_mismatch"
    WRONG_ZONE = "wrong_zone"
    DUPLICATE_EVALUATION = "duplicate_evaluation"
    INVALID_HORIZON = "invalid_horizon"
    INSUFFICIENT_REALITY_DATA = "insufficient_reality_data"
    MISSING_FORECAST_VALUE = "missing_forecast_value"
    MISSING_ACTUAL_VALUE = "missing_actual_value"
    NON_FINITE_VALUE = "non_finite_value"
    DISTURBED_REALITY = "disturbed_reality"
    STALE_DECISION = "stale_decision"
    UNSUPPORTED_FORECAST_FIELD = "unsupported_forecast_field"
    VERSION_MISMATCH = "version_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    WRONG_EVENT_TYPE = "wrong_event_type"
    SEVERE_OUTLIER = "severe_outlier"


_TO_CONTRACT = {
    ForecastRejection.MISSING_DECISION: RejectionReason.MISSING_REQUIRED_DATA,
    ForecastRejection.DECISION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    ForecastRejection.WRONG_ZONE: RejectionReason.NOT_ELIGIBLE,
    ForecastRejection.DUPLICATE_EVALUATION: RejectionReason.DUPLICATE,
    ForecastRejection.INVALID_HORIZON: RejectionReason.NOT_ELIGIBLE,
    ForecastRejection.INSUFFICIENT_REALITY_DATA: RejectionReason.LOW_RELIABILITY,
    ForecastRejection.MISSING_FORECAST_VALUE: RejectionReason.MISSING_REQUIRED_DATA,
    ForecastRejection.MISSING_ACTUAL_VALUE: RejectionReason.MISSING_REQUIRED_DATA,
    ForecastRejection.NON_FINITE_VALUE: RejectionReason.OUTLIER,
    ForecastRejection.DISTURBED_REALITY: RejectionReason.DISTURBED_REGIME,
    ForecastRejection.STALE_DECISION: RejectionReason.NOT_ELIGIBLE,
    ForecastRejection.UNSUPPORTED_FORECAST_FIELD: RejectionReason.NOT_ELIGIBLE,
    ForecastRejection.VERSION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    ForecastRejection.CONTEXT_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    ForecastRejection.WRONG_EVENT_TYPE: RejectionReason.NOT_ELIGIBLE,
    ForecastRejection.SEVERE_OUTLIER: RejectionReason.OUTLIER,
}


@dataclass(frozen=True)
class ForecastParameters:
    error_scale_c: float = 3.0            # abs error at which trust ~ 0 (documented unit: °C)
    max_abs_error_c: float = 30.0         # implausible -> guarded, not learned as provider error
    bias_trust_scale_c: float = 3.0       # |bias| penalty scale
    min_reality_quality: float = 0.3
    min_horizon_s: float = 0.0
    max_horizon_s: float = 7 * 24 * 3600.0
    matching_window_s: float = 6 * 3600.0  # eval must fall within this of the forecast target
    horizon_bucket_edges_s: tuple[float, ...] = (3600.0, 6 * 3600.0, 24 * 3600.0)
    max_open_decisions: int = 200
    max_sample_weight: float = 1.0
    bucket_min_samples: int = 5
    bucket_seed_evidence: float = 1.0
    outlier_mad_k_mild: float = 3.0
    outlier_mad_k_severe: float = 6.0
    min_samples_for_outlier: int = 5
    dedup_max_ids: int = 500
    research_sample_cap: int = 50
    full_confidence_samples: float = 20.0
    cold_start_confidence_cap: float = 0.35
    min_samples_for_correction: int = 8
    bias_correction_max_c: float = 2.0
    neutral_prior_trust: float = 0.5
    parameter_version: int = PARAMETER_VERSION


# -- robust dimension ---------------------------------------------------------

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
class ForecastBucket:
    key: str
    bias: _Dim = field(default_factory=_Dim)
    abs_error: _Dim = field(default_factory=_Dim)
    sample_count: int = 0

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0


def _bucket_dict(b: ForecastBucket) -> dict:
    return {"key": b.key, "bias": _dim_dict(b.bias), "abs_error": _dim_dict(b.abs_error),
            "sample_count": b.sample_count}


def _bucket_from(d: Mapping[str, Any]) -> ForecastBucket:
    return ForecastBucket(key=d["key"], bias=_dim_from(d["bias"]),
                          abs_error=_dim_from(d["abs_error"]), sample_count=d["sample_count"])


@dataclass(frozen=True)
class _OpenDecision:
    decision_id: str
    learning_zone_id: str
    created_ts: str
    forecast_high: float
    source_key: Optional[str] = None


@dataclass(frozen=True)
class ForecastUpdateContext:
    source_decision_id: str
    learning_zone_id: Optional[str] = None
    forecast_source_key: Optional[str] = None
    forecast_created_ts: Optional[str] = None
    forecast_target_ts: Optional[str] = None
    horizon_s: Optional[float] = None
    forecast_temperature_c: Optional[float] = None
    actual_temperature_c: Optional[float] = None
    reality_quality: float = 1.0
    weather_class: Optional[str] = None
    source_refs: tuple[str, ...] = ()
    source_feature_version: int = 1

    def __post_init__(self) -> None:
        if not self.source_decision_id:
            raise ValueError("source_decision_id must be non-empty")
        for name in ("horizon_s", "forecast_temperature_c", "actual_temperature_c",
                     "reality_quality"):
            v = getattr(self, name)
            if v is not None and not is_finite(v):
                raise ValueError(f"{name} must be finite (or None)")

    def to_dict(self) -> dict:
        return {
            "source_decision_id": self.source_decision_id,
            "learning_zone_id": self.learning_zone_id,
            "forecast_source_key": self.forecast_source_key,
            "forecast_created_ts": self.forecast_created_ts,
            "forecast_target_ts": self.forecast_target_ts, "horizon_s": self.horizon_s,
            "forecast_temperature_c": self.forecast_temperature_c,
            "actual_temperature_c": self.actual_temperature_c,
            "reality_quality": self.reality_quality, "weather_class": self.weather_class,
            "source_refs": list(self.source_refs),
            "source_feature_version": self.source_feature_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ForecastUpdateContext":
        return cls(
            source_decision_id=d["source_decision_id"],
            learning_zone_id=d.get("learning_zone_id"),
            forecast_source_key=d.get("forecast_source_key"),
            forecast_created_ts=d.get("forecast_created_ts"),
            forecast_target_ts=d.get("forecast_target_ts"), horizon_s=d.get("horizon_s"),
            forecast_temperature_c=d.get("forecast_temperature_c"),
            actual_temperature_c=d.get("actual_temperature_c"),
            reality_quality=d.get("reality_quality", 1.0), weather_class=d.get("weather_class"),
            source_refs=tuple(d.get("source_refs", [])),
            source_feature_version=d.get("source_feature_version", 1))


@dataclass(frozen=True)
class ForecastRebuildItem:
    decision: Optional[ForecastDecisionEvent] = None
    evaluation: Optional[ForecastEvaluationEvent] = None
    context: Optional[ForecastUpdateContext] = None


@dataclass(frozen=True)
class ForecastSample:
    source_decision_id: str
    evaluation_id: str
    learning_zone_id: str
    temperature_error_c: float
    absolute_error_c: float
    horizon_s: float
    horizon_bucket: str
    source_key: Optional[str]
    reality_quality: float
    effective_weight: float
    model_version: int


@dataclass(frozen=True)
class ForecastState:
    learning_zone_id: str
    general: ForecastBucket
    horizon_buckets: Mapping[str, ForecastBucket] = field(default_factory=dict)
    source_buckets: Mapping[str, ForecastBucket] = field(default_factory=dict)
    open_decisions: Mapping[str, _OpenDecision] = field(default_factory=dict)
    processed_eval_ids: tuple[str, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()
    recent_samples: tuple[ForecastSample, ...] = ()
    last_update_ts: Optional[str] = None
    reality_quality_sum: float = 0.0
    expired_decisions: int = 0
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION


@dataclass(frozen=True)
class ForecastTrustResult:
    trust: float
    reliability: float
    horizon_bucket: str
    source_bucket: Optional[str]
    bias_c: float
    mean_absolute_error_c: float
    effective_evidence: float
    prior_contribution: float
    learned_contribution: float
    missing_evidence: tuple[str, ...]
    reason_codes: tuple[str, ...]
    fallback_used: bool
    available: bool
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION

    def __post_init__(self) -> None:
        for name in ("trust", "reliability"):
            v = getattr(self, name)
            if not is_finite(v) or not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be finite within [0,1]")


@dataclass(frozen=True)
class ForecastDiagnostics:
    general_trust: float
    general_bias_c: float
    general_abs_error_c: float
    dispersion: float
    horizon_buckets: Mapping[str, float]
    source_buckets: Mapping[str, float]
    sample_counts: Mapping[str, int]
    mean_reality_quality: Optional[float]
    open_decisions: int
    expired_decisions: int
    rejection_counts: Mapping[str, int]
    outlier_counts: Mapping[str, int]
    confidence: float
    last_update_ts: Optional[str]
    model_version: int
    parameter_version: int


@dataclass(frozen=True)
class ForecastPredictionContext:
    horizon_s: Optional[float] = None
    source_key: Optional[str] = None
    raw_forecast_temperature_c: Optional[float] = None
    forecast_available: bool = True


def _horizon_bucket(horizon_s: float, edges: Sequence[float]) -> str:
    for i, edge in enumerate(edges):
        if horizon_s < edge:
            return f"h{i}"
    return f"h{len(edges)}"


# -- model --------------------------------------------------------------------

class ForecastModel:
    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    parameter_version = PARAMETER_VERSION
    consumed_raw_tracks = (RawTrackName.FORECAST_DECISION.value,
                           RawTrackName.FORECAST_EVALUATION.value)
    consumed_episode_types: tuple[str, ...] = ()
    required_features = ("forecast",)
    optional_features = ("outdoor_temp", "wind_speed", "solar_radiation")
    supported_prediction_types = (
        PredictionType.FORECAST_TRUST, PredictionType.FORECAST_BIAS,
        PredictionType.CORRECTED_FORECAST)
    min_tier = 0

    def __init__(self, learning_zone_id: str, params: Optional[ForecastParameters] = None) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or ForecastParameters()
        self._state = self._empty_state()

    def _empty_state(self) -> ForecastState:
        return ForecastState(learning_zone_id=self._zone, general=ForecastBucket(key="general"))

    def cold_start_prior(self) -> Mapping[str, Any]:
        return {"general_trust": self._params.neutral_prior_trust}

    # -- decision lifecycle ---------------------------------------------

    def register_decision(self, decision: ForecastDecisionEvent) -> None:
        if not isinstance(decision, ForecastDecisionEvent):
            raise ValueError("register_decision requires a ForecastDecisionEvent")
        if decision.learning_zone_id != self._zone:
            return  # foreign zone, ignore silently (never cross-bind)
        od = _OpenDecision(
            decision_id=decision.decision_id, learning_zone_id=decision.learning_zone_id,
            created_ts=_iso(decision.ts), forecast_high=decision.forecast_high,
            source_key=None)
        opened = dict(self._state.open_decisions)
        opened[decision.decision_id] = od
        expired = self._state.expired_decisions
        if len(opened) > self._params.max_open_decisions:
            # deterministically drop the oldest open decisions
            ordered = sorted(opened.values(), key=lambda d: (d.created_ts, d.decision_id))
            for victim in ordered[:len(opened) - self._params.max_open_decisions]:
                del opened[victim.decision_id]
                expired += 1
        self._state = replace(self._state, open_decisions=opened, expired_decisions=expired)

    # -- resolution / computation ---------------------------------------

    def _resolve(self, evaluation: ForecastEvaluationEvent,
                 context: Optional[ForecastUpdateContext]):
        """Return (forecast_temp, created_ts, source_key, reality_quality, actual_temp) or reject."""
        forecast_temp = None
        created_ts = None
        source_key = None
        reality_quality = 1.0
        actual_temp = evaluation.observed_temp_at_eval

        od = self._state.open_decisions.get(evaluation.decision_id)
        if od is not None:
            forecast_temp = od.forecast_high
            created_ts = od.created_ts
            source_key = od.source_key

        if context is not None:
            if context.source_decision_id != evaluation.decision_id or (
                    context.learning_zone_id is not None
                    and context.learning_zone_id != self._zone):
                return None, ForecastRejection.CONTEXT_MISMATCH
            if context.forecast_temperature_c is not None:
                forecast_temp = context.forecast_temperature_c
            if context.forecast_created_ts is not None:
                created_ts = context.forecast_created_ts
            if context.forecast_source_key is not None:
                source_key = context.forecast_source_key
            if context.actual_temperature_c is not None:
                actual_temp = context.actual_temperature_c
            reality_quality = context.reality_quality

        if forecast_temp is None:
            return None, ForecastRejection.MISSING_DECISION
        if actual_temp is None:
            return None, ForecastRejection.MISSING_ACTUAL_VALUE
        if not is_finite(forecast_temp):
            return None, ForecastRejection.MISSING_FORECAST_VALUE
        if not is_finite(actual_temp):
            return None, ForecastRejection.NON_FINITE_VALUE

        # horizon from explicit context or created->eval timestamps
        if context is not None and context.horizon_s is not None:
            horizon_s = context.horizon_s
        elif created_ts is not None:
            horizon_s = (evaluation.ts - _parse(created_ts)).total_seconds()
        else:
            return None, ForecastRejection.INVALID_HORIZON
        if not is_finite(horizon_s) or horizon_s < self._params.min_horizon_s \
                or horizon_s > self._params.max_horizon_s:
            return None, ForecastRejection.INVALID_HORIZON

        return (forecast_temp, actual_temp, horizon_s, source_key,
                clamp(reality_quality, 0.0, 1.0)), None

    # -- eligibility ----------------------------------------------------

    def update_eligibility(self, evaluation: Any) -> UpdateEligibilityResult:
        return self._eligibility(evaluation, None)

    def _eligibility(self, evaluation, context) -> UpdateEligibilityResult:
        p = self._params
        if not isinstance(evaluation, ForecastEvaluationEvent):
            return self._reject(ForecastRejection.WRONG_EVENT_TYPE)
        if evaluation.learning_zone_id != self._zone:
            return self._reject(ForecastRejection.WRONG_ZONE)
        if evaluation.evaluation_id in self._state.processed_eval_ids:
            return self._reject(ForecastRejection.DUPLICATE_EVALUATION)
        resolved, reject = self._resolve(evaluation, context)
        if reject is not None:
            return self._reject(reject)
        forecast_temp, actual_temp, horizon_s, source_key, reality_quality = resolved
        if reality_quality < p.min_reality_quality:
            return self._reject(ForecastRejection.INSUFFICIENT_REALITY_DATA)
        abs_error = abs(actual_temp - forecast_temp)
        if abs_error > p.max_abs_error_c:
            # implausible reality (likely disturbed), do not learn as provider error
            return self._reject(ForecastRejection.DISTURBED_REALITY)
        weight = clamp(reality_quality, 0.0, p.max_sample_weight)
        return UpdateEligibilityResult(
            accepted=True, reliability=reality_quality, weight=weight,
            outlier_status=OutlierStatus.NONE, regime=Regime.UNKNOWN,
            source_episode_id=evaluation.evaluation_id)

    def _reject(self, reason: ForecastRejection) -> UpdateEligibilityResult:
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

    def update(self, evaluation: Any,
               context: Optional[ForecastUpdateContext] = None) -> UpdateEligibilityResult:
        elig = self._eligibility(evaluation, context)
        if not elig.accepted:
            code = elig.confounder_flags[0] if elig.confounder_flags else "unknown"
            self._bump_rejection(code)
            return elig

        resolved, _ = self._resolve(evaluation, context)
        forecast_temp, actual_temp, horizon_s, source_key, reality_quality = resolved
        temp_error = actual_temp - forecast_temp
        abs_error = abs(temp_error)
        weight = elig.weight

        # outlier vs the zone's own forecast-error history
        g = self._state.general
        if g.abs_error.sample_count >= self._params.min_samples_for_outlier \
                and g.abs_error.has_evidence:
            z = outlier_z(abs_error, g.abs_error.value, g.abs_error.dispersion)
            if z > self._params.outlier_mad_k_severe:
                self._bump_outlier("severe")
                self._bump_rejection(ForecastRejection.SEVERE_OUTLIER.value)
                return self._reject(ForecastRejection.SEVERE_OUTLIER)
            if z > self._params.outlier_mad_k_mild:
                weight *= 0.25
                self._bump_outlier("mild")

        new_general = self._apply_bucket(g, temp_error, abs_error, weight)
        hkey = _horizon_bucket(horizon_s, self._params.horizon_bucket_edges_s)
        horizon_buckets = dict(self._state.horizon_buckets)
        hb = horizon_buckets.get(hkey) or self._seed_bucket(hkey, new_general)
        horizon_buckets[hkey] = self._apply_bucket(hb, temp_error, abs_error, weight)

        source_buckets = dict(self._state.source_buckets)
        if source_key:
            skey = f"src:{source_key}"
            sb = source_buckets.get(skey) or self._seed_bucket(skey, new_general)
            source_buckets[skey] = self._apply_bucket(sb, temp_error, abs_error, weight)

        sample = ForecastSample(
            source_decision_id=evaluation.decision_id, evaluation_id=evaluation.evaluation_id,
            learning_zone_id=self._zone, temperature_error_c=temp_error,
            absolute_error_c=abs_error, horizon_s=horizon_s, horizon_bucket=hkey,
            source_key=source_key, reality_quality=reality_quality, effective_weight=weight,
            model_version=MODEL_VERSION)

        opened = dict(self._state.open_decisions)
        opened.pop(evaluation.decision_id, None)
        processed_eval = (self._state.processed_eval_ids
                          + (evaluation.evaluation_id,))[-self._params.dedup_max_ids:]
        processed_dec = (self._state.processed_decision_ids
                         + (evaluation.decision_id,))[-self._params.dedup_max_ids:]
        recent = (self._state.recent_samples + (sample,))[-self._params.research_sample_cap:]
        self._state = replace(
            self._state, general=new_general, horizon_buckets=horizon_buckets,
            source_buckets=source_buckets, open_decisions=opened,
            processed_eval_ids=processed_eval, processed_decision_ids=processed_dec,
            recent_samples=recent, last_update_ts=_iso(evaluation.ts),
            reality_quality_sum=self._state.reality_quality_sum + reality_quality)
        return elig

    def _apply_bucket(self, b: ForecastBucket, temp_error: float, abs_error: float,
                      weight: float) -> ForecastBucket:
        return ForecastBucket(
            key=b.key, bias=_apply_dim(b.bias, temp_error, weight),
            abs_error=_apply_dim(b.abs_error, abs_error, weight),
            sample_count=b.sample_count + 1)

    def _seed_bucket(self, key: str, g: ForecastBucket) -> ForecastBucket:
        seed = lambda d: _Dim(value=d.value, effective_n=self._params.bucket_seed_evidence,
                              dispersion=d.dispersion, sample_count=0)
        return ForecastBucket(key=key, bias=seed(g.bias), abs_error=seed(g.abs_error),
                              sample_count=0)

    # -- trust / predictions --------------------------------------------

    def _trust_from(self, bucket: ForecastBucket) -> float:
        p = self._params
        abs_err = bucket.abs_error.value
        bias = abs(bucket.bias.value)
        base = clamp(1.0 - abs_err / p.error_scale_c, 0.0, 1.0)
        bias_factor = clamp(1.0 - bias / p.bias_trust_scale_c, 0.0, 1.0)
        evidence = clamp(bucket.abs_error.effective_n / p.full_confidence_samples, 0.0, 1.0)
        # few samples cannot yield high trust even if error looks small
        trust = base * bias_factor * (0.4 + 0.6 * evidence)
        return clamp(trust, 0.0, 1.0)

    def predict_trust(self, context: ForecastPredictionContext) -> ForecastTrustResult:
        p = self._params
        if context is not None and not context.forecast_available:
            return self._unavailable_trust("forecast_unavailable")
        bucket, bkey, source_bucket, fallback, reasons = self._select_bucket(context)
        if bucket is None:
            return self._unavailable_trust("no_forecast_evidence")
        trust = self._trust_from(bucket)
        eff = bucket.abs_error.effective_n
        if fallback:
            trust = min(trust, p.cold_start_confidence_cap)
        learned = 0.0 if fallback else 1.0
        return ForecastTrustResult(
            trust=trust, reliability=clamp(eff / p.full_confidence_samples, 0.0, 1.0),
            horizon_bucket=bkey, source_bucket=source_bucket, bias_c=bucket.bias.value,
            mean_absolute_error_c=bucket.abs_error.value, effective_evidence=eff,
            prior_contribution=1.0 - learned, learned_contribution=learned,
            missing_evidence=() if not fallback else ("bucket_support",),
            reason_codes=tuple(reasons), fallback_used=fallback, available=True)

    def _unavailable_trust(self, reason: str) -> ForecastTrustResult:
        return ForecastTrustResult(
            trust=self._params.neutral_prior_trust, reliability=0.0, horizon_bucket="none",
            source_bucket=None, bias_c=0.0, mean_absolute_error_c=0.0, effective_evidence=0.0,
            prior_contribution=1.0, learned_contribution=0.0,
            missing_evidence=("forecast_context",), reason_codes=(reason,),
            fallback_used=True, available=False)

    def _select_bucket(self, context: Optional[ForecastPredictionContext]):
        p = self._params
        reasons: list[str] = []
        source_bucket = None
        # 1. source+horizon (source bucket) if supported
        if context is not None and context.source_key:
            skey = f"src:{context.source_key}"
            sb = self._state.source_buckets.get(skey)
            if sb is not None and sb.sample_count >= p.bucket_min_samples:
                source_bucket = skey
                reasons.append("source_bucket")
                return sb, _hkey(context, p), source_bucket, False, reasons
        # 2. horizon general bucket
        if context is not None and context.horizon_s is not None:
            hkey = _horizon_bucket(context.horizon_s, p.horizon_bucket_edges_s)
            hb = self._state.horizon_buckets.get(hkey)
            if hb is not None and hb.sample_count >= p.bucket_min_samples:
                reasons.append("horizon_bucket")
                return hb, hkey, None, False, reasons
        # 3. general trust
        if self._state.general.has_evidence:
            reasons.append("general_state")
            return self._state.general, "general", None, False, reasons
        # 4. neutral conservative prior
        reasons.append("neutral_prior")
        return self._state.general, "general", None, True, reasons

    def predict(self, context: Any) -> Prediction:
        tr = self.predict_trust(context if isinstance(context, ForecastPredictionContext)
                                else ForecastPredictionContext())
        return Prediction(
            prediction_type=PredictionType.FORECAST_TRUST,
            values={"forecast_trust": tr.trust}, units={"forecast_trust": "score"},
            confidence=self._confidence_value(tr.effective_evidence, tr.fallback_used),
            reliability=tr.reliability, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION,
            prior_contribution=tr.prior_contribution,
            learned_contribution=tr.learned_contribution, fallback_used=tr.fallback_used,
            evidence_count=int(tr.effective_evidence), bucket=tr.horizon_bucket,
            confidence_cap=self._params.cold_start_confidence_cap if tr.fallback_used else None,
            cap_reasons=("cold_start",) if tr.fallback_used else (),
            missing_evidence=tr.missing_evidence, reason_codes=tr.reason_codes)

    def predict_bias(self, context: ForecastPredictionContext) -> Prediction:
        tr = self.predict_trust(context)
        return Prediction(
            prediction_type=PredictionType.FORECAST_BIAS,
            values={"forecast_bias": tr.bias_c}, units={"forecast_bias": "celsius"},
            confidence=self._confidence_value(tr.effective_evidence, tr.fallback_used),
            reliability=tr.reliability, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION, prior_contribution=tr.prior_contribution,
            learned_contribution=tr.learned_contribution, fallback_used=tr.fallback_used,
            evidence_count=int(tr.effective_evidence), bucket=tr.horizon_bucket,
            missing_evidence=tr.missing_evidence, reason_codes=tr.reason_codes)

    def predict_corrected_forecast(self, context: ForecastPredictionContext) -> Prediction:
        """Advisory bias-corrected forecast: raw + learned bias, hard-capped."""
        p = self._params
        tr = self.predict_trust(context)
        raw = context.raw_forecast_temperature_c if context is not None else None
        reasons = list(tr.reason_codes)
        correction = 0.0
        if (not tr.fallback_used and tr.effective_evidence >= p.min_samples_for_correction):
            correction = clamp(tr.bias_c, -p.bias_correction_max_c, p.bias_correction_max_c)
            reasons.append("bias_correction_applied")
        else:
            reasons.append("insufficient_evidence_for_correction")
        has_raw = raw is not None and is_finite(raw)
        corrected = (raw + correction) if has_raw else correction
        values = {"corrected_forecast": corrected, "applied_bias": correction}
        units = {"corrected_forecast": "celsius", "applied_bias": "celsius"}
        if has_raw:
            values["raw_forecast"] = raw
            units["raw_forecast"] = "celsius"
        return Prediction(
            prediction_type=PredictionType.CORRECTED_FORECAST,
            values=values, units=units,
            confidence=self._confidence_value(tr.effective_evidence, tr.fallback_used),
            reliability=tr.reliability, model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION, prior_contribution=tr.prior_contribution,
            learned_contribution=tr.learned_contribution, fallback_used=tr.fallback_used,
            evidence_count=int(tr.effective_evidence), bucket=tr.horizon_bucket,
            missing_evidence=tr.missing_evidence, reason_codes=tuple(reasons),
            warnings=("no_raw_forecast",) if raw is None else ())

    # -- confidence -----------------------------------------------------

    def _confidence_value(self, effective_evidence: float, fallback: bool) -> float:
        p = self._params
        if fallback or effective_evidence <= 0:
            return min(0.2, p.cold_start_confidence_cap)
        coverage = min(effective_evidence / p.full_confidence_samples, 1.0)
        abs_err = self._state.general.abs_error.value
        accuracy = clamp(1.0 - abs_err / p.error_scale_c, 0.2, 1.0)
        reality = self._mean_reality_quality() or 0.5
        return clamp(coverage * accuracy * reality, 0.0, 1.0)

    def confidence(self, *, now: Optional[str] = None) -> ConfidenceContribution:
        g = self._state.general
        reasons: list[str] = []
        if not g.has_evidence:
            reasons.append("cold_start")
        mrq = self._mean_reality_quality()
        if mrq is not None and mrq < 0.5:
            reasons.append("low_reality_quality")
        fallback = not g.has_evidence
        value = self._confidence_value(g.abs_error.effective_n, fallback)
        freshness, staleness = freshness_from_age(now, self._state.last_update_ts)
        status = "healthy" if g.has_evidence else "fallback"
        if g.has_evidence and staleness == "stale":
            status = "stale"
        return ConfidenceContribution(
            value=value, evidence_count=g.sample_count, reasons=tuple(reasons),
            component="forecast",
            reliability=clamp(g.abs_error.effective_n / self._params.full_confidence_samples,
                              0.0, 1.0),
            evidence_domains=("forecast", "weather"),
            source_count=len(self._state.source_buckets),
            prior_fraction=1.0 if fallback else 0.0,
            learned_fraction=0.0 if fallback else 1.0, fallback_used=fallback,
            freshness=freshness, status=status)

    def _mean_reality_quality(self) -> Optional[float]:
        n = self._state.general.sample_count
        return (self._state.reality_quality_sum / n) if n > 0 else None

    # -- diagnostics / export -------------------------------------------

    def diagnostics(self) -> ForecastDiagnostics:
        s = self._state
        g = s.general
        return ForecastDiagnostics(
            general_trust=self._trust_from(g), general_bias_c=g.bias.value,
            general_abs_error_c=g.abs_error.value, dispersion=g.abs_error.dispersion,
            horizon_buckets={k: self._trust_from(b) for k, b in s.horizon_buckets.items()},
            source_buckets={k: self._trust_from(b) for k, b in s.source_buckets.items()},
            sample_counts={"general": g.sample_count,
                           **{k: b.sample_count for k, b in s.horizon_buckets.items()}},
            mean_reality_quality=self._mean_reality_quality(),
            open_decisions=len(s.open_decisions), expired_decisions=s.expired_decisions,
            rejection_counts=dict(s.rejection_counts), outlier_counts=dict(s.outlier_counts),
            confidence=self.confidence().value, last_update_ts=s.last_update_ts,
            model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION)

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        s = self._state
        g = s.general
        if scope is ExportScope.SUPPORT:
            return {
                "general_trust": self._trust_from(g), "bias_c": g.bias.value,
                "abs_error_c": g.abs_error.value,
                "horizon_trust": {k: self._trust_from(b) for k, b in s.horizon_buckets.items()},
                "source_trust": {k: self._trust_from(b) for k, b in s.source_buckets.items()},
                "sample_count": g.sample_count, "confidence": self.confidence().value,
                "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
            }
        if scope is ExportScope.RESEARCH:
            return {
                "samples": [
                    {"temperature_error_c": x.temperature_error_c,
                     "absolute_error_c": x.absolute_error_c, "horizon_bucket": x.horizon_bucket,
                     "reality_quality": x.reality_quality, "model_version": x.model_version}
                    for x in s.recent_samples],
                "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
            }
        return {"unsupported": "raw_export_not_provided_by_model"}

    # -- lifecycle ------------------------------------------------------

    def reset(self) -> None:
        self._state = self._empty_state()

    def rebuild(self, raw: Any = None,
                items: Optional[Sequence[Any]] = None) -> ModelRebuildResult:
        self.reset()
        raw_items = list(items or [])
        # collect decisions and evaluations deterministically
        decisions: list[ForecastDecisionEvent] = []
        evaluations: list[tuple[ForecastEvaluationEvent, Optional[ForecastUpdateContext]]] = []
        for it in raw_items:
            if isinstance(it, ForecastRebuildItem):
                if it.decision is not None:
                    decisions.append(it.decision)
                if it.evaluation is not None:
                    evaluations.append((it.evaluation, it.context))
            elif isinstance(it, ForecastDecisionEvent):
                decisions.append(it)
            elif isinstance(it, ForecastEvaluationEvent):
                evaluations.append((it, None))
        for d in sorted(decisions, key=lambda d: (_iso(d.ts), d.decision_id)):
            self.register_decision(d)
        evaluations.sort(key=lambda pair: (_iso(pair[0].ts), pair[0].evaluation_id))
        accepted = rejected = duplicates = 0
        rejection_counts: dict[str, int] = {}
        for evaluation, context in evaluations:
            res = self.update(evaluation, context)
            if res.accepted:
                accepted += 1
            else:
                rejected += 1
                code = res.confounder_flags[0] if res.confounder_flags else "unknown"
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
                if code == ForecastRejection.DUPLICATE_EVALUATION.value:
                    duplicates += 1
        return ModelRebuildResult(
            processed_count=len(evaluations), accepted_count=accepted, rejected_count=rejected,
            duplicate_count=duplicates, error_count=0, rejection_counts=rejection_counts,
            started_from_prior=True, deterministic=True)

    # -- state ----------------------------------------------------------

    def serialize_state(self) -> Mapping[str, Any]:
        s = self._state
        return {
            "model_version": MODEL_VERSION, "parameter_version": PARAMETER_VERSION,
            "learning_zone_id": s.learning_zone_id, "general": _bucket_dict(s.general),
            "horizon_buckets": {k: _bucket_dict(b) for k, b in s.horizon_buckets.items()},
            "source_buckets": {k: _bucket_dict(b) for k, b in s.source_buckets.items()},
            "open_decisions": {k: _open_dict(d) for k, d in s.open_decisions.items()},
            "processed_eval_ids": list(s.processed_eval_ids),
            "processed_decision_ids": list(s.processed_decision_ids),
            "recent_samples": [_sample_dict(x) for x in s.recent_samples],
            "last_update_ts": s.last_update_ts, "reality_quality_sum": s.reality_quality_sum,
            "expired_decisions": s.expired_decisions,
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
        g = state.get("general", {})
        for dim in ("bias", "abs_error"):
            d = g.get(dim, {})
            for k in ("value", "effective_n", "dispersion"):
                v = d.get(k)
                if v is not None and not is_finite(v):
                    errors.append(f"non-finite general.{dim}.{k}")
        return errors

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        errors = self.validate_state(state)
        if errors:
            raise ValueError(f"invalid forecast state: {errors}")
        self._state = ForecastState(
            learning_zone_id=state["learning_zone_id"], general=_bucket_from(state["general"]),
            horizon_buckets={k: _bucket_from(b)
                             for k, b in state.get("horizon_buckets", {}).items()},
            source_buckets={k: _bucket_from(b)
                            for k, b in state.get("source_buckets", {}).items()},
            open_decisions={k: _open_from(d)
                            for k, d in state.get("open_decisions", {}).items()},
            processed_eval_ids=tuple(state.get("processed_eval_ids", [])),
            processed_decision_ids=tuple(state.get("processed_decision_ids", [])),
            recent_samples=tuple(_sample_from(x) for x in state.get("recent_samples", [])),
            last_update_ts=state.get("last_update_ts"),
            reality_quality_sum=state.get("reality_quality_sum", 0.0),
            expired_decisions=state.get("expired_decisions", 0),
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


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _hkey(context: Optional[ForecastPredictionContext], p: ForecastParameters) -> str:
    if context is not None and context.horizon_s is not None:
        return _horizon_bucket(context.horizon_s, p.horizon_bucket_edges_s)
    return "general"


def _open_dict(d: _OpenDecision) -> dict:
    return {"decision_id": d.decision_id, "learning_zone_id": d.learning_zone_id,
            "created_ts": d.created_ts, "forecast_high": d.forecast_high,
            "source_key": d.source_key}


def _open_from(d: Mapping[str, Any]) -> _OpenDecision:
    return _OpenDecision(decision_id=d["decision_id"], learning_zone_id=d["learning_zone_id"],
                         created_ts=d["created_ts"], forecast_high=d["forecast_high"],
                         source_key=d.get("source_key"))


def _sample_dict(x: ForecastSample) -> dict:
    return {
        "source_decision_id": x.source_decision_id, "evaluation_id": x.evaluation_id,
        "learning_zone_id": x.learning_zone_id, "temperature_error_c": x.temperature_error_c,
        "absolute_error_c": x.absolute_error_c, "horizon_s": x.horizon_s,
        "horizon_bucket": x.horizon_bucket, "source_key": x.source_key,
        "reality_quality": x.reality_quality, "effective_weight": x.effective_weight,
        "model_version": x.model_version,
    }


def _sample_from(d: Mapping[str, Any]) -> ForecastSample:
    return ForecastSample(
        source_decision_id=d["source_decision_id"], evaluation_id=d["evaluation_id"],
        learning_zone_id=d["learning_zone_id"], temperature_error_c=d["temperature_error_c"],
        absolute_error_c=d["absolute_error_c"], horizon_s=d["horizon_s"],
        horizon_bucket=d["horizon_bucket"], source_key=d.get("source_key"),
        reality_quality=d["reality_quality"], effective_weight=d["effective_weight"],
        model_version=d["model_version"])


def forecast_model_definition():
    from ..registry import ModelDefinition

    return ModelDefinition(
        model_name=MODEL_NAME, model_factory=lambda: ForecastModel("__definition__"),
        model_version=MODEL_VERSION, parameter_version=PARAMETER_VERSION,
        consumed_raw_tracks=(RawTrackName.FORECAST_DECISION, RawTrackName.FORECAST_EVALUATION),
        consumed_episode_types=(),
        supported_prediction_types=(
            PredictionType.FORECAST_TRUST, PredictionType.FORECAST_BIAS,
            PredictionType.CORRECTED_FORECAST),
        required_features=("forecast",),
        optional_features=("outdoor_temp", "wind_speed", "solar_radiation"),
        required_capability_flags=("has_forecast",), regime_dependent=False, advisory=True,
        control_relevant=False, rebuildable=True, can_be_not_available=True, min_trv_only=False)
