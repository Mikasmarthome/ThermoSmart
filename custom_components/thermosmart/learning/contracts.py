"""Typed contracts for the LE 2.0 model layer (pure Python, no Home Assistant).

Defines the cross-cutting vocabulary every learning component shares:

* data quality and measurement containers,
* the non-linear capability set,
* the structured ``Prediction`` and ``UpdateEligibilityResult`` results,
* the ``Model`` protocol all registered models must satisfy.

No concrete model is implemented here. Nothing in this module imports Home
Assistant, holds state, or performs I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, NewType, Optional, Protocol, runtime_checkable

# -- stable identity aliases (opaque strings, never visible entity names) -----

LearningZoneId = NewType("LearningZoneId", str)
TrvBindingId = NewType("TrvBindingId", str)
EpisodeId = NewType("EpisodeId", str)
DecisionId = NewType("DecisionId", str)
EvaluationId = NewType("EvaluationId", str)
EventId = NewType("EventId", str)


# -- enumerations -------------------------------------------------------------

class DataQuality(Enum):
    """Quality classification for an observed value."""

    OK = "ok"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class Regime(Enum):
    """Thermal regime produced by the (later) ThermalRegimeClassifier."""

    ACTIVE_HEATING = "active_heating"
    AFTERHEAT = "afterheat"
    PASSIVE_COOLING = "passive_cooling"
    STABLE = "stable"
    DISTURBED = "disturbed"
    UNKNOWN = "unknown"


class OutlierStatus(Enum):
    NONE = "none"
    MILD = "mild"
    SEVERE = "severe"


class RejectionReason(Enum):
    """Why a sample was not learned (or learned at reduced weight)."""

    DISTURBED_REGIME = "disturbed_regime"
    UNKNOWN_REGIME = "unknown_regime"
    MISSING_REQUIRED_DATA = "missing_required_data"
    OUTLIER = "outlier"
    DUPLICATE = "duplicate"
    LOW_RELIABILITY = "low_reliability"
    NOT_ELIGIBLE = "not_eligible"


class PredictionType(Enum):
    HEAT_RATE = "heat_rate"
    PREHEAT_MINUTES = "preheat_minutes"
    HEAT_LOSS_RATE = "heat_loss_rate"
    EXPECTED_OVERSHOOT = "expected_overshoot"
    AFTERHEAT_DURATION = "afterheat_duration"
    RECOMMENDED_EARLY_CUTOFF = "recommended_early_cutoff"
    FORECAST_TRUST = "forecast_trust"
    BOOST_FACTOR = "boost_factor"
    OUTCOME_SCORE = "outcome_score"
    HEATING_EFFORT = "heating_effort"


class ExportScope(Enum):
    SUPPORT = "support"
    RESEARCH = "research"
    RAW = "raw"
    DIAGNOSTICS = "diagnostics"


# -- helpers ------------------------------------------------------------------

def _check_unit_interval(name: str, value: float) -> None:
    if value is None or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0], got {value!r}")


def require_utc(ts: datetime, field_name: str = "ts") -> datetime:
    """Reject naive timestamps; normalise aware timestamps to UTC.

    Per the time contract all stored timestamps are UTC and timezone-aware.
    A naive datetime is a hard error (it cannot be interpreted unambiguously);
    an aware non-UTC datetime is explicitly normalised to UTC.
    """
    if not isinstance(ts, datetime):
        raise TypeError(f"{field_name} must be a datetime, got {type(ts)!r}")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (naive timestamps are rejected)")
    return ts.astimezone(timezone.utc)


def require_nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


# -- measurement container ----------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """An optional sensor value with explicit quality.

    ``value`` may be ``0.0`` (a perfectly valid measurement) — quality is the
    only thing that decides usability, never truthiness. ``None`` paired with a
    non-OK quality means "configured but currently not delivering a value".
    """

    value: Optional[float]
    quality: DataQuality = DataQuality.OK

    def __post_init__(self) -> None:
        if self.quality is DataQuality.OK and self.value is None:
            raise ValueError("a measurement with OK quality must carry a value")

    @property
    def is_usable(self) -> bool:
        return self.value is not None and self.quality is DataQuality.OK


# -- capability set (non-linear) ----------------------------------------------

@dataclass(frozen=True)
class CapabilitySet:
    """Concrete capability flags for a learning zone.

    The individual ``has_*`` flags are the authoritative truth. Capabilities are
    NOT linear: a zone may report valve opening while lacking outdoor/weather
    data. Models must therefore branch on the concrete flags they need — never
    on the numeric ``tier`` summary alone.
    """

    has_room_sensor: bool = False
    has_window_sensor: bool = False
    has_outdoor_temp: bool = False
    has_weather: bool = False
    has_forecast: bool = False
    has_wind: bool = False
    has_solar: bool = False
    has_humidity: bool = False
    has_rain: bool = False
    has_valve_opening: bool = False
    has_hvac_action: bool = False

    @property
    def tier(self) -> int:
        """Informative-only summary of the highest capability area present.

        WARNING: lossy. A higher tier does NOT imply lower tiers are fully
        satisfied (capabilities are non-linear). Provided for diagnostics and
        logging only — do not gate model behaviour on this value.
        """
        tier = 0
        if self.has_room_sensor or self.has_window_sensor:
            tier = max(tier, 1)
        if self.has_outdoor_temp or self.has_weather:
            tier = max(tier, 2)
        if self.has_valve_opening or self.has_hvac_action:
            tier = max(tier, 3)
        return tier


# -- prediction ---------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    """Structured, self-describing model output.

    Invariants enforced on construction:
    * confidence / reliability / contributions are within [0, 1];
    * prior_contribution + learned_contribution == 1.0;
    * at least one concrete (non-None) value with a unit — a control path never
      receives an unchecked ``None``;
    * fallbacks are transparent (fallback_used implies a positive prior share,
      a non-fallback prediction implies a positive learned share);
    * confidence never exceeds confidence_cap when a cap is present.
    """

    prediction_type: PredictionType
    values: Mapping[str, float]
    units: Mapping[str, str]
    confidence: float
    reliability: float
    model_version: int
    parameter_version: int
    prior_contribution: float
    learned_contribution: float
    fallback_used: bool
    evidence_count: int = 0
    bucket: Optional[str] = None
    valid_for: Optional[str] = None
    confidence_cap: Optional[float] = None
    cap_reasons: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_unit_interval("confidence", self.confidence)
        _check_unit_interval("reliability", self.reliability)
        _check_unit_interval("prior_contribution", self.prior_contribution)
        _check_unit_interval("learned_contribution", self.learned_contribution)
        if abs((self.prior_contribution + self.learned_contribution) - 1.0) > 1e-6:
            raise ValueError(
                "prior_contribution + learned_contribution must equal 1.0"
            )
        if not self.values:
            raise ValueError("a prediction must carry at least one value")
        for key, val in self.values.items():
            if val is None:
                raise ValueError(
                    f"value '{key}' must not be None in a control-path prediction"
                )
            if not isinstance(val, (int, float)):
                raise ValueError(f"value '{key}' must be numeric, got {type(val)!r}")
            if key not in self.units:
                raise ValueError(f"value '{key}' is missing a unit")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be >= 0")
        if self.confidence_cap is not None:
            _check_unit_interval("confidence_cap", self.confidence_cap)
            if self.confidence > self.confidence_cap + 1e-9:
                raise ValueError("confidence must not exceed confidence_cap")
        if self.fallback_used and self.prior_contribution <= 0.0:
            raise ValueError("fallback_used implies a positive prior_contribution")
        if (not self.fallback_used) and self.learned_contribution <= 0.0:
            raise ValueError(
                "a non-fallback prediction must have a positive learned_contribution"
            )

    def control_value(self, name: str) -> float:
        """Return a concrete value for a control path, never ``None``."""
        if name not in self.values:
            raise KeyError(name)
        val = self.values[name]
        if val is None:
            raise ValueError(f"control value '{name}' is None")
        return float(val)


# -- update eligibility -------------------------------------------------------

@dataclass(frozen=True)
class UpdateEligibilityResult:
    """Diagnoses whether (and how strongly) a sample is learned."""

    accepted: bool
    reliability: float
    weight: float = 0.0
    rejection_reason: Optional[RejectionReason] = None
    confounder_flags: tuple[str, ...] = ()
    required_data_missing: tuple[str, ...] = ()
    outlier_status: OutlierStatus = OutlierStatus.NONE
    regime: Optional[Regime] = None
    source_episode_id: Optional[str] = None
    feature_extractor_version: Optional[int] = None

    def __post_init__(self) -> None:
        _check_unit_interval("reliability", self.reliability)
        if self.weight < 0.0:
            raise ValueError("weight must be >= 0")
        if self.accepted:
            if self.rejection_reason is not None:
                raise ValueError("an accepted sample must not carry a rejection_reason")
        else:
            if self.weight != 0.0:
                raise ValueError("a rejected sample must have weight 0")
            if self.rejection_reason is None:
                raise ValueError("a rejected sample requires a rejection_reason")


# -- confidence contribution --------------------------------------------------

@dataclass(frozen=True)
class ConfidenceContribution:
    """A single model's contribution to the authoritative confidence."""

    value: float
    evidence_count: int = 0
    reasons: tuple[str, ...] = ()
    cap_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_unit_interval("value", self.value)
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be >= 0")


# -- model protocol -----------------------------------------------------------

@runtime_checkable
class Model(Protocol):
    """Contract every registered learning model must satisfy.

    Purity: ``predict``, ``confidence``, ``diagnostics``, ``export``,
    ``serialize_state`` and ``validate_state`` must not mutate state.
    ``update``, ``reset``, ``rebuild``, ``deserialize_state`` and
    ``migrate_state`` are the only mutating methods. ``update_eligibility`` is
    pure: it decides accept / weight / reject without mutating.

    Models never import Home Assistant, read entity states, call services, or
    send commands. They receive typed inputs and return typed results.
    """

    model_name: str
    model_version: int
    parameter_version: int
    consumed_raw_tracks: tuple[str, ...]
    consumed_episode_types: tuple[str, ...]
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    supported_prediction_types: tuple[PredictionType, ...]
    min_tier: int

    def cold_start_prior(self) -> Mapping[str, Any]:
        ...

    def update_eligibility(self, sample: Any) -> UpdateEligibilityResult:
        ...

    def update(self, sample: Any) -> UpdateEligibilityResult:
        ...

    def predict(self, context: Any) -> Prediction:
        ...

    def confidence(self) -> ConfidenceContribution:
        ...

    def diagnostics(self) -> Mapping[str, Any]:
        ...

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        ...

    def reset(self) -> None:
        ...

    def rebuild(self, raw: Any, episodes: Any) -> None:
        ...

    def serialize_state(self) -> Mapping[str, Any]:
        ...

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        ...

    def validate_state(self, state: Mapping[str, Any]) -> list[str]:
        ...

    def migrate_state(
        self, old_model_version: int, old_parameter_version: int,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...
