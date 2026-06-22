"""Deterministic, pure-Python thermal regime classifier for LE 2.0.

The classifier consumes FeatureExtractor results plus controller/TRV/telemetry
signals and returns a structured :class:`RegimeResult`. It is deterministic, has
no Home Assistant / storage / model dependency, mutates nothing, persists
nothing, and reaches no control decision. ``seconds_since_drive_end`` is passed
in so the classifier needs no clock.

Priority (highest first): hard disturbance / unusable data, active heating,
afterheat, passive cooling, stable, unknown. A weak heating heuristic can never
override a concrete disturbance or a better-evidenced afterheat phase, because
active heating requires current heating actuation/demand while afterheat
requires it to be absent.

An honest ``UNKNOWN`` is preferred over a falsely precise classification.
Missing optional telemetry alone never yields ``DISTURBED``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .contracts import DataQuality, Measurement, Regime
from .features import FeatureName, FeatureSet, FeatureStatus
from .raw_schemas import HeatingDriveEndEvent

CLASSIFIER_VERSION = 1


# -- structured codes ---------------------------------------------------------

class DisturbanceCode(Enum):
    WINDOW_OPEN = "window_open"
    HEATING_FAILURE = "heating_failure"
    IMPLAUSIBLE_JUMP = "implausible_jump"
    INVALID_CORE_MEASUREMENT = "invalid_core_measurement"
    HIGH_GAP_RATIO = "high_gap_ratio"
    KNOWN_EXTERNAL_DISTURBANCE = "known_external_disturbance"


class ReasonCode(Enum):
    ACTIVE_DEMAND = "active_demand"
    VALVE_OPEN = "valve_open"
    HVAC_HEATING = "hvac_heating"
    SETPOINT_OVER_INDOOR = "setpoint_over_indoor"
    POSITIVE_TREND = "positive_trend"
    NO_POSITIVE_TREND = "no_positive_trend"
    DECAYING_TREND = "decaying_trend"
    RECENT_DRIVE_END = "recent_drive_end"
    INDIRECT_DRIVE_END = "indirect_drive_end"
    DEMAND_OFF = "demand_off"
    NEGATIVE_TREND = "negative_trend"
    WITHIN_STABLE_EPS = "within_stable_eps"
    OUTSIDE_COMFORT_BAND = "outside_comfort_band"
    WITHIN_COMFORT_BAND = "within_comfort_band"
    SOLAR_CONFOUNDER = "solar_confounder"
    INSUFFICIENT_HISTORY = "insufficient_history"
    NO_INDOOR_TEMP = "no_indoor_temp"
    NO_TEMP_TREND = "no_temp_trend"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    RELATIVE_COOLING_ONLY = "relative_cooling_only"
    AMBIGUOUS_TREND = "ambiguous_trend"


class MissingEvidence(Enum):
    INDOOR_TEMP = "indoor_temp"
    VALVE_OPENING = "valve_opening"
    HVAC_ACTION = "hvac_action"
    WINDOW_STATE = "window_state"
    OUTDOOR_TEMP = "outdoor_temp"
    SOLAR = "solar"
    DRIVE_END_EVENT = "drive_end_event"
    CONTROLLER_DEMAND = "controller_demand"
    TEMPERATURE_TREND = "temperature_trend"


# -- parameters ---------------------------------------------------------------

@dataclass(frozen=True)
class RegimeParameters:
    """Versioned, conservative, provisional thresholds (calibrate after a season)."""

    min_history_points: int = 4
    stable_slope_eps: float = 0.005          # C/min: |slope| below this is stable
    cooling_slope_eps: float = 0.01          # C/min: slope at/below -this is cooling
    heating_slope_eps: float = 0.01          # C/min: slope at/above this is rising
    decay_second_derivative_max: float = 0.0  # second derivative below this = decaying
    max_seconds_since_drive_end: float = 1800.0  # 30 min afterheat window
    gap_cap: float = 0.5                      # gap ratio above this is a disturbance
    setpoint_over_indoor_eps: float = 0.3     # C: setpoint above indoor to count
    valve_open_eps: float = 0.0               # % above this counts as open
    solar_confounder_threshold: float = 400.0  # W/m2
    reliability_floor: float = 0.05
    reliability_ceiling: float = 0.95
    parameter_version: int = 1


# -- evidence & result --------------------------------------------------------

@dataclass(frozen=True)
class RegimeEvidence:
    code: str
    strength: float
    detail: str = ""

    def __post_init__(self) -> None:
        if self.strength < 0.0 or self.strength > 1.0:
            raise ValueError("evidence strength must be within [0.0, 1.0]")


@dataclass(frozen=True)
class RegimeResult:
    regime: Regime
    reliability: float
    evidence: tuple[RegimeEvidence, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    disturbance_reasons: tuple[str, ...] = ()
    feature_refs: tuple[str, ...] = ()
    reliability_caps: tuple[str, ...] = ()
    classifier_version: int = CLASSIFIER_VERSION
    parameter_version: int = 1

    def __post_init__(self) -> None:
        if self.reliability < 0.0 or self.reliability > 1.0:
            raise ValueError("reliability must be within [0.0, 1.0]")
        if self.regime is Regime.DISTURBED and not self.disturbance_reasons:
            raise ValueError("DISTURBED requires at least one disturbance reason")


# -- input --------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeInput:
    indoor_temp: Optional[float] = None
    target: Optional[float] = None
    trv_setpoint: Optional[float] = None
    controller_demand: Optional[bool] = None
    tpi_on: Optional[bool] = None
    valve_opening: Optional[Measurement] = None
    hvac_action: Optional[str] = None
    heating_drive_end: Optional[HeatingDriveEndEvent] = None
    seconds_since_drive_end: Optional[float] = None
    window_open: Optional[bool] = None
    heating_failure: bool = False
    outdoor_temp: Optional[Measurement] = None
    solar_radiation: Optional[Measurement] = None
    trajectory_features: Optional[FeatureSet] = None


# -- internal trend view ------------------------------------------------------

@dataclass(frozen=True)
class _Trend:
    available: bool
    slope: Optional[float]
    slope_reliability: float
    second_derivative: Optional[float]
    plausible: bool
    valid_points: int
    gap_ratio: Optional[float]


def _present_value(fs: FeatureSet, name: FeatureName) -> Optional[float]:
    if not fs.has(name):
        return None
    r = fs.get(name)
    return r.value if r.is_present else None


def _trend_of(inp: RegimeInput) -> _Trend:
    fs = inp.trajectory_features
    if fs is None:
        return _Trend(False, None, 0.0, None, True, 0, None)
    slope = _present_value(fs, FeatureName.TRAJ_ROBUST_SLOPE)
    slope_rel = fs.get(FeatureName.TRAJ_ROBUST_SLOPE).reliability \
        if fs.has(FeatureName.TRAJ_ROBUST_SLOPE) else 0.0
    if slope is None:
        slope = _present_value(fs, FeatureName.TRAJ_LINEAR_SLOPE)
        slope_rel = fs.get(FeatureName.TRAJ_LINEAR_SLOPE).reliability \
            if fs.has(FeatureName.TRAJ_LINEAR_SLOPE) else 0.0
    second = _present_value(fs, FeatureName.TRAJ_SECOND_DERIVATIVE)
    valid_points = int(_present_value(fs, FeatureName.TRAJ_VALID_POINTS) or 0)
    gap_ratio = _present_value(fs, FeatureName.TRAJ_GAP_RATIO)
    plausible = True
    if fs.has(FeatureName.TRAJ_PLAUSIBILITY):
        plausible = fs.get(FeatureName.TRAJ_PLAUSIBILITY).status is not FeatureStatus.INVALID
    return _Trend(slope is not None, slope, slope_rel, second, plausible,
                  valid_points, gap_ratio)


def _usable(m: Optional[Measurement]) -> Optional[float]:
    if m is not None and m.value is not None and m.quality in (DataQuality.OK, DataQuality.STALE):
        return m.value
    return None


# -- classifier ---------------------------------------------------------------

class ThermalRegimeClassifier:
    """Stateless deterministic classifier."""

    def __init__(self, params: Optional[RegimeParameters] = None) -> None:
        self._params = params or RegimeParameters()

    @property
    def params(self) -> RegimeParameters:
        return self._params

    def _result(self, regime, reliability, *, evidence=(), missing=(), reasons=(),
                disturbance=(), refs=(), caps=()) -> RegimeResult:
        p = self._params
        rel = max(p.reliability_floor if regime is not Regime.UNKNOWN else 0.0,
                  min(p.reliability_ceiling, reliability))
        rel = max(0.0, min(1.0, rel))
        return RegimeResult(
            regime=regime, reliability=rel, evidence=tuple(evidence),
            missing_evidence=tuple(m.value for m in missing),
            reason_codes=tuple(r.value for r in reasons),
            disturbance_reasons=tuple(d.value for d in disturbance),
            feature_refs=tuple(refs), reliability_caps=tuple(caps),
            parameter_version=p.parameter_version,
        )

    def classify(self, inp: RegimeInput) -> RegimeResult:
        p = self._params
        trend = _trend_of(inp)

        # --- 1. DISTURBED (concrete known disturbance / unusable data) ----
        disturbance: list[DisturbanceCode] = []
        if inp.heating_failure:
            disturbance.append(DisturbanceCode.HEATING_FAILURE)
        if inp.window_open is True:
            disturbance.append(DisturbanceCode.WINDOW_OPEN)
        if inp.indoor_temp is not None and not _finite(inp.indoor_temp):
            disturbance.append(DisturbanceCode.INVALID_CORE_MEASUREMENT)
        if trend.available and not trend.plausible:
            disturbance.append(DisturbanceCode.IMPLAUSIBLE_JUMP)
        if trend.gap_ratio is not None and trend.gap_ratio > p.gap_cap:
            disturbance.append(DisturbanceCode.HIGH_GAP_RATIO)
        if disturbance:
            concrete = inp.heating_failure or inp.window_open is True
            return self._result(
                Regime.DISTURBED, 0.9 if concrete else 0.6,
                disturbance=disturbance,
                evidence=(RegimeEvidence("disturbance", 1.0 if concrete else 0.6),),
            )

        missing = self._missing(inp, trend)
        solar_conf = self._solar_confounder(inp, trend)

        demand = inp.controller_demand is True
        valve_open = (_usable(inp.valve_opening) is not None
                      and _usable(inp.valve_opening) > p.valve_open_eps)
        hvac_heating = inp.hvac_action == "heating"
        heating_now = demand or valve_open or hvac_heating

        # --- 2. ACTIVE_HEATING --------------------------------------------
        if heating_now:
            return self._active_heating(inp, trend, missing, solar_conf,
                                        demand, valve_open, hvac_heating)

        # not actively heating: afterheat / cooling / stable / unknown
        # --- 3. AFTERHEAT --------------------------------------------------
        afterheat = self._try_afterheat(inp, trend, missing, solar_conf)
        if afterheat is not None:
            return afterheat

        # require a usable trend for the remaining thermal regimes
        if not trend.available:
            reasons = [ReasonCode.NO_TEMP_TREND]
            if inp.indoor_temp is None:
                reasons.append(ReasonCode.NO_INDOOR_TEMP)
            return self._result(Regime.UNKNOWN, 0.2, missing=missing, reasons=reasons)
        if trend.valid_points < p.min_history_points:
            return self._result(Regime.UNKNOWN, 0.25, missing=missing,
                                reasons=[ReasonCode.INSUFFICIENT_HISTORY],
                                refs=(FeatureName.TRAJ_VALID_POINTS.value,))

        slope = trend.slope or 0.0

        # --- 4. PASSIVE_COOLING -------------------------------------------
        if slope <= -p.cooling_slope_eps:
            return self._passive_cooling(inp, trend, missing, slope)

        # --- 5. STABLE -----------------------------------------------------
        if abs(slope) < p.stable_slope_eps:
            return self._stable(inp, trend, missing, slope)

        # --- 6. UNKNOWN (ambiguous mild / unexplained positive trend) ------
        reasons = [ReasonCode.AMBIGUOUS_TREND]
        if slope >= p.heating_slope_eps:
            reasons = [ReasonCode.POSITIVE_TREND, ReasonCode.DEMAND_OFF]
            if solar_conf:
                reasons.append(ReasonCode.SOLAR_CONFOUNDER)
        return self._result(Regime.UNKNOWN, 0.3 * (trend.slope_reliability or 1.0) + 0.1,
                            missing=missing, reasons=reasons,
                            refs=(FeatureName.TRAJ_ROBUST_SLOPE.value,))

    # -- regime helpers --------------------------------------------------

    def _active_heating(self, inp, trend, missing, solar_conf,
                        demand, valve_open, hvac_heating) -> RegimeResult:
        p = self._params
        evidence: list[RegimeEvidence] = []
        reasons: list[ReasonCode] = []
        caps: list[str] = []

        # conflicting: heating actuation but clearly cooling
        if trend.available and trend.slope is not None and trend.slope < -p.cooling_slope_eps:
            return self._result(Regime.UNKNOWN, 0.25, missing=missing,
                                reasons=[ReasonCode.CONFLICTING_EVIDENCE,
                                         ReasonCode.NEGATIVE_TREND],
                                refs=(FeatureName.TRAJ_ROBUST_SLOPE.value,))

        actuation = valve_open or hvac_heating
        positive_trend = (trend.available and trend.slope is not None
                          and trend.slope >= p.heating_slope_eps)
        # Demand alone, with neither heating actuation nor a usable positive
        # temperature response, is not enough to confirm active heating. Without
        # indoor temperature this stays an honest UNKNOWN (no invented regime).
        if not actuation and not positive_trend:
            reasons = [ReasonCode.ACTIVE_DEMAND]
            if not trend.available:
                reasons.append(ReasonCode.NO_TEMP_TREND)
                if inp.indoor_temp is None:
                    reasons.append(ReasonCode.NO_INDOOR_TEMP)
            else:
                reasons.append(ReasonCode.NO_POSITIVE_TREND)
            return self._result(Regime.UNKNOWN, 0.25, missing=missing, reasons=reasons)

        if demand:
            evidence.append(RegimeEvidence(ReasonCode.ACTIVE_DEMAND.value, 0.6))
            reasons.append(ReasonCode.ACTIVE_DEMAND)
        if valve_open:
            evidence.append(RegimeEvidence(ReasonCode.VALVE_OPEN.value, 0.8))
            reasons.append(ReasonCode.VALVE_OPEN)
        if hvac_heating:
            evidence.append(RegimeEvidence(ReasonCode.HVAC_HEATING.value, 0.8))
            reasons.append(ReasonCode.HVAC_HEATING)
        setpoint_over = (inp.indoor_temp is not None and inp.trv_setpoint is not None
                         and _finite(inp.indoor_temp) and _finite(inp.trv_setpoint)
                         and inp.trv_setpoint - inp.indoor_temp > p.setpoint_over_indoor_eps)
        if setpoint_over:
            evidence.append(RegimeEvidence(ReasonCode.SETPOINT_OVER_INDOOR.value, 0.4))
            reasons.append(ReasonCode.SETPOINT_OVER_INDOOR)

        strong = demand and (valve_open or hvac_heating)
        base = 0.85 if strong else (0.7 if (valve_open or hvac_heating) else 0.6)

        positive_trend = (trend.available and trend.slope is not None
                          and trend.slope >= p.heating_slope_eps)
        if positive_trend:
            reasons.append(ReasonCode.POSITIVE_TREND)
            base *= (0.6 + 0.4 * (trend.slope_reliability or 1.0))
        else:
            reasons.append(ReasonCode.NO_POSITIVE_TREND)
            base *= 0.8
            caps.append(ReasonCode.NO_POSITIVE_TREND.value)

        if solar_conf and not strong:
            base *= 0.7
            reasons.append(ReasonCode.SOLAR_CONFOUNDER)
            caps.append(ReasonCode.SOLAR_CONFOUNDER.value)
        if MissingEvidence.VALVE_OPENING in missing or MissingEvidence.HVAC_ACTION in missing:
            base *= 0.9
            caps.append("missing_actuation_telemetry")

        return self._result(Regime.ACTIVE_HEATING, base, evidence=evidence,
                            missing=missing, reasons=reasons, caps=caps,
                            refs=(FeatureName.TRAJ_ROBUST_SLOPE.value,))

    def _try_afterheat(self, inp, trend, missing, solar_conf) -> Optional[RegimeResult]:
        p = self._params
        recent_direct = (inp.heating_drive_end is not None
                         and inp.seconds_since_drive_end is not None
                         and 0 <= inp.seconds_since_drive_end <= p.max_seconds_since_drive_end)
        too_old = (inp.heating_drive_end is not None
                   and inp.seconds_since_drive_end is not None
                   and inp.seconds_since_drive_end > p.max_seconds_since_drive_end)
        rising = (trend.available and trend.slope is not None
                  and trend.slope >= p.heating_slope_eps)
        decaying = (trend.second_derivative is not None
                    and trend.second_derivative < p.decay_second_derivative_max)
        # indirect drive end: demand explicitly off and a rising+decaying tail
        indirect = (not recent_direct and not too_old
                    and inp.controller_demand is False and rising and decaying)

        if too_old:
            return None
        if not (recent_direct or indirect):
            return None
        if not (rising and decaying):
            return None

        reasons = [ReasonCode.DECAYING_TREND, ReasonCode.DEMAND_OFF]
        evidence = [RegimeEvidence(ReasonCode.DECAYING_TREND.value, 0.6)]
        if recent_direct:
            reasons.append(ReasonCode.RECENT_DRIVE_END)
            evidence.append(RegimeEvidence(ReasonCode.RECENT_DRIVE_END.value,
                                           inp.heating_drive_end.source_confidence))
            base = 0.7 * (0.5 + 0.5 * inp.heating_drive_end.source_confidence)
        else:
            reasons.append(ReasonCode.INDIRECT_DRIVE_END)
            base = 0.45

        if solar_conf:
            # cannot distinguish residual heat from solar gain -> stay honest
            return self._result(Regime.UNKNOWN, 0.3, missing=missing,
                                reasons=[ReasonCode.SOLAR_CONFOUNDER,
                                         ReasonCode.DECAYING_TREND],
                                refs=(FeatureName.TRAJ_SECOND_DERIVATIVE.value,))

        base *= (0.6 + 0.4 * (trend.slope_reliability or 1.0))
        caps: list[str] = []
        if MissingEvidence.VALVE_OPENING in missing or MissingEvidence.HVAC_ACTION in missing:
            base *= 0.85
            caps.append("missing_actuation_telemetry")
        return self._result(Regime.AFTERHEAT, base, evidence=evidence, missing=missing,
                            reasons=reasons, caps=caps,
                            refs=(FeatureName.TRAJ_SECOND_DERIVATIVE.value,
                                  FeatureName.TRAJ_ROBUST_SLOPE.value))

    def _passive_cooling(self, inp, trend, missing, slope) -> RegimeResult:
        p = self._params
        reasons = [ReasonCode.NEGATIVE_TREND, ReasonCode.DEMAND_OFF]
        caps: list[str] = []
        base = 0.7 * (0.6 + 0.4 * (trend.slope_reliability or 1.0))
        if _usable(inp.outdoor_temp) is None:
            reasons.append(ReasonCode.RELATIVE_COOLING_ONLY)
            caps.append(ReasonCode.RELATIVE_COOLING_ONLY.value)
            base *= 0.8
        return self._result(Regime.PASSIVE_COOLING, base, missing=missing,
                            reasons=reasons, caps=caps,
                            evidence=(RegimeEvidence(ReasonCode.NEGATIVE_TREND.value, 0.6),),
                            refs=(FeatureName.TRAJ_ROBUST_SLOPE.value,))

    def _stable(self, inp, trend, missing, slope) -> RegimeResult:
        reasons = [ReasonCode.WITHIN_STABLE_EPS]
        base = 0.75 * (0.6 + 0.4 * (trend.slope_reliability or 1.0))
        if inp.indoor_temp is not None and inp.target is not None \
                and _finite(inp.indoor_temp) and _finite(inp.target):
            tol = 0.5
            if abs(inp.indoor_temp - inp.target) <= tol:
                reasons.append(ReasonCode.WITHIN_COMFORT_BAND)
            else:
                reasons.append(ReasonCode.OUTSIDE_COMFORT_BAND)
        return self._result(Regime.STABLE, base, missing=missing, reasons=reasons,
                            evidence=(RegimeEvidence(ReasonCode.WITHIN_STABLE_EPS.value, 0.6),),
                            refs=(FeatureName.TRAJ_ROBUST_SLOPE.value,))

    # -- evidence helpers ------------------------------------------------

    def _solar_confounder(self, inp, trend) -> bool:
        solar = _usable(inp.solar_radiation)
        rising = trend.available and trend.slope is not None \
            and trend.slope >= self._params.heating_slope_eps
        return solar is not None and solar > self._params.solar_confounder_threshold and rising

    def _missing(self, inp, trend) -> list[MissingEvidence]:
        missing: list[MissingEvidence] = []
        if inp.indoor_temp is None and not trend.available:
            missing.append(MissingEvidence.INDOOR_TEMP)
        if inp.valve_opening is None:
            missing.append(MissingEvidence.VALVE_OPENING)
        if inp.hvac_action is None:
            missing.append(MissingEvidence.HVAC_ACTION)
        if inp.window_open is None:
            missing.append(MissingEvidence.WINDOW_STATE)
        if inp.outdoor_temp is None:
            missing.append(MissingEvidence.OUTDOOR_TEMP)
        if inp.solar_radiation is None:
            missing.append(MissingEvidence.SOLAR)
        if inp.controller_demand is None:
            missing.append(MissingEvidence.CONTROLLER_DEMAND)
        if not trend.available:
            missing.append(MissingEvidence.TEMPERATURE_TREND)
        return missing


def _finite(x: Optional[float]) -> bool:
    import math
    return x is not None and math.isfinite(x)
