"""Pure-Python FeatureExtractor for LE 2.0.

Computes derived features at read time from raw records and episode data. It
mutates nothing, stores nothing, holds no global state, and has no Home
Assistant / storage / model dependency. Every result is a typed
:class:`FeatureResult` that distinguishes available, stale, unavailable,
invalid, not-computable and fallback outcomes, and carries an explicit
reliability with reason codes. ``NaN``/``Infinity`` never escape as an ``OK``
value.

Tunable thresholds live in the versioned :class:`FeatureParameters` — no inline
magic numbers in the logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo
from enum import Enum
from typing import Mapping, Optional, Sequence

from .contracts import DataQuality, Measurement, require_utc
from .episode_schemas import Trajectory, TrajectoryPoint

FEATURE_EXTRACTOR_VERSION = 1


# -- feature identity & status ------------------------------------------------

class FeatureName(Enum):
    # time
    HOUR_OF_DAY = "hour_of_day"
    WEEKDAY = "weekday"
    DURATION = "duration"
    # temperature
    INDOOR_TARGET_DELTA = "indoor_target_delta"
    INDOOR_OUTDOOR_DELTA = "indoor_outdoor_delta"
    SETPOINT_EXCESS = "setpoint_excess"
    # trajectory
    TRAJ_START = "traj_start"
    TRAJ_END = "traj_end"
    TRAJ_DELTA = "traj_delta"
    TRAJ_DURATION = "traj_duration"
    TRAJ_LINEAR_SLOPE = "traj_linear_slope"
    TRAJ_ROBUST_SLOPE = "traj_robust_slope"
    TRAJ_SECOND_DERIVATIVE = "traj_second_derivative"
    TRAJ_PEAK = "traj_peak"
    TRAJ_TIME_TO_PEAK = "traj_time_to_peak"
    TRAJ_MIN = "traj_min"
    TRAJ_MAX = "traj_max"
    TRAJ_GAP_RATIO = "traj_gap_ratio"
    TRAJ_VALID_POINTS = "traj_valid_points"
    TRAJ_SAMPLE_DENSITY = "traj_sample_density"
    TRAJ_PLAUSIBILITY = "traj_plausibility"
    # comfort
    TIME_TO_FIRST_TARGET = "time_to_first_target"
    TIME_IN_COMFORT_BAND = "time_in_comfort_band"
    OVERSHOOT = "overshoot"
    UNDERSHOOT = "undershoot"
    STABILITY_AFTER_TARGET = "stability_after_target"
    OSCILLATION = "oscillation"
    LATE_DROP = "late_drop"
    # heating / charge
    HEATING_DURATION = "heating_duration"
    SETPOINT_EXCESS_INTEGRAL = "setpoint_excess_integral"
    THERMAL_CHARGE_PROXY = "thermal_charge_proxy"
    VALVE_OPEN_TIME = "valve_open_time"
    TPI_ON_TIME = "tpi_on_time"
    # context
    OUTDOOR_TEMP_BUCKET = "outdoor_temp_bucket"
    WIND_BUCKET = "wind_bucket"
    SOLAR_BUCKET = "solar_bucket"
    HUMIDITY_BUCKET = "humidity_bucket"
    WEATHER_QUALITY = "weather_quality"


class FeatureStatus(Enum):
    OK = "ok"                      # available and valid
    STALE = "stale"                # available but based on stale input
    UNAVAILABLE = "unavailable"    # input configured but not delivering
    INVALID = "invalid"            # input present but not usable (e.g. NaN)
    NOT_COMPUTABLE = "not_computable"  # missing required inputs
    FALLBACK = "fallback"          # computed via a fallback path


_PRESENT_STATUSES = (FeatureStatus.OK, FeatureStatus.STALE, FeatureStatus.FALLBACK)


@dataclass(frozen=True)
class FeatureResult:
    name: FeatureName
    unit: str
    status: FeatureStatus
    reliability: float
    value: Optional[float] = None
    quality: DataQuality = DataQuality.OK
    missing_evidence: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    extractor_version: int = FEATURE_EXTRACTOR_VERSION

    def __post_init__(self) -> None:
        if self.reliability < 0.0 or self.reliability > 1.0:
            raise ValueError("reliability must be within [0.0, 1.0]")
        if self.status in _PRESENT_STATUSES:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError(
                    f"{self.name.value}: a present feature requires a finite value"
                )
        else:
            if self.value is not None:
                raise ValueError(
                    f"{self.name.value}: a non-present feature must not carry a value"
                )

    @property
    def is_present(self) -> bool:
        return self.status in _PRESENT_STATUSES


@dataclass(frozen=True)
class FeatureSet:
    """Typed collection of feature results (not a free dict contract)."""

    features: Mapping[FeatureName, FeatureResult]

    def get(self, name: FeatureName) -> FeatureResult:
        return self.features[name]

    def has(self, name: FeatureName) -> bool:
        return name in self.features

    def names(self) -> tuple[FeatureName, ...]:
        return tuple(self.features.keys())

    def present(self) -> tuple[FeatureName, ...]:
        return tuple(n for n, r in self.features.items() if r.is_present)


# -- versioned parameters -----------------------------------------------------

@dataclass(frozen=True)
class FeatureParameters:
    """Versioned, conservative thresholds. No inline magic numbers in logic."""

    min_trajectory_points: int = 4
    min_slope_points: int = 2
    max_gap_ms: int = 20 * 60 * 1000          # gaps larger than this are not bridged
    plausibility_max_rate_per_min: float = 0.5  # |dT/dt| above this is implausible
    smoothing_window: int = 3
    comfort_tolerance_default: float = 0.5
    oscillation_noise: float = 0.1             # min swing counted as a reversal
    irregularity_threshold: float = 0.5        # rel. stdev of inter-sample spacing
    outdoor_temp_buckets: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    wind_buckets: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0)
    solar_buckets: tuple[float, ...] = (50.0, 200.0, 400.0, 700.0)
    humidity_buckets: tuple[float, ...] = (40.0, 60.0, 80.0)
    parameter_version: int = 1


# -- internal helpers ---------------------------------------------------------

@dataclass(frozen=True)
class _UsablePoint:
    offset_ms: int
    value: float


def _is_usable_measurement(m: Optional[Measurement]) -> bool:
    return m is not None and m.value is not None and math.isfinite(m.value) \
        and m.quality in (DataQuality.OK, DataQuality.STALE)


def _usable_value(m: Optional[Measurement]) -> Optional[float]:
    return m.value if _is_usable_measurement(m) else None


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    idx = 0
    for edge in edges:
        if value >= edge:
            idx += 1
        else:
            break
    return idx


@dataclass(frozen=True)
class _TrajectoryView:
    points: tuple[_UsablePoint, ...]
    total_points: int
    gap_points: int
    stale_points: int
    invalid_points: int

    @property
    def usable_count(self) -> int:
        return len(self.points)

    @property
    def gap_ratio(self) -> float:
        return self.gap_points / self.total_points if self.total_points else 0.0

    @property
    def stale_ratio(self) -> float:
        return self.stale_points / self.total_points if self.total_points else 0.0


def _view(trajectory: Trajectory) -> _TrajectoryView:
    usable: list[_UsablePoint] = []
    gap = stale = invalid = 0
    for p in trajectory.points:
        if p.gap:
            gap += 1
            continue
        if p.value is None or not math.isfinite(p.value) \
                or p.quality in (DataQuality.UNAVAILABLE, DataQuality.INVALID,
                                 DataQuality.UNKNOWN):
            invalid += 1
            continue
        if p.quality is DataQuality.STALE:
            stale += 1
        usable.append(_UsablePoint(p.offset_ms, float(p.value)))
    return _TrajectoryView(
        points=tuple(usable),
        total_points=len(trajectory.points),
        gap_points=gap, stale_points=stale, invalid_points=invalid,
    )


def _irregular(view: _TrajectoryView, params: FeatureParameters) -> bool:
    if view.usable_count < 3:
        return False
    spacings = [view.points[i + 1].offset_ms - view.points[i].offset_ms
                for i in range(view.usable_count - 1)]
    mean = sum(spacings) / len(spacings)
    if mean <= 0:
        return True
    var = sum((s - mean) ** 2 for s in spacings) / len(spacings)
    rel_std = math.sqrt(var) / mean
    return rel_std > params.irregularity_threshold


def _trajectory_reliability(
    view: _TrajectoryView, params: FeatureParameters, *,
    min_points: int, plausible: bool = True, fallback: bool = False,
) -> tuple[float, tuple[str, ...]]:
    reliability = 1.0
    reasons: list[str] = []
    if fallback:
        reliability *= 0.6
        reasons.append("fallback")
    if view.usable_count < min_points and min_points > 0:
        reliability *= view.usable_count / min_points
        reasons.append("few_points")
    if view.gap_ratio > 0:
        reliability *= max(0.0, 1.0 - 0.5 * view.gap_ratio)
        reasons.append("gaps")
    if view.stale_ratio > 0:
        reliability *= max(0.0, 1.0 - 0.3 * view.stale_ratio)
        reasons.append("stale_points")
    if _irregular(view, params):
        reliability *= 0.85
        reasons.append("irregular_sampling")
    if not plausible:
        reliability *= 0.5
        reasons.append("implausible_jump")
    return max(0.0, min(1.0, reliability)), tuple(reasons)


# -- extractor ----------------------------------------------------------------

class FeatureExtractor:
    """Stateless extractor. Methods are pure and deterministic."""

    def __init__(self, params: Optional[FeatureParameters] = None) -> None:
        self._params = params or FeatureParameters()

    @property
    def params(self) -> FeatureParameters:
        return self._params

    # -- construction helpers -------------------------------------------

    def _present(self, name, unit, value, reliability, *, status=FeatureStatus.OK,
                 quality=DataQuality.OK, reasons=(), missing=(), refs=()) -> FeatureResult:
        if value is None or not math.isfinite(value):
            return FeatureResult(name=name, unit=unit, status=FeatureStatus.INVALID,
                                 reliability=0.0, quality=DataQuality.INVALID,
                                 reason_codes=("non_finite",) + tuple(reasons),
                                 missing_evidence=tuple(missing), source_refs=tuple(refs))
        return FeatureResult(name=name, unit=unit, status=status, reliability=reliability,
                             value=float(value), quality=quality,
                             reason_codes=tuple(reasons), missing_evidence=tuple(missing),
                             source_refs=tuple(refs))

    def _not_computable(self, name, unit, missing, reasons=()) -> FeatureResult:
        return FeatureResult(name=name, unit=unit, status=FeatureStatus.NOT_COMPUTABLE,
                             reliability=0.0, missing_evidence=tuple(missing),
                             reason_codes=tuple(reasons))

    def _unavailable(self, name, unit, reasons=()) -> FeatureResult:
        return FeatureResult(name=name, unit=unit, status=FeatureStatus.UNAVAILABLE,
                             reliability=0.0, quality=DataQuality.UNAVAILABLE,
                             reason_codes=tuple(reasons))

    # -- time ------------------------------------------------------------

    def extract_time_features(self, ts: datetime, zone: tzinfo) -> FeatureSet:
        """Civil time derived from a UTC instant and an explicit timezone."""
        ts = require_utc(ts, "ts")
        local = ts.astimezone(zone)  # DST-correct via the provided tz
        out = {
            FeatureName.HOUR_OF_DAY: self._present(
                FeatureName.HOUR_OF_DAY, "hour", float(local.hour), 1.0),
            FeatureName.WEEKDAY: self._present(
                FeatureName.WEEKDAY, "weekday", float(local.weekday()), 1.0),
        }
        return FeatureSet(out)

    def duration_seconds(self, start_utc: datetime, end_utc: datetime) -> FeatureResult:
        start = require_utc(start_utc, "start_utc")
        end = require_utc(end_utc, "end_utc")
        seconds = (end - start).total_seconds()
        if seconds < 0:
            return self._not_computable(FeatureName.DURATION, "s", (),
                                        ("negative_duration",))
        return self._present(FeatureName.DURATION, "s", seconds, 1.0)

    def duration_from_offsets(self, start_offset_ms: int, end_offset_ms: int) -> FeatureResult:
        if end_offset_ms < start_offset_ms:
            return self._not_computable(FeatureName.DURATION, "s", (),
                                        ("non_monotonic_offsets",))
        return self._present(FeatureName.DURATION, "s",
                             (end_offset_ms - start_offset_ms) / 1000.0, 1.0)

    # -- temperature -----------------------------------------------------

    def extract_temperature_features(
        self, *, indoor_temp: Optional[float], target: Optional[float],
        outdoor_temp: Optional[Measurement] = None,
        trv_setpoint: Optional[float] = None,
    ) -> FeatureSet:
        out: dict[FeatureName, FeatureResult] = {}
        has_indoor = indoor_temp is not None and math.isfinite(indoor_temp)

        # indoor-target delta
        if has_indoor and target is not None and math.isfinite(target):
            out[FeatureName.INDOOR_TARGET_DELTA] = self._present(
                FeatureName.INDOOR_TARGET_DELTA, "C", target - indoor_temp, 1.0)
        else:
            out[FeatureName.INDOOR_TARGET_DELTA] = self._not_computable(
                FeatureName.INDOOR_TARGET_DELTA, "C",
                () if has_indoor else ("indoor_temp",))

        # indoor-outdoor delta (only with usable outdoor)
        if not has_indoor:
            out[FeatureName.INDOOR_OUTDOOR_DELTA] = self._not_computable(
                FeatureName.INDOOR_OUTDOOR_DELTA, "C", ("indoor_temp",))
        elif _is_usable_measurement(outdoor_temp):
            out[FeatureName.INDOOR_OUTDOOR_DELTA] = self._present(
                FeatureName.INDOOR_OUTDOOR_DELTA, "C",
                indoor_temp - outdoor_temp.value, 1.0,
                quality=outdoor_temp.quality,
                status=(FeatureStatus.STALE if outdoor_temp.quality is DataQuality.STALE
                        else FeatureStatus.OK))
        elif outdoor_temp is not None:
            out[FeatureName.INDOOR_OUTDOOR_DELTA] = self._unavailable(
                FeatureName.INDOOR_OUTDOOR_DELTA, "C", ("outdoor_temp_unavailable",))
        else:
            out[FeatureName.INDOOR_OUTDOOR_DELTA] = self._not_computable(
                FeatureName.INDOOR_OUTDOOR_DELTA, "C", ("outdoor_temp",))

        # setpoint excess
        if has_indoor and trv_setpoint is not None and math.isfinite(trv_setpoint):
            out[FeatureName.SETPOINT_EXCESS] = self._present(
                FeatureName.SETPOINT_EXCESS, "C", trv_setpoint - indoor_temp, 1.0)
        else:
            missing = []
            if not has_indoor:
                missing.append("indoor_temp")
            if trv_setpoint is None:
                missing.append("trv_setpoint")
            out[FeatureName.SETPOINT_EXCESS] = self._not_computable(
                FeatureName.SETPOINT_EXCESS, "C", tuple(missing))

        return FeatureSet(out)

    # -- trajectory ------------------------------------------------------

    def extract_trajectory_features(self, trajectory: Trajectory) -> FeatureSet:
        params = self._params
        view = _view(trajectory)
        out: dict[FeatureName, FeatureResult] = {}

        out[FeatureName.TRAJ_GAP_RATIO] = self._present(
            FeatureName.TRAJ_GAP_RATIO, "ratio", view.gap_ratio, 1.0)
        out[FeatureName.TRAJ_VALID_POINTS] = self._present(
            FeatureName.TRAJ_VALID_POINTS, "count", float(view.usable_count), 1.0)

        if view.usable_count == 0:
            for name, unit in (
                (FeatureName.TRAJ_START, "C"), (FeatureName.TRAJ_END, "C"),
                (FeatureName.TRAJ_DELTA, "C"), (FeatureName.TRAJ_DURATION, "s"),
                (FeatureName.TRAJ_LINEAR_SLOPE, "C/min"),
                (FeatureName.TRAJ_ROBUST_SLOPE, "C/min"),
                (FeatureName.TRAJ_SECOND_DERIVATIVE, "C/min2"),
                (FeatureName.TRAJ_PEAK, "C"), (FeatureName.TRAJ_TIME_TO_PEAK, "s"),
                (FeatureName.TRAJ_MIN, "C"), (FeatureName.TRAJ_MAX, "C"),
                (FeatureName.TRAJ_SAMPLE_DENSITY, "1/min"),
                (FeatureName.TRAJ_PLAUSIBILITY, "C/min"),
            ):
                out[name] = self._not_computable(name, unit, ("no_usable_points",))
            return FeatureSet(out)

        values = [p.value for p in view.points]
        offsets = [p.offset_ms for p in view.points]
        first_off, last_off = offsets[0], offsets[-1]
        duration_ms = last_off - first_off
        duration_min = duration_ms / 60000.0

        out[FeatureName.TRAJ_START] = self._present(
            FeatureName.TRAJ_START, "C", values[0], 1.0)
        out[FeatureName.TRAJ_END] = self._present(
            FeatureName.TRAJ_END, "C", values[-1], 1.0)
        out[FeatureName.TRAJ_DELTA] = self._present(
            FeatureName.TRAJ_DELTA, "C", values[-1] - values[0], 1.0)
        out[FeatureName.TRAJ_DURATION] = self._present(
            FeatureName.TRAJ_DURATION, "s", duration_ms / 1000.0, 1.0)
        out[FeatureName.TRAJ_MIN] = self._present(
            FeatureName.TRAJ_MIN, "C", min(values), 1.0)
        out[FeatureName.TRAJ_MAX] = self._present(
            FeatureName.TRAJ_MAX, "C", max(values), 1.0)

        peak = max(values)
        peak_off = offsets[values.index(peak)]
        out[FeatureName.TRAJ_PEAK] = self._present(FeatureName.TRAJ_PEAK, "C", peak, 1.0)
        out[FeatureName.TRAJ_TIME_TO_PEAK] = self._present(
            FeatureName.TRAJ_TIME_TO_PEAK, "s", (peak_off - first_off) / 1000.0, 1.0)

        out[FeatureName.TRAJ_SAMPLE_DENSITY] = self._present(
            FeatureName.TRAJ_SAMPLE_DENSITY, "1/min",
            view.usable_count / duration_min if duration_min > 0 else 0.0,
            1.0 if duration_min > 0 else 0.5)

        # consecutive slopes (per minute) for robustness + plausibility
        slopes: list[float] = []
        max_abs_rate = 0.0
        for i in range(view.usable_count - 1):
            dt_min = (offsets[i + 1] - offsets[i]) / 60000.0
            if dt_min <= 0:
                continue
            rate = (values[i + 1] - values[i]) / dt_min
            slopes.append(rate)
            max_abs_rate = max(max_abs_rate, abs(rate))

        plausible = max_abs_rate <= params.plausibility_max_rate_per_min
        out[FeatureName.TRAJ_PLAUSIBILITY] = (
            self._present(FeatureName.TRAJ_PLAUSIBILITY, "C/min", max_abs_rate, 1.0)
            if plausible else
            FeatureResult(name=FeatureName.TRAJ_PLAUSIBILITY, unit="C/min",
                          status=FeatureStatus.INVALID, reliability=0.0,
                          quality=DataQuality.INVALID,
                          reason_codes=("implausible_jump",)))

        # linear slope
        if view.usable_count >= params.min_slope_points and duration_min > 0:
            rel, reasons = _trajectory_reliability(
                view, params, min_points=params.min_slope_points, plausible=plausible)
            out[FeatureName.TRAJ_LINEAR_SLOPE] = self._present(
                FeatureName.TRAJ_LINEAR_SLOPE, "C/min",
                (values[-1] - values[0]) / duration_min, rel, reasons=reasons)
        else:
            out[FeatureName.TRAJ_LINEAR_SLOPE] = self._not_computable(
                FeatureName.TRAJ_LINEAR_SLOPE, "C/min", (), ("too_few_points",))

        # robust slope + second derivative need the full minimum
        if view.usable_count >= params.min_trajectory_points and slopes:
            rel, reasons = _trajectory_reliability(
                view, params, min_points=params.min_trajectory_points, plausible=plausible)
            out[FeatureName.TRAJ_ROBUST_SLOPE] = self._present(
                FeatureName.TRAJ_ROBUST_SLOPE, "C/min", _median(slopes), rel,
                reasons=reasons)
            second = [slopes[i + 1] - slopes[i] for i in range(len(slopes) - 1)]
            out[FeatureName.TRAJ_SECOND_DERIVATIVE] = self._present(
                FeatureName.TRAJ_SECOND_DERIVATIVE, "C/min2",
                _median(second) if second else 0.0, rel, reasons=reasons)
        else:
            out[FeatureName.TRAJ_ROBUST_SLOPE] = self._not_computable(
                FeatureName.TRAJ_ROBUST_SLOPE, "C/min", (), ("too_few_points",))
            out[FeatureName.TRAJ_SECOND_DERIVATIVE] = self._not_computable(
                FeatureName.TRAJ_SECOND_DERIVATIVE, "C/min2", (), ("too_few_points",))

        return FeatureSet(out)

    # -- comfort ---------------------------------------------------------

    def extract_comfort_features(
        self, trajectory: Trajectory, *, target: float,
        comfort_tolerance: Optional[float] = None,
    ) -> FeatureSet:
        params = self._params
        tol = comfort_tolerance if comfort_tolerance is not None \
            else params.comfort_tolerance_default
        view = _view(trajectory)
        out: dict[FeatureName, FeatureResult] = {}

        if view.usable_count == 0 or not math.isfinite(target):
            for name, unit in (
                (FeatureName.TIME_TO_FIRST_TARGET, "s"),
                (FeatureName.TIME_IN_COMFORT_BAND, "s"),
                (FeatureName.OVERSHOOT, "C"), (FeatureName.UNDERSHOOT, "C"),
                (FeatureName.STABILITY_AFTER_TARGET, "C"),
                (FeatureName.OSCILLATION, "count"), (FeatureName.LATE_DROP, "C"),
            ):
                out[name] = self._not_computable(name, unit, ("no_usable_points",))
            return FeatureSet(out)

        values = [p.value for p in view.points]
        offsets = [p.offset_ms for p in view.points]
        first_off = offsets[0]
        rel, reasons = _trajectory_reliability(
            view, params, min_points=params.min_slope_points)

        # time to first target (value reaches target)
        first_idx = next((i for i, v in enumerate(values) if v >= target), None)
        if first_idx is None:
            out[FeatureName.TIME_TO_FIRST_TARGET] = self._not_computable(
                FeatureName.TIME_TO_FIRST_TARGET, "s", (), ("target_never_reached",))
        else:
            out[FeatureName.TIME_TO_FIRST_TARGET] = self._present(
                FeatureName.TIME_TO_FIRST_TARGET, "s",
                (offsets[first_idx] - first_off) / 1000.0, rel, reasons=reasons)

        # time in comfort band (step integration over intervals)
        band_ms = 0
        for i in range(view.usable_count - 1):
            if abs(values[i] - target) <= tol:
                band_ms += offsets[i + 1] - offsets[i]
        out[FeatureName.TIME_IN_COMFORT_BAND] = self._present(
            FeatureName.TIME_IN_COMFORT_BAND, "s", band_ms / 1000.0, rel, reasons=reasons)

        out[FeatureName.OVERSHOOT] = self._present(
            FeatureName.OVERSHOOT, "C", max(0.0, max(values) - target), rel, reasons=reasons)
        out[FeatureName.UNDERSHOOT] = self._present(
            FeatureName.UNDERSHOOT, "C", max(0.0, target - min(values)), rel, reasons=reasons)

        # stability after first reaching target = range of values afterwards
        if first_idx is not None and first_idx < view.usable_count:
            after = values[first_idx:]
            stability = (max(after) - min(after)) if after else 0.0
            out[FeatureName.STABILITY_AFTER_TARGET] = self._present(
                FeatureName.STABILITY_AFTER_TARGET, "C", stability, rel, reasons=reasons)
            late_drop = max(0.0, max(after) - after[-1])
            out[FeatureName.LATE_DROP] = self._present(
                FeatureName.LATE_DROP, "C", late_drop, rel, reasons=reasons)
        else:
            out[FeatureName.STABILITY_AFTER_TARGET] = self._not_computable(
                FeatureName.STABILITY_AFTER_TARGET, "C", (), ("target_never_reached",))
            out[FeatureName.LATE_DROP] = self._not_computable(
                FeatureName.LATE_DROP, "C", (), ("target_never_reached",))

        # oscillation = direction reversals above the noise floor
        reversals = 0
        last_dir = 0
        for i in range(view.usable_count - 1):
            diff = values[i + 1] - values[i]
            if abs(diff) < params.oscillation_noise:
                continue
            direction = 1 if diff > 0 else -1
            if last_dir != 0 and direction != last_dir:
                reversals += 1
            last_dir = direction
        out[FeatureName.OSCILLATION] = self._present(
            FeatureName.OSCILLATION, "count", float(reversals), rel, reasons=reasons)

        return FeatureSet(out)

    # -- heating / charge ------------------------------------------------

    def extract_heating_features(
        self, *, heating_duration_s: Optional[float] = None,
        setpoint_excess_samples: Optional[Sequence[tuple[int, float]]] = None,
        valve_open_time_s: Optional[float] = None,
        tpi_on_time_s: Optional[float] = None,
    ) -> FeatureSet:
        out: dict[FeatureName, FeatureResult] = {}

        if heating_duration_s is not None and math.isfinite(heating_duration_s) \
                and heating_duration_s >= 0:
            out[FeatureName.HEATING_DURATION] = self._present(
                FeatureName.HEATING_DURATION, "s", heating_duration_s, 1.0)
        else:
            out[FeatureName.HEATING_DURATION] = self._not_computable(
                FeatureName.HEATING_DURATION, "s", ("heating_duration",))

        integral = None
        if setpoint_excess_samples and len(setpoint_excess_samples) >= 2:
            total = 0.0
            ok = True
            for i in range(len(setpoint_excess_samples) - 1):
                (t0, e0), (t1, e1) = setpoint_excess_samples[i], setpoint_excess_samples[i + 1]
                if t1 < t0 or not math.isfinite(e0) or not math.isfinite(e1):
                    ok = False
                    break
                dt_min = (t1 - t0) / 60000.0
                total += (e0 + e1) / 2.0 * dt_min  # trapezoid, C*min
            integral = total if ok else None
        if integral is not None and math.isfinite(integral):
            out[FeatureName.SETPOINT_EXCESS_INTEGRAL] = self._present(
                FeatureName.SETPOINT_EXCESS_INTEGRAL, "C*min", integral, 1.0)
        else:
            out[FeatureName.SETPOINT_EXCESS_INTEGRAL] = self._not_computable(
                FeatureName.SETPOINT_EXCESS_INTEGRAL, "C*min", ("setpoint_excess_samples",))

        # thermal charge proxy: prefer the excess integral; else duration alone
        charge = integral
        used_fallback = False
        if charge is None and heating_duration_s is not None \
                and math.isfinite(heating_duration_s) and heating_duration_s >= 0:
            charge = heating_duration_s / 60.0  # minutes as a crude proxy
            used_fallback = True
        if charge is not None and math.isfinite(charge):
            out[FeatureName.THERMAL_CHARGE_PROXY] = self._present(
                FeatureName.THERMAL_CHARGE_PROXY, "C*min" if not used_fallback else "min",
                charge, 0.6 if used_fallback else 1.0,
                status=FeatureStatus.FALLBACK if used_fallback else FeatureStatus.OK,
                reasons=("duration_only",) if used_fallback else ())
        else:
            out[FeatureName.THERMAL_CHARGE_PROXY] = self._not_computable(
                FeatureName.THERMAL_CHARGE_PROXY, "C*min",
                ("setpoint_excess_samples", "heating_duration"))

        out[FeatureName.VALVE_OPEN_TIME] = (
            self._present(FeatureName.VALVE_OPEN_TIME, "s", valve_open_time_s, 1.0)
            if valve_open_time_s is not None and math.isfinite(valve_open_time_s)
            and valve_open_time_s >= 0
            else self._unavailable(FeatureName.VALVE_OPEN_TIME, "s", ("valve_open_time",)))
        out[FeatureName.TPI_ON_TIME] = (
            self._present(FeatureName.TPI_ON_TIME, "s", tpi_on_time_s, 1.0)
            if tpi_on_time_s is not None and math.isfinite(tpi_on_time_s)
            and tpi_on_time_s >= 0
            else self._unavailable(FeatureName.TPI_ON_TIME, "s", ("tpi_on_time",)))

        return FeatureSet(out)

    # -- context ---------------------------------------------------------

    def extract_context_features(
        self, *, outdoor_temp: Optional[Measurement] = None,
        wind_speed: Optional[Measurement] = None,
        solar_radiation: Optional[Measurement] = None,
        humidity: Optional[Measurement] = None,
    ) -> FeatureSet:
        params = self._params
        out: dict[FeatureName, FeatureResult] = {}

        def bucket(name, m, edges, label):
            if _is_usable_measurement(m):
                status = (FeatureStatus.STALE if m.quality is DataQuality.STALE
                          else FeatureStatus.OK)
                return self._present(name, "index", float(_bucket_index(m.value, edges)),
                                     1.0, status=status, quality=m.quality)
            if m is not None:
                return self._unavailable(name, "index", (f"{label}_unavailable",))
            return self._not_computable(name, "index", (label,))

        out[FeatureName.OUTDOOR_TEMP_BUCKET] = bucket(
            FeatureName.OUTDOOR_TEMP_BUCKET, outdoor_temp,
            params.outdoor_temp_buckets, "outdoor_temp")
        out[FeatureName.WIND_BUCKET] = bucket(
            FeatureName.WIND_BUCKET, wind_speed, params.wind_buckets, "wind_speed")
        out[FeatureName.SOLAR_BUCKET] = bucket(
            FeatureName.SOLAR_BUCKET, solar_radiation, params.solar_buckets, "solar_radiation")
        out[FeatureName.HUMIDITY_BUCKET] = bucket(
            FeatureName.HUMIDITY_BUCKET, humidity, params.humidity_buckets, "humidity")

        usable = sum(1 for m in (outdoor_temp, wind_speed, solar_radiation, humidity)
                     if _is_usable_measurement(m))
        configured = sum(1 for m in (outdoor_temp, wind_speed, solar_radiation, humidity)
                         if m is not None)
        if configured == 0:
            out[FeatureName.WEATHER_QUALITY] = self._not_computable(
                FeatureName.WEATHER_QUALITY, "ratio", ("weather",))
        else:
            out[FeatureName.WEATHER_QUALITY] = self._present(
                FeatureName.WEATHER_QUALITY, "ratio", usable / configured, 1.0)

        return FeatureSet(out)
