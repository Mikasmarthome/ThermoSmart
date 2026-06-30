"""Typed episode schemas for LE 2.0 (pure Python, no Home Assistant).

Episodes are materialised, self-contained historical records. They carry their
own schema / builder / classifier versions so a later reclassification never
silently overwrites them. Derived model values (overshoot, durations, scores,
peak temperature when a trajectory exists) are NOT stored as authoritative
episode fields — they are recomputed from the trajectory at read time.

No builder logic lives here. Weather, valve and window data are never required
to construct a valid episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .contracts import (
    DataQuality,
    DecisionId,
    Measurement,
    Regime,
    require_nonempty,
    require_utc,
)


class EpisodeType(Enum):
    HEATING = "heating"
    AFTERHEAT = "afterheat"
    PASSIVE_COOLING = "passive_cooling"
    WINDOW_COOLING = "window_cooling"
    OUTCOME = "outcome"


class EpisodeReason(Enum):
    REACHED = "reached"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"          # controlled controller-end / generic interruption
    MANUAL_CORRECTION = "manual_correction"
    SUPERSEDED = "superseded"
    MODE_CHANGE = "mode_change"
    DISTURBED = "disturbed"
    INSUFFICIENT_DATA = "insufficient_data"


class ControllerKind(Enum):
    THERMOSMART = "ts"
    OBSERVED = "obs"


# -- trajectory ---------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryPoint:
    """One capped trajectory sample.

    ``offset_ms`` is derived from a monotonic clock so durations survive system
    clock corrections. Gaps are represented explicitly: a gap point carries no
    value. A non-gap OK point must carry a value.
    """

    offset_ms: int
    value: Optional[float] = None
    quality: DataQuality = DataQuality.OK
    gap: bool = False

    def __post_init__(self) -> None:
        if self.offset_ms < 0:
            raise ValueError("offset_ms must be >= 0")
        if self.gap:
            if self.value is not None:
                raise ValueError("a gap point must not carry a value")
        elif self.value is None and self.quality is DataQuality.OK:
            raise ValueError("a non-gap OK point requires a value")


@dataclass(frozen=True)
class Trajectory:
    """An immutable, capped, strictly time-ordered series of points."""

    points: tuple[TrajectoryPoint, ...]
    max_points: int

    def __post_init__(self) -> None:
        if self.max_points <= 0:
            raise ValueError("max_points must be > 0")
        if len(self.points) > self.max_points:
            raise ValueError("trajectory exceeds max_points")
        last = -1
        for point in self.points:
            if point.offset_ms <= last:
                raise ValueError("trajectory offsets must be strictly increasing")
            last = point.offset_ms

    @property
    def is_capped(self) -> bool:
        return len(self.points) == self.max_points

    @property
    def has_gaps(self) -> bool:
        return any(p.gap for p in self.points)


# -- common validation --------------------------------------------------------

def _validate_common(
    episode_id: str,
    learning_zone_id: str,
    start_ts: datetime,
    end_ts: datetime,
    reliability: float,
) -> tuple[datetime, datetime]:
    require_nonempty(episode_id, "episode_id")
    require_nonempty(learning_zone_id, "learning_zone_id")
    start = require_utc(start_ts, "start_ts")
    end = require_utc(end_ts, "end_ts")
    if end < start:
        raise ValueError("end_ts must not precede start_ts")
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability must be within [0.0, 1.0]")
    return start, end


# -- episodes -----------------------------------------------------------------

@dataclass(frozen=True)
class HeatingEpisode:
    """An active-heating phase within a zone."""

    episode_id: str
    learning_zone_id: str
    episode_schema_version: int
    builder_version: int
    classifier_version: int
    start_ts: datetime
    end_ts: datetime
    regime: Regime
    reliability: float
    start_temp: float
    target: float
    time_quality: DataQuality = DataQuality.OK
    confounder_flags: tuple[str, ...] = ()
    trajectory: Optional[Trajectory] = None
    controller: Optional[ControllerKind] = None
    trv_binding_id: Optional[str] = None

    def __post_init__(self) -> None:
        s, e = _validate_common(
            self.episode_id, self.learning_zone_id,
            self.start_ts, self.end_ts, self.reliability,
        )
        object.__setattr__(self, "start_ts", s)
        object.__setattr__(self, "end_ts", e)


@dataclass(frozen=True)
class AfterheatEpisode:
    """Residual warming after the heat drive ended (functions without valve%).

    ``start_ts`` IS the heating-drive-end timestamp, and ``indoor_temp_at_close``
    is the indoor temperature at that heating drive end (i.e. at episode start) —
    not at episode end. The authoritative residual rise is
    ``peak − temperature_at_heating_drive_end``.
    """

    episode_id: str
    learning_zone_id: str
    episode_schema_version: int
    builder_version: int
    classifier_version: int
    start_ts: datetime
    end_ts: datetime
    regime: Regime
    reliability: float
    indoor_temp_at_close: float
    target: float
    trv_setpoint_before: float
    trv_setpoint_after: float
    trajectory: Trajectory
    time_quality: DataQuality = DataQuality.OK
    confounder_flags: tuple[str, ...] = ()
    valve_before: Optional[Measurement] = None
    valve_after: Optional[Measurement] = None
    radiator_profile_id: Optional[str] = None
    trv_binding_id: Optional[str] = None

    def __post_init__(self) -> None:
        s, e = _validate_common(
            self.episode_id, self.learning_zone_id,
            self.start_ts, self.end_ts, self.reliability,
        )
        object.__setattr__(self, "start_ts", s)
        object.__setattr__(self, "end_ts", e)
        if not isinstance(self.trajectory, Trajectory):
            raise ValueError("AfterheatEpisode requires a Trajectory")


@dataclass(frozen=True)
class PassiveCoolingEpisode:
    """Steady cooling with the valve closed (used by HeatLoss; window excluded)."""

    episode_id: str
    learning_zone_id: str
    episode_schema_version: int
    builder_version: int
    classifier_version: int
    start_ts: datetime
    end_ts: datetime
    regime: Regime
    reliability: float
    start_temp: float
    end_temp: float
    trajectory: Trajectory
    time_quality: DataQuality = DataQuality.OK
    confounder_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        s, e = _validate_common(
            self.episode_id, self.learning_zone_id,
            self.start_ts, self.end_ts, self.reliability,
        )
        object.__setattr__(self, "start_ts", s)
        object.__setattr__(self, "end_ts", e)
        if not isinstance(self.trajectory, Trajectory):
            raise ValueError("PassiveCoolingEpisode requires a Trajectory")


@dataclass(frozen=True)
class WindowCoolingEpisode:
    """Cooling driven by an open window (kept separate from passive cooling)."""

    episode_id: str
    learning_zone_id: str
    episode_schema_version: int
    builder_version: int
    classifier_version: int
    start_ts: datetime
    end_ts: datetime
    regime: Regime
    reliability: float
    temp_at_open: float
    temp_at_close: float
    time_quality: DataQuality = DataQuality.OK
    confounder_flags: tuple[str, ...] = ()
    trajectory: Optional[Trajectory] = None

    def __post_init__(self) -> None:
        s, e = _validate_common(
            self.episode_id, self.learning_zone_id,
            self.start_ts, self.end_ts, self.reliability,
        )
        object.__setattr__(self, "start_ts", s)
        object.__setattr__(self, "end_ts", e)


@dataclass(frozen=True)
class OutcomeEpisode:
    """A heating session with a capped trajectory for comfort metrics.

    The outcome score and peak temperature are NOT stored: they are recomputed
    from the trajectory by the OutcomeModel at read time. ``comfort_tolerance_at_start``
    is stored (not a comfort band) so later config changes cannot distort the
    historical evaluation. ``legacy_peak_temp`` exists only for migrated records
    that lack a trajectory and stays ``None`` for v2-native episodes.

    ``decision_id`` is the authoritative, stable reference to the single
    controller decision this outcome scores. It is taken from the typed
    ``DecisionContext`` at open and stays constant across continue / snapshot /
    restore / materialisation. It uses the central :data:`DecisionId` identity
    alias (no second free string semantics) and carries no entity name.
    """

    episode_id: str
    learning_zone_id: str
    decision_id: DecisionId
    episode_schema_version: int
    builder_version: int
    classifier_version: int
    start_ts: datetime
    end_ts: datetime
    regime: Regime
    reliability: float
    start_temp: float
    end_temp: float
    target: float
    comfort_tolerance_at_start: float
    reason: EpisodeReason
    controller: ControllerKind
    trajectory: Trajectory
    time_quality: DataQuality = DataQuality.OK
    confounder_flags: tuple[str, ...] = ()
    legacy_peak_temp: Optional[float] = None

    def __post_init__(self) -> None:
        s, e = _validate_common(
            self.episode_id, self.learning_zone_id,
            self.start_ts, self.end_ts, self.reliability,
        )
        object.__setattr__(self, "start_ts", s)
        object.__setattr__(self, "end_ts", e)
        require_nonempty(self.decision_id, "decision_id")
        if not isinstance(self.reason, EpisodeReason):
            raise ValueError("reason must be an EpisodeReason")
        if not isinstance(self.controller, ControllerKind):
            raise ValueError("controller must be a ControllerKind")
        if not isinstance(self.trajectory, Trajectory):
            raise ValueError("OutcomeEpisode requires a Trajectory")
