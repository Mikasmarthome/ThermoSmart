"""Typed raw record schemas for LE 2.0 (pure Python, no Home Assistant).

Raw records contain ONLY measurements, observed states, controller decisions
and immutable events — never derived values (``delta``, ``heat_rate``,
``norm_heat_rate``, ``setpoint_excess``, ``efficiency`` …). Derived values are
computed at read time by the (later) FeatureExtractor.

TRV-only minimal setup is a hard invariant: environmental and telemetry data
(outdoor temperature, weather, forecast, wind, solar, humidity, rain, window
sensor, valve opening, HVAC action, a separate room sensor) are NEVER generally
required. Only stable identity, a UTC timestamp and the TRV/controller value
essential to the specific record are mandatory. A numeric ``0.0`` is always a
valid value (no ``x or default`` semantics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .contracts import (
    Measurement,
    require_nonempty,
    require_utc,
)


class RawTrackName(Enum):
    """Stable raw track identifiers used by the (later) registry/storage.

    Window cooling is intentionally absent: it is exclusively a materialised
    episode (``EpisodeType.WINDOW_COOLING``), never a raw record track.
    """

    ROOM = "room"
    TRV = "trv"
    HEATING_DRIVE_END = "heating_drive_end"
    FORECAST_DECISION = "forecast_decision"
    FORECAST_EVALUATION = "forecast_evaluation"
    MANUAL_CORRECTION = "manual_correction"


class DriveEndSource(Enum):
    """Where the end-of-heating-drive signal came from."""

    SETPOINT_REDUCTION = "setpoint_reduction"
    VALVE_CLOSING = "valve_closing"
    TPI_OFF_TRANSITION = "tpi_off_transition"
    HVAC_ACTION_OFF = "hvac_action_off"
    CENTRAL_HEAT_SOURCE_OFF = "central_heat_source_off"
    INFERRED_HEAT_INPUT_END = "inferred_heat_input_end"


class CorrectionActionType(Enum):
    SETPOINT_CHANGE = "setpoint_change"
    MODE_CHANGE = "mode_change"
    OVERRIDE = "override"


class CorrectionSource(Enum):
    UI = "ui"
    AUTOMATION = "automation"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


# -- room observation ---------------------------------------------------------

@dataclass(frozen=True)
class RoomObservation:
    """A room-temperature observation for a zone.

    Core requirement: a usable room temperature exists (from a TRV that reports
    it, or a separate room sensor — the schema is source-agnostic). If neither
    is available, no RoomObservation is produced; the field is therefore
    mandatory here rather than optional. All environmental fields are optional.
    """

    learning_zone_id: str
    ts: datetime
    indoor_temp: float
    target: float

    indoor_humidity: Optional[Measurement] = None
    outdoor_temp: Optional[Measurement] = None
    outdoor_humidity: Optional[Measurement] = None
    wind_speed: Optional[Measurement] = None
    solar_radiation: Optional[Measurement] = None
    rain: Optional[Measurement] = None
    forecast_high: Optional[Measurement] = None

    active_control: bool = False
    window_open: Optional[bool] = None
    preheat_active: bool = False
    heating_failure: bool = False
    vacation: bool = False
    summer_mode: bool = False
    control_reason: Optional[str] = None
    schedule_period: Optional[str] = None

    def __post_init__(self) -> None:
        require_nonempty(self.learning_zone_id, "learning_zone_id")
        object.__setattr__(self, "ts", require_utc(self.ts, "ts"))
        if self.indoor_temp is None or self.target is None:
            raise ValueError("RoomObservation requires indoor_temp and target")


# -- TRV observation ----------------------------------------------------------

@dataclass(frozen=True)
class TrvObservation:
    """A single TRV's reported setpoint within a zone.

    A zone may bind several TRVs (``trv_binding_id`` distinguishes them).
    ``indoor_temp`` is optional because not every TRV reports a temperature.
    """

    learning_zone_id: str
    ts: datetime
    trv_binding_id: str
    trv_setpoint: float
    target: float

    indoor_temp: Optional[Measurement] = None
    valve_opening: Optional[Measurement] = None
    outdoor_temp: Optional[Measurement] = None
    wind_speed: Optional[Measurement] = None
    solar_radiation: Optional[Measurement] = None

    def __post_init__(self) -> None:
        require_nonempty(self.learning_zone_id, "learning_zone_id")
        require_nonempty(self.trv_binding_id, "trv_binding_id")
        object.__setattr__(self, "ts", require_utc(self.ts, "ts"))
        if self.trv_setpoint is None or self.target is None:
            raise ValueError("TrvObservation requires trv_setpoint and target")


# -- heating drive end event --------------------------------------------------

@dataclass(frozen=True)
class HeatingDriveEndEvent:
    """Marks the end of active heat input (abstracted over its source)."""

    learning_zone_id: str
    ts: datetime
    event_id: str
    source: DriveEndSource
    source_confidence: float

    trv_binding_id: Optional[str] = None
    setpoint_before: Optional[float] = None
    setpoint_after: Optional[float] = None
    valve_before: Optional[Measurement] = None
    valve_after: Optional[Measurement] = None
    controller_state: Optional[str] = None
    tpi_state: Optional[str] = None

    def __post_init__(self) -> None:
        require_nonempty(self.learning_zone_id, "learning_zone_id")
        require_nonempty(self.event_id, "event_id")
        object.__setattr__(self, "ts", require_utc(self.ts, "ts"))
        if not isinstance(self.source, DriveEndSource):
            raise ValueError("source must be a DriveEndSource")
        if not 0.0 <= self.source_confidence <= 1.0:
            raise ValueError("source_confidence must be within [0.0, 1.0]")


# -- forecast events (immutable, decision and evaluation kept separate) -------

@dataclass(frozen=True)
class ForecastDecisionEvent:
    """A forecast-influenced control decision. Immutable; never mutated."""

    learning_zone_id: str
    ts: datetime
    decision_id: str
    decision_target: float
    raw_target: float
    forecast_high: float
    suppression: float

    current_temp_at_decision: Optional[float] = None
    outdoor_temp: Optional[Measurement] = None

    def __post_init__(self) -> None:
        require_nonempty(self.learning_zone_id, "learning_zone_id")
        require_nonempty(self.decision_id, "decision_id")
        object.__setattr__(self, "ts", require_utc(self.ts, "ts"))
        if not 0.0 <= self.suppression <= 1.0:
            raise ValueError("suppression must be within [0.0, 1.0]")


@dataclass(frozen=True)
class ForecastEvaluationEvent:
    """Evaluation of an earlier decision. Linked only via ``decision_id``."""

    learning_zone_id: str
    ts: datetime
    evaluation_id: str
    decision_id: str
    observed_temp_at_eval: float

    def __post_init__(self) -> None:
        require_nonempty(self.learning_zone_id, "learning_zone_id")
        require_nonempty(self.evaluation_id, "evaluation_id")
        require_nonempty(self.decision_id, "decision_id")
        object.__setattr__(self, "ts", require_utc(self.ts, "ts"))
        if self.observed_temp_at_eval is None:
            raise ValueError("ForecastEvaluationEvent requires observed_temp_at_eval")


# -- manual correction event --------------------------------------------------

@dataclass(frozen=True)
class ManualCorrectionEvent:
    """A user correction. Magnitude and direction are derived at read time."""

    learning_zone_id: str
    ts: datetime
    event_id: str
    action_type: CorrectionActionType
    prior_target: float
    new_target: float
    source: CorrectionSource

    def __post_init__(self) -> None:
        require_nonempty(self.learning_zone_id, "learning_zone_id")
        require_nonempty(self.event_id, "event_id")
        object.__setattr__(self, "ts", require_utc(self.ts, "ts"))
        if not isinstance(self.action_type, CorrectionActionType):
            raise ValueError("action_type must be a CorrectionActionType")
        if not isinstance(self.source, CorrectionSource):
            raise ValueError("source must be a CorrectionSource")
        if self.prior_target is None or self.new_target is None:
            raise ValueError("ManualCorrectionEvent requires prior_target and new_target")
