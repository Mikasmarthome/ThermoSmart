"""Typed event schemas for Learning (pure Python, no Home Assistant).

The in-memory events the builders and the forecast model work with: the end of a
heating drive and the forecast decision / evaluation pair. They contain only
measurements, observed states and controller decisions — never derived values.
A numeric ``0.0`` is always a valid value (no ``x or default`` semantics).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .contracts import (
    Measurement,
    require_nonempty,
    require_utc,
)


class DriveEndSource(Enum):
    """Where the end-of-heating-drive signal came from."""

    SETPOINT_REDUCTION = "setpoint_reduction"
    VALVE_CLOSING = "valve_closing"
    TPI_OFF_TRANSITION = "tpi_off_transition"
    HVAC_ACTION_OFF = "hvac_action_off"
    CENTRAL_HEAT_SOURCE_OFF = "central_heat_source_off"
    INFERRED_HEAT_INPUT_END = "inferred_heat_input_end"


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

