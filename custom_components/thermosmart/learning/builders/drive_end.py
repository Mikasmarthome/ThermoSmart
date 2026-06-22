"""Pure detection / normalisation of HeatingDriveEndEvent (Phase 6).

Direct signals (valve closing, HVAC off, TPI off, setpoint reduction) carry
higher source confidence than an inferred heat-input end. Near-simultaneous
signals are merged into one canonical event; pure temperature change without a
controller/drive context never invents a drive end. Works TRV-only via
setpoint / controller / TPI transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..contracts import DataQuality, Measurement, require_utc
from ..raw_schemas import DriveEndSource, HeatingDriveEndEvent

# Source confidence per source (direct > inferred).
_SOURCE_CONFIDENCE = {
    DriveEndSource.VALVE_CLOSING: 0.9,
    DriveEndSource.HVAC_ACTION_OFF: 0.9,
    DriveEndSource.CENTRAL_HEAT_SOURCE_OFF: 0.85,
    DriveEndSource.TPI_OFF_TRANSITION: 0.7,
    DriveEndSource.SETPOINT_REDUCTION: 0.6,
    DriveEndSource.INFERRED_HEAT_INPUT_END: 0.4,
}

# Priority for merging near-simultaneous signals (most direct wins).
_SOURCE_PRIORITY = [
    DriveEndSource.VALVE_CLOSING,
    DriveEndSource.HVAC_ACTION_OFF,
    DriveEndSource.CENTRAL_HEAT_SOURCE_OFF,
    DriveEndSource.TPI_OFF_TRANSITION,
    DriveEndSource.SETPOINT_REDUCTION,
    DriveEndSource.INFERRED_HEAT_INPUT_END,
]


@dataclass(frozen=True)
class DriveEndSignals:
    """The before/after controller state used to detect a heat-drive end."""

    ts: datetime
    learning_zone_id: str
    event_id: str
    valve_before: Optional[Measurement] = None
    valve_after: Optional[Measurement] = None
    hvac_before: Optional[str] = None
    hvac_after: Optional[str] = None
    tpi_before: Optional[bool] = None
    tpi_after: Optional[bool] = None
    demand_before: Optional[bool] = None
    demand_after: Optional[bool] = None
    setpoint_before: Optional[float] = None
    setpoint_after: Optional[float] = None
    central_before: Optional[bool] = None
    central_after: Optional[bool] = None
    trv_binding_id: Optional[str] = None


def _valve_value(m: Optional[Measurement]) -> Optional[float]:
    if m is not None and m.value is not None and m.quality in (DataQuality.OK, DataQuality.STALE):
        return m.value
    return None


def detect_drive_end(s: DriveEndSignals) -> Optional[HeatingDriveEndEvent]:
    """Return a single canonical drive-end event, or None if none applies."""
    sources: list[DriveEndSource] = []

    vb, va = _valve_value(s.valve_before), _valve_value(s.valve_after)
    if vb is not None and va is not None and vb > 0 and va <= 0:
        sources.append(DriveEndSource.VALVE_CLOSING)
    if s.hvac_before == "heating" and s.hvac_after not in (None, "heating"):
        sources.append(DriveEndSource.HVAC_ACTION_OFF)
    if s.central_before is True and s.central_after is False:
        sources.append(DriveEndSource.CENTRAL_HEAT_SOURCE_OFF)
    if s.tpi_before is True and s.tpi_after is False:
        sources.append(DriveEndSource.TPI_OFF_TRANSITION)
    if (s.setpoint_before is not None and s.setpoint_after is not None
            and s.setpoint_after < s.setpoint_before):
        sources.append(DriveEndSource.SETPOINT_REDUCTION)
    # demand off alone, with no other signal, is an inferred end (TRV-only)
    if not sources and s.demand_before is True and s.demand_after is False:
        sources.append(DriveEndSource.INFERRED_HEAT_INPUT_END)

    if not sources:
        return None  # pure temperature change never invents a drive end

    canonical = next(src for src in _SOURCE_PRIORITY if src in sources)
    return HeatingDriveEndEvent(
        learning_zone_id=s.learning_zone_id,
        ts=require_utc(s.ts, "ts"),
        event_id=s.event_id,
        source=canonical,
        source_confidence=_SOURCE_CONFIDENCE[canonical],
        trv_binding_id=s.trv_binding_id,
        setpoint_before=s.setpoint_before,
        setpoint_after=s.setpoint_after,
        valve_before=s.valve_before,
        valve_after=s.valve_after,
    )


class DriveEndDetector:
    """Stateful de-duplicator: collapses repeated signals within a window."""

    def __init__(self, dedup_window_s: float = 120.0) -> None:
        self._window = dedup_window_s
        self._last_ts: Optional[datetime] = None

    def detect(self, signals: DriveEndSignals) -> Optional[HeatingDriveEndEvent]:
        event = detect_drive_end(signals)
        if event is None:
            return None
        if self._last_ts is not None and \
                (event.ts - self._last_ts).total_seconds() < self._window:
            return None  # merged into the previous canonical event
        self._last_ts = event.ts
        return event

    def reset(self) -> None:
        self._last_ts = None
