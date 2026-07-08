"""Authoritative JSON-compatible serialization for Learning raw records.

One serializer, no competing implementations. Rules:

* datetimes -> canonical UTC string ``YYYY-MM-DDTHH:MM:SS.mmmZ``;
* enums -> their stable ``.value`` string;
* ``Measurement`` -> ``{"value": <float|null>, "quality": <str>}`` (so a
  configured-but-unavailable value survives), or ``null`` when the field is not
  configured at all;
* ``0.0`` / ``False`` / empty optionals are preserved;
* load validates by reconstructing the dataclass (which enforces its invariants);
* unknown future fields are ignored on decode but reported via
  :func:`unknown_raw_fields`;
* no pickle / no Python-specific persistence.
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from ..contracts import DataQuality, Measurement, require_utc
from ..raw_schemas import (
    CorrectionActionType,
    CorrectionSource,
    DriveEndSource,
    ForecastDecisionEvent,
    ForecastEvaluationEvent,
    HeatingDriveEndEvent,
    ManualCorrectionEvent,
    RawTrackName,
    RoomObservation,
    TrvObservation,
)


class SerializationError(Exception):
    """Raised on an invalid payload during decode/encode."""


_SCHEMA_FOR_TRACK: dict[RawTrackName, type] = {
    RawTrackName.ROOM: RoomObservation,
    RawTrackName.TRV: TrvObservation,
    RawTrackName.HEATING_DRIVE_END: HeatingDriveEndEvent,
    RawTrackName.FORECAST_DECISION: ForecastDecisionEvent,
    RawTrackName.FORECAST_EVALUATION: ForecastEvaluationEvent,
    RawTrackName.MANUAL_CORRECTION: ManualCorrectionEvent,
}


# -- primitive codecs ---------------------------------------------------------

def encode_utc(value: datetime) -> str:
    dt = require_utc(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def decode_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SerializationError(f"invalid UTC timestamp: {value!r}")
    try:
        naive = datetime.fromisoformat(value[:-1])
    except ValueError as err:
        raise SerializationError(f"invalid UTC timestamp: {value!r}") from err
    return naive.replace(tzinfo=timezone.utc)


def encode_measurement(value: Optional[Measurement]) -> Optional[dict]:
    if value is None:
        return None  # field not configured
    return {"value": value.value, "quality": value.quality.value}


def decode_measurement(value: Any) -> Optional[Measurement]:
    if value is None:
        return None
    if not isinstance(value, Mapping) or "quality" not in value:
        raise SerializationError(f"invalid measurement payload: {value!r}")
    try:
        quality = DataQuality(value["quality"])
    except ValueError as err:
        raise SerializationError(f"unknown quality: {value['quality']!r}") from err
    return Measurement(value.get("value"), quality)


# -- per-record encoders ------------------------------------------------------

def _enc_room(r: RoomObservation) -> dict:
    return {
        "learning_zone_id": r.learning_zone_id,
        "ts": encode_utc(r.ts),
        "indoor_temp": r.indoor_temp,
        "target": r.target,
        "indoor_humidity": encode_measurement(r.indoor_humidity),
        "outdoor_temp": encode_measurement(r.outdoor_temp),
        "outdoor_humidity": encode_measurement(r.outdoor_humidity),
        "wind_speed": encode_measurement(r.wind_speed),
        "solar_radiation": encode_measurement(r.solar_radiation),
        "rain": encode_measurement(r.rain),
        "forecast_high": encode_measurement(r.forecast_high),
        "active_control": r.active_control,
        "window_open": r.window_open,
        "preheat_active": r.preheat_active,
        "heating_failure": r.heating_failure,
        "vacation": r.vacation,
        "summer_mode": r.summer_mode,
        "control_reason": r.control_reason,
        "schedule_period": r.schedule_period,
    }


def _dec_room(d: Mapping[str, Any]) -> RoomObservation:
    return RoomObservation(
        learning_zone_id=d["learning_zone_id"],
        ts=decode_utc(d["ts"]),
        indoor_temp=d["indoor_temp"],
        target=d["target"],
        indoor_humidity=decode_measurement(d.get("indoor_humidity")),
        outdoor_temp=decode_measurement(d.get("outdoor_temp")),
        outdoor_humidity=decode_measurement(d.get("outdoor_humidity")),
        wind_speed=decode_measurement(d.get("wind_speed")),
        solar_radiation=decode_measurement(d.get("solar_radiation")),
        rain=decode_measurement(d.get("rain")),
        forecast_high=decode_measurement(d.get("forecast_high")),
        active_control=d.get("active_control", False),
        window_open=d.get("window_open"),
        preheat_active=d.get("preheat_active", False),
        heating_failure=d.get("heating_failure", False),
        vacation=d.get("vacation", False),
        summer_mode=d.get("summer_mode", False),
        control_reason=d.get("control_reason"),
        schedule_period=d.get("schedule_period"),
    )


def _enc_trv(r: TrvObservation) -> dict:
    return {
        "learning_zone_id": r.learning_zone_id,
        "ts": encode_utc(r.ts),
        "trv_binding_id": r.trv_binding_id,
        "trv_setpoint": r.trv_setpoint,
        "target": r.target,
        "indoor_temp": encode_measurement(r.indoor_temp),
        "valve_opening": encode_measurement(r.valve_opening),
        "outdoor_temp": encode_measurement(r.outdoor_temp),
        "wind_speed": encode_measurement(r.wind_speed),
        "solar_radiation": encode_measurement(r.solar_radiation),
    }


def _dec_trv(d: Mapping[str, Any]) -> TrvObservation:
    return TrvObservation(
        learning_zone_id=d["learning_zone_id"],
        ts=decode_utc(d["ts"]),
        trv_binding_id=d["trv_binding_id"],
        trv_setpoint=d["trv_setpoint"],
        target=d["target"],
        indoor_temp=decode_measurement(d.get("indoor_temp")),
        valve_opening=decode_measurement(d.get("valve_opening")),
        outdoor_temp=decode_measurement(d.get("outdoor_temp")),
        wind_speed=decode_measurement(d.get("wind_speed")),
        solar_radiation=decode_measurement(d.get("solar_radiation")),
    )


def _enc_drive_end(r: HeatingDriveEndEvent) -> dict:
    return {
        "learning_zone_id": r.learning_zone_id,
        "ts": encode_utc(r.ts),
        "event_id": r.event_id,
        "source": r.source.value,
        "source_confidence": r.source_confidence,
        "trv_binding_id": r.trv_binding_id,
        "setpoint_before": r.setpoint_before,
        "setpoint_after": r.setpoint_after,
        "valve_before": encode_measurement(r.valve_before),
        "valve_after": encode_measurement(r.valve_after),
        "controller_state": r.controller_state,
        "tpi_state": r.tpi_state,
    }


def _dec_drive_end(d: Mapping[str, Any]) -> HeatingDriveEndEvent:
    try:
        source = DriveEndSource(d["source"])
    except ValueError as err:
        raise SerializationError(f"unknown drive end source: {d.get('source')!r}") from err
    return HeatingDriveEndEvent(
        learning_zone_id=d["learning_zone_id"],
        ts=decode_utc(d["ts"]),
        event_id=d["event_id"],
        source=source,
        source_confidence=d["source_confidence"],
        trv_binding_id=d.get("trv_binding_id"),
        setpoint_before=d.get("setpoint_before"),
        setpoint_after=d.get("setpoint_after"),
        valve_before=decode_measurement(d.get("valve_before")),
        valve_after=decode_measurement(d.get("valve_after")),
        controller_state=d.get("controller_state"),
        tpi_state=d.get("tpi_state"),
    )


def _enc_forecast_decision(r: ForecastDecisionEvent) -> dict:
    return {
        "learning_zone_id": r.learning_zone_id,
        "ts": encode_utc(r.ts),
        "decision_id": r.decision_id,
        "decision_target": r.decision_target,
        "raw_target": r.raw_target,
        "forecast_high": r.forecast_high,
        "suppression": r.suppression,
        "current_temp_at_decision": r.current_temp_at_decision,
        "outdoor_temp": encode_measurement(r.outdoor_temp),
    }


def _dec_forecast_decision(d: Mapping[str, Any]) -> ForecastDecisionEvent:
    return ForecastDecisionEvent(
        learning_zone_id=d["learning_zone_id"],
        ts=decode_utc(d["ts"]),
        decision_id=d["decision_id"],
        decision_target=d["decision_target"],
        raw_target=d["raw_target"],
        forecast_high=d["forecast_high"],
        suppression=d["suppression"],
        current_temp_at_decision=d.get("current_temp_at_decision"),
        outdoor_temp=decode_measurement(d.get("outdoor_temp")),
    )


def _enc_forecast_eval(r: ForecastEvaluationEvent) -> dict:
    return {
        "learning_zone_id": r.learning_zone_id,
        "ts": encode_utc(r.ts),
        "evaluation_id": r.evaluation_id,
        "decision_id": r.decision_id,
        "observed_temp_at_eval": r.observed_temp_at_eval,
    }


def _dec_forecast_eval(d: Mapping[str, Any]) -> ForecastEvaluationEvent:
    return ForecastEvaluationEvent(
        learning_zone_id=d["learning_zone_id"],
        ts=decode_utc(d["ts"]),
        evaluation_id=d["evaluation_id"],
        decision_id=d["decision_id"],
        observed_temp_at_eval=d["observed_temp_at_eval"],
    )


def _enc_manual(r: ManualCorrectionEvent) -> dict:
    return {
        "learning_zone_id": r.learning_zone_id,
        "ts": encode_utc(r.ts),
        "event_id": r.event_id,
        "action_type": r.action_type.value,
        "prior_target": r.prior_target,
        "new_target": r.new_target,
        "source": r.source.value,
    }


def _dec_manual(d: Mapping[str, Any]) -> ManualCorrectionEvent:
    try:
        action_type = CorrectionActionType(d["action_type"])
        source = CorrectionSource(d["source"])
    except (ValueError, KeyError) as err:
        raise SerializationError("invalid manual correction payload") from err
    return ManualCorrectionEvent(
        learning_zone_id=d["learning_zone_id"],
        ts=decode_utc(d["ts"]),
        event_id=d["event_id"],
        action_type=action_type,
        prior_target=d["prior_target"],
        new_target=d["new_target"],
        source=source,
    )


_ENCODERS = {
    RawTrackName.ROOM: _enc_room,
    RawTrackName.TRV: _enc_trv,
    RawTrackName.HEATING_DRIVE_END: _enc_drive_end,
    RawTrackName.FORECAST_DECISION: _enc_forecast_decision,
    RawTrackName.FORECAST_EVALUATION: _enc_forecast_eval,
    RawTrackName.MANUAL_CORRECTION: _enc_manual,
}

_DECODERS = {
    RawTrackName.ROOM: _dec_room,
    RawTrackName.TRV: _dec_trv,
    RawTrackName.HEATING_DRIVE_END: _dec_drive_end,
    RawTrackName.FORECAST_DECISION: _dec_forecast_decision,
    RawTrackName.FORECAST_EVALUATION: _dec_forecast_eval,
    RawTrackName.MANUAL_CORRECTION: _dec_manual,
}


# -- public API ---------------------------------------------------------------

def encode_raw(track_name: RawTrackName, record: Any) -> dict:
    schema = _SCHEMA_FOR_TRACK.get(track_name)
    if schema is None:
        raise SerializationError(f"no schema for track {track_name!r}")
    if not isinstance(record, schema):
        raise SerializationError(
            f"record is {type(record).__name__}, expected {schema.__name__}"
        )
    return _ENCODERS[track_name](record)


def decode_raw(track_name: RawTrackName, data: Mapping[str, Any]) -> Any:
    decoder = _DECODERS.get(track_name)
    if decoder is None:
        raise SerializationError(f"no decoder for track {track_name!r}")
    if not isinstance(data, Mapping):
        raise SerializationError("record payload must be a mapping")
    try:
        return decoder(data)
    except SerializationError:
        raise
    except (KeyError, ValueError, TypeError) as err:
        raise SerializationError(f"invalid {track_name.value} payload: {err}") from err


def unknown_raw_fields(track_name: RawTrackName, data: Mapping[str, Any]) -> tuple[str, ...]:
    """Return payload keys that are not part of the track's current schema."""
    schema = _SCHEMA_FOR_TRACK.get(track_name)
    if schema is None:
        raise SerializationError(f"no schema for track {track_name!r}")
    known = {f.name for f in fields(schema)}
    return tuple(sorted(k for k in data.keys() if k not in known))
