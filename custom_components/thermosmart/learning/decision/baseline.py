"""Baseline adapter (Phase 19A, pure Python).

Adapts the existing controller's recommendation dictionary into a typed
``ControllerBaselineDecision`` at the architecture boundary, WITHOUT changing how
the deterministic controller computes it. The legacy controller remains the safe
fallback; this only gives it a typed, side-effect-free output contract.
"""
from __future__ import annotations

from typing import Mapping, Optional

from .contracts import ControllerBaselineDecision, ZoneRuntimeInput


def _f(d: Mapping, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return default
    return default


def baseline_from_recommendation(zone_id: str, rec: Mapping,
                                 *, active_control: bool = False) -> ControllerBaselineDecision:
    """Build the typed baseline from an existing recommendation dict (read-only)."""
    target = _f(rec, "effective_target", "adjusted_target", "target")
    # Prefer tpi_baseline_setpoint (written by adjust_recommendation_safe before boost).
    # Falls back to trv_setpoint in shadow mode where the original TPI value is unchanged.
    setpoint = _f(rec, "tpi_baseline_setpoint", "trv_setpoint")
    # Use the coordinator's pre-computed le2 boost offset directly (not reconstructed).
    # boost_offset_c = 0.0 when no LE2 boost is active; positive when boost is applied.
    # This replaces the previous max(0, setpoint-target) which conflated TPI duty with le2.
    boost_offset = _f(rec, "boost_offset_c") or 0.0
    return ControllerBaselineDecision(
        zone_id=zone_id,
        target_c=target,
        trv_setpoint_c=setpoint,
        preheat_minutes=_f(rec, "preheat_minutes", default=0.0) or 0.0,
        boost_offset_c=boost_offset,
        duty_cycle=_f(rec, "tpi_duty_cycle", default=0.0) or 0.0,
        control_reason=str(rec.get("control_reason", "schedule")),
        summer=bool(rec.get("is_summer")),
        window_open=bool(rec.get("window_open")),
        active_control=active_control,
    )


def zone_input_from_recommendation(zone_id: str, ts: str, rec: Mapping,
                                   *, decision_id: Optional[str] = None) -> ZoneRuntimeInput:
    """Build the normalised ZoneRuntimeInput from an existing recommendation dict."""
    indoor = _f(rec, "indoor_temperature", "current_temp", "indoor_temp")
    return ZoneRuntimeInput(
        zone_id=zone_id, ts=ts,
        target_c=_f(rec, "effective_target", "adjusted_target", "target"),
        trv_setpoint_c=_f(rec, "trv_setpoint"),
        indoor_temp_c=indoor, indoor_temp_valid=indoor is not None,
        comfort_temperature_c=_f(rec, "comfort_temperature", "next_comfort_temp", "adjusted_target"),
        comfort_time_utc=rec.get("next_comfort_time_utc") or rec.get("comfort_time_utc"),
        mode=rec.get("mode") or rec.get("effective_mode"),
        controller_demand=rec.get("controller_demand"),
        window_open=bool(rec.get("window_open")), summer=bool(rec.get("is_summer")),
        frost_protection=bool(rec.get("frost_protection")),
        absence=bool(rec.get("absence") or rec.get("away")),
        heating_failure=bool(rec.get("heating_failure")),
        decision_id=decision_id,
        outdoor_temp_c=_f(rec, "outdoor_temperature", "outdoor_temp"),
        forecast_high_c=_f(rec, "forecast_high"),
        early_cutoff_state=rec.get("early_cutoff_state"),
        tpi_valve_direct=bool(rec.get("tpi_valve_direct", False)),
        preheat_minutes_le2=_f(rec, "preheat_minutes") if rec.get("preheat_status") not in (None, "not_available", "unavailable") else None,
        preheat_status=rec.get("preheat_status"),
        deterministic_baseline_preheat_min=_f(rec, "preheat_baseline_minutes"),
        onset_delay_min=_f(rec, "onset_delay_min"),
        onset_delay_status=rec.get("onset_delay_status"),
        temperature_gap_c=_f(rec, "temperature_gap_c"),
        target_temperature_c=_f(rec, "comfort_temperature", "next_comfort_temp"),
        context_time_bucket=rec.get("schedule_period"),
    )
