"""Home Assistant shadow integration shell for LE 2.0 (Phase 17C).

The ONLY glue between the live ThermoSmart coordinator and the passive LE-2.0
runtime. It builds a typed ``RuntimeCycleInput`` from values the coordinator has
already computed, runs the passive shadow cycle behind a hard guard, and stores
only diagnostics. It NEVER calls a service, never mutates a control value, and
never lets an exception reach the heating path.

This module imports Home Assistant (it is the shell, not the pure core); it is
deliberately NOT exported from ``runtime/__init__`` so the pure runtime stays
importable without HA.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping, Optional

from ...const import TPI_MAX_BOOST_CELSIUS
from ..contracts import DataQuality, Measurement
from .capture import ControllerDecisionInput, DecisionType, RuntimeCycleInput, ScheduleTarget
from .ha_store import HomeAssistantStoreAdapter
from .lifecycle import (
    CoordinatorBridge,
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)

_LOGGER = logging.getLogger(__name__)
_ERROR_LOG_THROTTLE = 50  # log at most 1 of every N identical errors


def _zone_segment(zone_id: str) -> str:
    """Non-identifying, stable token for the store key (never an entity name)."""
    return hashlib.sha256(zone_id.encode("utf-8")).hexdigest()[:16]


def _utcnow_iso() -> str:
    from homeassistant.util import dt as dt_util
    return dt_util.utcnow().isoformat()


def _decision_type(recommendation: Mapping[str, Any]) -> DecisionType:
    try:
        if recommendation.get("preheat_minutes", 0) and recommendation.get("preheat_minutes", 0) > 0:
            return DecisionType.PREHEAT
        if recommendation.get("boost_active"):
            return DecisionType.BOOST
        if recommendation.get("window_open") or recommendation.get("is_summer"):
            return DecisionType.IDLE
        return DecisionType.NORMAL
    except Exception:
        return DecisionType.UNKNOWN


def build_runtime_cycle_input(zone_id: str, recommendation: Mapping[str, Any], *,
                              weather: Optional[Mapping[str, Any]] = None,
                              schedule_comfort_time_utc: Optional[str] = None,
                              schedule_comfort_temperature_c: Optional[float] = None,
                              ts_iso: Optional[str] = None,
                              heating_failure: bool = False) -> RuntimeCycleInput:
    """Materialise a typed cycle input from already-computed coordinator values.

    Defensive: only reads present values, never invents a temperature, keeps 0.0,
    and uses Measurement/None for missing data. Pure given its inputs.
    """
    rec = recommendation or {}
    ts = ts_iso or _utcnow_iso()
    target = rec.get("effective_target")
    if target is None:
        target = rec.get("adjusted_target")
    setpoint = rec.get("trv_setpoint")
    current = rec.get("current_temp")
    indoor = Measurement(current, DataQuality.OK) if current is not None else None

    demand = None
    duty = rec.get("tpi_duty_cycle")
    if duty is not None:
        demand = duty > 0.0
    elif setpoint is not None and current is not None:
        demand = setpoint > current + 0.3

    boost_offset = None
    if setpoint is not None and target is not None and setpoint > target:
        boost_offset = min(setpoint - target, TPI_MAX_BOOST_CELSIUS)

    outdoor = None
    if weather:
        ot = weather.get("outdoor_temp") or weather.get("temperature")
        if ot is not None:
            outdoor = Measurement(ot, DataQuality.OK)

    decision_type = _decision_type(rec)
    return RuntimeCycleInput(
        zone_id=zone_id, ts=ts, target_c=target, trv_setpoint_c=setpoint, indoor_temp=indoor,
        schedule=ScheduleTarget(comfort_time_utc=schedule_comfort_time_utc,
                                comfort_temperature_c=schedule_comfort_temperature_c)
        if schedule_comfort_time_utc and schedule_comfort_temperature_c is not None else None,
        mode=rec.get("mode"), controller_demand=demand,
        controller_decision=ControllerDecisionInput(
            decision_type=decision_type, target_c=target, trv_setpoint_c=setpoint,
            boost_offset_c=boost_offset, boost_active=bool(rec.get("boost_active")),
            preheat_active=bool(rec.get("preheat_minutes", 0)),
            mode=rec.get("mode")),
        window_open=rec.get("window_open"), heating_failure=heating_failure,
        outdoor_temp=outdoor, forecast_high=rec.get("forecast_high"))


class LearningShadowController:
    """Owns the passive LE-2.0 runtime for one config entry / zone.

    Every public method is guarded: a learning/store/model failure is counted and
    logged (throttled), never raised. The existing heating control is untouched.
    """

    def __init__(self, hass: Any, zone_id: str, *, store: Any = None,
                 mode: LearningRuntimeMode = LearningRuntimeMode.SHADOW) -> None:
        self._zone = zone_id
        self._enabled = True
        self._errors = 0
        self._error_signatures: dict[str, int] = {}
        self._last_result = None
        adapter = store
        if adapter is None:
            try:
                adapter = HomeAssistantStoreAdapter(hass, _zone_segment(zone_id))
            except Exception:  # store construction must never break setup
                adapter = None
        self._runtime = LearningRuntime(
            LearningRuntimeConfig(mode=mode), store=adapter, clock=_utcnow_iso)

    @property
    def runtime(self) -> LearningRuntime:
        return self._runtime

    @property
    def errors(self) -> int:
        return self._errors

    async def async_setup(self) -> bool:
        try:
            return await self._runtime.async_setup()
        except Exception as err:  # learning setup failure must not fail the entry
            self._record_error("setup", err)
            self._enabled = False
            return False

    def observe_safe(self, recommendation: Mapping[str, Any], *,
                     weather: Optional[Mapping[str, Any]] = None,
                     schedule_comfort_time_utc: Optional[str] = None,
                     schedule_comfort_temperature_c: Optional[float] = None,
                     heating_failure: bool = False) -> None:
        """Run one passive shadow cycle. Never raises, never controls."""
        if not self._enabled:
            return
        try:
            inp = build_runtime_cycle_input(
                self._zone, recommendation, weather=weather,
                schedule_comfort_time_utc=schedule_comfort_time_utc,
                schedule_comfort_temperature_c=schedule_comfort_temperature_c,
                heating_failure=heating_failure)
            self._last_result = CoordinatorBridge(self._runtime).process(inp)
        except Exception as err:
            self._record_error("cycle", err)

    async def async_save_if_due(self) -> None:
        if not self._enabled:
            return
        try:
            await self._runtime.async_save_if_due()
        except Exception as err:
            self._record_error("save", err)

    async def async_flush(self) -> None:
        try:
            await self._runtime.async_flush()
        except Exception as err:
            self._record_error("flush", err)

    async def async_unload(self) -> None:
        self._enabled = False
        try:
            await self._runtime.async_unload()
        except Exception as err:
            self._record_error("unload", err)

    def diagnostics(self) -> dict:
        h = self._runtime.health()
        return {
            "mode": h.mode, "initialized": h.initialized, "dirty": h.dirty,
            "last_cycle_ts": h.last_cycle_ts, "last_save": h.last_save,
            "open_decisions": h.open_decisions, "open_comparisons": h.open_comparisons,
            "model_update_total": h.model_update_total,
            "model_update_counts": dict(h.model_update_counts),
            "prediction_snapshots": h.prediction_snapshots,
            "learning_errors": self._errors, "enabled": self._enabled,
        }

    def _record_error(self, kind: str, err: Exception) -> None:
        self._errors += 1
        sig = f"{kind}:{type(err).__name__}"
        n = self._error_signatures.get(sig, 0)
        self._error_signatures[sig] = n + 1
        if n % _ERROR_LOG_THROTTLE == 0:  # throttle identical errors
            _LOGGER.warning("ThermoSmart LE2 shadow %s error (zone hidden): %s", kind,
                            type(err).__name__)
