"""Shared builders for LE 2.0 shadow runtime tests."""
from __future__ import annotations

from typing import Optional

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput,
    DecisionType,
    RuntimeCycleInput,
    ScheduleTarget,
)

COMFORT_TIME = "2026-06-22T07:00:00+00:00"


class MemoryStore:
    """In-memory injected async store (no HA)."""

    def __init__(self, data=None, *, fail_load=False, fail_save=False):
        self.data = data
        self.saves = 0
        self._fail_load = fail_load
        self._fail_save = fail_save

    async def async_load(self):
        if self._fail_load:
            raise IOError("load boom")
        return self.data

    async def async_save(self, data):
        if self._fail_save:
            raise IOError("save boom")
        self.data = data
        self.saves += 1


def cycle_input(ts, *, zone="lr", target=21.0, setpoint=21.0, indoor=19.0,
                indoor_quality=DataQuality.OK, schedule=True,
                decision_type=DecisionType.NORMAL, boost_offset=None, boost_active=False,
                preheat_active=False, legacy_preheat=None, legacy_boost=None,
                sent_commands=(), manual_corrections=(), window_open=None,
                heating_failure=False, superseded=False, decision_target=None,
                forecast_high=None) -> RuntimeCycleInput:
    indoor_m = None if indoor is None else Measurement(indoor, indoor_quality)
    sched = ScheduleTarget(comfort_time_utc=COMFORT_TIME, comfort_temperature_c=21.0) \
        if schedule else None
    dt = decision_target if decision_target is not None else target
    return RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setpoint, indoor_temp=indoor_m,
        schedule=sched,
        controller_decision=ControllerDecisionInput(
            decision_type=decision_type, target_c=dt, trv_setpoint_c=setpoint,
            boost_offset_c=boost_offset, boost_active=boost_active,
            preheat_active=preheat_active, superseded=superseded),
        window_open=window_open, heating_failure=heating_failure,
        forecast_high=forecast_high, sent_commands=tuple(sent_commands),
        manual_corrections=tuple(manual_corrections),
        legacy_preheat_start_utc=legacy_preheat, legacy_boost_offset_c=legacy_boost)
