"""Deterministic multi-cycle scenario driver for Learning runtime tests (no real time)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput,
    DecisionType,
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
    RuntimeCycleInput,
    ScheduleTarget,
)

T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
COMFORT = "2025-01-01T07:00:00+00:00"

# Default clock for scenario helpers — deterministic anchor at T0.
_DEFAULT_CLOCK = lambda: T0.isoformat()


def runtime(mode=LearningRuntimeMode.SHADOW, **kw):
    if "clock" not in kw:
        kw["clock"] = _DEFAULT_CLOCK
    return LearningRuntime(LearningRuntimeConfig(mode=mode, startup_grace_cycles=0), **kw)


def step(rt, i, temp, *, heating, zone="lz", target=21.0, schedule=True,
         indoor_quality=DataQuality.OK, window_open=None, legacy_preheat=None,
         decision_type=DecisionType.NORMAL, step_min=5):
    ts = (T0 + timedelta(minutes=i * step_min)).isoformat()
    setp = (target + 3.0) if heating else (target - 4.0)
    indoor = None if temp is None else Measurement(temp, indoor_quality)
    inp = RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=indoor,
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=target)
        if schedule else None,
        controller_decision=ControllerDecisionInput(decision_type=decision_type, target_c=target,
                                                    trv_setpoint_c=setp),
        window_open=window_open, legacy_preheat_start_utc=legacy_preheat)
    return rt.run_cycle(inp)


def heating_ramp_then_settle(rt, *, zone="lz", target=21.0, start=18.0, rise=0.5,
                             ramp=7, settle=5, legacy_preheat=None):
    """Drive a full heating ramp to target, then a settle/cooling tail."""
    results = []
    temp = start
    i = 0
    for _ in range(ramp):
        results.append(step(rt, i, round(temp, 3), heating=True, zone=zone, target=target,
                            legacy_preheat=legacy_preheat))
        temp = min(target, temp + rise)
        i += 1
    for _ in range(settle):
        results.append(step(rt, i, round(temp, 3), heating=False, zone=zone, target=target))
        temp = max(start, temp - 0.4)
        i += 1
    return results
