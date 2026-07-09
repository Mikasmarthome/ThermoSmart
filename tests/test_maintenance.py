"""Regression coverage for MaintenanceMixin's Active Control gate.

Real Heating Plausibility Audit finding: _async_valve_maintenance() checked
valve_likely_idle = is_summer or in_vacation without ever checking
active_control, so the weekly valve-exercise routine could dispatch a real
climate.set_temperature call even with Active Control switched off, as long
as the zone was in summer or vacation mode. Fixed by adding a hard
active_control gate alongside the existing early-exit checks.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_coordinator, make_state, set_hass_states

_SUNDAY_3AM = datetime(2025, 1, 12, 3, 0, tzinfo=timezone.utc)  # matches VALVE_MAINTENANCE_WEEKDAY/HOUR


async def _run_maintenance(coord, cfg: dict, rec: dict) -> list[dict]:
    """Run _async_valve_maintenance() and collect any dispatched service calls."""
    calls: list[dict] = []

    async def _svc(domain, service, data, *, blocking=True, **kw):
        calls.append(data)

    coord.hass.services.async_call = _svc
    coord.hass.async_create_task = lambda coro: asyncio.ensure_future(coro)
    set_hass_states(coord, {
        "climate.trv": make_state("heat", {
            "temperature": 19.0, "min_temp": 5.0, "max_temp": 35.0, "target_temp_step": 0.5,
        }),
    })
    coord._device_profiles = {}
    coord._maintenance_running = False
    coord._last_maintenance = None
    coord._clock = FakeClock(_SUNDAY_3AM)

    await coord._async_valve_maintenance(cfg, rec)
    task = coord._maintenance_task
    if task is not None:
        await task
    return calls


def _cfg() -> dict:
    return {
        "climate_entities": ["climate.trv"],
        "valve_maintenance": True,
        "vacation_temp": 12.0,
        "comfort_temp": 21.0,
    }


class TestValveMaintenanceActiveControlGate:
    async def test_active_control_off_in_summer_dispatches_nothing(self):
        coord = make_coordinator()
        coord._is_summer = True
        coord._active_control = False
        calls = await _run_maintenance(coord, _cfg(), {"mode": "auto", "adjusted_target": 21.0})
        assert calls == []
        assert coord._maintenance_task is None

    async def test_active_control_off_in_vacation_dispatches_nothing(self):
        coord = make_coordinator()
        coord._is_summer = False
        coord._active_control = False
        rec = {"mode": "vacation", "adjusted_target": 12.0}
        calls = await _run_maintenance(coord, _cfg(), rec)
        assert calls == []
        assert coord._maintenance_task is None

    async def test_active_control_on_in_summer_still_dispatches(self):
        coord = make_coordinator()
        coord._is_summer = True
        coord._active_control = True
        calls = await _run_maintenance(coord, _cfg(), {"mode": "auto", "adjusted_target": 21.0})
        assert len(calls) == 2  # boost dispatch + return dispatch
        assert calls[0]["entity_id"] == "climate.trv"

    async def test_active_control_on_in_vacation_still_dispatches(self):
        coord = make_coordinator()
        coord._is_summer = False
        coord._active_control = True
        rec = {"mode": "vacation", "adjusted_target": 12.0}
        calls = await _run_maintenance(coord, _cfg(), rec)
        assert len(calls) == 2  # boost dispatch + return dispatch

    async def test_neither_summer_nor_vacation_dispatches_nothing_regardless_of_active_control(self):
        coord = make_coordinator()
        coord._is_summer = False
        coord._active_control = True
        rec = {"mode": "auto", "adjusted_target": 21.0}
        calls = await _run_maintenance(coord, _cfg(), rec)
        assert calls == []
        assert coord._maintenance_task is None
