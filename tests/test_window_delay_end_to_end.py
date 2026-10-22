"""End-to-end window open delay: real contact sensor -> real listener ->
real coordinator refresh -> real _apply_temperature -> climate.set_temperature.

Reported symptom: with Window Open Delay = 5 min and window temperature 5 C the
radiator dropped to 5 C practically immediately after a window was opened.
These tests drive the whole path, not just the detector.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.thermosmart.coordinator import ThermoSmartCoordinator
from custom_components.thermosmart.trv_control import _DispatchStats
from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_coordinator, make_state

_T0 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
_CONTACT = "binary_sensor.window"
_TRV = "climate.test_trv"
_TEMP = "sensor.test_temp"


class _E2ECoordinator(ThermoSmartCoordinator):
    """Stubs only maintenance / non-setpoint side effects; the decision
    pipeline and the real _apply_temperature actuator path run unchanged."""

    async def _async_apply_quirks(self, cfg): pass
    async def _watchdog_hvac(self, cfg, recommendation): pass
    async def _async_calibrate_trvs(self, cfg, recommendation): pass
    async def _async_set_valve_percent(self, cfg, duty): return _DispatchStats()
    async def _apply_frost_protection(self, cfg): pass
    async def _async_valve_maintenance(self, cfg, recommendation): pass
    async def _async_write_external_temp(self, cfg, recommendation): pass
    async def _async_manage_temp_source(self, cfg, recommendation): pass
    async def _async_observe_trv_setpoints(self, cfg, recommendation, weather): pass
    def _check_boost_outcome(self, cfg): pass
    def _check_heating_failure(self, recommendation): pass


def _trv_state(setpoint):
    return make_state("heat", {
        "temperature": setpoint, "current_temperature": 20.0,
        "min_temp": 5.0, "max_temp": 30.0, "target_temp_step": 0.5,
        "hvac_modes": ["heat", "off"],
    })


class _Rig:
    def __init__(self, *, open_delay=5, contact=True, temps=None, normal=21.0):
        self.clock = FakeClock(_T0)
        base = make_coordinator({
            "window_sensors": [_CONTACT] if contact else [],
            "window_open_delay": open_delay,
            "window_close_delay": 2,
            "window_open_temp": 5.0,
            "temp_sensors": [_TEMP],
            "climate_entities": [_TRV],
            "schedule_enabled": False,
        }, clock=self.clock)
        with patch("homeassistant.helpers.frame.report_usage"):
            self.coord = _E2ECoordinator(
                base.hass, base.entry, base.weather_engine, base.learning_engine,
                clock=self.clock,
            )
        self.coord.learning_engine.async_get_base_target = AsyncMock(return_value=normal)
        self.coord._active_control = True
        self.contact = ("off", _T0 - timedelta(days=1))
        self.temp = 21.0
        self.setpoint = normal
        self.calls: list[float] = []
        h = self.coord.hass
        h.states.get = MagicMock(side_effect=self._get)
        h.async_create_task = lambda c: getattr(c, "close", lambda: None)()

        async def _call(domain, service, data, blocking=False, **kw):
            if domain == "climate" and service == "set_temperature":
                self.calls.append(data["temperature"])
                self.setpoint = data["temperature"]
        h.services.async_call = AsyncMock(side_effect=_call)

        captured = {}
        def _track(hass, ids, cb):
            captured[tuple(ids)] = cb
            return lambda: None
        with patch("custom_components.thermosmart.coordinator.async_track_state_change_event",
                   side_effect=_track), \
             patch("custom_components.thermosmart.coordinator.async_call_later",
                   side_effect=lambda hass, d, cb: (lambda: None)):
            self.coord.setup_event_listeners()
        self.listener = next((cb for e, cb in captured.items() if _CONTACT in e), None)

    def _get(self, eid):
        if eid == _CONTACT:
            s = make_state(self.contact[0]); s.last_changed = self.contact[1]; return s
        if eid == _TEMP:
            return make_state(str(self.temp))
        if eid == _TRV:
            return _trv_state(self.setpoint)
        return None

    def open_window(self):
        old = self.contact[0]
        self.contact = ("on", self.clock.now_utc())
        ev = MagicMock()
        ev.data = {"entity_id": _CONTACT, "old_state": MagicMock(state=old),
                   "new_state": MagicMock(state="on")}
        self.listener(ev)

    def wait(self, *, minutes=0, seconds=0):
        self.clock.advance(minutes * 60 + seconds)

    async def refresh(self):
        return await self.coord._async_update_data()


class TestContactDelayEndToEnd:
    async def test_target_stays_until_open_delay_elapsed_then_drops_to_window_temp(self):
        rig = _Rig(open_delay=5)
        await rig.refresh()
        rig.calls.clear()

        rig.open_window()
        data = await rig.refresh()                      # t = 0
        assert data["zone"]["window_open"] is False
        assert 5.0 not in rig.calls

        rig.wait(minutes=1)
        data = await rig.refresh()
        assert data["zone"]["window_open"] is False
        assert 5.0 not in rig.calls

        rig.wait(minutes=3, seconds=59)                 # t = 4:59
        data = await rig.refresh()
        assert data["zone"]["window_open"] is False
        assert 5.0 not in rig.calls

        rig.wait(seconds=1)                             # t = 5:00
        data = await rig.refresh()
        assert data["zone"]["window_open"] is True
        assert rig.calls and rig.calls[-1] == 5.0

    async def test_falling_room_temperature_does_not_bypass_the_contact_delay(self):
        rig = _Rig(open_delay=5)
        await rig.refresh()
        rig.calls.clear()
        rig.open_window()
        for _ in range(12):                              # 12 min-ish of fast cooling, 20 s apart
            rig.temp -= 0.3
            rig.wait(seconds=20)
            data = await rig.refresh()
            assert data["zone"]["window_open"] is False, (rig.clock.now_utc() - _T0)
        assert 5.0 not in rig.calls


class TestWithoutContactSensor:
    """No window sensor configured: drop-based (slope) detection is the only
    window detection and, by design, has no open delay. This characterises
    that behaviour end to end so it is not mistaken for a delay bug."""

    async def test_fast_cooling_eventually_triggers_window_temp_without_open_delay(self):
        rig = _Rig(open_delay=5, contact=False)
        await rig.refresh()
        rig.calls.clear()
        first_trigger = None
        for i in range(1, 11):
            rig.temp -= 0.15                 # 0.15 C per minute, e.g. window opened
            rig.wait(minutes=1)
            data = await rig.refresh()
            if data["zone"]["window_open"] and first_trigger is None:
                first_trigger = i
        assert first_trigger is not None
        assert rig.calls and rig.calls[-1] == 5.0


class TestConfiguredDelayReachesRuntime:
    """Options-flow value (stored in entry.options as a float by the
    NumberSelector) overrides the original entry.data value at runtime."""

    async def test_options_value_overrides_data_value_including_zero(self):
        rig = _Rig(open_delay=0)                       # entry.data says 0
        rig.coord.entry.options = {"window_open_delay": 5.0}   # options flow saved 5
        await rig.refresh()
        rig.calls.clear()
        rig.open_window()
        data = await rig.refresh()
        assert data["zone"]["window_open"] is False
        assert 5.0 not in rig.calls
        rig.wait(minutes=5)
        data = await rig.refresh()
        assert data["zone"]["window_open"] is True
