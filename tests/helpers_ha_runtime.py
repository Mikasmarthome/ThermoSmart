"""Harness for LE 2.0 HA shadow-integration tests.

Runs the REAL ThermoSmartCoordinator._async_update_data (with control side-effects
stubbed and recorded) so the production LE-2.0 shadow hook is exercised. Pure
Python + MagicMock hass — runs on Windows and in Docker.
"""
from __future__ import annotations

from unittest.mock import patch as _patch

from custom_components.thermosmart.coordinator import ThermoSmartCoordinator
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)
from custom_components.thermosmart.trv_control import _DispatchStats
from tests.helpers import make_coordinator, make_state, set_hass_states


class RecordingCoordinator(ThermoSmartCoordinator):
    """Coordinator that records (not performs) every control side-effect."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.control_calls: list = []

    async def _reset_valve_opening_degree(self) -> bool:
        return True

    async def _async_apply_quirks(self, cfg):
        pass

    async def _watchdog_hvac(self, cfg, recommendation):
        pass

    async def _async_calibrate_trvs(self, cfg, recommendation):
        pass

    async def _apply_temperature(self, cfg, recommendation):
        self.control_calls.append(("apply_temperature", recommendation.get("trv_setpoint"),
                                   recommendation.get("effective_target")))
        return _DispatchStats()

    async def _async_set_valve_percent(self, cfg, duty):
        self.control_calls.append(("set_valve_percent", duty))
        return _DispatchStats()

    async def _apply_frost_protection(self, cfg):
        self.control_calls.append(("frost_protection",))

    async def _async_valve_maintenance(self, cfg, recommendation):
        pass

    async def _async_write_external_temp(self, cfg, recommendation):
        pass

    async def _async_manage_temp_source(self, cfg, recommendation):
        pass

    async def _async_observe_trv_setpoints(self, cfg, recommendation, weather):
        pass

    def _check_heating_failure(self, recommendation):
        pass


def make_recording_coordinator(zone_cfg_overrides=None, *, indoor="19.0"):
    base = make_coordinator(zone_cfg_overrides)
    with _patch("homeassistant.helpers.frame.report_usage"):
        coord = RecordingCoordinator(base.hass, base.entry, base.weather_engine,
                                     base.learning_engine)
    coord._valve_reset_done = True
    set_hass_states(coord, {"sensor.test_temp": make_state(indoor)})
    return coord


class FakeStore:
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


def attach_shadow(coord, *, store=None, zone_id=None):
    shadow = LearningShadowController(hass=coord.hass, zone_id=zone_id or coord.zone_id,
                                     store=store or FakeStore())
    coord.attach_le2_shadow(shadow)
    return shadow
