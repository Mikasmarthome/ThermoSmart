"""Observation mode must not write to the controlled devices.

HACS review of ThermoSmart: the README promises that Observation mode (Active
Control off) changes nothing on the TRVs and makes no service calls. Two code
paths broke that promise:

* ``_reset_valve_opening_degree`` reset every SONOFF ``valve_opening_degree``
  below 100 back to 100 on every setup and refresh — overwriting limits the
  user set on purpose. It only ever served installs of v1.0.0-v1.0.3 (which
  wrote the TPI duty-cycle there) and has been removed without replacement.
* ``external_temperature_input`` was written regardless of Active Control. It
  is now only written while Active Control is on.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import make_coordinator, make_state

_CLIMATE = "climate.trv"
_EXT_TEMP = "number.trv_external_temperature_input"
_VALVE_LIMIT = "number.trv_valve_opening_degree"


def _coordinator_with_ext_temp(*, active_control: bool):
    coord = make_coordinator({"climate_entities": [_CLIMATE]})
    coord._active_control = active_control
    coord._auto_ext_temp_map = {_CLIMATE: _EXT_TEMP}
    coord._device_profiles = {}
    state = make_state("15.0", {"unit_of_measurement": "°C"})
    coord.hass.states.get = MagicMock(
        side_effect=lambda eid: state if eid == _EXT_TEMP else None
    )
    coord.hass.services.async_call = AsyncMock()
    coord.hass.async_create_task = MagicMock(side_effect=lambda c: getattr(c, "close", lambda: None)())
    return coord


class TestExternalTemperatureInputGating:
    async def test_observation_mode_writes_nothing(self):
        coord = _coordinator_with_ext_temp(active_control=False)

        await coord._async_write_external_temp(coord.zone_cfg, {"current_temp": 21.3})

        coord.hass.services.async_call.assert_not_called()
        coord.hass.async_create_task.assert_not_called()

    async def test_active_control_still_writes_the_room_temperature(self):
        coord = _coordinator_with_ext_temp(active_control=True)

        await coord._async_write_external_temp(coord.zone_cfg, {"current_temp": 21.3})

        coord.hass.services.async_call.assert_called_once_with(
            "number", "set_value",
            {"entity_id": _EXT_TEMP, "value": 21.3},
            blocking=False,
        )

    async def test_switching_active_control_off_stops_the_writes(self):
        coord = _coordinator_with_ext_temp(active_control=True)
        await coord._async_write_external_temp(coord.zone_cfg, {"current_temp": 21.3})
        assert coord.hass.services.async_call.call_count == 1

        coord.set_active_control(False)
        await coord._async_write_external_temp(coord.zone_cfg, {"current_temp": 24.0})

        assert coord.hass.services.async_call.call_count == 1


def _registry_with_sonoff_valve_limit():
    """Entity/device registry fakes for one SONOFF TRVZB with a user-set
    valve_opening_degree limit of 40 %."""
    trv_entry = SimpleNamespace(entity_id=_CLIMATE, device_id="dev1")
    device_entities = [
        SimpleNamespace(entity_id=_VALVE_LIMIT, translation_key="valve_opening_degree",
                        original_name="Valve opening degree"),
        SimpleNamespace(entity_id=_EXT_TEMP, translation_key="external_temperature_input",
                        original_name="External temperature input"),
    ]
    ent_reg = MagicMock()
    ent_reg.async_get = MagicMock(side_effect=lambda eid: trv_entry if eid == _CLIMATE else None)
    dev_reg = MagicMock()
    dev_reg.async_get = MagicMock(
        return_value=SimpleNamespace(manufacturer="SONOFF", model="TRVZB")
    )
    er = MagicMock()
    er.async_get = MagicMock(return_value=ent_reg)
    er.async_entries_for_device = MagicMock(return_value=device_entities)
    dr = MagicMock()
    dr.async_get = MagicMock(return_value=dev_reg)
    return er, dr


class TestValveOpeningDegreeIsNeverWritten:
    def test_the_reset_routine_and_its_state_are_gone(self):
        coord = make_coordinator()
        assert not hasattr(coord, "_reset_valve_opening_degree")
        assert not hasattr(coord, "_valve_reset_done")
        assert not hasattr(coord, "_valve_reset_attempts")

    @pytest.mark.parametrize("active_control", [False, True])
    async def test_device_detection_never_writes_a_user_set_limit(self, active_control):
        coord = make_coordinator({"climate_entities": [_CLIMATE]})
        coord._active_control = active_control
        coord.hass.states.get = MagicMock(
            side_effect=lambda eid: make_state("40") if eid == _VALVE_LIMIT else None
        )
        coord.hass.services.async_call = AsyncMock()
        coord.hass.async_create_task = MagicMock(
            side_effect=lambda c: getattr(c, "close", lambda: None)()
        )
        er, dr = _registry_with_sonoff_valve_limit()

        with patch("custom_components.thermosmart.trv_control.er", er), \
             patch("custom_components.thermosmart.trv_control.dr", dr):
            # setup, and two more times as an integration reload would do
            for _ in range(3):
                await coord.async_detect_device_entities()

        coord.hass.services.async_call.assert_not_called()
        coord.hass.async_create_task.assert_not_called()
        # the entity is still not treated as a direct-valve target either
        assert _VALVE_LIMIT not in coord._auto_valve_map.values()

    async def test_coordinator_refreshes_never_write_the_limit(self):
        """The first refreshes used to retry the reset; none of that remains."""
        import inspect
        from custom_components.thermosmart import coordinator as coordinator_module
        from custom_components.thermosmart import trv_control

        for module in (coordinator_module, trv_control):
            source = inspect.getsource(module)
            assert "_reset_valve_opening_degree" not in source
            assert "_VALVE_RESET" not in source
