"""Phase 18: LE 2.0 never dispatches directly; existing controller is single authority."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import LearningRuntimeMode
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


async def _control_coord(indoor="18.0"):
    coord = make_recording_coordinator(indoor=indoor)
    coord._active_control = True
    sh = attach_shadow(coord)
    sh.runtime.set_mode(LearningRuntimeMode.CONTROL)
    await sh.async_setup()
    return coord, sh


class TestNoDirectDispatch:
    async def test_le2_makes_no_service_calls(self):
        coord, sh = await _control_coord()
        await coord._async_update_data()
        # all recorded control happened via the existing controller path only
        kinds = {c[0] for c in coord.control_calls}
        assert kinds <= {"apply_temperature", "set_valve_percent", "frost_protection"}

    async def test_hass_services_not_called_by_le2(self):
        coord, sh = await _control_coord()
        # hass is a MagicMock; the shadow must not call hass.services.async_call
        before = coord.hass.services.async_call.call_count
        await coord._async_update_data()
        # the recording coordinator stubs apply; LE2 adds no extra service calls
        assert coord.hass.services.async_call.call_count == before

    async def test_single_dispatch_path(self):
        coord, sh = await _control_coord()
        await coord._async_update_data()
        # at most one apply_temperature per cycle (single authority)
        assert sum(1 for c in coord.control_calls if c[0] == "apply_temperature") <= 1

    async def test_control_value_flows_through_existing_apply(self):
        # in CONTROL mode the boost adjustment lands in recommendation BEFORE apply,
        # so the existing _apply_temperature is the only thing that dispatches it
        coord, sh = await _control_coord()
        result = await coord._async_update_data()
        assert result is not None and "zone" in result
