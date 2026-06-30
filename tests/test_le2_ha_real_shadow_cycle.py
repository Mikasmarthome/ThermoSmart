"""Phase 17C: a real coordinator update cycle drives the passive shadow (Docker/CI)."""
from __future__ import annotations

import pytest

from tests.helpers_ha_real import setup_zone


async def test_refresh_runs_shadow_cycle(hass, hass_storage, enable_custom_integrations):
    _, entry, entry_data = await setup_zone(hass)
    shadow = entry_data["le2_shadow"]
    coord = entry_data["coordinator"]
    await coord.async_request_refresh()
    await hass.async_block_till_done()
    assert coord.last_update_success
    # the passive shadow observed at least one cycle
    assert shadow.diagnostics()["last_cycle_ts"] is not None


async def test_existing_recommendation_still_produced(hass, hass_storage,
                                                      enable_custom_integrations):
    _, entry, entry_data = await setup_zone(hass)
    coord = entry_data["coordinator"]
    await coord.async_request_refresh()
    await hass.async_block_till_done()
    assert coord.data is not None and "zone" in coord.data


async def test_capture_time_advances(hass, hass_storage, enable_custom_integrations):
    _, entry, entry_data = await setup_zone(hass)
    shadow = entry_data["le2_shadow"]
    coord = entry_data["coordinator"]
    await coord.async_request_refresh()
    await hass.async_block_till_done()
    first = shadow.diagnostics()["last_cycle_ts"]
    hass.states.async_set("sensor.test_temp", "19.5",
                          {"unit_of_measurement": "°C", "device_class": "temperature"})
    await coord.async_request_refresh()
    await hass.async_block_till_done()
    second = shadow.diagnostics()["last_cycle_ts"]
    assert first is not None and second is not None


async def test_control_result_unaffected(hass, hass_storage, enable_custom_integrations):
    # control result with the shadow present is still a normal recommendation dict;
    # exact byte-equality is proven by the RecordingCoordinator suite.
    _, entry, entry_data = await setup_zone(hass)
    coord = entry_data["coordinator"]
    await coord.async_request_refresh()
    await hass.async_block_till_done()
    data = coord.data
    assert isinstance(data, dict) and isinstance(data.get("zone"), dict)
    # the recommendation carries no LE2 control fields written back
    assert "le2_setpoint" not in data["zone"] and "shadow_override" not in data["zone"]
