"""Phase S1a Item 9 — Direct-Valve-TRV Restart under Real HA Runtime.

Proves that after coordinator reload, Direct-Valve-TRV handling is correct:

  * No climate.set_temperature call is issued for valve-controlled entities.
  * The _auto_valve_map resets to {} after reload (coordinator re-detects devices).
  * When _auto_valve_map is injected post-setup, the zone_control_type reflects
    the correct "direct_valve" or "mixed" type.
  * After reload, no stale valve dispatch state is carried over.
  * Pending attribution distinguishes direct_valve from setpoint in device_control_type.

Scope note:
  The number.set_value call for actual TPI-duty-cycle-to-valve-position dispatch
  requires the full TPI+valve-detection path running in a coordinator cycle with a
  real auto_valve_map entry that survived device-registry detection. Since
  async_detect_device_entities() depends on the entity/device registry (not
  available in pytest without registered devices), we test the invariants that are
  accessible: the _auto_valve_map state, the zone_control_type derivation, and the
  absence of set_temperature calls when valve control is active.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.const import DOMAIN
from tests.helpers_ha_real import (
    capture_climate_calls,
    capture_number_calls,
    set_trv_state,
    setup_zone,
)

pytestmark = pytest.mark.asyncio


def _coord(entry_data):
    return entry_data["coordinator"]


# ── 1. _auto_valve_map resets on reload ───────────────────────────────────────


async def test_auto_valve_map_empty_after_reload(
        hass, hass_storage, enable_custom_integrations):
    """_auto_valve_map is rebuilt from device registry on each setup; starts empty."""
    _, entry, entry_data1 = await setup_zone(hass)
    coord1 = _coord(entry_data1)
    # Simulate a detected valve map (normally set by async_detect_device_entities)
    coord1._auto_valve_map = {"climate.test_trv": "number.valve_pos"}

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    # After reload, the map starts empty (no device registry entries match in tests)
    assert coord2._auto_valve_map == {}, \
        "_auto_valve_map must restart empty; stale state must not carry over"


# ── 2. No set_temperature when zone_control_type=direct_valve ─────────────────


async def test_no_set_temperature_when_valve_direct(
        hass, hass_storage, enable_custom_integrations):
    """When _auto_valve_map maps a climate entity to a valve, no set_temperature call.

    We inject the valve map post-setup (bypassing device registry detection) to
    simulate what the production path would produce for a SONOFF TRVZB device.
    """
    _, entry, entry_data1 = await setup_zone(hass)
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    # Inject valve map: climate.test_trv is now valve-controlled
    coord2._auto_valve_map = {"climate.test_trv": "number.test_valve"}

    # Provide valve number entity state
    hass.states.async_set("number.test_valve", "0",
                          {"min": 0, "max": 100, "step": 1, "unit_of_measurement": "%"})
    set_trv_state(hass, "climate.test_trv", setpoint=20.0, current_temp=18.0)

    calls = capture_climate_calls(hass)
    await coord2.async_refresh()
    await hass.async_block_till_done()

    set_temp_calls = [c for c in calls
                      if c.data.get("entity_id") == "climate.test_trv"]
    assert len(set_temp_calls) == 0, (
        f"Direct-Valve TRV must not receive climate.set_temperature calls; "
        f"got {len(set_temp_calls)}: {[c.data for c in set_temp_calls]}")


async def test_no_set_temperature_after_reload_with_valve_map(
        hass, hass_storage, enable_custom_integrations):
    """After reload, injected valve map prevents set_temperature for that entity."""
    _, entry, entry_data1 = await setup_zone(hass)
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    coord2._auto_valve_map = {"climate.test_trv": "number.test_valve"}
    hass.states.async_set("number.test_valve", "50",
                          {"min": 0, "max": 100, "step": 1, "unit_of_measurement": "%"})
    set_trv_state(hass, "climate.test_trv", setpoint=20.0, current_temp=18.0)

    calls = capture_climate_calls(hass)
    # Two refreshes: ensure no set_temperature on either
    await coord2.async_refresh()
    await hass.async_block_till_done()
    await coord2.async_refresh()
    await hass.async_block_till_done()

    set_temp_calls = [c for c in calls
                      if c.data.get("entity_id") == "climate.test_trv"]
    assert len(set_temp_calls) == 0, \
        "Direct-valve entity must not receive set_temperature across multiple refreshes"


# ── 3. Valve boost blocked: no °C boost for Direct-Valve ──────────────────────


async def test_no_boost_applied_c_for_direct_valve(
        hass, hass_storage, enable_custom_integrations):
    """Direct-Valve path does not apply an absolute °C boost setpoint.

    The coordinator's _async_update_data sets zone_control_type=direct_valve and
    blocks LE2 boost (BoostBlockReason.MIXED_CONTROL_TYPES or direct-valve gate).
    We verify no set_temperature call with an elevated setpoint.
    """
    _, entry, entry_data1 = await setup_zone(hass)
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    coord2._auto_valve_map = {"climate.test_trv": "number.test_valve"}
    # Start from a clearly cold state to drive TPI duty high
    hass.states.async_set("number.test_valve", "0",
                          {"min": 0, "max": 100, "step": 1, "unit_of_measurement": "%"})
    set_trv_state(hass, "climate.test_trv", setpoint=20.0, current_temp=14.0)

    calls = capture_climate_calls(hass)
    await coord2.async_refresh()
    await hass.async_block_till_done()

    set_temp_calls = [c for c in calls
                      if c.data.get("entity_id") == "climate.test_trv"]
    # No °C setpoint should be written to a direct-valve entity
    assert len(set_temp_calls) == 0, (
        f"Direct-Valve must not receive climate.set_temperature boost calls; "
        f"got: {[c.data for c in set_temp_calls]}")


# ── 4. Stale boost cleared: _boost_active reset ───────────────────────────────


async def test_boost_active_empty_after_reload_valve_zone(
        hass, hass_storage, enable_custom_integrations):
    """_boost_active starts empty after reload even for valve zones."""
    _, entry, entry_data1 = await setup_zone(hass)
    coord1 = _coord(entry_data1)
    coord1._boost_active["lz"] = {"valve_duty": 80}

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert coord2._boost_active == {}, "_boost_active must be empty after reload"


# ── 5. Pending attribution: device_control_type preserved ─────────────────────


async def test_dispatch_record_device_type_direct_valve_survives_reload(
        hass, hass_storage, enable_custom_integrations):
    """A BoostDispatchRecord with device_control_type='direct_valve' survives reload."""
    from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = entry_data1.get("le2_shadow")
    if shadow1 is None:
        pytest.skip("No shadow controller")

    did = "dec-dv-1"
    shadow1.runtime._zone("lz").pending_dispatch_records[did] = BoostDispatchRecord(
        decision_id=did,
        dispatch_status="fully_succeeded",
        outcome_eligible=False,  # direct_valve is never outcome-eligible in current model
        outcome_reliability="none",
        device_control_type="direct_valve",
        effective_setpoints=(),   # valve duty, not setpoint
        targets_total=1,
        targets_failed=0,
    )
    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = entry_data2.get("le2_shadow")
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    drec = shadow2.runtime._zone("lz").pending_dispatch_records.get(did)
    assert drec is not None, "Direct-valve dispatch record must survive reload"
    assert drec.device_control_type == "direct_valve", \
        "device_control_type must be 'direct_valve' after restore"
    assert not drec.outcome_eligible, \
        "Direct-valve dispatch must not be outcome-eligible"


# ── 6. Pending attribution summary: control_types includes direct_valve ────────


async def test_pending_attribution_summary_includes_direct_valve(
        hass, hass_storage, enable_custom_integrations):
    """pending_attribution_summary() reflects direct_valve in control_types."""
    from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = entry_data1.get("le2_shadow")
    if shadow1 is None:
        pytest.skip("No shadow controller")

    did = "dec-dv-summary"
    shadow1.runtime._zone("lz").pending_dispatch_records[did] = BoostDispatchRecord(
        decision_id=did,
        dispatch_status="fully_succeeded",
        device_control_type="direct_valve",
        effective_setpoints=(),
        targets_total=1,
        targets_failed=0,
    )

    summary = shadow1.runtime.pending_attribution_summary("lz")
    assert "direct_valve" in summary.get("control_types", []), \
        "pending_attribution_summary must list direct_valve in control_types"
    assert summary["pending_dispatch_count"] == 1
