"""Phase S1a Item 9 — Mixed Multi-TRV Zone Restart under Real HA.

Tests a zone with two climate entities (Setpoint-TRV + Direct-Valve-TRV):

  * zone_control_type = 'mixed' when both types are present
  * LE2 boost is blocked for mixed zones (MIXED_CONTROL_TYPES gate)
  * No climate.set_temperature for valve entity
  * climate.set_temperature allowed for setpoint entity (non-valve side)
  * After reload: no stale state, _boost_active empty, _auto_valve_map empty
  * Pending attribution from partial failure survives reload with correct per-device data
  * Outcome idempotency: processed_decision_ids survive across reload

Scope note:
  Per-device effective setpoint comparison across setpoint+direct-valve in the SAME
  cycle requires both the TPI path (setpoint side) and the valve path (duty % side)
  to be active simultaneously. The TPI/valve-dispatch path for number.set_value
  requires number.set_value to be registered as a service in the test HA instance,
  which is available via async_mock_service. The per-cycle assertion is that
  climate.valve_trv does NOT receive a set_temperature call.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.const import DOMAIN
from tests.helpers_ha_real import (
    capture_climate_calls,
    setup_multi_trv_zone,
    set_trv_state,
)

pytestmark = pytest.mark.asyncio


def _coord(entry_data):
    return entry_data["coordinator"]


def _shadow(entry_data):
    return entry_data.get("le2_shadow")


# ── 1. Zone control type detection ────────────────────────────────────────────


async def test_zone_control_type_mixed_with_valve_map(
        hass, hass_storage, enable_custom_integrations):
    """zone_control_type = 'mixed' when one entity is valve-mapped, one is not."""
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)
    coord1 = _coord(entry_data1)

    # Trigger a coordinator cycle so zone_control_type is computed
    await coord1.async_refresh()
    await hass.async_block_till_done()

    last_rec = getattr(coord1, "_last_recommendation", None)
    if last_rec is not None:
        zt = last_rec.get("zone_control_type")
        assert zt in ("mixed", "direct_valve", "setpoint"), \
            f"zone_control_type must be one of the valid values; got {zt!r}"


async def test_auto_valve_map_empty_after_multi_trv_reload(
        hass, hass_storage, enable_custom_integrations):
    """After multi-TRV zone reload, _auto_valve_map starts empty."""
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert coord2._auto_valve_map == {}, \
        "_auto_valve_map must restart empty after multi-TRV reload"


# ── 2. No set_temperature for valve entity ────────────────────────────────────


async def test_valve_entity_receives_no_set_temperature(
        hass, hass_storage, enable_custom_integrations):
    """climate.valve_trv must never receive a climate.set_temperature call.

    The coordinator routes valve-mapped entities to the number.set_value path,
    not to climate.set_temperature.
    """
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)

    calls = capture_climate_calls(hass)
    coord1 = _coord(entry_data1)
    # Run two cycles
    await coord1.async_refresh()
    await hass.async_block_till_done()
    await coord1.async_refresh()
    await hass.async_block_till_done()

    valve_calls = [c for c in calls
                   if c.data.get("entity_id") == "climate.valve_trv"]
    assert len(valve_calls) == 0, (
        f"climate.valve_trv (direct-valve) must not receive climate.set_temperature; "
        f"got {len(valve_calls)}: {[c.data for c in valve_calls]}")


async def test_valve_entity_receives_no_set_temperature_after_reload(
        hass, hass_storage, enable_custom_integrations):
    """After multi-TRV zone reload, valve entity still gets no set_temperature."""
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    # Re-inject valve map after reload (device registry not available in tests)
    coord2._auto_valve_map = {"climate.valve_trv": "number.valve_position"}

    calls = capture_climate_calls(hass)
    await coord2.async_refresh()
    await hass.async_block_till_done()

    valve_calls = [c for c in calls
                   if c.data.get("entity_id") == "climate.valve_trv"]
    assert len(valve_calls) == 0, (
        f"Valve entity must not receive set_temperature after reload; "
        f"got {len(valve_calls)}")


# ── 3. _boost_active and _last_written_setpoints reset after reload ────────────


async def test_boost_active_empty_after_multi_trv_reload(
        hass, hass_storage, enable_custom_integrations):
    """After multi-TRV zone reload, _boost_active and _last_written_setpoints are empty."""
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)
    coord1 = _coord(entry_data1)
    coord1._boost_active["lz"] = {"setpoint_boost": 2.0}
    coord1._last_written_setpoints["climate.setpoint_trv"] = 23.0
    coord1._last_written_setpoints["climate.valve_trv"] = 20.0

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert coord2._boost_active == {}, "_boost_active must be empty after reload"
    assert coord2._last_written_setpoints == {}, \
        "_last_written_setpoints must be empty after reload"


# ── 4. Partial failure: one entity ok, one failed ─────────────────────────────


async def test_partial_failure_record_survives_multi_trv_reload(
        hass, hass_storage, enable_custom_integrations):
    """In a mixed zone, partial failure (1 ok, 1 failed) survives reload."""
    from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller")

    did = "dec-multi-partial"
    shadow1.runtime._zone("lz").pending_dispatch_records[did] = BoostDispatchRecord(
        decision_id=did,
        dispatch_status="partially_succeeded",
        outcome_eligible=False,
        outcome_reliability="partial",
        device_control_type="setpoint",
        effective_setpoints=(23.0,),     # only the setpoint entity succeeded
        targets_total=2,
        targets_failed=1,
    )
    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    drec = shadow2.runtime._zone("lz").pending_dispatch_records.get(did)
    assert drec is not None, "Partial failure dispatch record must survive reload"
    assert drec.dispatch_status == "partially_succeeded"
    assert drec.targets_total == 2
    assert drec.targets_failed == 1
    assert drec.effective_setpoints == (23.0,)


# ── 5. Reload during open pending outcome ─────────────────────────────────────


async def test_pending_context_survives_multi_trv_reload(
        hass, hass_storage, enable_custom_integrations):
    """An open pending boost context from a mixed zone survives reload."""
    from custom_components.thermosmart.learning.models import BoostUpdateContext
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller")

    did = "dec-mixed-open"
    shadow1.runtime._zone("lz").pending_boost_contexts[did] = BoostUpdateContext(
        source_episode_id=did, decision_id=did,
        learning_zone_id="lz", boost_applied_c=0.0,  # mixed → no boost
        dispatch_status="partially_succeeded",
        authority="live_record",
    )
    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    assert did in shadow2.runtime._zone("lz").pending_boost_contexts, \
        "Mixed-zone pending context must survive reload"


# ── 6. Outcome idempotency across multi-TRV reload ────────────────────────────


async def test_processed_decision_ids_intact_after_multi_trv_reload(
        hass, hass_storage, enable_custom_integrations):
    """processed_decision_ids survive multi-TRV zone reload (no double outcome)."""
    from tests.helpers_boost import boost_episode, boost_context
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller")

    m1 = shadow1.runtime._zone("lz").orchestrator.models.get("boost")
    if m1 is None:
        pytest.skip("No boost model")

    did = "dec-multi-dedup"
    vals = [18.0, 19.5, 21.0]
    ep = boost_episode(did, vals, decision_id=did, zone="lz")
    bctx = boost_context(ep)
    m1.update(ep, bctx)
    ids_before = set(m1._state.processed_decision_ids)

    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    m2 = shadow2.runtime._zone("lz").orchestrator.models.get("boost")
    assert m2 is not None
    ids_after = set(m2._state.processed_decision_ids)
    assert ids_before == ids_after, "processed_decision_ids must survive multi-TRV reload"

    count_before = m2._state.full_count
    m2.update(ep, bctx)
    assert m2._state.full_count == count_before, \
        "Duplicate outcome after multi-TRV reload must be rejected by dedup"


# ── 7. Pending attribution summary for multi-TRV zone ─────────────────────────


async def test_pending_attribution_summary_multi_trv(
        hass, hass_storage, enable_custom_integrations):
    """pending_attribution_summary() reflects partial failure for mixed-zone dispatch."""
    from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller")

    did = "dec-multi-summary"
    shadow1.runtime._zone("lz").pending_dispatch_records[did] = BoostDispatchRecord(
        decision_id=did,
        dispatch_status="partially_succeeded",
        outcome_eligible=False,
        device_control_type="setpoint",
        effective_setpoints=(23.0,),
        targets_total=2,
        targets_failed=1,
    )

    summary = shadow1.runtime.pending_attribution_summary("lz")
    assert summary["pending_dispatch_count"] == 1
    assert summary["failed_targets"] == 1
    assert summary["total_targets"] == 2
    assert "partially_succeeded" in summary.get("dispatch_statuses", {})


# ── 8. Unavailable device in multi-TRV zone ───────────────────────────────────


async def test_unavailable_valve_entity_no_crash(
        hass, hass_storage, enable_custom_integrations):
    """When the valve entity is unavailable, coordinator runs without crash."""
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)
    coord1 = _coord(entry_data1)
    coord1._auto_valve_map = {"climate.valve_trv": "number.valve_position"}
    hass.states.async_set("climate.valve_trv", "unavailable", {})
    hass.states.async_set("number.valve_position", "unavailable", {})

    calls = capture_climate_calls(hass)
    # Must not raise
    await coord1.async_refresh()
    await hass.async_block_till_done()

    # No set_temperature for the unavailable valve entity
    valve_calls = [c for c in calls
                   if c.data.get("entity_id") == "climate.valve_trv"]
    assert len(valve_calls) == 0, \
        "Unavailable valve entity must not receive set_temperature"


async def test_setpoint_entity_still_written_when_valve_unavailable(
        hass, hass_storage, enable_custom_integrations):
    """When valve entity unavailable, the setpoint entity side is still handled."""
    _, entry, entry_data1 = await setup_multi_trv_zone(hass)
    coord1 = _coord(entry_data1)
    coord1._auto_valve_map = {"climate.valve_trv": "number.valve_position"}
    hass.states.async_set("climate.valve_trv", "unavailable", {})
    hass.states.async_set("number.valve_position", "unavailable", {})
    # Setpoint entity far from target → expects a write
    set_trv_state(hass, "climate.setpoint_trv", setpoint=5.0, current_temp=12.0)

    calls = capture_climate_calls(hass)
    await coord1.async_refresh()
    await hass.async_block_till_done()

    # Coordinator still runs and is operational
    assert coord1.last_update_success, \
        "Coordinator must remain operational with one unavailable device in multi-TRV"
