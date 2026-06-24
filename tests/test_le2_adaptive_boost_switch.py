"""B2c: ThermoSmartAdaptiveBoostSwitch — entity semantics, persistence, migration.

Tests:
  - Default OFF on first install
  - RestoreEntity: ON restored from "on" state
  - RestoreEntity: OFF restored from "off" state
  - RestoreEntity: OFF when no previous state (first boot)
  - Multiple zones are independent (different unique_ids, different devices)
  - set_adaptive_boost_enabled propagated on restore
  - Turning ON propagates to coordinator
  - Turning OFF propagates to coordinator and releases active boost
  - Unique ID format: {entry_id}_adaptive_boost
  - translation_key: adaptive_boost
  - icon: mdi:rocket-launch
  - extra_state_attributes contains adaptive_boost_active
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch as _patch

from custom_components.thermosmart.switch import ThermoSmartAdaptiveBoostSwitch


def _make_entry(entry_id: str = "zone_abc"):
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.data = {"name": f"Zone {entry_id}"}
    entry.options = {}
    return entry


def _make_coordinator(entry_id: str = "zone_abc"):
    coord = MagicMock()
    coord.zone_id = entry_id
    coord.zone_name = f"Zone {entry_id}"
    coord._adaptive_boost_enabled = False
    coord.set_adaptive_boost_enabled = MagicMock()
    coord.async_request_refresh = AsyncMock()
    return coord


def _make_switch(entry_id: str = "zone_abc"):
    entry = _make_entry(entry_id)
    coord = _make_coordinator(entry_id)
    sw = ThermoSmartAdaptiveBoostSwitch(coord, entry)
    sw.hass = MagicMock()
    return sw, coord


# ── Default state ─────────────────────────────────────────────────────────────

class TestDefaultOff:
    def test_default_is_off(self):
        sw, _ = _make_switch()
        assert sw.is_on is False

    def test_default_unique_id(self):
        sw, _ = _make_switch("test_zone")
        assert sw._attr_unique_id == "test_zone_adaptive_boost"

    def test_translation_key(self):
        sw, _ = _make_switch()
        assert sw._attr_translation_key == "adaptive_boost"

    def test_icon(self):
        sw, _ = _make_switch()
        assert sw._attr_icon == "mdi:rocket-launch"

    def test_extra_state_attributes_off(self):
        sw, _ = _make_switch()
        assert sw.extra_state_attributes == {"adaptive_boost_active": False}


# ── RestoreEntity semantics ───────────────────────────────────────────────────

class TestRestoreEntity:
    async def test_restore_on(self):
        sw, coord = _make_switch()
        last_state = MagicMock()
        last_state.state = "on"
        with _patch.object(sw, "async_get_last_state", AsyncMock(return_value=last_state)):
            await sw.async_added_to_hass()
        assert sw.is_on is True
        coord.set_adaptive_boost_enabled.assert_called_once_with(True)

    async def test_restore_off(self):
        sw, coord = _make_switch()
        last_state = MagicMock()
        last_state.state = "off"
        with _patch.object(sw, "async_get_last_state", AsyncMock(return_value=last_state)):
            await sw.async_added_to_hass()
        assert sw.is_on is False
        coord.set_adaptive_boost_enabled.assert_called_once_with(False)

    async def test_restore_no_previous_state_defaults_off(self):
        sw, coord = _make_switch()
        with _patch.object(sw, "async_get_last_state", AsyncMock(return_value=None)):
            await sw.async_added_to_hass()
        assert sw.is_on is False
        coord.set_adaptive_boost_enabled.assert_called_once_with(False)

    async def test_restore_unknown_state_defaults_off(self):
        """Unexpected/corrupt state values must produce OFF (conservative fallback)."""
        sw, coord = _make_switch()
        last_state = MagicMock()
        last_state.state = "unavailable"
        with _patch.object(sw, "async_get_last_state", AsyncMock(return_value=last_state)):
            await sw.async_added_to_hass()
        assert sw.is_on is False


# ── Turn on / Turn off ────────────────────────────────────────────────────────

class TestTurnOnOff:
    async def test_turn_on(self):
        sw, coord = _make_switch()
        sw.async_write_ha_state = MagicMock()
        await sw.async_turn_on()
        assert sw.is_on is True
        coord.set_adaptive_boost_enabled.assert_called_once_with(True)
        coord.async_request_refresh.assert_awaited_once()

    async def test_turn_off(self):
        sw, coord = _make_switch()
        sw._is_on = True
        sw.async_write_ha_state = MagicMock()
        await sw.async_turn_off()
        assert sw.is_on is False
        coord.set_adaptive_boost_enabled.assert_called_once_with(False)
        coord.async_request_refresh.assert_awaited_once()

    async def test_turn_on_writes_ha_state(self):
        sw, _ = _make_switch()
        write_calls = []
        sw.async_write_ha_state = MagicMock(side_effect=lambda: write_calls.append("write"))
        await sw.async_turn_on()
        assert write_calls, "async_write_ha_state must be called"

    async def test_turn_off_writes_ha_state(self):
        sw, _ = _make_switch()
        sw._is_on = True
        write_calls = []
        sw.async_write_ha_state = MagicMock(side_effect=lambda: write_calls.append("write"))
        await sw.async_turn_off()
        assert write_calls, "async_write_ha_state must be called"

    def test_extra_state_attributes_on(self):
        sw, _ = _make_switch()
        sw._is_on = True
        assert sw.extra_state_attributes == {"adaptive_boost_active": True}


# ── Multi-zone independence ───────────────────────────────────────────────────

class TestMultiZoneIndependence:
    def test_unique_ids_are_per_zone(self):
        sw_a, _ = _make_switch("zone_A")
        sw_b, _ = _make_switch("zone_B")
        assert sw_a._attr_unique_id != sw_b._attr_unique_id
        assert sw_a._attr_unique_id == "zone_A_adaptive_boost"
        assert sw_b._attr_unique_id == "zone_B_adaptive_boost"

    async def test_turning_on_zone_a_does_not_affect_zone_b(self):
        sw_a, coord_a = _make_switch("zone_A")
        sw_b, coord_b = _make_switch("zone_B")
        sw_a.async_write_ha_state = MagicMock()
        sw_b.async_write_ha_state = MagicMock()
        await sw_a.async_turn_on()
        coord_b.set_adaptive_boost_enabled.assert_not_called()

    async def test_restoring_zone_a_on_does_not_touch_zone_b(self):
        sw_a, coord_a = _make_switch("zone_A")
        sw_b, coord_b = _make_switch("zone_B")
        last_on = MagicMock(); last_on.state = "on"
        last_off = MagicMock(); last_off.state = "off"
        with _patch.object(sw_a, "async_get_last_state", AsyncMock(return_value=last_on)):
            await sw_a.async_added_to_hass()
        with _patch.object(sw_b, "async_get_last_state", AsyncMock(return_value=last_off)):
            await sw_b.async_added_to_hass()
        assert sw_a.is_on is True
        assert sw_b.is_on is False


# ── Migration: switch absent in old installs → OFF by default ─────────────────

class TestMigrationNewSwitchDefaultOff:
    """Existing users who never had the switch must not see it auto-enabled."""

    async def test_no_prior_state_in_entity_registry_gives_off(self):
        # Simulated: async_get_last_state returns None (entity never registered before)
        sw, coord = _make_switch()
        with _patch.object(sw, "async_get_last_state", AsyncMock(return_value=None)):
            await sw.async_added_to_hass()
        assert sw.is_on is False

    async def test_coordinator_adaptive_boost_stays_false_without_switch_restore(self):
        from tests.helpers_ha_runtime import make_recording_coordinator
        coord = make_recording_coordinator()
        # Coordinator starts with False; no switch entity was added in this test
        assert coord._adaptive_boost_enabled is False
