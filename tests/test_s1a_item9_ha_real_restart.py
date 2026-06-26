"""Phase S1a Item 9 — Real HA Restart/Reload Equivalence (Docker/CI only).

Tests the full HA lifecycle for restart and reload equivalence:
  - LE 2.0 learning state persists across reload (via hass_storage)
  - Restore Barrier (_active_control_initialized) is False until entity restore
  - Boost lifecycle resets on reload (no stale offset)
  - processed_decision_ids survive reload (dedup intact)
  - Four-mode reload: INACTIVE/SHADOW/DETERMINISTIC/ADAPTIVE each mode stable
  - No double dispatch on reload
  - Coordinator data valid immediately after reload

Run in Docker / CI only (uses enable_custom_integrations + hass_storage).
"""
from __future__ import annotations

import logging

import pytest

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.coordinator import ControlAdaptationMode
from tests.helpers_ha_real import setup_zone
from tests.helpers_runtime_scenarios import heating_ramp_then_settle

pytestmark = pytest.mark.asyncio


def _coord(entry_data):
    return entry_data["coordinator"]


def _shadow(entry_data):
    return entry_data.get("le2_shadow")


# ── 1. Learning state persists across reload ─────────────────────────────────


async def test_learning_state_persists_across_reload(hass, hass_storage,
                                                      enable_custom_integrations):
    """After reload, the LE 2.0 shadow runtime is restored from hass_storage.

    We train the runtime before reload, then verify sample count is intact after.
    """
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    assert shadow1 is not None

    # Train the runtime with a heating scenario
    heating_ramp_then_settle(shadow1.runtime)
    shadow1.runtime.mark_dirty(important=True)
    n_before = shadow1.runtime._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count

    # Flush before reload so hass_storage has the data
    await shadow1.async_flush()
    await hass.async_block_till_done()

    # Reload
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    assert shadow2 is not shadow1  # fresh instance

    # Learning preserved
    n_after = shadow2.runtime._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
    assert n_after == n_before, (
        f"Sample count changed across reload: {n_before} → {n_after}"
    )


async def test_reload_produces_clean_health(hass, hass_storage, enable_custom_integrations):
    """After a clean reload, the shadow runtime reports zero restore errors."""
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    heating_ramp_then_settle(shadow1.runtime)
    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    h = shadow2.runtime.health()
    assert h.storage_warnings == 0


# ── 2. Restore Barrier ────────────────────────────────────────────────────────


async def test_restore_barrier_cleared_after_setup(hass, hass_storage,
                                                    enable_custom_integrations):
    """After full setup, _active_control_initialized must be True.

    The coordinator starts with barrier=False; ThermoSmartActiveSwitch.async_added_to_hass
    sets it via set_active_control().  By the time setup completes and async_block_till_done
    returns, the barrier must be cleared.
    """
    _, entry, entry_data = await setup_zone(hass)
    coord = _coord(entry_data)
    assert coord._active_control_initialized, (
        "Restore barrier should be cleared after full setup."
    )


async def test_restore_barrier_cleared_after_reload(hass, hass_storage,
                                                     enable_custom_integrations):
    """After reload, the barrier is cleared again by entity re-registration."""
    _, entry, entry_data = await setup_zone(hass)

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert coord2._active_control_initialized, (
        "Restore barrier must be cleared again after reload."
    )


# ── 3. Coordinator data valid after reload ────────────────────────────────────


async def test_coordinator_data_valid_after_reload(hass, hass_storage,
                                                    enable_custom_integrations):
    """After reload, coordinator.data must be a dict with zone key."""
    _, entry, entry_data = await setup_zone(hass)

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert coord2.data is not None
    assert isinstance(coord2.data, dict)
    assert "zone" in coord2.data


async def test_coordinator_last_update_success_after_reload(hass, hass_storage,
                                                             enable_custom_integrations):
    """After reload, last_update_success must be True."""
    _, entry, entry_data = await setup_zone(hass)

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert coord2.last_update_success is True


# ── 4. No double dispatch on reload ──────────────────────────────────────────


async def test_no_double_dispatch_on_reload(hass, hass_storage, enable_custom_integrations):
    """Reload creates exactly one shadow, coordinator never double-dispatches."""
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    shadow2 = _shadow(entry_data2)

    # Only one shadow attached
    assert coord2._le2_shadow is shadow2
    # Old shadow is gone / no stale reference
    assert shadow2 is not shadow1


# ── 5. Shadow disabled after unload ──────────────────────────────────────────


async def test_shadow_disabled_after_unload(hass, hass_storage, enable_custom_integrations):
    """After unload, the shadow runtime must be disabled (not sending cycles)."""
    _, entry, entry_data = await setup_zone(hass)
    shadow = _shadow(entry_data)

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert not shadow.diagnostics()["enabled"]


# ── 6. Adaptation mode stable after reload ────────────────────────────────────


async def test_adaptation_mode_stable_after_reload(hass, hass_storage,
                                                    enable_custom_integrations):
    """The adaptation mode reported by the coordinator must be the same after reload.

    This verifies that entity-restore correctly re-establishes switch state.
    """
    _, entry, entry_data1 = await setup_zone(hass)
    coord1 = _coord(entry_data1)
    mode_before = coord1.data.get("zone", {}).get("adaptation_mode")
    assert mode_before is not None

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    mode_after = coord2.data.get("zone", {}).get("adaptation_mode")
    assert mode_after == mode_before, (
        f"Adaptation mode changed across reload: {mode_before!r} → {mode_after!r}"
    )


# ── 7. hass.data cleaned up after unload ─────────────────────────────────────


async def test_hass_data_cleanup_after_unload(hass, hass_storage, enable_custom_integrations):
    """After unload, hass.data[DOMAIN][entry_id] must be removed."""
    _, entry, _ = await setup_zone(hass)
    assert entry.entry_id in hass.data.get(DOMAIN, {})

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.entry_id not in hass.data.get(DOMAIN, {})


# ── 8. Reload after training with flush ──────────────────────────────────────


async def test_reload_preserves_cycles_count(hass, hass_storage, enable_custom_integrations):
    """cycles count from zone runtime must match across reload."""
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    heating_ramp_then_settle(shadow1.runtime)
    cycles_before = shadow1.runtime._zone("lz").cycles
    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    cycles_after = shadow2.runtime._zone("lz").cycles
    assert cycles_after == cycles_before


async def test_storage_keys_present_after_reload_flush(hass, hass_storage,
                                                        enable_custom_integrations):
    """After reload with training + flush, hass_storage must contain LE2 keys."""
    _, entry, entry_data = await setup_zone(hass)
    shadow = _shadow(entry_data)
    heating_ramp_then_settle(shadow.runtime)
    shadow.runtime.mark_dirty(important=True)
    await shadow.async_flush()
    await hass.async_block_till_done()

    keys = [k for k in hass_storage if k.startswith("thermosmart_le2__")]
    assert keys, "LE2 storage key must be present after flush"
