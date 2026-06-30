"""HA integration tests for ThermoSmartModeSelect listener lifecycle (Docker/CI only).

Validates that setup, unload, and reload of the select entity do not produce
any 'async_remove_listener' AttributeError and that the coordinator listener
is properly registered and removed via the async_on_remove pattern.
"""
from __future__ import annotations

import logging

import pytest

from custom_components.thermosmart.const import DOMAIN
from tests.helpers_ha_real import setup_zone


def _select_attribute_errors(caplog) -> list[str]:
    """Return messages of ERROR-level AttributeErrors from select or entity platform."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR
        and "AttributeError" in r.getMessage()
        and "async_remove_listener" in r.getMessage()
    ]


# ── Test 7: no teardown AttributeError for async_remove_listener ─────────────

async def test_no_async_remove_listener_error_on_unload(
    hass, hass_storage, enable_custom_integrations, caplog
):
    """Unloading the entry must not raise AttributeError for async_remove_listener."""
    _, entry, _ = await setup_zone(hass)

    with caplog.at_level(logging.ERROR):
        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

    errors = _select_attribute_errors(caplog)
    assert not errors, f"Unexpected AttributeErrors: {errors}"


# ── Test 8: entry removed from hass.data after unload ────────────────────────

async def test_unload_removes_select_listener(
    hass, hass_storage, enable_custom_integrations, caplog
):
    """After unload, no stale entry in hass.data and no select AttributeError."""
    _, entry, entry_data = await setup_zone(hass)
    assert len(entry_data["coordinator"]._listeners) > 0

    with caplog.at_level(logging.ERROR):
        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.entry_id not in hass.data.get(DOMAIN, {})
    assert not _select_attribute_errors(caplog)


# ── Test 9: reload does not double-register ───────────────────────────────────

async def test_reload_does_not_double_register_listener(
    hass, hass_storage, enable_custom_integrations, caplog
):
    """Reload creates a fresh coordinator with exactly 1 mode-select listener, not 2."""
    _, entry, entry_data1 = await setup_zone(hass)

    with caplog.at_level(logging.ERROR):
        await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()

    assert not _select_attribute_errors(caplog)

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = entry_data2["coordinator"]

    # Verify the mode-select entity's listener is present exactly once.
    # We identify it by finding callbacks bound to ThermoSmartModeSelect instances.
    from custom_components.thermosmart.select import ThermoSmartModeSelect
    mode_select_callbacks = [
        cb for cb, _ctx in coord2._listeners.values()
        if hasattr(cb, "__self__") and isinstance(cb.__self__, ThermoSmartModeSelect)
    ]
    assert len(mode_select_callbacks) == 1, (
        f"Expected exactly 1 mode-select listener after reload, got {len(mode_select_callbacks)}"
    )


# ── Test 10: setup/unload/reload stays clean end-to-end ──────────────────────

async def test_setup_unload_reload_no_errors(
    hass, hass_storage, enable_custom_integrations, caplog
):
    """Full lifecycle: setup → unload → reload → unload — no AttributeError."""
    with caplog.at_level(logging.ERROR):
        _, entry, _ = await setup_zone(hass)
        await hass.async_block_till_done()

        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

    assert not _select_attribute_errors(caplog)
