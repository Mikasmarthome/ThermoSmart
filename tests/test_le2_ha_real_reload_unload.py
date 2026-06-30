"""Phase 17C: real reload/unload of the LE 2.0 shadow runtime (Docker/CI only).

Verifies the awaited (block-on-finish) unload flush, hass.data cleanup and the
absence of double instances / task leaks on reload.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.const import DOMAIN
from tests.helpers_ha_real import setup_zone


async def test_unload_flushes_pending_state(hass, hass_storage, enable_custom_integrations):
    _, entry, entry_data = await setup_zone(hass)
    shadow = entry_data["le2_shadow"]
    # produce a real authoritative change + force pending
    from tests.helpers_runtime_scenarios import heating_ramp_then_settle
    heating_ramp_then_settle(shadow.runtime)
    shadow.runtime.mark_dirty(important=True)

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    # awaited flush during unload -> state persisted by the time unload returns
    keys = [k for k in hass_storage if k.startswith("thermosmart_le2__")]
    assert keys, "expected the shadow state to be flushed on unload"


async def test_unload_cleans_hass_data(hass, hass_storage, enable_custom_integrations):
    _, entry, _ = await setup_zone(hass)
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.entry_id not in hass.data.get(DOMAIN, {})


async def test_unload_disables_shadow(hass, hass_storage, enable_custom_integrations):
    _, entry, entry_data = await setup_zone(hass)
    shadow = entry_data["le2_shadow"]
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert not shadow.diagnostics()["enabled"]


async def test_reload_single_instance(hass, hass_storage, enable_custom_integrations):
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = entry_data1["le2_shadow"]
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = entry_data2["le2_shadow"]
    coord2 = entry_data2["coordinator"]
    assert shadow2 is not shadow1                 # fresh instance after reload
    assert coord2._le2_shadow is shadow2          # exactly one hook, current instance


async def test_reload_no_runtime_warning(hass, hass_storage, enable_custom_integrations,
                                         recwarn):
    _, entry, _ = await setup_zone(hass)
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
