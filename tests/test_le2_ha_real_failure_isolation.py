"""Phase 17C: real failure isolation — LE 2.0 never breaks the entry/heating (Docker/CI)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.const import DOMAIN
from tests.helpers_ha_real import setup_zone


async def test_setup_failure_keeps_entry_loaded(hass, hass_storage, enable_custom_integrations,
                                                monkeypatch):
    from custom_components.thermosmart.learning.runtime import ha_integration

    async def _boom(self):
        raise RuntimeError("boom")
    monkeypatch.setattr(ha_integration.LearningShadowController, "async_setup", _boom)
    ok, _, entry_data = await setup_zone(hass)
    assert ok and "coordinator" in entry_data


async def test_cycle_failure_isolated(hass, hass_storage, enable_custom_integrations,
                                      monkeypatch):
    _, entry, entry_data = await setup_zone(hass)
    shadow = entry_data["le2_shadow"]

    def _boom(inp):
        raise RuntimeError("cycle boom")
    monkeypatch.setattr(shadow._runtime, "run_cycle_safe", _boom)
    coord = entry_data["coordinator"]
    # a coordinator cycle must succeed even though the shadow cycle raises
    result = await coord._async_update_data()
    assert result is not None              # no UpdateFailed from the shadow
    assert shadow.errors >= 1              # error counted, isolated


async def test_store_failure_isolated(hass, hass_storage, enable_custom_integrations,
                                      monkeypatch):
    _, entry, entry_data = await setup_zone(hass)
    shadow = entry_data["le2_shadow"]

    async def _boom(_data):
        raise IOError("disk full")
    if shadow._runtime._persistence is not None:
        monkeypatch.setattr(shadow._runtime._persistence._store, "async_save", _boom)
    shadow.runtime.mark_dirty(important=True)
    await shadow.async_flush()                # swallowed
    coord = entry_data["coordinator"]
    result = await coord._async_update_data()
    assert result is not None


async def test_heating_path_unaffected_by_le2(hass, hass_storage, enable_custom_integrations,
                                              monkeypatch):
    _, entry, entry_data = await setup_zone(hass)
    shadow = entry_data["le2_shadow"]
    monkeypatch.setattr(shadow, "observe_safe",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    coord = entry_data["coordinator"]
    result = await coord._async_update_data()
    assert result is not None and "zone" in result
