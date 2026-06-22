"""Phase 17C: real Config-Entry setup with the LE 2.0 shadow runtime (Docker/CI only)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.learning.runtime import LearningRuntimeMode
from tests.helpers_ha_real import setup_zone


async def test_entry_loads_with_shadow(hass, hass_storage, enable_custom_integrations):
    ok, entry, entry_data = await setup_zone(hass)
    assert ok
    assert "coordinator" in entry_data
    shadow = entry_data.get("le2_shadow")
    assert shadow is not None
    assert shadow.runtime.mode is LearningRuntimeMode.SHADOW


async def test_coordinator_has_single_shadow_reference(hass, hass_storage,
                                                       enable_custom_integrations):
    _, _, entry_data = await setup_zone(hass)
    coord = entry_data["coordinator"]
    assert coord._le2_shadow is entry_data["le2_shadow"]


async def test_no_store_written_on_setup(hass, hass_storage, enable_custom_integrations):
    _, _, entry_data = await setup_zone(hass)
    shadow = entry_data["le2_shadow"]
    # an empty initial setup must not persist a store
    assert shadow.runtime._persistence is None or not [
        k for k in hass_storage if k.startswith("thermosmart_le2__")]


async def test_diagnostics_available(hass, hass_storage, enable_custom_integrations):
    _, _, entry_data = await setup_zone(hass)
    d = entry_data["le2_shadow"].diagnostics()
    assert d["mode"] == "shadow" and d["initialized"] is True


async def test_setup_failure_isolated(hass, hass_storage, enable_custom_integrations,
                                      monkeypatch):
    # force the LE 2.0 shadow setup to raise; the zone must still load
    from custom_components.thermosmart.learning.runtime import ha_integration

    async def _boom(self):
        raise RuntimeError("le2 setup boom")
    monkeypatch.setattr(ha_integration.LearningShadowController, "async_setup", _boom)

    ok, entry, entry_data = await setup_zone(hass)
    assert ok                                   # zone loaded despite LE2 failure
    assert "coordinator" in entry_data          # heating runtime present
    # shadow either absent or attached-but-disabled; never breaks the entry
    coord = entry_data["coordinator"]
    assert coord is not None
