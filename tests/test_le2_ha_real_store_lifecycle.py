"""Phase 17C: real Home Assistant Store lifecycle (Docker/CI only).

Exercises the actual ``homeassistant.helpers.storage.Store`` via the LE 2.0
adapter + runtime — restore, debounced/force save, key namespacing, corruption
isolation and dedup-after-restore. Auto-skipped on Windows (see conftest).
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from custom_components.thermosmart.learning.runtime.ha_store import (
    STORE_KEY_PREFIX,
    HomeAssistantStoreAdapter,
)
from custom_components.thermosmart.learning.runtime.ha_integration import _zone_segment
from tests.helpers_runtime_scenarios import heating_ramp_then_settle


def _runtime(hass, zone, clock="2025-01-01T07:00:00+00:00"):
    adapter = HomeAssistantStoreAdapter(hass, _zone_segment(zone))
    rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                               startup_grace_cycles=0),
                         store=adapter, clock=lambda: clock)
    return rt, adapter


async def test_store_key_namespaced_no_names(hass):
    adapter = HomeAssistantStoreAdapter(hass, _zone_segment("climate.living_room"))
    assert adapter.key.startswith(STORE_KEY_PREFIX)
    assert "living_room" not in adapter.key and "climate." not in adapter.key


async def test_no_store_written_on_empty_setup(hass, hass_storage):
    rt, adapter = _runtime(hass, "zone_a")
    await rt.async_setup()
    await hass.async_block_till_done()
    # an empty setup must not write a store file
    assert adapter.key not in hass_storage


async def test_dirty_then_flush_writes_state(hass, hass_storage):
    rt, adapter = _runtime(hass, "zone_a")
    await rt.async_setup()
    heating_ramp_then_settle(rt)          # produces an authoritative change
    rt.mark_dirty(important=True)
    await rt.async_flush()
    await hass.async_block_till_done()
    assert adapter.key in hass_storage
    assert hass_storage[adapter.key]["data"]["runtime_schema_version"] == 1


async def test_restore_after_new_runtime_instance(hass, hass_storage):
    rt1, _ = _runtime(hass, "zone_a")
    await rt1.async_setup()
    heating_ramp_then_settle(rt1)
    samples1 = rt1._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
    rt1.mark_dirty(important=True)
    await rt1.async_flush()
    await hass.async_block_till_done()

    rt2, _ = _runtime(hass, "zone_a")
    await rt2.async_setup()              # restores from the real Store
    # re-feeding identical timestamps must not re-learn (dedup survives restore)
    heating_ramp_then_settle(rt2)
    samples2 = rt2._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
    assert samples1 == 1 and samples2 == 1


async def test_corrupt_store_isolated(hass, hass_storage):
    rt, adapter = _runtime(hass, "zone_a")
    # pre-seed a corrupt payload at the real store key
    hass_storage[adapter.key] = {"version": 2, "data": {
        "runtime_schema_version": 1, "le_version": "2.0.0",
        "zones": {"lz": {"models": {"heat_rate": {"garbage": True}}}}}}
    ok = await rt.async_setup()         # corrupt segment isolated, no raise
    assert ok and rt.initialized


async def test_save_failure_does_not_raise(hass, monkeypatch):
    rt, adapter = _runtime(hass, "zone_a")
    await rt.async_setup()
    heating_ramp_then_settle(rt)
    rt.mark_dirty(important=True)

    async def _boom(_data):
        raise IOError("disk full")
    monkeypatch.setattr(adapter, "async_save", _boom)
    # flush swallows the store error; the runtime/heating keep working
    await rt.async_flush()
    assert rt._zone("lz") is not None
