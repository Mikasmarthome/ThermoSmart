"""Phase 17B HA store adapter (tested against a fake Store, no real HA)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime.ha_store import (
    STORE_KEY_PREFIX,
    HomeAssistantStoreAdapter,
    store_key,
)
from custom_components.thermosmart.learning.runtime import PersistenceOrchestrator, SavePolicy


class _FakeStore:
    def __init__(self):
        self.data = None
        self.saves = 0

    async def async_load(self):
        return self.data

    async def async_save(self, data):
        self.data = data
        self.saves += 1


class TestHaStore:
    def test_key_namespaced(self):
        assert store_key("abc123").startswith(STORE_KEY_PREFIX)

    def test_key_no_empty(self):
        with pytest.raises(ValueError):
            store_key("")

    async def test_adapter_load_save(self):
        fake = _FakeStore()
        adapter = HomeAssistantStoreAdapter(hass=None, zone_segment="z1", store=fake)
        await adapter.async_save({"x": 1})
        assert fake.saves == 1
        assert await adapter.async_load() == {"x": 1}

    async def test_adapter_satisfies_persistence_protocol(self):
        fake = _FakeStore()
        adapter = HomeAssistantStoreAdapter(hass=None, zone_segment="z1", store=fake)
        p = PersistenceOrchestrator(adapter, policy=SavePolicy())
        p.mark_dirty("2025-01-01T05:00:00+00:00")
        assert await p.flush("2025-01-01T05:00:01+00:00", lambda: {"state": True})
        assert fake.data == {"state": True}

    def test_key_has_no_entity_name(self):
        # the caller must pass a non-identifying token; the key carries only that
        k = store_key("8f2ac1")
        assert "climate." not in k and "living_room" not in k
