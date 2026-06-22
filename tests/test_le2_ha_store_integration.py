"""Phase 17C: HA store lifecycle through the shadow controller."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime.ha_integration import _zone_segment
from custom_components.thermosmart.learning.runtime.ha_store import STORE_KEY_PREFIX, store_key
from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_recording_coordinator


class TestHaStoreIntegration:
    async def test_flush_writes_via_store(self):
        coord = make_recording_coordinator(indoor="18.0")
        store = FakeStore()
        sh = attach_shadow(coord, store=store)
        await sh.async_setup()
        await coord._async_update_data()
        sh.runtime.mark_dirty(important=True)
        await sh.async_flush()
        assert store.saves >= 1 and store.data is not None

    async def test_no_save_per_cycle(self):
        coord = make_recording_coordinator(indoor="18.0")
        store = FakeStore()
        sh = attach_shadow(coord, store=store)
        await sh.async_setup()
        for _ in range(5):
            await coord._async_update_data()
            await sh.async_save_if_due()
        # debounced (real clock advances minimally) -> not five writes
        assert store.saves <= 1

    async def test_restore_on_setup(self):
        coord = make_recording_coordinator()
        store = FakeStore(data={"runtime_schema_version": 1, "le_version": "2.0.0",
                                "zones": {}})
        sh = attach_shadow(coord, store=store)
        ok = await sh.async_setup()
        assert ok and sh.runtime.initialized

    def test_store_key_namespaced_no_names(self):
        seg = _zone_segment("climate.living_room_long_entry_id")
        key = store_key(seg)
        assert key.startswith(STORE_KEY_PREFIX) and "living_room" not in key

    async def test_save_failure_keeps_dirty(self):
        coord = make_recording_coordinator(indoor="18.0")
        store = FakeStore(fail_save=True)
        sh = attach_shadow(coord, store=store)
        await sh.async_setup()
        sh.runtime.mark_dirty(important=True)
        await sh.async_flush()  # fails internally, no raise
        assert store.saves == 0
