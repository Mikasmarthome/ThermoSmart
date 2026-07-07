"""Phase 17C: reload/unload behaviour of the shadow controller."""
from __future__ import annotations

import pytest

from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_recording_coordinator


class TestReloadUnload:
    async def test_unload_flushes_and_disables(self):
        coord = make_recording_coordinator(indoor="18.0")
        store = FakeStore()
        sh = attach_shadow(coord, store=store)
        await sh.async_setup()
        await coord._async_update_data()
        await sh.async_unload()
        assert not sh.diagnostics()["enabled"]

    async def test_unload_stops_further_cycles(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        await sh.async_unload()
        before = sh.diagnostics()["last_cycle_ts"]
        # after unload, observe is a no-op (disabled)
        sh.observe_safe({"current_temp": 19.0, "effective_target": 21.0})
        assert sh.diagnostics()["last_cycle_ts"] == before

    async def test_unload_isolated_on_store_failure(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord, store=FakeStore(fail_save=True))
        await sh.async_setup()
        sh.runtime.mark_dirty(important=True)
        await sh.async_unload()  # flush fails internally, no raise
        assert not sh.diagnostics()["enabled"]

    async def test_reload_no_double_runtime(self):
        # simulate reload: new controller from persisted store, single runtime
        coord = make_recording_coordinator(indoor="18.0")
        store = FakeStore()
        sh1 = attach_shadow(coord, store=store)
        await sh1.async_setup()
        await coord._async_update_data()
        sh1.runtime.mark_dirty(important=True)
        await sh1.async_flush()
        await sh1.async_unload()

        coord2 = make_recording_coordinator(indoor="18.0")
        sh2 = attach_shadow(coord2, store=FakeStore(data=store.data))
        await sh2.async_setup()
        assert sh2.runtime is not sh1.runtime and sh2.runtime.initialized
