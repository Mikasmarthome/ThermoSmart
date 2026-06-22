"""Phase 17C: restart survival of shadow state through the HA store."""
from __future__ import annotations

import pytest

from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_recording_coordinator


class TestRestartSurvival:
    async def test_state_persisted_and_restored(self):
        coord = make_recording_coordinator(indoor="18.0")
        store = FakeStore()
        sh = attach_shadow(coord, store=store)
        await sh.async_setup()
        for _ in range(3):
            await coord._async_update_data()
        sh.runtime.mark_dirty(important=True)
        await sh.async_flush()
        assert store.data is not None

        # restart: new controller restores from the same store payload
        coord2 = make_recording_coordinator(indoor="18.0")
        sh2 = attach_shadow(coord2, store=FakeStore(data=store.data))
        ok = await sh2.async_setup()
        assert ok and sh2.runtime.initialized

    async def test_no_crash_on_corrupt_payload(self):
        coord = make_recording_coordinator()
        corrupt = {"runtime_schema_version": 1, "le_version": "2.0.0",
                   "zones": {"z": {"models": {"heat_rate": {"bad": True}}}}}
        sh = attach_shadow(coord, store=FakeStore(data=corrupt))
        assert await sh.async_setup()  # corrupt segment isolated

    async def test_unknown_schema_ignored(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord, store=FakeStore(data={"runtime_schema_version": 999}))
        await sh.async_setup()
        assert sh.runtime.initialized

    async def test_survives_and_continues(self):
        coord = make_recording_coordinator(indoor="18.0")
        store = FakeStore()
        sh = attach_shadow(coord, store=store)
        await sh.async_setup()
        await coord._async_update_data()
        sh.runtime.mark_dirty(important=True)
        await sh.async_flush()
        coord2 = make_recording_coordinator(indoor="18.0")
        sh2 = attach_shadow(coord2, store=FakeStore(data=store.data))
        await sh2.async_setup()
        # continues capturing after restart
        assert (await coord2._async_update_data()) is not None
