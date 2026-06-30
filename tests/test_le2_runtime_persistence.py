"""Phase 17 persistence: save policy, dirty tracking, flush, failure isolation."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
    PersistenceOrchestrator,
    SavePolicy,
    SaveTrigger,
)
from tests.helpers_runtime import MemoryStore, cycle_input

pytestmark = pytest.mark.asyncio


class TestSavePolicy:
    async def test_clean_no_save(self):
        p = PersistenceOrchestrator(MemoryStore(), policy=SavePolicy())
        assert p.decide("2026-06-22T05:00:00+00:00") is SaveTrigger.NONE

    async def test_debounced_save(self):
        store = MemoryStore()
        p = PersistenceOrchestrator(store, policy=SavePolicy(debounce_s=30))
        p.mark_dirty("2026-06-22T05:00:00+00:00")
        assert p.decide("2026-06-22T05:00:10+00:00") is SaveTrigger.NONE
        t = await p.maybe_save("2026-06-22T05:00:40+00:00", lambda: {"x": 1})
        assert t is SaveTrigger.AUTHORITATIVE_CHANGE and store.saves == 1

    async def test_important_quick_save(self):
        store = MemoryStore()
        p = PersistenceOrchestrator(store, policy=SavePolicy(important_debounce_s=2))
        p.mark_dirty("2026-06-22T05:00:00+00:00", important=True)
        t = await p.maybe_save("2026-06-22T05:00:05+00:00", lambda: {"x": 1})
        assert t is SaveTrigger.IMPORTANT_OUTCOME

    async def test_hourly_pending_flush(self):
        store = MemoryStore()
        p = PersistenceOrchestrator(store, policy=SavePolicy(debounce_s=30, max_pending_s=3600))
        p.mark_dirty("2026-06-22T05:00:00+00:00")
        # keep re-dirtying so debounce never fires, but hourly net catches it
        p.mark_dirty("2026-06-22T05:59:59+00:00")
        t = await p.maybe_save("2026-06-22T06:00:05+00:00", lambda: {"x": 1})
        assert t is SaveTrigger.PERIODIC

    async def test_flush_unconditional(self):
        store = MemoryStore()
        p = PersistenceOrchestrator(store)
        p.mark_dirty("2026-06-22T05:00:00+00:00")
        assert await p.flush("2026-06-22T05:00:01+00:00", lambda: {"x": 1})
        assert not p.dirty and store.saves == 1

    async def test_flush_when_clean_noop(self):
        p = PersistenceOrchestrator(MemoryStore())
        assert not await p.flush("2026-06-22T05:00:00+00:00", lambda: {"x": 1})

    async def test_save_failure_isolated(self):
        store = MemoryStore(fail_save=True)
        p = PersistenceOrchestrator(store)
        p.mark_dirty("2026-06-22T05:00:00+00:00")
        ok = await p.flush("2026-06-22T05:00:01+00:00", lambda: {"x": 1})
        assert not ok and p.dirty  # still dirty, but no exception
        assert any("save_failed" in w for w in p.warnings)

    async def test_load_failure_isolated(self):
        p = PersistenceOrchestrator(MemoryStore(fail_load=True))
        assert await p.load() is None
        assert any("load_failed" in w for w in p.warnings)


class TestRuntimePersistence:
    async def test_save_and_restore_roundtrip(self):
        store = MemoryStore()
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW),
                             store=store, clock=lambda: "2026-06-22T05:00:00+00:00")
        await rt.async_setup()
        rt.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        rt.mark_dirty(important=True)
        await rt.async_flush()
        assert store.data is not None

        rt2 = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW),
                              store=MemoryStore(data=store.data),
                              clock=lambda: "2026-06-22T05:10:00+00:00")
        await rt2.async_setup()
        # restored capture state carries the open decision
        assert rt2.health().zones >= 1

    async def test_unload_flushes(self):
        store = MemoryStore()
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW),
                             store=store, clock=lambda: "2026-06-22T05:00:00+00:00")
        await rt.async_setup()
        rt.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        rt.mark_dirty()
        assert await rt.async_unload()

    async def test_store_failure_does_not_break_runtime(self):
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW),
                             store=MemoryStore(fail_save=True),
                             clock=lambda: "2026-06-22T05:00:00+00:00")
        await rt.async_setup()
        rt.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        rt.mark_dirty()
        # flush fails but does not raise; cycle still works afterwards
        await rt.async_flush()
        r = rt.run_cycle(cycle_input("2026-06-22T05:05:00+00:00"))
        assert r.captured
