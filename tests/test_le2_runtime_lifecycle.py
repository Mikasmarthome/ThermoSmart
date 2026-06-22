"""Phase 17 runtime lifecycle: setup/restore/unload/reset/health/zone add-remove."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
    RuntimeHealth,
)
from tests.helpers_runtime import MemoryStore, cycle_input

pytestmark = pytest.mark.asyncio


def rt(store=None, mode=LearningRuntimeMode.SHADOW):
    return LearningRuntime(LearningRuntimeConfig(mode=mode), store=store,
                           clock=lambda: "2026-06-22T05:00:00+00:00")


class TestLifecycle:
    async def test_setup_initializes(self):
        r = rt(MemoryStore())
        assert await r.async_setup() and r.initialized

    async def test_setup_without_store(self):
        r = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW))
        assert await r.async_setup() and r.initialized

    async def test_restore_survives_restart(self):
        store = MemoryStore()
        a = rt(store)
        await a.async_setup()
        a.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        a.mark_dirty()
        await a.async_flush()
        b = rt(MemoryStore(data=store.data))
        await b.async_setup()
        assert b.initialized and b.health().zones >= 1

    async def test_reset_clears_state(self):
        r = rt()
        await r.async_setup()
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        r.reset()
        assert r.health().zones == 0 and not r.initialized

    async def test_remove_zone(self):
        r = rt()
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="a"))
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="b"))
        r.remove_zone("a")
        assert r.health().zones == 1

    async def test_health_contract(self):
        r = rt()
        await r.async_setup()
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        h = r.health()
        assert isinstance(h, RuntimeHealth)
        assert h.mode == "shadow" and h.zones == 1 and h.last_cycle_ts is not None

    async def test_corrupt_segment_isolated_on_restore(self):
        store = MemoryStore(data={
            "runtime_schema_version": 1, "le_version": "2.0.0",
            "zones": {"lr": {"models": {"heat_rate": {"garbage": True}}}}})
        r = rt(MemoryStore(data=store.data))
        # restore must not raise even with a corrupt model segment
        await r.async_setup()
        assert r.initialized

    async def test_unknown_schema_ignored(self):
        store = MemoryStore(data={"runtime_schema_version": 999, "zones": {"lr": {}}})
        r = rt(MemoryStore(data=store.data))
        await r.async_setup()
        assert r.health().zones == 0  # incompatible payload ignored

    async def test_startup_grace(self):
        # first cycle within grace must not crash; subsequent cycles proceed
        r = rt()
        await r.async_setup()
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        r.run_cycle(cycle_input("2026-06-22T05:05:00+00:00"))
        assert r.health().zones == 1

    async def test_idempotent_setup(self):
        r = rt(MemoryStore())
        await r.async_setup()
        await r.async_setup()
        assert r.initialized
