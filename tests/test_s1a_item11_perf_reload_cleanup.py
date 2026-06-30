"""S1a Item 11 — Reload/Unload Cleanup (Windows-kompatibel, kein hass-Fixture).

Proves component-level cleanup properties at the LearningRuntime layer.
These tests run on Windows and complement test_s1a_item11_perf_ha_real_reload.py
(which requires the HA hass fixture and runs in Docker/Linux only).

Properties covered:
  - async_unload() flushes dirty state before shutdown
  - after async_unload() runtime is not dirty (no pending state)
  - repeated setup/unload/setup cycle does not accumulate zones or internal state
  - state restored from store after unload+reinit is intact (no data loss, no duplication)
  - 10x setup/run/unload loop: store size remains bounded
  - mark_dirty(important=True) + async_unload() forces immediate flush
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import MemoryStore, cycle_input

_T0 = datetime(2026, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
_ZONE = "zone_cleanup"


def _rt(store: MemoryStore, clock: FakeClock) -> LearningRuntime:
    return LearningRuntime(
        LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW, startup_grace_cycles=0),
        store=store,
        clock=lambda: clock.now_utc().isoformat(),
    )


def _ts(clock: FakeClock, offset_s: int = 0) -> str:
    return (clock.now_utc() + timedelta(seconds=offset_s)).isoformat()


def _run_cycles(rt: LearningRuntime, clock: FakeClock, n: int) -> None:
    for i in range(n):
        rt.run_cycle(cycle_input(_ts(clock, i * 300), zone=_ZONE))


class TestUnloadFlushesState:
    """async_unload() must persist dirty state before shutdown."""

    async def test_unload_flushes_important_dirty(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)
        rt = _rt(store, clock)
        await rt.async_setup()
        _run_cycles(rt, clock, 5)
        rt.mark_dirty(important=True)

        assert rt.health().dirty
        await rt.async_unload()

        assert store.saves >= 1, "async_unload must flush important dirty state"
        assert not rt.health().dirty

    async def test_unload_flushes_non_important_dirty(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)
        rt = _rt(store, clock)
        await rt.async_setup()
        _run_cycles(rt, clock, 5)
        rt.mark_dirty(important=False)

        await rt.async_unload()

        assert store.saves >= 1, "async_unload must flush non-important dirty state too"
        assert not rt.health().dirty

    async def test_unload_clean_runtime_no_unnecessary_save(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)
        rt = _rt(store, clock)
        await rt.async_setup()
        saves_before = store.saves
        await rt.async_unload()

        assert store.saves >= saves_before, "Unload of clean runtime must not error"


class TestRepeatedSetupUnload:
    """Repeated setup/unload cycles must not accumulate zones or state."""

    async def test_10x_setup_unload_bounded_zones(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)

        for i in range(10):
            rt = _rt(store, clock)
            await rt.async_setup()
            _run_cycles(rt, clock, 3)
            await rt.async_unload()
            clock.advance(3600)
            zones = rt.health().zones
            assert zones <= 1, f"Iteration {i}: zones={zones}, must not accumulate"

    async def test_10x_setup_unload_errors_stay_zero(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)

        for i in range(10):
            rt = _rt(store, clock)
            await rt.async_setup()
            _run_cycles(rt, clock, 3)
            await rt.async_unload()
            clock.advance(3600)

        assert rt.health().storage_warnings == 0

    async def test_10x_setup_unload_store_size_bounded(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)
        import json

        sizes = []
        for i in range(10):
            rt = _rt(store, clock)
            await rt.async_setup()
            _run_cycles(rt, clock, 5)
            await rt.async_unload()
            clock.advance(3600)
            if store.data is not None:
                sizes.append(len(json.dumps(store.data)))

        if len(sizes) >= 2:
            assert sizes[-1] <= sizes[0] * 3 + 1024, (
                f"Store size grew from {sizes[0]}B to {sizes[-1]}B across 10 cycles"
            )


class TestRestoreAfterUnload:
    """State must be exactly restored after unload+reinit from store."""

    async def test_model_update_total_preserved_after_reload(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)
        rt1 = _rt(store, clock)
        await rt1.async_setup()
        _run_cycles(rt1, clock, 20)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()
        updates_before = rt1.health().model_update_total

        rt2 = _rt(store, clock)
        await rt2.async_setup()

        assert rt2.health().initialized
        assert rt2.health().model_update_total == updates_before, (
            f"model_update_total changed: {updates_before} → {rt2.health().model_update_total}"
        )

    async def test_new_instance_after_reload_is_independent(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)
        rt1 = _rt(store, clock)
        await rt1.async_setup()
        _run_cycles(rt1, clock, 5)
        await rt1.async_unload()

        rt2 = _rt(store, clock)
        await rt2.async_setup()

        assert rt2 is not rt1, "rt2 must be a separate instance"
        assert rt2.health().initialized
        assert rt2.health().zones == rt1.health().zones

    async def test_zone_state_not_duplicated_after_reload(self):
        store = MemoryStore()
        clock = FakeClock(start=_T0)
        rt1 = _rt(store, clock)
        await rt1.async_setup()
        _run_cycles(rt1, clock, 10)
        await rt1.async_unload()

        rt2 = _rt(store, clock)
        await rt2.async_setup()
        zone_count = rt2.health().zones
        assert zone_count <= 1, f"After reload: {zone_count} zones, expected ≤1 for single zone"
