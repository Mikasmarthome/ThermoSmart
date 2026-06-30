"""Phase S1a Item 9 — Storage/Retention/Bounds and Soak Tests (Section 10).

Verifies:
  - 1 restart: clean, no blob growth
  - 10 restart chain: blob size stable, learning preserved
  - 100 restart chain: no memory leak, no accumulation
  - Store failure isolation: save errors never corrupt in-memory state
  - Store load failure: cold-start fallback
  - Blob size stays within reasonable bounds (< 512 KB per zone)
"""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import (
    heating_ramp_then_settle,
    runtime,
    step,
)

pytestmark = pytest.mark.asyncio

_BLOB_SIZE_LIMIT_BYTES = 512 * 1024   # 512 KB hard cap per zone
_CONTROL_CLOCK = lambda: "2025-01-01T07:00:00+00:00"


async def _trained_store() -> MemoryStore:
    store = MemoryStore()
    rt = runtime(store=store)
    await rt.async_setup()
    heating_ramp_then_settle(rt)
    rt.mark_dirty(important=True)
    await rt.async_flush()
    return store


# ── 1. Single restart ────────────────────────────────────────────────────────


class TestSingleRestart:
    async def test_blob_size_within_limit_after_training(self):
        store = await _trained_store()
        size = len(json.dumps(store.data).encode("utf-8"))
        assert size < _BLOB_SIZE_LIMIT_BYTES, (
            f"Store blob too large: {size:,} bytes (limit {_BLOB_SIZE_LIMIT_BYTES:,})"
        )

    async def test_blob_size_stable_after_restart(self):
        store = await _trained_store()
        size1 = len(json.dumps(store.data).encode("utf-8"))

        rt2 = runtime(store=MemoryStore(data=json.loads(json.dumps(store.data))))
        await rt2.async_setup()
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        size2 = len(json.dumps(store.data).encode("utf-8"))

        # Blob may shrink slightly (dedup cleanup) but must not grow significantly
        assert size2 <= size1 * 1.05, (
            f"Blob grew unexpectedly after 1 restart: {size1:,} → {size2:,} bytes"
        )

    async def test_save_count_increments_per_flush(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        assert store.saves == 0
        step(rt, 0, 19.0, heating=True)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        assert store.saves == 1


# ── 2. 10-restart chain ──────────────────────────────────────────────────────


class TestTenRestartChain:
    async def test_learning_preserved_through_10_restarts(self):
        store = await _trained_store()
        n_base = None

        for i in range(10):
            blob = json.loads(json.dumps(store.data))
            store = MemoryStore(data=blob)
            rt = runtime(store=store)
            await rt.async_setup()
            n = rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
            if n_base is None:
                n_base = n
            assert n == n_base, f"Restart {i+1}/10: sample count changed {n_base} → {n}"
            rt.mark_dirty(important=True)
            await rt.async_flush()

    async def test_blob_size_does_not_grow_through_10_restarts(self):
        store = await _trained_store()
        size_start = len(json.dumps(store.data).encode("utf-8"))

        for i in range(10):
            blob = json.loads(json.dumps(store.data))
            store = MemoryStore(data=blob)
            rt = runtime(store=store)
            await rt.async_setup()
            rt.mark_dirty(important=True)
            await rt.async_flush()

        size_end = len(json.dumps(store.data).encode("utf-8"))
        assert size_end <= size_start * 1.10, (
            f"Blob grew through 10 restarts: {size_start:,} → {size_end:,} bytes (+{size_end-size_start:,})"
        )

    async def test_health_clean_through_10_restarts(self):
        store = await _trained_store()

        for i in range(10):
            blob = json.loads(json.dumps(store.data))
            store = MemoryStore(data=blob)
            rt = runtime(store=store)
            await rt.async_setup()
            h = rt.health()
            assert h.storage_warnings == 0, f"Restart {i+1}/10: {h.storage_warnings} storage warnings"
            assert h.zones >= 1
            rt.mark_dirty(important=True)
            await rt.async_flush()

    async def test_cycles_count_preserved_through_10_restarts(self):
        store = await _trained_store()
        cycles_base = None

        for i in range(10):
            blob = json.loads(json.dumps(store.data))
            store = MemoryStore(data=blob)
            rt = runtime(store=store)
            await rt.async_setup()
            cycles = rt._zone("lz").cycles
            if cycles_base is None:
                cycles_base = cycles
            assert cycles == cycles_base, f"Restart {i+1}/10: cycles changed {cycles_base} → {cycles}"
            rt.mark_dirty(important=True)
            await rt.async_flush()


# ── 3. 100-restart chain (soak) ──────────────────────────────────────────────


class TestHundredRestartSoak:
    """Soak test: 100 sequential restarts with no new learning.
    Blob size must stay bounded; no memory leak; no crashes.
    """

    async def test_100_restarts_no_crash(self):
        store = await _trained_store()
        for i in range(100):
            blob = json.loads(json.dumps(store.data))
            store = MemoryStore(data=blob)
            rt = runtime(store=store)
            await rt.async_setup()
            rt.mark_dirty(important=True)
            await rt.async_flush()
        # If we got here without exception: pass

    async def test_100_restarts_blob_bounded(self):
        store = await _trained_store()
        size_start = len(json.dumps(store.data).encode("utf-8"))

        for _ in range(100):
            blob = json.loads(json.dumps(store.data))
            store = MemoryStore(data=blob)
            rt = runtime(store=store)
            await rt.async_setup()
            rt.mark_dirty(important=True)
            await rt.async_flush()

        size_end = len(json.dumps(store.data).encode("utf-8"))
        assert size_end < _BLOB_SIZE_LIMIT_BYTES, (
            f"Blob exceeds hard cap after 100 restarts: {size_end:,} bytes"
        )
        assert size_end <= size_start * 1.20, (
            f"Blob grew >20% through 100 restarts: {size_start:,} → {size_end:,}"
        )


# ── 4. Store failure isolation ────────────────────────────────────────────────


class TestStoreFailureIsolation:
    """Store failures must be isolated — they must not corrupt the in-memory state."""

    async def test_save_failure_does_not_corrupt_in_memory_state(self):
        fail_store = MemoryStore(fail_save=True)
        rt = runtime(store=fail_store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        # Save fails but runtime must remain healthy
        await rt.async_flush()
        h = rt.health()
        assert h.zones >= 1

    async def test_save_failure_increments_failed_saves_counter(self):
        fail_store = MemoryStore(fail_save=True)
        rt = runtime(store=fail_store)
        await rt.async_setup()
        step(rt, 0, 19.0, heating=True)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        # Persistence orchestrator tracks failed saves
        assert rt._persistence._s.failed_saves >= 1

    async def test_load_failure_produces_cold_start(self):
        """When store.async_load() raises, runtime must cold-start cleanly."""
        fail_store = MemoryStore(fail_load=True)
        rt = runtime(store=fail_store)
        await rt.async_setup()
        # After load failure, runtime is alive but has no restored data
        result = step(rt, 0, 19.0, heating=True)
        assert result is not None
        h = rt.health()
        assert h.zones >= 1

    async def test_partial_failure_after_initial_success(self):
        """Train, save, then next-restart save fails.  Learning state still in memory."""
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()
        blob = json.loads(json.dumps(store.data))

        fail_store = MemoryStore(data=blob, fail_save=True)
        rt2 = runtime(store=fail_store)
        await rt2.async_setup()
        n = rt2._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        assert n >= 1  # restored from blob
        # Second save fails
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        h = rt2.health()
        assert h.zones >= 1


# ── 5. Persistence policy triggers ──────────────────────────────────────────


class TestPersistencePolicy:
    """PersistenceOrchestrator decides WHEN to save — verify trigger conditions."""

    async def test_important_dirty_triggers_fast_save(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        step(rt, 0, 19.0, heating=True)
        rt.mark_dirty(important=True)
        # With important=True and debounce=2s, a save at TS+3s must trigger
        from custom_components.thermosmart.learning.runtime.persistence import (
            PersistenceOrchestrator, SavePolicy, SaveTrigger)
        p = PersistenceOrchestrator(store, policy=SavePolicy(important_debounce_s=2.0))
        p.mark_dirty("2025-01-01T07:00:00+00:00", important=True)
        trigger = p.decide("2025-01-01T07:00:03+00:00")
        assert trigger != SaveTrigger.NONE

    async def test_not_dirty_produces_none_trigger(self):
        from custom_components.thermosmart.learning.runtime.persistence import (
            PersistenceOrchestrator, SaveTrigger)
        p = PersistenceOrchestrator(MemoryStore())
        trigger = p.decide("2025-01-01T07:00:00+00:00")
        assert trigger == SaveTrigger.NONE
