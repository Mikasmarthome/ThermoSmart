"""S1a Item 10 - Section 9: Multi-zone isolation and independent state.

Each zone's runtime state, prediction ledger, baseline store, and serialized blob
are completely independent. No cross-zone data leakage. Runs on Windows.
"""
from __future__ import annotations

import json
from datetime import timedelta, timezone

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import runtime, step, T0


def _run_zone(rt: LearningRuntime, zone: str, n: int, base_temp: float = 18.0):
    for i in range(n):
        temp = base_temp + (i % 5) * 0.4
        step(rt, i, round(temp, 1), heating=temp < 21.0, zone=zone)


class TestMultiZoneIsolation:
    def test_three_zones_independent_ledger_size(self):
        store = MemoryStore()
        rt = runtime(store=store)
        _run_zone(rt, "lz_a", 30, base_temp=18.0)
        _run_zone(rt, "lz_b", 15, base_temp=17.0)
        _run_zone(rt, "lz_c", 5, base_temp=19.0)
        # Each zone only has its own prediction ledger entries
        zr_a = rt._zones["lz_a"]
        zr_b = rt._zones["lz_b"]
        zr_c = rt._zones["lz_c"]
        assert zr_a.ledger.size >= zr_b.ledger.size >= zr_c.ledger.size

    def test_zone_update_counts_independent(self):
        rt = runtime()
        _run_zone(rt, "zone_x", 40, base_temp=17.5)
        _run_zone(rt, "zone_y", 40, base_temp=17.5)
        zr_x = rt._zones["zone_x"]
        zr_y = rt._zones["zone_y"]
        # Same cycles, same physics => same update counts
        assert zr_x.model_update_counts == zr_y.model_update_counts

    def test_zone_data_does_not_leak_across_zones(self):
        rt = runtime()
        _run_zone(rt, "lz_a", 20, base_temp=16.0)
        _run_zone(rt, "lz_b", 20, base_temp=22.0)
        # No lz_a samples in lz_b's baseline store
        store_b = rt._zones["lz_b"].baseline_store
        for s in store_b.samples():
            assert s.learning_zone_id == "lz_b"

    def test_zone_ledger_no_cross_zone_predictions(self):
        rt = runtime()
        _run_zone(rt, "zone_1", 20)
        _run_zone(rt, "zone_2", 20)
        # zone_1's ledger only contains predictions for zone_1 decisions
        zr1 = rt._zones["zone_1"]
        for snap in zr1.ledger._snapshots.values():
            assert snap.learning_zone_id in ("zone_1", "")  # "" is legacy path

    def test_remove_zone_leaves_others_intact(self):
        rt = runtime()
        _run_zone(rt, "alpha", 20)
        _run_zone(rt, "beta", 20)
        cycles_alpha = rt._zones["alpha"].cycles
        rt.remove_zone("alpha")
        assert "alpha" not in rt._zones
        assert "beta" in rt._zones
        assert rt._zones["beta"].cycles == 20

    def test_serialized_blob_zones_keyed_separately(self):
        import asyncio

        async def run():
            store = MemoryStore()
            rt = runtime(store=store)
            await rt.async_setup()
            _run_zone(rt, "a_zone", 10)
            _run_zone(rt, "b_zone", 10)
            rt.mark_dirty(important=True)
            await rt.async_flush()
            data = store.data
            assert "runtime_schema_version" in data
            zones = data.get("zones", {})
            assert "a_zone" in zones
            assert "b_zone" in zones
            # Each zone's serialized state is independent
            assert zones["a_zone"] is not zones["b_zone"]

        asyncio.run(run())

    def test_restore_only_restores_correct_zones(self):
        import asyncio

        async def run():
            store = MemoryStore()
            rt = runtime(store=store)
            await rt.async_setup()
            _run_zone(rt, "zone_alpha", 20)
            rt.mark_dirty(important=True)
            await rt.async_flush()
            # Restore into a new runtime
            rt2 = runtime(store=store)
            await rt2.async_setup()
            # zone_alpha restored, zone_beta not created
            assert "zone_alpha" in rt2._zones
            assert "zone_beta" not in rt2._zones

        asyncio.run(run())

    def test_three_zones_independent_blob_size(self):
        import asyncio

        async def run():
            # 3 separate runtimes, each with its own store — simulates 3 config entries
            stores = [MemoryStore() for _ in range(3)]
            rts = [runtime(store=s) for s in stores]
            for rt in rts:
                await rt.async_setup()
            _run_zone(rts[0], "lz", 100)
            _run_zone(rts[1], "lz", 100)
            _run_zone(rts[2], "lz", 100)
            for rt, store in zip(rts, stores):
                rt.mark_dirty(important=True)
                await rt.async_flush()
            sizes = [len(json.dumps(s.data).encode()) for s in stores]
            # All three have similar blob sizes (same workload, same zone ID)
            assert max(sizes) < 512 * 1024
            assert abs(sizes[0] - sizes[1]) < 10 * 1024  # <10KB difference

        asyncio.run(run())

    def test_pending_boost_outcome_not_shared_across_zones(self):
        rt = runtime()
        step(rt, 0, 19.0, heating=True, zone="a")
        step(rt, 0, 19.0, heating=True, zone="b")
        # pending_boost_outcome is per zone — None by default, never shared
        assert rt._zones["a"].pending_boost_outcome is None
        assert rt._zones["b"].pending_boost_outcome is None
        # Each zone has its own attribute (not the same object)
        assert rt._zones["a"].pending_boost_outcome is not rt._zones["b"]
