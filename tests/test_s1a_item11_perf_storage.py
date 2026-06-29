"""S1a Item 11 — Storage timing: serialize/deserialize/flush/no-op gates.

Validates that serialize(), restore(), async_flush(), and maybe_save() all
stay within generous bounds regardless of zone count or history depth.
All tests are pure Python, no HA fixtures required.
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timedelta, timezone

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    PersistenceOrchestrator,
    SavePolicy,
)
from tests.helpers_runtime import MemoryStore, cycle_input
from tests.helpers_runtime_scenarios import runtime as make_rt

_T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
_STEP_MIN = 5
_BLOB_LIMIT = 512 * 1024


def _ts(i: int) -> str:
    return (_T0 + timedelta(minutes=i * _STEP_MIN)).isoformat()


def _populate(rt: LearningRuntime, zone_ids: list[str], n_cycles: int) -> None:
    for i in range(n_cycles):
        ts = _ts(i)
        for zid in zone_ids:
            rt.run_cycle(cycle_input(ts, zone=zid))


def _serialize_one_zone(rt: LearningRuntime, zone_id: str) -> dict:
    return rt._zone(zone_id).serialize()


def _blob_bytes(rt: LearningRuntime) -> int:
    payload = rt._build_payload()
    return len(json.dumps(payload).encode("utf-8"))


class TestSerializeTiming:
    """_ZoneRuntime.serialize() must be fast at all history depths."""

    def test_serialize_1zone_p99_below_200ms(self):
        rt = make_rt()
        _populate(rt, ["z1"], 200)
        samples = []
        for _ in range(50):
            t0 = time.perf_counter()
            _serialize_one_zone(rt, "z1")
            samples.append(time.perf_counter() - t0)
        s = sorted(samples)
        p99 = s[min(len(s) - 1, int(len(s) * 0.99))]
        assert p99 < 0.200, f"serialize p99={p99*1000:.1f}ms > 200ms"

    def test_serialize_post_cap_not_slower_than_pre_cap(self):
        """After caps fill (ledger=500, pending=64), serialize must not slow down."""
        rt_pre = make_rt()
        rt_post = make_rt()
        _populate(rt_pre, ["z1"], 50)
        _populate(rt_post, ["z1"], 700)  # fills ledger cap (500 entries)

        def _mean_serialize(rt: LearningRuntime, n: int = 30) -> float:
            times = []
            for _ in range(n):
                t0 = time.perf_counter()
                _serialize_one_zone(rt, "z1")
                times.append(time.perf_counter() - t0)
            return statistics.mean(times)

        t_pre = _mean_serialize(rt_pre)
        t_post = _mean_serialize(rt_post)
        assert t_post < t_pre * 3.0 + 0.010, (
            f"post-cap serialize {t_post*1000:.1f}ms > 3× pre-cap {t_pre*1000:.1f}ms"
        )

    def test_serialize_10zones_linear_vs_1zone(self):
        """Serializing 10 zones should scale approximately linearly with zone count."""
        rt = make_rt()
        _populate(rt, [f"z{i}" for i in range(10)], 100)

        def _mean_serialize_all(n: int = 20) -> float:
            times = []
            for _ in range(n):
                t0 = time.perf_counter()
                for i in range(10):
                    _serialize_one_zone(rt, f"z{i}")
                times.append(time.perf_counter() - t0)
            return statistics.mean(times)

        rt1 = make_rt()
        _populate(rt1, ["z1"], 100)

        def _mean_serialize_1(n: int = 20) -> float:
            times = []
            for _ in range(n):
                t0 = time.perf_counter()
                _serialize_one_zone(rt1, "z1")
                times.append(time.perf_counter() - t0)
            return statistics.mean(times)

        t10 = _mean_serialize_all()
        t1 = _mean_serialize_1()
        # 10-zone total should be < 15× 1-zone (linear=10, allow 50%)
        assert t10 < t1 * 15.0 + 0.020, (
            f"10-zone total serialize {t10*1000:.1f}ms > 15× 1-zone {t1*1000:.1f}ms"
        )

    def test_blob_bytes_below_limit_1zone(self):
        rt = make_rt()
        _populate(rt, ["z1"], 300)
        size = _blob_bytes(rt)
        assert size < _BLOB_LIMIT, f"1-zone blob {size/1024:.1f}KB > 512KB"


class TestDeserializeTiming:
    """_ZoneRuntime.restore() must be fast regardless of history depth."""

    async def test_restore_1zone_p99_below_200ms(self):
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        _populate(rt, ["z1"], 200)
        rt.mark_dirty(important=True)
        await rt.async_flush()

        raw = store.data
        samples = []
        for _ in range(30):
            rt2 = make_rt(store=store)
            t0 = time.perf_counter()
            await rt2.async_setup()
            samples.append(time.perf_counter() - t0)

        s = sorted(samples)
        p99 = s[min(len(s) - 1, int(len(s) * 0.99))]
        assert p99 < 0.200, f"restore p99={p99*1000:.1f}ms > 200ms"

    async def test_restore_10zones_p99_below_2000ms(self):
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        zones = [f"z{i}" for i in range(10)]
        _populate(rt, zones, 100)
        rt.mark_dirty(important=True)
        await rt.async_flush()

        samples = []
        for _ in range(10):
            rt2 = make_rt(store=store)
            t0 = time.perf_counter()
            await rt2.async_setup()
            samples.append(time.perf_counter() - t0)

        s = sorted(samples)
        p99 = s[min(len(s) - 1, int(len(s) * 0.99))]
        assert p99 < 2.000, f"10-zone restore p99={p99*1000:.1f}ms > 2000ms"

    async def test_restore_scales_linearly_with_zones(self):
        """10-zone restore < 15× 1-zone restore (linear=10, allow 50%)."""
        store1 = MemoryStore()
        rt1 = make_rt(store=store1)
        await rt1.async_setup()
        _populate(rt1, ["z1"], 100)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        store10 = MemoryStore()
        rt10 = make_rt(store=store10)
        await rt10.async_setup()
        _populate(rt10, [f"z{i}" for i in range(10)], 100)
        rt10.mark_dirty(important=True)
        await rt10.async_flush()

        async def _mean_restore(store: MemoryStore, n: int = 10) -> float:
            times = []
            for _ in range(n):
                rt = make_rt(store=store)
                t0 = time.perf_counter()
                await rt.async_setup()
                times.append(time.perf_counter() - t0)
            return statistics.mean(times)

        t1 = await _mean_restore(store1)
        t10 = await _mean_restore(store10)
        assert t10 < t1 * 15.0 + 0.050, (
            f"10-zone restore {t10*1000:.1f}ms > 15× 1-zone {t1*1000:.1f}ms"
        )

    async def test_restore_idempotent(self):
        """Restoring the same blob twice yields the same blob size."""
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        _populate(rt, ["z1"], 100)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        blob1 = json.dumps(store.data)

        rt2 = make_rt(store=store)
        await rt2.async_setup()
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        blob2 = json.dumps(store.data)

        assert len(blob2) == len(blob1), (
            f"Restore+flush changed blob size: {len(blob1)} → {len(blob2)} bytes"
        )


class TestFlushTiming:
    """async_flush() and maybe_save() performance."""

    async def test_flush_empty_runtime_below_100ms(self):
        """Flushing a runtime with no cycles is very cheap."""
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        t0 = time.perf_counter()
        await rt.async_flush()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.100, f"empty flush took {elapsed*1000:.1f}ms > 100ms"

    async def test_flush_1zone_200cycles_below_2000ms(self):
        """Dirty flush with 1 zone at 200 cycles stays within generous bound."""
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        _populate(rt, ["z1"], 200)
        rt.mark_dirty(important=True)
        t0 = time.perf_counter()
        await rt.async_flush()
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.000, f"1-zone flush took {elapsed*1000:.1f}ms > 2000ms"

    async def test_flush_10zones_below_10000ms(self):
        """Dirty flush with 10 zones stays bounded."""
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        _populate(rt, [f"z{i}" for i in range(10)], 100)
        rt.mark_dirty(important=True)
        t0 = time.perf_counter()
        await rt.async_flush()
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.000, f"10-zone flush took {elapsed*1000:.1f}ms > 10000ms"

    async def test_repeated_flush_is_idempotent(self):
        """Calling async_flush() twice produces the same byte count."""
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        _populate(rt, ["z1"], 50)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        b1 = len(json.dumps(store.data))
        rt.mark_dirty(important=True)
        await rt.async_flush()
        b2 = len(json.dumps(store.data))
        assert b1 == b2, f"repeated flush changed blob: {b1} → {b2} bytes"


class TestNoopSave:
    """maybe_save() when not dirty must not call payload_fn and must be cheap."""

    async def test_maybe_save_noop_skips_payload(self):
        store = MemoryStore()
        po = PersistenceOrchestrator(store, policy=SavePolicy(debounce_s=30))
        calls = {"n": 0}

        def payload():
            calls["n"] += 1
            return {}

        await po.maybe_save("2025-01-01T06:00:00+00:00", payload)
        assert calls["n"] == 0, "payload_fn must not be called when not dirty"

    async def test_maybe_save_noop_below_10ms(self):
        store = MemoryStore()
        po = PersistenceOrchestrator(store, policy=SavePolicy(debounce_s=30))

        t0 = time.perf_counter()
        for _ in range(100):
            await po.maybe_save("2025-01-01T06:00:00+00:00", lambda: {})
        elapsed = (time.perf_counter() - t0) / 100

        assert elapsed < 0.010, f"no-op save mean={elapsed*1000:.2f}ms > 10ms"
        assert store.saves == 0

    async def test_dirty_save_executes_payload(self):
        store = MemoryStore()
        po = PersistenceOrchestrator(store, policy=SavePolicy(debounce_s=30))
        po.mark_dirty("2025-01-01T05:00:00+00:00")
        calls = {"n": 0}

        def payload():
            calls["n"] += 1
            return {"x": 1}

        await po.maybe_save("2025-01-01T06:00:00+00:00", payload)
        assert calls["n"] == 1, "payload_fn must be called when dirty"
        assert store.saves == 1


class TestBlobScaling:
    """Blob size must scale linearly with zones, not super-linearly."""

    async def test_blob_1zone_under_512kb(self):
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        _populate(rt, ["z1"], 300)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        size = len(json.dumps(store.data).encode("utf-8"))
        assert size < _BLOB_LIMIT, f"1-zone blob {size/1024:.1f}KB ≥ 512KB"

    async def test_blob_5zones_under_5x_limit(self):
        """5-zone blob < 5 × 512 KB."""
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        _populate(rt, [f"z{i}" for i in range(5)], 300)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        size = len(json.dumps(store.data).encode("utf-8"))
        assert size < 5 * _BLOB_LIMIT, (
            f"5-zone blob {size/1024:.1f}KB ≥ {5*512}KB"
        )

    async def test_blob_scales_roughly_linearly_with_zones(self):
        """10-zone blob < 15× 1-zone blob (linear=10, allow 50%)."""
        async def _blob(n_zones: int) -> int:
            store = MemoryStore()
            rt = make_rt(store=store)
            await rt.async_setup()
            _populate(rt, [f"z{i}" for i in range(n_zones)], 100)
            rt.mark_dirty(important=True)
            await rt.async_flush()
            return len(json.dumps(store.data).encode("utf-8"))

        b1 = await _blob(1)
        b10 = await _blob(10)
        assert b10 < b1 * 15, (
            f"10-zone blob {b10/1024:.0f}KB > 15× 1-zone {b1/1024:.0f}KB"
        )
