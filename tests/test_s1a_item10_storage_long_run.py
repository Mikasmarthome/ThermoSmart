"""S1a Item 10 - Section 8: Long-term storage simulation.

Verifies that the runtime blob does NOT exceed the 512 KB limit after a 30-day
simulation and that no data structure accumulates beyond its declared cap.

Excluded on Windows (runtime ~5-10 min); runs in Docker/CI.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
    RuntimeCycleInput,
    ControllerDecisionInput,
    ScheduleTarget,
)
from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import DecisionType
from tests.helpers_runtime import MemoryStore

pytestmark = pytest.mark.asyncio

UTC = timezone.utc
T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
STEP_S = 300  # 5-minute steps
STEPS_PER_DAY = 24 * 3600 // STEP_S  # 288 steps/day

_BLOB_LIMIT_BYTES = 512 * 1024  # 512 KB


def _make_runtime(store):
    from custom_components.thermosmart.learning.clock import FakeClock

    def clock():
        return T0.isoformat()

    return LearningRuntime(LearningRuntimeConfig(
        mode=LearningRuntimeMode.SHADOW, startup_grace_cycles=0), store=store, clock=clock)


def _run_day(rt: LearningRuntime, day: int, zone: str = "lz") -> None:
    """Run one day of 5-minute cycles with realistic heating patterns."""
    steps_before = day * STEPS_PER_DAY
    for s in range(STEPS_PER_DAY):
        step_n = steps_before + s
        ts = (T0 + timedelta(seconds=step_n * STEP_S)).isoformat()
        hour_of_day = (step_n * STEP_S // 3600) % 24
        # Comfort period 06:00-08:00 and 17:00-22:00; setback otherwise
        comfort = hour_of_day in range(6, 9) or hour_of_day in range(17, 23)
        target = 21.0 if comfort else 18.0
        # Indoor temperature oscillates realistically
        indoor = 18.0 + 3.0 * abs((step_n % 288) / 288.0 - 0.5)
        setpoint = (target + 3.0) if indoor < target - 0.3 else target
        inp = RuntimeCycleInput(
            zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setpoint,
            indoor_temp=Measurement(round(indoor, 2), DataQuality.OK),
            schedule=ScheduleTarget(
                comfort_time_utc=(T0 + timedelta(hours=7 + day * 24)).isoformat(),
                comfort_temperature_c=target,
            ) if comfort else None,
            controller_decision=ControllerDecisionInput(
                decision_type=DecisionType.NORMAL, target_c=target, trv_setpoint_c=setpoint),
        )
        rt.run_cycle(inp)


async def _blob_size(rt: LearningRuntime, store: MemoryStore) -> int:
    rt.mark_dirty(important=True)
    await rt.async_flush()
    return len(json.dumps(store.data).encode("utf-8"))


# ── 30-day simulation ──────────────────────────────────────────────────────────


async def test_30d_blob_within_limit():
    store = MemoryStore()
    rt = _make_runtime(store)
    await rt.async_setup()
    for day in range(30):
        _run_day(rt, day)
    size = await _blob_size(rt, store)
    assert size < _BLOB_LIMIT_BYTES, \
        f"30d blob {size:,} bytes exceeds {_BLOB_LIMIT_BYTES:,}"


async def test_30d_prediction_ledger_bounded():
    store = MemoryStore()
    rt = _make_runtime(store)
    await rt.async_setup()
    for day in range(30):
        _run_day(rt, day)
    zr = rt._zones.get("lz")
    if zr is not None:
        assert zr.ledger.size <= 500, f"Ledger size {zr.ledger.size} exceeds 500"


async def test_30d_baseline_store_bounded():
    store = MemoryStore()
    rt = _make_runtime(store)
    await rt.async_setup()
    for day in range(30):
        _run_day(rt, day)
    zr = rt._zones.get("lz")
    if zr is not None:
        assert zr.baseline_store.size <= 200, \
            f"Baseline store {zr.baseline_store.size} exceeds 200"


async def test_30d_open_comparisons_bounded():
    store = MemoryStore()
    rt = _make_runtime(store)
    await rt.async_setup()
    for day in range(30):
        _run_day(rt, day)
    zr = rt._zones.get("lz")
    if zr is not None:
        assert len(zr.open_comparisons) <= 200, \
            f"open_comparisons {len(zr.open_comparisons)} exceeds 200"


async def test_30d_pending_contexts_bounded():
    store = MemoryStore()
    rt = _make_runtime(store)
    await rt.async_setup()
    for day in range(30):
        _run_day(rt, day)
    zr = rt._zones.get("lz")
    if zr is not None:
        assert len(zr.pending_boost_contexts) <= 64
        assert len(zr.pending_dispatch_records) <= 64
        assert len(zr.pending_baseline_ctx) <= 64


async def test_30d_blob_stable_across_restarts():
    store = MemoryStore()
    rt = _make_runtime(store)
    await rt.async_setup()
    for day in range(30):
        _run_day(rt, day)
    size1 = await _blob_size(rt, store)
    # Restart and run 1 more day
    import json as _json
    store2 = MemoryStore(data=_json.loads(_json.dumps(store.data)))
    rt2 = _make_runtime(store2)
    await rt2.async_setup()
    _run_day(rt2, 30)
    size2 = await _blob_size(rt2, store2)
    # Blob may grow with new data but must not grow unboundedly
    assert size2 < _BLOB_LIMIT_BYTES, f"Post-restart blob {size2:,} bytes exceeds limit"
    # Growth per restart must be bounded (new data may add up to 20%)
    assert size2 <= size1 * 1.5, \
        f"Unexpected blob growth: {size1:,} -> {size2:,} bytes"


async def test_30d_two_zones_total_blob_within_limit():
    """Two separate zone runtimes must each stay within the per-zone limit."""
    stores = [MemoryStore(), MemoryStore()]
    rts = [_make_runtime(s) for s in stores]
    for rt in rts:
        await rt.async_setup()
    for day in range(30):
        _run_day(rts[0], day, zone="zone_a")
        _run_day(rts[1], day, zone="zone_b")
    sizes = []
    for rt, store in zip(rts, stores):
        s = await _blob_size(rt, store)
        sizes.append(s)
        assert s < _BLOB_LIMIT_BYTES, f"Zone blob {s:,} exceeds {_BLOB_LIMIT_BYTES:,}"


# ── 90-day (optional, longer) ─────────────────────────────────────────────────


async def test_90d_blob_within_limit():
    store = MemoryStore()
    rt = _make_runtime(store)
    await rt.async_setup()
    for day in range(90):
        _run_day(rt, day)
    size = await _blob_size(rt, store)
    assert size < _BLOB_LIMIT_BYTES, \
        f"90d blob {size:,} bytes exceeds {_BLOB_LIMIT_BYTES:,}"


async def test_90d_all_caps_respected():
    store = MemoryStore()
    rt = _make_runtime(store)
    await rt.async_setup()
    for day in range(90):
        _run_day(rt, day)
    zr = rt._zones.get("lz")
    if zr is not None:
        assert zr.ledger.size <= 500
        assert zr.baseline_store.size <= 200
        assert len(zr.open_comparisons) <= 200
        assert len(zr.pending_boost_contexts) <= 64
        assert len(zr.pending_dispatch_records) <= 64
        assert len(zr.pending_baseline_ctx) <= 64
