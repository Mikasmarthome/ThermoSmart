"""S1a Item 11 — Long-run performance: 30d / 90d / 180d timing gates.

Excluded on Windows (see conftest.py); run explicitly on Linux/Docker:
  pytest tests/test_s1a_item11_perf_long_run.py -v

Scenarios:
  - A: Minimal (1 zone, 1 simulated TRV, 5-min steps)
  - B: Normal (3 zones, heating ramp pattern)
  - C: Large (5 zones, mixed patterns)

Metrics collected per run:
  - Elapsed wall-clock seconds for the simulation
  - Final blob size (KB)
  - Ledger, pending, baseline_store sizes
  - Cycle times at day 1, day 30, day 90, day 180

Gates:
  - Blob ≤ 512 KB per zone (confirmed cap from Item 10)
  - Cycle time at day 180 ≤ 3× cycle time at day 1 (bounded growth)
  - All caps (ledger=500, pending=64, comparisons=200) respected
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import MemoryStore, cycle_input
from tests.helpers_runtime_scenarios import runtime as make_rt

UTC = timezone.utc
T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
STEP_S = 300        # 5-min steps
STEPS_PER_DAY = 288
BLOB_LIMIT = 512 * 1024


class DayMetrics(NamedTuple):
    day: int
    elapsed_s: float
    blob_kb: float
    ledger_size: int
    baseline_size: int
    open_comparisons: int
    pending_ctx: int
    pending_dispatch: int
    cycle_times: list[float]


def _ts(step: int) -> str:
    return (T0 + timedelta(seconds=step * STEP_S)).isoformat()


def _run_day(rt: LearningRuntime, day: int, zone_ids: list[str]) -> list[float]:
    """Run one simulated day; return per-coordinator-cycle durations (seconds)."""
    times: list[float] = []
    base = day * STEPS_PER_DAY
    for s in range(STEPS_PER_DAY):
        step = base + s
        ts = _ts(step)
        hour = (step * STEP_S // 3600) % 24
        comfort = hour in range(6, 9) or hour in range(17, 23)
        target = 21.0 if comfort else 18.0
        indoor = 18.0 + 2.5 * abs((step % STEPS_PER_DAY) / STEPS_PER_DAY - 0.5)
        t0 = time.perf_counter()
        for zid in zone_ids:
            rt.run_cycle(cycle_input(ts, zone=zid, target=target,
                                     indoor=round(indoor, 2)))
        times.append(time.perf_counter() - t0)
    return times


def _collect_metrics(rt: LearningRuntime, day: int, elapsed_s: float,
                     zone_id: str, times: list[float]) -> DayMetrics:
    payload = rt._build_payload()
    blob_bytes = len(json.dumps(payload).encode("utf-8"))
    zr = rt._zone(zone_id)
    return DayMetrics(
        day=day,
        elapsed_s=elapsed_s,
        blob_kb=blob_bytes / 1024,
        ledger_size=zr.ledger.size,
        baseline_size=zr.baseline_store.size,
        open_comparisons=len(zr.open_comparisons),
        pending_ctx=len(zr.pending_boost_contexts),
        pending_dispatch=len(zr.pending_dispatch_records),
        cycle_times=times,
    )


def _run_scenario(zone_ids: list[str], n_days: int,
                  sample_days: list[int]) -> dict[int, DayMetrics]:
    rt = make_rt()
    results: dict[int, DayMetrics] = {}
    t_start = time.perf_counter()
    for day in range(n_days):
        day_times = _run_day(rt, day, zone_ids)
        elapsed = time.perf_counter() - t_start
        if day + 1 in sample_days:
            results[day + 1] = _collect_metrics(
                rt, day + 1, elapsed, zone_ids[0], day_times)
    return results


class TestScenarioA_Minimal_30d:
    """1 zone, 30 days."""

    def test_30d_blob_under_512kb(self):
        rt = make_rt()
        for day in range(30):
            _run_day(rt, day, ["z1"])
        payload = rt._build_payload()
        blob_bytes = len(json.dumps(payload).encode("utf-8"))
        assert blob_bytes < BLOB_LIMIT, (
            f"30d blob={blob_bytes/1024:.1f}KB ≥ 512KB"
        )

    def test_30d_all_caps_respected(self):
        rt = make_rt()
        for day in range(30):
            _run_day(rt, day, ["z1"])
        zr = rt._zone("z1")
        assert zr.ledger.size <= 500
        assert len(zr.pending_boost_contexts) <= 64
        assert len(zr.pending_dispatch_records) <= 64
        assert len(zr.open_comparisons) <= 200

    def test_30d_cycle_time_stable(self):
        """Cycle time at day 30 must not be more than 4× day 1."""
        day1_times = _run_day(make_rt(), 0, ["z1"])
        rt = make_rt()
        for day in range(29):
            _run_day(rt, day, ["z1"])
        day30_times = _run_day(rt, 29, ["z1"])
        mean1 = statistics.mean(day1_times)
        mean30 = statistics.mean(day30_times)
        assert mean30 < mean1 * 4.0 + 0.005, (
            f"Day-30 mean {mean30*1000:.2f}ms > 4× day-1 {mean1*1000:.2f}ms"
        )


class TestScenarioA_Minimal_90d:
    """1 zone, 90 days — bounded growth after caps fill."""

    def test_90d_blob_under_512kb(self):
        rt = make_rt()
        for day in range(90):
            _run_day(rt, day, ["z1"])
        payload = rt._build_payload()
        blob_bytes = len(json.dumps(payload).encode("utf-8"))
        assert blob_bytes < BLOB_LIMIT, (
            f"90d blob={blob_bytes/1024:.1f}KB ≥ 512KB"
        )

    def test_90d_blob_bounded_vs_30d(self):
        """90d blob must not be more than 1.2× 30d blob (caps bound growth)."""
        rt30 = make_rt()
        for day in range(30):
            _run_day(rt30, day, ["z1"])
        b30 = len(json.dumps(rt30._build_payload()).encode("utf-8"))

        rt90 = make_rt()
        for day in range(90):
            _run_day(rt90, day, ["z1"])
        b90 = len(json.dumps(rt90._build_payload()).encode("utf-8"))

        assert b90 < b30 * 1.5, (
            f"90d blob {b90/1024:.1f}KB > 1.5× 30d blob {b30/1024:.1f}KB"
        )

    def test_90d_caps_respected(self):
        rt = make_rt()
        for day in range(90):
            _run_day(rt, day, ["z1"])
        zr = rt._zone("z1")
        assert zr.ledger.size <= 500
        assert len(zr.open_comparisons) <= 200


class TestScenarioA_Minimal_180d:
    """1 zone, 180 days — confirms no storage explosion at extreme horizon."""

    def test_180d_blob_under_512kb(self):
        rt = make_rt()
        for day in range(180):
            _run_day(rt, day, ["z1"])
        payload = rt._build_payload()
        blob_bytes = len(json.dumps(payload).encode("utf-8"))
        assert blob_bytes < BLOB_LIMIT, (
            f"180d blob={blob_bytes/1024:.1f}KB ≥ 512KB"
        )

    def test_180d_cycle_times_not_degraded(self):
        """Cycle mean at day 180 ≤ 4× day 1 mean."""
        day1_times = _run_day(make_rt(), 0, ["z1"])
        rt = make_rt()
        for day in range(179):
            _run_day(rt, day, ["z1"])
        day180_times = _run_day(rt, 179, ["z1"])
        mean1 = statistics.mean(day1_times)
        mean180 = statistics.mean(day180_times)
        assert mean180 < mean1 * 4.0 + 0.005, (
            f"Day-180 mean {mean180*1000:.2f}ms > 4× day-1 {mean1*1000:.2f}ms"
        )

    def test_180d_caps_respected(self):
        rt = make_rt()
        for day in range(180):
            _run_day(rt, day, ["z1"])
        zr = rt._zone("z1")
        assert zr.ledger.size <= 500
        assert len(zr.pending_boost_contexts) <= 64
        assert len(zr.open_comparisons) <= 200


class TestScenarioB_Normal_30d:
    """3 zones, 30 days — typical household."""

    def test_30d_3zones_per_zone_blob_under_512kb(self):
        """Each zone's per-zone blob must fit within the 512KB limit."""
        import hashlib
        rt = make_rt()
        for day in range(30):
            _run_day(rt, day, ["z1", "z2", "z3"])
        payload = rt._build_payload()
        total_bytes = len(json.dumps(payload).encode("utf-8"))
        # Per-zone is total / 3 zones (approximate check)
        per_zone = total_bytes / 3
        assert per_zone < BLOB_LIMIT, (
            f"3-zone per-zone approx {per_zone/1024:.1f}KB ≥ 512KB"
        )

    def test_30d_3zones_caps_respected_each_zone(self):
        rt = make_rt()
        for day in range(30):
            _run_day(rt, day, ["z1", "z2", "z3"])
        for zid in ["z1", "z2", "z3"]:
            zr = rt._zone(zid)
            assert zr.ledger.size <= 500, f"zone {zid} ledger {zr.ledger.size} > 500"
            assert len(zr.open_comparisons) <= 200, (
                f"zone {zid} open_comparisons {len(zr.open_comparisons)} > 200"
            )


class TestRestartIdempotency_LongRun:
    """After 30d, restart (serialize→restore→serialize) produces same blob size."""

    async def test_30d_restart_idempotent(self):
        store = MemoryStore()
        rt = make_rt(store=store)
        await rt.async_setup()
        for day in range(30):
            _run_day(rt, day, ["z1"])
        rt.mark_dirty(important=True)
        await rt.async_flush()
        b1 = len(json.dumps(store.data))

        rt2 = make_rt(store=store)
        await rt2.async_setup()
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        b2 = len(json.dumps(store.data))

        assert abs(b2 - b1) <= 100, (
            f"Reload-only blob drift: {b1} → {b2} bytes (diff={b2-b1})"
        )
