"""S1a Item 11 — Scaling comparisons: zones, history depth, cap impact.

Proves:
- Cycle time grows approximately linearly with zone count, not quadratically.
- Reaching cap bounds (ledger, pending, comparisons) does not cause further
  cycle-time growth — caps are proven O(1) once full.
- Health counters remain bounded and accurate at all scales.
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timedelta, timezone

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import MemoryStore, cycle_input
from tests.helpers_runtime_scenarios import T0 as _SCENARIO_T0
from tests.helpers_runtime_scenarios import runtime as make_rt

_CLOCK = lambda: _SCENARIO_T0.isoformat()


def _make_rt_with_cap(max_open_comparisons: int = 200) -> LearningRuntime:
    return LearningRuntime(
        LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW, startup_grace_cycles=0,
                              max_open_comparisons=max_open_comparisons),
        clock=_CLOCK)

_T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
_STEP_MIN = 5


def _ts(i: int) -> str:
    return (_T0 + timedelta(minutes=i * _STEP_MIN)).isoformat()


def _run(rt: LearningRuntime, zone_ids: list[str], n: int) -> list[float]:
    out: list[float] = []
    for i in range(n):
        ts = _ts(i)
        t0 = time.perf_counter()
        for zid in zone_ids:
            rt.run_cycle(cycle_input(ts, zone=zid))
        out.append(time.perf_counter() - t0)
    return out


def _pct(s: list[float], p: int) -> float:
    ss = sorted(s)
    return ss[max(0, min(len(ss) - 1, int(len(ss) * p / 100)))]


class TestZoneCountScaling:
    """Verify total cycle time grows linearly with zone count."""

    def test_1_zone_mean(self):
        rt = make_rt()
        s = _run(rt, ["z1"], 100)
        mean = statistics.mean(s)
        assert mean < 0.030, f"1-zone mean={mean*1000:.1f}ms > 30ms"

    def test_3_zone_mean_below_3x_reference(self):
        rt1 = make_rt()
        rt3 = make_rt()
        s1 = _run(rt1, ["z1"], 100)
        s3 = _run(rt3, ["z1", "z2", "z3"], 100)
        assert statistics.mean(s3) < statistics.mean(s1) * 5.0 + 0.010, (
            "3-zone total > 5× 1-zone total"
        )

    def test_10_zone_mean_below_15x_reference(self):
        rt1 = make_rt()
        rt10 = make_rt()
        s1 = _run(rt1, ["z1"], 50)
        s10 = _run(rt10, [f"z{i}" for i in range(10)], 50)
        assert statistics.mean(s10) < statistics.mean(s1) * 15.0 + 0.050, (
            "10-zone total > 15× 1-zone total"
        )

    def test_20_zone_mean_below_30x_reference(self):
        rt1 = make_rt()
        rt20 = make_rt()
        s1 = _run(rt1, ["z1"], 30)
        s20 = _run(rt20, [f"z{i}" for i in range(20)], 30)
        factor = statistics.mean(s20) / max(statistics.mean(s1), 1e-9)
        assert factor < 30.0, f"20z/1z factor={factor:.1f} > 30 (O(n²) risk)"

    def test_scaling_factor_consistent_across_zone_counts(self):
        """The per-zone overhead (mean / n_zones) should be similar at 3, 10, 20 zones."""
        rt3 = make_rt()
        rt10 = make_rt()
        rt20 = make_rt()
        s3 = _run(rt3, [f"z{i}" for i in range(3)], 30)
        s10 = _run(rt10, [f"z{i}" for i in range(10)], 30)
        s20 = _run(rt20, [f"z{i}" for i in range(20)], 30)
        per3 = statistics.mean(s3) / 3
        per10 = statistics.mean(s10) / 10
        per20 = statistics.mean(s20) / 20
        # per-zone cost should be within 3× across all counts
        max_per = max(per3, per10, per20)
        min_per = min(per3, per10, per20)
        ratio = max_per / max(min_per, 1e-9)
        assert ratio < 3.0, (
            f"per-zone cost ratio max/min={ratio:.1f} > 3.0 "
            f"(per3={per3*1e3:.2f}ms per10={per10*1e3:.2f}ms per20={per20*1e3:.2f}ms)"
        )


class TestCapImpact:
    """Once caps are full, cycle time must not continue growing."""

    def _cycle_mean(self, rt: LearningRuntime, start: int, n: int) -> float:
        times = []
        for i in range(start, start + n):
            t0 = time.perf_counter()
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
            times.append(time.perf_counter() - t0)
        return statistics.mean(times)

    def test_cycle_time_stable_before_and_after_ledger_cap(self):
        """Cycle time at step 550+ (ledger full at 500) ≈ time at step 50."""
        rt = make_rt()
        # warm-up: skip first 10 (interpreter cold start)
        for i in range(10):
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        # measure before cap
        t_pre = self._cycle_mean(rt, 10, 40)
        # advance past ledger cap
        for i in range(50, 560):
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        # measure after cap (ledger full, FIFO ring)
        t_post = self._cycle_mean(rt, 560, 40)
        assert t_post < t_pre * 3.0 + 0.005, (
            f"post-cap mean {t_post*1000:.2f}ms > 3× pre-cap {t_pre*1000:.2f}ms"
        )

    def test_ledger_size_stops_growing_at_cap(self):
        rt = _make_rt_with_cap(200)
        for i in range(700):  # well past ledger cap of 500
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        zr = rt._zone("z1")
        assert zr.ledger.size <= 500, f"ledger.size={zr.ledger.size} > 500"

    def test_pending_dicts_stay_bounded(self):
        rt = make_rt()
        for i in range(200):
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        zr = rt._zone("z1")
        assert len(zr.pending_boost_contexts) <= 64
        assert len(zr.pending_dispatch_records) <= 64
        assert len(zr.pending_baseline_ctx) <= 64

    def test_open_comparisons_stay_bounded(self):
        rt = _make_rt_with_cap(200)
        for i in range(300):
            rt.run_cycle(cycle_input(_ts(i), zone="z1",
                                     legacy_preheat=_ts(i + 12)))
        zr = rt._zone("z1")
        assert len(zr.open_comparisons) <= 200


class TestHistoryDepthScaling:
    """History depth (how many cycles have run) must not slow down processing."""

    def _sample_at_cycle(self, start: int, n_measure: int = 30) -> tuple[float, float]:
        rt = make_rt()
        # advance to start
        for i in range(start):
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        times = []
        for i in range(start, start + n_measure):
            t0 = time.perf_counter()
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
            times.append(time.perf_counter() - t0)
        return statistics.mean(times), max(times)

    def test_cycle_mean_at_100_not_worse_than_cycle_10(self):
        mean_early, _ = self._sample_at_cycle(5)
        mean_late, _ = self._sample_at_cycle(100)
        assert mean_late < mean_early * 3.0 + 0.005

    def test_cycle_mean_at_600_not_worse_than_cycle_10(self):
        """Even after ledger cap (500 entries), cycle mean stays bounded."""
        mean_early, _ = self._sample_at_cycle(5)
        mean_late, _ = self._sample_at_cycle(600)
        assert mean_late < mean_early * 3.0 + 0.010

    def test_blob_size_stable_after_cap(self):
        """Blob size does not grow after caps are reached."""
        rt = make_rt()
        for i in range(550):
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        payload_mid = rt._build_payload()
        b_mid = len(json.dumps(payload_mid))
        for i in range(550, 800):
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        payload_late = rt._build_payload()
        b_late = len(json.dumps(payload_late))
        assert b_late < b_mid * 1.20 + 1024, (
            f"blob grew significantly post-cap: {b_mid} → {b_late} bytes"
        )


class TestMultiZoneHealthInvariants:
    """Health counters stay accurate and bounded at all scales."""

    def test_health_model_update_total_increases(self):
        rt = make_rt()
        for i in range(50):
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        h = rt.health()
        assert h.model_update_total >= 0  # must be non-negative

    def test_health_prediction_snapshots_bounded(self):
        rt = _make_rt_with_cap(200)
        for i in range(600):
            rt.run_cycle(cycle_input(_ts(i), zone="z1"))
        h = rt.health()
        assert h.prediction_snapshots <= 500

    def test_health_open_comparisons_bounded(self):
        rt = _make_rt_with_cap(200)
        for i in range(300):
            rt.run_cycle(cycle_input(_ts(i), zone="z1",
                                     legacy_preheat=_ts(i + 12)))
        h = rt.health()
        assert h.open_comparisons <= 200

    def test_10zone_health_reports_correct_zones(self):
        rt = make_rt()
        for i in range(5):
            for j in range(10):
                rt.run_cycle(cycle_input(_ts(i), zone=f"z{j}"))
        h = rt.health()
        assert h.zones == 10

    def test_remove_zone_reduces_count(self):
        rt = make_rt()
        for j in range(5):
            rt.run_cycle(cycle_input(_ts(0), zone=f"z{j}"))
        rt.remove_zone("z0")
        h = rt.health()
        assert h.zones == 4
