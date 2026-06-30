"""S1a Item 11 — Cycle timing: p95/p99 gates, scaling linearity.

Tests LearningRuntime.run_cycle() performance across scenarios A–D
(minimal → normal → large → stress). Gates are conservative (10–20×
buffer over measured 5-min-step timing) and p95/p99-based to tolerate
GC spikes. All tests are pure Python, no HA fixtures required.
"""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timedelta, timezone

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import cycle_input
from tests.helpers_runtime_scenarios import runtime as make_rt

_T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
_STEP_MIN = 5


def _ts(i: int) -> str:
    return (_T0 + timedelta(minutes=i * _STEP_MIN)).isoformat()


def _pct(samples: list[float], p: int) -> float:
    s = sorted(samples)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[idx]


def _run_cycles(rt: LearningRuntime, zone_ids: list[str], n: int) -> list[float]:
    """Run n coordinator cycles; return per-cycle total duration (seconds)."""
    out: list[float] = []
    for i in range(n):
        ts = _ts(i)
        t0 = time.perf_counter()
        for zid in zone_ids:
            rt.run_cycle(cycle_input(ts, zone=zid))
        out.append(time.perf_counter() - t0)
    return out


class TestScenarioA_Minimal:
    """Scenario A: 1 zone — baseline overhead must be very small."""

    def test_p99_below_100ms(self):
        rt = make_rt()
        s = _run_cycles(rt, ["z1"], 200)
        p99 = _pct(s, 99)
        assert p99 < 0.100, f"1-zone p99={p99*1000:.1f}ms > 100ms"

    def test_p95_below_50ms(self):
        rt = make_rt()
        s = _run_cycles(rt, ["z1"], 200)
        p95 = _pct(s, 95)
        assert p95 < 0.050, f"1-zone p95={p95*1000:.1f}ms > 50ms"

    def test_mean_below_30ms(self):
        rt = make_rt()
        s = _run_cycles(rt, ["z1"], 200)
        mean = statistics.mean(s)
        assert mean < 0.030, f"1-zone mean={mean*1000:.1f}ms > 30ms"

    def test_max_below_500ms(self):
        """Worst-case outlier stays bounded (GC spike, interpreter warm-up)."""
        rt = make_rt()
        s = _run_cycles(rt, ["z1"], 200)
        assert max(s) < 0.500, f"1-zone max={max(s)*1000:.1f}ms > 500ms"

    def test_cycle_returns_result_with_zone_id(self):
        rt = make_rt()
        result = rt.run_cycle(cycle_input(_ts(0), zone="z1"))
        assert result.zone_id == "z1"
        assert result.mode is not None

    def test_cycle_completes_without_errors(self):
        rt = make_rt()
        result = rt.run_cycle(cycle_input(_ts(0), zone="z1"))
        assert len(result.errors) == 0


class TestScenarioB_Normal:
    """Scenario B: 3 zones — typical household."""

    def test_total_p99_below_300ms(self):
        rt = make_rt()
        s = _run_cycles(rt, ["z1", "z2", "z3"], 100)
        p99 = _pct(s, 99)
        assert p99 < 0.300, f"3-zone p99={p99*1000:.1f}ms > 300ms"

    def test_per_zone_overhead_within_3x_of_single(self):
        """3-zone per-zone mean should be within 3× of 1-zone per-zone mean."""
        rt1 = make_rt()
        rt3 = make_rt()
        s1 = _run_cycles(rt1, ["z1"], 100)
        s3 = _run_cycles(rt3, ["z1", "z2", "z3"], 100)
        per1 = statistics.mean(s1)
        per3 = statistics.mean(s3) / 3
        assert per3 < per1 * 3.0, (
            f"3-zone per-zone {per3*1000:.1f}ms > 3× 1-zone {per1*1000:.1f}ms"
        )

    def test_three_independent_zones_no_cross_talk(self):
        """Cycles on different zone_ids accumulate independent state."""
        rt = make_rt()
        for i in range(20):
            rt.run_cycle(cycle_input(_ts(i), zone="za"))
            rt.run_cycle(cycle_input(_ts(i), zone="zb"))
        h = rt.health()
        assert h.zones == 2


class TestScenarioC_Large:
    """Scenario C: 10 zones — upper end of real HA installations."""

    def test_total_p99_below_1000ms(self):
        rt = make_rt()
        zones = [f"z{i}" for i in range(10)]
        s = _run_cycles(rt, zones, 50)
        p99 = _pct(s, 99)
        assert p99 < 1.000, f"10-zone p99={p99*1000:.1f}ms > 1000ms"

    def test_per_zone_not_more_than_5x_single_zone(self):
        """10-zone per-zone mean < 5× 1-zone mean (no super-linear growth)."""
        rt1 = make_rt()
        rt10 = make_rt()
        s1 = _run_cycles(rt1, ["z1"], 50)
        s10 = _run_cycles(rt10, [f"z{i}" for i in range(10)], 50)
        per1 = statistics.mean(s1)
        per10 = statistics.mean(s10) / 10
        assert per10 < per1 * 5.0, (
            f"10-zone per-zone {per10*1000:.1f}ms > 5× 1-zone {per1*1000:.1f}ms"
        )

    def test_large_setup_produces_no_errors(self):
        rt = make_rt()
        zones = [f"z{i}" for i in range(10)]
        for i in range(30):
            for zid in zones:
                result = rt.run_cycle(cycle_input(_ts(i), zone=zid))
                assert len(result.errors) == 0, f"cycle error at zone={zid} step={i}"


class TestScenarioD_Stress:
    """Scenario D: 20 zones — robustness/bounds proof, not normal deployment."""

    def test_total_p99_below_3000ms(self):
        rt = make_rt()
        zones = [f"z{i}" for i in range(20)]
        s = _run_cycles(rt, zones, 30)
        p99 = _pct(s, 99)
        assert p99 < 3.000, f"20-zone p99={p99*1000:.1f}ms > 3000ms"

    def test_growth_factor_not_quadratic(self):
        """Total time for 20 zones < 30× total time for 1 zone (linear=20, allow 50%)."""
        rt1 = make_rt()
        rt20 = make_rt()
        s1 = _run_cycles(rt1, ["z1"], 30)
        s20 = _run_cycles(rt20, [f"z{i}" for i in range(20)], 30)
        factor = statistics.mean(s20) / statistics.mean(s1)
        assert factor < 30.0, (
            f"20-zone growth factor {factor:.1f}× > 30 (possible O(n²) risk)"
        )

    def test_per_zone_p99_stable_at_20_zones(self):
        """p99 per zone in 20-zone config < 3× p99 per zone in 1-zone config."""
        rt1 = make_rt()
        rt20 = make_rt()
        s1 = _run_cycles(rt1, ["z1"], 50)
        s20 = _run_cycles(rt20, [f"z{i}" for i in range(20)], 30)
        p99_1 = _pct(s1, 99)
        p99_20_per_zone = _pct(s20, 99) / 20
        assert p99_20_per_zone < p99_1 * 3.0, (
            f"20-zone per-zone p99 {p99_20_per_zone*1000:.1f}ms > 3× 1-zone {p99_1*1000:.1f}ms"
        )


class TestCycleConsistency:
    """Cross-cutting: timing stability, no runaway, zone isolation."""

    def test_no_systematic_slowdown_over_200_cycles(self):
        """Later cycles must not be significantly slower than early cycles."""
        rt = make_rt()
        all_s = _run_cycles(rt, ["z1"], 200)
        early_mean = statistics.mean(all_s[:50])
        late_mean = statistics.mean(all_s[150:])
        assert late_mean < early_mean * 3.0, (
            f"Late cycles {late_mean*1000:.1f}ms > 3× early {early_mean*1000:.1f}ms"
        )

    def test_zone_dict_size_does_not_slow_individual_zone(self):
        """z1 in a 20-zone runtime is not slower than z1 alone."""
        rt_small = make_rt()
        rt_large = make_rt()
        # warm up both runtimes
        for i in range(20):
            rt_small.run_cycle(cycle_input(_ts(i), zone="z1"))
            for zid in [f"z{j}" for j in range(20)]:
                rt_large.run_cycle(cycle_input(_ts(i), zone=zid))

        def _z1_samples(rt: LearningRuntime, n: int = 50) -> list[float]:
            out = []
            for i in range(n):
                t0 = time.perf_counter()
                rt.run_cycle(cycle_input(_ts(i + 100), zone="z1"))
                out.append(time.perf_counter() - t0)
            return out

        s_small = _z1_samples(rt_small)
        s_large = _z1_samples(rt_large)
        assert statistics.mean(s_large) < statistics.mean(s_small) * 2.0, (
            "z1 in 20-zone runtime significantly slower than z1 alone"
        )

    def test_startup_grace_vs_full_processing_bounded(self):
        """Full processing (grace=0) vs grace-only: overhead bounded within 4×."""
        rt_grace = LearningRuntime(
            LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW, startup_grace_cycles=9999),
            clock=lambda: _ts(0))
        rt_full = LearningRuntime(
            LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW, startup_grace_cycles=0),
            clock=lambda: _ts(0))
        n = 50
        t0 = time.perf_counter()
        for i in range(n):
            rt_grace.run_cycle(cycle_input(_ts(i), zone="z1"))
        t_grace = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for i in range(n):
            rt_full.run_cycle(cycle_input(_ts(i), zone="z1"))
        t_full = (time.perf_counter() - t0) / n

        assert t_full < t_grace * 4.0 + 0.020, (
            f"Full processing {t_full*1000:.1f}ms > 4× grace {t_grace*1000:.1f}ms"
        )

    def test_health_reports_correct_zone_count(self):
        rt = make_rt()
        zones = [f"z{i}" for i in range(5)]
        for zid in zones:
            rt.run_cycle(cycle_input(_ts(0), zone=zid))
        h = rt.health()
        assert h.zones == 5

    def test_cycle_safe_never_raises(self):
        """run_cycle_safe must always return a result, never raise."""
        rt = make_rt()
        result = rt.run_cycle_safe(cycle_input(_ts(0), zone="z1"))
        assert result is not None
        assert result.zone_id == "z1"
