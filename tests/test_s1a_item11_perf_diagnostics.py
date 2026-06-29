"""S1a Item 11 — Diagnostics timing and output size bounds.

DiagnosticsOrchestrator.build_diagnostics() must remain fast and bounded
regardless of zone count or model count. All tests are pure Python
(no HA fixtures required).
"""
from __future__ import annotations

import json
import statistics
import time

import pytest

from custom_components.thermosmart.learning.diagnostics import DiagnosticsOrchestrator
from tests.helpers_orchestration import diagnostics_input, zone_input


def _build(n_zones: int = 1) -> dict:
    zones = tuple(zone_input(f"zone_{i}") for i in range(n_zones))
    di = diagnostics_input(zones=zones)
    result = DiagnosticsOrchestrator(di).build_diagnostics()
    return result.payload


def _build_timed(n_zones: int = 1, n_reps: int = 20) -> list[float]:
    zones = tuple(zone_input(f"zone_{i}") for i in range(n_zones))
    di = diagnostics_input(zones=zones)
    times: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        DiagnosticsOrchestrator(di).build_diagnostics()
        times.append(time.perf_counter() - t0)
    return times


def _pct(s: list[float], p: int) -> float:
    ss = sorted(s)
    return ss[max(0, min(len(ss) - 1, int(len(ss) * p / 100)))]


def _payload_bytes(payload: dict) -> int:
    return len(json.dumps(payload).encode("utf-8"))


class TestDiagnosticsTiming:
    """build_diagnostics() must be fast at all zone counts."""

    def test_1zone_p95_below_500ms(self):
        times = _build_timed(n_zones=1, n_reps=30)
        p95 = _pct(times, 95)
        assert p95 < 0.500, f"1-zone p95={p95*1000:.1f}ms > 500ms"

    def test_5zones_p95_below_2500ms(self):
        times = _build_timed(n_zones=5, n_reps=20)
        p95 = _pct(times, 95)
        assert p95 < 2.500, f"5-zone p95={p95*1000:.1f}ms > 2500ms"

    def test_10zones_p95_below_5000ms(self):
        times = _build_timed(n_zones=10, n_reps=10)
        p95 = _pct(times, 95)
        assert p95 < 5.000, f"10-zone p95={p95*1000:.1f}ms > 5000ms"

    def test_20zones_mean_below_10000ms(self):
        times = _build_timed(n_zones=20, n_reps=5)
        mean = statistics.mean(times)
        assert mean < 10.000, f"20-zone mean={mean*1000:.1f}ms > 10000ms"


class TestDiagnosticsScaling:
    """build_diagnostics() must scale approximately linearly with zone count."""

    def test_10zones_not_more_than_15x_slower_than_1zone(self):
        t1 = statistics.mean(_build_timed(n_zones=1, n_reps=20))
        t10 = statistics.mean(_build_timed(n_zones=10, n_reps=10))
        assert t10 < t1 * 15.0 + 0.100, (
            f"10-zone {t10*1000:.1f}ms > 15× 1-zone {t1*1000:.1f}ms"
        )

    def test_zone_count_scales_linearly_not_quadratic(self):
        """per-zone mean should be similar at 1, 5, and 10 zones."""
        t1 = statistics.mean(_build_timed(n_zones=1, n_reps=20))
        t5 = statistics.mean(_build_timed(n_zones=5, n_reps=15))
        t10 = statistics.mean(_build_timed(n_zones=10, n_reps=10))
        per1 = t1 / 1
        per5 = t5 / 5
        per10 = t10 / 10
        max_per = max(per1, per5, per10)
        min_per = min(per1, per5, per10)
        ratio = max_per / max(min_per, 1e-9)
        assert ratio < 4.0, (
            f"per-zone cost ratio max/min={ratio:.1f} > 4.0 "
            f"(per1={per1*1e3:.2f}ms per5={per5*1e3:.2f}ms per10={per10*1e3:.2f}ms)"
        )


class TestDiagnosticsOutputSize:
    """Output payload must be bounded and not grow unboundedly with zones."""

    def test_1zone_payload_below_100kb(self):
        payload = _build(n_zones=1)
        size = _payload_bytes(payload)
        assert size < 100 * 1024, f"1-zone diagnostics {size/1024:.1f}KB ≥ 100KB"

    def test_10zones_payload_below_1mb(self):
        payload = _build(n_zones=10)
        size = _payload_bytes(payload)
        assert size < 1024 * 1024, f"10-zone diagnostics {size/1024:.1f}KB ≥ 1MB"

    def test_20zones_payload_scales_linearly(self):
        """20-zone payload < 25× 1-zone payload (linear=20, allow 25%)."""
        b1 = _payload_bytes(_build(n_zones=1))
        b20 = _payload_bytes(_build(n_zones=20))
        assert b20 < b1 * 25, (
            f"20-zone {b20/1024:.0f}KB > 25× 1-zone {b1/1024:.0f}KB"
        )

    def test_payload_contains_all_zones(self):
        payload = _build(n_zones=5)
        assert len(payload["zones"]) == 5

    def test_payload_has_no_privacy_violations(self):
        """build_diagnostics() must not embed raw private data."""
        zones = tuple(zone_input(f"zone_{i}") for i in range(3))
        di = diagnostics_input(zones=zones)
        result = DiagnosticsOrchestrator(di).build_diagnostics()
        # Privacy scanner runs inside build_diagnostics(); errors signal violations
        privacy_errors = [e for e in result.errors if e.startswith("privacy:")]
        assert privacy_errors == [], f"Privacy violations: {privacy_errors}"


class TestDiagnosticsStructure:
    """build_diagnostics() output has required structural invariants."""

    def test_result_has_schema_version(self):
        payload = _build(1)
        assert "diagnostics_schema_version" in payload

    def test_result_has_engine_block(self):
        payload = _build(1)
        assert "engine" in payload
        assert "le_version" in payload["engine"]

    def test_result_has_zones_list(self):
        payload = _build(3)
        assert "zones" in payload
        assert isinstance(payload["zones"], list)

    def test_empty_zones_produces_valid_payload(self):
        di = diagnostics_input(zones=())
        result = DiagnosticsOrchestrator(di).build_diagnostics()
        assert result.payload["zones"] == []
        assert len(result.errors) == 0

    def test_missing_model_handled_gracefully(self):
        """A zone with no models must not cause build_diagnostics() to raise."""
        zones = (zone_input("empty_zone", models={}),)
        di = diagnostics_input(zones=zones)
        result = DiagnosticsOrchestrator(di).build_diagnostics()
        assert result is not None
        assert len(result.payload["zones"]) == 1


class TestPendingAttributionSummary:
    """pending_attribution_summary() must be fast and privacy-safe."""

    def test_summary_is_fast(self):
        from tests.helpers_runtime_scenarios import runtime as make_rt
        from tests.helpers_runtime import cycle_input
        from datetime import datetime, timedelta, timezone

        T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
        rt = make_rt()
        for i in range(100):
            ts = (T0 + timedelta(minutes=i * 5)).isoformat()
            rt.run_cycle(cycle_input(ts, zone="z1"))

        zr = rt._zone("z1")
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            zr.pending_attribution_summary()
            times.append(time.perf_counter() - t0)

        mean = statistics.mean(times)
        assert mean < 0.010, f"pending_attribution_summary mean={mean*1000:.2f}ms > 10ms"

    def test_summary_has_required_keys(self):
        from tests.helpers_runtime_scenarios import runtime as make_rt
        from tests.helpers_runtime import cycle_input
        from datetime import datetime, timedelta, timezone

        T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
        rt = make_rt()
        for i in range(30):
            ts = (T0 + timedelta(minutes=i * 5)).isoformat()
            rt.run_cycle(cycle_input(ts, zone="z1"))

        zr = rt._zone("z1")
        s = zr.pending_attribution_summary()
        assert "schema_version" in s
        assert "pending_context_count" in s
        assert "pending_dispatch_count" in s

    def test_summary_counts_are_non_negative(self):
        from tests.helpers_runtime_scenarios import runtime as make_rt
        from tests.helpers_runtime import cycle_input
        from datetime import datetime, timedelta, timezone

        T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
        rt = make_rt()
        for i in range(50):
            ts = (T0 + timedelta(minutes=i * 5)).isoformat()
            rt.run_cycle(cycle_input(ts, zone="z1"))

        zr = rt._zone("z1")
        s = zr.pending_attribution_summary()
        assert s["pending_context_count"] >= 0
        assert s["pending_dispatch_count"] >= 0
