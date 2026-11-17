"""Phase D2-A.2 — final persistence hardening.

Restore-invariant and conditioning-readiness regression tests for the remaining
weaknesses the D2-A.1 audit flagged:

* zero-weight / nonzero-weighted-sum corruption
* reconstructed- and finalized-mean range validation
* implausible persisted temperatures (plausibility contract)
* missing first/last when usable samples exist
* unsupported / inconsistent schema versions (v2 missing cov/lo; cov != wsum)
* overflow safety in finalize
* duration/weight trust for legacy v1 uncapped coverage
* conditioning_ready behaviour across source switching, unknown source, stale
  quality, extreme spread and short/sparse episodes (observational, not a D2-B
  trust gate)

The section-4 mutation checks (removing each guard reintroduces a defect these
tests catch) are run independently at commit time; see the task report.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.outdoor_evidence import (
    OUTDOOR_EVIDENCE_SCHEMA_VERSION,
    OutdoorEvidenceAccumulator,
)


def _m(value, quality=DataQuality.OK):
    return Measurement(value, quality)


def _valid_v2() -> dict:
    """Well-formed v2 snapshot: usable 0.0 then 10.0, 300 s apart (one interval)."""
    acc = OutdoorEvidenceAccumulator()
    acc.observe(0, _m(0.0), "sensor")
    acc.observe(300_000, _m(10.0), "sensor")
    d = acc.serialize()
    assert d["v"] == OUTDOOR_EVIDENCE_SCHEMA_VERSION == 2
    return d


def _empty(acc: OutdoorEvidenceAccumulator) -> bool:
    return acc.finalize(duration_s=600.0, end_offset_ms=600_000) is None


# ── 1. restore invariants ─────────────────────────────────────────────────────

class TestRestoreInvariants:
    @pytest.mark.parametrize("mutate,label", [
        (lambda d: d.update(wsum=0.0, vwsum=5.0, cov=0.0), "zero-weight nonzero sum"),
        (lambda d: d.update(max=60.0), "implausible max (>50C)"),
        (lambda d: d.update(min=-40.0), "implausible min (<-30C)"),
        (lambda d: d.update(first=None), "missing first with usable samples"),
        (lambda d: d.update(last=None), "missing last with usable samples"),
        (lambda d: d.update(cov=d["wsum"] + 1000.0), "v2 cov != wsum"),
        (lambda d: d.pop("cov"), "v2 missing cov"),
        (lambda d: d.pop("lo"), "v2 missing lo"),
        (lambda d: d.update(vwsum=9.0e9), "restored mean out of range"),
        (lambda d: d.update(first=20.0), "first outside [min,max]"),
    ])
    def test_corrupt_restores_empty(self, mutate, label):
        d = _valid_v2()
        mutate(d)
        assert _empty(OutdoorEvidenceAccumulator.restore(d)), label

    def test_zero_weight_zero_sum_is_accepted(self):
        # a single usable sample never closes an interval: wsum == vwsum == 0 is valid
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        d = acc.serialize()
        assert d["wsum"] == 0.0 and d["vwsum"] == 0.0
        restored = OutdoorEvidenceAccumulator.restore(d)
        ev = restored.finalize(duration_s=0.0, end_offset_ms=0)
        assert ev is not None and ev.usable_count == 1 and ev.time_weighted_mean_c == 5.0

    def test_v1_snapshot_without_cov_lo_is_accepted(self):
        d = _valid_v2()
        d["v"] = 1
        d.pop("cov")
        d.pop("lo")
        ev = OutdoorEvidenceAccumulator.restore(d).finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev is not None and ev.usable_count == 2


# ── 2. finalize guards (direct, data-free) ────────────────────────────────────

class TestFinalizeGuards:
    def test_finalized_mean_out_of_range_fails_closed(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        acc.observe(300_000, _m(5.0), "sensor")
        # tamper the internal sums so the weighted mean would read 9e9
        acc._vwsum = 9.0e9
        acc._wsum = 1.0
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev.time_weighted_mean_c is None          # not a fabricated 9e9
        assert ev.has_usable_evidence is False
        assert ev.conditioning_ready is False

    def test_finalize_overflow_is_contained(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        acc.observe(300_000, _m(5.0), "sensor")
        acc._vwsum = 1.0e308
        acc._wsum = 1.0e-10                              # division overflows
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)   # must not raise
        assert ev.time_weighted_mean_c is None

    def test_legacy_v1_uncapped_weight_is_not_trusted(self):
        # a v1 snapshot whose weight sum vastly exceeds the episode duration: the
        # temporal basis is unverifiable, so it must not read conditioning-ready.
        d = _valid_v2()
        d["v"] = 1
        d.pop("cov")
        d.pop("lo")
        d.update(n=2, u=2, min=0.0, max=10.0, first=5.0, last=5.0, luv=5.0,
                 wsum=50_000_000.0, vwsum=250_000_000.0, ssum=10.0, luo=50_000_000)
        acc = OutdoorEvidenceAccumulator.restore(d)
        ev = acc.finalize(duration_s=600.0, end_offset_ms=600_000)   # 600 s episode
        assert ev is not None
        assert ev.temporal_coverage == pytest.approx(1.0)            # clamped
        assert ev.conditioning_ready is False                        # but not trusted


# ── 3. conditioning readiness (observational) ─────────────────────────────────

def _good_coverage(values_sources, *, qualities=None, step=300_000, duration_s=600.0):
    """Drive one usable sample per cadence step (fully time-covered)."""
    acc = OutdoorEvidenceAccumulator()
    n = len(values_sources)
    quals = qualities or [DataQuality.OK] * n
    for i, ((val, src), q) in enumerate(zip(values_sources, quals)):
        acc.observe(i * step, Measurement(val, q), src)
    return acc.finalize(duration_s=duration_s, end_offset_ms=(n - 1) * step)


class TestConditioningReadiness:
    def test_source_switching_blocks_readiness(self):
        ev = _good_coverage([(5.0, "sensor"), (6.0, "weather_entity"), (7.0, "sensor")])
        assert ev.source_changed is True
        assert ev.conditioning_ready is False

    def test_unknown_source_does_not_block_readiness(self):
        # observational: an unknown (None) source with dense coverage is not gated
        ev = _good_coverage([(5.0, None), (6.0, None), (7.0, None)])
        assert ev.source is None and ev.source_changed is False
        assert ev.temporal_coverage == pytest.approx(1.0)
        assert ev.conditioning_ready is True

    def test_stale_quality_does_not_block_readiness(self):
        ev = _good_coverage(
            [(5.0, "sensor"), (5.0, "sensor"), (5.0, "sensor")],
            qualities=[DataQuality.OK, DataQuality.STALE, DataQuality.OK])
        assert ev.quality is DataQuality.STALE
        assert ev.conditioning_ready is True

    def test_extreme_temperature_spread_does_not_block_readiness(self):
        ev = _good_coverage([(-25.0, "sensor"), (45.0, "sensor"), (0.0, "sensor")])
        assert ev.min_c == -25.0 and ev.max_c == 45.0
        assert ev.conditioning_ready is True

    def test_short_sparse_episode_blocks_readiness(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")                # single usable sample
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev.usable_count == 1
        assert ev.conditioning_ready is False
