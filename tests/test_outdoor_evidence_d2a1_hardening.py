"""Phase D2-A.1 — outdoor-evidence hardening.

Covers the two mandatory D2-A audit fixes plus the additional-robustness items:

1. Restore safety: a corrupt / malformed / inconsistent / unsupported-version
   accumulator snapshot restores as empty (unusable) evidence, never as a
   fabricated measurement, and never raises — while valid D2-A (v1) and D2-A.1
   (v2) snapshots round-trip.
2. Long-gap handling: the time-weighted aggregation no longer extrapolates the
   last valid reading across missing/unavailable periods. Temporal coverage is
   tracked distinctly from sample coverage, a usable reading holds for at most
   two cadence intervals, an unusable reading suspends the previous interval, and
   low/uncertain coverage marks the evidence unsuitable for conditioned learning.
3. Robustness: booleans are not accepted as temperatures, duplicate/out-of-order
   timestamps are ignored, unusable samples never flag a source change, restore
   works mid-heating and mid-cooling, and source identity propagates from the
   weather dict through capture.

The five-hour-dropout probe (test_five_hour_dropout_is_not_fully_covered) is the
auditor's reported failure; with the pre-hardening uncapped hold it reported full
coverage, which the task-report mutation re-confirms.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement, Regime
from custom_components.thermosmart.learning.outdoor_evidence import (
    OUTDOOR_EVIDENCE_SCHEMA_VERSION,
    OutdoorEvidenceAccumulator,
    _MAX_HOLD_MS,
)
from custom_components.thermosmart.learning.builders.base import BuilderAction, BuilderInput
from custom_components.thermosmart.learning.builders.cooling import PassiveCoolingEpisodeBuilder
from custom_components.thermosmart.learning.builders.heating import HeatingEpisodeBuilder
from custom_components.thermosmart.learning.regime import RegimeResult

_T0 = datetime(2026, 1, 12, 6, 0, tzinfo=timezone.utc)


def _m(value, quality=DataQuality.OK):
    return Measurement(value, quality)


# ── 1. restore safety ─────────────────────────────────────────────────────────

def _valid_v2() -> dict:
    """A well-formed v2 snapshot (2 usable samples 0.0 then 10.0, 300 s apart)."""
    acc = OutdoorEvidenceAccumulator()
    acc.observe(0, _m(0.0), "sensor")
    acc.observe(300_000, _m(10.0), "sensor")
    d = acc.serialize()
    assert d["v"] == OUTDOOR_EVIDENCE_SCHEMA_VERSION == 2
    return d


def _is_empty(acc: OutdoorEvidenceAccumulator) -> bool:
    return acc.finalize(duration_s=600.0, end_offset_ms=600_000) is None


class TestRestoreSafety:
    def test_valid_v2_roundtrips(self):
        acc = OutdoorEvidenceAccumulator.restore(_valid_v2())
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev is not None and ev.usable_count == 2 and ev.time_weighted_mean_c == 0.0

    def test_valid_v1_without_temporal_state_roundtrips(self):
        d = _valid_v2()
        d["v"] = 1
        d.pop("cov")          # v1 (D2-A) had no explicit temporal-coverage state
        d.pop("lo")
        acc = OutdoorEvidenceAccumulator.restore(d)
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev is not None and ev.usable_count == 2

    def test_pre_d2a_absent_state_is_empty(self):
        assert OutdoorEvidenceAccumulator.restore(None).finalize(duration_s=0.0) is None

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update(v=3),                       # unsupported version
        lambda d: d.pop("v"),                          # missing version
        lambda d: d.pop("n"),                          # missing required int
        lambda d: d.pop("wsum"),                       # missing required float
        lambda d: d.update(wsum=float("inf")),         # infinity
        lambda d: d.update(vwsum=float("nan")),        # NaN
        lambda d: d.update(min=float("nan")),          # NaN in optional
        lambda d: d.update(n=True),                    # bool where int expected
        lambda d: d.update(wsum=True),                 # bool where float expected
        lambda d: d.update(luo=True),                  # bool where offset expected
        lambda d: d.update(u=9),                       # usable > total
        lambda d: d.update(wsum=-1.0),                 # negative weight
        lambda d: d.update(cov=-5.0),                  # negative coverage
        lambda d: d.update(luo=-10),                   # negative offset
        lambda d: d.update(luo=500_000, lo=300_000),   # luo > lo
        lambda d: d.update(min=None),                  # usable but no min
        lambda d: d.update(min=20.0, max=0.0),         # min > max
        lambda d: d.update(luv=99.0),                  # last usable value outside [min,max]
        lambda d: d.update(vwsum=9.0e9),               # mean outside [min,max]
        lambda d: d.update(sources="sensor"),          # sources not a list
        lambda d: d.update(sources=[123]),             # non-string source
        lambda d: d.update(stale="yes"),               # non-bool stale
        lambda d: d.update(wsum=10 ** 400),            # OverflowError on float()
    ])
    def test_corrupt_snapshot_restores_empty(self, mutate):
        d = _valid_v2()
        mutate(d)
        acc = OutdoorEvidenceAccumulator.restore(d)      # must not raise
        assert _is_empty(acc), "corrupt snapshot must restore as empty evidence"

    def test_u0_with_sources_is_rejected(self):
        d = {"v": 2, "n": 1, "u": 0, "min": None, "max": None, "first": None,
             "last": None, "stale": False, "sources": ["sensor"], "wsum": 0.0,
             "vwsum": 0.0, "ssum": 0.0, "cov": 0.0, "luo": None, "luv": None, "lo": 0}
        assert _is_empty(OutdoorEvidenceAccumulator.restore(d))

    def test_u0_consistent_snapshot_is_accepted(self):
        # an all-unusable episode (n>0, u=0) is a legitimate state, not corruption
        d = {"v": 2, "n": 3, "u": 0, "min": None, "max": None, "first": None,
             "last": None, "stale": False, "sources": [], "wsum": 0.0,
             "vwsum": 0.0, "ssum": 0.0, "cov": 0.0, "luo": None, "luv": None, "lo": 600_000}
        ev = OutdoorEvidenceAccumulator.restore(d).finalize(duration_s=600.0, end_offset_ms=600_000)
        assert ev is not None and ev.usable_count == 0 and ev.sample_count == 3


# ── 2. long-gap / temporal coverage ───────────────────────────────────────────

class TestTemporalCoverage:
    def test_five_hour_dropout_is_not_fully_covered(self):
        # usable 10 °C at t0, then the sensor goes unavailable for ~5 h with no fresh
        # reading, episode ends. Pre-hardening this credited the full 5 h to 10 °C
        # (fully "observed"); now only one max-hold interval is covered.
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(10.0), "sensor")
        for i in range(1, 60):                              # 59 unavailable 5-min cycles
            acc.observe(i * 300_000, Measurement(None, DataQuality.UNAVAILABLE), "sensor")
        ev = acc.finalize(duration_s=18_000.0, end_offset_ms=18_000_000)
        # the first unavailable reading (at +300 s) terminates the single valid
        # interval; only that one cadence step is credited, not the whole 5 h.
        assert ev.temporal_coverage == pytest.approx(300_000 / 18_000_000, abs=1e-4)
        assert ev.temporal_coverage < 0.05
        assert ev.conditioning_ready is False
        assert ev.time_weighted_mean_c == pytest.approx(10.0)   # not fabricated

    def test_silent_dropout_caps_hold_at_max(self):
        # no readings at all for ~5 h (not even "unavailable"): the single pre-gap
        # reading can hold for at most one max-hold interval, never the whole gap.
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(10.0), "sensor")
        acc.observe(18_000_000, _m(12.0), "sensor")       # next reading 5 h later
        ev = acc.finalize(duration_s=18_000.0, end_offset_ms=18_000_000)
        assert ev.temporal_coverage == pytest.approx(_MAX_HOLD_MS / 18_000_000, abs=1e-4)
        assert ev.conditioning_ready is False

    def test_sample_coverage_high_but_temporal_coverage_low(self):
        # two usable readings 2000 s apart: every sample is usable (sample coverage
        # 1.0) yet most of the time is unobserved (temporal coverage ~0.3).
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(10.0), "sensor")
        acc.observe(2_000_000, _m(12.0), "sensor")
        ev = acc.finalize(duration_s=2_000.0, end_offset_ms=2_000_000)
        assert ev.coverage == pytest.approx(1.0)                     # sample coverage
        assert ev.temporal_coverage == pytest.approx(_MAX_HOLD_MS / 2_000_000, abs=1e-4)
        assert ev.temporal_coverage == pytest.approx(0.3, abs=1e-4)
        assert ev.conditioning_ready is False

    def test_stable_sensor_every_cycle_is_fully_covered(self):
        # identical value every 5 min for 30 min: unchanged != stale/unobserved.
        acc = OutdoorEvidenceAccumulator()
        for i in range(7):
            acc.observe(i * 300_000, _m(5.0), "sensor")
        ev = acc.finalize(duration_s=1_800.0, end_offset_ms=1_800_000)
        assert ev.temporal_coverage == pytest.approx(1.0)
        assert ev.quality is DataQuality.OK                          # not downgraded to stale
        assert ev.conditioning_ready is True
        assert ev.time_weighted_mean_c == pytest.approx(5.0)

    def test_unusable_reading_terminates_previous_interval(self):
        # usable@0, unusable@300k, usable@600k: the gap 300k–600k is NOT credited to
        # the first reading; only its single pre-gap interval counts.
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(10.0), "sensor")
        acc.observe(300_000, None, "sensor")
        acc.observe(600_000, _m(12.0), "sensor")
        ev = acc.finalize(duration_s=600.0, end_offset_ms=600_000)
        assert ev.temporal_coverage == pytest.approx(0.5)            # 300k of 600k
        assert ev.time_weighted_mean_c == pytest.approx(10.0)        # only the 1st interval weighted

    def test_irregular_cadence_caps_each_hold(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(4.0), "sensor")
        acc.observe(100_000, _m(4.0), "sensor")          # 100 s gap -> 100k held
        acc.observe(900_000, _m(4.0), "sensor")          # 800 s gap -> capped at 600k
        ev = acc.finalize(duration_s=900.0, end_offset_ms=900_000)
        assert ev.temporal_coverage == pytest.approx(700_000 / 900_000, abs=1e-4)


# ── 3. additional robustness ──────────────────────────────────────────────────

class TestRobustness:
    def test_boolean_temperature_is_not_accepted_as_one_degree(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, Measurement(True, DataQuality.OK), "sensor")   # bool value
        acc.observe(300_000, Measurement(False, DataQuality.OK), "sensor")
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev.usable_count == 0                                   # neither True->1.0 nor False->0.0
        assert ev.time_weighted_mean_c is None

    def test_duplicate_and_out_of_order_timestamps_ignored(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(300_000, _m(5.0), "sensor")
        acc.observe(300_000, _m(9.0), "sensor")   # duplicate offset -> ignored
        acc.observe(100_000, _m(9.0), "sensor")   # out of order -> ignored
        acc.observe(600_000, _m(7.0), "sensor")
        ev = acc.finalize(duration_s=600.0, end_offset_ms=600_000)
        assert ev.sample_count == 2                                   # duplicates/reorders not counted
        assert ev.usable_count == 2
        assert ev.min_c == 5.0 and ev.max_c == 7.0                    # the 9.0 reorders never landed

    def test_unusable_sample_does_not_flag_source_change(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        acc.observe(300_000, Measurement(None, DataQuality.UNAVAILABLE), "weather_entity")
        acc.observe(600_000, _m(6.0), "sensor")
        ev = acc.finalize(duration_s=600.0, end_offset_ms=600_000)
        assert ev.source_changed is False
        assert ev.source == "sensor"

    def _drive(self, builder, open_regime, close_regime, outdoor_series, indoor_fn):
        n = len(outdoor_series)
        res = None
        for i, od in enumerate(outdoor_series):
            regime = open_regime if i < n - 1 else close_regime
            res = builder.process(BuilderInput(
                ts=_T0 + timedelta(seconds=i * 300), regime=RegimeResult(regime, 0.9),
                indoor_temp=indoor_fn(i), target=21.0, outdoor_temp=od, outdoor_source="sensor"))
        return res

    def test_restore_midway_during_heating(self):
        b1 = HeatingEpisodeBuilder("lz")
        b1.process(BuilderInput(ts=_T0, regime=RegimeResult(Regime.ACTIVE_HEATING, 0.9),
                                indoor_temp=18.0, target=21.0, outdoor_temp=_m(0.0),
                                outdoor_source="sensor"))
        b1.process(BuilderInput(ts=_T0 + timedelta(seconds=300),
                                regime=RegimeResult(Regime.ACTIVE_HEATING, 0.9),
                                indoor_temp=18.5, target=21.0, outdoor_temp=_m(10.0),
                                outdoor_source="sensor"))
        snap = b1.snapshot_state()
        b2 = HeatingEpisodeBuilder("lz")
        b2.restore_state(snap)
        res = b2.process(BuilderInput(ts=_T0 + timedelta(seconds=600),
                                      regime=RegimeResult(Regime.STABLE, 0.9),
                                      indoor_temp=19.0, target=21.0, outdoor_temp=_m(20.0),
                                      outdoor_source="sensor"))
        assert res.action is BuilderAction.CLOSED
        ev = res.outdoor_evidence
        assert ev.usable_count == 3 and ev.temporal_coverage == pytest.approx(1.0)
        assert ev.time_weighted_mean_c == pytest.approx(5.0)

    def test_restore_midway_during_passive_cooling(self):
        b1 = PassiveCoolingEpisodeBuilder("lz")
        b1.process(BuilderInput(ts=_T0, regime=RegimeResult(Regime.PASSIVE_COOLING, 0.9),
                                indoor_temp=21.0, target=21.0, outdoor_temp=_m(2.0),
                                outdoor_source="sensor"))
        b1.process(BuilderInput(ts=_T0 + timedelta(seconds=300),
                                regime=RegimeResult(Regime.PASSIVE_COOLING, 0.9),
                                indoor_temp=20.7, target=21.0, outdoor_temp=_m(2.0),
                                outdoor_source="sensor"))
        snap = b1.snapshot_state()
        b2 = PassiveCoolingEpisodeBuilder("lz")
        b2.restore_state(snap)
        res = b2.process(BuilderInput(ts=_T0 + timedelta(seconds=600),
                                      regime=RegimeResult(Regime.STABLE, 0.9),
                                      indoor_temp=20.4, target=21.0, outdoor_temp=_m(4.0),
                                      outdoor_source="sensor"))
        assert res.action is BuilderAction.CLOSED
        ev = res.outdoor_evidence
        assert ev.usable_count == 3 and ev.min_c == 2.0 and ev.max_c == 4.0
        assert ev.temporal_coverage == pytest.approx(1.0)


class TestSourcePropagation:
    def test_weather_source_propagates_through_capture(self):
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            build_runtime_cycle_input)
        from custom_components.thermosmart.learning.runtime.capture import CaptureCoordinator

        inp = build_runtime_cycle_input(
            "lz", {"current_temp": 19.0, "effective_target": 21.0},
            weather={"temperature": 5.0, "temperature_source": "sensor"},
            ts_iso=_T0.isoformat())
        assert inp.outdoor_source == "sensor"
        snap = CaptureCoordinator("lz").capture(inp)
        assert snap.outdoor_source == "sensor"

    def test_absent_weather_source_is_none(self):
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            build_runtime_cycle_input)
        inp = build_runtime_cycle_input(
            "lz", {"current_temp": 19.0, "effective_target": 21.0},
            weather={"temperature": 5.0}, ts_iso=_T0.isoformat())
        assert inp.outdoor_source is None
