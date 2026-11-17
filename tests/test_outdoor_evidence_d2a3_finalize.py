"""Phase D2-A.3 — finalize safety for corrupt persisted offsets.

F1 — a persisted offset so large it overflows float arithmetic (e.g. 10**400)
must not make finalize raise and must not leave the heating/cooling episode stuck
open; it fails closed to unusable outdoor evidence while the episode still closes.

F3 — a persisted guard offset (``lo``) beyond the actual episode end offset
silently suppresses subsequent readings while the evidence looks complete. Such a
state is detected at finalize (where the real end offset is known), the evidence
is invalidated (mean None, temporal_coverage 0, conditioning_ready False), and no
dropped subsequent reading can rehabilitate it without an explicit reset (a new
episode). Normal duplicate / out-of-order handling is preserved.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement, Regime
from custom_components.thermosmart.learning.outdoor_evidence import OutdoorEvidenceAccumulator
from custom_components.thermosmart.learning.builders.base import BuilderAction, BuilderInput
from custom_components.thermosmart.learning.builders.cooling import PassiveCoolingEpisodeBuilder
from custom_components.thermosmart.learning.builders.heating import HeatingEpisodeBuilder
from custom_components.thermosmart.learning.regime import RegimeResult

_T0 = datetime(2026, 1, 12, 6, 0, tzinfo=timezone.utc)
_BIG = 10 ** 400               # not representable as a finite float


def _m(value, quality=DataQuality.OK):
    return Measurement(value, quality)


def _binp(i, regime, *, indoor, outdoor, source="sensor", target=21.0, step_s=300):
    return BuilderInput(
        ts=_T0 + timedelta(seconds=i * step_s), regime=RegimeResult(regime, 0.9),
        indoor_temp=indoor, target=target, outdoor_temp=outdoor, outdoor_source=source)


# ── F1: overflow in finalize ──────────────────────────────────────────────────

class TestF1Overflow:
    def test_restore_rejects_non_float_representable_offset(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        acc.observe(300_000, _m(6.0), "sensor")
        d = acc.serialize()
        d["luo"] = _BIG
        d["lo"] = _BIG
        restored = OutdoorEvidenceAccumulator.restore(d)         # must not raise
        assert restored.finalize(duration_s=600.0, end_offset_ms=600_000) is None

    def test_finalize_does_not_raise_on_huge_offset_state(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        acc.observe(300_000, _m(6.0), "sensor")
        # force an impossible offset directly (bypassing restore)
        acc._last_offset = _BIG
        acc._last_usable_offset = _BIG
        ev = acc.finalize(duration_s=600.0, end_offset_ms=600_000)   # must not raise
        assert ev is not None
        assert ev.time_weighted_mean_c is None                   # failed closed
        assert ev.conditioning_ready is False

    def test_heating_episode_closes_after_corrupt_offset_restore(self):
        b1 = HeatingEpisodeBuilder("lz")
        b1.process(_binp(0, Regime.ACTIVE_HEATING, indoor=18.0, outdoor=_m(0.0)))
        b1.process(_binp(1, Regime.ACTIVE_HEATING, indoor=18.5, outdoor=_m(10.0)))
        snap = b1.snapshot_state()
        snap["outdoor"]["luo"] = _BIG                            # F1 corruption
        snap["outdoor"]["lo"] = _BIG
        b2 = HeatingEpisodeBuilder("lz")
        b2.restore_state(snap)                                   # must not raise
        res = b2.process(_binp(2, Regime.STABLE, indoor=19.0, outdoor=_m(20.0)))
        assert res.action is BuilderAction.CLOSED                # episode NOT stuck open
        # corrupt outdoor state was dropped on restore; the episode still completes
        assert res.outdoor_evidence is not None


# ── F3: future offset freezes evidence ────────────────────────────────────────

class TestF3FutureOffset:
    def test_future_guard_offset_invalidates_evidence(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        acc.observe(300_000, _m(6.0), "sensor")
        d = acc.serialize()
        d["lo"] = 900_000                                        # guard offset in the future
        restored = OutdoorEvidenceAccumulator.restore(d)
        # a subsequent valid reading at the real cadence is silently dropped (<= lo)
        restored.observe(600_000, _m(7.0), "sensor")
        ev = restored.finalize(duration_s=600.0, end_offset_ms=600_000)   # real end < lo
        assert ev.time_weighted_mean_c is None
        assert ev.temporal_coverage == pytest.approx(0.0)
        assert ev.conditioning_ready is False

    def test_dropped_readings_do_not_rehabilitate_without_reset(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        acc.observe(300_000, _m(6.0), "sensor")
        d = acc.serialize()
        d["lo"] = 900_000
        restored = OutdoorEvidenceAccumulator.restore(d)
        for off in (600_000, 450_000, 300_000):                 # all <= future lo -> dropped
            restored.observe(off, _m(8.0), "sensor")
        ev = restored.finalize(duration_s=600.0, end_offset_ms=600_000)
        assert ev.conditioning_ready is False                   # still invalid
        assert ev.time_weighted_mean_c is None
        # an explicit reset (a fresh accumulator / new episode) is the only recovery
        fresh = OutdoorEvidenceAccumulator()
        fresh.observe(0, _m(5.0), "sensor")
        fresh.observe(300_000, _m(5.0), "sensor")
        assert fresh.finalize(duration_s=300.0, end_offset_ms=300_000).conditioning_ready is True

    def test_normal_duplicate_and_out_of_order_still_handled(self):
        # F3 detection must not disturb legitimate dedup/reorder behaviour
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")
        acc.observe(300_000, _m(6.0), "sensor")
        acc.observe(300_000, _m(99.0), "sensor")   # duplicate -> ignored
        acc.observe(100_000, _m(99.0), "sensor")   # out of order -> ignored
        acc.observe(600_000, _m(7.0), "sensor")
        ev = acc.finalize(duration_s=600.0, end_offset_ms=600_000)
        assert ev.sample_count == 3 and ev.usable_count == 3
        assert ev.max_c == 7.0                                   # the 99.0 never landed
        assert ev.conditioning_ready is True

    def _future_offset_episode(self, builder, open_regime, close_regime, indoor_fn):
        builder.process(BuilderInput(ts=_T0, regime=RegimeResult(open_regime, 0.9),
                                     indoor_temp=indoor_fn(0), target=21.0,
                                     outdoor_temp=_m(2.0), outdoor_source="sensor"))
        builder.process(BuilderInput(ts=_T0 + timedelta(seconds=300),
                                     regime=RegimeResult(open_regime, 0.9),
                                     indoor_temp=indoor_fn(1), target=21.0,
                                     outdoor_temp=_m(4.0), outdoor_source="sensor"))
        snap = builder.snapshot_state()
        snap["outdoor"]["lo"] = 5_000_000                        # far beyond the episode end
        restored = type(builder)("lz")
        restored.restore_state(snap)
        return restored.process(BuilderInput(
            ts=_T0 + timedelta(seconds=600), regime=RegimeResult(close_regime, 0.9),
            indoor_temp=indoor_fn(2), target=21.0, outdoor_temp=_m(6.0), outdoor_source="sensor"))

    def test_heating_lifecycle_with_future_offset(self):
        res = self._future_offset_episode(
            HeatingEpisodeBuilder("lz"), Regime.ACTIVE_HEATING, Regime.STABLE,
            indoor_fn=lambda i: 18.0 + i * 0.5)
        assert res.action is BuilderAction.CLOSED
        assert res.outdoor_evidence.conditioning_ready is False
        assert res.outdoor_evidence.time_weighted_mean_c is None

    def test_passive_cooling_lifecycle_with_future_offset(self):
        res = self._future_offset_episode(
            PassiveCoolingEpisodeBuilder("lz"), Regime.PASSIVE_COOLING, Regime.STABLE,
            indoor_fn=lambda i: 21.0 - i * 0.3)
        assert res.action is BuilderAction.CLOSED
        assert res.outdoor_evidence.conditioning_ready is False
        assert res.outdoor_evidence.time_weighted_mean_c is None
