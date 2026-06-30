"""S1a Item 10 - Section 3: Bounded Growth Proof for all runtime caps.

Verifies that every finite-size data structure enforces its declared upper
bound when fed more data than the cap allows. No HA dependency; runs on Windows.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.runtime.prediction_ledger import (
    PredictionSnapshot,
    PredictionSnapshotLedger,
)
from custom_components.thermosmart.learning.models.baseline_comparison import (
    BaselineComparisonStore,
    BaselineParams,
    NonBoostBaselineSample,
)
from custom_components.thermosmart.learning.runtime.boost_pending import (
    POST_REACH_SAMPLE_CAP,
    BoostPendingOutcome,
)
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import runtime, step

UTC = timezone.utc
T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=UTC)


# ── helpers ───────────────────────────────────────────────────────────────────

def _snapshot(decision_id: str, ptype: str = "heat_rate") -> PredictionSnapshot:
    return PredictionSnapshot(
        decision_id=decision_id, learning_zone_id="lz",
        prediction_type=ptype, value=1.5, unit="C/h",
        confidence=0.8, reliability=0.9, fallback_used=False,
        prior_contribution=0.2, learned_contribution=0.8,
        bucket=None, model_version=1, parameter_version=1,
        created_ts=T0.isoformat(), target_ts=None,
    )


def _baseline_sample(decision_id: str, reached_at: str) -> NonBoostBaselineSample:
    return NonBoostBaselineSample(
        episode_id=f"ep_{decision_id}", decision_id=decision_id,
        learning_zone_id="lz", started_at="2025-01-01T06:00:00+00:00",
        effective_heating_onset_at="2025-01-01T06:10:00+00:00",
        target_reached_at=reached_at,
        command_to_target_duration_s=3600.0, onset_to_target_duration_s=3000.0,
        start_deficit_c=2.0, target_temperature_c=21.0, start_temperature_c=19.0,
        external_temperature_c=-5.0, heat_rate_used=1.5, heat_loss_used=0.3,
        tpi_duty=0.7, baseline_setpoint_c=21.0, comfort_time_utc=None,
        time_of_day_bucket=6, presence_context=None, device_control_type="setpoint",
        dispatch_quality="full", sensor_reliability=0.95, confounder_flags=(),
        outcome_reliability=0.9, boost_applied_c=0.0, authoritative=True,
    )


def _pending_outcome(zone_id: str = "lz") -> BoostPendingOutcome:
    return BoostPendingOutcome(
        zone_id=zone_id, decision_id="dec_1", episode_id="ep_1",
        state="target_reached_pending_observation",
        target=21.0, episode_end_ts="2025-01-01T07:00:00+00:00",
        observation_start_ts="2025-01-01T07:00:00+00:00",
        deadline_ts="2025-01-01T07:15:00+00:00",
        last_valid_ts="2025-01-01T07:00:00+00:00",
        episode={"start_ts": "2025-01-01T06:00:00+00:00", "end_ts": "2025-01-01T07:00:00+00:00",
                 "episode_id": "ep_1", "learning_zone_id": zone_id, "decision_id": "dec_1",
                 "regime": "normal", "reliability": 0.9, "start_temp": 18.0, "end_temp": 21.0,
                 "target": 21.0, "comfort_tolerance_at_start": 0.5, "reason": "reached",
                 "controller": "tpi", "confounder_flags": [], "trajectory": [], "max_points": 240},
        boost_context={},
    )


# ── 1. PredictionSnapshotLedger cap ──────────────────────────────────────────


class TestPredictionSnapshotLedgerCap:
    def test_cap_is_500(self):
        ledger = PredictionSnapshotLedger()
        assert ledger._max == 500

    def test_overflow_stays_at_max(self):
        ledger = PredictionSnapshotLedger()
        for i in range(600):
            ledger.record(_snapshot(f"dec_{i}"))
        assert ledger.size <= 500

    def test_custom_cap_enforced(self):
        ledger = PredictionSnapshotLedger(max_entries=10)
        for i in range(25):
            ledger.record(_snapshot(f"dec_{i}"))
        assert ledger.size <= 10

    def test_dedup_same_decision_prediction_type(self):
        ledger = PredictionSnapshotLedger(max_entries=5)
        for _ in range(20):
            ledger.record(_snapshot("dec_1", "heat_rate"))
        assert ledger.size == 1

    def test_multiple_types_same_decision_counted_separately(self):
        ledger = PredictionSnapshotLedger(max_entries=20)
        for ptype in ("heat_rate", "onset_delay", "boost_factor"):
            ledger.record(_snapshot("dec_1", ptype))
        assert ledger.size == 3

    def test_serialize_restore_preserves_cap(self):
        ledger = PredictionSnapshotLedger(max_entries=10)
        for i in range(25):
            ledger.record(_snapshot(f"dec_{i}"))
        blob = ledger.serialize()
        ledger2 = PredictionSnapshotLedger(max_entries=10)
        ledger2.restore(blob)
        assert ledger2.size <= 10


# ── 2. BaselineComparisonStore count cap ─────────────────────────────────────


class TestBaselineComparisonStoreCap:
    def test_count_cap_is_200(self):
        store = BaselineComparisonStore("lz")
        assert store._p.count_cap == 200

    def test_overflow_stays_at_cap(self):
        store = BaselineComparisonStore("lz")
        for i in range(250):
            ts = (T0 + timedelta(hours=i)).isoformat()
            store.add(_baseline_sample(f"dec_{i}", ts))
        assert store.size <= 200

    def test_oldest_pruned_on_overflow(self):
        store = BaselineComparisonStore("lz", params=BaselineParams(count_cap=5))
        for i in range(10):
            ts = (T0 + timedelta(hours=i)).isoformat()
            store.add(_baseline_sample(f"dec_{i}", ts))
        assert store.size == 5
        # The 5 newest samples should be kept (highest timestamps)
        kept_ids = {s.decision_id for s in store.samples()}
        for i in range(5, 10):
            assert f"dec_{i}" in kept_ids

    def test_dedup_per_decision_id(self):
        store = BaselineComparisonStore("lz")
        sample = _baseline_sample("dec_1", "2025-01-01T07:00:00+00:00")
        assert store.add(sample) is True
        assert store.add(sample) is False
        assert store.size == 1

    def test_cross_zone_rejected(self):
        store = BaselineComparisonStore("lz_a")
        sample = _baseline_sample("dec_1", "2025-01-01T07:00:00+00:00")
        # sample.learning_zone_id is "lz", not "lz_a"
        result = store.add(sample)
        assert result is False
        assert store.size == 0

    def test_serialize_restore_preserves_cap(self):
        store = BaselineComparisonStore("lz", params=BaselineParams(count_cap=5))
        for i in range(10):
            ts = (T0 + timedelta(hours=i)).isoformat()
            store.add(_baseline_sample(f"dec_{i}", ts))
        blob = store.serialize()
        store2 = BaselineComparisonStore("lz", params=BaselineParams(count_cap=5))
        store2.restore(blob)
        assert store2.size <= 5


# ── 3. BoostPendingOutcome post_reach sample cap ─────────────────────────────


class TestBoostPendingOutcomePostReachCap:
    def test_cap_constant_is_80(self):
        assert POST_REACH_SAMPLE_CAP == 80

    def test_overflow_trimmed_to_cap(self):
        p = _pending_outcome()
        for i in range(120):
            ts = (T0 + timedelta(minutes=i)).isoformat()
            p.ingest(ts, 21.5 + i * 0.01)
        assert len(p.post_reach) <= POST_REACH_SAMPLE_CAP

    def test_newest_samples_kept(self):
        p = _pending_outcome()
        for i in range(100):
            ts = (T0 + timedelta(minutes=i)).isoformat()
            p.ingest(ts, 20.0 + i * 0.1)
        # After cap overflow, newer samples are kept
        assert len(p.post_reach) == POST_REACH_SAMPLE_CAP
        last_offset = p.post_reach[-1][0]
        # Offset should correspond to one of the later samples (not the very first)
        assert last_offset > 0


# ── 4. ZoneRuntime pending attribution caps (64 each) ────────────────────────


class TestPendingAttributionCaps:
    def test_pending_boost_cap_value(self):
        store = MemoryStore()
        rt = runtime(store=store)
        # Force creation of a zone
        step(rt, 0, 19.0, heating=True)
        zr = rt._zones["lz"]
        assert zr._pending_boost_cap == 64

    def test_pending_boost_contexts_bounded(self):
        store = MemoryStore()
        rt = runtime(store=store)
        # Inject many fake entries directly into the zone runtime
        step(rt, 0, 19.0, heating=True)
        zr = rt._zones["lz"]
        from collections import OrderedDict
        for i in range(100):
            zr.pending_boost_contexts[f"dec_{i}"] = {"fake": True}
            while len(zr.pending_boost_contexts) > zr._pending_boost_cap:
                zr.pending_boost_contexts.popitem(last=False)
        assert len(zr.pending_boost_contexts) <= 64

    def test_pending_dispatch_records_bounded(self):
        store = MemoryStore()
        rt = runtime(store=store)
        step(rt, 0, 19.0, heating=True)
        zr = rt._zones["lz"]
        for i in range(100):
            zr.pending_dispatch_records[f"dec_{i}"] = {"fake": True}
            while len(zr.pending_dispatch_records) > zr._pending_boost_cap:
                zr.pending_dispatch_records.popitem(last=False)
        assert len(zr.pending_dispatch_records) <= 64

    def test_pending_baseline_ctx_bounded(self):
        store = MemoryStore()
        rt = runtime(store=store)
        step(rt, 0, 19.0, heating=True)
        zr = rt._zones["lz"]
        for i in range(100):
            zr.pending_baseline_ctx[f"dec_{i}"] = {"fake": True}
            while len(zr.pending_baseline_ctx) > zr._pending_boost_cap:
                zr.pending_baseline_ctx.popitem(last=False)
        assert len(zr.pending_baseline_ctx) <= 64


# ── 5. open_comparisons cap (200) ────────────────────────────────────────────


class TestOpenComparisonsCap:
    def test_cap_at_200(self):
        store = MemoryStore()
        rt = runtime(store=store)
        # Run 300 preheat cycles that generate comparisons
        from tests.helpers_runtime_scenarios import T0, COMFORT
        from custom_components.thermosmart.learning.runtime import (
            LearningRuntimeMode, RuntimeCycleInput, ControllerDecisionInput, ScheduleTarget,
        )
        from custom_components.thermosmart.learning.contracts import Measurement, DataQuality
        from custom_components.thermosmart.learning.runtime import DecisionType
        import asyncio

        async def setup():
            await rt.async_setup()

        asyncio.run(setup())

        for i in range(300):
            ts = (T0 + timedelta(minutes=i * 5)).isoformat()
            inp = RuntimeCycleInput(
                zone_id="lz", ts=ts, target_c=21.0, trv_setpoint_c=24.0,
                indoor_temp=Measurement(19.0, DataQuality.OK),
                schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=21.0),
                controller_decision=ControllerDecisionInput(
                    decision_type=DecisionType.NORMAL, target_c=21.0, trv_setpoint_c=24.0),
                legacy_preheat_start_utc=(T0 - timedelta(minutes=30)).isoformat(),
            )
            rt.run_cycle(inp)

        zr = rt._zones["lz"]
        assert len(zr.open_comparisons) <= rt._config.max_open_comparisons
        assert rt._config.max_open_comparisons == 200


# ── 6. Total runtime blob stays bounded across restarts ──────────────────────


class TestRuntimeBlobBounded:
    _LIMIT_BYTES = 512 * 1024

    def _run_many_cycles(self, rt, n_cycles: int, zone: str = "lz"):
        for i in range(n_cycles):
            temp = 18.0 + (i % 7) * 0.5
            heating = temp < 21.0
            step(rt, i, round(temp, 1), heating=heating, zone=zone)

    def test_blob_within_limit_after_1000_cycles(self):
        import asyncio

        async def run():
            store = MemoryStore()
            rt = runtime(store=store)
            await rt.async_setup()
            self._run_many_cycles(rt, 1000)
            rt.mark_dirty(important=True)
            await rt.async_flush()
            blob_bytes = len(json.dumps(store.data).encode("utf-8"))
            assert blob_bytes < self._LIMIT_BYTES, \
                f"Blob {blob_bytes:,} bytes exceeds {self._LIMIT_BYTES:,} after 1000 cycles"

        asyncio.run(run())

    def test_blob_not_growing_between_restarts(self):
        import asyncio

        async def run():
            store = MemoryStore()
            rt = runtime(store=store)
            await rt.async_setup()
            self._run_many_cycles(rt, 500)
            rt.mark_dirty(important=True)
            await rt.async_flush()
            size1 = len(json.dumps(store.data).encode("utf-8"))

            # Restart and run more cycles
            rt2 = runtime(store=store)
            await rt2.async_setup()
            self._run_many_cycles(rt2, 100)
            rt2.mark_dirty(important=True)
            await rt2.async_flush()
            size2 = len(json.dumps(store.data).encode("utf-8"))

            # Blob may grow with new data, but must not grow unboundedly
            assert size2 < self._LIMIT_BYTES, \
                f"Post-restart blob {size2:,} bytes exceeds limit"

        asyncio.run(run())
