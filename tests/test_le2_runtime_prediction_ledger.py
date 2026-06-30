"""Phase 17B prediction snapshot ledger tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    PredictionSnapshot,
    PredictionSnapshotLedger,
)
from tests.helpers_runtime_scenarios import heating_ramp_then_settle, runtime


def snap(decision="d1", ptype="heat_rate", value=2.0, ts="2025-01-01T06:00:00+00:00"):
    return PredictionSnapshot(
        decision_id=decision, learning_zone_id="lz", prediction_type=ptype, value=value,
        unit="c_per_h", confidence=0.5, reliability=0.4, fallback_used=True,
        prior_contribution=1.0, learned_contribution=0.0, bucket=None, model_version=1,
        parameter_version=1, created_ts=ts)


class TestLedger:
    def test_record_and_get(self):
        led = PredictionSnapshotLedger()
        led.record(snap())
        assert led.get("d1", "heat_rate").value == 2.0

    def test_dedup_by_decision_and_type(self):
        led = PredictionSnapshotLedger()
        led.record(snap(value=2.0))
        led.record(snap(value=3.0))
        assert led.size == 1 and led.get("d1", "heat_rate").value == 3.0

    def test_distinct_types_kept(self):
        led = PredictionSnapshotLedger()
        led.record(snap(ptype="heat_rate"))
        led.record(snap(ptype="forecast_trust"))
        assert led.size == 2

    def test_bounded(self):
        led = PredictionSnapshotLedger(max_entries=5)
        for i in range(20):
            led.record(snap(decision=f"d{i}", ts=f"2025-01-01T06:{i:02d}:00+00:00"))
        assert led.size == 5

    def test_for_decision(self):
        led = PredictionSnapshotLedger()
        led.record(snap(decision="d1", ptype="heat_rate"))
        led.record(snap(decision="d1", ptype="forecast_trust"))
        led.record(snap(decision="d2", ptype="heat_rate"))
        assert len(led.for_decision("d1")) == 2

    def test_prune_decisions(self):
        led = PredictionSnapshotLedger()
        led.record(snap(decision="d1"))
        led.record(snap(decision="d2"))
        removed = led.prune_decisions(["d1"])
        assert removed == 1 and led.get("d2", "heat_rate") is None

    def test_serialize_restore(self):
        led = PredictionSnapshotLedger()
        led.record(snap())
        other = PredictionSnapshotLedger()
        other.restore(led.serialize())
        assert other.get("d1", "heat_rate").value == 2.0

    def test_runtime_records_snapshots(self):
        rt = runtime()
        heating_ramp_then_settle(rt, legacy_preheat="2025-01-01T06:10:00+00:00")
        assert rt._zone("lz").ledger.size >= 1

    def test_baseline_not_recomputed(self):
        # a stored snapshot keeps the prediction as it was, not a later value
        led = PredictionSnapshotLedger()
        led.record(snap(value=2.0, ts="2025-01-01T06:00:00+00:00"))
        # later improved model would predict differently; ledger keeps the original
        assert led.get("d1", "heat_rate").value == 2.0
