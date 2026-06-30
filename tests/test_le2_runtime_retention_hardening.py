"""Phase 17B retention / bounded-state hardening."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    CommandLedger,
    CommandType,
    PredictionSnapshotLedger,
    RuntimePipeline,
    SentCommand,
)
from custom_components.thermosmart.learning.runtime.capture import CaptureCoordinator
from custom_components.thermosmart.learning.runtime.prediction_ledger import PredictionSnapshot
from tests.helpers_runtime import cycle_input
from tests.helpers_runtime_scenarios import runtime, step


def _snap(i):
    return PredictionSnapshot(
        decision_id=f"d{i}", learning_zone_id="lz", prediction_type="heat_rate", value=2.0,
        unit="c", confidence=0.5, reliability=0.4, fallback_used=True, prior_contribution=1.0,
        learned_contribution=0.0, bucket=None, model_version=1, parameter_version=1,
        created_ts=f"2025-01-01T06:{i % 60:02d}:00+00:00")


class TestRetention:
    def test_prediction_ledger_bounded(self):
        led = PredictionSnapshotLedger(max_entries=20)
        for i in range(200):
            led.record(_snap(i))
        assert led.size == 20

    def test_command_ledger_bounded(self):
        led = CommandLedger(max_entries=10)
        for i in range(100):
            led.record(SentCommand(f"c{i}", CommandType.SETPOINT, "t", float(i),
                                   f"2025-01-01T06:{i % 60:02d}:00+00:00"))
        assert led.size == 10

    def test_pipeline_trajectory_bounded(self):
        pipe = RuntimePipeline("lz", max_trajectory_points=12)
        cap = CaptureCoordinator("lz")
        for i in range(50):
            pipe.process(cap.capture(cycle_input(
                f"2025-01-01T{6 + i // 60:02d}:{i % 60:02d}:00+00:00", zone="lz",
                indoor=18.0 + (i % 5) * 0.2)))
        assert len(pipe._traj) <= 12 and len(pipe._processed_ts) <= 24

    def test_open_comparisons_bounded(self):
        rt = runtime()
        for i in range(60):
            step(rt, i, round(18.0 + (i % 6) * 0.4, 3), heating=(i % 2 == 0),
                 legacy_preheat="2025-01-01T06:10:00+00:00")
        assert len(rt._zone("lz").open_comparisons) <= rt._config.max_open_comparisons

    def test_no_full_serialize_per_cycle(self):
        rt = runtime()
        calls = {"n": 0}
        zr = rt._zone("lz")
        orig = zr.orchestrator.serialize_models
        zr.orchestrator.serialize_models = lambda: (calls.__setitem__("n", calls["n"] + 1)
                                                    or orig())
        for i in range(10):
            step(rt, i, 19.0, heating=True)
        assert calls["n"] == 0

    def test_prediction_ledger_prune(self):
        led = PredictionSnapshotLedger()
        for i in range(5):
            led.record(_snap(i))
        kept = led.prune_decisions(["d0", "d1"])
        assert kept == 3 and led.size == 2
