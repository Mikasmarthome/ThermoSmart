"""Phase 17B cross-cycle episode pipeline tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import RuntimePipeline
from custom_components.thermosmart.learning.runtime.capture import CaptureCoordinator
from tests.helpers_runtime import cycle_input
from tests.helpers_runtime_scenarios import heating_ramp_then_settle, runtime, step


class TestEpisodePipeline:
    def test_heating_episode_completes_and_updates_heat_rate(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        zr = rt._zone("lz")
        assert zr.model_update_counts.get("heat_rate", 0) >= 1
        assert zr.orchestrator.models["heat_rate"]._state.general.sample_count >= 1

    def test_outcome_episode_completes(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        assert rt._zone("lz").model_update_counts.get("outcome", 0) >= 1

    def test_snapshot_dedup(self):
        pipe = RuntimePipeline("lz")
        cap = CaptureCoordinator("lz")
        snap = cap.capture(cycle_input("2025-01-01T06:00:00+00:00", zone="lz"))
        r1 = pipe.process(snap)
        r2 = pipe.process(snap)   # same ts -> deduplicated
        assert not r1.deduplicated and r2.deduplicated

    def test_regime_present(self):
        pipe = RuntimePipeline("lz")
        cap = CaptureCoordinator("lz")
        for i in range(5):
            snap = cap.capture(cycle_input(f"2025-01-01T06:0{i}:00+00:00", indoor=18.0 + i * 0.5, zone="lz"))
            r = pipe.process(snap)
        assert r.regime is not None

    def test_one_episode_per_heating_cycle(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        # a single ramp+settle yields exactly one heating episode
        assert rt._zone("lz").model_update_counts.get("heat_rate", 0) == 1

    def test_two_zones_no_cross_contamination(self):
        rt = runtime()
        heating_ramp_then_settle(rt, zone="a")
        heating_ramp_then_settle(rt, zone="b")
        a = rt._zone("a").orchestrator.models["heat_rate"]._state.general.sample_count
        b = rt._zone("b").orchestrator.models["heat_rate"]._state.general.sample_count
        assert a >= 1 and b >= 1
        assert rt._zone("a").pipeline is not rt._zone("b").pipeline

    def test_pipeline_serialize_restore(self):
        pipe = RuntimePipeline("lz")
        cap = CaptureCoordinator("lz")
        for i in range(4):
            pipe.process(cap.capture(cycle_input(f"2025-01-01T06:0{i}:00+00:00",
                                                 indoor=18.0 + i * 0.4, zone="lz")))
        data = pipe.serialize()
        other = RuntimePipeline("lz")
        assert other.restore(data) == ()

    def test_disturbed_blocks_updates(self):
        rt = runtime()
        # window open during heating -> DISTURBED -> no thermal model update
        for i in range(8):
            step(rt, i, 18.0 + i * 0.3, heating=True, window_open=True)
        assert rt._zone("lz").model_update_counts.get("heat_rate", 0) == 0
