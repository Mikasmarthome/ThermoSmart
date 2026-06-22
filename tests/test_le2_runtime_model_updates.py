"""Phase 17B model-update transaction tests."""
from __future__ import annotations

import pytest

from tests.helpers_runtime_scenarios import heating_ramp_then_settle, runtime, step


class TestModelUpdates:
    def test_heat_rate_updated_from_pipeline(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        assert rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count >= 1

    def test_update_counts_tracked(self):
        rt = runtime()
        heating_ramp_then_settle(rt)
        counts = rt._zone("lz").model_update_counts
        assert counts.get("heat_rate", 0) >= 1

    def test_no_update_under_startup_grace(self):
        from custom_components.thermosmart.learning.runtime import (
            LearningRuntime, LearningRuntimeConfig, LearningRuntimeMode)
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                   startup_grace_cycles=999))
        heating_ramp_then_settle(rt)
        assert rt._zone("lz").model_update_counts.get("heat_rate", 0) == 0

    def test_no_update_when_disturbed(self):
        rt = runtime()
        for i in range(8):
            step(rt, i, 18.0 + i * 0.3, heating=True, window_open=True)
        assert sum(rt._zone("lz").model_update_counts.values()) == 0

    def test_model_error_isolated(self):
        rt = runtime()
        # break one model; others must still update
        class _Bad:
            def update(self, ep, ctx=None):
                raise RuntimeError("boom")
        rt._zone("lz").orchestrator._models["heat_rate"] = _Bad()
        heating_ramp_then_settle(rt)
        # outcome still updates despite heat_rate failing
        assert rt._zone("lz").model_update_counts.get("outcome", 0) >= 1

    def test_dirty_marked_on_authoritative_change(self):
        from tests.helpers_runtime import MemoryStore
        from custom_components.thermosmart.learning.runtime import (
            LearningRuntime, LearningRuntimeConfig, LearningRuntimeMode)
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                   startup_grace_cycles=0),
                             store=MemoryStore(), clock=lambda: "2025-01-01T07:00:00+00:00")
        import asyncio
        asyncio.get_event_loop().run_until_complete(rt.async_setup()) if False else None
        heating_ramp_then_settle(rt)
        assert rt._persistence.dirty

    def test_two_zones_isolated_updates(self):
        rt = runtime()
        heating_ramp_then_settle(rt, zone="a")
        heating_ramp_then_settle(rt, zone="b")
        assert rt._zone("a").model_update_counts.get("heat_rate", 0) >= 1
        assert rt._zone("b").model_update_counts.get("heat_rate", 0) >= 1
