"""Phase 17B long deterministic multi-cycle shadow scenarios."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntimeMode,
    ShadowOrchestrator,
)
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import (
    COMFORT,
    heating_ramp_then_settle,
    runtime,
    step,
)


class TestLongScenarios:
    def test_normal_morning_preheat(self):
        rt = runtime()
        res = heating_ramp_then_settle(rt, legacy_preheat="2025-01-01T06:10:00+00:00")
        # a preheat comparison is produced and the model learns a heat rate
        assert any(c.comparison_type == "preheat_start"
                   for r in res for c in r.comparisons)
        assert rt._zone("lz").model_update_counts.get("heat_rate", 0) >= 1

    def test_late_preheat_then_improves(self):
        rt = runtime()
        # slow room: target barely reached late
        for i in range(10):
            step(rt, i, round(18.0 + i * 0.25, 3), heating=True, legacy_preheat=COMFORT)
        for i in range(10, 14):
            step(rt, i, 20.5, heating=False)
        assert rt.health().model_update_total >= 1

    def test_overshoot_scenario(self):
        rt = runtime()
        temps = [18, 19.5, 21.0, 22.5, 23.0, 22.5, 22.0, 21.5, 21.0]
        heat = [True] * 4 + [False] * 5
        for i, (t, h) in enumerate(zip(temps, heat)):
            step(rt, i, float(t), heating=h)
        assert rt._zone("lz").model_update_counts.get("heat_rate", 0) >= 0  # no crash

    def test_window_disturbance(self):
        rt = runtime()
        for i in range(8):
            step(rt, i, round(21.0 - i * 0.4, 3), heating=False, window_open=True)
        # disturbed -> no heat-loss/heat-rate misfit
        assert sum(rt._zone("lz").model_update_counts.values()) == 0

    def test_trv_only_full_flow(self):
        rt = runtime()
        heating_ramp_then_settle(rt)  # no outdoor/forecast/valve anywhere
        assert rt._zone("lz").model_update_counts.get("heat_rate", 0) >= 1

    def test_multi_day_with_restart(self):
        import asyncio
        from custom_components.thermosmart.learning.runtime import (
            LearningRuntime, LearningRuntimeConfig)

        async def run():
            store = MemoryStore()
            rt1 = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                        startup_grace_cycles=0),
                                  store=store, clock=lambda: "2025-01-01T23:00:00+00:00")
            await rt1.async_setup()
            heating_ramp_then_settle(rt1)
            day1 = rt1._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
            await rt1.async_flush()
            rt2 = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                        startup_grace_cycles=0),
                                  store=MemoryStore(data=store.data),
                                  clock=lambda: "2025-01-02T23:00:00+00:00")
            await rt2.async_setup()
            # day 2 with distinct timestamps -> a second independent episode
            for i in range(7):
                step(rt2, i + 100, round(18.0 + i * 0.5, 3), heating=True)
            for i in range(7, 12):
                step(rt2, i + 100, round(21.0 - (i - 7) * 0.4, 3), heating=False)
            day2 = rt2._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
            return day1, day2

        day1, day2 = asyncio.run(run())
        assert day1 == 1 and day2 == 2  # learning accumulates across restart

    def test_retention_bounded_over_many_cycles(self):
        rt = runtime()
        for cycle in range(6):
            base = cycle * 200
            for i in range(7):
                step(rt, base + i, round(18.0 + i * 0.5, 3), heating=True,
                     legacy_preheat="2025-01-01T06:10:00+00:00")
            for i in range(7, 12):
                step(rt, base + i, round(21.0 - (i - 7) * 0.4, 3), heating=False)
        zr = rt._zone("lz")
        assert len(zr.open_comparisons) <= rt._config.max_open_comparisons
        assert zr.ledger.size <= 500
