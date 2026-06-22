"""Phase 17 performance/retention budgets (operation counts, not ms benchmarks)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    CommandLedger,
    CommandType,
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
    PersistenceOrchestrator,
    SavePolicy,
    SentCommand,
)
from tests.helpers_runtime import MemoryStore, cycle_input


class TestPerformance:
    async def test_no_save_per_cycle(self):
        store = MemoryStore()
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW),
                             store=store, clock=lambda: "2026-06-22T05:00:00+00:00")
        await rt.async_setup()
        for i in range(10):
            rt.run_cycle(cycle_input(f"2026-06-22T05:0{i}:00+00:00"))
            await rt.async_save_if_due()
        # debounced: no per-cycle writes (clock is fixed so debounce never elapses)
        assert store.saves == 0

    async def test_open_comparisons_bounded(self):
        rt = LearningRuntime(LearningRuntimeConfig(
            mode=LearningRuntimeMode.SHADOW, max_open_comparisons=5))
        for i in range(20):
            rt.run_cycle(cycle_input(f"2026-06-22T05:{i:02d}:00+00:00",
                                     legacy_preheat="2026-06-22T06:00:00+00:00",
                                     decision_target=21.0 + i * 0.1))
        assert rt.health().open_comparisons <= 5

    def test_command_ledger_bounded(self):
        led = CommandLedger(max_entries=10)
        for i in range(50):
            led.record(SentCommand(f"c{i}", CommandType.SETPOINT, "trv1", float(i),
                                   f"2026-06-22T05:{i:02d}:00+00:00"))
        assert led.size == 10

    def test_cycle_does_not_full_serialize(self):
        # a plain cycle must not serialise model state (only mark dirty)
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW))
        calls = {"n": 0}
        orig = rt._zone("lr").orchestrator.serialize_models

        def counting():
            calls["n"] += 1
            return orig()
        rt._zone("lr").orchestrator.serialize_models = counting
        rt.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert calls["n"] == 0

    async def test_debounce_collapses_many_dirty(self):
        store = MemoryStore()
        p = PersistenceOrchestrator(store, policy=SavePolicy(debounce_s=30))
        for i in range(5):
            p.mark_dirty(f"2026-06-22T05:00:0{i}+00:00")
        # one save after debounce, not five
        await p.maybe_save("2026-06-22T05:01:00+00:00", lambda: {"x": 1})
        assert store.saves == 1
