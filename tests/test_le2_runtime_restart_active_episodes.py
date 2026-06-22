"""Phase 17B restart/reload survival of active builder state."""
from __future__ import annotations

import asyncio

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import step

pytestmark = pytest.mark.asyncio


def _rt(store):
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                 startup_grace_cycles=0),
                           store=store, clock=lambda: "2025-01-01T07:00:00+00:00")


class TestRestart:
    async def test_active_heating_episode_survives_restart(self):
        store = MemoryStore()
        rt1 = _rt(store)
        await rt1.async_setup()
        # open a heating episode but do NOT close it (keep heating)
        for i in range(5):
            step(rt1, i, 18.0 + i * 0.5, heating=True)
        assert rt1._zone("lz").model_update_counts.get("heat_rate", 0) == 0  # still open
        await rt1.async_flush()

        # restart: restore builder state, then finish the episode
        rt2 = _rt(MemoryStore(data=store.data))
        await rt2.async_setup()
        for i in range(5, 10):
            step(rt2, i, 21.0, heating=False)
        # exactly one heating episode materialises after restart
        assert rt2._zone("lz").model_update_counts.get("heat_rate", 0) == 1

    async def test_no_duplicate_episode_after_restart(self):
        from tests.helpers_runtime_scenarios import heating_ramp_then_settle
        store = MemoryStore()
        rt1 = _rt(store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        count1 = rt1._zone("lz").model_update_counts.get("heat_rate", 0)
        await rt1.async_flush()
        samples1 = rt1._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        rt2 = _rt(MemoryStore(data=store.data))
        await rt2.async_setup()
        # re-feeding the same already-processed timestamps must not re-learn (dedup)
        heating_ramp_then_settle(rt2)
        samples2 = rt2._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        assert count1 == 1 and samples1 == 1
        assert samples2 == 1  # restored sample, not doubled

    async def test_command_ledger_survives_restart(self):
        store = MemoryStore()
        rt1 = _rt(store)
        await rt1.async_setup()
        step(rt1, 0, 20.0, heating=True)
        # ledger state is part of capture serialize -> survives
        await rt1.async_flush()
        rt2 = _rt(MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2._zone("lz").capture.ledger is not None

    async def test_health_after_restart(self):
        store = MemoryStore()
        rt1 = _rt(store)
        await rt1.async_setup()
        step(rt1, 0, 19.0, heating=True)
        await rt1.async_flush()
        rt2 = _rt(MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2.health().zones >= 1
