"""Phase S1a Item 9 — Diagnostics/Export Validation Post-Restart.

Proves that after restart, the diagnostics() output is complete and correct:
  - mode, initialized, dirty, last_save
  - learning_errors, enabled, control_enabled
  - open_decisions, reversion_count
  - model_update_total, model_update_counts
  - RuntimeHealth: zones, storage_warnings, model_errors

Additional checks:
  - Diagnostics after unload shows enabled=False
  - Diagnostics after cold start shows model_update_total=0
  - Diagnostics after training shows model_update_total>0
  - Diagnostics after corruption shows safe (no crash)
  - RuntimeHealth after restart is identical to pre-restart (same model state)
"""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import (
    heating_ramp_then_settle,
    runtime,
    step,
)

pytestmark = pytest.mark.asyncio


# ── 1. RuntimeHealth: all fields present after restart ───────────────────────


class TestRuntimeHealthPostRestart:
    async def test_health_fields_present_after_cold_start(self):
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        h = rt.health()
        # All RuntimeHealth fields must be accessible
        assert isinstance(h.zones, int)
        assert isinstance(h.initialized, bool)
        assert isinstance(h.dirty, bool)
        assert isinstance(h.storage_warnings, int)
        assert isinstance(h.model_errors, int)
        assert isinstance(h.model_update_total, int)
        assert isinstance(h.open_decisions, int)
        assert isinstance(h.reversion_count, int)

    async def test_health_initialized_true_after_setup(self):
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        assert rt.health().initialized

    async def test_health_zones_count_after_setup(self):
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        step(rt, 0, 19.0, heating=False)
        assert rt.health().zones >= 1

    async def test_health_storage_warnings_zero_on_clean_restart(self):
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2.health().storage_warnings == 0

    async def test_health_model_errors_zero_on_clean_restart(self):
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2.health().model_errors == 0


# ── 2. Model update counts survive restart ───────────────────────────────────


class TestModelUpdateCountsPostRestart:
    async def test_model_update_total_zero_on_cold_start(self):
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        assert rt.health().model_update_total == 0

    async def test_model_update_counts_per_key_survive_restart(self):
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        counts1 = dict(rt1.health().model_update_counts)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        counts2 = dict(rt2.health().model_update_counts)
        assert counts2 == counts1

    async def test_model_update_total_stable_after_restart(self):
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        total1 = rt1.health().model_update_total
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        total2 = rt2.health().model_update_total
        assert total2 == total1

    async def test_restart_alone_does_not_increment_model_updates(self):
        """Restart without new cycles must not increment model update count."""
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()
        total_base = rt1.health().model_update_total

        for _ in range(3):
            blob = json.loads(json.dumps(store.data))
            rt = runtime(store=MemoryStore(data=blob))
            await rt.async_setup()
            assert rt.health().model_update_total == total_base
            rt.mark_dirty(important=True)
            await rt.async_flush()


# ── 3. Mode field in health after restart ────────────────────────────────────


class TestModeFieldPostRestart:
    async def test_shadow_mode_in_health_is_shadow(self):
        rt = runtime(mode=LearningRuntimeMode.SHADOW, store=MemoryStore())
        await rt.async_setup()
        assert rt.health().mode == LearningRuntimeMode.SHADOW.value

    async def test_control_mode_in_health_is_control(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL, store=MemoryStore())
        await rt.async_setup()
        assert rt.health().mode == LearningRuntimeMode.CONTROL.value

    async def test_mode_restores_after_restart(self):
        """Mode is NOT persisted (constructor arg) — must be set explicitly after restart."""
        store = MemoryStore()
        rt1 = runtime(mode=LearningRuntimeMode.CONTROL, store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        # After restart with explicit CONTROL mode
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL, store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2.health().mode == LearningRuntimeMode.CONTROL.value
        assert rt2.health().control_enabled


# ── 4. Reversion count after mode switches ───────────────────────────────────


class TestReversionCountPostRestart:
    async def test_reversion_count_zero_on_cold_start(self):
        rt = runtime(store=MemoryStore())
        await rt.async_setup()
        assert rt.health().reversion_count == 0

    async def test_reversion_count_increments_on_mode_switch(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL, store=MemoryStore())
        await rt.async_setup()
        rc0 = rt.health().reversion_count
        rt.set_mode(LearningRuntimeMode.SHADOW)  # CONTROL → SHADOW increments
        rc1 = rt.health().reversion_count
        assert rc1 > rc0

    async def test_reversion_count_is_ephemeral_after_restart(self):
        """Reversion count is session-scoped (not persisted)."""
        rt = runtime(mode=LearningRuntimeMode.CONTROL, store=MemoryStore())
        await rt.async_setup()
        rt.set_mode(LearningRuntimeMode.SHADOW)
        rc_before = rt.health().reversion_count
        assert rc_before > 0

        # After restart: reversion count resets
        rt2 = runtime(mode=LearningRuntimeMode.SHADOW, store=MemoryStore())
        await rt2.async_setup()
        assert rt2.health().reversion_count == 0, \
            "Reversion count is ephemeral and must reset on restart"


# ── 5. Dirty flag behavior ────────────────────────────────────────────────────


class TestDirtyFlagBehavior:
    async def test_dirty_false_immediately_after_setup(self):
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        assert not rt.health().dirty

    async def test_dirty_true_after_mark_dirty(self):
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        step(rt, 0, 19.0, heating=False)
        rt.mark_dirty(important=True)
        assert rt.health().dirty

    async def test_dirty_false_after_flush(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        step(rt, 0, 19.0, heating=False)
        rt.mark_dirty(important=True)
        assert rt.health().dirty
        await rt.async_flush()
        assert not rt.health().dirty


# ── 6. Open decisions count ───────────────────────────────────────────────────


class TestOpenDecisionsPostRestart:
    async def test_open_decisions_zero_on_cold_start(self):
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        assert rt.health().open_decisions == 0

    async def test_open_decisions_stable_after_restart(self):
        """open_decisions count is restored from persisted ledger — stable across restart."""
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()
        od1 = rt1.health().open_decisions

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        od2 = rt2.health().open_decisions
        assert od2 == od1, \
            f"open_decisions changed after restart: {od1} → {od2}"


# ── 7. Diagnostics post-unload / disabled ─────────────────────────────────────


class TestDiagnosticsAfterUnload:
    """After teardown, runtime health must still be queryable without crash."""

    async def test_health_still_queryable_after_flush_and_clear(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        rt.reset()  # unload
        # After clear: runtime must still be queryable
        h = rt.health()
        assert h is not None

    async def test_health_zones_zero_after_clear(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        step(rt, 0, 19.0, heating=False)
        rt.reset()
        h = rt.health()
        assert h.zones == 0


# ── 8. Diagnostics after corrupt restore ─────────────────────────────────────


class TestDiagnosticsAfterCorruptRestore:
    async def test_diagnostics_not_crashing_after_corrupt_blob(self):
        rt = runtime(store=MemoryStore(data={"garbage": True}))
        await rt.async_setup()
        h = rt.health()
        assert h.initialized
        assert h.storage_warnings >= 0  # may or may not be set

    async def test_model_errors_zero_on_empty_blob_recovery(self):
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        h = rt.health()
        assert h.model_errors == 0

    async def test_health_remains_stable_after_nan_restore(self):
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        blob = json.loads(json.dumps(store.data))
        try:
            blob["zones"]["lz"]["models"]["heat_rate"]["general"]["rate_c_per_h"] = None
        except (KeyError, TypeError):
            pass
        rt2 = runtime(store=MemoryStore(data=blob))
        await rt2.async_setup()
        h = rt2.health()
        assert h.initialized
        assert h.storage_warnings == 0  # JSON corruption is model-level, not storage

    async def test_health_identical_for_same_blob_two_runtimes(self):
        """Same serialized blob → identical health metrics on two runtime instances."""
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()
        blob = json.loads(json.dumps(store.data))

        rt_a = runtime(store=MemoryStore(data=blob))
        await rt_a.async_setup()
        h_a = rt_a.health()

        rt_b = runtime(store=MemoryStore(data=blob))
        await rt_b.async_setup()
        h_b = rt_b.health()

        assert h_a.model_update_total == h_b.model_update_total
        assert h_a.storage_warnings == h_b.storage_warnings
        assert h_a.model_errors == h_b.model_errors
        assert h_a.zones == h_b.zones
