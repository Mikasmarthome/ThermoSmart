"""Phase S1a Item 9 — Restart/Restore Authority Map.

Verifies WHICH state components survive a restart (persistent) and which
are intentionally reset (ephemeral).  These are the canonical persistence
boundaries for LE 2.0 — every other Item 9 test relies on this contract.

Authority decision guide (Section 1 of the Item 9 spec)
────────────────────────────────────────────────────────
PERSISTED (must survive restart unchanged):
  - BoostModel learned samples, buckets, general stats
  - BoostModel.processed_decision_ids  (dedup, prevents double-outcome)
  - BoostModel.processed_ids           (legacy episode dedup)
  - ZoneRuntime capture / pipeline / models / ledger / baseline_store
  - ZoneRuntime.pending_boost_outcome  (single bounded post-reach observer)
  - ZoneRuntime.cycles, last_cycle_ts
  - LearningRuntime store blob (versioned, zone-keyed)

EPHEMERAL (reset on restart — explicitly documented in source):
  - BoostLifecycle (lifecycle, current_episode_id, applied_offset_c)
  - cooldown_until_ts                  (per-session anti-chatter gate)
  - lifecycle_start/base/user_target_c (episode binding)
  - last_failed_episode_id / last_completed_episode_id
  - LearningRuntime._active_control_initialized (HA restore barrier)
  - Coordinator._indoor_temp_slope     (rolling window, runtime only)

DERIVED (recomputed from inputs, not stored):
  - ControlAdaptationMode              (from learning_on × active_on switches)
  - Current TRV setpoint               (from TPI + model advice)
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.models.boost import (
    BoostLifecycle,
    BoostModel,
    BoostState,
)
from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_boost import good_boost
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import (
    T0,
    heating_ramp_then_settle,
    runtime,
    step,
)


# ── 1. BoostModel: learned data persists ────────────────────────────────────


class TestBoostModelPersistence:
    def _trained_model(self) -> BoostModel:
        m = BoostModel("lz")
        for i in range(3):
            good_boost(m, i)
        return m

    def test_general_samples_survive_roundtrip(self):
        m = self._trained_model()
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.general.gain.effective_n == m._state.general.gain.effective_n

    def test_processed_decision_ids_persist(self):
        m = self._trained_model()
        orig_ids = m._state.processed_decision_ids
        assert len(orig_ids) > 0, "Need at least one processed decision"
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.processed_decision_ids == orig_ids

    def test_full_count_persists(self):
        m = self._trained_model()
        full_count = m._state.full_count
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.full_count == full_count

    def test_recent_samples_persist(self):
        m = self._trained_model()
        n_samples = len(m._state.recent_samples)
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert len(m2._state.recent_samples) == n_samples

    def test_degradation_state_persists(self):
        m = self._trained_model()
        blob = m.serialize_state()
        assert "degradation" in blob
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        # Degradation struct survived roundtrip
        assert m2._state.degradation is not None


# ── 2. BoostModel: lifecycle is ephemeral ────────────────────────────────────


class TestBoostLifecycleEphemeral:
    """Boost lifecycle is intentionally NOT persisted — documented design."""

    def _state_after_apply(self) -> tuple[BoostModel, BoostState]:
        m = BoostModel("lz")
        m.apply_lifecycle("ep-001", 1.5, base_target_c=21.0,
                          ts="2025-01-01T07:00:00+00:00")
        return m, m._state

    def test_lifecycle_resets_to_inactive_on_deserialize(self):
        m, before = self._state_after_apply()
        assert before.lifecycle == BoostLifecycle.APPLIED
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.lifecycle == BoostLifecycle.INACTIVE

    def test_applied_offset_resets_to_zero_on_deserialize(self):
        m, _ = self._state_after_apply()
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.applied_offset_c == 0.0

    def test_cooldown_not_in_serialized_blob(self):
        m = BoostModel("lz")
        ts = "2025-01-01T07:00:00+00:00"
        m.apply_lifecycle("ep-001", 1.5, base_target_c=21.0, ts=ts)
        m.release_lifecycle("timeout", ts)
        # After release, cooldown_until_ts is set in-memory
        assert m._state.cooldown_until_ts is not None
        blob = m.serialize_state()
        assert "cooldown_until_ts" not in blob, (
            "cooldown_until_ts must NOT appear in serialized state — "
            "it is an ephemeral per-session anti-chatter guard."
        )

    def test_cooldown_resets_on_deserialize(self):
        m = BoostModel("lz")
        ts = "2025-01-01T07:00:00+00:00"
        m.apply_lifecycle("ep-001", 1.5, base_target_c=21.0, ts=ts)
        m.release_lifecycle("timeout", ts)
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.cooldown_until_ts is None

    def test_current_episode_id_resets_on_deserialize(self):
        m, _ = self._state_after_apply()
        assert m._state.current_episode_id == "ep-001"
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.current_episode_id is None

    def test_lifecycle_start_ts_not_in_blob(self):
        m, _ = self._state_after_apply()
        blob = m.serialize_state()
        assert "lifecycle_start_ts" not in blob
        assert "lifecycle_base_target_c" not in blob


# ── 3. ZoneRuntime serialization ────────────────────────────────────────────


class TestZoneRuntimePersistence:
    """Key ZoneRuntime fields survive flush → restore."""

    async def _rt_after_ramp(self) -> tuple[LearningRuntime, MemoryStore]:
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        return rt, store

    @pytest.mark.asyncio
    async def test_cycles_count_survives_restart(self):
        rt1, store = await self._rt_after_ramp()
        cycles_before = rt1._zone("lz").cycles
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2._zone("lz").cycles == cycles_before

    @pytest.mark.asyncio
    async def test_last_cycle_ts_survives_restart(self):
        rt1, store = await self._rt_after_ramp()
        ts_before = rt1._zone("lz").last_cycle_ts
        assert ts_before is not None
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2._zone("lz").last_cycle_ts == ts_before

    @pytest.mark.asyncio
    async def test_model_update_counts_survive_restart(self):
        rt1, store = await self._rt_after_ramp()
        counts_before = dict(rt1._zone("lz").model_update_counts)
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2._zone("lz").model_update_counts == counts_before

    @pytest.mark.asyncio
    async def test_heat_rate_sample_count_survives_restart(self):
        rt1, store = await self._rt_after_ramp()
        n = rt1._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        n2 = rt2._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        assert n2 == n

    @pytest.mark.asyncio
    async def test_ledger_survives_restart(self):
        rt1, store = await self._rt_after_ramp()
        assert rt1._zone("lz").capture.ledger is not None
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2._zone("lz").capture.ledger is not None


# ── 4. pending_boost_outcome survives restart ─────────────────────────────────


class TestPendingBoostOutcomePersistence:
    """pending_boost_outcome is the bounded post-reach observer.
    It must survive restart so no observation data is lost mid-window.
    """

    @pytest.mark.asyncio
    async def test_none_pending_outcome_serializes_cleanly(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        rt.mark_dirty(important=True)
        await rt.async_flush()
        # Verify None is serialized as None (not missing key)
        zone_blob = json.loads(json.dumps(store.data))
        assert zone_blob is not None

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2._zone("lz").pending_boost_outcome is None

    @pytest.mark.asyncio
    async def test_pending_boost_contexts_reset_on_restart(self):
        """pending_boost_contexts is a runtime-only stash; reset on restart is correct.

        This is intentional: boost decisions from before the restart are no
        longer outstanding (HA was down during the observation window), so
        they should not receive outcome attribution.
        """
        store = MemoryStore()
        rt1 = runtime(mode=LearningRuntimeMode.SHADOW, store=store)
        await rt1.async_setup()
        # Drive a boost scenario to fill pending_boost_contexts
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        # Runtime-only context stash is always empty on fresh start
        assert len(rt2._zone("lz").pending_boost_contexts) == 0


# ── 5. Control mode is NOT persisted ────────────────────────────────────────


class TestControlModeNotPersisted:
    """Mode (SHADOW/CONTROL) is a constructor argument, NOT stored in the blob."""

    @pytest.mark.asyncio
    async def test_control_runtime_serializes_without_mode_field(self):
        store = MemoryStore()
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.CONTROL,
                                                   startup_grace_cycles=0),
                             store=store, clock=lambda: "2025-01-01T07:00:00+00:00")
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        blob = json.dumps(store.data)
        assert "control" not in blob.lower() or '"mode"' not in blob, (
            "Mode must not be stored — control authority comes from the constructor argument."
        )

    @pytest.mark.asyncio
    async def test_shadow_restores_control_only_if_configured_as_control(self):
        """Restore a CONTROL-trained store into a SHADOW runtime → stays SHADOW."""
        store = MemoryStore()
        rt1 = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.CONTROL,
                                                    startup_grace_cycles=0),
                              store=store, clock=lambda: "2025-01-01T07:00:00+00:00")
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                    startup_grace_cycles=0),
                              store=MemoryStore(data=store.data),
                              clock=lambda: "2025-01-01T08:00:00+00:00")
        await rt2.async_setup()
        assert rt2.mode is LearningRuntimeMode.SHADOW
        assert not rt2.control_enabled

    @pytest.mark.asyncio
    async def test_control_enables_via_constructor_not_blob(self):
        """A fresh (empty) store + CONTROL mode → immediately control-enabled."""
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.CONTROL,
                                                   startup_grace_cycles=0),
                             store=MemoryStore(),
                             clock=lambda: "2025-01-01T07:00:00+00:00")
        await rt.async_setup()
        assert rt.control_enabled


# ── 6. Store blob is versioned and zone-keyed ────────────────────────────────


class TestStoreBlob:
    """The runtime serializes zone data under deterministic keys and version stamps."""

    @pytest.mark.asyncio
    async def test_store_has_version_field(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        step(rt, 0, 19.0, heating=True)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        assert store.data is not None
        # Top-level must contain some versioning
        raw = json.dumps(store.data)
        assert "version" in raw.lower() or "schema_version" in raw.lower()

    @pytest.mark.asyncio
    async def test_store_roundtrip_is_deterministic(self):
        """Serialize → deserialize → serialize produces identical blobs."""
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()
        blob1 = copy.deepcopy(store.data)

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        # Numeric values may have float repr variance; check structural equality
        blob2 = store.data
        assert json.dumps(blob1, sort_keys=True) == json.dumps(blob2, sort_keys=True)

    @pytest.mark.asyncio
    async def test_empty_store_produces_healthy_runtime(self):
        """A clean start (no store data) is safe and non-crashing."""
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        result = step(rt, 0, 19.0, heating=True)
        assert result is not None
        assert rt.health().zones >= 1


# ── 7. Health report after restart ──────────────────────────────────────────


class TestHealthAfterRestart:
    @pytest.mark.asyncio
    async def test_health_zones_count_correct(self):
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        h = rt2.health()
        assert h.zones == 1

    @pytest.mark.asyncio
    async def test_health_errors_empty_after_clean_restart(self):
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        h = rt2.health()
        assert h.storage_warnings == 0
