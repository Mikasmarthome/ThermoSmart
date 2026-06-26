"""Phase S1a Item 9 — Failure Injection and Safe Degradation (Section 13).

Tests that the system degrades safely when storage or restore fails:
  - Corrupt blob (invalid JSON structure)
  - Wrong model_version → rejected, cold-start fallback
  - Wrong learning_zone_id → rejected, cold-start fallback
  - Missing required fields → partial restore, not crash
  - Non-finite values in blob → rejected via validate_state
  - Unknown extra fields → ignored (forward-compatible)
  - Schema version mismatch → migrate or reject gracefully
  - Store save IOError → warning recorded, no exception propagation
  - Store load IOError → cold-start, no exception propagation
  - Corrupt pending_boost_outcome → None, no crash
  - Corrupt capture/pipeline/models individual sections → partial restore
"""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models.boost import BoostModel
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

_TS0 = "2025-01-01T07:00:00+00:00"


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _trained_store() -> MemoryStore:
    store = MemoryStore()
    rt = runtime(store=store)
    await rt.async_setup()
    heating_ramp_then_settle(rt)
    rt.mark_dirty(important=True)
    await rt.async_flush()
    return store


# ── 1. BoostModel corrupt payload ────────────────────────────────────────────


class TestBoostModelCorruptPayload:
    def test_wrong_model_version_raises_value_error(self):
        m = BoostModel("lz")
        blob = m.serialize_state()
        blob["model_version"] = 9999
        with pytest.raises(ValueError, match="model_version"):
            m2 = BoostModel("lz")
            m2.deserialize_state(blob)

    def test_wrong_learning_zone_id_raises_value_error(self):
        m = BoostModel("lz")
        blob = m.serialize_state()
        blob["learning_zone_id"] = "wrong_zone"
        with pytest.raises(ValueError, match="learning_zone_id"):
            m2 = BoostModel("lz")
            m2.deserialize_state(blob)

    def test_non_finite_gain_value_raises_value_error(self):
        m = BoostModel("lz")
        blob = m.serialize_state()
        blob["general"]["gain"]["value"] = float("inf")
        with pytest.raises((ValueError, Exception)):
            m2 = BoostModel("lz")
            m2.deserialize_state(blob)

    def test_completely_empty_blob_raises_key_error(self):
        with pytest.raises((KeyError, ValueError)):
            m = BoostModel("lz")
            m.deserialize_state({})

    def test_none_blob_raises(self):
        with pytest.raises((TypeError, AttributeError, KeyError, ValueError)):
            m = BoostModel("lz")
            m.deserialize_state(None)


# ── 2. Runtime-level corrupt payload ─────────────────────────────────────────


class TestRuntimeCorruptPayload:
    @pytest.mark.asyncio
    async def test_corrupt_json_structure_cold_starts_safely(self):
        """A fundamentally wrong blob structure must not crash the runtime."""
        corrupt_store = MemoryStore(data={"garbage": True, "no_zones": None})
        rt = runtime(store=corrupt_store)
        await rt.async_setup()
        # Must still be runnable
        result = step(rt, 0, 19.0, heating=True)
        assert result is not None

    @pytest.mark.asyncio
    async def test_missing_capture_key_continues_without_crash(self):
        """If 'capture' key is absent in zone blob, the zone restores partially."""
        store = await _trained_store()
        blob = json.loads(json.dumps(store.data))
        # Remove 'capture' from the zone data
        for key in list(blob.keys()):
            if isinstance(blob[key], dict) and "zones" in blob[key]:
                for zone_key in blob[key]["zones"]:
                    blob[key]["zones"][zone_key].pop("capture", None)
        corrupt_store = MemoryStore(data=blob)
        rt = runtime(store=corrupt_store)
        await rt.async_setup()
        result = step(rt, 0, 19.0, heating=True)
        assert result is not None

    @pytest.mark.asyncio
    async def test_corrupt_pending_boost_outcome_is_discarded(self):
        """Corrupt pending_boost_outcome blob: discarded as None, no crash."""
        store = await _trained_store()
        blob = json.loads(json.dumps(store.data))
        # Inject corrupt pending_boost_outcome at zone level
        for key in list(blob.keys()):
            if isinstance(blob[key], dict) and "zones" in blob[key]:
                for zone_key in blob[key]["zones"]:
                    blob[key]["zones"][zone_key]["pending_boost_outcome"] = {
                        "schema_version": 9999, "garbage": True}
        corrupt_store = MemoryStore(data=blob)
        rt = runtime(store=corrupt_store)
        await rt.async_setup()
        # pending_boost_outcome must be None, not a crash
        zr = rt._zone("lz")
        assert zr.pending_boost_outcome is None

    @pytest.mark.asyncio
    async def test_unknown_extra_fields_in_blob_are_ignored(self):
        """Unknown fields in zone data must not cause errors (forward compatibility)."""
        store = await _trained_store()
        blob = json.loads(json.dumps(store.data))
        # Add unknown field at top level (safe — _restore_payload only reads known keys)
        blob["unknown_future_field_v99"] = "future_value"
        # Add unknown field inside each zone's data (safe — _ZoneRuntime.restore reads known keys)
        for zone_data in blob.get("zones", {}).values():
            if isinstance(zone_data, dict):
                zone_data["unknown_zone_field_v99"] = "future_value"
        rt = runtime(store=MemoryStore(data=blob))
        await rt.async_setup()
        h = rt.health()
        assert h.zones >= 1


# ── 3. Store I/O failure isolation ───────────────────────────────────────────


class TestStoreIOFailureIsolation:
    @pytest.mark.asyncio
    async def test_save_ioerror_does_not_crash(self):
        fail_store = MemoryStore(fail_save=True)
        rt = runtime(store=fail_store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        # Must not raise
        await rt.async_flush()

    @pytest.mark.asyncio
    async def test_save_ioerror_records_warning(self):
        fail_store = MemoryStore(fail_save=True)
        rt = runtime(store=fail_store)
        await rt.async_setup()
        step(rt, 0, 19.0, heating=True)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        warnings = rt._persistence.warnings
        assert any("save_failed" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_load_ioerror_produces_cold_start(self):
        fail_store = MemoryStore(fail_load=True)
        rt = runtime(store=fail_store)
        await rt.async_setup()
        result = step(rt, 0, 19.0, heating=True)
        assert result is not None
        h = rt.health()
        assert h.zones >= 1

    @pytest.mark.asyncio
    async def test_load_ioerror_records_warning(self):
        fail_store = MemoryStore(fail_load=True)
        rt = runtime(store=fail_store)
        await rt.async_setup()
        warnings = rt._persistence.warnings
        assert any("load_failed" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_runtime_remains_healthy_after_repeated_save_failures(self):
        fail_store = MemoryStore(fail_save=True)
        rt = runtime(store=fail_store)
        await rt.async_setup()
        for i in range(5):
            step(rt, i, 19.0, heating=True)
            rt.mark_dirty(important=True)
            await rt.async_flush()
        h = rt.health()
        assert h.zones >= 1


# ── 4. Schema migration stubs ────────────────────────────────────────────────


class TestSchemaMigration:
    def test_same_version_returns_state_unchanged(self):
        from custom_components.thermosmart.learning.models.boost import (
            BoostModel, MODEL_VERSION, PARAMETER_VERSION)
        m = BoostModel("lz")
        blob = m.serialize_state()
        migrated = m.migrate_state(MODEL_VERSION, PARAMETER_VERSION, blob)
        assert migrated == blob

    def test_unknown_version_raises_value_error(self):
        from custom_components.thermosmart.learning.models.boost import BoostModel
        m = BoostModel("lz")
        blob = m.serialize_state()
        with pytest.raises(ValueError):
            m.migrate_state(0, 0, blob)


# ── 5. Multi-failure scenario ────────────────────────────────────────────────


class TestMultiFailure:
    @pytest.mark.asyncio
    async def test_corrupt_blob_followed_by_save_failure_cold_starts_safely(self):
        """Corrupt load → cold-start; then save fails → still alive."""
        corrupt_store = MemoryStore(data={"garbage": "blob"}, fail_save=True)
        rt = runtime(store=corrupt_store)
        await rt.async_setup()
        result = step(rt, 0, 19.0, heating=True)
        assert result is not None
        rt.mark_dirty(important=True)
        await rt.async_flush()  # must not raise
        h = rt.health()
        assert h.zones >= 1

    @pytest.mark.asyncio
    async def test_10_alternating_success_failure_save_cycles(self):
        """Alternating save success / failure must keep learning intact."""
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        n_base = rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        last_good = json.loads(json.dumps(store.data))

        for i in range(10):
            if i % 2 == 0:
                new_store = MemoryStore(data=json.loads(json.dumps(last_good)))
            else:
                new_store = MemoryStore(data=json.loads(json.dumps(last_good)), fail_save=True)
            rt = runtime(store=new_store)
            await rt.async_setup()
            n = rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
            assert n == n_base, f"Cycle {i}: samples changed {n_base} → {n}"
            rt.mark_dirty(important=True)
            await rt.async_flush()
            if i % 2 == 0 and new_store.data is not None:
                last_good = json.loads(json.dumps(new_store.data))
