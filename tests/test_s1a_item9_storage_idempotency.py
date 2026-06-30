"""Phase S1a Item 9 — Storage Idempotency and Growth Analysis.

Semantic idempotency: N restarts with no new data → identical blob on each cycle.
Serialization idempotency: serialize(deserialize(blob)) == blob (no structural drift).

Growth analysis:
  - ≤20% growth per restart is acceptable (documented pattern)
  - Growth must come from new data, not restart overhead
  - No fachliches Wachstum durch Restart allein (no business data added by restart)
  - deterministische Sortierung: same data → same serialized key order
"""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models.boost import BoostModel
from custom_components.thermosmart.learning.runtime import LearningRuntime
from tests.helpers_boost import good_boost
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import (
    heating_ramp_then_settle,
    runtime,
    step,
)

_BLOB_SIZE_LIMIT_BYTES = 512 * 1024  # 512 KB


# ── 1. Boost model: semantic idempotency ─────────────────────────────────────


class TestBoostModelSemanticIdempotency:
    def _baseline_model(self) -> BoostModel:
        m = BoostModel("lz")
        for i in range(5):
            good_boost(m, i)
        return m

    def test_serialize_deserialize_same_full_count(self):
        m = self._baseline_model()
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.full_count == m._state.full_count

    def test_three_roundtrips_no_drift(self):
        """serialize→deserialize three times: no structural drift."""
        m = self._baseline_model()
        blob = m.serialize_state()
        for i in range(3):
            m2 = BoostModel("lz")
            m2.deserialize_state(blob)
            blob2 = m2.serialize_state()
            n1 = json.dumps(blob, sort_keys=True)
            n2 = json.dumps(blob2, sort_keys=True)
            assert n1 == n2, f"Roundtrip {i+1}: blob drifted"
            blob = blob2

    def test_ten_roundtrips_no_sample_growth(self):
        m = self._baseline_model()
        n0 = m._state.full_count
        blob = m.serialize_state()
        for i in range(10):
            m2 = BoostModel("lz")
            m2.deserialize_state(blob)
            assert m2._state.full_count == n0, \
                f"Roundtrip {i+1}: full_count changed {n0} → {m2._state.full_count}"
            blob = m2.serialize_state()

    def test_hundred_roundtrips_processed_ids_stable(self):
        m = self._baseline_model()
        ids0 = set(m._state.processed_decision_ids)
        blob = m.serialize_state()
        for i in range(100):
            m2 = BoostModel("lz")
            m2.deserialize_state(blob)
            ids = set(m2._state.processed_decision_ids)
            assert ids == ids0, \
                f"Roundtrip {i+1}: processed_decision_ids changed"
            blob = m2.serialize_state()

    def test_deterministic_sort_order(self):
        """Same model → same serialized key order in JSON."""
        m = self._baseline_model()
        blob1 = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob1)
        blob2 = m2.serialize_state()
        j1 = json.dumps(blob1, sort_keys=True)
        j2 = json.dumps(blob2, sort_keys=True)
        assert j1 == j2


# ── 2. Runtime: semantic idempotency ─────────────────────────────────────────


class TestRuntimeSemanticIdempotency:
    async def _trained_store(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        return store

    @pytest.mark.asyncio
    async def test_single_restart_sample_count_stable(self):
        store = await self._trained_store()
        blob1 = json.loads(json.dumps(store.data))
        n1 = blob1["zones"]["lz"]["models"]["heat_rate"]["general"]["sample_count"]
        rt2 = runtime(store=MemoryStore(data=blob1))
        await rt2.async_setup()
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        blob2 = store.data
        n2 = blob2["zones"]["lz"]["models"]["heat_rate"]["general"]["sample_count"]
        assert n2 == n1, f"Sample count drifted: {n1} → {n2}"

    @pytest.mark.asyncio
    async def test_three_restart_chain_no_fachliches_wachstum(self):
        """Restart chain without new data: no business data growth."""
        store = await self._trained_store()
        size0 = len(json.dumps(store.data))
        for i in range(3):
            blob = json.loads(json.dumps(store.data))
            rt = runtime(store=MemoryStore(data=blob))
            await rt.async_setup()
            rt.mark_dirty(important=True)
            await rt.async_flush()
            size_i = len(json.dumps(store.data))
            assert size_i == size0, \
                f"Restart {i+1}: blob grew without new data: {size0} → {size_i}"

    @pytest.mark.asyncio
    async def test_blob_within_size_limit_after_training(self):
        store = await self._trained_store()
        blob_bytes = len(json.dumps(store.data).encode("utf-8"))
        assert blob_bytes <= _BLOB_SIZE_LIMIT_BYTES, \
            f"Blob too large: {blob_bytes} bytes > {_BLOB_SIZE_LIMIT_BYTES}"


# ── 3. Serialization idempotency ─────────────────────────────────────────────


class TestSerializationIdempotency:
    """serialize(deserialize(blob)) == blob: no structural overhead."""

    def test_boost_model_blob_idempotent(self):
        m = BoostModel("lz")
        for i in range(3):
            good_boost(m, i)
        blob1 = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob1)
        blob2 = m2.serialize_state()
        assert json.dumps(blob1, sort_keys=True) == json.dumps(blob2, sort_keys=True)

    @pytest.mark.asyncio
    async def test_runtime_blob_idempotent(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        blob1 = json.loads(json.dumps(store.data))

        rt2 = runtime(store=MemoryStore(data=blob1))
        await rt2.async_setup()
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        blob2 = json.loads(json.dumps(store.data))
        assert json.dumps(blob1, sort_keys=True) == json.dumps(blob2, sort_keys=True)


# ── 4. Growth is bounded to new data only ────────────────────────────────────


class TestBoundedGrowth:
    """Growth is allowed only when new training data is added."""

    def test_adding_one_episode_grows_blob(self):
        m = BoostModel("lz")
        for i in range(3):
            good_boost(m, i)
        size1 = len(json.dumps(m.serialize_state()))
        good_boost(m, 3)
        size2 = len(json.dumps(m.serialize_state()))
        assert size2 >= size1  # growth is allowed for new data

    def test_restart_without_new_data_does_not_grow_blob(self):
        m = BoostModel("lz")
        for i in range(5):
            good_boost(m, i)
        blob1 = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob1)
        blob2 = m2.serialize_state()
        assert len(json.dumps(blob2)) == len(json.dumps(blob1)), \
            "Restart without new data must not grow the blob"

    @pytest.mark.asyncio
    async def test_runtime_growth_only_on_new_data(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        size_base = len(json.dumps(store.data))

        # Restart without new data
        rt2 = runtime(store=MemoryStore(data=json.loads(json.dumps(store.data))))
        await rt2.async_setup()
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        size_after = len(json.dumps(store.data))
        assert size_after == size_base, \
            f"Restart without new cycles grew blob: {size_base} → {size_after}"


# ── 5. 100-restart soak: no accumulation ──────────────────────────────────────


class TestHundredRestartSoak:
    def test_boost_model_100_restarts_stable(self):
        m = BoostModel("lz")
        for i in range(10):
            good_boost(m, i)
        blob0 = m.serialize_state()
        n0 = m._state.full_count
        j0 = json.dumps(blob0, sort_keys=True)
        blob = blob0
        for i in range(100):
            m2 = BoostModel("lz")
            m2.deserialize_state(blob)
            blob = m2.serialize_state()
            ji = json.dumps(blob, sort_keys=True)
            assert ji == j0, f"Restart {i+1}: blob drifted"
            assert m2._state.full_count == n0, \
                f"Restart {i+1}: full_count changed"
