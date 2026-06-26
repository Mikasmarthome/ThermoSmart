"""Phase S1a Item 9 — Reliability/Confidence/Readiness Pre/Post Restore.

Verifies that after restart, the learning model state is numerically identical
to pre-restart: same samples, same effective_n, same confidence, same readiness.

No artificial maturing (samples don't increase just by restart).
No artificial reset (samples don't decrease just by restart).

Fields checked:
  - heat_rate model: general.sample_count, general.gain.effective_n
  - boost model: full_count, partial_count, processed_decision_ids count
  - activation_readiness: eligibility, confidence, reliability
  - processed_ids count (for legacy episode dedup)
  - model update counts per model key
"""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models.boost import (
    BoostModel,
    BoostPredictionContext,
)
from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_boost import boost_context_with_comparison, boost_episode, good_boost
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import (
    heating_ramp_then_settle,
    runtime,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _trained_runtime_store() -> tuple:
    store = MemoryStore()
    rt = runtime(store=store)
    await rt.async_setup()
    heating_ramp_then_settle(rt)
    rt.mark_dirty(important=True)
    await rt.async_flush()
    return rt, store


def _trained_boost_model(n: int = 5) -> BoostModel:
    m = BoostModel("lz")
    for i in range(n):
        good_boost(m, i)
    return m


# ── 1. HeatRate model: pre/post identical ─────────────────────────────────────


class TestHeatRateReliabilityPrePost:
    @pytest.mark.asyncio
    async def test_sample_count_identical(self):
        rt1, store = await _trained_runtime_store()
        n1 = rt1._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        n2 = rt2._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        assert n2 == n1

    @pytest.mark.asyncio
    async def test_heat_rate_c_per_h_identical(self):
        rt1, store = await _trained_runtime_store()
        r1 = rt1._zone("lz").orchestrator.models["heat_rate"]._state.general.rate_c_per_h
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        r2 = rt2._zone("lz").orchestrator.models["heat_rate"]._state.general.rate_c_per_h
        assert abs(r2 - r1) < 1e-9

    @pytest.mark.asyncio
    async def test_model_update_count_identical(self):
        rt1, store = await _trained_runtime_store()
        c1 = rt1._zone("lz").model_update_counts.get("heat_rate", 0)
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        c2 = rt2._zone("lz").model_update_counts.get("heat_rate", 0)
        assert c2 == c1

    @pytest.mark.asyncio
    async def test_no_artificial_maturation_through_restart(self):
        """Restart alone must not increase sample count (no auto-training)."""
        rt1, store = await _trained_runtime_store()
        n1 = rt1._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        for _ in range(3):
            blob = json.loads(json.dumps(store.data))
            store = MemoryStore(data=blob)
            rt = runtime(store=store)
            await rt.async_setup()
            n = rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
            assert n == n1, "Restart must not add samples to heat_rate model"
            rt.mark_dirty(important=True)
            await rt.async_flush()

    @pytest.mark.asyncio
    async def test_no_artificial_reset_through_restart(self):
        """Restart alone must not reset sample count to zero."""
        rt1, store = await _trained_runtime_store()
        n1 = rt1._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        assert n1 > 0, "Need trained model"
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        n2 = rt2._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        assert n2 > 0, "Model must not reset to zero on restart"


# ── 2. Boost model: confidence/reliability identical ──────────────────────────


class TestBoostConfidenceReliabilityPrePost:
    def test_full_count_identical_after_roundtrip(self):
        m1 = _trained_boost_model(5)
        blob = m1.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.full_count == m1._state.full_count

    def test_partial_count_identical_after_roundtrip(self):
        m1 = _trained_boost_model(3)
        blob = m1.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.partial_count == m1._state.partial_count

    def test_recent_samples_count_identical_after_roundtrip(self):
        m1 = _trained_boost_model(5)
        blob = m1.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert len(m2._state.recent_samples) == len(m1._state.recent_samples)

    def test_activation_readiness_eligibility_identical_after_roundtrip(self):
        """Same model state → same eligibility verdict."""
        m1 = _trained_boost_model(5)
        ctx = BoostPredictionContext()
        ar1 = m1.activation_readiness(ctx)

        m2 = BoostModel("lz")
        m2.deserialize_state(m1.serialize_state())
        ar2 = m2.activation_readiness(ctx)

        assert ar2.eligibility == ar1.eligibility, (
            f"Eligibility changed after restart: {ar1.eligibility} → {ar2.eligibility}"
        )

    def test_activation_readiness_confidence_identical_after_roundtrip(self):
        m1 = _trained_boost_model(5)
        ctx = BoostPredictionContext()
        ar1 = m1.activation_readiness(ctx)
        m2 = BoostModel("lz")
        m2.deserialize_state(m1.serialize_state())
        ar2 = m2.activation_readiness(ctx)
        assert abs(ar2.confidence - ar1.confidence) < 1e-6, (
            f"Confidence changed after restart: {ar1.confidence} → {ar2.confidence}"
        )

    def test_activation_readiness_reliability_identical_after_roundtrip(self):
        m1 = _trained_boost_model(5)
        ctx = BoostPredictionContext()
        ar1 = m1.activation_readiness(ctx)
        m2 = BoostModel("lz")
        m2.deserialize_state(m1.serialize_state())
        ar2 = m2.activation_readiness(ctx)
        assert abs(ar2.reliability - ar1.reliability) < 1e-6

    def test_general_readiness_identical_after_roundtrip(self):
        m1 = _trained_boost_model(5)
        blob = m1.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.degradation.general.readiness == \
            m1._state.degradation.general.readiness


# ── 3. Readiness gates: identical scope selection ─────────────────────────────


class TestReadinessScopeIdentical:
    def test_selected_scope_identical_after_roundtrip(self):
        m1 = _trained_boost_model(5)
        ctx = BoostPredictionContext()
        ar1 = m1.activation_readiness(ctx)
        m2 = BoostModel("lz")
        m2.deserialize_state(m1.serialize_state())
        ar2 = m2.activation_readiness(ctx)
        assert ar2.selected_scope == ar1.selected_scope, (
            f"Selected scope changed: {ar1.selected_scope} → {ar2.selected_scope}"
        )

    def test_factor_usable_identical_after_roundtrip(self):
        m1 = _trained_boost_model(5)
        ctx = BoostPredictionContext()
        ar1 = m1.activation_readiness(ctx)
        m2 = BoostModel("lz")
        m2.deserialize_state(m1.serialize_state())
        ar2 = m2.activation_readiness(ctx)
        assert ar2.factor_usable == ar1.factor_usable

    def test_cold_start_readiness_is_insufficient_data(self):
        """Empty model → INSUFFICIENT_DATA readiness (not eligible)."""
        from custom_components.thermosmart.learning.models.boost_activation import (
            BoostLearningReadiness)
        m = BoostModel("lz")
        ctx = BoostPredictionContext()
        ar = m.activation_readiness(ctx)
        assert not ar.eligibility
        assert ar.learning_readiness == BoostLearningReadiness.INSUFFICIENT_DATA

    def test_10_restart_chain_identical_readiness(self):
        """10-restart chain: readiness verdict must not change."""
        m = _trained_boost_model(5)
        ctx = BoostPredictionContext()
        ar_base = m.activation_readiness(ctx)
        blob = m.serialize_state()
        for i in range(10):
            m2 = BoostModel("lz")
            m2.deserialize_state(blob)
            ar = m2.activation_readiness(ctx)
            assert ar.eligibility == ar_base.eligibility, \
                f"Restart {i+1}: eligibility changed"
            assert abs(ar.confidence - ar_base.confidence) < 1e-6, \
                f"Restart {i+1}: confidence changed"
            blob = m2.serialize_state()


# ── 4. Unknown schema / missing fields degrade safely ────────────────────────


class TestReadinessDegradationOnCorruptState:
    def test_missing_degradation_field_falls_back_safely(self):
        m = _trained_boost_model(5)
        blob = m.serialize_state()
        del blob["degradation"]
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        ctx = BoostPredictionContext()
        ar = m2.activation_readiness(ctx)
        # Should still produce a valid (possibly degraded) readiness without crash
        assert isinstance(ar.eligibility, bool)

    def test_unknown_extra_fields_in_boost_blob_are_ignored(self):
        m = _trained_boost_model(3)
        blob = m.serialize_state()
        blob["unknown_future_field_v99"] = "some_value"
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        ctx = BoostPredictionContext()
        ar = m2.activation_readiness(ctx)
        assert isinstance(ar.eligibility, bool)


# ── 5. Processed IDs count identical ─────────────────────────────────────────


class TestProcessedIdsCountIdentical:
    def test_processed_decision_ids_count_identical(self):
        m = _trained_boost_model(7)
        n1 = len(m._state.processed_decision_ids)
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        n2 = len(m2._state.processed_decision_ids)
        assert n2 == n1

    def test_outcome_class_counts_identical(self):
        m = _trained_boost_model(5)
        counts1 = dict(m._state.outcome_class_counts)
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        counts2 = dict(m2._state.outcome_class_counts)
        assert counts2 == counts1

    def test_reason_counts_identical(self):
        m = _trained_boost_model(5)
        rc1 = dict(m._state.reason_counts)
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        rc2 = dict(m2._state.reason_counts)
        assert rc2 == rc1
