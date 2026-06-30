"""Phase S1a Item 9 — Boost Restart Semantics and Pending Outcome Attribution
(Sections 5, 6 of the Item 9 spec).

Tests:
  - Boost lifecycle is ephemeral (no stale offset after restart)
  - processed_decision_ids prevents double-outcome attribution
  - pending_boost_outcome survives restart (bounded post-reach observer)
  - Cooldown resets on restart (allowed deviation, documented)
  - Boost can re-fire after restart if model says so (no false blocking)
  - No double-attribution of a completed outcome across restart boundary
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.models.boost import (
    BoostLifecycle,
    BoostModel,
)
from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from custom_components.thermosmart.learning.runtime.boost_pending import (
    BoostPendingOutcome,
    PENDING_SCHEMA_VERSION,
)
from tests.helpers_boost import (
    boost_context_with_comparison,
    boost_episode,
    good_boost,
)
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import (
    heating_ramp_then_settle,
    runtime,
    step,
)

_TS0 = "2025-01-01T07:00:00+00:00"
_TS1 = "2025-01-01T08:00:00+00:00"


# ── 1. Boost lifecycle resets — no stale offset ───────────────────────────────


class TestBoostLifecycleEphemeral:
    """After a restart there must be no active boost offset left over from
    the pre-restart session.  This is the primary safety invariant:
    a missed TRV override cannot persist across HA restarts.
    """

    def test_applied_offset_zero_after_deserialize(self):
        m = BoostModel("lz")
        m.apply_lifecycle("ep-001", 2.0, base_target_c=21.0, ts=_TS0)
        assert m._state.applied_offset_c == 2.0
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.applied_offset_c == 0.0, (
            "Stale boost offset must not survive restart."
        )

    def test_lifecycle_inactive_after_deserialize(self):
        m = BoostModel("lz")
        m.apply_lifecycle("ep-001", 1.5, base_target_c=21.0, ts=_TS0)
        assert m._state.lifecycle == BoostLifecycle.APPLIED
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.lifecycle == BoostLifecycle.INACTIVE

    def test_no_stale_boost_in_cold_start(self):
        """A brand-new model (cold start) has zero applied offset."""
        m = BoostModel("lz")
        assert m._state.applied_offset_c == 0.0
        assert m._state.lifecycle == BoostLifecycle.INACTIVE

    def test_episode_binding_cleared_after_deserialize(self):
        m = BoostModel("lz")
        m.apply_lifecycle("ep-session-1", 1.5, base_target_c=21.0, ts=_TS0)
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.current_episode_id is None
        assert m2._state.lifecycle_start_ts is None
        assert m2._state.lifecycle_base_target_c is None


# ── 2. Cooldown resets — documented allowed deviation ─────────────────────────


class TestCooldownEphemeral:
    """Cooldown is a per-session anti-chatter gate.  It MUST NOT survive
    restart — a restart is a session boundary.

    Documented deviation: boost may re-fire sooner than the cooldown
    duration would suggest if HA restarted during the cooldown window.
    This is a deliberate design decision (see BoostState docstring).
    """

    def test_cooldown_resets_after_release_and_deserialize(self):
        m = BoostModel("lz")
        m.apply_lifecycle("ep-001", 1.5, base_target_c=21.0, ts=_TS0)
        m.release_lifecycle("timeout", _TS0)
        assert m._state.cooldown_until_ts is not None, "Release must set cooldown"
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert m2._state.cooldown_until_ts is None, (
            "Cooldown must reset on restart — it is ephemeral by design."
        )

    def test_lifecycle_cooldown_not_at_top_level_of_serialized_blob(self):
        """The BoostLifecycle cooldown is ephemeral.  The serialized blob top-level
        must not contain a 'cooldown_until_ts' key.  (The degradation sub-dict has
        its own cooldown tracking, which IS persisted — that's a different thing.)
        """
        m = BoostModel("lz")
        m.apply_lifecycle("ep-001", 1.5, base_target_c=21.0, ts=_TS0)
        m.release_lifecycle("timeout", _TS0)
        blob = m.serialize_state()
        assert "cooldown_until_ts" not in blob, (
            "Top-level BoostState.cooldown_until_ts must not be serialized."
        )

    def test_boost_eligible_after_restart_despite_active_cooldown(self):
        """After restart the cooldown is gone, so the model is technically eligible again.
        The model's in_cooldown check must return False on missing cooldown.
        """
        m = BoostModel("lz")
        # Seed with one good sample
        good_boost(m, 0)
        # Apply + release to set cooldown
        m.apply_lifecycle("ep-001", 1.5, base_target_c=21.0, ts=_TS0)
        m.release_lifecycle("timeout", _TS0)
        # Simulate restart
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        # cooldown_active should return False (no cooldown registered after restart)
        assert not m2.cooldown_active(_TS1)


# ── 3. processed_decision_ids prevents double-outcome ─────────────────────────


class TestDedup:
    """If a boost outcome was already attributed (processed_decision_ids),
    re-feeding the same episode after restart must NOT re-update the model.
    """

    def test_processed_decision_id_persists_and_blocks_double_update(self):
        m = BoostModel("lz")
        ep = boost_episode("ep-001", [18.0, 19.5, 21.0, 21.0, 21.0])
        ctx = boost_context_with_comparison(ep, requested_offset_c=1.5)
        m.update(ep, ctx)
        n1 = m._state.general.gain.effective_n
        decision_id = ep.decision_id
        assert decision_id in m._state.processed_decision_ids

        # Simulate restart
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        assert decision_id in m2._state.processed_decision_ids

        # Re-feed same episode — must be blocked
        m2.update(ep, ctx)
        n2 = m2._state.general.gain.effective_n
        assert n2 == n1, "Double attribution blocked by processed_decision_ids"

    def test_new_episode_after_restart_is_accepted(self):
        """A NEW decision (different decision_id) after restart must be accepted."""
        m = BoostModel("lz")
        ep1 = boost_episode("ep-001", [18.0, 19.5, 21.0, 21.0, 21.0])
        m.update(ep1, boost_context_with_comparison(ep1, requested_offset_c=1.5))
        n1 = m._state.general.gain.effective_n
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)

        ep2 = boost_episode("ep-002", [19.0, 20.0, 21.5, 21.5, 21.0])
        m2.update(ep2, boost_context_with_comparison(ep2, requested_offset_c=1.5))
        n2 = m2._state.general.gain.effective_n
        assert n2 > n1, "New episode after restart must be accepted"

    def test_processed_ids_preserved_across_10_restart_chain(self):
        """processed_decision_ids must be retained through 10 serialization roundtrips."""
        m = BoostModel("lz")
        for i in range(5):
            ep = boost_episode(f"ep-{i:04d}", [18.0, 19.5, 21.0, 21.0, 21.0],
                               decision_id=f"dec-{i:04d}")
            m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        ids_original = set(m._state.processed_decision_ids)
        assert len(ids_original) == 5

        blob = m.serialize_state()
        for _ in range(10):
            m_new = BoostModel("lz")
            m_new.deserialize_state(blob)
            blob = m_new.serialize_state()

        m_final = BoostModel("lz")
        m_final.deserialize_state(blob)
        assert set(m_final._state.processed_decision_ids) == ids_original


# ── 4. pending_boost_outcome survives restart ─────────────────────────────────


def _make_pending_outcome(decision_id: str = "ep-001") -> BoostPendingOutcome:
    """Create a minimal valid BoostPendingOutcome for testing."""
    ts = "2025-01-01T07:00:00+00:00"
    ts_end = "2025-01-01T07:30:00+00:00"
    ts_deadline = "2025-01-01T07:45:00+00:00"
    return BoostPendingOutcome(
        zone_id="lz",
        decision_id=decision_id,
        episode_id=f"ep_{decision_id}",
        state="TARGET_REACHED_PENDING_OBSERVATION",
        target=21.0,
        episode_end_ts=ts_end,
        observation_start_ts=ts_end,
        deadline_ts=ts_deadline,
        last_valid_ts=ts_end,
        episode={"episode_id": f"ep_{decision_id}", "decision_id": decision_id,
                 "start_ts": ts, "end_ts": ts_end, "target": 21.0},
        boost_context={"decision_id": decision_id, "requested_offset_c": 1.5},
    )


class TestPendingBoostOutcome:
    """The single bounded post-reach observer is serialized/restored.
    After a restart within the observation window, attribution continues.
    """

    def test_pending_outcome_roundtrip(self):
        po = _make_pending_outcome()
        blob = po.to_dict()
        po2 = BoostPendingOutcome.from_dict(blob)
        assert po2 is not None
        assert po2.decision_id == po.decision_id

    def test_pending_outcome_schema_version_present(self):
        po = _make_pending_outcome()
        blob = po.to_dict()
        assert blob.get("schema_version") == PENDING_SCHEMA_VERSION

    def test_pending_outcome_from_wrong_version_returns_none(self):
        blob = _make_pending_outcome().to_dict()
        blob["schema_version"] = 9999
        result = BoostPendingOutcome.from_dict(blob)
        assert result is None

    def test_pending_outcome_from_empty_blob_returns_none(self):
        result = BoostPendingOutcome.from_dict({})
        assert result is None

    def test_pending_outcome_from_corrupt_blob_returns_none(self):
        result = BoostPendingOutcome.from_dict(
            {"schema_version": PENDING_SCHEMA_VERSION, "garbage": True})
        assert result is None

    @pytest.mark.asyncio
    async def test_pending_outcome_in_zone_runtime_survives_restart(self):
        """When a ZoneRuntime has a pending_boost_outcome, it survives flush→restore."""
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        # Manually inject a pending outcome into the zone
        po = _make_pending_outcome("ep-001")
        zr = rt1._zone("lz")
        zr.pending_boost_outcome = po
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        restored = rt2._zone("lz").pending_boost_outcome
        assert restored is not None
        assert restored.decision_id == "ep-001"

    @pytest.mark.asyncio
    async def test_pending_outcome_episode_data_survives_restart(self):
        po = _make_pending_outcome("ep-007")
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        rt1._zone("lz").pending_boost_outcome = po
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        restored = rt2._zone("lz").pending_boost_outcome
        assert restored is not None
        assert restored.episode["decision_id"] == "ep-007"


# ── 5. Boost runtime across restart chain ─────────────────────────────────────


class TestBoostRestartChain:
    """End-to-end: boost model trains, serializes, restores, then trains more."""

    @pytest.mark.asyncio
    async def test_boost_model_persists_through_10_restart_chain(self):
        """10-restart chain — store blob stays stable, learning preserved."""
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        n_base = rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count

        for i in range(10):
            blob = json.loads(json.dumps(store.data))
            new_store = MemoryStore(data=blob)
            rt = runtime(store=new_store)
            await rt.async_setup()
            n = rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
            assert n == n_base, f"Restart {i+1}/10: samples changed ({n} != {n_base})"
            rt.mark_dirty(important=True)
            await rt.async_flush()
            store = new_store

    @pytest.mark.asyncio
    async def test_pending_boost_outcome_clear_after_none_stored(self):
        """None pending_boost_outcome restores as None — no phantom outcome."""
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        assert rt1._zone("lz").pending_boost_outcome is None
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert rt2._zone("lz").pending_boost_outcome is None

    @pytest.mark.asyncio
    async def test_boost_contexts_are_runtime_only_reset_on_restart(self):
        """pending_boost_contexts is a runtime-only stash; always empty after restart."""
        store = MemoryStore()
        rt1 = runtime(store=store)
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        assert len(rt2._zone("lz").pending_boost_contexts) == 0
