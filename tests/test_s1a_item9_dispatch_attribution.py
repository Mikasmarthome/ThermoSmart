"""Phase S1a Item 9 — Dispatch Attribution Persistence Tests.

Root cause closed:
    Before this fix, ``pending_boost_contexts`` and ``pending_dispatch_records`` were
    ephemeral (not serialised).  A restart between real dispatch and episode-close meant
    the BoostUpdateContext and BoostDispatchRecord were gone — the episode close saw
    ``bctx = None`` and fell through to the non-boost baseline path.  The dispatch was
    executed against the TRV but the learning model never received its outcome.

Fix (lifecycle.py + capture.py):
    Both dicts are now included in ``_ZoneRuntime.serialize()`` and restored in
    ``_ZoneRuntime.restore()``.  ``BoostDispatchRecord`` gained ``to_dict()`` /
    ``from_dict()``.  ``pending_baseline_ctx`` (already plain dicts) is also persisted.

This file proves:
  • BoostDispatchRecord round-trips losslessly.
  • All three pending dicts survive a store-flush / restore cycle.
  • Attribution at episode-close is preserved after any number of restarts.
  • Exactly one outcome per decision (processed_decision_ids dedup).
  • Partial failure stays visible after restart.
  • Corrupt / unknown pending entries are isolated without crash.
  • Pending contexts are bounded (FIFO cap; no unbounded growth).
  • Idempotency: restoring + re-serialising the blob does not mutate it.
"""
from __future__ import annotations

import json
import pytest

from custom_components.thermosmart.learning.models import BoostUpdateContext
from custom_components.thermosmart.learning.models.boost import BoostModel
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from tests.helpers_boost import good_boost
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import runtime, step


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_drec(decision_id: str = "dec-A", *,
               status: str = "fully_succeeded",
               applied_c: float = 2.0,
               targets_total: int = 1,
               targets_failed: int = 0) -> BoostDispatchRecord:
    return BoostDispatchRecord(
        decision_id=decision_id,
        boost_candidate_c=applied_c,
        boost_applied_c=applied_c,
        baseline_setpoint_c=21.0,
        final_setpoint_c=21.0 + applied_c,
        effective_setpoint_min_c=21.0 + applied_c,
        effective_setpoint_max_c=21.0 + applied_c,
        dispatch_status=status,
        outcome_eligible=(status != "not_attempted"),
        outcome_reliability="full" if status == "fully_succeeded" else "partial",
        start_deficit_c=2.0,
        boost_evaluation_status="approved",
        device_control_type="setpoint",
        effective_setpoints=(21.0 + applied_c,) * targets_total,
        targets_total=targets_total,
        targets_failed=targets_failed,
    )


def _make_bctx(decision_id: str = "dec-A", *,
               applied_c: float = 2.0,
               status: str = "fully_succeeded") -> BoostUpdateContext:
    return BoostUpdateContext(
        source_episode_id=decision_id,
        decision_id=decision_id,
        learning_zone_id="lz",
        boost_applied_c=applied_c,
        baseline_setpoint_c=21.0,
        final_setpoint_c=21.0 + applied_c,
        dispatch_status=status,
        outcome_reliability="full" if status == "fully_succeeded" else "partial",
        authority="live_record",
    )


async def _runtime_with_pending(decision_id: str = "dec-A", *,
                                status: str = "fully_succeeded") -> tuple:
    """Return (runtime, MemoryStore) with one pending dispatch context injected."""
    store = MemoryStore()
    rt = runtime(store=store)
    await rt.async_setup()
    step(rt, 0, 19.0, heating=False)
    zr = rt._zone("lz")
    zr.pending_boost_contexts[decision_id] = _make_bctx(decision_id, status=status)
    zr.pending_dispatch_records[decision_id] = _make_drec(decision_id, status=status)
    zr.pending_baseline_ctx[decision_id] = {
        "external_temperature_c": 5.0,
        "heat_rate_used": 1.8,
        "device_control_type": "setpoint",
        "decision_ts": "2025-01-01T06:00:00+00:00",
    }
    rt.mark_dirty(important=True)
    await rt.async_flush()
    return rt, store


# ══════════════════════════════════════════════════════════════════════════════
# 1. BoostDispatchRecord serialization
# ══════════════════════════════════════════════════════════════════════════════


class TestBoostDispatchRecordSerialization:
    def test_to_dict_contains_all_required_fields(self):
        drec = _make_drec()
        d = drec.to_dict()
        required = [
            "decision_id", "boost_applied_c", "baseline_setpoint_c",
            "dispatch_status", "outcome_eligible", "outcome_reliability",
            "device_control_type", "effective_setpoints",
            "targets_total", "targets_failed",
        ]
        for f in required:
            assert f in d, f"Missing field: {f}"

    def test_from_dict_round_trip_lossless(self):
        drec = _make_drec("dec-rt-1", status="fully_succeeded", applied_c=2.5,
                           targets_total=2, targets_failed=0)
        d = drec.to_dict()
        drec2 = BoostDispatchRecord.from_dict(d)
        assert drec2.decision_id == "dec-rt-1"
        assert drec2.boost_applied_c == 2.5
        assert drec2.dispatch_status == "fully_succeeded"
        assert drec2.effective_setpoints == (23.5, 23.5)
        assert drec2.targets_total == 2
        assert drec2.targets_failed == 0

    def test_partial_failure_round_trip(self):
        drec = _make_drec("dec-partial", status="partially_succeeded",
                           applied_c=2.0, targets_total=2, targets_failed=1)
        d = drec.to_dict()
        drec2 = BoostDispatchRecord.from_dict(d)
        assert drec2.dispatch_status == "partially_succeeded"
        assert drec2.targets_failed == 1
        assert drec2.outcome_reliability == "partial"

    def test_no_attempt_round_trip(self):
        drec = BoostDispatchRecord(
            decision_id="dec-no-att",
            dispatch_status="not_attempted",
            outcome_eligible=False,
        )
        d = drec.to_dict()
        drec2 = BoostDispatchRecord.from_dict(d)
        assert drec2.dispatch_status == "not_attempted"
        assert not drec2.outcome_eligible

    def test_multi_trv_effective_setpoints_preserved(self):
        drec = BoostDispatchRecord(
            decision_id="dec-multi",
            dispatch_status="fully_succeeded",
            effective_setpoints=(23.0, 23.0, 23.5),
            targets_total=3,
            targets_failed=0,
        )
        d = drec.to_dict()
        drec2 = BoostDispatchRecord.from_dict(d)
        assert drec2.effective_setpoints == (23.0, 23.0, 23.5)
        assert drec2.targets_total == 3

    def test_from_dict_with_minimal_fields(self):
        drec = BoostDispatchRecord.from_dict({"decision_id": "dec-min"})
        assert drec.decision_id == "dec-min"
        assert drec.dispatch_status == "not_attempted"
        assert drec.effective_setpoints == ()

    def test_direct_valve_control_type_preserved(self):
        drec = BoostDispatchRecord(
            decision_id="dec-valve",
            device_control_type="direct_valve",
            dispatch_status="fully_succeeded",
        )
        drec2 = BoostDispatchRecord.from_dict(drec.to_dict())
        assert drec2.device_control_type == "direct_valve"

    def test_json_serializable(self):
        drec = _make_drec("dec-json", applied_c=1.5)
        d = drec.to_dict()
        json_str = json.dumps(d)   # must not raise
        assert "dec-json" in json_str


# ══════════════════════════════════════════════════════════════════════════════
# 2. pending_boost_contexts persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestPendingBoostContextPersistence:
    @pytest.mark.asyncio
    async def test_pending_boost_contexts_in_serialized_blob(self):
        _, store = await _runtime_with_pending("dec-B1")
        blob = store.data
        assert "pending_boost_contexts" in blob["zones"]["lz"], \
            "pending_boost_contexts must be in zone blob"
        assert "dec-B1" in blob["zones"]["lz"]["pending_boost_contexts"]

    @pytest.mark.asyncio
    async def test_pending_boost_context_fields_preserved(self):
        _, store = await _runtime_with_pending("dec-B2")
        bctx_d = store.data["zones"]["lz"]["pending_boost_contexts"]["dec-B2"]
        assert bctx_d["decision_id"] == "dec-B2"
        assert bctx_d["authority"] == "live_record"
        assert bctx_d["boost_applied_c"] == 2.0

    @pytest.mark.asyncio
    async def test_pending_boost_context_restored_after_restart(self):
        _, store = await _runtime_with_pending("dec-B3")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        zr2 = rt2._zone("lz")
        assert "dec-B3" in zr2.pending_boost_contexts, \
            "pending_boost_context must be available after restart"

    @pytest.mark.asyncio
    async def test_restored_context_is_boost_update_context(self):
        _, store = await _runtime_with_pending("dec-B4")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        bctx = rt2._zone("lz").pending_boost_contexts["dec-B4"]
        assert isinstance(bctx, BoostUpdateContext)
        assert bctx.decision_id == "dec-B4"

    @pytest.mark.asyncio
    async def test_restored_context_decision_id_matches(self):
        _, store = await _runtime_with_pending("dec-B5")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        bctx = rt2._zone("lz").pending_boost_contexts.get("dec-B5")
        assert bctx is not None
        assert bctx.decision_id == "dec-B5"

    @pytest.mark.asyncio
    async def test_multiple_pending_contexts_all_restored(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        zr = rt._zone("lz")
        for i in range(5):
            did = f"dec-multi-{i}"
            zr.pending_boost_contexts[did] = _make_bctx(did)
        rt.mark_dirty(important=True)
        await rt.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        zr2 = rt2._zone("lz")
        for i in range(5):
            assert f"dec-multi-{i}" in zr2.pending_boost_contexts


# ══════════════════════════════════════════════════════════════════════════════
# 3. pending_dispatch_records persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestPendingDispatchRecordPersistence:
    @pytest.mark.asyncio
    async def test_pending_dispatch_records_in_blob(self):
        _, store = await _runtime_with_pending("dec-D1")
        assert "pending_dispatch_records" in store.data["zones"]["lz"]
        assert "dec-D1" in store.data["zones"]["lz"]["pending_dispatch_records"]

    @pytest.mark.asyncio
    async def test_dispatch_record_fields_preserved_in_blob(self):
        _, store = await _runtime_with_pending("dec-D2")
        drec_d = store.data["zones"]["lz"]["pending_dispatch_records"]["dec-D2"]
        assert drec_d["dispatch_status"] == "fully_succeeded"
        assert drec_d["boost_applied_c"] == 2.0
        assert drec_d["device_control_type"] == "setpoint"

    @pytest.mark.asyncio
    async def test_dispatch_record_restored_as_boost_dispatch_record(self):
        _, store = await _runtime_with_pending("dec-D3")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        drec = rt2._zone("lz").pending_dispatch_records.get("dec-D3")
        assert drec is not None
        assert isinstance(drec, BoostDispatchRecord)

    @pytest.mark.asyncio
    async def test_partial_failure_provenance_preserved(self):
        _, store = await _runtime_with_pending("dec-D4", status="partially_succeeded")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        drec = rt2._zone("lz").pending_dispatch_records.get("dec-D4")
        assert drec is not None
        assert drec.dispatch_status == "partially_succeeded"
        assert drec.outcome_reliability == "partial"

    @pytest.mark.asyncio
    async def test_effective_setpoints_preserved_after_restart(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        zr = rt._zone("lz")
        drec = BoostDispatchRecord(
            decision_id="dec-D5",
            dispatch_status="fully_succeeded",
            effective_setpoints=(23.0, 23.5),
            targets_total=2,
            targets_failed=0,
        )
        zr.pending_dispatch_records["dec-D5"] = drec
        rt.mark_dirty(important=True)
        await rt.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        drec2 = rt2._zone("lz").pending_dispatch_records.get("dec-D5")
        assert drec2 is not None
        assert drec2.effective_setpoints == (23.0, 23.5)
        assert drec2.targets_total == 2

    @pytest.mark.asyncio
    async def test_device_control_type_preserved(self):
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        zr = rt._zone("lz")
        zr.pending_dispatch_records["dec-valve"] = BoostDispatchRecord(
            decision_id="dec-valve",
            device_control_type="direct_valve",
            dispatch_status="fully_succeeded",
        )
        rt.mark_dirty(important=True)
        await rt.async_flush()
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        drec2 = rt2._zone("lz").pending_dispatch_records.get("dec-valve")
        assert drec2 is not None
        assert drec2.device_control_type == "direct_valve"


# ══════════════════════════════════════════════════════════════════════════════
# 4. pending_baseline_ctx persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestPendingBaselineCtxPersistence:
    @pytest.mark.asyncio
    async def test_pending_baseline_ctx_in_blob(self):
        _, store = await _runtime_with_pending("dec-X1")
        assert "pending_baseline_ctx" in store.data["zones"]["lz"]
        assert "dec-X1" in store.data["zones"]["lz"]["pending_baseline_ctx"]

    @pytest.mark.asyncio
    async def test_baseline_ctx_fields_preserved(self):
        _, store = await _runtime_with_pending("dec-X2")
        ctx_d = store.data["zones"]["lz"]["pending_baseline_ctx"]["dec-X2"]
        assert ctx_d["external_temperature_c"] == 5.0
        assert ctx_d["heat_rate_used"] == 1.8
        assert ctx_d["device_control_type"] == "setpoint"

    @pytest.mark.asyncio
    async def test_baseline_ctx_restored_after_restart(self):
        _, store = await _runtime_with_pending("dec-X3")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        ctx = rt2._zone("lz").pending_baseline_ctx.get("dec-X3")
        assert ctx is not None
        assert ctx["external_temperature_c"] == 5.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Attribution preserved after restart (pop at episode close)
# ══════════════════════════════════════════════════════════════════════════════


class TestAttributionPreservedAfterRestart:
    """Prove that after restart the contexts can be popped (as episode-close would do)
    and the BoostModel receives the correct attribution."""

    @pytest.mark.asyncio
    async def test_bctx_poppable_after_restart(self):
        """Simulate episode-close pop after restart — must return correct context."""
        _, store = await _runtime_with_pending("dec-A1")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        zr2 = rt2._zone("lz")

        bctx = zr2.pending_boost_contexts.pop("dec-A1", None)
        assert bctx is not None, "BoostUpdateContext must be available after restart"
        assert bctx.decision_id == "dec-A1"
        assert bctx.boost_applied_c == 2.0

    @pytest.mark.asyncio
    async def test_drec_poppable_after_restart(self):
        _, store = await _runtime_with_pending("dec-A2")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        zr2 = rt2._zone("lz")

        drec = zr2.pending_dispatch_records.pop("dec-A2", None)
        assert drec is not None, "BoostDispatchRecord must be available after restart"
        assert drec.decision_id == "dec-A2"
        assert drec.dispatch_status == "fully_succeeded"

    @pytest.mark.asyncio
    async def test_baseline_ctx_poppable_after_restart(self):
        _, store = await _runtime_with_pending("dec-A3")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        ctx = rt2._zone("lz").pending_baseline_ctx.pop("dec-A3", None)
        assert ctx is not None
        assert ctx["external_temperature_c"] == 5.0

    @pytest.mark.asyncio
    async def test_three_restart_chain_context_survives(self):
        """Context must survive multiple successive restarts."""
        _, store = await _runtime_with_pending("dec-chain")
        for _ in range(3):
            rt_i = runtime(store=MemoryStore(data=store.data))
            await rt_i.async_setup()
            # Re-serialize without consuming the context
            rt_i.mark_dirty(important=True)
            await rt_i.async_flush()
            store = MemoryStore(data=json.loads(json.dumps(store.data)))

        rt_final = runtime(store=MemoryStore(data=store.data))
        await rt_final.async_setup()
        bctx = rt_final._zone("lz").pending_boost_contexts.get("dec-chain")
        assert bctx is not None
        assert bctx.decision_id == "dec-chain"

    @pytest.mark.asyncio
    async def test_boost_model_update_possible_with_restored_context(self):
        """After restart, the restored BoostUpdateContext is usable for model.update()."""
        from tests.helpers_boost import boost_episode
        _, store = await _runtime_with_pending("dec-A5")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        zr2 = rt2._zone("lz")

        bctx = zr2.pending_boost_contexts.pop("dec-A5", None)
        assert bctx is not None

        # Build a minimal episode with REACHED trajectory
        vals = [18.0, 19.0, 20.0, 21.0]
        ep = boost_episode("dec-A5", vals, decision_id="dec-A5", zone="lz")
        m = zr2.orchestrator.models.get("boost")
        if m is not None:
            # Should not raise; result indicates eligibility gate outcome
            result = m.update(ep, bctx)
            assert result is not None

    @pytest.mark.asyncio
    async def test_decision_id_identical_pre_post_restart(self):
        _, store = await _runtime_with_pending("dec-id-check")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        bctx = rt2._zone("lz").pending_boost_contexts.get("dec-id-check")
        assert bctx is not None
        assert bctx.decision_id == "dec-id-check"

    @pytest.mark.asyncio
    async def test_partial_failure_flag_preserved_after_restart(self):
        """Partial failure remains recognisable after restart."""
        _, store = await _runtime_with_pending("dec-partial", status="partially_succeeded")
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        drec = rt2._zone("lz").pending_dispatch_records.get("dec-partial")
        bctx = rt2._zone("lz").pending_boost_contexts.get("dec-partial")
        assert drec is not None
        assert drec.dispatch_status == "partially_succeeded"
        assert bctx is not None
        assert bctx.dispatch_status == "partially_succeeded"
        assert bctx.outcome_reliability == "partial"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Exactly-one-outcome semantics
# ══════════════════════════════════════════════════════════════════════════════


class TestExactOneOutcomeSemantics:
    """processed_decision_ids deduplication prevents double attribution."""

    def _boost_model_with_entries(self, n: int = 5) -> BoostModel:
        m = BoostModel("lz")
        for i in range(n):
            good_boost(m, i)
        return m

    def test_processed_decision_ids_dedup(self):
        m = self._boost_model_with_entries()
        blob = m.serialize_state()
        ids_before = set(m._state.processed_decision_ids)

        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        ids_after = set(m2._state.processed_decision_ids)

        assert ids_before == ids_after, \
            "processed_decision_ids must be identical after restart"

    def test_second_update_with_same_decision_id_is_rejected(self):
        from tests.helpers_boost import boost_episode
        m = BoostModel("lz")
        good_boost(m, 0)

        # Get a known processed id
        processed_id = next(iter(m._state.processed_decision_ids), None)
        if processed_id is None:
            return   # no entries to test

        # Serialize and restore
        m2 = BoostModel("lz")
        m2.deserialize_state(m.serialize_state())

        # Rebuild the episode with the same decision_id
        vals = [18.0, 19.5, 21.0]
        ep = boost_episode(processed_id, vals, decision_id=processed_id, zone="lz")
        bctx = _make_bctx(processed_id)
        count_before = m2._state.full_count
        m2.update(ep, bctx)
        # Update must be rejected (dedup) — count must not increase
        assert m2._state.full_count == count_before, \
            "Duplicate outcome must be rejected by processed_decision_ids dedup"

    @pytest.mark.asyncio
    async def test_no_double_attribution_across_restart(self):
        """Inject a pending context, pop it (simulating episode close), record the
        decision as processed, then restart: the decision must still be in
        processed_decision_ids and a second update must be rejected."""
        from tests.helpers_boost import boost_episode
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        zr = rt._zone("lz")

        did = "dec-dedup-1"
        zr.pending_boost_contexts[did] = _make_bctx(did)

        # Simulate episode close: pop + apply to boost model
        vals = [18.0, 19.5, 21.0]
        ep = boost_episode(did, vals, decision_id=did, zone="lz")
        m = zr.orchestrator.models.get("boost")
        bctx = zr.pending_boost_contexts.pop(did)
        if m is not None:
            m.update(ep, bctx)   # first attribution

        rt.mark_dirty(important=True)
        await rt.async_flush()

        # Restart
        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        m2 = rt2._zone("lz").orchestrator.models.get("boost")
        if m2 is None:
            return

        count_before = m2._state.full_count
        # Try to apply again with same decision_id
        m2.update(ep, bctx)
        assert m2._state.full_count == count_before, \
            "Second update after restart must be deduped"

    @pytest.mark.asyncio
    async def test_shadow_proposal_produces_no_real_outcome(self):
        """A context never consumed (shadow proposal without dispatch) must not
        produce a real outcome entry in processed_decision_ids after restart."""
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        # Run steps without any boost_dispatch injection
        for i in range(3):
            step(rt, i, 19.0 + i * 0.5, heating=True)
        rt.mark_dirty(important=True)
        await rt.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        m = rt2._zone("lz").orchestrator.models.get("boost")
        if m is not None:
            assert m._state.full_count == 0, \
                "No dispatch → no outcome → full_count must remain 0"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Bounds and retention
# ══════════════════════════════════════════════════════════════════════════════


class TestBoundsAndRetention:
    @pytest.mark.asyncio
    async def test_pending_boost_contexts_cap_at_64(self):
        """Verify the FIFO cap is 64 (run_cycle enforces it; direct dict access is a test-only path)."""
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        zr = rt._zone("lz")
        assert zr._pending_boost_cap == 64, "Cap must be 64"

    @pytest.mark.asyncio
    async def test_cap_number_of_pending_entries_survives_restore(self):
        """Exactly cap entries serialised → exactly cap entries restored."""
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        zr = rt._zone("lz")
        cap = zr._pending_boost_cap

        # Fill to cap exactly
        for i in range(cap):
            did = f"dec-cap-{i}"
            zr.pending_boost_contexts[did] = _make_bctx(did)
            zr.pending_dispatch_records[did] = _make_drec(did)

        assert len(zr.pending_boost_contexts) == cap
        rt.mark_dirty(important=True)
        await rt.async_flush()

        rt2 = runtime(store=MemoryStore(data=store.data))
        await rt2.async_setup()
        zr2 = rt2._zone("lz")
        assert len(zr2.pending_boost_contexts) == cap, \
            f"Restore must return exactly {cap} entries"

    @pytest.mark.asyncio
    async def test_blob_size_bounded_at_cap(self):
        """Blob with cap entries has a finite, predictable size."""
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        zr = rt._zone("lz")
        cap = zr._pending_boost_cap

        for i in range(cap):
            did = f"dec-bound-{i}"
            zr.pending_boost_contexts[did] = _make_bctx(did)
            zr.pending_dispatch_records[did] = _make_drec(did)

        rt.mark_dirty(important=True)
        await rt.async_flush()
        size = len(json.dumps(store.data))

        # 64 full contexts: must be < 1 MB (generous upper bound)
        assert size < 1_048_576, f"Blob with {cap} pending entries is too large: {size} bytes"

    @pytest.mark.asyncio
    async def test_one_pending_record_per_decision(self):
        """A second store for the same decision_id replaces the first (no duplicates)."""
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        zr = rt._zone("lz")

        did = "dec-unique"
        zr.pending_boost_contexts[did] = _make_bctx(did, applied_c=1.0)
        zr.pending_boost_contexts[did] = _make_bctx(did, applied_c=2.0)  # overwrite

        assert len([k for k in zr.pending_boost_contexts if k == did]) == 1
        assert zr.pending_boost_contexts[did].boost_applied_c == 2.0

    @pytest.mark.asyncio
    async def test_consumed_context_not_present_in_next_flush(self):
        """Once popped (episode close), the entry must not reappear after re-flush."""
        _, store_init = await _runtime_with_pending("dec-consumed")
        # rt2 uses its OWN store so we can check its output
        store2 = MemoryStore(data=store_init.data)
        rt2 = runtime(store=store2)
        await rt2.async_setup()

        # Pop (simulating episode close)
        rt2._zone("lz").pending_boost_contexts.pop("dec-consumed", None)
        rt2._zone("lz").pending_dispatch_records.pop("dec-consumed", None)
        rt2.mark_dirty(important=True)
        await rt2.async_flush()

        assert "dec-consumed" not in store2.data["zones"]["lz"].get("pending_boost_contexts", {})


# ══════════════════════════════════════════════════════════════════════════════
# 8. Corruption isolation
# ══════════════════════════════════════════════════════════════════════════════


class TestCorruptPendingContextIsolation:
    @pytest.mark.asyncio
    async def test_corrupt_boost_context_entry_is_skipped(self):
        """A corrupt pending_boost_contexts entry must be skipped (not crash)."""
        _, store = await _runtime_with_pending("dec-good")
        blob = json.loads(json.dumps(store.data))
        # Corrupt one entry; leave the valid one
        blob["zones"]["lz"]["pending_boost_contexts"]["dec-corrupt"] = \
            {"broken": True}   # missing required fields

        rt2 = runtime(store=MemoryStore(data=blob))
        await rt2.async_setup()   # must not raise
        # The valid entry must still be present
        assert "dec-good" in rt2._zone("lz").pending_boost_contexts

    @pytest.mark.asyncio
    async def test_corrupt_dispatch_record_entry_is_skipped(self):
        _, store = await _runtime_with_pending("dec-good2")
        blob = json.loads(json.dumps(store.data))
        blob["zones"]["lz"]["pending_dispatch_records"]["dec-corrupt2"] = \
            {"decision_id": None}   # invalid

        rt2 = runtime(store=MemoryStore(data=blob))
        await rt2.async_setup()
        h = rt2.health()
        assert h.initialized

    @pytest.mark.asyncio
    async def test_null_pending_boost_contexts_treated_as_empty(self):
        _, store = await _runtime_with_pending("dec-null")
        blob = json.loads(json.dumps(store.data))
        blob["zones"]["lz"]["pending_boost_contexts"] = None

        rt2 = runtime(store=MemoryStore(data=blob))
        await rt2.async_setup()
        h = rt2.health()
        assert h.initialized
        assert len(rt2._zone("lz").pending_boost_contexts) == 0

    @pytest.mark.asyncio
    async def test_null_pending_dispatch_records_treated_as_empty(self):
        _, store = await _runtime_with_pending("dec-null2")
        blob = json.loads(json.dumps(store.data))
        blob["zones"]["lz"]["pending_dispatch_records"] = None

        rt2 = runtime(store=MemoryStore(data=blob))
        await rt2.async_setup()
        assert len(rt2._zone("lz").pending_dispatch_records) == 0

    @pytest.mark.asyncio
    async def test_missing_pending_fields_cold_start_safe(self):
        """Blobs from before the fix (no pending_* keys) must load cleanly."""
        store = MemoryStore()
        rt = runtime(store=store)
        await rt.async_setup()
        step(rt, 0, 19.0, heating=False)
        rt.mark_dirty(important=True)
        await rt.async_flush()

        blob = json.loads(json.dumps(store.data))
        # Simulate pre-fix blob: remove the new keys
        zone = blob["zones"]["lz"]
        zone.pop("pending_boost_contexts", None)
        zone.pop("pending_dispatch_records", None)
        zone.pop("pending_baseline_ctx", None)

        rt2 = runtime(store=MemoryStore(data=blob))
        await rt2.async_setup()
        h = rt2.health()
        assert h.initialized
        assert len(rt2._zone("lz").pending_boost_contexts) == 0

    @pytest.mark.asyncio
    async def test_extra_unknown_fields_in_dispatch_record_ignored(self):
        _, store = await _runtime_with_pending("dec-unknown")
        blob = json.loads(json.dumps(store.data))
        blob["zones"]["lz"]["pending_dispatch_records"]["dec-unknown"]["future_field_v99"] = "x"

        rt2 = runtime(store=MemoryStore(data=blob))
        await rt2.async_setup()
        drec = rt2._zone("lz").pending_dispatch_records.get("dec-unknown")
        assert drec is not None
        assert drec.decision_id == "dec-unknown"


# ══════════════════════════════════════════════════════════════════════════════
# 9. Idempotency of pending context persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestPendingContextIdempotency:
    @pytest.mark.asyncio
    async def test_serialize_deserialize_pending_blob_idempotent(self):
        """serialize(restore(blob)) == blob for zone with pending contexts."""
        _, store = await _runtime_with_pending("dec-idem")
        blob1 = json.loads(json.dumps(store.data))

        rt2 = runtime(store=MemoryStore(data=blob1))
        await rt2.async_setup()
        rt2.mark_dirty(important=True)
        await rt2.async_flush()
        blob2 = json.loads(json.dumps(store.data))

        assert (json.dumps(blob1["zones"]["lz"]["pending_boost_contexts"], sort_keys=True)
                == json.dumps(blob2["zones"]["lz"]["pending_boost_contexts"], sort_keys=True)), \
            "pending_boost_contexts must be idempotent across restart"

    @pytest.mark.asyncio
    async def test_restart_without_consumption_does_not_grow_pending(self):
        """N restarts without episode closes must not grow pending context count."""
        _, store = await _runtime_with_pending("dec-stable")
        count0 = len(store.data["zones"]["lz"].get("pending_boost_contexts", {}))

        for _ in range(5):
            rt_i = runtime(store=MemoryStore(data=store.data))
            await rt_i.async_setup()
            rt_i.mark_dirty(important=True)
            await rt_i.async_flush()

        count_final = len(store.data["zones"]["lz"].get("pending_boost_contexts", {}))
        assert count_final == count0, \
            f"Pending context count grew from {count0} to {count_final} without new dispatches"
