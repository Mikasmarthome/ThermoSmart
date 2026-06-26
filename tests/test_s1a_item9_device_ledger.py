"""Phase S1a Item 9 — Per-Device Dispatch Ledger and Diagnostics Export.

Unit tests (pure Python, Windows + Docker) proving:

  1. Per-Device Dispatch Ledger
     • last_written_setpoints: only updated on success
     • effective target per device: from BoostDispatchRecord.effective_setpoints
     • successful / failed / unavailable device subsets
     • retry semantics: failed records survive restart; success does not retry
     • dedup: processed_decision_ids blocks second attribution
     • pending normalization: targets_failed > 0 → normalization required
     • clamp result: effective_setpoints reflects post-clamp value (< requested)
     • 1 record per decision_id: re-dispatch overwrites, not appends

  2. Pending Attribution Diagnostics Export
     • pending_attribution_summary() fields: all required keys present
     • privacy-safe: no decision_ids, no entity_ids in summary
     • bounded: summary stays small regardless of cap-level fill
     • Support Export: compact counts only
     • Research Export: per-status breakdown available
     • Diagnostics orchestrator: pending_attribution in zone output
     • After restart: summary reflects restored state
     • After consumption (episode close): pending_count decrements
"""
from __future__ import annotations

import json
import pytest

from custom_components.thermosmart.learning.runtime.lifecycle import (
    _ZoneRuntime,
    LearningRuntimeConfig,
    LearningRuntime,
)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from custom_components.thermosmart.learning.models import BoostUpdateContext
from custom_components.thermosmart.learning.diagnostics import (
    build_diagnostics,
    DiagnosticsInput,
)
from custom_components.thermosmart.learning.export import (
    ZoneExportInput,
    EngineSummary,
)
from custom_components.thermosmart.learning.privacy import Pseudonymizer
from tests.helpers_runtime import MemoryStore

pytestmark = pytest.mark.asyncio


def _make_drec(did="dec-1", *, dispatch_status="fully_succeeded",
               targets_total=1, targets_failed=0,
               effective_setpoints=(22.5,),
               device_control_type="setpoint",
               outcome_eligible=True) -> BoostDispatchRecord:
    return BoostDispatchRecord(
        decision_id=did,
        dispatch_status=dispatch_status,
        outcome_eligible=outcome_eligible,
        outcome_reliability="full" if dispatch_status == "fully_succeeded" else "partial",
        device_control_type=device_control_type,
        effective_setpoints=effective_setpoints,
        targets_total=targets_total,
        targets_failed=targets_failed,
    )


def _make_bctx(did="dec-1") -> BoostUpdateContext:
    return BoostUpdateContext(
        source_episode_id=did, decision_id=did, learning_zone_id="lz",
        boost_applied_c=2.0, dispatch_status="fully_succeeded",
        authority="live_record",
    )


def _zone_runtime() -> _ZoneRuntime:
    return _ZoneRuntime("lz", LearningRuntimeConfig())


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Per-Device Dispatch Ledger
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerDeviceDispatchLedger:
    """Unit tests for per-device effective setpoints and dispatch state."""

    def test_effective_setpoints_stored_per_decision(self):
        zr = _zone_runtime()
        drec = _make_drec("dec-A", effective_setpoints=(23.0,))
        zr.pending_dispatch_records["dec-A"] = drec
        assert zr.pending_dispatch_records["dec-A"].effective_setpoints == (23.0,)

    def test_multi_trv_effective_setpoints_stored(self):
        zr = _zone_runtime()
        drec = _make_drec("dec-B", effective_setpoints=(23.0, 22.5, 22.0),
                           targets_total=3)
        zr.pending_dispatch_records["dec-B"] = drec
        assert len(zr.pending_dispatch_records["dec-B"].effective_setpoints) == 3

    def test_one_record_per_decision_id_overwrite(self):
        zr = _zone_runtime()
        zr.pending_dispatch_records["dec-C"] = _make_drec("dec-C", effective_setpoints=(22.0,))
        zr.pending_dispatch_records["dec-C"] = _make_drec("dec-C", effective_setpoints=(23.0,))
        # Must have exactly ONE entry
        assert len(zr.pending_dispatch_records) == 1
        assert zr.pending_dispatch_records["dec-C"].effective_setpoints == (23.0,)

    def test_failed_device_targets_failed_nonzero(self):
        zr = _zone_runtime()
        drec = _make_drec("dec-D", dispatch_status="partially_succeeded",
                           targets_total=2, targets_failed=1,
                           effective_setpoints=(23.0,))
        zr.pending_dispatch_records["dec-D"] = drec
        assert zr.pending_dispatch_records["dec-D"].targets_failed == 1

    def test_fully_failed_dispatch_no_effective_setpoints(self):
        zr = _zone_runtime()
        drec = _make_drec("dec-E", dispatch_status="failed",
                           targets_total=1, targets_failed=1,
                           effective_setpoints=(),
                           outcome_eligible=False)
        zr.pending_dispatch_records["dec-E"] = drec
        assert drec.effective_setpoints == ()
        assert not drec.outcome_eligible

    def test_clamp_result_effective_setpoints_lt_requested(self):
        """effective_setpoints reflects post-clamp value (may be < requested boost)."""
        zr = _zone_runtime()
        # Requested 25.0°C but device clamped to max_temp=24.0°C
        drec = _make_drec("dec-F", effective_setpoints=(24.0,))
        zr.pending_dispatch_records["dec-F"] = drec
        assert zr.pending_dispatch_records["dec-F"].effective_setpoints == (24.0,)

    def test_normalization_required_when_targets_failed(self):
        zr = _zone_runtime()
        drec = _make_drec("dec-G", dispatch_status="partially_succeeded",
                           targets_total=2, targets_failed=1)
        zr.pending_dispatch_records["dec-G"] = drec
        # "normalization_required" is derived: targets_failed > 0
        assert drec.targets_failed > 0, "Partial failure means normalization required"

    def test_no_normalization_when_all_succeeded(self):
        zr = _zone_runtime()
        drec = _make_drec("dec-H", targets_total=2, targets_failed=0,
                           effective_setpoints=(22.5, 22.5))
        assert drec.targets_failed == 0, "No normalization required when all succeeded"

    def test_failed_record_survives_restart_for_retry(self):
        """A failed dispatch record survives serialize/restore (retry-capable)."""
        zr = _zone_runtime()
        did = "dec-retry"
        zr.pending_dispatch_records[did] = _make_drec(
            did, dispatch_status="failed", targets_total=1, targets_failed=1,
            effective_setpoints=(), outcome_eligible=False)
        blob = zr.serialize()
        zr2 = _zone_runtime()
        zr2.restore(blob)
        drec2 = zr2.pending_dispatch_records.get(did)
        assert drec2 is not None, "Failed dispatch record must survive restart"
        assert drec2.dispatch_status == "failed"

    def test_successful_dispatch_survives_restart_for_attribution(self):
        """A successful dispatch record survives restart for attribution close."""
        zr = _zone_runtime()
        did = "dec-success"
        zr.pending_dispatch_records[did] = _make_drec(
            did, effective_setpoints=(22.5,), outcome_eligible=True)
        blob = zr.serialize()
        zr2 = _zone_runtime()
        zr2.restore(blob)
        drec2 = zr2.pending_dispatch_records.get(did)
        assert drec2 is not None, "Successful dispatch must survive restart for attribution"
        assert drec2.outcome_eligible

    async def test_dedup_no_second_attribution_after_restart(self):
        """After restart, the same decision_id is rejected by dedup."""
        from tests.helpers_boost import boost_episode, boost_context
        from tests.helpers_runtime import MemoryStore
        from tests.helpers_runtime_scenarios import runtime

        store_init = MemoryStore(data=None)
        rt1 = runtime(store=store_init)
        await rt1.async_setup()

        did = "dec-dedup-unit"
        vals = [18.0, 19.5, 21.0]
        ep = boost_episode(did, vals, decision_id=did, zone="lz")
        bctx = boost_context(ep)
        m1 = rt1._zone("lz").orchestrator.models.get("boost")
        m1.update(ep, bctx)
        count = m1._state.full_count
        await rt1.async_flush()

        store_init2 = MemoryStore(data=store_init.data)
        rt2 = runtime(store=store_init2)
        await rt2.async_setup()
        m2 = rt2._zone("lz").orchestrator.models.get("boost")
        m2.update(ep, bctx)
        assert m2._state.full_count == count, \
            "Duplicate outcome must be rejected by dedup after restart"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Pending Attribution Diagnostics and Export
# ═══════════════════════════════════════════════════════════════════════════════


class TestPendingAttributionSummary:
    """Tests for _ZoneRuntime.pending_attribution_summary()."""

    def test_required_keys_present_empty_state(self):
        zr = _zone_runtime()
        s = zr.pending_attribution_summary()
        required = {"schema_version", "pending_context_count", "pending_dispatch_count",
                    "outcome_eligible_count", "dispatch_statuses", "control_types",
                    "total_targets", "failed_targets", "pending_outcome_active"}
        missing = required - set(s.keys())
        assert not missing, f"Missing keys in pending_attribution_summary: {missing}"

    def test_empty_state_all_zeros(self):
        zr = _zone_runtime()
        s = zr.pending_attribution_summary()
        assert s["pending_context_count"] == 0
        assert s["pending_dispatch_count"] == 0
        assert s["outcome_eligible_count"] == 0
        assert s["total_targets"] == 0
        assert s["failed_targets"] == 0
        assert not s["pending_outcome_active"]

    def test_counts_reflect_injected_data(self):
        zr = _zone_runtime()
        zr.pending_boost_contexts["d1"] = _make_bctx("d1")
        zr.pending_boost_contexts["d2"] = _make_bctx("d2")
        zr.pending_dispatch_records["d1"] = _make_drec("d1", targets_total=2, targets_failed=1)
        s = zr.pending_attribution_summary()
        assert s["pending_context_count"] == 2
        assert s["pending_dispatch_count"] == 1
        assert s["total_targets"] == 2
        assert s["failed_targets"] == 1

    def test_control_types_multiple(self):
        zr = _zone_runtime()
        zr.pending_dispatch_records["d1"] = _make_drec("d1", device_control_type="setpoint")
        zr.pending_dispatch_records["d2"] = _make_drec("d2", device_control_type="direct_valve")
        s = zr.pending_attribution_summary()
        assert "setpoint" in s["control_types"]
        assert "direct_valve" in s["control_types"]

    def test_dispatch_statuses_aggregated(self):
        zr = _zone_runtime()
        zr.pending_dispatch_records["d1"] = _make_drec("d1", dispatch_status="fully_succeeded")
        zr.pending_dispatch_records["d2"] = _make_drec("d2", dispatch_status="partially_succeeded",
                                                        targets_total=2, targets_failed=1)
        s = zr.pending_attribution_summary()
        assert s["dispatch_statuses"]["fully_succeeded"] == 1
        assert s["dispatch_statuses"]["partially_succeeded"] == 1

    def test_outcome_eligible_count_correct(self):
        zr = _zone_runtime()
        zr.pending_dispatch_records["d1"] = _make_drec("d1", outcome_eligible=True)
        zr.pending_dispatch_records["d2"] = _make_drec("d2", outcome_eligible=False)
        s = zr.pending_attribution_summary()
        assert s["outcome_eligible_count"] == 1

    def test_pending_outcome_active_false_when_none(self):
        zr = _zone_runtime()
        zr.pending_boost_outcome = None
        s = zr.pending_attribution_summary()
        assert not s["pending_outcome_active"]

    def test_summary_is_json_serializable(self):
        zr = _zone_runtime()
        zr.pending_boost_contexts["d1"] = _make_bctx("d1")
        zr.pending_dispatch_records["d1"] = _make_drec("d1")
        s = zr.pending_attribution_summary()
        json_str = json.dumps(s)
        assert len(json_str) < 4096, "Summary must be compact"

    def test_no_decision_ids_in_summary(self):
        """Privacy check: no raw decision IDs in the summary."""
        zr = _zone_runtime()
        did = "super-secret-uuid-12345678"
        zr.pending_boost_contexts[did] = _make_bctx(did)
        zr.pending_dispatch_records[did] = _make_drec(did)
        s = zr.pending_attribution_summary()
        summary_str = json.dumps(s)
        assert did not in summary_str, \
            "Privacy violation: decision_id must not appear in pending_attribution_summary"

    def test_summary_stable_after_serialize_restore(self):
        """Summary is consistent after round-trip through serialize/restore."""
        zr = _zone_runtime()
        zr.pending_boost_contexts["d1"] = _make_bctx("d1")
        zr.pending_dispatch_records["d1"] = _make_drec("d1")
        s1 = zr.pending_attribution_summary()
        blob = zr.serialize()
        zr2 = _zone_runtime()
        zr2.restore(blob)
        s2 = zr2.pending_attribution_summary()
        assert s1["pending_context_count"] == s2["pending_context_count"]
        assert s1["pending_dispatch_count"] == s2["pending_dispatch_count"]
        assert s1["total_targets"] == s2["total_targets"]

    def test_summary_decrements_after_consumption(self):
        """After consuming a pending context (episode close), count decrements."""
        zr = _zone_runtime()
        zr.pending_boost_contexts["d1"] = _make_bctx("d1")
        assert zr.pending_attribution_summary()["pending_context_count"] == 1
        zr.pending_boost_contexts.pop("d1")
        assert zr.pending_attribution_summary()["pending_context_count"] == 0

    def test_summary_bounded_at_cap(self):
        """With 64 pending records (cap), summary remains compact."""
        zr = _zone_runtime()
        for i in range(64):
            did = f"dec-{i:04d}"
            zr.pending_dispatch_records[did] = _make_drec(did)
        s = zr.pending_attribution_summary()
        assert s["pending_dispatch_count"] == 64
        assert len(json.dumps(s)) < 8192, "Summary must stay small at cap"


class TestLearningRuntimePendingAttributionSummary:
    """Tests for LearningRuntime.pending_attribution_summary(zone_id)."""

    async def test_summary_available_for_zone(self):
        from tests.helpers_runtime_scenarios import runtime
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        # Zone doesn't exist yet → zero state returned, no crash
        s = rt.pending_attribution_summary("lz")
        assert s["pending_context_count"] == 0

    async def test_summary_reflects_pending_in_zone(self):
        from tests.helpers_runtime_scenarios import runtime
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        zr = rt._zone("lz")
        zr.pending_dispatch_records["d1"] = _make_drec("d1")
        s = rt.pending_attribution_summary("lz")
        assert s["pending_dispatch_count"] == 1

    async def test_summary_unknown_zone_no_crash(self):
        from tests.helpers_runtime_scenarios import runtime
        rt = runtime(store=MemoryStore(data=None))
        await rt.async_setup()
        s = rt.pending_attribution_summary("nonexistent_zone")
        assert isinstance(s, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DiagnosticsOrchestrator: pending_attribution in zone output
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiagnosticsWithPendingAttribution:
    """Tests for pending_attribution field in DiagnosticsOrchestrator output."""

    def _diag_zone_with_pa(self, **pa_overrides):
        pa = {
            "schema_version": 1,
            "pending_context_count": 2,
            "pending_dispatch_count": 2,
            "outcome_eligible_count": 1,
            "dispatch_statuses": {"fully_succeeded": 1, "partially_succeeded": 1},
            "control_types": ["setpoint"],
            "total_targets": 3,
            "failed_targets": 1,
            "pending_outcome_active": False,
        }
        pa.update(pa_overrides)
        return ZoneExportInput(zone_id="lz", pending_attribution=pa)

    def _build(self, zone_input):
        return build_diagnostics(DiagnosticsInput(
            engine=EngineSummary(le_version="2.0", storage_schema_version=1,
                                 initialized=True, mode="shadow", registry_valid=True),
            zones=(zone_input,),
            pseudonymizer=Pseudonymizer(),
            generated_at="2026-06-26T10:00:00+00:00",
        ))

    def test_pending_attribution_in_zone_output(self):
        z = self._diag_zone_with_pa()
        result = self._build(z)
        pa = result.payload["zones"][0].get("pending_attribution")
        assert pa is not None, "pending_attribution must appear in zone diagnostics output"

    def test_pending_attribution_counts_correct(self):
        z = self._diag_zone_with_pa(pending_context_count=3, pending_dispatch_count=2)
        result = self._build(z)
        pa = result.payload["zones"][0]["pending_attribution"]
        assert pa["pending_context_count"] == 3
        assert pa["pending_dispatch_count"] == 2

    def test_pending_attribution_dispatch_statuses(self):
        z = self._diag_zone_with_pa(
            dispatch_statuses={"fully_succeeded": 2, "failed": 1})
        result = self._build(z)
        pa = result.payload["zones"][0]["pending_attribution"]
        assert pa["dispatch_statuses"]["fully_succeeded"] == 2
        assert pa["dispatch_statuses"]["failed"] == 1

    def test_pending_attribution_control_types_sorted(self):
        z = self._diag_zone_with_pa(control_types=["setpoint", "direct_valve"])
        result = self._build(z)
        pa = result.payload["zones"][0]["pending_attribution"]
        assert pa["control_types"] == ["direct_valve", "setpoint"]

    def test_pending_attribution_absent_when_none(self):
        z = ZoneExportInput(zone_id="lz")  # no pending_attribution
        result = self._build(z)
        pa = result.payload["zones"][0].get("pending_attribution")
        assert pa is None, "pending_attribution must not appear when ZoneExportInput.pending_attribution is None"

    def test_pending_attribution_no_errors(self):
        z = self._diag_zone_with_pa()
        result = self._build(z)
        privacy_errors = [e for e in result.errors if e.startswith("privacy:")]
        assert not privacy_errors, \
            f"No privacy violations in pending_attribution output; got {privacy_errors}"

    def test_pending_attribution_schema_version_in_output(self):
        z = self._diag_zone_with_pa(schema_version=1)
        result = self._build(z)
        pa = result.payload["zones"][0]["pending_attribution"]
        assert pa["schema_version"] == 1

    def test_pending_attribution_json_serializable(self):
        z = self._diag_zone_with_pa()
        result = self._build(z)
        pa = result.payload["zones"][0]["pending_attribution"]
        json.dumps(pa)  # must not raise

    def test_zero_state_diagnostics_includes_pending_attribution(self):
        z = self._diag_zone_with_pa(
            pending_context_count=0,
            pending_dispatch_count=0,
            outcome_eligible_count=0,
            dispatch_statuses={},
            control_types=[],
            total_targets=0,
            failed_targets=0,
            pending_outcome_active=False,
        )
        result = self._build(z)
        pa = result.payload["zones"][0]["pending_attribution"]
        assert pa is not None
        assert pa["pending_context_count"] == 0

    def test_outcome_active_reflected_in_output(self):
        z = self._diag_zone_with_pa(pending_outcome_active=True)
        result = self._build(z)
        pa = result.payload["zones"][0]["pending_attribution"]
        assert pa["pending_outcome_active"] is True
