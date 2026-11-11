"""Phase B — M1/M2/L1: dispatch status must not hide Boost exposure, legacy
exposure fallback, and malformed-persistence unreliability.

M1: episode Boost exposure is driven SOLELY by the authoritative per-device
    "any device boosted" predicate — never gated on the aggregate dispatch status.
    A zone-level ``failed`` status can coexist with a TRV HELD above its unboosted
    baseline (a real exposure); when the capture decision_id changes while the
    outcome episode stays open, the episode-scoped ``boost_exposed`` flag is the
    ONLY thing standing between that cycle and a contaminated clean baseline.
M2: a legacy persisted dispatch record (no ``any_device_boosted`` flag, and
    possibly no per-device setpoints) is still protected from being read as a clean
    baseline — first from trustworthy per-device setpoints, else from a conservative
    approved/effective-offset fallback gated on an applied/held dispatch status.
L1: a malformed persisted setpoint tuple marks the record UNRELIABLE instead of
    silently dropping confounding evidence; an unreliable record fails closed for
    both adaptive Boost learning and clean-baseline attribution.

The M1/M2 reproductions are version-agnostic enough to also run against the
pre-remediation commit 50a5b44, where they FAIL (a contaminated baseline / a
spurious adaptive Boost update); after the fix they pass.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, LearningRuntimeMode, RuntimeCycleInput,
    ScheduleTarget)
from custom_components.thermosmart.learning.runtime.capture import (
    BoostDispatchRecord, record_any_device_boosted)
from custom_components.thermosmart.learning.runtime.evidence import EvidenceMaterializer
from custom_components.thermosmart.learning.runtime.ha_integration import (
    _boost_dispatch_from_live_record)
from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_dispatching_coordinator
from tests.helpers_runtime_scenarios import COMFORT, T0, runtime

_ZONE = "lz"
_TARGET = 21.0
# ramp 18->21 then hold at target long enough for the stability window.
_TEMPS = [18.0, 18.6, 19.2, 19.8, 20.4, 21.0, 21.0, 21.0, 21.0]


# ── record / snapshot / cycle builders ───────────────────────────────────────

def _rec(*, status="fully_succeeded", base=_TARGET, written=(), held=(), approved=0.0,
         effective=0.0, any_boosted=None, decision_id="x"):
    """A BoostDispatchRecord with explicit per-field control. ``any_boosted=None``
    emulates a LEGACY record that predates the authoritative flag."""
    return BoostDispatchRecord(
        decision_id=decision_id, boost_candidate_c=(approved or None),
        boost_applied_c=approved, baseline_setpoint_c=base,
        final_setpoint_c=base + (approved or 0.0), dispatch_status=status,
        effective_offset_c=effective, effective_setpoints=tuple(written),
        held_effective_setpoints=tuple(held), any_device_boosted=any_boosted,
        targets_total=1)


def _snap(bd, *, decision_type=DecisionType.BOOST):
    from custom_components.thermosmart.learning.runtime.capture import CapturedSnapshot
    return CapturedSnapshot(
        zone_id=_ZONE, ts=T0.isoformat(), decision_id="x", target_c=_TARGET,
        trv_setpoint_c=_TARGET + 0.5, indoor_temp=Measurement(19.0, DataQuality.OK),
        indoor_temp_quality=DataQuality.OK, decision_type=decision_type,
        controller_demand=True, tpi_on=True, boost_active=True, preheat_active=False,
        fallback_active=False, window_open=False, heating_failure=False,
        schedule_comfort_time_utc=None, schedule_comfort_temperature_c=None,
        outdoor_temp=None, valve_opening=None, hvac_action="heating", forecast_high=None,
        boost_dispatch=bd)


def _cycle(rt, i, temp, *, zone=_ZONE, target=_TARGET, dt=DecisionType.NORMAL,
           rec=None, heating=True, window=None):
    """One real runtime cycle. A clean NORMAL cycle still carries a (clean) dispatch
    record, as the live coordinator always supplies one — baseline storage requires it."""
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = target + (3.0 if heating else -4.0)
    bd = rec if rec is not None else _rec(status="fully_succeeded", base=target)
    inp = RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK), window_open=window,
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(
            decision_type=dt, target_c=target, trv_setpoint_c=setp,
            boost_active=(dt is DecisionType.BOOST)),
        boost_dispatch=bd)
    return rt.run_cycle(inp)


def _failed_held(base=_TARGET):
    """A real mixed cycle: aggregate status ``failed`` (a write raised) while one TRV
    remains HELD above baseline — authoritative exposure True, conservative min 0.0."""
    return _rec(status="failed", base=base, written=(), held=(base + 1.0,),
                approved=0.8, effective=0.0, any_boosted=True)


# ── M1: dispatch status must not hide real Boost exposure ─────────────────────

class TestM1FailedExposureWithHeld:
    async def test_1_real_coordinator_failed_write_held_device_exposed(self):
        # A real coordinator cycle: climate.a already HOLDS the boost (no write),
        # climate.b's write RAISES -> aggregate status is not fully_succeeded, yet the
        # authoritative any_device_boosted flag is True (the held device is above
        # baseline). Exposure must not be hidden by the failed aggregate status.
        def _side(domain, service, data=None, **kw):
            eid = (data or {}).get("entity_id")
            ents = eid if isinstance(eid, (list, tuple)) else [eid]
            if service == "set_temperature" and "climate.b" in ents:
                raise RuntimeError("boom")

        coord = make_dispatching_coordinator(
            {"schedule_enabled": False, "boost_bootstrap_prior_c": 1.0}, indoor="19.5",
            climate_entities=["climate.a", "climate.b"],
            climate_states={"climate.a": {"temperature": 28.2},
                            "climate.b": {"temperature": 21.0}},
            service_side_effect=_side)
        coord.set_active_control(True)
        sh = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
        await sh.async_setup()
        coord.data = await coord._async_update_data()
        ldr = coord._last_live_decision
        assert ldr is not None
        assert ldr.dispatch_status in ("failed", "partially_succeeded")   # a write failed
        assert ldr.dispatch_any_device_boosted is True                    # held above baseline
        assert ldr.dispatch_held_effective_setpoints                      # held setpoint captured
        bd = _boost_dispatch_from_live_record(ldr)
        assert record_any_device_boosted(bd) is True

    def test_2_regression_target_change_failed_held_not_baseline(self):
        # REGRESSION (fails on 50a5b44): outcome episode opens at target 21; the target
        # steps to 23 mid-episode (a NEW capture decision_id) and a FAILED boost leaves
        # one TRV held above baseline. The active episode stays under the ORIGINAL id,
        # so the dispatch record sits under a different id and close-time pops a clean
        # record — the episode-scoped boost_exposed flag is the SOLE protection. On
        # 50a5b44 the status gate suppressed that flag for `failed`, so a clean baseline
        # was stored (size 1).
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        _cycle(rt, 0, 18.0, target=21.0)
        _cycle(rt, 1, 19.0, target=21.0)
        _cycle(rt, 2, 20.0, target=23.0, dt=DecisionType.BOOST, rec=_failed_held(21.0))
        for i, t in enumerate([21.0, 22.0, 23.0, 23.0, 23.0], start=3):
            _cycle(rt, i, t, target=23.0, heating=(t < 23.0))
        assert zr.baseline_store.size == 0
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) >= 1

    def test_3_preheat_then_failed_held_boost_not_baseline(self):
        # PREHEAT -> BOOST class change (a new capture decision_id) with the same mixed
        # failure. The outcome episode opened under PREHEAT stays open and latches the
        # exposure; it must not close as a clean baseline.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        _cycle(rt, 0, 18.0, dt=DecisionType.PREHEAT)
        _cycle(rt, 1, 19.0, dt=DecisionType.PREHEAT)
        _cycle(rt, 2, 20.0, dt=DecisionType.BOOST, rec=_failed_held(21.0))
        for i, t in enumerate([20.6, 21.0, 21.0, 21.0], start=3):
            _cycle(rt, i, t, heating=(t < 21.0))
        assert zr.baseline_store.size == 0

    def test_4_restart_between_exposed_cycle_and_close_not_baseline(self):
        # The failed+held exposure happens under target 21, the target then steps to 23
        # (episode stays open under the original id), we serialize, restore into a fresh
        # runtime, and only then reach. The episode-scoped provenance must persist across
        # the restart so the restored episode is still excluded.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        _cycle(rt, 0, 18.0, target=21.0)
        _cycle(rt, 1, 19.0, target=21.0)
        _cycle(rt, 2, 20.0, target=23.0, dt=DecisionType.BOOST, rec=_failed_held(21.0))
        payload = rt._build_payload()
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        for i, t in enumerate([21.0, 22.0, 23.0, 23.0, 23.0], start=3):
            _cycle(rt2, i, t, target=23.0, heating=(t < 23.0))
        assert rt2._zone(_ZONE).baseline_store.size == 0

    def test_5_failed_without_boosted_device_stays_baseline(self):
        # Control: a failed dispatch with NO device above baseline (no held, no
        # successful write) is a genuinely non-boost cycle -> baseline eligible. The
        # removed status gate must not cause over-exclusion.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        for i, t in enumerate(_TEMPS):
            if i == 2:
                _cycle(rt, i, t, dt=DecisionType.BOOST,
                       rec=_rec(status="failed", base=_TARGET, written=(), held=(),
                                approved=0.8, effective=0.0, any_boosted=False),
                       heating=(t < _TARGET))
            else:
                _cycle(rt, i, t, heating=(t < _TARGET))
        assert zr.baseline_store.size == 1

    def test_6_multi_zone_isolation(self):
        # Zone A sees a failed+held (exposed) boost; zone B is clean. Exclusion in A must
        # not leak into B, and B's clean baseline must not be suppressed by A.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        for i, t in enumerate(_TEMPS):
            if i == 2:
                _cycle(rt, i, t, zone="A", dt=DecisionType.BOOST, rec=_failed_held(21.0),
                       heating=(t < _TARGET))
            else:
                _cycle(rt, i, t, zone="A", heating=(t < _TARGET))
            _cycle(rt, i, t, zone="B", heating=(t < _TARGET))
        assert rt._zone("A").baseline_store.size == 0
        assert rt._zone("B").baseline_store.size == 1

    def test_7_abort_then_clean_episode_stores_baseline(self):
        # A failed+held boosted episode aborted by a window-open; a subsequent CLEAN
        # episode under a new target must still store a baseline (the flag did not leak).
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        _cycle(rt, 0, 18.0, dt=DecisionType.BOOST, rec=_failed_held(21.0))
        _cycle(rt, 1, 19.0, window=True)   # abort
        for i, t in enumerate([18.0, 18.8, 19.6, 20.4, 22.0, 22.0, 22.0, 22.0], start=2):
            _cycle(rt, i, t, target=22.0, heating=(t < 22.0))
        assert zr.baseline_store.size == 1
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) == 0


# ── M2: legacy persisted records without the exposure flag ────────────────────

class TestM2LegacyExposure:
    def test_1_legacy_held_only_derives_exposure_from_setpoints(self):
        # Pre-flag held-only Boost payload (no any_device_boosted) -> exposure from the
        # persisted per-device HELD setpoint above baseline.
        bd = _rec(status="already_satisfied", base=21.0, held=(21.6,), approved=0.6,
                  effective=0.6, any_boosted=None)
        assert record_any_device_boosted(bd) is True

    def test_2_legacy_written_boost_derives_exposure(self):
        bd = _rec(status="fully_succeeded", base=21.0, written=(22.0,), approved=1.0,
                  effective=1.0, any_boosted=None)
        assert record_any_device_boosted(bd) is True

    def test_3_legacy_mixed_held_written_derives_exposure(self):
        bd = _rec(status="fully_succeeded", base=21.0, written=(22.0,), held=(21.0,),
                  approved=1.0, effective=0.0, any_boosted=None)
        assert record_any_device_boosted(bd) is True

    def test_4_legacy_clean_episode_is_baseline_eligible(self):
        # No boost, all at baseline, no flag -> not exposed.
        bd = _rec(status="fully_succeeded", base=21.0, written=(), held=(), approved=0.0,
                  effective=0.0, any_boosted=None)
        assert record_any_device_boosted(bd) is False

    def test_5_legacy_failed_without_device_evidence_not_exposed(self):
        # failed / not_attempted with no positive device evidence must not be read as
        # confirmed exposure even under the conservative fallback.
        assert record_any_device_boosted(
            _rec(status="failed", base=21.0, approved=0.8, effective=None,
                 any_boosted=None)) is False
        assert record_any_device_boosted(
            _rec(status="not_attempted", base=21.0, approved=0.8, effective=None,
                 any_boosted=None)) is False

    def test_6_legacy_fallback_when_perdevice_evidence_absent(self):
        # The M2 fix: a legacy applied/held record with NO persisted per-device setpoints
        # but a finite positive APPROVED offset -> conservative exposure (baseline
        # protection). On 50a5b44 this returned False (misclassified as clean).
        bd = _rec(status="fully_succeeded", base=21.0, written=(), held=(), approved=0.8,
                  effective=None, any_boosted=None)
        assert record_any_device_boosted(bd) is True

    def test_7_regression_legacy_inflight_boost_not_contaminated_baseline(self):
        # REGRESSION (fails on 50a5b44): a legacy in-flight Boost (status succeeded,
        # positive approved offset, but no per-device setpoints and no flag) must not
        # close as a clean baseline. On 50a5b44 the conservative fallback did not exist,
        # so record_any_device_boosted returned False and a baseline was stored (size 1).
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        for i, t in enumerate(_TEMPS):
            if i == 2:
                _cycle(rt, i, t, dt=DecisionType.BOOST,
                       rec=_rec(status="fully_succeeded", base=_TARGET, written=(), held=(),
                                approved=0.8, effective=None, any_boosted=None),
                       heating=(t < _TARGET))
            else:
                _cycle(rt, i, t, heating=(t < _TARGET))
        assert zr.baseline_store.size == 0
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) >= 1

    def test_8_no_adaptive_update_from_unverified_approved_offset(self):
        # The legacy fallback protects the baseline but must NEVER feed adaptive learning:
        # a record with a positive approved offset but NO effective offset forms no Boost
        # context (fail closed), so no adaptive Boost magnitude is learned.
        bd = _rec(status="fully_succeeded", base=21.0, approved=0.8, effective=None,
                  any_boosted=None)
        ev = EvidenceMaterializer().materialize(_snap(bd))
        assert ev.boost_context is None

    def test_9_later_unboosted_cycle_keeps_legacy_evidence(self):
        # A legacy exposed record followed by clean cycles of the SAME decision is
        # retained (relevance uses the authoritative predicate) so the episode stays
        # excluded — later unboosted cycles do not erase the historical evidence.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        _cycle(rt, 0, 18.0)
        _cycle(rt, 1, 19.0, dt=DecisionType.BOOST,
               rec=_rec(status="fully_succeeded", base=_TARGET, written=(), held=(),
                        approved=0.8, effective=None, any_boosted=None))
        for i, t in enumerate([19.8, 20.4, 21.0, 21.0, 21.0], start=2):
            _cycle(rt, i, t, heating=(t < _TARGET))
        assert zr.baseline_store.size == 0


# ── L1: malformed persisted setpoint tuples -> unreliable, fail closed ────────

class TestL1MalformedPersistence:
    def test_malformed_member_marks_record_unreliable(self):
        d = _rec(status="fully_succeeded", base=21.0, written=(23.0,), held=(21.0,),
                 approved=2.0, effective=2.0).to_dict()
        d["held_effective_setpoints"] = [21.0, "bad", float("nan"), float("inf"), None]
        rec = BoostDispatchRecord.from_dict(d)
        assert rec.persistence_unreliable is True
        assert rec.held_effective_setpoints == (21.0,)   # valid member kept, bad ones out

    def test_non_list_container_marks_unreliable_and_empty(self):
        d = _rec(status="fully_succeeded", base=21.0, written=(23.0,), approved=2.0,
                 effective=2.0).to_dict()
        d["held_effective_setpoints"] = "not-a-list"
        rec = BoostDispatchRecord.from_dict(d)
        assert rec.persistence_unreliable is True
        assert rec.held_effective_setpoints == ()

    def test_valid_legacy_record_restores_clean(self):
        d = _rec(status="fully_succeeded", base=21.0, written=(23.0,), held=(21.0,),
                 approved=2.0, effective=2.0, any_boosted=None).to_dict()
        rec = BoostDispatchRecord.from_dict(d)
        assert rec.persistence_unreliable is False
        assert rec.effective_setpoints == (23.0,)
        assert rec.held_effective_setpoints == (21.0,)

    def test_unreliable_record_forms_no_adaptive_context(self):
        # A reliable record forms a Boost context; the SAME record made unreliable by a
        # malformed held container forms NONE — unreliable per-device evidence can never
        # produce an adaptive Boost context (the confounder must not be silently dropped).
        reliable = _rec(status="fully_succeeded", base=21.0, written=(23.0,), held=(23.0,),
                        approved=2.0, effective=2.0, any_boosted=True)
        assert EvidenceMaterializer().materialize(_snap(reliable)).boost_context is not None
        d = reliable.to_dict()
        d["held_effective_setpoints"] = "garbage"
        unrel = BoostDispatchRecord.from_dict(d)
        assert unrel.persistence_unreliable is True
        ev = EvidenceMaterializer().materialize(_snap(unrel))
        assert ev.boost_context is None
        assert "boost_evidence_unreliable" in ev.notes

    def test_unreliable_record_does_not_crash_relevance_or_cycle(self):
        d = _rec(status="fully_succeeded", base=21.0, written=(23.0,), approved=2.0,
                 effective=2.0).to_dict()
        d["held_effective_setpoints"] = [21.0, "bad"]
        rec = BoostDispatchRecord.from_dict(d)
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        assert rt._dispatch_boost_relevance(rec) >= 0.0
        inp = RuntimeCycleInput(
            zone_id=_ZONE, ts=T0.isoformat(), target_c=_TARGET, trv_setpoint_c=24.0,
            controller_demand=True, indoor_temp=Measurement(19.0, DataQuality.OK),
            schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=_TARGET),
            controller_decision=ControllerDecisionInput(
                decision_type=DecisionType.BOOST, target_c=_TARGET, trv_setpoint_c=24.0,
                boost_active=True),
            boost_dispatch=rec)
        rt.run_cycle(inp)   # must not raise

    def test_regression_restored_unreliable_record_fails_closed(self):
        # REGRESSION: a boosted episode whose persisted dispatch record is corrupted so
        # the ONLY divergent (held) device is lost. On 50a5b44 the malformed member was
        # silently dropped, the multi-TRV confounder vanished, and the boost was learned
        # as a clean adaptive sample. After the fix the record is unreliable -> fail
        # closed: no adaptive Boost update AND no clean baseline.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        _cycle(rt, 0, 18.0)
        _cycle(rt, 1, 19.0)
        _cycle(rt, 2, 19.5, dt=DecisionType.BOOST,
               rec=_rec(status="fully_succeeded", base=_TARGET, written=(23.0,),
                        held=(21.0,), approved=2.0, effective=2.0, any_boosted=True))
        payload = rt._build_payload()
        for z in payload["zones"].values():
            for drec_d in (z.get("pending_dispatch_records") or {}).values():
                drec_d["held_effective_setpoints"] = "corrupt"   # non-list -> unreliable
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        zr2 = rt2._zone(_ZONE)
        recs = list(zr2.pending_dispatch_records.values())
        assert recs and recs[0].persistence_unreliable is True
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _cycle(rt2, i, t, heating=(t < _TARGET))
        assert zr2.model_update_counts.get("boost", 0) == 0     # no adaptive update
        assert zr2.pending_boost_outcome is None
        assert zr2.baseline_store.size == 0                     # not a clean baseline
        assert zr2.baseline_store.rejection_counts().get("boost_applied", 0) >= 1


# ── L3: direct exposure-path assertions (close remaining test gaps) ───────────

class TestL3DirectAssertions:
    def test_exposure_threshold_is_0_05(self):
        # Legacy per-device derivation must use the 0.05 °C exposure threshold.
        assert record_any_device_boosted(
            _rec(status="fully_succeeded", base=21.0, held=(21.04,), any_boosted=None)) is False
        assert record_any_device_boosted(
            _rec(status="fully_succeeded", base=21.0, held=(21.06,), any_boosted=None)) is True

    def test_explicit_flag_overrides_perdevice_derivation(self):
        # Current-format records trust the authoritative flag alone: True even when the
        # conservative per-device min is 0 (mixed zone), and the derivation is not
        # consulted.
        bd = _rec(status="fully_succeeded", base=21.0, written=(22.0,), held=(21.0,),
                  approved=1.0, effective=0.0, any_boosted=True)
        assert record_any_device_boosted(bd) is True

    def test_none_record_not_exposed(self):
        assert record_any_device_boosted(None) is False

    async def test_real_coordinator_sets_any_device_boosted_on_mixed_boost(self):
        # Direct assertion on the real coordinator: a mixed held/written boost sets the
        # authoritative dispatch_any_device_boosted flag True and records the per-device
        # held setpoint.
        coord = make_dispatching_coordinator(
            {"schedule_enabled": False, "boost_bootstrap_prior_c": 1.0}, indoor="19.5",
            climate_entities=["climate.a", "climate.b"],
            climate_states={"climate.a": {"temperature": 28.2},
                            "climate.b": {"temperature": 21.0}})
        coord.set_active_control(True)
        sh = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
        await sh.async_setup()
        coord.data = await coord._async_update_data()
        ldr = coord._last_live_decision
        assert ldr.dispatch_any_device_boosted is True
        assert ldr.dispatch_held_effective_setpoints == (pytest.approx(28.2),)


# ── F-A: partial legacy evidence must not suppress the conservative fallback ──

class TestFAPartialLegacyEvidence:
    def test_fa_written_at_baseline_with_approved_offset_is_exposed(self):
        # Legacy record (no flag): the ONLY persisted per-device setpoint is a WRITTEN
        # device at the zone baseline (offset 0.0); the actually-boosted HELD device was
        # never persisted by the old payload. The present-but-at-baseline written setpoint
        # must NOT be treated as conclusive "not exposed" — the conservative fallback
        # (status applied/held + positive approved offset) still applies. On 6bf7c84 this
        # short-circuited to False.
        bd = _rec(status="fully_succeeded", base=21.0, written=(21.0,), held=(),
                  approved=0.8, effective=None, any_boosted=None)
        assert record_any_device_boosted(bd) is True

    def test_fa_written_above_baseline_still_exposed(self):
        # The per-device branch alone still suffices when it shows a boost.
        bd = _rec(status="fully_succeeded", base=21.0, written=(22.0,), held=(),
                  approved=0.8, effective=1.0, any_boosted=None)
        assert record_any_device_boosted(bd) is True

    def test_fa_written_at_baseline_no_offset_not_exposed(self):
        # Control: written at baseline AND no positive approved/effective offset -> a
        # genuinely clean legacy cycle stays non-exposed (no over-exclusion).
        bd = _rec(status="fully_succeeded", base=21.0, written=(21.0,), held=(),
                  approved=0.0, effective=0.0, any_boosted=None)
        assert record_any_device_boosted(bd) is False

    def test_fa_explicit_false_not_overridden_by_perdevice(self):
        # A current-format explicit False (e.g. a real per-device min_temp clamp) is
        # authoritative and must NOT be overridden to True by a zone-baseline comparison,
        # even if a written setpoint happens to sit above baseline.
        bd = _rec(status="fully_succeeded", base=21.0, written=(22.0,), held=(),
                  approved=0.8, effective=0.0, any_boosted=False)
        assert record_any_device_boosted(bd) is False

    def test_fa_regression_partial_written_not_contaminated_baseline(self):
        # REGRESSION (fails on 6bf7c84): a boost cycle whose legacy-shaped record has a
        # written device AT baseline, no held setpoint, no flag, but a positive approved
        # offset. On 6bf7c84 exposure read False (per-device short-circuit) and a clean
        # baseline was stored; after F-A the fallback marks it exposed -> excluded.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        for i, t in enumerate(_TEMPS):
            if i == 2:
                _cycle(rt, i, t, dt=DecisionType.BOOST,
                       rec=_rec(status="fully_succeeded", base=_TARGET, written=(_TARGET,),
                                held=(), approved=0.8, effective=None, any_boosted=None),
                       heating=(t < _TARGET))
            else:
                _cycle(rt, i, t, heating=(t < _TARGET))
        assert zr.baseline_store.size == 0
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) >= 1


# ── F-B: unreliable provenance must survive a later clean dispatch ────────────

def _corrupt_unreliable_no_evidence(payload, *, strip_boost_seen=True, flag=None,
                                    only_zone=None):
    """Corrupt persisted dispatch records in `payload` so they are unreliable with NO
    positive exposure evidence (flag absent/False, no setpoints, no offsets). Optionally
    strip the episode-scoped boost_seen so the retained record is the SOLE protection.
    ``only_zone`` restricts corruption to one zone (isolation tests)."""
    for zid, zdata in payload["zones"].items():
        if only_zone is not None and zid != only_zone:
            continue
        for drec_d in (zdata.get("pending_dispatch_records") or {}).values():
            drec_d["held_effective_setpoints"] = "corrupt"   # non-list -> unreliable
            drec_d["effective_setpoints"] = []
            drec_d["effective_offset_c"] = None
            drec_d["boost_applied_c"] = None
            drec_d["any_device_boosted"] = flag
        # Drop any surviving boost context / baseline ctx so the unreliable dispatch
        # record is the SOLE provenance that this episode was boost-exposed — the exact
        # F-B scenario (older/lost-context payload). Without this a surviving boost
        # context would bind the outcome and mask the clean-baseline contamination.
        zdata["pending_boost_contexts"] = {}
        zdata["pending_baseline_ctx"] = {}
        if strip_boost_seen:
            for bstate in (zdata.get("pipeline", {}).get("builders", {}) or {}).values():
                if isinstance(bstate, dict) and isinstance(bstate.get("extra"), dict):
                    bstate["extra"].pop("boost_seen", None)
    return payload


class TestFBUnreliableSurvivesCleanCycle:
    def _drive_restore_corrupt(self, *, flag=None, zone=_ZONE):
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        _cycle(rt, 0, 18.0, zone=zone)
        _cycle(rt, 1, 19.0, zone=zone)
        _cycle(rt, 2, 19.5, zone=zone, dt=DecisionType.BOOST,
               rec=_rec(status="fully_succeeded", base=_TARGET, written=(23.0,),
                        held=(21.0,), approved=2.0, effective=2.0, any_boosted=True))
        payload = _corrupt_unreliable_no_evidence(rt._build_payload(), flag=flag)
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        return rt2

    def test_fb_regression_clean_cycles_do_not_erase_unreliable(self):
        # REGRESSION (fails on 6bf7c84): the restored unreliable record has NO exposure
        # evidence (relevance 0 on 6bf7c84), so the first later CLEAN cycle of the same
        # decision overwrote it and the episode closed as a clean baseline. After F-B the
        # unreliable record outranks a clean one and is retained -> failed closed.
        rt2 = self._drive_restore_corrupt()
        zr2 = rt2._zone(_ZONE)
        recs = list(zr2.pending_dispatch_records.values())
        assert recs and recs[0].persistence_unreliable is True
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _cycle(rt2, i, t, heating=(t < _TARGET))
        assert zr2.baseline_store.size == 0
        assert zr2.model_update_counts.get("boost", 0) == 0
        assert zr2.baseline_store.rejection_counts().get("boost_applied", 0) >= 1

    def test_fb_explicit_false_exposure_still_retained(self):
        # The unreliable marker alone (independent of an explicit False exposure flag)
        # keeps the record above a later clean dispatch.
        rt2 = self._drive_restore_corrupt(flag=False)
        zr2 = rt2._zone(_ZONE)
        assert list(zr2.pending_dispatch_records.values())[0].persistence_unreliable is True
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _cycle(rt2, i, t, heating=(t < _TARGET))
        assert zr2.baseline_store.size == 0

    def test_fb_multi_zone_isolation(self):
        # Zone A: a boosted episode whose persisted record is corrupted to unreliable.
        # Zone B: a clean boost-free episode. After restore only A's record is corrupted;
        # driving both to REACHED, A is excluded (unreliable retained) and B still stores
        # a clean baseline — no cross-zone leak either way.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        for i, t in enumerate([18.0, 19.0, 19.5]):
            _cycle(rt, i, t, zone="A",
                   dt=(DecisionType.BOOST if i == 2 else DecisionType.NORMAL),
                   rec=(_rec(status="fully_succeeded", base=_TARGET, written=(23.0,),
                             held=(21.0,), approved=2.0, effective=2.0, any_boosted=True)
                        if i == 2 else None),
                   heating=(t < _TARGET))
            _cycle(rt, i, t, zone="B", heating=(t < _TARGET))
        payload = _corrupt_unreliable_no_evidence(rt._build_payload(), only_zone="A")
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _cycle(rt2, i, t, zone="A", heating=(t < _TARGET))
            _cycle(rt2, i, t, zone="B", heating=(t < _TARGET))
        assert rt2._zone("A").baseline_store.size == 0
        assert rt2._zone("B").baseline_store.size == 1

    def test_fb_relevance_unreliable_outranks_clean(self):
        # Direct: an unreliable record (even with no exposure evidence) outranks a clean
        # reliable record, so retention keeps it.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        clean = _rec(status="fully_succeeded", base=_TARGET, written=(), held=(),
                     approved=0.0, effective=0.0, any_boosted=False)
        d = clean.to_dict()
        d["held_effective_setpoints"] = "corrupt"
        unreliable = BoostDispatchRecord.from_dict(d)
        assert unreliable.persistence_unreliable is True
        assert rt._dispatch_boost_relevance(unreliable) > rt._dispatch_boost_relevance(clean)

    def test_fb_f4_pruning_still_bounded(self):
        # F4: an unreliable record whose decision can no longer close an episode is still
        # pruned (no indefinite retention of unrelated records).
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        for k in range(30):
            tgt = 20.0 + (k % 5) * 0.5
            _cycle(rt, k * 2, 18.0, target=tgt, dt=DecisionType.BOOST,
                   rec=_rec(status="fully_succeeded", base=tgt, written=(tgt + 1.0,),
                            approved=1.0, effective=1.0, any_boosted=True))
            _cycle(rt, k * 2 + 1, 18.5, target=tgt, window=True)   # abort
        assert len(zr.pending_dispatch_records) <= 2


# ── F-C: invalid persisted fields are consistently marked unreliable ──────────

def _base_dict(**over):
    d = _rec(status="fully_succeeded", base=21.0, written=(22.0,), held=(21.0,),
             approved=1.0, effective=1.0, any_boosted=True).to_dict()
    d.update(over)
    return d


class TestFCMalformedFields:
    @pytest.mark.parametrize("field", [
        "boost_applied_c", "boost_candidate_c", "baseline_setpoint_c", "final_setpoint_c",
        "effective_offset_c", "effective_setpoint_min_c", "effective_setpoint_max_c",
        "start_deficit_c", "expected_non_boost_duration_s"])
    @pytest.mark.parametrize("bad", ["x", "nan", float("nan"), float("inf"),
                                     float("-inf"), [1, 2], {"a": 1}])
    def test_malformed_offset_marks_unreliable(self, field, bad):
        rec = BoostDispatchRecord.from_dict(_base_dict(**{field: bad}))
        assert rec.persistence_unreliable is True
        assert getattr(rec, field) is None

    @pytest.mark.parametrize("field", [
        "boost_applied_c", "effective_offset_c", "baseline_setpoint_c"])
    def test_explicit_null_offset_is_not_corruption(self, field):
        rec = BoostDispatchRecord.from_dict(_base_dict(**{field: None}))
        assert rec.persistence_unreliable is False
        assert getattr(rec, field) is None

    @pytest.mark.parametrize("field", ["boost_applied_c", "effective_offset_c"])
    def test_missing_offset_key_is_not_corruption(self, field):
        d = _base_dict()
        d.pop(field, None)
        rec = BoostDispatchRecord.from_dict(d)
        assert rec.persistence_unreliable is False
        assert getattr(rec, field) is None

    def test_valid_zero_offset_preserved(self):
        rec = BoostDispatchRecord.from_dict(_base_dict(boost_applied_c=0.0,
                                                       effective_offset_c=0.0))
        assert rec.persistence_unreliable is False
        assert rec.boost_applied_c == 0.0
        assert rec.effective_offset_c == 0.0

    @pytest.mark.parametrize("bad", [0, 1, "", "true", "false", 1.0, [], {}])
    def test_malformed_exposure_flag_marks_unreliable(self, bad):
        # Non-bool, non-None values are corruption, not authoritative flags. The flag is
        # dropped to None (never truthiness-coerced) and the record is unreliable.
        rec = BoostDispatchRecord.from_dict(_base_dict(any_device_boosted=bad))
        assert rec.persistence_unreliable is True
        assert rec.any_device_boosted is None

    @pytest.mark.parametrize("val", [True, False])
    def test_valid_explicit_flag_preserved(self, val):
        rec = BoostDispatchRecord.from_dict(_base_dict(any_device_boosted=val))
        assert rec.persistence_unreliable is False
        assert rec.any_device_boosted is val

    def test_null_and_missing_flag_are_legacy_absence(self):
        rec_null = BoostDispatchRecord.from_dict(_base_dict(any_device_boosted=None))
        assert rec_null.persistence_unreliable is False
        assert rec_null.any_device_boosted is None
        d = _base_dict()
        d.pop("any_device_boosted", None)
        rec_missing = BoostDispatchRecord.from_dict(d)
        assert rec_missing.persistence_unreliable is False
        assert rec_missing.any_device_boosted is None

    @pytest.mark.parametrize("field", ["effective_setpoints", "held_effective_setpoints"])
    @pytest.mark.parametrize("bad", ["x", {"a": 1}, 21.0, [21.0, "bad"],
                                     [float("nan")], [None]])
    def test_malformed_setpoint_container_marks_unreliable(self, field, bad):
        rec = BoostDispatchRecord.from_dict(_base_dict(**{field: bad}))
        assert rec.persistence_unreliable is True

    @pytest.mark.parametrize("field", ["effective_setpoints", "held_effective_setpoints"])
    def test_valid_and_empty_setpoint_containers_clean(self, field):
        assert BoostDispatchRecord.from_dict(
            _base_dict(**{field: []})).persistence_unreliable is False
        assert BoostDispatchRecord.from_dict(
            _base_dict(**{field: [21.0, 22.0]})).persistence_unreliable is False

    def test_unreliable_marker_round_trips(self):
        rec = BoostDispatchRecord.from_dict(_base_dict(boost_applied_c="x"))
        assert rec.persistence_unreliable is True
        # serialize -> restore must preserve the marker even though the offending field
        # is now a clean None.
        rec2 = BoostDispatchRecord.from_dict(rec.to_dict())
        assert rec2.persistence_unreliable is True

    def test_malformed_field_fails_closed_for_learning(self):
        # A record with a malformed present offset forms no adaptive Boost context.
        rec = BoostDispatchRecord.from_dict(_base_dict(boost_applied_c="x"))
        ev = EvidenceMaterializer().materialize(_snap(rec))
        assert ev.boost_context is None
        assert "boost_evidence_unreliable" in ev.notes
