"""Phase B — F1-F4: multi-TRV Boost learning integrity, dispatch-supported
provenance, older/malformed persisted state, and orphaned-context cleanup.

F1: adaptive Boost learning consumes the POST-dispatch CONFIRMED effective offset
    (not the pre-dispatch APPROVED offset); held devices are represented in the
    multi-TRV ambiguity check; a later unboosted cycle does not erase a boosted
    cycle's dispatch record.
F2: Boost exposure is derived from dispatch EVIDENCE (status + effective offset),
    so failed / not-attempted Boost cannot suppress clean-baseline learning.
F3: missing/malformed provenance never turns a Boost-exposed episode into a clean
    baseline when other persisted evidence exists.
F4: pending Boost contexts / dispatch records do not accumulate when their episode
    can no longer consume them.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from custom_components.thermosmart.export import _learning_application_export
from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, LearningRuntimeMode, RuntimeCycleInput,
    ScheduleTarget)
from custom_components.thermosmart.learning.runtime.capture import (
    BoostDispatchRecord, CapturedSnapshot)
from custom_components.thermosmart.learning.runtime.evidence import EvidenceMaterializer
from custom_components.thermosmart.learning.runtime.ha_integration import (
    _boost_dispatch_from_live_record)
from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_dispatching_coordinator
from tests.helpers_runtime_scenarios import COMFORT, T0, runtime

_ZONE = "lz"
_TARGET = 21.0
_TEMPS = [18.0, 18.6, 19.2, 19.8, 20.4, 21.0, 21.0, 21.0, 21.0]


# ── shared builders ───────────────────────────────────────────────────────────

def _bd(status="fully_succeeded", *, approved=0.5, effective=0.5, held=(), written=None,
        base=21.0):
    return BoostDispatchRecord(
        decision_id="d1", boost_candidate_c=1.0, boost_applied_c=approved,
        baseline_setpoint_c=base, final_setpoint_c=base + (approved or 0.0),
        dispatch_status=status, effective_offset_c=effective,
        held_effective_setpoints=tuple(held),
        effective_setpoints=tuple(written if written is not None
                                  else ((base + effective,) if effective and effective > 0 else ())),
        targets_total=1)


def _snap(bd, *, decision_type=DecisionType.BOOST):
    return CapturedSnapshot(
        zone_id=_ZONE, ts=T0.isoformat(), decision_id="d1", target_c=_TARGET,
        trv_setpoint_c=_TARGET + 0.5, indoor_temp=Measurement(19.0, DataQuality.OK),
        indoor_temp_quality=DataQuality.OK, decision_type=decision_type,
        controller_demand=True, tpi_on=True, boost_active=True, preheat_active=False,
        fallback_active=False, window_open=False, heating_failure=False,
        schedule_comfort_time_utc=None, schedule_comfort_temperature_c=None,
        outdoor_temp=None, valve_opening=None, hvac_action="heating", forecast_high=None,
        boost_dispatch=bd)


def _cycle(rt, i, temp, *, zone=_ZONE, target=_TARGET, dt=DecisionType.NORMAL,
           approved=0.0, effective=0.0, status="fully_succeeded", held=(), written=None,
           heating=True, window=None):
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = target + (3.0 if heating else -4.0)
    bd = BoostDispatchRecord(
        decision_id="x", boost_applied_c=approved, baseline_setpoint_c=target,
        final_setpoint_c=setp, dispatch_status=status, effective_offset_c=effective,
        held_effective_setpoints=tuple(held),
        effective_setpoints=tuple(written if written is not None
                                  else ((setp,) if effective and effective > 0 else ())),
        targets_total=1)
    inp = RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK), window_open=window,
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(
            decision_type=dt, target_c=target, trv_setpoint_c=setp,
            boost_active=(dt is DecisionType.BOOST)),
        boost_dispatch=bd)
    return rt.run_cycle(inp)


# ── F1: adaptive learning consumes the effective (not approved) offset ────────

class TestLearningConsumesEffectiveOffset:
    def test_context_uses_effective_offset(self):
        # approved 0.5, effective 0.2 -> the learning context must carry 0.2.
        ev = EvidenceMaterializer().materialize(_snap(_bd(approved=0.5, effective=0.2)))
        assert ev.boost_context is not None
        assert ev.boost_context.boost_applied_c == pytest.approx(0.2)
        assert ev.boost_context.requested_offset_c == pytest.approx(0.2)

    def test_effective_none_falls_back_to_approved(self):
        # Older record: effective_offset_c is None -> fall back to approved 0.5.
        ev = EvidenceMaterializer().materialize(_snap(_bd(approved=0.5, effective=None)))
        assert ev.boost_context is not None
        assert ev.boost_context.boost_applied_c == pytest.approx(0.5)

    def test_effective_zero_forms_no_context(self):
        # Boost approved (0.5) but clamped away to 0 effective -> no learning outcome.
        ev = EvidenceMaterializer().materialize(_snap(_bd(approved=0.5, effective=0.0)))
        assert ev.boost_context is None
        assert "boost_zero_applied" in ev.notes

    async def test_real_coordinator_effective_offset_propagates(self):
        # Mixed held(28.2)/written(28.5): approved 0.5, effective (M4 min) 0.2.
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
        assert ldr.boost_applied_c == pytest.approx(0.5)                      # approved
        assert ldr.dispatch_effective_boost_offset_c == pytest.approx(0.2)    # effective
        assert ldr.dispatch_held_effective_setpoints == (pytest.approx(28.2),)
        bd = _boost_dispatch_from_live_record(ldr)
        assert bd.effective_offset_c == pytest.approx(0.2)                    # learning uses this
        assert bd.boost_applied_c == pytest.approx(0.5)                       # baseline keeps approved

    async def test_regression_real_cycle_context_uses_effective_not_approved(self):
        # VERSION-AGNOSTIC regression proof (no new-field construction, so it also runs
        # against ca73672): after a real mixed held(28.2)/written(28.5) coordinator cycle,
        # the pending Boost-learning context must carry the POST-dispatch effective offset
        # (0.2), NOT the pre-dispatch approved offset (0.5). Against ca73672 this is 0.5.
        coord = make_dispatching_coordinator(
            {"schedule_enabled": False, "boost_bootstrap_prior_c": 1.0}, indoor="19.5",
            climate_entities=["climate.a", "climate.b"],
            climate_states={"climate.a": {"temperature": 28.2},
                            "climate.b": {"temperature": 21.0}})
        coord.set_active_control(True)
        sh = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
        await sh.async_setup()
        coord.data = await coord._async_update_data()
        zr = sh._runtime._zone(coord.zone_id)
        ctxs = list(zr.pending_boost_contexts.values())
        assert len(ctxs) == 1
        assert ctxs[0].boost_applied_c == pytest.approx(0.2)         # effective, not 0.5
        assert ctxs[0].requested_offset_c == pytest.approx(0.2)


# ── F1.4: held devices are represented in multi-TRV ambiguity ─────────────────

class TestMultiTrvIncludesHeld:
    def test_divergent_held_device_raises_ambiguity(self):
        # One WRITTEN device at the full boost (base 21 -> 23, offset 2.0) and one
        # already-HELD device still at baseline (21.0, offset 0.0). With only the
        # written device the spread check cannot run (single device); INCLUDING the
        # held device yields a 2.0 spread (> the ~0.68 tolerance) -> MULTI_TRV_AMBIGUOUS.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        drec = BoostDispatchRecord(
            decision_id="d", boost_applied_c=2.0, baseline_setpoint_c=21.0,
            dispatch_status="fully_succeeded", effective_offset_c=0.0,
            effective_setpoints=(23.0,), held_effective_setpoints=(21.0,), targets_total=1)
        flags = rt._runtime_confounders(rt._zone(_ZONE), None, drec, None)
        assert any("multi_trv" in f for f in flags)

    def test_omitting_held_would_miss_ambiguity(self):
        # Control: the SAME written device alone (no held device represented) does not
        # flag — proving the ambiguity above comes from including the held device.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        drec = BoostDispatchRecord(
            decision_id="d", boost_applied_c=2.0, baseline_setpoint_c=21.0,
            dispatch_status="fully_succeeded", effective_offset_c=2.0,
            effective_setpoints=(23.0,), held_effective_setpoints=(), targets_total=1)
        flags = rt._runtime_confounders(rt._zone(_ZONE), None, drec, None)
        assert not any("multi_trv" in f for f in flags)

    def test_uniform_devices_no_ambiguity(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        drec = BoostDispatchRecord(
            decision_id="d", boost_applied_c=2.0, baseline_setpoint_c=21.0,
            dispatch_status="fully_succeeded", effective_offset_c=2.0,
            effective_setpoints=(23.0,), held_effective_setpoints=(23.0,), targets_total=1)
        flags = rt._runtime_confounders(rt._zone(_ZONE), None, drec, None)
        assert not any("multi_trv" in f for f in flags)


# ── F2: Boost exposure from dispatch evidence (failed/not-attempted don't suppress)

def _run(boost_cycles):
    rt = runtime(mode=LearningRuntimeMode.CONTROL)
    zr = rt._zone(_ZONE)
    for i, t in enumerate(_TEMPS):
        kw = boost_cycles.get(i, {})
        _cycle(rt, i, t, heating=(t < _TARGET), **kw)
    return zr


class TestExposureFromDispatchEvidence:
    def test_failed_boost_does_not_suppress_baseline(self):
        # A failed boost (approved>0 but effective 0, status failed) has no thermal
        # effect; a later clean reach must still store a baseline.
        zr = _run({2: dict(dt=DecisionType.BOOST, approved=0.5, effective=0.0, status="failed")})
        assert zr.baseline_store.size == 1

    def test_not_attempted_boost_does_not_suppress_baseline(self):
        zr = _run({2: dict(dt=DecisionType.BOOST, approved=0.5, effective=0.0,
                           status="not_attempted")})
        assert zr.baseline_store.size == 1

    def test_partial_boost_suppresses_baseline(self):
        zr = _run({2: dict(dt=DecisionType.BOOST, approved=0.5, effective=0.0,
                           status="partially_succeeded")})
        assert zr.baseline_store.size == 0

    def test_already_satisfied_boost_suppresses_baseline(self):
        zr = _run({2: dict(dt=DecisionType.BOOST, approved=0.5, effective=0.4, held=(21.4,),
                           status="already_satisfied")})
        assert zr.baseline_store.size == 0


# ── F1.5: a later unboosted cycle does not erase boost-device ambiguity ───────

class TestLastCycleDispatchRetention:
    def test_partial_boost_then_clean_cycles_not_baseline(self):
        # cycle 2 is a partial (ambiguous) boost; cycles 3+ are clean. The partial
        # record must be retained for the decision so the episode is not a baseline.
        zr = _run({2: dict(dt=DecisionType.BOOST, approved=0.5, effective=0.0,
                           status="partially_succeeded")})
        assert zr.baseline_store.size == 0


# ── F4: orphaned pending contexts do not accumulate ───────────────────────────

class TestOrphanCleanup:
    def test_repeated_aborts_do_not_accumulate_pending(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        # Many short boosted episodes each aborted by a window-open, under changing
        # targets (new decision each time). Pending dicts must stay bounded/small.
        for k in range(30):
            tgt = 20.0 + (k % 5) * 0.5
            _cycle(rt, k * 2, 18.0, target=tgt, dt=DecisionType.BOOST,
                   approved=1.0, effective=1.0, written=[tgt + 1.0])
            _cycle(rt, k * 2 + 1, 18.5, target=tgt, window=True)   # abort
        assert len(zr.pending_boost_contexts) <= 2
        assert len(zr.pending_dispatch_records) <= 2
        assert len(zr.pending_baseline_ctx) <= 2

    def test_pending_does_not_survive_as_orphan_across_decisions(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        _cycle(rt, 0, 18.0, dt=DecisionType.BOOST, approved=1.0, effective=1.0, written=[22.0])
        # move to a new decision (target change) without closing an episode under the old id
        for i, t in enumerate([19.0, 20.0, 21.0, 22.0, 22.0, 22.0], start=1):
            _cycle(rt, i, t, target=22.0, heating=(t < 22.0))
        # the old boost decision's pending entries were pruned (never consumable)
        assert len(zr.pending_boost_contexts) <= 2
        assert len(zr.pending_dispatch_records) <= 2


# ── F3: older / malformed persisted provenance ────────────────────────────────

class TestOlderMalformedProvenance:
    def test_old_record_without_effective_offset_still_learns_via_fallback(self):
        # effective_offset_c missing (None) on an older record -> fallback to approved.
        bd = _bd(approved=0.6, effective=None)
        ev = EvidenceMaterializer().materialize(_snap(bd))
        assert ev.boost_context is not None
        assert ev.boost_context.boost_applied_c == pytest.approx(0.6)
