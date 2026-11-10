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
    # N1: authoritative exposure — a boost that was approved on an exposed dispatch
    # status reached at least one device (independent of the conservative min).
    _exposed = status in ("fully_succeeded", "partially_succeeded", "already_satisfied")
    _any_boosted = bool((approved or 0.0) > 0.0 and _exposed)
    bd = BoostDispatchRecord(
        decision_id="x", boost_applied_c=approved, baseline_setpoint_c=target,
        final_setpoint_c=setp, dispatch_status=status, effective_offset_c=effective,
        held_effective_setpoints=tuple(held),
        effective_setpoints=tuple(written if written is not None
                                  else ((setp,) if effective and effective > 0 else ())),
        targets_total=1, any_device_boosted=_any_boosted)
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

    def test_effective_none_fails_closed(self):
        # N3: effective_offset_c absent -> adaptive learning FAILS CLOSED (no context);
        # the approved offset is NEVER treated as a confirmed post-dispatch application.
        ev = EvidenceMaterializer().materialize(_snap(_bd(approved=0.5, effective=None)))
        assert ev.boost_context is None
        assert "boost_zero_applied" in ev.notes

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
    def test_old_record_without_effective_offset_fails_closed_but_protects_baseline(self):
        # N3: an older record lacking effective_offset_c forms NO adaptive Boost
        # context (fail closed), but baseline protection is preserved via the
        # authoritative exposure signal derived from the persisted per-device
        # setpoints — here a written device above baseline -> exposed.
        from custom_components.thermosmart.learning.runtime.capture import (
            record_any_device_boosted)
        bd = _bd(approved=0.6, effective=None, written=[21.6], base=21.0)
        ev = EvidenceMaterializer().materialize(_snap(bd))
        assert ev.boost_context is None                       # fail closed for learning
        assert record_any_device_boosted(bd) is True          # but still exposed -> no clean baseline


# ── N1: mixed-TRV exposure vs conservative minimum ───────────────────────────

def _vcycle(rt, i, temp, *, zone=_ZONE, target=_TARGET, dt=DecisionType.NORMAL,
            effective=0.0, written=(), held=(), status="fully_succeeded", heating=True,
            window=None):
    """VERSION-AGNOSTIC driver (no any_device_boosted field, so it also runs against
    5448096): builds the record from effective_offset_c + per-device written/held
    setpoints only. Exposure must be derived from the per-device setpoints, not the
    conservative minimum."""
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = target + (3.0 if heating else -4.0)
    bd = BoostDispatchRecord(
        decision_id="x", boost_applied_c=(0.5 if dt is DecisionType.BOOST else 0.0),
        baseline_setpoint_c=target, final_setpoint_c=setp, dispatch_status=status,
        effective_offset_c=effective, effective_setpoints=tuple(written),
        held_effective_setpoints=tuple(held), targets_total=1)
    inp = RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK), window_open=window,
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(
            decision_type=dt, target_c=target, trv_setpoint_c=setp,
            boost_active=(dt is DecisionType.BOOST)),
        boost_dispatch=bd)
    return rt.run_cycle(inp)


class TestN1MixedTrvExposure:
    def test_regression_written_above_held_at_baseline_min_zero_not_baseline(self):
        # N1 REGRESSION PROOF (version-agnostic): device A written above baseline
        # (22.0, offset 1.0), device B held at baseline (21.0, offset 0.0) -> the
        # conservative effective MINIMUM is 0.0, but a device WAS boosted. The episode
        # must be exposed and NOT stored as a clean baseline. Against 5448096 (exposure
        # computed from the minimum) this stored a clean baseline (size 1).
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        for i, t in enumerate(_TEMPS):
            if i == 2:
                _vcycle(rt, i, t, dt=DecisionType.BOOST, effective=0.0,
                        written=(22.0,), held=(21.0,), heating=(t < _TARGET))
            else:
                _vcycle(rt, i, t, heating=(t < _TARGET))
        assert zr.baseline_store.size == 0
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) >= 1

    def test_mixed_then_unboosted_cycles_not_baseline(self):
        # Same mixed boost, then purely unboosted cycles before REACHED. The boosted
        # record is retained (relevance uses the authoritative exposure) so the
        # episode is still excluded.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        _vcycle(rt, 0, 18.0); _vcycle(rt, 1, 19.0)
        _vcycle(rt, 2, 19.5, dt=DecisionType.BOOST, effective=0.0, written=(22.0,), held=(21.0,))
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _vcycle(rt, i, t, heating=(t < _TARGET))
        assert zr.baseline_store.size == 0

    def test_mixed_exposure_across_restart_not_baseline(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        _vcycle(rt, 0, 18.0); _vcycle(rt, 1, 19.0)
        _vcycle(rt, 2, 19.5, dt=DecisionType.BOOST, effective=0.0, written=(22.0,), held=(21.0,))
        payload = rt._build_payload()
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _vcycle(rt2, i, t, heating=(t < _TARGET))
        assert rt2._zone(_ZONE).baseline_store.size == 0

    def test_clean_control_effective_zero_no_device_boosted_stores_baseline(self):
        # Control: effective 0.0 AND no device above baseline (all at baseline) -> a
        # genuinely clean non-boost episode -> baseline IS stored.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        for i, t in enumerate(_TEMPS):
            _vcycle(rt, i, t, heating=(t < _TARGET))     # NORMAL, no written/held above base
        assert zr.baseline_store.size == 1

    def test_no_adaptive_update_when_effective_min_zero(self):
        # N1 req 6: a zero conservative effective offset must NOT generate adaptive
        # Boost evidence, even though the episode is exposed (baseline-excluded).
        bd = BoostDispatchRecord(
            decision_id="d1", boost_applied_c=0.5, baseline_setpoint_c=21.0,
            final_setpoint_c=22.0, dispatch_status="fully_succeeded",
            effective_offset_c=0.0, effective_setpoints=(22.0,),
            held_effective_setpoints=(21.0,), any_device_boosted=True, targets_total=1)
        ev = EvidenceMaterializer().materialize(_snap(bd))
        assert ev.boost_context is None                 # no adaptive magnitude (min 0)
        from custom_components.thermosmart.learning.runtime.capture import (
            record_any_device_boosted)
        assert record_any_device_boosted(bd) is True     # but exposed -> no clean baseline


# ── N2: malformed persisted effective offset ─────────────────────────────────

class TestN2MalformedOffset:
    def test_malformed_effective_offset_normalized_on_restore(self):
        from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord as BR
        d = _bd("fully_succeeded", approved=0.5).to_dict()
        d["effective_offset_c"] = "x"               # malformed
        d["boost_applied_c"] = "nan"
        d["held_effective_setpoints"] = [21.0, "bad", float("inf")]
        rec = BR.from_dict(d)
        assert rec.effective_offset_c is None           # coerced, no crash
        assert rec.boost_applied_c is None
        assert rec.held_effective_setpoints == (21.0,)  # bad/inf dropped

    def test_malformed_offset_does_not_crash_relevance_or_cycle(self):
        # A malformed record must not raise in relevance evaluation or a full cycle.
        from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord as BR
        d = _bd("fully_succeeded", approved=0.5).to_dict()
        d["effective_offset_c"] = "x"
        rec = BR.from_dict(d)
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        # relevance must not raise
        assert rt._dispatch_boost_relevance(rec) >= 0.0
        # and a full learning cycle carrying it must not raise
        inp = RuntimeCycleInput(
            zone_id=_ZONE, ts=T0.isoformat(), target_c=_TARGET, trv_setpoint_c=21.5,
            controller_demand=True, indoor_temp=Measurement(19.0, DataQuality.OK),
            schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=_TARGET),
            controller_decision=ControllerDecisionInput(
                decision_type=DecisionType.BOOST, target_c=_TARGET, trv_setpoint_c=21.5,
                boost_active=True),
            boost_dispatch=rec)
        rt.run_cycle(inp)      # must not raise
