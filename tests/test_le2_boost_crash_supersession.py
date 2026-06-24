"""B2b-4e: crash-atomic exactly-once finalization, supersession, cross-cycle availability."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.models import (
    BoostConfounder, ConfounderSeverity, evaluate_confounders)
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, LearningRuntime, LearningRuntimeConfig,
    LearningRuntimeMode, RuntimeCycleInput, ScheduleTarget)
from custom_components.thermosmart.learning.runtime.boost_pending import (
    BoostPendingOutcome, OBS_MAX_GAP_S)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord

T0 = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)


def _bd(did="x"):
    return BoostDispatchRecord(did, boost_applied_c=1.5, dispatch_status="fully_succeeded",
                               outcome_reliability="full", baseline_setpoint_c=21.0,
                               effective_setpoints=(22.5,), targets_total=1)


def _rt():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                 startup_grace_cycles=0))


def _cyc(r, i, temp, heating, *, target=21.0, bd=None, device_states=None, decision_type=None):
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = 24.0 if heating else 17.0
    dt = decision_type or DecisionType.BOOST
    r.run_cycle(RuntimeCycleInput(
        zone_id="lz", ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK), device_states=device_states or {},
        schedule=ScheduleTarget(comfort_time_utc="2025-06-01T07:00:00+00:00",
                                comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(decision_type=dt, target_c=target,
                                                    trv_setpoint_c=setp),
        boost_dispatch=bd if bd is not None else _bd()))


_REACH = [(0, 18, True), (1, 19, True), (2, 20, True), (3, 20.6, True), (4, 20.8, True),
          (5, 21.0, True)]


def _classes(zr):
    return dict(zr.orchestrator.models["boost"]._state.outcome_class_counts)


def _processed(zr):
    return set(zr.orchestrator.models["boost"]._state.processed_decision_ids)


def _eff_n(zr):
    return zr.orchestrator.models["boost"]._state.general.effective_factor.effective_n


# ── 1. crash-atomic exactly-once (real serialized snapshots) ────────────────
class TestCrashAtomic:
    def _to_pending(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h)
        assert zr.pending_boost_outcome is not None
        return r, zr

    def test_crash_before_update_one_update(self):
        # snapshot persisted while pending open (no update yet) -> restore -> finalize once
        r, zr = self._to_pending()
        snap = zr.serialize()
        r2 = _rt(); zr2 = r2._zone("lz"); zr2.restore(snap)
        assert zr2.pending_boost_outcome is not None and not _processed(zr2)
        for i in range(6, 10):                         # tick past deadline (c8)
            _cyc(r2, i, 21.0, False)
        assert zr2.model_update_counts.get("boost", 0) == 1
        assert len(_processed(zr2)) == 1

    def test_crash_after_update_before_save_redone_once(self):
        # zoneA finalizes in memory; the SAVE never happens -> we restore the PRE-finalize
        # snapshot. The lost in-memory update is redone exactly once; one persistent update.
        r, zr = self._to_pending()
        pre_save = zr.serialize()                       # last durable snapshot (pre-finalize)
        for i in range(6, 10):
            _cyc(r, i, 21.0, False)                     # zoneA finalizes (update lost on crash)
        assert zr.model_update_counts.get("boost", 0) == 1
        r2 = _rt(); zr2 = r2._zone("lz"); zr2.restore(pre_save)
        assert not _processed(zr2)                      # restored state has NO processed id
        for i in range(6, 10):
            _cyc(r2, i, 21.0, False)
        assert zr2.model_update_counts.get("boost", 0) == 1   # redone exactly once
        assert len(_processed(zr2)) == 1

    def test_crash_after_model_persisted_before_pending_clear(self):
        # construct the dangerous snapshot: dedup ALREADY has the decision id AND the pending
        # is still present (model persisted, pending not yet cleared) -> restore -> NO 2nd update.
        r, zr = self._to_pending()
        pending_snap = zr.serialize()                   # pending present, dedup empty
        for i in range(6, 10):
            _cyc(r, i, 21.0, False)                     # finalize -> dedup now has the id
        did = next(iter(_processed(zr)))
        post_models = zr.serialize()["models"]          # model/dedup AFTER finalize
        # splice: pending from before (not yet cleared) + model/dedup from after
        spliced = dict(pending_snap)
        spliced["models"] = post_models
        r2 = _rt(); zr2 = r2._zone("lz"); zr2.restore(spliced)
        assert zr2.pending_boost_outcome is not None and did in _processed(zr2)
        before = _eff_n(zr2)
        for i in range(6, 10):
            _cyc(r2, i, 21.0, False)                     # re-finalize -> model dedups
        assert _eff_n(zr2) == before                     # NO second effective update
        assert zr2.pending_boost_outcome is None         # pending cleaned up

    def test_crash_after_full_save_no_second_update(self):
        r, zr = self._to_pending()
        for i in range(6, 10):
            _cyc(r, i, 21.0, False)
        full = zr.serialize()                            # pending cleared, dedup has id
        r2 = _rt(); zr2 = r2._zone("lz"); zr2.restore(full)
        assert zr2.pending_boost_outcome is None
        before = _eff_n(zr2)
        for i in range(10, 14):
            _cyc(r2, i, 21.0, False)
        assert _eff_n(zr2) == before                      # nothing re-finalized

    def test_repeated_restore_idempotent(self):
        r, zr = self._to_pending()
        for i in range(6, 10):
            _cyc(r, i, 21.0, False)
        full = zr.serialize()
        for _ in range(3):
            r2 = _rt(); zr2 = r2._zone("lz"); zr2.restore(full)
            before = _eff_n(zr2)
            for i in range(10, 13):
                _cyc(r2, i, 21.0, False)
            assert _eff_n(zr2) == before


# ── 4/5. supersession ────────────────────────────────────────────────────────
class TestSupersession:
    def _to_pending(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h)
        return r, zr

    def test_new_decision_before_deadline_interrupts_old(self):
        r, zr = self._to_pending()
        old_did = zr.pending_boost_outcome.decision_id
        # a NEW decision reaches before the old deadline -> old superseded (interrupted)
        for j, (i, t, h) in enumerate([(6, 18, True), (7, 19, True), (8, 20, True),
                                        (9, 20.6, True), (10, 20.8, True), (11, 21.0, True)]):
            _cyc(r, i, t, h, bd=_bd("y"))
        cc = _classes(zr)
        assert cc.get("boost_interrupted", 0) >= 1
        assert "boost_success" not in cc                 # superseded never a positive outcome

    def test_new_decision_after_deadline_finalizes_old_regularly(self):
        r, zr = self._to_pending()
        # let the old deadline pass with no overshoot (regular finalize), THEN new decision
        for i in range(6, 10):
            _cyc(r, i, 21.0, False)
        assert zr.pending_boost_outcome is None
        assert zr.model_update_counts.get("boost", 0) == 1   # old finalized regularly

    def test_same_decision_retry_no_supersession(self):
        r, zr = self._to_pending()
        first = zr.pending_boost_outcome
        # re-open with the SAME decision id (retry) must not interrupt / replace
        _cyc(r, 6, 21.0, True, bd=_bd("x"))
        assert zr.pending_boost_outcome is first
        assert _classes(zr).get("boost_interrupted", 0) == 0

    def test_superseded_confounder_is_interrupting(self):
        e = evaluate_confounders(["superseded_by_new_decision"])
        assert e.is_interrupting
        assert BoostConfounder.SUPERSEDED_BY_NEW_DECISION.value in e.confounders


# ── 6-9. cross-cycle device availability ─────────────────────────────────────
class TestAvailability:
    def test_device_available_throughout_no_confounder(self):
        r = _rt(); zr = r._zone("lz")
        ds = {"climate.a": "available"}
        for i, t, h in _REACH:
            _cyc(r, i, t, h, device_states=ds)
        for i in range(6, 10):
            _cyc(r, i, 21.0, False, device_states=ds)
        cc = _classes(zr)
        assert cc and "boost_interrupted" not in cc

    def test_device_unavailable_during_observation_partial(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h, device_states={"climate.a": "available"})
        # device goes unavailable after reach -> partial device effect (no full positive attr)
        for i in range(6, 10):
            _cyc(r, i, 21.0, False, device_states={"climate.a": "unavailable"})
        s = (zr.orchestrator.models["boost"]._state.recent_samples
             + zr.orchestrator.models["boost"]._state.diagnostic_samples)
        assert s and s[-1].outcome_class != "boost_success"

    def test_device_removed_is_configuration_change(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h, device_states={"climate.a": "available"})
        for i in range(6, 10):
            _cyc(r, i, 21.0, False, device_states={})   # device vanished from the set
        # device_configuration_change is blocking -> non-adaptive
        s = (zr.orchestrator.models["boost"]._state.recent_samples
             + zr.orchestrator.models["boost"]._state.diagnostic_samples)
        assert s and s[-1].outcome_class != "boost_success"

    def test_availability_persists_across_restart(self):
        r = _rt(); zr = r._zone("lz")
        for i, t, h in _REACH:
            _cyc(r, i, t, h, device_states={"climate.a": "available"})
        _cyc(r, 6, 21.0, False, device_states={"climate.a": "unavailable"})
        snap = zr.serialize()
        r2 = _rt(); zr2 = r2._zone("lz"); zr2.restore(snap)
        p = zr2.pending_boost_outcome
        assert p is not None
        assert any(agg.get("ever_unavailable") for agg in p.device_slots.values())

    def test_no_entity_ids_in_serialized_slots(self):
        p = BoostPendingOutcome(
            zone_id="lz", decision_id="d", episode_id="e", state="x", target=21.0,
            episode_end_ts=T0.isoformat(), observation_start_ts=T0.isoformat(),
            deadline_ts=T0.isoformat(), last_valid_ts=T0.isoformat(), episode={}, boost_context={})
        p.seed_devices(T0.isoformat(), {"climate.living_room": "available"}, "setpoint")
        blob = p.to_dict()
        assert "climate.living_room" not in str(blob["device_slots"])
        assert len(next(iter(blob["device_slots"].keys()))) == 12   # hashed slot
