"""Phase B — Boost dispatch truth (D5a), decision attribution (D4) and
multi-device dispatch accounting (O1).

Real-coordinator integration tests (via DispatchingCoordinator with the real
``_apply_temperature`` actuator path and mocked HA services) for the dispatch
accounting and applied-offset semantics, plus focused unit tests for the two
authoritative boundaries the defects live at:

  * D4 — ``_decision_type`` / ``build_runtime_cycle_input`` read the authoritative
    ``learning_boost_adjusted`` flag (the former ``recommendation["boost_active"]``
    key never existed, so every boost cycle was misclassified).
  * D5a — a device already holding the requested (boosted) setpoint is accounted
    as HELD (status ``already_satisfied``, applied offset reflected), never as a
    new successful write nor as ``not_attempted``, and never forms a NEW boost
    learning outcome on its own (no silent eligibility increase).

Includes explicit regression tests that FAIL under the pre-Phase-B D4/D5a code.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.export import _learning_application_export
from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.privacy import scan_payload
from custom_components.thermosmart.learning.runtime.capture import (
    BoostDispatchRecord,
    CapturedSnapshot,
    DecisionType,
)
from custom_components.thermosmart.learning.runtime.evidence import EvidenceMaterializer
from custom_components.thermosmart.learning.runtime.ha_integration import (
    _decision_type,
    build_runtime_cycle_input,
)
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from custom_components.thermosmart.trv_control import _DispatchStats
from tests.helpers_ha_runtime import (
    FakeStore,
    attach_shadow,
    make_dispatching_coordinator,
)

_TS = "2026-01-12T06:00:00+00:00"


async def _boost_cycle(*, indoor="19.5", climate_entities=None, climate_states=None,
                       service_side_effect=None, active=True, cfg=None,
                       mode=LearningRuntimeMode.CONTROL):
    coord = make_dispatching_coordinator(
        {"schedule_enabled": False, "boost_bootstrap_prior_c": 1.0, **(cfg or {})},
        indoor=indoor, climate_entities=climate_entities, climate_states=climate_states,
        service_side_effect=service_side_effect)
    coord.set_active_control(active)
    shadow = attach_shadow(coord, mode=mode, store=FakeStore())
    await shadow.async_setup()
    coord.data = await coord._async_update_data()
    return coord, _learning_application_export(coord)["boost"]


# ── D5a / O1: real-cycle dispatch accounting ──────────────────────────────────

class TestDispatchAccounting:
    async def test_1_single_trv_already_holding_boost_target(self):
        # Device already at the boosted setpoint (28.5°C) → held, no write.
        coord, b = await _boost_cycle(climate_states={"climate.trv_test": {"temperature": 28.5}})
        assert b["dispatch_status"] == "already_satisfied"
        assert b["dispatch_targets_held"] == 1
        assert b["dispatch_targets_total"] == 0            # no new write attempted
        assert b["dispatch_targets_considered"] == 1
        assert b["dispatch_write_issued"] is False
        assert b["service_call_succeeded"] is None
        assert b["requested_state_already_satisfied"] is True
        assert b["approved_offset_c"] > 0
        # D5a: the held boost IS applied — confirmed offset > 0 (was 0.0 pre-fix).
        assert b["dispatch_confirmed_offset_c"] > 0
        assert b["dispatch_confirmed_offset_c"] == coord.data["zone"]["applied_boost_offset_c"]
        assert b["outcome_eligible"] is False              # no new actuation evidence
        assert coord._service_calls == []

    async def test_2_single_trv_new_successful_write(self):
        coord, b = await _boost_cycle(climate_states={"climate.trv_test": {"temperature": 21.0}})
        assert b["dispatch_status"] == "fully_succeeded"
        assert b["dispatch_targets_total"] == 1
        assert b["dispatch_targets_succeeded"] == 1
        assert b["dispatch_targets_held"] == 0
        assert b["dispatch_write_issued"] is True
        assert b["service_call_succeeded"] is True
        assert b["approved_offset_c"] > 0
        assert b["dispatch_confirmed_offset_c"] > 0
        assert b["outcome_eligible"] is True
        assert len(coord._service_calls) == 1

    async def test_3_service_failure(self):
        coord, b = await _boost_cycle(
            climate_states={"climate.trv_test": {"temperature": 21.0}},
            service_side_effect=[IOError("boom")])
        assert b["dispatch_status"] == "failed"
        assert b["dispatch_targets_failed"] == 1
        assert b["dispatch_targets_succeeded"] == 0
        assert b["service_call_succeeded"] is False
        assert b["approved_offset_c"] > 0
        # failed dispatch → confirmed offset 0.0 (nothing applied)
        assert b["dispatch_confirmed_offset_c"] == 0.0
        assert b["outcome_eligible"] is False

    async def test_4_two_trvs_one_held_one_written(self):
        coord, b = await _boost_cycle(
            climate_entities=["climate.a", "climate.b"],
            climate_states={"climate.a": {"temperature": 28.5},   # held
                            "climate.b": {"temperature": 21.0}})   # needs write
        assert b["dispatch_targets_considered"] == 2
        assert b["dispatch_targets_held"] == 1
        assert b["dispatch_targets_total"] == 1                # one new write
        assert b["dispatch_targets_succeeded"] == 1
        assert b["dispatch_status"] == "fully_succeeded"       # the write succeeded
        assert b["dispatch_write_issued"] is True
        assert len(coord._service_calls) == 1                  # only the non-held device

    async def test_5_two_trvs_one_success_one_failed(self):
        coord, b = await _boost_cycle(
            climate_entities=["climate.a", "climate.b"],
            climate_states={"climate.a": {"temperature": 21.0},
                            "climate.b": {"temperature": 21.0}},
            service_side_effect=[None, IOError("boom")])
        assert b["dispatch_status"] == "partially_succeeded"
        assert b["dispatch_targets_total"] == 2
        assert b["dispatch_targets_succeeded"] == 1
        assert b["dispatch_targets_failed"] == 1
        assert b["dispatch_targets_held"] == 0
        assert b["service_call_succeeded"] is False            # not fully succeeded
        assert b["dispatch_confirmed_offset_c"] == 0.0         # partial → 0 applied truth

    async def test_8_boost_approved_but_dispatch_unavailable(self):
        # Active control OFF: boost may be evaluated but the dispatch path does not
        # run — approved-but-not-dispatched must not surface as applied.
        coord, b = await _boost_cycle(active=False, mode=LearningRuntimeMode.SHADOW)
        assert b["dispatch_attempted"] is False
        assert b["dispatch_status"] == "not_attempted"
        assert b["dispatch_targets_total"] == 0
        assert b["dispatch_targets_held"] == 0
        assert b["dispatch_confirmed_offset_c"] == 0.0
        assert b["outcome_eligible"] is False
        assert coord._service_calls == []

    async def test_9_normal_heating_without_boost_not_mislabelled(self):
        # Cold room, NO boost prior: a generic schedule/TPI setpoint write occurs,
        # but it must not be labelled a boost.
        coord, b = await _boost_cycle(cfg={"boost_bootstrap_prior_c": 0.0}, indoor="18.0",
                                      climate_states={"climate.trv_test": {"temperature": 21.0}})
        assert b["dispatch_write_issued"] is True
        assert b["lifecycle_approved"] is False
        assert b["approved_offset_c"] in (None, 0.0)
        assert b["dispatch_confirmed_offset_c"] == 0.0
        assert len(coord._service_calls) == 1

    async def test_excluded_devices_are_accounted(self):
        # A device with an inactive profile is counted as excluded (privacy-safe
        # reason), not silently dropped. Exercised through the real
        # _apply_temperature actuator path.
        from custom_components.thermosmart.device_profiles import DeviceProfile
        coord = make_dispatching_coordinator(
            {"schedule_enabled": False}, indoor="18.0",
            climate_states={"climate.trv_test": {"temperature": 21.0}})
        coord.set_active_control(True)
        coord._device_profiles = {
            "climate.trv_test": DeviceProfile(identifier="t", display_name="T", is_active=False)}
        stats = await coord._apply_temperature(
            {"climate_entities": ["climate.trv_test"]},
            {"adjusted_target": 21.0, "trv_setpoint": 24.0})
        assert stats.targets_excluded == 1
        assert "profile_inactive" in stats.excluded_reasons
        assert stats.targets_total == 0 and stats.targets_held == 0
        assert stats.status == "not_attempted"              # excluded-only → nothing actionable
        assert coord._service_calls == []


# ── D5a / O1: _DispatchStats unit semantics ───────────────────────────────────

class TestDispatchStats:
    def test_held_is_already_satisfied_not_not_attempted(self):
        s = _DispatchStats()
        s.record_held(effective_c=22.0, entity_id="climate.x")
        assert s.status == "already_satisfied"
        assert s.targets_held == 1
        assert s.targets_total == 0
        assert s.considered == 1

    def test_empty_is_not_attempted(self):
        assert _DispatchStats().status == "not_attempted"

    def test_held_plus_write_is_fully_succeeded(self):
        s = _DispatchStats()
        s.record_held(effective_c=22.0, entity_id="climate.a")
        s.record(None, effective_c=23.0, entity_id="climate.b")
        assert s.status == "fully_succeeded"
        assert s.considered == 2
        assert s.targets_held == 1
        assert s.targets_total == 1

    def test_excluded_only_is_not_attempted(self):
        s = _DispatchStats()
        s.record_excluded("unsupported")
        assert s.status == "not_attempted"
        assert s.targets_excluded == 1
        assert s.considered == 0

    def test_merge_combines_held_and_excluded(self):
        a = _DispatchStats(); a.record_held(effective_c=22.0, entity_id="x")
        bstat = _DispatchStats(); bstat.record_excluded("no_state")
        m = a.merge(bstat)
        assert m.targets_held == 1 and m.targets_excluded == 1
        assert m.status == "already_satisfied"


# ── D4: decision-type / controller boost_active attribution ───────────────────

class TestDecisionAttribution:
    def test_boost_applied_classified_as_boost(self):
        rec = {"learning_boost_adjusted": True, "preheat_minutes": 0,
               "effective_target": 21.0, "trv_setpoint": 22.5, "current_temp": 19.0}
        assert _decision_type(rec) is DecisionType.BOOST
        inp = build_runtime_cycle_input("z", rec, ts_iso=_TS)
        assert inp.controller_decision.boost_active is True
        assert inp.controller_decision.decision_type is DecisionType.BOOST

    def test_rejected_boost_is_not_boost(self):
        # Boost candidate present but NOT applied → not a BOOST decision.
        rec = {"learning_boost_adjusted": False, "_boost_rejection_reason": "guard_rejected",
               "preheat_minutes": 0, "effective_target": 21.0, "trv_setpoint": 21.0,
               "current_temp": 19.0}
        assert _decision_type(rec) is DecisionType.NORMAL
        inp = build_runtime_cycle_input("z", rec, ts_iso=_TS)
        assert inp.controller_decision.boost_active is False

    def test_preheat_without_boost_is_preheat(self):
        rec = {"learning_boost_adjusted": False, "preheat_minutes": 40.0,
               "effective_target": 21.0, "trv_setpoint": 24.0, "current_temp": 18.0}
        assert _decision_type(rec) is DecisionType.PREHEAT
        inp = build_runtime_cycle_input("z", rec, ts_iso=_TS)
        assert inp.controller_decision.boost_active is False
        assert inp.controller_decision.preheat_active is True

    def test_normal_heating_is_normal(self):
        rec = {"learning_boost_adjusted": False, "preheat_minutes": 0,
               "effective_target": 21.0, "trv_setpoint": 21.0, "current_temp": 20.0}
        assert _decision_type(rec) is DecisionType.NORMAL

    def test_D4_regression_nonexistent_boost_active_key_is_not_used(self):
        # REGRESSION: under the old code `_decision_type` read rec["boost_active"],
        # which never exists — so a real boost cycle was misclassified as NORMAL.
        # The authoritative flag is learning_boost_adjusted. A recommendation that
        # carries ONLY the old (never-set) key must NOT classify as BOOST, and one
        # carrying the authoritative flag MUST.
        rec_old_key_only = {"boost_active": True, "learning_boost_adjusted": False,
                            "preheat_minutes": 0}
        assert _decision_type(rec_old_key_only) is not DecisionType.BOOST
        rec_authoritative = {"learning_boost_adjusted": True, "preheat_minutes": 0}
        assert _decision_type(rec_authoritative) is DecisionType.BOOST


# ── D5a: learning-eligibility boundary (EvidenceMaterializer) ─────────────────

def _snap(decision_type=DecisionType.BOOST, boost_dispatch=None, **kw):
    base = dict(
        zone_id="z", ts=_TS, decision_id="z:BOOST:1",
        target_c=21.0, trv_setpoint_c=22.5,
        indoor_temp=Measurement(19.0, DataQuality.OK), indoor_temp_quality=DataQuality.OK,
        decision_type=decision_type, controller_demand=True, tpi_on=True,
        boost_active=True, preheat_active=False, fallback_active=False,
        window_open=False, heating_failure=False,
        schedule_comfort_time_utc=None, schedule_comfort_temperature_c=None,
        outdoor_temp=None, valve_opening=None, hvac_action="heating", forecast_high=None,
        boost_dispatch=boost_dispatch)
    base.update(kw)
    return CapturedSnapshot(**base)


def _bd(status, applied=1.0, **kw):
    # Model the live coordinator: the POST-dispatch effective offset (F1) equals the
    # applied offset for these single-device records, and the authoritative "any device
    # boosted" exposure flag (N1) follows from a positive offset on an exposed status.
    _exposed = status in ("fully_succeeded", "already_satisfied", "partially_succeeded")
    base = dict(decision_id="z:BOOST:1", boost_candidate_c=1.5, boost_applied_c=applied,
                baseline_setpoint_c=21.0, final_setpoint_c=22.5, dispatch_status=status,
                effective_offset_c=applied,
                any_device_boosted=bool((applied or 0.0) > 0.0 and _exposed))
    base.update(kw)
    return BoostDispatchRecord(**base)


class TestEligibilityBoundary:
    def test_fully_succeeded_forms_boost_outcome(self):
        ev = EvidenceMaterializer().materialize(
            _snap(boost_dispatch=_bd("fully_succeeded", applied=1.0)))
        assert ev.boost_context is not None
        assert "boost_authoritative" in ev.notes

    def test_already_satisfied_forms_no_boost_outcome(self):
        # D5a: a held cycle (even with applied > 0) must NOT create a new boost
        # outcome — no silent eligibility increase.
        ev = EvidenceMaterializer().materialize(
            _snap(boost_dispatch=_bd("already_satisfied", applied=1.0)))
        assert ev.boost_context is None
        assert "boost_held_no_new_dispatch" in ev.notes

    def test_failed_forms_no_boost_outcome(self):
        ev = EvidenceMaterializer().materialize(
            _snap(boost_dispatch=_bd("failed", applied=0.0)))
        assert ev.boost_context is None
        assert "boost_dispatch_failed" in ev.notes

    def test_not_attempted_forms_no_boost_outcome(self):
        ev = EvidenceMaterializer().materialize(
            _snap(boost_dispatch=_bd("not_attempted", applied=0.0)))
        assert ev.boost_context is None
        assert "boost_not_attempted" in ev.notes


# ── multi-zone isolation, privacy, diagnostics consistency ────────────────────

class TestIsolationAndPrivacy:
    async def test_13_multi_zone_isolation(self):
        held_coord, held_b = await _boost_cycle(
            climate_states={"climate.trv_test": {"temperature": 28.5}})
        write_coord, write_b = await _boost_cycle(
            climate_states={"climate.trv_test": {"temperature": 21.0}})
        assert held_b["dispatch_status"] == "already_satisfied"
        assert write_b["dispatch_status"] == "fully_succeeded"
        assert held_coord._last_live_decision is not write_coord._last_live_decision

    async def test_14_privacy_and_diagnostics_consistency(self):
        coord, b = await _boost_cycle(
            climate_entities=["climate.a", "climate.b"],
            climate_states={"climate.a": {"temperature": 28.5},
                            "climate.b": {"temperature": 21.0}})
        out = _learning_application_export(coord)
        assert scan_payload(out) == []
        # excluded reasons are bounded enums, never entity IDs
        for r in b["dispatch_excluded_reasons"]:
            assert r in ("profile_inactive", "unsupported", "no_state")
        # considered == held + attempted (internal consistency)
        assert b["dispatch_targets_considered"] == (
            b["dispatch_targets_held"] + b["dispatch_targets_total"])

    async def test_D5a_regression_held_boost_applied_offset(self):
        # REGRESSION: under the old code a held boost reported
        # applied_boost_offset_c == 0.0 and dispatch_status == "not_attempted".
        # Phase B reports the real held offset and "already_satisfied".
        coord, b = await _boost_cycle(
            climate_states={"climate.trv_test": {"temperature": 28.5}})
        assert b["dispatch_status"] == "already_satisfied"      # was "not_attempted"
        assert coord.data["zone"]["applied_boost_offset_c"] > 0  # was 0.0


# ── M1: held-offset accuracy (actual observed setpoint, not the request) ──────

class TestHeldOffsetAccuracy:
    async def test_held_below_target_within_tolerance_uses_actual(self):
        # Device holds 28.3 °C while the requested boosted setpoint is 28.5 °C
        # (within the default 0.5 tolerance). The confirmed offset must reflect the
        # ACTUAL held value (28.3 - 28.0 baseline = 0.3), NOT the requested 0.5.
        coord, b = await _boost_cycle(climate_states={"climate.trv_test": {"temperature": 28.3}})
        assert b["dispatch_status"] == "already_satisfied"
        assert b["approved_offset_c"] == pytest.approx(0.5)          # what was requested
        assert b["dispatch_confirmed_offset_c"] == pytest.approx(0.3, abs=0.051)  # what is held
        assert b["dispatch_confirmed_offset_c"] < b["approved_offset_c"]

    async def test_held_exactly_at_target_confirms_full_offset(self):
        coord, b = await _boost_cycle(climate_states={"climate.trv_test": {"temperature": 28.5}})
        assert b["dispatch_confirmed_offset_c"] == pytest.approx(0.5)

    async def test_held_below_baseline_reports_no_confirmed_offset(self):
        # Non-default tolerance 1.0: device holds 27.8 °C — within tolerance of the
        # 28.5 request, but BELOW the 28.0 baseline. Observation does not support a
        # confirmed positive boost, so the confirmed offset is 0.0 (uncertainty
        # reported, not a fabricated 0.5).
        coord, b = await _boost_cycle(cfg={"temp_tolerance": 1.0},
                                      climate_states={"climate.trv_test": {"temperature": 27.8}})
        assert b["dispatch_status"] == "already_satisfied"
        assert b["approved_offset_c"] == pytest.approx(0.5)
        assert b["dispatch_confirmed_offset_c"] == pytest.approx(0.0)


# ── M4: mixed held/written conservative minimum includes held devices ─────────

class TestMixedHeldWrittenAggregation:
    async def test_mixed_zone_minimum_includes_held_device(self):
        # TRV A already holds 28.2 (offset 0.2); TRV B is newly written to 28.5
        # (offset 0.5). The conservative zone offset is the MINIMUM across both —
        # 0.2 — not 0.5. Before M4 the held device was ignored and the offset was
        # overstated as 0.5.
        coord, b = await _boost_cycle(
            climate_entities=["climate.a", "climate.b"],
            climate_states={"climate.a": {"temperature": 28.2},
                            "climate.b": {"temperature": 21.0}})
        assert b["dispatch_status"] == "fully_succeeded"
        assert b["dispatch_targets_held"] == 1
        assert b["dispatch_targets_total"] == 1
        assert b["dispatch_confirmed_offset_c"] == pytest.approx(0.2, abs=0.051)
        assert b["dispatch_confirmed_offset_c"] < b["approved_offset_c"]


# ── M2: legacy fallback cannot produce adaptive evidence ──────────────────────

class TestLegacyFallbackSafety:
    def test_boost_decision_without_dispatch_record_forms_no_context(self):
        # D4 makes DecisionType.BOOST reachable. With NO authoritative dispatch
        # record we cannot confirm the boost was applied/succeeded, so the former
        # setpoint-heuristic fallback is disabled: NO boost context is formed.
        ev = EvidenceMaterializer().materialize(
            _snap(decision_type=DecisionType.BOOST, boost_dispatch=None,
                  trv_setpoint_c=25.0, target_c=21.0))     # large setpoint>target
        assert ev.boost_context is None
        assert "boost_unconfirmed_no_dispatch_record" in ev.notes

    def test_failed_boost_record_forms_no_context(self):
        ev = EvidenceMaterializer().materialize(
            _snap(decision_type=DecisionType.BOOST, boost_dispatch=_bd("failed", applied=0.0)))
        assert ev.boost_context is None

    def test_not_attempted_boost_record_forms_no_context(self):
        ev = EvidenceMaterializer().materialize(
            _snap(decision_type=DecisionType.BOOST, boost_dispatch=_bd("not_attempted", applied=0.0)))
        assert ev.boost_context is None


# ── M3: accounting scope / no double-count across setpoint+valve ──────────────

class TestAccountingScope:
    def test_considered_is_setpoint_scoped(self):
        # The exported considered/held counts come from the SETPOINT stats only
        # (the coordinator uses _sp_stats, never the merged stats, for these), so a
        # valve write for the same TRV cannot inflate them.
        sp = _DispatchStats()
        sp.record_held(effective_c=22.0, entity_id="climate.x")   # held via setpoint
        sp.record(None, effective_c=23.0, entity_id="climate.y")  # written via setpoint
        assert sp.considered == 2
        assert sp.targets_held == 1

    def test_merge_sums_each_count_once(self):
        sp = _DispatchStats()
        sp.record_held(effective_c=22.0, entity_id="climate.x")
        sp.record_excluded("unsupported")
        vv = _DispatchStats()
        vv.record(None)                                           # a valve write
        m = sp.merge(vv)
        # held/excluded come only from the path that recorded them (no doubling);
        # total is the sum of both paths' write attempts.
        assert m.targets_held == 1
        assert m.targets_excluded == 1
        assert m.targets_total == sp.targets_total + vv.targets_total
