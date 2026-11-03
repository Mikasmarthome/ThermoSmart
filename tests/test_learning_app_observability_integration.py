"""Phase A — I4: REAL coordinator-cycle integration tests for the learning
APPLICATION observability diagnostic and the restore/health path.

Unlike test_learning_app_observability.py (which feeds manually constructed
LiveDecisionRecord objects), these drive the real coordinator decision + the
real ``_apply_temperature`` actuator path through ``DispatchingCoordinator``
(HA services mocked for success/failure/partial) and then read the diagnostic
back from the coordinator's own post-cycle state. They verify that the
diagnostic faithfully reflects what the live cycle actually did — dispatch
success/failure/partial, held-setpoint no-write, active-control off, learning
off, learned vs deterministic preheat — without the diagnostic itself mutating
any control, ledger or counter state, and that the restore path surfaces
store-load / corrupt-segment / schema-mismatch errors through RuntimeHealth.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.export import (
    _learning_application_export,
    _learning_health_data,
)
from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.privacy import scan_payload
from custom_components.thermosmart.learning.runtime.lifecycle import (
    LearningRuntimeMode,
    RUNTIME_SCHEMA_VERSION,
)
from tests.helpers import make_state
from tests.helpers_ha_runtime import (
    FakeStore,
    attach_shadow,
    inject_learned_thermal_state,
    make_dispatching_coordinator,
)

_START = datetime(2026, 1, 12, 6, 0, tzinfo=timezone.utc)


async def _run_boost_cycle(*, cfg=None, indoor="19.5", active=True,
                           climate_entities=None, service_side_effect=None,
                           climate_states=None, mode=LearningRuntimeMode.CONTROL):
    """Drive one real coordinator cycle and return (coord, diagnostic)."""
    coord = make_dispatching_coordinator(
        {"schedule_enabled": False, **(cfg or {})}, indoor=indoor,
        climate_entities=climate_entities, climate_states=climate_states,
        service_side_effect=service_side_effect)
    coord.set_active_control(active)
    shadow = attach_shadow(coord, mode=mode, store=FakeStore())
    await shadow.async_setup()
    coord.data = await coord._async_update_data()
    return coord, _learning_application_export(coord)


# ── I4: real boost dispatch cycles ────────────────────────────────────────────

class TestRealBoostDispatch:
    async def test_successful_dispatch_reports_confirmed_offset(self):
        coord, out = await _run_boost_cycle(cfg={"boost_bootstrap_prior_c": 1.0})
        b = out["boost"]
        assert b["available"] is True
        assert b["lifecycle_approved"] is True
        # approved (pre-dispatch) and coordinator-confirmed (post-dispatch) agree
        # here, and both equal the zone's authoritative applied offset.
        assert b["approved_offset_c"] == b["dispatch_confirmed_offset_c"]
        assert b["dispatch_confirmed_offset_c"] == coord.data["zone"]["applied_boost_offset_c"]
        assert b["dispatch_attempted"] is True
        assert b["dispatch_path"] == "setpoint"
        assert b["dispatch_status"] == "fully_succeeded"
        assert b["service_call_succeeded"] is True
        assert b["dispatch_write_issued"] is True
        assert b["dispatch_targets_failed"] == 0
        # effective per-device setpoint actually written is exposed
        assert b["dispatch_effective_setpoint_min_c"] is not None
        assert b["heating_effect_confirmed"] is None

    async def test_failed_dispatch_reports_failure_not_success(self):
        coord, out = await _run_boost_cycle(
            cfg={"boost_bootstrap_prior_c": 1.0},
            service_side_effect=[IOError("boom")])
        b = out["boost"]
        assert b["dispatch_attempted"] is True
        assert b["dispatch_status"] == "failed"
        assert b["service_call_succeeded"] is False          # attempted and failed, not null
        assert b["dispatch_targets_failed"] >= 1
        assert b["dispatch_targets_succeeded"] == 0
        assert b["dispatch_failure_reasons"]                 # populated, privacy-safe labels
        assert all(":" not in r or not r.startswith("climate.") for r in b["dispatch_failure_reasons"])
        assert b["heating_effect_confirmed"] is None

    async def test_partial_dispatch_is_distinguished(self):
        coord, out = await _run_boost_cycle(
            cfg={"boost_bootstrap_prior_c": 1.0},
            climate_entities=["climate.a", "climate.b"],
            service_side_effect=[None, IOError("boom")])
        b = out["boost"]
        assert b["dispatch_status"] == "partially_succeeded"
        assert b["service_call_succeeded"] is False          # not fully succeeded
        assert b["dispatch_targets_total"] == 2
        assert b["dispatch_targets_succeeded"] == 1
        assert b["dispatch_targets_failed"] == 1

    async def test_active_control_off_no_dispatch_attempted(self):
        coord, out = await _run_boost_cycle(
            cfg={"boost_bootstrap_prior_c": 1.0}, active=False,
            mode=LearningRuntimeMode.SHADOW)
        b = out["boost"]
        assert b["dispatch_attempted"] is False
        assert b["dispatch_status"] == "not_attempted"
        assert b["dispatch_targets_total"] == 0
        assert b["service_call_succeeded"] is None           # tri-state null: no call attempted
        assert b["dispatch_write_issued"] is False
        assert coord._service_calls == []                    # real path issued nothing

    async def test_learning_disabled_is_deterministic(self):
        # Learning off: boost bootstrap prior is ignored, no learned boost applied.
        coord, out = await _run_boost_cycle(
            cfg={"learning_enabled": False, "boost_bootstrap_prior_c": 1.0})
        b = out["boost"]
        # A write may still occur (deterministic schedule), but no learned boost.
        assert b["lifecycle_approved"] is False
        assert b["approved_offset_c"] in (None, 0.0)
        assert out["tpi"]["coef_source"] is not None


# ── I4: real preheat cycles (learned vs deterministic) ────────────────────────

async def _run_preheat(learned: bool):
    clock = FakeClock(_START)
    coord = make_dispatching_coordinator({"schedule_enabled": False}, indoor="17.5")
    coord._clock = clock
    coord.set_active_control(True)
    shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore(),
                           clock=clock)
    await shadow.async_setup()
    if learned:
        inject_learned_thermal_state(shadow)
    coord._resolve_base_target = MagicMock(return_value=17.0)
    coord._minutes_until_next_comfort = lambda cfg: 45.0
    _orig = coord.hass.states.get.side_effect

    def _get(eid):
        if eid == "sensor.test_temp":
            return make_state("17.5")
        return _orig(eid)

    coord.hass.states.get = MagicMock(side_effect=_get)
    zone = {}
    for _ in range(3):           # let last_predictions populate across cycles
        zone = (await coord._async_update_data())["zone"]
        clock.advance(300)
    coord.data = {"zone": zone}
    return zone, _learning_application_export(coord)["preheat"]


class TestRealPreheat:
    async def test_learned_preheat_reported_as_learned(self):
        zone, ph = await _run_preheat(learned=True)
        assert zone["preheat_status"] in ("valid", "valid_hl_prior")
        assert ph["source"] == "learned"
        assert ph["status"] == zone["preheat_status"]        # authoritative passthrough
        # baseline is the coordinator's OWN same-cycle baseline, not recomputed,
        # and the learned selection differs from it (not a mere echo).
        assert ph["baseline_minutes"] == zone["preheat_baseline_minutes"]
        assert ph["selected_minutes"] == zone["preheat_minutes"]
        assert ph["selected_minutes"] != ph["baseline_minutes"]

    async def test_deterministic_preheat_reported_as_baseline(self):
        zone, ph = await _run_preheat(learned=False)
        assert zone["preheat_status"] == "deterministic_baseline"
        assert ph["source"] == "deterministic_baseline"
        assert ph["status"] == "deterministic_baseline"
        assert ph["baseline_minutes"] == zone["preheat_baseline_minutes"]
        # deterministic selection equals the baseline it fell back to
        assert ph["selected_minutes"] == ph["baseline_minutes"]


# ── I4: real TPI selection through a cycle ────────────────────────────────────

class TestRealTpi:
    async def test_coef_source_is_authoritative_from_real_cycle(self):
        coord, out = await _run_boost_cycle(cfg={"boost_bootstrap_prior_c": 1.0})
        tpi = out["tpi"]
        assert tpi["available"] is True
        assert tpi["coef_source"] == coord.data["zone"]["tpi_coef_source"]
        assert tpi["coef_int_selected"] == coord.data["zone"]["tpi_coef_int"]
        assert tpi["is_selection_only"] is True

    async def test_coef_step_limit_progression_across_cycles(self):
        # A learned zone's coefficients converge under the real step-limit guard
        # over successive cycles; the diagnostic reports each cycle's SELECTED
        # (post-step-limit) coefficient, matching the zone's own value.
        clock = FakeClock(_START)
        coord = make_dispatching_coordinator({"schedule_enabled": False}, indoor="20.4")
        coord._clock = clock
        coord.set_active_control(True)
        shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL,
                               store=FakeStore(), clock=clock)
        await shadow.async_setup()
        inject_learned_thermal_state(shadow)
        seen = []
        for _ in range(4):
            zone = (await coord._async_update_data())["zone"]
            coord.data = {"zone": zone}
            tpi = _learning_application_export(coord)["tpi"]
            assert tpi["coef_int_selected"] == zone["tpi_coef_int"]   # selected, post step-limit
            seen.append(tpi["coef_source"])
            clock.advance(300)
        assert any(s == "valid_learning" for s in seen)      # learning engaged at least once


# ── I4: diagnostics must not mutate control / ledger / counters ───────────────

class TestDiagnosticsAreInert:
    async def test_export_does_not_mutate_or_dispatch(self):
        coord, _ = await _run_boost_cycle(cfg={"boost_bootstrap_prior_c": 1.0})
        shadow = coord._learning_shadow
        before_health = _learning_health_data(coord)
        before_zone = dict(coord.data["zone"])
        before_calls = len(coord._service_calls)
        before_ldr = coord._last_live_decision

        # Repeated downloads are pure reads and byte-identical.
        out1 = _learning_application_export(coord)
        out2 = _learning_application_export(coord)
        assert out1 == out2

        after_health = _learning_health_data(coord)
        assert coord.data["zone"] == before_zone             # recommendation untouched
        assert len(coord._service_calls) == before_calls     # no new actuator write
        assert coord._last_live_decision is before_ldr       # LDR identity unchanged
        # ledger size, model-update counts and restore counters are unchanged
        for key in ("prediction_snapshots", "model_update_total", "model_update_counts",
                    "control_applied", "restore_errors"):
            assert after_health[key] == before_health[key]

    async def test_privacy_clean_on_real_cycle(self):
        coord, out = await _run_boost_cycle(
            cfg={"boost_bootstrap_prior_c": 1.0},
            climate_entities=["climate.a", "climate.b"],
            service_side_effect=[None, IOError("boom")])
        assert scan_payload(out) == []
        assert scan_payload(_learning_health_data(coord)) == []


# ── I4: multi-zone isolation at the diagnostic level ──────────────────────────

class TestMultiZoneIsolation:
    async def test_two_coordinators_report_independently(self):
        ok_coord, ok_out = await _run_boost_cycle(cfg={"boost_bootstrap_prior_c": 1.0})
        fail_coord, fail_out = await _run_boost_cycle(
            cfg={"boost_bootstrap_prior_c": 1.0},
            service_side_effect=[IOError("boom")])
        assert ok_out["boost"]["dispatch_status"] == "fully_succeeded"
        assert fail_out["boost"]["dispatch_status"] == "failed"
        assert ok_out["boost"]["service_call_succeeded"] is True
        assert fail_out["boost"]["service_call_succeeded"] is False
        # one coordinator's outcome never bleeds into the other's diagnostic
        assert ok_coord._last_live_decision is not fail_coord._last_live_decision


# ── I4: missing / unavailable runtime state through the real path ─────────────

class TestMissingRuntimeState:
    async def test_no_cycle_run_boost_unavailable(self):
        # Real coordinator constructed but no decision cycle has run yet:
        # there is no LiveDecisionRecord, so the boost block is unavailable
        # (never fabricated) while the overall export still succeeds.
        coord = make_dispatching_coordinator({"schedule_enabled": False}, indoor="19.0")
        assert getattr(coord, "_last_live_decision", None) is None
        out = _learning_application_export(coord)
        assert out["available"] is True
        assert out["boost"]["available"] is False
        assert out["boost"]["reason"] == "no_decision_record"

    async def test_cycle_without_shadow_still_coherent(self):
        # A real completed cycle with no learning shadow attached still yields a
        # coherent boost block (the coordinator builds its own baseline decision
        # record); no learned boost is applied.
        coord = make_dispatching_coordinator({"schedule_enabled": False}, indoor="19.0")
        coord.set_active_control(True)
        coord.data = await coord._async_update_data()
        b = _learning_application_export(coord)["boost"]
        assert b["available"] is True
        assert b["lifecycle_approved"] is False
        assert b["heating_effect_confirmed"] is None


# ── I4/I5: restore through the real store + health path ───────────────────────

async def _setup_runtime(store):
    coord = make_dispatching_coordinator({"schedule_enabled": False}, indoor="19.0")
    coord.set_active_control(True)
    shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=store)
    await shadow.async_setup()
    return coord, shadow._runtime


class TestRestoreHealthPath:
    async def test_clean_store_reports_zero_errors(self):
        coord, rt = await _setup_runtime(FakeStore(data=None))
        h = rt.health()
        assert h.restore_errors == 0
        assert h.restore_error_reasons == ()
        assert h.restore_schema_mismatch is False

    async def test_store_load_failure_is_surfaced(self):
        coord, rt = await _setup_runtime(FakeStore(fail_load=True))
        h = rt.health()
        assert h.restore_errors >= 1
        assert any(r.startswith("store:load_failed:") for r in h.restore_error_reasons)
        # the existing persistence warning is preserved (not replaced)
        assert h.storage_warnings >= 1
        assert rt._initialized is True                       # setup stays non-fatal

    async def test_populated_payload_restores_without_errors(self):
        # Round-trip: a payload the runtime itself serialized restores clean.
        coord, rt = await _setup_runtime(FakeStore(data=None))
        payload = rt._build_payload()
        coord2, rt2 = await _setup_runtime(FakeStore(data=payload))
        h = rt2.health()
        assert h.restore_errors == 0
        assert h.restore_schema_mismatch is False

    async def test_corrupt_segment_surfaced_but_isolated(self):
        bad = {"runtime_schema_version": RUNTIME_SCHEMA_VERSION,
               "zones": {"z": {"models": {"heat_rate": "corrupt-state"}}}}
        coord, rt = await _setup_runtime(FakeStore(data=bad))
        h = rt.health()
        assert h.restore_errors >= 1
        assert any("heat_rate" in r for r in h.restore_error_reasons)

    async def test_schema_mismatch_discards_payload(self):
        bad = {"runtime_schema_version": RUNTIME_SCHEMA_VERSION + 999,
               "zones": {"z": {"models": {}}}}
        coord, rt = await _setup_runtime(FakeStore(data=bad))
        h = rt.health()
        assert h.restore_schema_mismatch is True
        assert "schema_version_mismatch" in h.restore_error_reasons

    async def test_health_export_exposes_restore_fields(self):
        coord, rt = await _setup_runtime(FakeStore(fail_load=True))
        hd = _learning_health_data(coord)
        assert hd["restore_errors"] >= 1
        assert any(r.startswith("store:load_failed:") for r in hd["restore_error_reasons"])
        assert scan_payload(hd) == []
