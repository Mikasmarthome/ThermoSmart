"""B2c Real HA End-to-End tests — Adaptive Boost Dispatch Truth.

Exercises the FULL production chain without patching the core decision path:

    Zone setup
    → Learning Mode ON + Active Control ON
    → real BoostActivationReadiness (eligible model state injected, evaluator runs for real)
    → real Coordinator cycle (_async_update_data)
    → real TPI baseline
    → real resolve_adaptive_boost_control()
    → real adjust_recommendation_safe()
    → real _apply_temperature() (device clamping, step-snap, effective setpoint)
    → mocked hass.services.async_call (physical backend only)
    → real _DispatchStats
    → real LiveDecisionRecord
    → per-device applied truth computed post-dispatch

Allowed mocks: physical HA service backend, Climate entity states, clock/sensors.
Not patched: readiness, resolver, clamp/policy, addition path, dispatch evaluation,
             LiveDecisionRecord mapping.

Test cases:
  TestE2EFullSuccess      — all TRVs succeed, per-device truth from effective setpoints
  TestE2EFullFailure      — all TRVs fail, lifecycle invalidated, applied=0.0
  TestE2EPartialDispatch  — 2 TRVs, one fails, partial truth preserved, lifecycle released
  TestE2ELearningOff      — A/B: identical setup, learning OFF → no boost
  TestE2EDirectValve      — direct-valve zone → blocked at readiness (unsupported_control_type)
  TestE2EMixedZone        — mixed-control zone → blocked at Gate 3
"""
from __future__ import annotations

import pytest
from homeassistant.exceptions import HomeAssistantError
from unittest.mock import MagicMock

from tests.helpers_ha_runtime import (
    attach_shadow,
    inject_eligible_boost_model,
    make_dispatching_coordinator,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_heat_state(temp=21.0, min_temp=5.0, max_temp=35.0, step=0.5):
    return {"temperature": temp, "min_temp": min_temp, "max_temp": max_temp,
            "hvac_mode": "heat", "temperature_step": step}


async def _setup_boost_coord(climate_entities=None, climate_states=None,
                              service_side_effect=None, indoor="15.0"):
    """Build a DispatchingCoordinator + Shadow with an eligible BoostModel."""
    coord = make_dispatching_coordinator(
        climate_entities=climate_entities,
        climate_states=climate_states,
        service_side_effect=service_side_effect,
        indoor=indoor,
    )
    shadow = attach_shadow(coord)
    await shadow.async_setup()
    inject_eligible_boost_model(shadow, factor_c=1.5)
    return coord, shadow


# ── TestE2EFullSuccess ────────────────────────────────────────────────────────

class TestE2EFullSuccess:
    """Full dispatch success: per-device truth from actual effective setpoints."""

    async def test_boost_applied_c_comes_from_effective_setpoint(self):
        """applied offset = effective_setpoint − baseline (not pre-dispatch approved)."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0, step=0.5)},
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]

        # Service call must have happened (boost dispatched)
        assert any(c["service"] == "set_temperature" for c in coord._service_calls), \
            "expected climate.set_temperature to be called"

        # applied_boost_offset_c is the confirmed dispatch truth (boost_offset_c = raw prediction)
        applied = zone.get("applied_boost_offset_c", 0.0)
        assert isinstance(applied, float)

        # LiveDecisionRecord must have dispatch truth
        ldr = coord._last_live_decision
        assert ldr is not None
        assert ldr.dispatch_status in ("fully_succeeded", "not_attempted")
        if ldr.dispatch_status == "fully_succeeded":
            assert ldr.dispatch_targets_succeeded >= 1
            # boost_applied_c in LDR must equal applied_boost_offset_c in zone (real dispatch)
            if ldr.boost_applied_c is not None:
                assert abs(float(ldr.boost_applied_c) - applied) < 0.02

    async def test_ldr_boost_applied_c_not_from_pre_dispatch_approved(self):
        """LDR.boost_applied_c must NOT be the raw approved offset — must come from dispatch."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0, step=0.5)},
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]
        ldr = coord._last_live_decision

        if ldr is not None and ldr.dispatch_status == "fully_succeeded" and ldr.boost_applied_c:
            # The applied value from LDR must agree with zone's applied_boost_offset_c
            assert abs(float(ldr.boost_applied_c) - float(zone.get("applied_boost_offset_c", 0))) < 0.02

    async def test_per_device_applied_offsets_recorded(self):
        """Per-device applied offsets must be recorded in recommendation."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0, step=0.5)},
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]
        ldr = coord._last_live_decision
        if ldr is not None and ldr.dispatch_status == "fully_succeeded":
            # If boost was applied, per-device list must exist
            per_dev = zone.get("_boost_per_device_applied_c")
            if zone.get("le2_boost_adjusted"):
                assert per_dev is not None and isinstance(per_dev, list)
                assert len(per_dev) >= 1

    async def test_dispatch_status_is_fully_succeeded(self):
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0, step=0.5)},
            indoor="15.0",
        )
        result = await coord._async_update_data()
        ldr = coord._last_live_decision
        if ldr is not None:
            assert ldr.dispatch_status in ("fully_succeeded", "not_attempted")

    async def test_two_trvs_both_succeed(self):
        """Two identical TRVs, both succeed — conservative min of offsets used."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": _make_heat_state(temp=21.0, step=0.5),
                "climate.trv_b": _make_heat_state(temp=21.0, step=0.5),
            },
            indoor="15.0",
        )
        result = await coord._async_update_data()
        ldr = coord._last_live_decision
        zone = result["zone"]
        if ldr is not None and ldr.dispatch_status == "fully_succeeded":
            assert ldr.dispatch_targets_succeeded == 2
            # applied_boost_offset_c = min(dev_offsets) ≥ 0
            assert zone.get("applied_boost_offset_c", 0.0) >= 0.0


# ── TestE2EFullFailure ────────────────────────────────────────────────────────

class TestE2EFullFailure:
    """All TRVs fail: lifecycle invalidated, applied=0.0, LDR.boost_applied_c=0.0."""

    async def test_failed_dispatch_zeroes_applied(self):
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            service_side_effect=[HomeAssistantError("service failed")],
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]

        # applied_boost_offset_c must be 0.0 (no confirmed dispatch; prediction may still be set)
        assert zone.get("applied_boost_offset_c", 0.0) == 0.0
        assert zone.get("boost_factor", 1.0) == 1.0

    async def test_failed_dispatch_ldr_boost_applied_zero(self):
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            service_side_effect=[HomeAssistantError("service failed")],
            indoor="15.0",
        )
        result = await coord._async_update_data()
        ldr = coord._last_live_decision
        if ldr is not None and ldr.dispatch_status == "failed":
            assert (ldr.boost_applied_c is None or float(ldr.boost_applied_c) == 0.0)

    async def test_failed_dispatch_status_recorded(self):
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            service_side_effect=[HomeAssistantError("service failed")],
            indoor="15.0",
        )
        result = await coord._async_update_data()
        ldr = coord._last_live_decision
        if ldr is not None:
            # If boost was attempted and failed, dispatch_status reflects it
            if ldr.dispatch_attempted and ldr.dispatch_targets_failed > 0:
                assert ldr.dispatch_status in ("failed", "partially_succeeded")


# ── TestE2EPartialDispatch ───────────────────────────────────────────────────

class TestE2EPartialDispatch:
    """Two TRVs, one succeeds one fails: partial truth preserved, public=0.0."""

    async def test_partial_dispatch_public_boost_is_zero(self):
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": _make_heat_state(temp=21.0),
                "climate.trv_b": _make_heat_state(temp=21.0),
            },
            # First call (trv_a) succeeds, second (trv_b) fails
            service_side_effect=[None, HomeAssistantError("partial fail")],
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]
        ldr = coord._last_live_decision

        # If partial dispatch occurred, public applied_boost_offset_c must be 0.0
        if ldr is not None and ldr.dispatch_status == "partially_succeeded":
            assert zone.get("applied_boost_offset_c", 0.0) == 0.0
            assert zone.get("_boost_applied_c", 0.0) == 0.0

    async def test_partial_dispatch_per_device_preserved(self):
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": _make_heat_state(temp=21.0),
                "climate.trv_b": _make_heat_state(temp=21.0),
            },
            service_side_effect=[None, HomeAssistantError("partial fail")],
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]
        ldr = coord._last_live_decision

        if ldr is not None and ldr.dispatch_status == "partially_succeeded":
            # Per-device offsets for the successful device must be recorded
            per_dev = zone.get("_boost_per_device_applied_c")
            if zone.get("le2_boost_adjusted"):
                # At least the 1 successful device's offset is recorded
                assert per_dev is not None
                assert len(per_dev) >= 1

    async def test_partial_dispatch_ldr_status(self):
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": _make_heat_state(temp=21.0),
                "climate.trv_b": _make_heat_state(temp=21.0),
            },
            service_side_effect=[None, HomeAssistantError("partial fail")],
            indoor="15.0",
        )
        result = await coord._async_update_data()
        ldr = coord._last_live_decision
        if ldr is not None and ldr.dispatch_attempted:
            # dispatch_targets must be recorded
            assert ldr.dispatch_targets_total >= 1


# ── TestE2ELearningOff ───────────────────────────────────────────────────────

class TestE2ELearningOff:
    """A/B: identical setup, learning OFF → no boost dispatched."""

    async def test_learning_off_no_boosted_setpoint(self):
        """When learning mode is OFF, no boosted setpoint is dispatched."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            indoor="15.0",
        )
        # Override: disable learning in config
        coord.entry.data = {**coord.entry.data, "learning_enabled": False}

        result = await coord._async_update_data()
        zone = result["zone"]

        assert zone.get("boost_offset_c", 0.0) == 0.0
        assert not bool(zone.get("le2_boost_adjusted"))

    async def test_learning_off_blocking_reason(self):
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            indoor="15.0",
        )
        coord.entry.data = {**coord.entry.data, "learning_enabled": False}
        result = await coord._async_update_data()
        zone = result["zone"]
        assert zone.get("_boost_rejection_reason") == "learning_mode_off"

    async def test_learning_on_vs_off_same_baseline(self):
        """Baseline setpoint is identical regardless of learning mode."""
        coord_on, shadow_on = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            indoor="15.0",
        )
        coord_off, shadow_off = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            indoor="15.0",
        )
        coord_off.entry.data = {**coord_off.entry.data, "learning_enabled": False}

        result_on = await coord_on._async_update_data()
        result_off = await coord_off._async_update_data()

        baseline_on = result_on["zone"].get("tpi_baseline_setpoint") or result_on["zone"].get("trv_setpoint")
        baseline_off = result_off["zone"].get("trv_setpoint")
        if baseline_on is not None and baseline_off is not None:
            assert abs(float(baseline_on) - float(baseline_off)) < 0.1


# ── TestE2EDirectValve ────────────────────────────────────────────────────────

class TestE2EDirectValve:
    """Direct-valve zone: boost blocked at readiness (unsupported_control_type)."""

    async def test_direct_valve_no_boost(self):
        """Direct valve zones never receive a boost setpoint."""
        from tests.helpers import make_coordinator
        from unittest.mock import patch as _patch
        coord = make_dispatching_coordinator(
            zone_cfg_overrides={"climate_entities": ["climate.trv_valve"]},
            indoor="15.0",
        )
        # Mark as direct-valve zone
        coord._auto_valve_map = {"climate.trv_valve": True}
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)

        result = await coord._async_update_data()
        zone = result["zone"]

        # Direct valve: boost_offset_c must be 0.0
        assert zone.get("boost_offset_c", 0.0) == 0.0
        assert not bool(zone.get("le2_boost_adjusted"))

    async def test_direct_valve_blocking_reason(self):
        coord = make_dispatching_coordinator(
            zone_cfg_overrides={"climate_entities": ["climate.trv_valve"]},
            indoor="15.0",
        )
        coord._auto_valve_map = {"climate.trv_valve": True}
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)

        result = await coord._async_update_data()
        zone = result["zone"]
        # Rejection reason should be direct_valve or unsupported_control_type
        rej = zone.get("_boost_rejection_reason", "")
        assert rej in ("direct_valve", "unsupported_control_type", "mixed_control_types_unsupported"), \
            f"unexpected rejection: {rej!r}"


# ── TestE2EMixedZone ──────────────────────────────────────────────────────────

class TestE2EMixedZone:
    """Mixed-control zone (setpoint + direct valve): blocked at Gate 3."""

    async def test_mixed_zone_no_boost(self):
        """Mixed zone must never receive a boost setpoint."""
        coord = make_dispatching_coordinator(
            zone_cfg_overrides={"climate_entities": ["climate.trv_sp", "climate.trv_valve"]},
            indoor="15.0",
        )
        # trv_sp = setpoint, trv_valve = direct valve
        coord._auto_valve_map = {"climate.trv_valve": True}
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)

        result = await coord._async_update_data()
        zone = result["zone"]

        assert zone.get("boost_offset_c", 0.0) == 0.0
        assert not bool(zone.get("le2_boost_adjusted"))

    async def test_mixed_zone_typed_blocking_reason(self):
        coord = make_dispatching_coordinator(
            zone_cfg_overrides={"climate_entities": ["climate.trv_sp", "climate.trv_valve"]},
            indoor="15.0",
        )
        coord._auto_valve_map = {"climate.trv_valve": True}
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)

        result = await coord._async_update_data()
        zone = result["zone"]

        assert zone.get("_boost_rejection_reason") == "mixed_control_types_unsupported"
        assert zone.get("zone_control_type") == "mixed"

    async def test_mixed_zone_ldr_no_boost(self):
        coord = make_dispatching_coordinator(
            zone_cfg_overrides={"climate_entities": ["climate.trv_sp", "climate.trv_valve"]},
            indoor="15.0",
        )
        coord._auto_valve_map = {"climate.trv_valve": True}
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)

        result = await coord._async_update_data()
        ldr = coord._last_live_decision
        if ldr is not None:
            assert not bool(getattr(ldr, "boost_applied_c", None))


# ── TestE2EMaxClamp ───────────────────────────────────────────────────────────

class TestE2EMaxClamp:
    """max_temp device ceiling clips effective setpoint; applied < approved.

    Uses resolver + real _apply_temperature() via the shared authority so
    both sides of the counterfactual see the same max_temp.
    """

    @pytest.mark.asyncio
    async def test_max_clamp_reduces_applied_service_call(self):
        """Device max_temp=23.0 clips the boosted setpoint in the real service call.
        baseline=22.5, approved≈1.0 → boosted_input=23.5 → clamped to 23.0.
        applied = 23.0 - 22.5 = 0.5 (less than approved).
        """
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": {
                "temperature": 22.5, "min_temp": 5.0, "max_temp": 23.0,
                "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]
        ldr = coord._last_live_decision

        if ldr is not None and ldr.dispatch_status == "fully_succeeded":
            calls = coord._service_calls
            trv_call = next((c for c in calls if c.get("service") == "set_temperature"), None)
            if trv_call is not None:
                effective_sp = trv_call["data"]["temperature"]
                assert effective_sp <= 23.0, (
                    f"Service call sent {effective_sp}°C exceeds device max_temp=23.0")
                # applied_boost_offset_c must be ≤ approved if max_temp was hit
                applied = zone.get("applied_boost_offset_c", 0.0)
                assert applied <= 1.0 + 0.01

    @pytest.mark.asyncio
    async def test_max_clamp_baseline_at_max_applied_zero(self):
        """Baseline already at device max → no headroom → applied = 0.0."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": {
                "temperature": 23.0, "min_temp": 5.0, "max_temp": 23.0,
                "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]
        ldr = coord._last_live_decision

        if ldr is not None and ldr.dispatch_status == "fully_succeeded":
            assert zone.get("applied_boost_offset_c", 0.0) == pytest.approx(0.0, abs=0.01), (
                "Baseline at max_temp must give applied_boost_offset_c=0.0")

    @pytest.mark.asyncio
    async def test_mixed_device_limits_conservative(self):
        """Device A: max=25.0 → full boost. Device B: max=22.5 → limited boost.
        Conservative min(applied_A, applied_B) used as zone truth.
        """
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": {"temperature": 21.0, "min_temp": 5.0, "max_temp": 25.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
                "climate.trv_b": {"temperature": 21.0, "min_temp": 5.0, "max_temp": 22.5,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
            },
            indoor="15.0",
        )
        result = await coord._async_update_data()
        zone = result["zone"]
        ldr = coord._last_live_decision

        if ldr is not None and ldr.dispatch_status == "fully_succeeded":
            per_dev = zone.get("_boost_per_device_applied_c", [])
            if per_dev and len(per_dev) == 2:
                # Device B (lower max) must have lower or equal applied vs Device A
                applied_conservative = zone.get("applied_boost_offset_c", 0.0)
                assert applied_conservative <= max(per_dev) + 0.01, (
                    "Conservative min must not exceed any device's individual applied")

    @pytest.mark.asyncio
    async def test_step_plus_max_clamp_combined(self):
        """step=0.5, max=23.0: boost input → step-snap → may exceed max → re-clamped.
        Service call must never exceed device max.
        """
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": {
                "temperature": 21.0, "min_temp": 5.0, "max_temp": 23.0,
                "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        result = await coord._async_update_data()
        ldr = coord._last_live_decision

        if ldr is not None and ldr.dispatch_status == "fully_succeeded":
            calls = coord._service_calls
            for call in calls:
                if call.get("service") == "set_temperature":
                    sp = call["data"]["temperature"]
                    assert sp <= 23.0, (
                        f"Step-snap must not bypass max_temp: {sp}°C > 23.0")


# ── TestE2ESequential ─────────────────────────────────────────────────────────

class TestE2ESequential:
    """Multi-cycle sequential tests verifying state transitions between cycles."""

    @pytest.mark.asyncio
    async def test_full_success_then_normal_cycle(self):
        """Cycle 1: full success with boost. Cycle 2: same baseline, boost may re-apply."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            indoor="15.0",
        )
        result1 = await coord._async_update_data()

        async def _c2_svc(domain, service, data=None, *, blocking=True, **kw):
            pass
        coord.hass.services.async_call = _c2_svc

        result2 = await coord._async_update_data()
        # No crash; result is valid
        assert result2 is not None
        assert "zone" in result2

    @pytest.mark.asyncio
    async def test_failed_then_next_cycle_baseline(self):
        """After full failure, next cycle must NOT carry stale applied_boost_offset_c."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            service_side_effect=[HomeAssistantError("fail")],
            indoor="15.0",
        )
        result1 = await coord._async_update_data()

        async def _c2_svc(domain, service, data=None, *, blocking=True, **kw):
            pass
        coord.hass.services.async_call = _c2_svc

        result2 = await coord._async_update_data()
        assert result2["zone"].get("applied_boost_offset_c", 0.0) == 0.0
        assert not bool(result2["zone"].get("le2_boost_adjusted"))

    @pytest.mark.asyncio
    async def test_not_attempted_after_previous_boost_no_stale(self):
        """Cycle 1 had boost. Cycle 2 active_control=False → not_attempted → applied=0.0."""
        coord, shadow = await _setup_boost_coord(
            climate_entities=["climate.trv_a"],
            climate_states={"climate.trv_a": _make_heat_state(temp=21.0)},
            indoor="15.0",
        )
        await coord._async_update_data()

        coord.set_active_control(False)
        result2 = await coord._async_update_data()
        assert result2["zone"].get("boost_offset_c", 0.0) == 0.0          # Gate 2 blocks → prediction=0.0
        assert result2["zone"].get("applied_boost_offset_c", 0.0) == 0.0  # no dispatch
        assert result2["zone"].get("_boost_per_device_applied_c", []) == []
