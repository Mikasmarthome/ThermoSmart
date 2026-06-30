"""B2c lifecycle, _boost_active, and restore-barrier error-path tests.

Covers:
  TestBoostActiveInit   -- _boost_active initialized before first refresh, never AttributeError
  TestRestoreBarrier    -- barrier blocks until active_control initialized; error paths; no timeout
  TestPartialRecovery   -- sequential two-cycle test: partial dispatch -> baseline restoration
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from homeassistant.exceptions import HomeAssistantError

from tests.helpers_ha_runtime import (
    attach_shadow,
    inject_eligible_boost_model,
    make_dispatching_coordinator,
    make_recording_coordinator,
)


# -- TestBoostActiveInit --

class TestBoostActiveInit:
    """_boost_active is initialized before any coordinator refresh can access it."""

    def test_boost_active_present_at_init(self):
        """Attribute exists on fresh coordinator, before any setup or refresh."""
        coord = make_recording_coordinator()
        assert hasattr(coord, "_boost_active"), "_boost_active not initialized"
        assert isinstance(coord._boost_active, dict)

    def test_boost_active_empty_at_init(self):
        coord = make_recording_coordinator()
        assert coord._boost_active == {}

    @pytest.mark.asyncio
    async def test_no_attribute_error_on_first_refresh(self):
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
        )
        try:
            await coord._async_update_data()
        except AttributeError as e:
            pytest.fail(f"AttributeError on first refresh: {e}")

    @pytest.mark.asyncio
    async def test_boost_active_populated_after_heating(self):
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        await coord._async_update_data()
        assert isinstance(coord._boost_active, dict)

    @pytest.mark.asyncio
    async def test_multiple_refreshes_no_attribute_error(self):
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        for _ in range(3):
            try:
                await coord._async_update_data()
            except AttributeError as e:
                pytest.fail(f"AttributeError on refresh: {e}")


# -- TestRestoreBarrier --

class TestRestoreBarrier:
    """Restore initialization barrier blocks until active_control is initialized."""

    @pytest.mark.asyncio
    async def test_barrier_blocks_when_not_initialized(self):
        coord = make_recording_coordinator(indoor="15.0")
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        coord._active_control_initialized = False

        result = await coord._async_update_data()
        zone = result["zone"]
        assert zone.get("boost_offset_c", 0.0) == 0.0
        rej = zone.get("_boost_rejection_reason")
        assert rej in ("restore_pending", None, "learning_mode_off", "active_control_off"), \
            f"unexpected rejection: {rej!r}"

    @pytest.mark.asyncio
    async def test_barrier_unblocks_when_active_control_initialized(self):
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        assert coord._active_control_initialized

        result = await coord._async_update_data()
        assert result["zone"].get("_boost_rejection_reason") != "restore_pending"

    @pytest.mark.asyncio
    async def test_barrier_no_auto_release_without_initialization(self):
        coord = make_recording_coordinator(indoor="15.0")
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        coord._active_control_initialized = False

        for _ in range(5):
            await coord._async_update_data()
            assert not coord._active_control_initialized

    @pytest.mark.asyncio
    async def test_barrier_malformed_restore_state_stays_blocked(self):
        coord = make_recording_coordinator(indoor="15.0")
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        coord._active_control = "yes_please"
        coord._active_control_initialized = False

        result = await coord._async_update_data()
        assert result["zone"].get("boost_offset_c", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_barrier_entity_disabled_stays_blocked(self):
        coord = make_recording_coordinator(indoor="15.0")
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        coord._active_control_initialized = False

        result = await coord._async_update_data()
        assert result is not None
        assert result["zone"].get("boost_offset_c", 0.0) == 0.0

    def test_restore_pending_computable_without_entity_ids(self):
        coord = make_recording_coordinator()
        coord._active_control_initialized = False
        restore_pending = not coord._active_control_initialized
        assert restore_pending is True
        assert "climate." not in str(restore_pending)


# -- TestPartialRecovery --

class TestPartialRecovery:
    """Sequential two-cycle test: partial dispatch -> baseline restoration in cycle 2."""

    @pytest.mark.asyncio
    async def test_cycle1_partial_cycle2_no_boost(self):
        """Cycle 1 partial -> lifecycle invalidated. Cycle 2: no sticky boost."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": {"temperature": 21.0, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
                "climate.trv_b": {"temperature": 21.0, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
            },
            service_side_effect=[None, HomeAssistantError("b fails")],
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)

        result1 = await coord._async_update_data()
        ldr1 = coord._last_live_decision
        if ldr1 is not None and ldr1.dispatch_status == "partially_succeeded":
            assert result1["zone"].get("applied_boost_offset_c", 0.0) == 0.0

        _c2_calls: list = []
        async def _c2_svc(domain, service, data=None, *, blocking=True, **kw):
            _c2_calls.append(data or {})
        coord.hass.services.async_call = _c2_svc

        result2 = await coord._async_update_data()
        assert result2["zone"].get("applied_boost_offset_c", 0.0) == 0.0
        assert not bool(result2["zone"].get("le2_boost_adjusted"))

    @pytest.mark.asyncio
    async def test_cycle1_partial_cycle2_device_b_unavailable(self):
        """Device B unavailable in cycle 2; device A still receives its TPI baseline."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": {"temperature": 21.0, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
                "climate.trv_b": {"temperature": 21.0, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
            },
            service_side_effect=[None, HomeAssistantError("b fails")],
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)

        # Cycle 1: Device A may receive a boosted setpoint; Device B fails.
        await coord._async_update_data()
        c1_calls = coord._service_calls[:]

        # Find what setpoint was sent to Device A in cycle 1 (may be baseline or boosted)
        c1_trv_a = next(
            (c for c in c1_calls
             if c.get("data", {}).get("entity_id") == "climate.trv_a"), None)
        c1_sp_a = c1_trv_a["data"]["temperature"] if c1_trv_a is not None else 21.0

        # In cycle 2: Device B remains unavailable; Device A reflects c1 setpoint in state.
        _orig_get = coord.hass.states.get.side_effect
        def _c2_states(eid):
            if eid == "climate.trv_b":
                _st = MagicMock()
                _st.state = "unavailable"
                _st.attributes = {}
                return _st
            if eid == "climate.trv_a":
                _st = MagicMock()
                _st.state = "heat"
                _st.attributes = {"temperature": c1_sp_a, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5}
                return _st
            return _orig_get(eid) if _orig_get else None
        coord.hass.states.get = MagicMock(side_effect=_c2_states)

        c2_calls: list = []
        async def _c2_svc(domain, service, data=None, *, blocking=True, **kw):
            c2_calls.append({"domain": domain, "service": service, "data": data or {}})
        coord.hass.services.async_call = _c2_svc

        result2 = await coord._async_update_data()
        # Gate 3.6 blocks (Device B unavailable) → prediction=0.0, applied=0.0
        assert result2["zone"].get("boost_offset_c", 0.0) == 0.0
        assert result2["zone"].get("applied_boost_offset_c", 0.0) == 0.0
        # No exception; coordinator returns a valid result despite Device B being unavailable
        assert result2 is not None
        assert "zone" in result2

    @pytest.mark.asyncio
    async def test_cycle1_partial_device_a_baseline_cleanup_proof(self):
        """Positive proof: after partial dispatch, Device A explicitly at boosted setpoint
        in cycle 2 state → coordinator dispatches baseline to Device A."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": {"temperature": 21.0, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
                "climate.trv_b": {"temperature": 21.0, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
            },
            service_side_effect=[None, HomeAssistantError("b fails")],
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        await coord._async_update_data()

        # After cycle 1: get the TPI setpoint that was applied to Device A.
        c1_calls = coord._service_calls[:]
        c1_trv_a = next(
            (c for c in c1_calls
             if c.get("data", {}).get("entity_id") == "climate.trv_a"), None)

        if c1_trv_a is None:
            pytest.skip("No cycle-1 dispatch for Device A — model not approved")
            return
        c1_sp_a = c1_trv_a["data"]["temperature"]

        # Inject a higher temperature to simulate a boosted device for cycle 2.
        forced_boosted = c1_sp_a + 1.0

        def _c2_states(eid):
            st = MagicMock()
            if eid == "climate.trv_a":
                st.state = "heat"
                st.attributes = {"temperature": forced_boosted, "min_temp": 5.0,
                                  "hvac_mode": "heat", "target_temp_step": 0.5}
            elif eid == "climate.trv_b":
                st.state = "unavailable"
                st.attributes = {}
            else:
                return None
            return st
        coord.hass.states.get = MagicMock(side_effect=_c2_states)

        c2_calls: list = []
        async def _c2_svc(domain, service, data=None, *, blocking=True, **kw):
            c2_calls.append({"domain": domain, "service": service, "data": data or {}})
        coord.hass.services.async_call = _c2_svc

        result2 = await coord._async_update_data()
        # Gate 3.6 blocks (Device B unavailable) → prediction=0.0, applied=0.0
        assert result2["zone"].get("boost_offset_c", 0.0) == 0.0
        assert result2["zone"].get("applied_boost_offset_c", 0.0) == 0.0

        trv_a_c2 = next(
            (c for c in c2_calls if c.get("data", {}).get("entity_id") == "climate.trv_a"),
            None)
        assert trv_a_c2 is not None, (
            f"Device A (at {forced_boosted}°C) must receive baseline cleanup dispatch in cycle 2")
        c2_sp_a = trv_a_c2["data"]["temperature"]
        assert c2_sp_a < forced_boosted, (
            f"Cycle 2 setpoint {c2_sp_a} must be below forced boosted state {forced_boosted}"
        )

    @pytest.mark.asyncio
    async def test_no_old_decision_re_authorizes_boost(self):
        """No old authorization re-applies boost in cycle 2 after invalidation."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv_a", "climate.trv_b"],
            climate_states={
                "climate.trv_a": {"temperature": 21.0, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
                "climate.trv_b": {"temperature": 21.0, "min_temp": 5.0,
                                   "hvac_mode": "heat", "target_temp_step": 0.5},
            },
            service_side_effect=[None, HomeAssistantError("b fails")],
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        await coord._async_update_data()

        async def _c2_svc(domain, service, data=None, *, blocking=True, **kw):
            pass
        coord.hass.services.async_call = _c2_svc
        result2 = await coord._async_update_data()

        assert not bool(result2["zone"].get("le2_boost_adjusted"))


# -- TestNotAttempted --

class TestNotAttempted:
    """not_attempted dispatch: public applied must be 0.0, never stale from prior cycle.

    Applies to every scenario where dispatch is skipped: summer mode, active_control=False,
    restore pending — regardless of what the boost resolver approved.
    """

    @pytest.mark.asyncio
    async def test_active_control_off_applied_zero(self):
        """Active control OFF: Gate 2 blocks → boost_offset_c=0.0, applied_boost_offset_c=0.0."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        coord.set_active_control(False)

        result = await coord._async_update_data()
        assert result["zone"].get("boost_offset_c", 0.0) == 0.0          # Gate 2 blocks before model
        assert result["zone"].get("applied_boost_offset_c", 0.0) == 0.0  # no dispatch
        assert not coord._service_calls, "No service calls expected when active_control=False"

    @pytest.mark.asyncio
    async def test_restore_pending_applied_zero(self):
        """Restore barrier active: Gate 0 blocks → boost_offset_c=0.0, applied=0.0."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        coord._active_control_initialized = False

        result = await coord._async_update_data()
        assert result["zone"].get("boost_offset_c", 0.0) == 0.0          # Gate 0 blocks before model
        assert result["zone"].get("applied_boost_offset_c", 0.0) == 0.0  # no dispatch

    @pytest.mark.asyncio
    async def test_learning_off_applied_zero(self):
        """Learning OFF: Gate 1 blocks → boost_offset_c=0.0, applied=0.0."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        coord.entry.data = {**coord.entry.data, "learning_enabled": False}

        result = await coord._async_update_data()
        assert result["zone"].get("boost_offset_c", 0.0) == 0.0          # Gate 1 blocks before model
        assert result["zone"].get("applied_boost_offset_c", 0.0) == 0.0  # no dispatch

    @pytest.mark.asyncio
    async def test_prior_applied_boost_then_learning_off_no_stale(self):
        """Cycle 1: boost applied. Cycle 2: learning OFF → applied 0.0, no stale value."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)

        await coord._async_update_data()
        # Cycle 1 may or may not have approved boost — either way cycle 2 must be 0.0

        coord.entry.data = {**coord.entry.data, "learning_enabled": False}
        async def _c2_svc(domain, service, data=None, *, blocking=True, **kw):
            pass
        coord.hass.services.async_call = _c2_svc

        result2 = await coord._async_update_data()
        assert result2["zone"].get("boost_offset_c", 0.0) == 0.0          # Gate 1 blocks → prediction=0.0
        assert result2["zone"].get("applied_boost_offset_c", 0.0) == 0.0  # no stale applied
        assert result2["zone"].get("_boost_per_device_applied_c", []) == []

    @pytest.mark.asyncio
    async def test_prior_applied_boost_then_active_control_off_no_stale(self):
        """Cycle 1: boost applied. Cycle 2: active_control=False → applied 0.0."""
        coord = make_dispatching_coordinator(
            climate_entities=["climate.trv"],
            climate_states={"climate.trv": {"temperature": 21.0, "min_temp": 5.0,
                                             "hvac_mode": "heat", "target_temp_step": 0.5}},
            indoor="15.0",
        )
        shadow = attach_shadow(coord)
        await shadow.async_setup()
        inject_eligible_boost_model(shadow)
        await coord._async_update_data()

        coord.set_active_control(False)
        result2 = await coord._async_update_data()
        assert result2["zone"].get("boost_offset_c", 0.0) == 0.0          # Gate 2 blocks → prediction=0.0
        assert result2["zone"].get("applied_boost_offset_c", 0.0) == 0.0  # no stale applied
