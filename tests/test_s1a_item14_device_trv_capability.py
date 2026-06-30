"""S1a Item 14 — Device / TRV Capability and Failure Audit.

Tests the TRV control dispatch layer directly:
  - _DispatchStats accumulation, merge, and status
  - resolve_device_effective_setpoint 6-step transform
  - _apply_temperature setpoint dispatch (success, dedup, skip, failure, profile effects)
  - _apply_temperature window-open frost path
  - _apply_temperature multi-TRV partial-success handling
  - _async_set_valve_percent direct valve dispatch (duty, no-op, fractional, failure)
  - DeviceProfile effects (is_active, minimum/maximum_setpoint, setpoint_method)
  - _last_written_setpoints semantics (success-only update, clean after failure)
  - _trv_offline / _auto_valve_map / _device_profiles initial state at restart
  - _boost_active tracking when trv_setpoint > adjusted_target

Item 13 covers dispatch gating (active_control / summer).  This file covers
dispatch internals — what happens when dispatch IS attempted.

Pure Python + MagicMock.  No hass fixture (setup.cfg -p no:homeassistant).
asyncio_mode = auto → no @pytest.mark.asyncio needed.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, call

from custom_components.thermosmart.trv_control import _DispatchStats
from custom_components.thermosmart.device_profiles.capabilities import (
    DeviceProfile,
    SETPOINT_HVAC_FIRST,
)
from custom_components.thermosmart.const import WINDOW_OPEN_SETPOINT

from tests.helpers import make_coordinator, make_state, set_hass_states


# ── Helpers ──────────────────────────────────────────────────────────────────

def _trv_state(
    *,
    setpoint: float = 19.0,
    current_temp: float = 18.5,
    min_temp: float = 5.0,
    max_temp: float = 30.0,
    step: float | None = 0.5,
    state: str = "heat",
) -> object:
    attrs: dict = {
        "temperature": setpoint,
        "current_temperature": current_temp,
        "min_temp": min_temp,
        "max_temp": max_temp,
    }
    if step is not None:
        attrs["target_temp_step"] = step
    return make_state(state, attrs)


def _coord_with_svc(*, fail: bool = False, delay: float = 0.0) -> object:
    """Coordinator with async_call wired up as AsyncMock."""
    coord = make_coordinator()
    if fail:
        coord.hass.services.async_call = AsyncMock(side_effect=Exception("Service error"))
    else:
        coord.hass.services.async_call = AsyncMock(return_value=None)
    return coord


_CFG_ONE = {
    "climate_entities": ["climate.trv"],
    "temp_tolerance": 0.5,
    "window_open_temp": WINDOW_OPEN_SETPOINT,
}

_CFG_TWO = {
    "climate_entities": ["climate.trv1", "climate.trv2"],
    "temp_tolerance": 0.5,
    "window_open_temp": WINDOW_OPEN_SETPOINT,
}

_REC_NORMAL = {"adjusted_target": 21.0, "trv_setpoint": 23.5, "window_open": False}
_REC_WINDOW = {"adjusted_target": None, "trv_setpoint": None, "window_open": True}


# ── §1 _DispatchStats unit tests ─────────────────────────────────────────────

class TestDispatchStats:
    def test_initial_status_is_not_attempted(self):
        assert _DispatchStats().status == "not_attempted"

    def test_initial_counts_are_zero(self):
        s = _DispatchStats()
        assert s.targets_total == 0
        assert s.targets_succeeded == 0
        assert s.targets_failed == 0
        assert s.failure_reasons == []
        assert s.effective_setpoints == []
        assert s.effective_setpoints_by_entity == {}

    def test_record_success_increments_succeeded(self):
        s = _DispatchStats()
        s.record(None, effective_c=21.0, entity_id="climate.trv")
        assert s.targets_total == 1
        assert s.targets_succeeded == 1
        assert s.targets_failed == 0

    def test_record_success_stores_setpoint(self):
        s = _DispatchStats()
        s.record(None, effective_c=21.0, entity_id="climate.trv")
        assert 21.0 in s.effective_setpoints
        assert s.effective_setpoints_by_entity["climate.trv"] == 21.0

    def test_record_failure_increments_failed(self):
        s = _DispatchStats()
        s.record(Exception("Service error"))
        assert s.targets_total == 1
        assert s.targets_failed == 1
        assert s.targets_succeeded == 0

    def test_status_fully_succeeded_after_one_success(self):
        s = _DispatchStats()
        s.record(None, effective_c=21.0, entity_id="climate.trv")
        assert s.status == "fully_succeeded"

    def test_status_failed_after_one_failure(self):
        s = _DispatchStats()
        s.record(Exception("error"))
        assert s.status == "failed"

    def test_status_partially_succeeded_mixed(self):
        s = _DispatchStats()
        s.record(None, effective_c=21.0, entity_id="climate.trv1")
        s.record(Exception("error"))
        assert s.status == "partially_succeeded"

    def test_merge_two_succeeded_is_fully_succeeded(self):
        a = _DispatchStats()
        a.record(None, effective_c=21.0, entity_id="climate.trv1")
        b = _DispatchStats()
        b.record(None, effective_c=22.0, entity_id="climate.trv2")
        merged = a.merge(b)
        assert merged.status == "fully_succeeded"
        assert merged.targets_total == 2
        assert merged.targets_succeeded == 2

    def test_merge_success_and_failure_is_partially_succeeded(self):
        a = _DispatchStats()
        a.record(None, effective_c=21.0, entity_id="climate.trv1")
        b = _DispatchStats()
        b.record(Exception("error"))
        merged = a.merge(b)
        assert merged.status == "partially_succeeded"

    def test_merge_two_empty_is_not_attempted(self):
        merged = _DispatchStats().merge(_DispatchStats())
        assert merged.status == "not_attempted"

    def test_record_gather_all_success(self):
        s = _DispatchStats()
        s.record_gather([None, None])
        assert s.targets_succeeded == 2
        assert s.targets_failed == 0
        assert s.status == "fully_succeeded"

    def test_record_gather_mixed(self):
        s = _DispatchStats()
        s.record_gather([None, Exception("error")])
        assert s.targets_succeeded == 1
        assert s.targets_failed == 1
        assert s.status == "partially_succeeded"


# ── §2 resolve_device_effective_setpoint ─────────────────────────────────────

class TestResolveDeviceEffectiveSetpoint:
    def _coord(self):
        return make_coordinator()

    def test_no_transform_needed(self):
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=0.5)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 21.0)
        assert result == pytest.approx(21.0)

    def test_min_temp_clamp_below_device_min(self):
        coord = self._coord()
        state = _trv_state(min_temp=15.0, max_temp=30.0, step=0.5)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 10.0)
        assert result >= 15.0

    def test_min_temp_clamp_exact_boundary(self):
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=0.5)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 5.0)
        assert result == pytest.approx(5.0)

    def test_max_temp_clamp_above_device_max(self):
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=25.0, step=0.5)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 28.0)
        assert result <= 25.0

    def test_step_snap_whole_degree(self):
        """step=1.0, target=20.3 → rounds to 20.0."""
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=1.0)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 20.3)
        assert result == pytest.approx(20.0)

    def test_step_snap_half_degree(self):
        """step=0.5, target=20.7 → rounds to 20.5."""
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=0.5)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 20.7)
        assert result == pytest.approx(20.5)

    def test_step_snap_round_up(self):
        """step=1.0, target=20.7 → rounds to 21.0."""
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=1.0)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 20.7)
        assert result == pytest.approx(21.0)

    def test_zero_min_temp_is_not_used_as_clamp(self):
        """min_temp=0 sentinel → no clamp; setpoint passes through."""
        coord = self._coord()
        state = _trv_state(min_temp=0.0, max_temp=30.0, step=None)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 10.0)
        assert result == pytest.approx(10.0)

    def test_zero_max_temp_is_not_used_as_clamp(self):
        """max_temp=0 sentinel → no clamp; setpoint passes through."""
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=0.0, step=None)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 28.0)
        assert result == pytest.approx(28.0)

    def test_profile_minimum_setpoint_stricter_than_device(self):
        """Profile floor overrides when it is stricter than device min_temp."""
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=0.5)
        profile = DeviceProfile(identifier="test", display_name="Test", minimum_setpoint=16.0)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 10.0, profile)
        assert result >= 16.0

    def test_profile_maximum_setpoint_stricter_than_device(self):
        """Profile ceiling overrides when it is stricter than device max_temp."""
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=0.5)
        profile = DeviceProfile(identifier="test", display_name="Test", maximum_setpoint=22.0)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 28.0, profile)
        assert result <= 22.0

    def test_no_profile_does_not_apply_profile_clamps(self):
        """Without a profile steps 2 and 6 are skipped; target passes unchanged."""
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=0.5)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 21.0)
        assert result == pytest.approx(21.0)

    def test_missing_step_attribute_leaves_setpoint_unchanged(self):
        """No target_temp_step → step-snap is skipped."""
        coord = self._coord()
        state = _trv_state(min_temp=5.0, max_temp=30.0, step=None)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 20.3)
        assert result == pytest.approx(20.3)

    def test_snap_below_min_is_reclamped_after_snap(self):
        """If snap rounds below min_temp, step 4 re-clamps to min."""
        coord = self._coord()
        state = _trv_state(min_temp=15.0, max_temp=30.0, step=1.0)
        result = coord.resolve_device_effective_setpoint("climate.trv", state, 15.3)
        assert result >= 15.0


# ── §3 _apply_temperature — setpoint dispatch ─────────────────────────────────

class TestApplyTemperatureSetpointDispatch:
    async def test_success_calls_service_and_updates_last_written(self):
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "fully_succeeded"
        assert coord.hass.services.async_call.called
        assert coord._last_written_setpoints.get("climate.trv") == pytest.approx(23.5)

    async def test_success_service_call_uses_correct_setpoint(self):
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        args = coord.hass.services.async_call.call_args
        assert args[0][0] == "climate"
        assert args[0][1] == "set_temperature"
        assert args[0][2]["temperature"] == pytest.approx(23.5)

    async def test_dedup_exact_match_skips_service_call(self):
        """Current setpoint == trv_setpoint → within 0.5°C tolerance → no dispatch."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=23.5)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called

    async def test_dedup_within_tolerance_skips_service_call(self):
        """Difference 0.3°C < 0.5°C tolerance → no dispatch."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=23.3)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "not_attempted"

    async def test_dedup_just_outside_tolerance_dispatches(self):
        """Difference 0.6°C > 0.5°C tolerance → dispatches."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=22.9)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "fully_succeeded"

    async def test_unavailable_trv_is_skipped_without_service_call(self):
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": make_state("unavailable")})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called

    async def test_unknown_trv_is_skipped_without_service_call(self):
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": make_state("unknown")})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called

    async def test_service_failure_records_failed_and_does_not_update_last_written(self):
        coord = _coord_with_svc(fail=True)
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "failed"
        assert coord._last_written_setpoints.get("climate.trv") is None

    async def test_profile_is_active_false_skips_dispatch(self):
        coord = _coord_with_svc()
        coord._device_profiles = {
            "climate.trv": DeviceProfile(identifier="t", display_name="T", is_active=False)
        }
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called

    async def test_profile_minimum_setpoint_clamps_dispatch_value(self):
        """Profile floor 20.0 > trv_setpoint 15.0 → dispatched at 20.0."""
        coord = _coord_with_svc()
        coord._device_profiles = {
            "climate.trv": DeviceProfile(
                identifier="t", display_name="T", minimum_setpoint=20.0
            )
        }
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=10.0)})
        rec = {"adjusted_target": 14.0, "trv_setpoint": 15.0, "window_open": False}
        await coord._apply_temperature(_CFG_ONE, rec)
        args = coord.hass.services.async_call.call_args
        assert args[0][2]["temperature"] == pytest.approx(20.0)

    async def test_profile_maximum_setpoint_clamps_dispatch_value(self):
        """Profile ceiling 22.0 < trv_setpoint 28.0 → dispatched at 22.0."""
        coord = _coord_with_svc()
        coord._device_profiles = {
            "climate.trv": DeviceProfile(
                identifier="t", display_name="T", maximum_setpoint=22.0
            )
        }
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=10.0)})
        rec = {"adjusted_target": 21.0, "trv_setpoint": 28.0, "window_open": False}
        await coord._apply_temperature(_CFG_ONE, rec)
        args = coord.hass.services.async_call.call_args
        assert args[0][2]["temperature"] == pytest.approx(22.0)

    async def test_setpoint_hvac_first_sends_set_temperature(self):
        """SETPOINT_HVAC_FIRST: set_temperature is always called (state=heat → hvac_mode call skipped)."""
        coord = _coord_with_svc()
        coord._device_profiles = {
            "climate.trv": DeviceProfile(
                identifier="t", display_name="T", setpoint_method=SETPOINT_HVAC_FIRST
            )
        }
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0, state="heat")})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "fully_succeeded"
        calls = coord.hass.services.async_call.call_args_list
        svc_names = [c[0][1] for c in calls]
        assert "set_temperature" in svc_names

    async def test_setpoint_hvac_first_sends_hvac_mode_when_state_differs(self):
        """SETPOINT_HVAC_FIRST: set_hvac_mode is called when TRV state != target mode."""
        coord = _coord_with_svc()
        coord._device_profiles = {
            "climate.trv": DeviceProfile(
                identifier="t", display_name="T", setpoint_method=SETPOINT_HVAC_FIRST
            )
        }
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0, state="off")})
        await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        calls = coord.hass.services.async_call.call_args_list
        svc_names = [c[0][1] for c in calls]
        assert "set_hvac_mode" in svc_names
        assert "set_temperature" in svc_names

    async def test_boost_active_tracked_when_trv_setpoint_exceeds_target(self):
        """When trv_setpoint > adjusted_target, _boost_active is updated."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert coord.zone_id in coord._boost_active

    async def test_boost_not_tracked_when_setpoint_equals_target(self):
        """When trv_setpoint == adjusted_target, _boost_active not set."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        rec = {"adjusted_target": 21.0, "trv_setpoint": 21.0, "window_open": False}
        await coord._apply_temperature(_CFG_ONE, rec)
        assert coord.zone_id not in coord._boost_active

    async def test_adjusted_target_outside_safety_range_aborts_dispatch(self):
        """adjusted_target < 5.0 → safety check fails → no dispatch."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        rec = {"adjusted_target": 4.0, "trv_setpoint": 4.0, "window_open": False}
        stats = await coord._apply_temperature(_CFG_ONE, rec)
        assert stats.status == "not_attempted"


# ── §4 _apply_temperature — window-open frost path ────────────────────────────

class TestApplyTemperatureWindowOpen:
    async def test_window_open_sends_frost_setpoint(self):
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_WINDOW)
        assert stats.status == "fully_succeeded"
        args = coord.hass.services.async_call.call_args
        assert args[0][2]["temperature"] == pytest.approx(WINDOW_OPEN_SETPOINT)

    async def test_window_open_updates_last_written_setpoints(self):
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG_ONE, _REC_WINDOW)
        assert coord._last_written_setpoints.get("climate.trv") == pytest.approx(
            WINDOW_OPEN_SETPOINT
        )

    async def test_window_open_dedup_skips_if_already_at_frost(self):
        """Current setpoint == WINDOW_OPEN_SETPOINT → within 0.3°C → no dispatch."""
        coord = _coord_with_svc()
        set_hass_states(
            coord, {"climate.trv": _trv_state(setpoint=WINDOW_OPEN_SETPOINT)}
        )
        stats = await coord._apply_temperature(_CFG_ONE, _REC_WINDOW)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called

    async def test_window_open_service_failure_does_not_update_last_written(self):
        coord = _coord_with_svc(fail=True)
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG_ONE, _REC_WINDOW)
        assert coord._last_written_setpoints.get("climate.trv") is None

    async def test_window_open_profile_is_active_false_skips(self):
        coord = _coord_with_svc()
        coord._device_profiles = {
            "climate.trv": DeviceProfile(identifier="t", display_name="T", is_active=False)
        }
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_WINDOW)
        assert stats.status == "not_attempted"


# ── §5 _apply_temperature — multi-TRV ─────────────────────────────────────────

class TestApplyTemperatureMultiTRV:
    async def test_two_trvs_both_succeed_fully_succeeded(self):
        coord = _coord_with_svc()
        set_hass_states(
            coord,
            {
                "climate.trv1": _trv_state(setpoint=19.0),
                "climate.trv2": _trv_state(setpoint=18.0),
            },
        )
        stats = await coord._apply_temperature(_CFG_TWO, _REC_NORMAL)
        assert stats.status == "fully_succeeded"
        assert stats.targets_succeeded == 2

    async def test_two_trvs_both_fail_is_failed(self):
        coord = _coord_with_svc(fail=True)
        set_hass_states(
            coord,
            {
                "climate.trv1": _trv_state(setpoint=19.0),
                "climate.trv2": _trv_state(setpoint=18.0),
            },
        )
        stats = await coord._apply_temperature(_CFG_TWO, _REC_NORMAL)
        assert stats.status == "failed"
        assert coord._last_written_setpoints.get("climate.trv1") is None
        assert coord._last_written_setpoints.get("climate.trv2") is None

    async def test_partial_success_one_unavailable(self):
        """1 TRV unavailable + 1 TRV succeeds → not_attempted + fully_succeeded = ???

        Unavailable TRVs are silently skipped (no record at all).
        So the result is fully_succeeded from the one that was attempted.
        """
        coord = _coord_with_svc()
        set_hass_states(
            coord,
            {
                "climate.trv1": make_state("unavailable"),
                "climate.trv2": _trv_state(setpoint=18.0),
            },
        )
        stats = await coord._apply_temperature(_CFG_TWO, _REC_NORMAL)
        assert stats.status == "fully_succeeded"
        assert stats.targets_succeeded == 1

    async def test_partial_success_one_service_fails(self):
        """1 success + 1 service failure → partially_succeeded."""
        call_count = [0]

        async def _svc(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("TRV2 error")

        coord = make_coordinator()
        coord.hass.services.async_call = _svc
        set_hass_states(
            coord,
            {
                "climate.trv1": _trv_state(setpoint=19.0),
                "climate.trv2": _trv_state(setpoint=18.0),
            },
        )
        stats = await coord._apply_temperature(_CFG_TWO, _REC_NORMAL)
        assert stats.status == "partially_succeeded"

    async def test_partial_success_healthy_trv_still_gets_last_written_updated(self):
        """After partial failure, the successful entity IS in _last_written_setpoints."""
        call_count = [0]

        async def _svc(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("TRV2 error")

        coord = make_coordinator()
        coord.hass.services.async_call = _svc
        set_hass_states(
            coord,
            {
                "climate.trv1": _trv_state(setpoint=19.0),
                "climate.trv2": _trv_state(setpoint=18.0),
            },
        )
        await coord._apply_temperature(_CFG_TWO, _REC_NORMAL)
        assert coord._last_written_setpoints.get("climate.trv1") is not None
        assert coord._last_written_setpoints.get("climate.trv2") is None

    async def test_all_unavailable_gives_not_attempted(self):
        coord = _coord_with_svc()
        set_hass_states(
            coord,
            {
                "climate.trv1": make_state("unavailable"),
                "climate.trv2": make_state("unavailable"),
            },
        )
        stats = await coord._apply_temperature(_CFG_TWO, _REC_NORMAL)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called

    async def test_no_service_call_storm_on_repeated_unavailable(self):
        """Multiple cycles with unavailable TRVs never accumulate pending calls."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": make_state("unavailable")})
        for _ in range(5):
            await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert not coord.hass.services.async_call.called

    async def test_dedup_prevents_redundant_writes_across_cycles(self):
        """After successful write, next cycle with unchanged TRV state is deduplicated."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        coord.hass.services.async_call.reset_mock()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=23.5)})
        stats = await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called


# ── §6 _async_set_valve_percent ───────────────────────────────────────────────

class TestAsyncSetValvePercent:
    def _coord_valve(self, *, valve_state: str = "50", fail: bool = False):
        coord = _coord_with_svc(fail=fail)
        coord._auto_valve_map = {"climate.trv": "number.valve"}
        set_hass_states(coord, {"number.valve": make_state(valve_state)})
        return coord

    async def test_success_calls_set_value_with_correct_duty(self):
        coord = self._coord_valve(valve_state="50")
        stats = await coord._async_set_valve_percent(_CFG_ONE, 75.0)
        assert stats.status == "fully_succeeded"
        args = coord.hass.services.async_call.call_args
        assert args[0][0] == "number"
        assert args[0][1] == "set_value"
        assert args[0][2]["value"] == pytest.approx(75.0)

    async def test_empty_valve_map_returns_not_attempted(self):
        coord = _coord_with_svc()
        coord._auto_valve_map = {}
        stats = await coord._async_set_valve_percent(_CFG_ONE, 75.0)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called

    async def test_noop_when_valve_already_at_target(self):
        coord = self._coord_valve(valve_state="75")
        stats = await coord._async_set_valve_percent(_CFG_ONE, 75.0)
        assert stats.status == "not_attempted"
        assert not coord.hass.services.async_call.called

    async def test_unavailable_valve_entity_is_skipped(self):
        coord = _coord_with_svc()
        coord._auto_valve_map = {"climate.trv": "number.valve"}
        set_hass_states(coord, {"number.valve": make_state("unavailable")})
        stats = await coord._async_set_valve_percent(_CFG_ONE, 75.0)
        assert not coord.hass.services.async_call.called

    async def test_fractional_scale_for_level_entities(self):
        """Entities with 'level' in their name use 0.0-1.0 range."""
        coord = _coord_with_svc()
        coord._auto_valve_map = {"climate.trv": "number.valve_level"}
        set_hass_states(coord, {"number.valve_level": make_state("0.5")})
        await coord._async_set_valve_percent(_CFG_ONE, 75.0)
        args = coord.hass.services.async_call.call_args
        assert args[0][2]["value"] == pytest.approx(0.75)

    async def test_service_failure_recorded_in_stats(self):
        coord = self._coord_valve(valve_state="50", fail=True)
        stats = await coord._async_set_valve_percent(_CFG_ONE, 75.0)
        assert stats.targets_failed >= 1

    async def test_duty_at_zero_sends_zero(self):
        coord = self._coord_valve(valve_state="50")
        await coord._async_set_valve_percent(_CFG_ONE, 0.0)
        args = coord.hass.services.async_call.call_args
        assert args[0][2]["value"] == pytest.approx(0.0)

    async def test_duty_at_one_hundred_sends_one_hundred(self):
        coord = self._coord_valve(valve_state="50")
        await coord._async_set_valve_percent(_CFG_ONE, 100.0)
        args = coord.hass.services.async_call.call_args
        assert args[0][2]["value"] == pytest.approx(100.0)

    async def test_no_valve_map_entry_for_climate_entity_skips(self):
        coord = _coord_with_svc()
        coord._auto_valve_map = {"climate.other": "number.valve"}  # different key
        set_hass_states(coord, {"number.valve": make_state("50")})
        stats = await coord._async_set_valve_percent(_CFG_ONE, 75.0)
        assert not coord.hass.services.async_call.called

    async def test_two_valves_both_succeed(self):
        coord = _coord_with_svc()
        coord._auto_valve_map = {
            "climate.trv1": "number.valve1",
            "climate.trv2": "number.valve2",
        }
        set_hass_states(
            coord,
            {
                "number.valve1": make_state("50"),
                "number.valve2": make_state("30"),
            },
        )
        stats = await coord._async_set_valve_percent(_CFG_TWO, 75.0)
        assert stats.targets_succeeded == 2


# ── §7 Per-device state attribution ──────────────────────────────────────────

class TestPerDeviceState:
    def test_last_written_setpoints_is_empty_at_startup(self):
        coord = make_coordinator()
        assert coord._last_written_setpoints == {}

    def test_trv_offline_set_is_empty_at_startup(self):
        coord = make_coordinator()
        assert coord._trv_offline == set()

    def test_auto_valve_map_is_empty_at_startup(self):
        coord = make_coordinator()
        assert coord._auto_valve_map == {}

    def test_device_profiles_is_empty_at_startup(self):
        coord = make_coordinator()
        assert coord._device_profiles == {}

    async def test_failed_service_does_not_pollute_last_written(self):
        """A failure must not leave stale data in _last_written_setpoints."""
        coord = _coord_with_svc(fail=True)
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert "climate.trv" not in coord._last_written_setpoints

    async def test_success_then_failure_preserves_last_written_from_first_success(self):
        """Second cycle failure should not clear previously-stored setpoint."""
        coord = _coord_with_svc()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        first = coord._last_written_setpoints.get("climate.trv")
        assert first is not None

        coord.hass.services.async_call = AsyncMock(side_effect=Exception("fail"))
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=10.0)})
        await coord._apply_temperature(_CFG_ONE, _REC_NORMAL)
        assert coord._last_written_setpoints.get("climate.trv") == pytest.approx(first)

    async def test_effective_setpoints_by_entity_tracks_per_trv(self):
        """Two TRVs → both appear in effective_setpoints_by_entity with correct values."""
        coord = _coord_with_svc()
        set_hass_states(
            coord,
            {
                "climate.trv1": _trv_state(setpoint=19.0, min_temp=5.0, max_temp=30.0),
                "climate.trv2": _trv_state(setpoint=18.0, min_temp=5.0, max_temp=30.0),
            },
        )
        rec = {"adjusted_target": 21.0, "trv_setpoint": 23.5, "window_open": False}
        stats = await coord._apply_temperature(_CFG_TWO, rec)
        assert "climate.trv1" in stats.effective_setpoints_by_entity
        assert "climate.trv2" in stats.effective_setpoints_by_entity

    async def test_valve_percent_stats_reflect_two_successful_writes(self):
        coord = _coord_with_svc()
        coord._auto_valve_map = {
            "climate.trv1": "number.valve1",
            "climate.trv2": "number.valve2",
        }
        set_hass_states(
            coord,
            {"number.valve1": make_state("10"), "number.valve2": make_state("20")},
        )
        stats = await coord._async_set_valve_percent(_CFG_TWO, 60.0)
        assert stats.targets_total == 2
        assert stats.targets_succeeded == 2
