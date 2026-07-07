"""Device Compatibility / TRV Quirk Audit — P0/P1/P2-small fix block regression tests.

Covers:
  - P0: is_plausible_temperature_c() helper + its application in
    coordinator._filter_sensor_value() (EMA baseline garbage guard) and
    coordinator._read_trv_avg_temp() (TRV-only fallback garbage guard).
  - P1: maintenance.py valve exercise routed through the same
    resolve_device_effective_setpoint() clamp/step logic as regular dispatch.
  - P1: per-entity consecutive dispatch-failure counter + rate-limited
    Support Critical Event after DISPATCH_FAILURE_SUPPORT_EVENT_THRESHOLD.
  - P2 (small): dual-setpoint / HEAT_COOL-only entities get a one-time
    warning and are skipped, never a silent per-cycle failure loop.
  - Heating-Failure-Confounder verification: recommendation["heating_failure"]
    reaches ZoneRuntimeInput.heating_failure (the shared input contract feeding
    HeatRate/HeatLoss/Afterheat/Onset/Boost/Outcome builders and models).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from homeassistant.const import UnitOfTemperature

from custom_components.thermosmart.const import DISPATCH_FAILURE_SUPPORT_EVENT_THRESHOLD
from custom_components.thermosmart.learning.decision.baseline import (
    zone_input_from_recommendation,
)
from custom_components.thermosmart.learning.support_event_schemas import (
    SupportEventType,
)
from custom_components.thermosmart.temperature_units import (
    PLAUSIBLE_TEMPERATURE_MAX_C,
    PLAUSIBLE_TEMPERATURE_MIN_C,
    is_plausible_temperature_c,
    to_internal_temperature_c,
)
from custom_components.thermosmart.trv_control import _DispatchStats

from tests.helpers import make_coordinator, make_state, set_hass_states

_CFG = {
    "climate_entities": ["climate.trv"],
    "temp_tolerance": 0.5,
    "window_open_temp": 5.0,
}
_REC = {"adjusted_target": 21.0, "trv_setpoint": 23.5, "window_open": False}


def _trv_state(setpoint: float = 19.0, *, supported_features: int = 1) -> object:
    return make_state(
        "heat",
        {
            "temperature": setpoint,
            "current_temperature": 18.5,
            "min_temp": 5.0,
            "max_temp": 35.0,
            "target_temp_step": 0.5,
            "supported_features": supported_features,
        },
    )


# ── P0: is_plausible_temperature_c() helper ──────────────────────────────────

class TestIsPlausibleTemperatureC:
    def test_none_is_implausible(self):
        assert is_plausible_temperature_c(None) is False

    def test_nan_is_implausible(self):
        assert is_plausible_temperature_c(float("nan")) is False

    def test_inf_is_implausible(self):
        assert is_plausible_temperature_c(float("inf")) is False
        assert is_plausible_temperature_c(float("-inf")) is False

    def test_zigbee_sentinel_127_is_implausible(self):
        assert is_plausible_temperature_c(127.0) is False

    def test_zigbee_sentinel_126_5_is_implausible(self):
        assert is_plausible_temperature_c(126.5) is False

    def test_error_code_minus_100_is_implausible(self):
        assert is_plausible_temperature_c(-100.0) is False

    def test_extreme_high_is_implausible(self):
        assert is_plausible_temperature_c(999.0) is False

    def test_normal_room_temperature_is_plausible(self):
        assert is_plausible_temperature_c(21.0) is True

    def test_normal_outdoor_cold_is_plausible(self):
        assert is_plausible_temperature_c(-15.0) is True

    def test_band_boundaries_inclusive(self):
        assert is_plausible_temperature_c(PLAUSIBLE_TEMPERATURE_MIN_C) is True
        assert is_plausible_temperature_c(PLAUSIBLE_TEMPERATURE_MAX_C) is True
        assert is_plausible_temperature_c(PLAUSIBLE_TEMPERATURE_MIN_C - 0.1) is False
        assert is_plausible_temperature_c(PLAUSIBLE_TEMPERATURE_MAX_C + 0.1) is False

    def test_context_accepted_but_does_not_affect_result(self):
        assert is_plausible_temperature_c(21.0, context="sensor.room") is True
        assert is_plausible_temperature_c(127.0, context="sensor.room") is False

    def test_non_numeric_string_is_implausible(self):
        assert is_plausible_temperature_c("not_a_number") is False

    def test_never_raises_on_garbage_input(self):
        assert is_plausible_temperature_c(object()) is False
        assert is_plausible_temperature_c([1, 2, 3]) is False


# ── P0: coordinator._filter_sensor_value() garbage-baseline guard ───────────

class TestFilterSensorValueGarbageGuard:
    def test_first_garbage_value_not_stored_as_baseline(self):
        coord = make_coordinator()
        result = coord._filter_sensor_value("sensor.temp", 127.0)
        assert result is None
        assert "sensor.temp" not in coord._sensor_ema

    def test_first_garbage_minus_100_not_stored_as_baseline(self):
        coord = make_coordinator()
        result = coord._filter_sensor_value("sensor.temp", -100.0)
        assert result is None
        assert "sensor.temp" not in coord._sensor_ema

    def test_next_plausible_value_becomes_baseline_normally(self):
        coord = make_coordinator()
        coord._filter_sensor_value("sensor.temp", 127.0)  # garbage, rejected
        result = coord._filter_sensor_value("sensor.temp", 21.0)  # first real value
        assert result == pytest.approx(21.0)
        assert coord._sensor_ema["sensor.temp"] == pytest.approx(21.0)

    def test_plausible_value_after_garbage_not_treated_as_spike(self):
        coord = make_coordinator()
        coord._filter_sensor_value("sensor.temp", 127.0)  # garbage, rejected
        coord._filter_sensor_value("sensor.temp", 21.0)    # baseline = 21.0
        result = coord._filter_sensor_value("sensor.temp", 21.3)  # normal follow-up reading
        # A genuine spike (> NOISE_FILTER_SPIKE_THRESHOLD from the 21.0 baseline)
        # would return None — this small, real follow-up reading must not.
        assert result is not None
        assert 21.0 <= result <= 21.3

    def test_normal_celsius_values_unaffected(self):
        coord = make_coordinator()
        first = coord._filter_sensor_value("sensor.temp", 20.0)
        second = coord._filter_sensor_value("sensor.temp", 20.2)
        assert first == pytest.approx(20.0)
        assert second is not None

    def test_genuine_spike_still_rejected_after_valid_baseline(self):
        """The existing spike-detection behavior must survive unchanged."""
        coord = make_coordinator()
        coord._filter_sensor_value("sensor.temp", 20.0)  # baseline
        result = coord._filter_sensor_value("sensor.temp", 30.0)  # spike, > threshold
        assert result is None


# ── P0: coordinator._read_trv_avg_temp() TRV-only fallback guard ───────────

class TestReadTrvAvgTempGarbageGuard:
    def test_garbage_trv_fallback_ignored(self):
        coord = make_coordinator({"climate_entities": ["climate.trv"], "temp_sensors": []})
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 127.0}),
        })
        assert coord._read_trv_avg_temp(["climate.trv"]) is None

    def test_minus_100_trv_fallback_ignored(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": -100.0}),
        })
        assert coord._read_trv_avg_temp(["climate.trv"]) is None

    def test_plausible_trv_fallback_accepted(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 20.5}),
        })
        assert coord._read_trv_avg_temp(["climate.trv"]) == pytest.approx(20.5)

    def test_one_garbage_one_plausible_trv_only_plausible_averaged(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv1": make_state("heat", {"current_temperature": 126.5}),
            "climate.trv2": make_state("heat", {"current_temperature": 20.0}),
        })
        assert coord._read_trv_avg_temp(["climate.trv1", "climate.trv2"]) == pytest.approx(20.0)


# ── P0: Fahrenheit-unit-fix + plausibility-guard interplay ──────────────────

class TestFahrenheitPlausibilityInterplay:
    def test_normal_fahrenheit_room_temp_converts_and_is_plausible(self):
        hass = MagicMock()
        hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS
        c = to_internal_temperature_c(hass, "68", unit=UnitOfTemperature.FAHRENHEIT)
        assert c == pytest.approx(20.0, abs=0.01)
        assert is_plausible_temperature_c(c) is True

    def test_fahrenheit_sentinel_260_6_converts_to_implausible_127c(self):
        """260.6°F == 127°C — a Fahrenheit-reported Zigbee sentinel must be
        caught by the plausibility guard just like a native-Celsius one."""
        hass = MagicMock()
        hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS
        c = to_internal_temperature_c(hass, "260.6", unit=UnitOfTemperature.FAHRENHEIT)
        assert c == pytest.approx(127.0, abs=0.1)
        assert is_plausible_temperature_c(c) is False


# ── P1: Maintenance routed through resolve_device_effective_setpoint() ──────

class TestMaintenanceClampedDispatch:
    async def _run_and_capture(self, coord, *, min_temp, max_temp, step, boost_temp,
                               target_after, current_temp=19.0):
        calls = []

        async def _svc(domain, service, data, *, blocking=True, **kw):
            calls.append(data)

        coord.hass.services.async_call = _svc
        coord.hass.async_create_task = lambda coro: asyncio.ensure_future(coro)
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {
                "temperature": current_temp,
                "min_temp": min_temp,
                "max_temp": max_temp,
                "target_temp_step": step,
            }),
        })
        coord._device_profiles = {}
        coord._is_summer = True
        coord._maintenance_running = False
        coord._last_maintenance = None
        from custom_components.thermosmart.learning.clock import FakeClock
        from datetime import datetime, timezone
        coord._clock = FakeClock(datetime(2025, 1, 12, 3, 0, tzinfo=timezone.utc))  # Sunday 03:00
        cfg = {
            "climate_entities": ["climate.trv"],
            "valve_maintenance": True,
            "vacation_temp": 12.0,
            "comfort_temp": 21.0,
        }
        rec = {"mode": "auto", "adjusted_target": target_after}
        await coord._async_valve_maintenance(cfg, rec)
        task = coord._maintenance_task
        assert task is not None
        await task
        return calls

    async def test_unusual_min_clamps_boost_temperature(self):
        """VALVE_MAINTENANCE_BOOST_TEMP is high (28°C); an unusually low device
        max_temp must clamp it down instead of sending the raw constant."""
        coord = make_coordinator()
        calls = await self._run_and_capture(
            coord, min_temp=5.0, max_temp=22.0, step=0.5,
            boost_temp=30.0, target_after=21.0,
        )
        assert len(calls) == 2  # boost dispatch + return dispatch
        boost_call = calls[0]
        assert boost_call["entity_id"] == "climate.trv"
        # Must be clamped to the device max (22.0), never the raw 30.0 constant.
        assert boost_call["temperature"] <= 22.0

    async def test_step_snap_applied_to_return_temperature(self):
        coord = make_coordinator()
        calls = await self._run_and_capture(
            coord, min_temp=5.0, max_temp=35.0, step=1.0,
            boost_temp=30.0, target_after=21.3,
        )
        return_call = calls[1]
        # Snapped to a 1.0°C step grid — never the raw 21.3 constant.
        assert return_call["temperature"] % 1.0 == pytest.approx(0.0, abs=1e-6)

    async def test_no_double_unit_conversion_in_fahrenheit_system(self):
        coord = make_coordinator()
        coord.hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT
        # min_temp/max_temp are HA climate capability attributes — already
        # normalized to the system unit by HA itself, so a Fahrenheit-system
        # device reports Fahrenheit-equivalent bounds here (41-95°F == 5-35°C),
        # not the raw Celsius numbers.
        calls = await self._run_and_capture(
            coord, min_temp=41.0, max_temp=95.0, step=0.5,
            boost_temp=30.0, target_after=21.0,
        )
        boost_call = calls[0]
        # VALVE_MAINTENANCE_BOOST_TEMP (28°C) clamped/snapped then converted
        # ONCE to °F system unit (28°C == 82.4°F), never converted twice
        # (which would land far outside any real range, e.g. ~180).
        assert 80.0 <= boost_call["temperature"] <= 90.0


# ── P1: per-entity consecutive dispatch-failure counter + Support Event ────

class TestDispatchFailureCounter:
    def test_failed_entity_ids_populated_on_failure(self):
        stats = _DispatchStats()
        stats.record(Exception("boom"), entity_id="climate.trv")
        assert stats.failed_entity_ids == ["climate.trv"]

    def test_success_does_not_populate_failed_entity_ids(self):
        stats = _DispatchStats()
        stats.record(None, effective_c=21.0, entity_id="climate.trv")
        assert stats.failed_entity_ids == []

    def test_merge_combines_failed_entity_ids(self):
        a = _DispatchStats()
        a.record(Exception("x"), entity_id="climate.trv1")
        b = _DispatchStats()
        b.record(Exception("y"), entity_id="climate.trv2")
        merged = a.merge(b)
        assert set(merged.failed_entity_ids) == {"climate.trv1", "climate.trv2"}

    def test_counter_increments_on_repeated_failure(self):
        coord = make_coordinator()
        stats = _DispatchStats()
        stats.record(Exception("boom"), entity_id="climate.trv")
        coord._update_dispatch_failure_counters(stats)
        coord._update_dispatch_failure_counters(stats)
        assert coord._dispatch_consecutive_failures["climate.trv"] == 2

    def test_success_resets_counter(self):
        coord = make_coordinator()
        fail_stats = _DispatchStats()
        fail_stats.record(Exception("boom"), entity_id="climate.trv")
        coord._update_dispatch_failure_counters(fail_stats)
        coord._update_dispatch_failure_counters(fail_stats)
        assert coord._dispatch_consecutive_failures["climate.trv"] == 2

        success_stats = _DispatchStats()
        success_stats.record(None, effective_c=21.0, entity_id="climate.trv")
        coord._update_dispatch_failure_counters(success_stats)
        assert coord._dispatch_consecutive_failures["climate.trv"] == 0

    def test_crossing_returns_true_exactly_at_threshold(self):
        coord = make_coordinator()
        stats = _DispatchStats()
        stats.record(Exception("boom"), entity_id="climate.trv")
        crossed = False
        for _ in range(DISPATCH_FAILURE_SUPPORT_EVENT_THRESHOLD):
            crossed = coord._update_dispatch_failure_counters(stats)
        assert crossed is True

    def test_no_crossing_below_threshold(self):
        coord = make_coordinator()
        stats = _DispatchStats()
        stats.record(Exception("boom"), entity_id="climate.trv")
        crossed = False
        for _ in range(DISPATCH_FAILURE_SUPPORT_EVENT_THRESHOLD - 1):
            crossed = coord._update_dispatch_failure_counters(stats)
        assert crossed is False

    def test_support_event_fires_exactly_once_at_threshold(self):
        coord = make_coordinator()
        coord._le2_shadow = MagicMock()
        stats = _DispatchStats()
        stats.record(Exception("boom"), entity_id="climate.trv")

        for _ in range(DISPATCH_FAILURE_SUPPORT_EVENT_THRESHOLD):
            crossed = coord._update_dispatch_failure_counters(stats)
            coord._maybe_record_dispatch_failure_event(crossed)

        assert coord._le2_shadow.record_support_critical_event_safe.call_count == 1
        event = coord._le2_shadow.record_support_critical_event_safe.call_args[0][0]
        assert event.event_type == SupportEventType.TRV_COMMAND_FAILED

    def test_no_spam_beyond_threshold(self):
        coord = make_coordinator()
        coord._le2_shadow = MagicMock()
        stats = _DispatchStats()
        stats.record(Exception("boom"), entity_id="climate.trv")

        for _ in range(DISPATCH_FAILURE_SUPPORT_EVENT_THRESHOLD + 5):
            crossed = coord._update_dispatch_failure_counters(stats)
            coord._maybe_record_dispatch_failure_event(crossed)

        assert coord._le2_shadow.record_support_critical_event_safe.call_count == 1

    def test_event_refires_after_recovery_and_new_failure_streak(self):
        coord = make_coordinator()
        coord._le2_shadow = MagicMock()
        fail_stats = _DispatchStats()
        fail_stats.record(Exception("boom"), entity_id="climate.trv")
        success_stats = _DispatchStats()
        success_stats.record(None, effective_c=21.0, entity_id="climate.trv")

        for _ in range(DISPATCH_FAILURE_SUPPORT_EVENT_THRESHOLD):
            coord._maybe_record_dispatch_failure_event(
                coord._update_dispatch_failure_counters(fail_stats))
        assert coord._le2_shadow.record_support_critical_event_safe.call_count == 1

        coord._update_dispatch_failure_counters(success_stats)  # recovers

        for _ in range(DISPATCH_FAILURE_SUPPORT_EVENT_THRESHOLD):
            coord._maybe_record_dispatch_failure_event(
                coord._update_dispatch_failure_counters(fail_stats))
        assert coord._le2_shadow.record_support_critical_event_safe.call_count == 2

    def test_no_event_without_shadow_attached(self):
        coord = make_coordinator()
        assert coord._le2_shadow is None
        coord._maybe_record_dispatch_failure_event(True)  # must not raise


# ── P2 (small): dual-setpoint / HEAT_COOL-only detection ────────────────────

class TestDualSetpointDetection:
    async def test_entity_without_target_temperature_is_skipped_and_warned(self, caplog):
        coord = make_coordinator()
        coord.hass.services.async_call = MagicMock()
        set_hass_states(coord, {
            "climate.trv": _trv_state(supported_features=2),  # TARGET_TEMPERATURE_RANGE only
        })
        stats = await coord._apply_temperature(_CFG, _REC)
        assert stats.status == "not_attempted"
        coord.hass.services.async_call.assert_not_called()
        assert "climate.trv" in coord._dual_setpoint_warned

    async def test_entity_with_target_temperature_dispatches_normally(self):
        coord = make_coordinator()
        called = []

        async def _svc(domain, service, data, *, blocking=True, **kw):
            called.append(data)

        coord.hass.services.async_call = _svc
        set_hass_states(coord, {
            "climate.trv": _trv_state(supported_features=1),  # TARGET_TEMPERATURE
        })
        stats = await coord._apply_temperature(_CFG, _REC)
        assert stats.status == "fully_succeeded"
        assert len(called) == 1

    async def test_warning_issued_only_once_across_cycles(self):
        coord = make_coordinator()
        coord.hass.services.async_call = MagicMock()
        set_hass_states(coord, {
            "climate.trv": _trv_state(supported_features=2),
        })
        await coord._apply_temperature(_CFG, _REC)
        await coord._apply_temperature(_CFG, _REC)
        assert coord._dual_setpoint_warned == {"climate.trv"}

    async def test_dual_setpoint_skip_does_not_count_toward_failure_counter(self):
        coord = make_coordinator()
        coord.hass.services.async_call = MagicMock()
        set_hass_states(coord, {
            "climate.trv": _trv_state(supported_features=2),
        })
        stats = await coord._apply_temperature(_CFG, _REC)
        coord._update_dispatch_failure_counters(stats)
        assert "climate.trv" not in coord._dispatch_consecutive_failures

    async def test_missing_supported_features_attribute_does_not_crash(self):
        """A device that omits supported_features entirely (attributes.get
        default 0) has no positive dual-setpoint-only evidence — treated as
        single-setpoint-capable (matching prior behavior before this guard
        existed), not silently skipped. Must not crash either way."""
        coord = make_coordinator()

        async def _svc(domain, service, data, *, blocking=True, **kw):
            pass

        coord.hass.services.async_call = _svc
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {
                "temperature": 19.0, "min_temp": 5.0, "max_temp": 35.0,
                "target_temp_step": 0.5,
            }),
        })
        stats = await coord._apply_temperature(_CFG, _REC)
        assert stats.status == "fully_succeeded"
        assert "climate.trv" not in coord._dual_setpoint_warned


# ── Heating-Failure-Confounder verification ─────────────────────────────────

class TestHeatingFailureConfounderReachesBuilders:
    def test_heating_failure_true_propagates_into_zone_runtime_input(self):
        rec = {**_REC, "heating_failure": True}
        zin = zone_input_from_recommendation("zone1", "2025-01-01T00:00:00+00:00", rec)
        assert zin.heating_failure is True

    def test_heating_failure_false_propagates_into_zone_runtime_input(self):
        rec = {**_REC, "heating_failure": False}
        zin = zone_input_from_recommendation("zone1", "2025-01-01T00:00:00+00:00", rec)
        assert zin.heating_failure is False

    def test_missing_heating_failure_key_defaults_false_no_crash(self):
        rec = {**_REC}
        rec.pop("heating_failure", None)
        zin = zone_input_from_recommendation("zone1", "2025-01-01T00:00:00+00:00", rec)
        assert zin.heating_failure is False
