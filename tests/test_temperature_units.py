"""P0/P1 Temperature-Unit / HA-Unit-Compatibility fix regression tests.

Covers:
  - temperature_units.py's to_internal_temperature_c() / from_internal_temperature_c()
    (Celsius/Fahrenheit/Kelvin, missing unit + system-unit fallback, invalid states),
  - coordinator.py's temperature reads (_read_avg_sensor, _read_raw_avg_sensor,
    _read_trv_avg_temp) normalize Fahrenheit sensors correctly,
  - weather_engine.py's outdoor temperature + forecast reads,
  - trv_control.py's min/max clamp + climate.set_temperature dispatch conversion,
  - season.py's frost-protection setpoint read/dispatch,
  - Learning/TPI: a Fahrenheit-reported room temperature yields the identical
    internal Celsius value (and therefore identical downstream behaviour) as
    the equivalent Celsius sensor.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.const import UnitOfTemperature

from custom_components.thermosmart.temperature_units import (
    from_internal_temperature_c,
    to_internal_temperature_c,
)
from custom_components.thermosmart.weather_engine import WeatherEngine
from tests.helpers import make_coordinator, make_state, set_hass_states


def _hass_with_unit(unit: str) -> MagicMock:
    hass = MagicMock()
    hass.config.units.temperature_unit = unit
    return hass


# ── to_internal_temperature_c() ──────────────────────────────────────────────

class TestToInternalTemperatureC:
    def test_celsius_state_unchanged(self):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        assert to_internal_temperature_c(hass, "21.0", unit=UnitOfTemperature.CELSIUS) == \
            pytest.approx(21.0)

    def test_fahrenheit_state_converted(self):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        # 68°F == 20°C
        assert to_internal_temperature_c(hass, "68", unit=UnitOfTemperature.FAHRENHEIT) == \
            pytest.approx(20.0, abs=0.01)

    def test_kelvin_state_converted(self):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        # 293.15 K == 20°C
        assert to_internal_temperature_c(hass, "293.15", unit=UnitOfTemperature.KELVIN) == \
            pytest.approx(20.0, abs=0.01)

    def test_missing_unit_falls_back_to_celsius_system(self):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        assert to_internal_temperature_c(hass, "21.0") == pytest.approx(21.0)

    def test_missing_unit_falls_back_to_fahrenheit_system(self):
        hass = _hass_with_unit(UnitOfTemperature.FAHRENHEIT)
        # no explicit unit -> system unit is Fahrenheit -> 68°F == 20°C
        assert to_internal_temperature_c(hass, "68") == pytest.approx(20.0, abs=0.01)

    def test_unrecognized_unit_falls_back_to_system_unit(self):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        assert to_internal_temperature_c(hass, "21.0", unit="banana") == pytest.approx(21.0)

    @pytest.mark.parametrize("bad", [None, "", "unknown", "unavailable", "None", "not-a-number"])
    def test_invalid_values_return_none(self, bad):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        assert to_internal_temperature_c(hass, bad) is None

    def test_numeric_value_accepted_directly(self):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        assert to_internal_temperature_c(hass, 21.0) == pytest.approx(21.0)


class TestFromInternalTemperatureC:
    def test_celsius_system_unchanged(self):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        assert from_internal_temperature_c(hass, 21.0) == pytest.approx(21.0)

    def test_fahrenheit_system_converted(self):
        hass = _hass_with_unit(UnitOfTemperature.FAHRENHEIT)
        assert from_internal_temperature_c(hass, 20.0) == pytest.approx(68.0, abs=0.01)

    def test_none_stays_none(self):
        hass = _hass_with_unit(UnitOfTemperature.CELSIUS)
        assert from_internal_temperature_c(hass, None) is None

    def test_roundtrip_fahrenheit(self):
        hass = _hass_with_unit(UnitOfTemperature.FAHRENHEIT)
        c = to_internal_temperature_c(hass, "68", unit=UnitOfTemperature.FAHRENHEIT)
        back = from_internal_temperature_c(hass, c)
        assert back == pytest.approx(68.0, abs=0.01)


# ── Coordinator temperature reads ────────────────────────────────────────────

class TestCoordinatorTemperatureReads:
    def test_fahrenheit_room_sensor_normalized(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room": make_state("68.0", {"unit_of_measurement": "°F"}),
        })
        result = coord._read_avg_sensor(["sensor.room"], is_temperature=True)
        assert result == pytest.approx(20.0, abs=0.1)

    def test_celsius_room_sensor_unchanged(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room": make_state("20.0", {"unit_of_measurement": "°C"}),
        })
        result = coord._read_avg_sensor(["sensor.room"], is_temperature=True)
        assert result == pytest.approx(20.0, abs=0.1)

    def test_mixed_unit_sensors_averaged_correctly(self):
        """20°C and 68°F (== 20°C) must average to 20°C, not a nonsense mixed value."""
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.a": make_state("20.0", {"unit_of_measurement": "°C"}),
            "sensor.b": make_state("68.0", {"unit_of_measurement": "°F"}),
        })
        result = coord._read_avg_sensor(["sensor.a", "sensor.b"], is_temperature=True)
        assert result == pytest.approx(20.0, abs=0.1)

    def test_invalid_sensor_ignored_as_before(self):
        coord = make_coordinator()
        set_hass_states(coord, {"sensor.bad": make_state("unknown", {})})
        result = coord._read_avg_sensor(["sensor.bad"], is_temperature=True)
        assert result is None

    def test_humidity_sensor_never_unit_converted(self):
        """is_temperature defaults to False — humidity (%) passes through raw."""
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.hum": make_state("55.0", {"unit_of_measurement": "%"}),
        })
        result = coord._read_avg_sensor(["sensor.hum"])
        assert result == pytest.approx(55.0)

    def test_raw_avg_sensor_normalizes_fahrenheit(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room": make_state("68.0", {"unit_of_measurement": "°F"}),
        })
        result = coord._read_raw_avg_sensor(["sensor.room"])
        assert result == pytest.approx(20.0, abs=0.1)

    def test_trv_current_temperature_fallback_uses_system_unit(self):
        """climate.current_temperature carries no unit attribute — HA already
        normalizes it to the system unit; a Fahrenheit-system coordinator must
        interpret the raw attribute value as Fahrenheit.
        """
        coord = make_coordinator()
        coord.hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 68.0}),
        })
        result = coord._read_trv_avg_temp(["climate.trv"])
        assert result == pytest.approx(20.0, abs=0.1)


# ── Weather / Season ──────────────────────────────────────────────────────────

class TestWeatherEngineUnits:
    async def test_fahrenheit_outdoor_sensor_normalized(self):
        hass = MagicMock()
        hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS
        hass.states.get.side_effect = lambda eid: {
            "weather.home": None,
            "sensor.outdoor": make_state("50.0", {"unit_of_measurement": "°F"}),  # 10°C
        }.get(eid)

        async def _fail(*a, **kw):
            raise RuntimeError("no forecast service in this test")
        hass.services.async_call = AsyncMock(side_effect=_fail)

        engine = WeatherEngine(hass, "weather.home", outdoor_temp_sensor="sensor.outdoor")
        data = await engine.async_get_data()
        assert data["temperature"] == pytest.approx(10.0, abs=0.1)

    async def test_weather_entity_temperature_unit_attribute_used(self):
        hass = MagicMock()
        hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS
        hass.states.get.side_effect = lambda eid: {
            "weather.home": make_state("sunny", {
                "temperature": 68.0, "temperature_unit": "°F",
                "humidity": 40, "wind_speed": 5,
            }),
        }.get(eid)

        async def _fail(*a, **kw):
            raise RuntimeError("no forecast service in this test")
        hass.services.async_call = AsyncMock(side_effect=_fail)

        engine = WeatherEngine(hass, "weather.home")
        data = await engine.async_get_data()
        assert data["temperature"] == pytest.approx(20.0, abs=0.1)
        # humidity/wind must never be unit-converted
        assert data["humidity"] == pytest.approx(40)
        assert data["wind_speed"] == pytest.approx(5)


class TestSeasonFrostProtectionUnits:
    async def test_frost_protection_dispatch_converts_for_fahrenheit_system(self):
        from custom_components.thermosmart.const import TEMP_FROST_PROTECTION

        coord = make_coordinator()
        coord.hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT
        coord.hass.services.async_call = AsyncMock(return_value=None)
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"temperature": 68.0}),  # 20°C, far from frost temp
        })
        await coord._apply_frost_protection({"climate_entities": ["climate.trv"]})

        assert coord.hass.services.async_call.called
        call_data = coord.hass.services.async_call.call_args[0][2]
        expected_f = from_internal_temperature_c(coord.hass, TEMP_FROST_PROTECTION)
        # dispatched value must be TEMP_FROST_PROTECTION converted to °F, not
        # the raw internal °C number passed straight through.
        assert call_data["temperature"] == pytest.approx(expected_f, abs=0.1)
        assert call_data["temperature"] != pytest.approx(TEMP_FROST_PROTECTION, abs=0.1)


# ── TRV/Climate dispatch conversion ──────────────────────────────────────────

class TestClimateSetTemperatureDispatchConversion:
    async def test_dispatch_converts_to_fahrenheit_system_unit(self):
        from tests.helpers import make_zone_config
        coord = make_coordinator(zone_cfg_overrides={"climate_entities": ["climate.trv"]})
        coord.hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT
        coord.hass.services.async_call = AsyncMock(return_value=None)
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {
                "temperature": 60.0, "current_temperature": 60.0,
                "min_temp": 41.0, "max_temp": 86.0,
            }),
        })
        recommendation = {"adjusted_target": 21.0, "trv_setpoint": 21.0, "window_open": False}
        await coord._apply_temperature({"climate_entities": ["climate.trv"], "temp_tolerance": 0.5},
                                       recommendation)

        assert coord.hass.services.async_call.called
        call_data = coord.hass.services.async_call.call_args[0][2]
        # 21°C dispatched must be ~69.8°F, never the raw 21 passed straight through
        assert call_data["temperature"] == pytest.approx(69.8, abs=0.2)

    async def test_dispatch_unchanged_for_celsius_system(self):
        coord = make_coordinator(zone_cfg_overrides={"climate_entities": ["climate.trv"]})
        coord.hass.services.async_call = AsyncMock(return_value=None)
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {
                "temperature": 18.0, "current_temperature": 18.0,
                "min_temp": 5.0, "max_temp": 30.0,
            }),
        })
        recommendation = {"adjusted_target": 21.0, "trv_setpoint": 21.0, "window_open": False}
        await coord._apply_temperature({"climate_entities": ["climate.trv"], "temp_tolerance": 0.5},
                                       recommendation)
        call_data = coord.hass.services.async_call.call_args[0][2]
        assert call_data["temperature"] == pytest.approx(21.0, abs=0.1)


# ── Learning/TPI: unit-independent internal representation ──────────────────

class TestLearningReceivesUnitIndependentCelsius:
    def test_fahrenheit_and_celsius_sensors_yield_identical_internal_reading(self):
        """Whatever feeds HeatRate/HeatLoss/TPI must be identical regardless of
        the reporting sensor's unit — the coordinator is the single
        normalization point; models must never see a raw Fahrenheit float.
        """
        coord_f = make_coordinator()
        set_hass_states(coord_f, {
            "sensor.room": make_state("68.0", {"unit_of_measurement": "°F"}),
        })
        result_f = coord_f._read_avg_sensor(["sensor.room"], is_temperature=True)

        coord_c = make_coordinator()
        set_hass_states(coord_c, {
            "sensor.room": make_state("20.0", {"unit_of_measurement": "°C"}),
        })
        result_c = coord_c._read_avg_sensor(["sensor.room"], is_temperature=True)

        assert result_f == pytest.approx(result_c, abs=0.1)
