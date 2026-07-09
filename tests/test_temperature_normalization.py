"""Regression coverage for outdoor/weather-entity temperature plausibility
filtering in WeatherEngine.

Device Compatibility / Sensor Handling Audit finding: is_plausible_temperature_c()
(temperature_units.py) was already applied to room/TRV-fallback readings
(coordinator.py) but not to the dedicated outdoor temperature sensor or the
weather entity's own temperature attribute in WeatherEngine — a garbage/
sentinel value that survives unit conversion (e.g. a misconfigured sensor
reporting a raw 200) could feed straight into TPI/weather-offset. Fixed by
applying the same shared plausibility band to both WeatherEngine temperature
sources, right after unit normalization, mirroring the guard already used for
room/TRV readings.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.weather_engine import WeatherEngine


def _sensor_state(value, unit_of_measurement: str | None = None) -> MagicMock:
    st = MagicMock()
    st.state = str(value)
    st.attributes = {"unit_of_measurement": unit_of_measurement} if unit_of_measurement else {}
    return st


def _weather_state(temperature=None, temperature_unit=None) -> MagicMock:
    st = MagicMock()
    st.state = "sunny"
    attrs: dict = {}
    if temperature is not None:
        attrs["temperature"] = temperature
    if temperature_unit is not None:
        attrs["temperature_unit"] = temperature_unit
    st.attributes = attrs
    return st


def _engine(*, temp_sensor=None, states: dict) -> WeatherEngine:
    hass = MagicMock()
    hass.config.units.temperature_unit = "°C"
    hass.states.get = MagicMock(side_effect=lambda eid: states.get(eid))
    hass.services.async_call = MagicMock(side_effect=Exception("no forecast service in tests"))
    return WeatherEngine(
        hass=hass, weather_entity="weather.home", outdoor_temp_sensor=temp_sensor,
    )


class TestDedicatedOutdoorSensorPlausibility:
    async def test_implausible_200c_is_ignored(self):
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={"sensor.outdoor_temp": _sensor_state(200.0, unit_of_measurement="°C")},
        )
        data = await eng.async_get_data()
        assert data["temperature"] is None

    async def test_implausible_minus_100_is_ignored(self):
        """A -100°C sentinel some firmwares use as an error code."""
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={"sensor.outdoor_temp": _sensor_state(-100.0, unit_of_measurement="°C")},
        )
        data = await eng.async_get_data()
        assert data["temperature"] is None

    async def test_plausible_cold_value_stays_valid(self):
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={"sensor.outdoor_temp": _sensor_state(-10.0, unit_of_measurement="°C")},
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(-10.0)

    async def test_plausible_warm_value_stays_valid(self):
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={"sensor.outdoor_temp": _sensor_state(35.0, unit_of_measurement="°C")},
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(35.0)

    async def test_fahrenheit_converted_then_plausible(self):
        """68°F -> 20°C, well inside the plausible band."""
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={"sensor.outdoor_temp": _sensor_state(68.0, unit_of_measurement="°F")},
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(20.0, abs=0.1)

    async def test_fahrenheit_converted_then_implausible(self):
        """451°F -> ~232.8°C, well outside the plausible band even after a
        correct unit conversion — a garbage/sentinel Fahrenheit reading must
        not slip through just because the conversion math succeeds."""
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={"sensor.outdoor_temp": _sensor_state(451.0, unit_of_measurement="°F")},
        )
        data = await eng.async_get_data()
        assert data["temperature"] is None


class TestWeatherEntityTemperaturePlausibility:
    async def test_implausible_200c_is_ignored(self):
        eng = _engine(states={
            "weather.home": _weather_state(temperature=200.0, temperature_unit="°C"),
        })
        data = await eng.async_get_data()
        assert data["temperature"] is None

    async def test_plausible_value_stays_valid(self):
        eng = _engine(states={
            "weather.home": _weather_state(temperature=12.5, temperature_unit="°C"),
        })
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(12.5)

    async def test_dedicated_sensor_overrides_implausible_weather_entity(self):
        """Dedicated sensor still takes priority over the weather entity,
        exactly as before this fix — the plausibility guard doesn't change
        that priority order, it just also applies to the weather-entity path."""
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={
                "weather.home": _weather_state(temperature=200.0, temperature_unit="°C"),
                "sensor.outdoor_temp": _sensor_state(-5.0, unit_of_measurement="°C"),
            },
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(-5.0)
