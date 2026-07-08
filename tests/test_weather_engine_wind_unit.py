"""Regression coverage for wind-speed unit normalization in WeatherEngine.

The config UI explicitly advertises the outdoor wind sensor as accepting
"m/s or km/h" (strings.json / translations), but WeatherEngine used to pass
the raw sensor value straight through to WIND_THRESHOLD_MS (defined in m/s)
with no unit conversion at all — unlike temperature, which is normalized via
to_internal_temperature_c(). A wind sensor reporting km/h (a common
convention for European weather stations) would misfire the wind-chill
boost at roughly 1/3.6 of the intended real-world wind speed. Fixed by
normalizing both the dedicated sensor's unit_of_measurement and the weather
entity's dedicated wind_speed_unit attribute to m/s before use.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.weather_engine import WeatherEngine
from custom_components.thermosmart.const import WIND_CHILL_BOOST


def _sensor_state(value, unit_of_measurement: str | None = None) -> MagicMock:
    st = MagicMock()
    st.state = str(value)
    st.attributes = {"unit_of_measurement": unit_of_measurement} if unit_of_measurement else {}
    return st


def _weather_state(wind_speed=None, wind_speed_unit=None) -> MagicMock:
    st = MagicMock()
    st.state = "sunny"
    attrs: dict = {}
    if wind_speed is not None:
        attrs["wind_speed"] = wind_speed
    if wind_speed_unit is not None:
        attrs["wind_speed_unit"] = wind_speed_unit
    st.attributes = attrs
    return st


def _engine(*, wind_sensor=None, states: dict) -> WeatherEngine:
    hass = MagicMock()
    hass.states.get = MagicMock(side_effect=lambda eid: states.get(eid))
    return WeatherEngine(
        hass=hass, weather_entity="weather.home", outdoor_wind_sensor=wind_sensor,
    )


class TestDedicatedWindSensorUnitNormalization:
    async def test_kmh_sensor_normalized_to_ms(self):
        eng = _engine(
            wind_sensor="sensor.outdoor_wind",
            states={"sensor.outdoor_wind": _sensor_state(36.0, unit_of_measurement="km/h")},
        )
        data = await eng.async_get_data()
        assert data["wind_speed"] == pytest.approx(10.0)

    async def test_ms_sensor_unchanged(self):
        eng = _engine(
            wind_sensor="sensor.outdoor_wind",
            states={"sensor.outdoor_wind": _sensor_state(7.5, unit_of_measurement="m/s")},
        )
        data = await eng.async_get_data()
        assert data["wind_speed"] == pytest.approx(7.5)

    async def test_no_unit_assumed_ms_legacy_behavior(self):
        eng = _engine(
            wind_sensor="sensor.outdoor_wind",
            states={"sensor.outdoor_wind": _sensor_state(6.0)},
        )
        data = await eng.async_get_data()
        assert data["wind_speed"] == pytest.approx(6.0)


class TestWeatherEntityWindSpeedUnitNormalization:
    async def test_kmh_weather_entity_normalized_to_ms(self):
        eng = _engine(states={
            "weather.home": _weather_state(wind_speed=36.0, wind_speed_unit="km/h"),
        })
        data = await eng.async_get_data()
        assert data["wind_speed"] == pytest.approx(10.0)


class TestWindChillBoostRegression:
    async def test_kmh_reading_does_not_misfire_boost(self):
        """Same raw number '15', different unit: 15 km/h ≈ 4.2 m/s (below
        WIND_THRESHOLD_MS=10.0, no boost) vs a genuine 15 m/s (above
        threshold, boost applies) — proves the value is unit-converted, not
        passed through raw."""
        eng = _engine(
            wind_sensor="sensor.outdoor_wind",
            states={"sensor.outdoor_wind": _sensor_state(15.0, unit_of_measurement="km/h")},
        )
        data = await eng.async_get_data()
        data["temperature"] = -5.0

        offset_kmh = eng.compute_temperature_offset(data)
        offset_ms = eng.compute_temperature_offset({**data, "wind_speed": 15.0})

        assert offset_kmh < offset_ms
        assert (offset_ms - offset_kmh) == pytest.approx(WIND_CHILL_BOOST, abs=1e-6)
