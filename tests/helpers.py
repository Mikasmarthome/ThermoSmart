"""Shared test helpers and mock factories for ThermoSmart tests."""
from __future__ import annotations

from unittest.mock import MagicMock


def make_zone_config(**overrides) -> dict:
    """Return a minimal valid zone configuration dict."""
    defaults: dict = {
        "entry_type": "zone",
        "name": "Test Zone",
        "climate_entities": ["climate.test_trv"],
        "temp_sensors": ["sensor.test_temp"],
        "humidity_sensors": [],
        "window_sensors": [],
        "comfort_temp": 21.0,
        "night_temp": 18.0,
        "away_temp": 17.0,
        "vacation_temp": 12.0,
        "eco_temp": 19.0,
        "window_open_temp": 5.0,
        "temp_tolerance": 0.5,
        "schedule_enabled": True,
        "sched_wd_morning": "06:00",
        "sched_wd_night": "22:00",
        "sched_we_morning": "08:00",
        "sched_we_night": "23:00",
        "learning_enabled": True,
        "window_open_delay": 5,
        "window_close_delay": 2,
        "valve_maintenance": True,
        "manage_temp_source": False,
        "calibration_invert": False,
    }
    defaults.update(overrides)
    return defaults


def make_weather_data(
    outdoor_temp: float | None = 10.0,
    humidity: float | None = 70.0,
    wind_speed: float | None = None,
    solar_radiation: float | None = None,
    forecast_high: float | None = None,
    forecast_low: float | None = None,
    condition: str | None = "sunny",
    rain: float | None = None,
) -> dict:
    """Return a weather data dict matching WeatherEngine.async_get_data() output."""
    data: dict = {}
    if outdoor_temp is not None:
        data["temperature"] = outdoor_temp
    if humidity is not None:
        data["humidity"] = humidity
    if wind_speed is not None:
        data["wind_speed"] = wind_speed
    if solar_radiation is not None:
        data["solar_radiation"] = solar_radiation
    if forecast_high is not None:
        data["forecast_high"] = forecast_high
    if forecast_low is not None:
        data["forecast_low"] = forecast_low
    if condition is not None:
        data["weather_condition"] = condition
    if rain is not None:
        data["rain"] = rain
    return data


def make_mock_hass() -> MagicMock:
    """Return a minimal hass mock for unit tests that instantiate engines."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.services = MagicMock()
    return hass
