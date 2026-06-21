"""Shared test helpers and mock factories for ThermoSmart tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def make_state(value, attributes: dict | None = None) -> MagicMock:
    """Return a minimal mock HA State object."""
    state = MagicMock()
    state.state = str(value)
    state.attributes = attributes or {}
    return state


def set_hass_states(coordinator, states_dict: dict) -> None:
    """Configure coordinator.hass.states.get() to return mock states from a dict."""
    coordinator.hass.states.get.side_effect = lambda eid: states_dict.get(eid)


def make_coordinator(zone_cfg_overrides: dict | None = None):
    """Create a ThermoSmartCoordinator with mocked dependencies for unit testing.

    Uses MagicMock hass – suitable for testing pure logic methods without HA
    fixtures. Compatible with Windows (no fcntl/homeassistant.runner needed).

    DataUpdateCoordinator.__init__ calls homeassistant.helpers.frame.report_usage()
    which requires an active HA event loop.  We patch it to a no-op so the
    coordinator can be instantiated in plain pytest without a full HA runtime.
    """
    from unittest.mock import patch as _patch
    from custom_components.thermosmart.coordinator import ThermoSmartCoordinator

    hass = make_mock_hass()

    entry = MagicMock()
    entry.entry_id = "test_zone_001"
    entry.options = {}
    entry.data = make_zone_config(**(zone_cfg_overrides or {}))

    weather_engine = MagicMock()
    weather_engine.compute_temperature_offset.return_value = 0.0
    weather_engine.compute_forecast_suppression.return_value = 1.0
    weather_engine.async_get_data = AsyncMock(return_value={"temperature": 10.0})

    learning_engine = MagicMock()
    learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
    learning_engine.async_get_preheat_minutes = AsyncMock(return_value=0)
    learning_engine.get_forecast_bias.return_value = 1.0
    learning_engine.get_confidence.return_value = 0.5
    learning_engine.get_tpi_coefficients.return_value = (0.6, 0.01)
    learning_engine.get_boost_factor.return_value = 1.0
    learning_engine.async_observe = AsyncMock()
    learning_engine.evaluate_forecast_decisions = MagicMock()
    learning_engine.update_heating_session = MagicMock()
    learning_engine.record_forecast_decision = MagicMock()

    with _patch("homeassistant.helpers.frame.report_usage"):
        return ThermoSmartCoordinator(hass, entry, weather_engine, learning_engine)


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


def make_learning_engine():
    """Create a LearningEngine with a mocked hass for pure-logic unit tests.

    Suitable for Welle 4a characterization tests: methods that only read or
    compute derived values, or mutate in-memory dicts.  The underlying Store is
    never written to in these tests — persistence is covered separately in a
    later wave.
    """
    from custom_components.thermosmart.learning_engine import LearningEngine

    return LearningEngine(make_mock_hass())
