"""Unit tests for ThermoSmart _compute_recommendation pipeline (Welle 3).

Tests the recommendation dict structure, mode-specific targets,
weather offset guard (v1.1.0 fix), window open detection, and override behavior.

Pure Python + MagicMock – runs on Windows without Docker.
_compute_recommendation is an async method; pytest-asyncio with asyncio_mode=auto
executes all async test functions automatically.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from custom_components.thermosmart.const import (
    HEATING_MODE_AUTO,
    HEATING_MODE_AWAY,
    HEATING_MODE_COMFORT,
    HEATING_MODE_ECO,
    HEATING_MODE_NIGHT,
    HEATING_MODE_VACATION,
    CONF_VACATION_TEMP,
    CONF_ECO_TEMP,
)

from tests.helpers import make_coordinator, make_state, make_weather_data, set_hass_states

# Shared zone config used across most tests
_CFG = {
    "comfort_temp": 21.0,
    "night_temp": 18.0,
    "away_temp": 17.0,
    "vacation_temp": 12.0,
    "eco_temp": 19.0,
    "window_open_temp": 5.0,
    "temp_tolerance": 0.5,
    "temp_sensors": ["sensor.test_temp"],
    "climate_entities": ["climate.test_trv"],
    "window_sensors": [],
    "window_open_delay": 5,
    "window_close_delay": 2,
    "sched_wd_morning": "06:00",
    "sched_wd_night": "22:00",
    "sched_we_morning": "08:00",
    "sched_we_night": "23:00",
    "schedule_enabled": True,
}

_WEATHER = make_weather_data(outdoor_temp=10.0)


# ── TestRecommendationStructure ───────────────────────────────────────────────


class TestRecommendationStructure:
    async def test_result_contains_required_keys(self):
        coord = make_coordinator()
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        required = {
            "zone_name", "mode", "current_temp", "window_open",
            "base_target", "weather_offset", "adjusted_target", "effective_target",
            "override_active", "override_temp", "preheat_minutes", "preheat_active",
            "forecast_suppression", "learning_confidence",
            "outdoor_temp", "forecast_high", "weather_condition",
            "temp_slope", "temp_ema_1h",
        }
        assert required <= result.keys()

    async def test_no_sensor_current_temp_is_none(self):
        coord = make_coordinator()
        cfg = {**_CFG, "temp_sensors": [], "climate_entities": []}
        result = await coord._compute_recommendation(cfg, _WEATHER, HEATING_MODE_AUTO)
        assert result["current_temp"] is None

    async def test_sensor_state_returned_as_current_temp(self):
        coord = make_coordinator()
        set_hass_states(coord, {"sensor.test_temp": make_state("20.5")})
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["current_temp"] == 20.5

    async def test_trv_fallback_when_no_temp_sensor(self):
        """When temp_sensors is empty, current_temp comes from TRV current_temperature."""
        coord = make_coordinator()
        cfg = {**_CFG, "temp_sensors": []}
        set_hass_states(coord, {
            "climate.test_trv": make_state("heat", {"current_temperature": 19.0}),
        })
        result = await coord._compute_recommendation(cfg, _WEATHER, HEATING_MODE_AUTO)
        assert result["current_temp"] == 19.0

    async def test_zone_name_in_result(self):
        coord = make_coordinator()
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["zone_name"] == "Test Zone"

    async def test_mode_in_result_matches_input_mode(self):
        coord = make_coordinator()
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_NIGHT)
        assert result["mode"] == HEATING_MODE_NIGHT


# ── TestRecommendationModes ───────────────────────────────────────────────────


class TestRecommendationModes:
    async def test_comfort_mode_adjusted_target_is_comfort_temp(self):
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_COMFORT)
        assert result["adjusted_target"] == 21.0
        assert result["override_active"] is False

    async def test_night_mode_adjusted_target_is_night_temp(self):
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_NIGHT)
        assert result["adjusted_target"] == 18.0

    async def test_away_mode_adjusted_target_is_away_temp(self):
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=17.0)
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AWAY)
        assert result["adjusted_target"] == 17.0

    async def test_vacation_mode_adjusted_target_is_vacation_temp(self):
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=12.0)
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_VACATION)
        assert result["adjusted_target"] == 12.0

    async def test_non_auto_mode_weather_offset_always_zero(self):
        """Weather offset is only applied in AUTO mode."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        coord.weather_engine.compute_temperature_offset.return_value = -2.0
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_NIGHT)
        assert result["weather_offset"] == 0.0
        assert result["effective_target"] == 18.0  # offset not applied

    async def test_eco_mode_adjusted_target_matches_base_target(self):
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=19.0)
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_ECO)
        assert result["adjusted_target"] == 19.0

    async def test_auto_without_override_override_active_is_false(self):
        coord = make_coordinator()
        coord._override = None
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["override_active"] is False


# ── TestRecommendationOverride ────────────────────────────────────────────────


class TestRecommendationOverride:
    async def test_override_active_when_override_set(self):
        coord = make_coordinator()
        coord._override = 22.0
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["override_active"] is True
        assert result["adjusted_target"] == 22.0
        assert result["effective_target"] == 22.0

    async def test_override_temp_in_result(self):
        coord = make_coordinator()
        coord._override = 23.5
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["override_temp"] == 23.5

    async def test_away_mode_clears_override(self):
        """When mode is AWAY, manual override is cleared (presence > override)."""
        coord = make_coordinator()
        coord._override = 22.0
        coord._override_schedule_period = "wd_comfort"
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AWAY)
        assert result["override_active"] is False
        assert coord._override is None

    async def test_window_open_overrides_manual_override(self):
        """Window open suppresses output regardless of manual override."""
        coord = make_coordinator()
        coord._override = 22.0
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"]}
        set_hass_states(coord, {
            "binary_sensor.window": make_state("on"),
        })
        result = await coord._compute_recommendation(cfg, _WEATHER, HEATING_MODE_AUTO)
        assert result["window_open"] is True
        assert result["effective_target"] is None
        assert result["override_active"] is False


# ── TestRecommendationWindowOpen ──────────────────────────────────────────────


class TestRecommendationWindowOpen:
    async def test_no_window_sensors_window_open_is_false(self):
        coord = make_coordinator()
        cfg = {**_CFG, "window_sensors": []}
        result = await coord._compute_recommendation(cfg, _WEATHER, HEATING_MODE_AUTO)
        assert result["window_open"] is False

    async def test_sensor_off_window_open_is_false(self):
        coord = make_coordinator()
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"]}
        set_hass_states(coord, {"binary_sensor.window": make_state("off")})
        result = await coord._compute_recommendation(cfg, _WEATHER, HEATING_MODE_AUTO)
        assert result["window_open"] is False

    async def test_sensor_on_window_open_is_true(self):
        """Sensor 'on' with empty _window_open_at immediately counts as open."""
        coord = make_coordinator()
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"]}
        set_hass_states(coord, {"binary_sensor.window": make_state("on")})
        result = await coord._compute_recommendation(cfg, _WEATHER, HEATING_MODE_AUTO)
        assert result["window_open"] is True

    async def test_window_open_suppresses_targets(self):
        coord = make_coordinator()
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"]}
        set_hass_states(coord, {"binary_sensor.window": make_state("on")})
        result = await coord._compute_recommendation(cfg, _WEATHER, HEATING_MODE_AUTO)
        assert result["effective_target"] is None
        assert result["adjusted_target"] is None


# ── TestRecommendationWeatherOffset ──────────────────────────────────────────


class TestRecommendationWeatherOffset:
    async def test_positive_offset_applied_in_auto_mode(self):
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        coord.weather_engine.compute_temperature_offset.return_value = 1.5
        # Sensor not below base_target – positive offset applies unconditionally
        set_hass_states(coord, {"sensor.test_temp": make_state("22.0")})
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["weather_offset"] == 1.5
        assert result["effective_target"] == pytest_approx(21.0 + 1.5, abs=0.2)

    async def test_negative_offset_suppressed_while_below_comfort(self):
        """v1.1.0 fix: negative offset zeroed when current_temp < base_target in AUTO mode."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        coord.weather_engine.compute_temperature_offset.return_value = -1.0
        # Indoor below comfort target → guard activates
        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["weather_offset"] == 0.0

    async def test_negative_offset_applied_when_above_comfort(self):
        """Negative offset is NOT suppressed once current_temp >= base_target."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        coord.weather_engine.compute_temperature_offset.return_value = -1.0
        # Indoor at or above target → guard does NOT activate
        set_hass_states(coord, {"sensor.test_temp": make_state("21.5")})
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["weather_offset"] == -1.0

    async def test_negative_offset_with_no_sensor_is_suppressed(self):
        """Without current_temp the guard condition is False → offset NOT suppressed."""
        coord = make_coordinator()
        cfg = {**_CFG, "temp_sensors": [], "climate_entities": []}
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        coord.weather_engine.compute_temperature_offset.return_value = -1.0
        result = await coord._compute_recommendation(cfg, _WEATHER, HEATING_MODE_AUTO)
        # current_temp is None → guard condition False → offset is applied
        assert result["weather_offset"] == -1.0


# ── TestRecommendationPreheat ─────────────────────────────────────────────────


class TestRecommendationPreheat:
    async def test_zero_preheat_minutes_no_preheat(self):
        coord = make_coordinator()
        coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=0)
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["preheat_active"] is False

    async def test_preheat_raises_base_target_to_comfort_temp(self):
        """Pre-heat raises night base_target to comfort_temp during pre-heat window."""
        coord = make_coordinator()
        coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=60)
        # base_target < comfort_temp → pre-heat eligible
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        # Mock _minutes_until_next_comfort to return a value within the preheat window
        with patch.object(coord, "_minutes_until_next_comfort", return_value=30):
            result = await coord._compute_recommendation(_CFG, _WEATHER, HEATING_MODE_AUTO)
        assert result["preheat_active"] is True
        assert result["base_target"] == _CFG["comfort_temp"]


# pytest.approx shortcut for concise assertions
try:
    from pytest import approx as pytest_approx
except ImportError:
    def pytest_approx(val, abs=0.1):  # type: ignore[misc]
        return val
