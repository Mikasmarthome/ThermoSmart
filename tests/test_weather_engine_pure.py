"""Tests for WeatherEngine pure computation methods.

compute_temperature_offset() and compute_forecast_suppression() accept
only a weather dict and return a float. They never read HA state, so a
MagicMock hass is sufficient — no HA fixtures required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.weather_engine import WeatherEngine
from custom_components.thermosmart.const import (
    WEATHER_COLD_THRESHOLD,
    WEATHER_MILD_THRESHOLD,
    WEATHER_WARM_THRESHOLD,
    WEATHER_BOOST_OFFSET,
    WEATHER_REDUCE_OFFSET,
    WIND_THRESHOLD_MS,
    WIND_CHILL_BOOST,
    SOLAR_OFFSET_THRESHOLD_W,
    SOLAR_OFFSET_MIN_OUTDOOR_C,
)


@pytest.fixture
def engine() -> WeatherEngine:
    """WeatherEngine instance with a mock hass — sufficient for pure method tests."""
    return WeatherEngine(hass=MagicMock(), weather_entity="weather.home")


# ── compute_temperature_offset ───────────────────────────────────────────────

class TestComputeTemperatureOffset:
    def test_no_temperature_in_data_returns_zero(self, engine):
        """Empty weather dict → no outdoor temp → offset 0.0."""
        assert engine.compute_temperature_offset({}) == 0.0

    def test_none_temperature_returns_zero(self, engine):
        assert engine.compute_temperature_offset({"temperature": None}) == 0.0

    def test_very_cold_returns_full_boost(self, engine):
        """outdoor < 0°C (WEATHER_COLD_THRESHOLD) → WEATHER_BOOST_OFFSET."""
        offset = engine.compute_temperature_offset({"temperature": -10.0})
        assert offset == pytest.approx(WEATHER_BOOST_OFFSET, abs=0.01)

    def test_cold_boundary_just_below_zero(self, engine):
        """outdoor = -0.1°C → still counts as very cold → full boost."""
        offset = engine.compute_temperature_offset({"temperature": -0.1})
        assert offset == pytest.approx(WEATHER_BOOST_OFFSET, abs=0.01)

    def test_mild_weather_returns_partial_boost(self, engine):
        """0°C ≤ outdoor < 10°C → reduced boost (1/3 of full boost)."""
        offset = engine.compute_temperature_offset({"temperature": 5.0})
        assert offset == pytest.approx(WEATHER_BOOST_OFFSET / 3.0, abs=0.3)
        # More precisely: rounded to nearest 0.5
        assert offset == pytest.approx(0.5, abs=0.01)

    def test_mild_boundary_at_zero(self, engine):
        """outdoor = 0.0°C → boundary: cold threshold → full boost."""
        offset = engine.compute_temperature_offset({"temperature": 0.0})
        # 0.0 is NOT < 0.0, so falls into mild range
        assert offset == pytest.approx(WEATHER_BOOST_OFFSET / 3.0, abs=0.3)

    def test_moderate_weather_no_offset(self, engine):
        """10°C ≤ outdoor < 18°C → offset = 0.0."""
        offset = engine.compute_temperature_offset({"temperature": 15.0})
        assert offset == 0.0

    def test_warm_boundary_at_10(self, engine):
        """outdoor = 10.0°C → WEATHER_MILD_THRESHOLD boundary → no boost."""
        offset = engine.compute_temperature_offset({"temperature": 10.0})
        assert offset == 0.0

    def test_hot_weather_returns_negative_offset(self, engine):
        """outdoor ≥ 18°C (WEATHER_WARM_THRESHOLD) → WEATHER_REDUCE_OFFSET."""
        offset = engine.compute_temperature_offset({"temperature": 25.0})
        assert offset == pytest.approx(WEATHER_REDUCE_OFFSET, abs=0.01)

    def test_hot_boundary_at_warm_threshold(self, engine):
        """outdoor = 18.0°C → exactly at WEATHER_WARM_THRESHOLD → reduce offset."""
        offset = engine.compute_temperature_offset({"temperature": 18.0})
        assert offset == pytest.approx(WEATHER_REDUCE_OFFSET, abs=0.01)

    def test_wind_chill_bonus_when_cold_and_windy(self, engine):
        """Cold outdoor + high wind → extra WIND_CHILL_BOOST added to offset."""
        data_no_wind = {"temperature": -5.0}
        data_with_wind = {"temperature": -5.0, "wind_speed": WIND_THRESHOLD_MS + 1}
        offset_base = engine.compute_temperature_offset(data_no_wind)
        offset_windy = engine.compute_temperature_offset(data_with_wind)
        assert offset_windy == pytest.approx(offset_base + WIND_CHILL_BOOST, abs=0.3)

    def test_wind_chill_not_applied_in_moderate_weather(self, engine):
        """Wind chill only adds boost when outdoor < WEATHER_MILD_THRESHOLD."""
        data = {"temperature": 15.0, "wind_speed": 20.0}
        offset = engine.compute_temperature_offset(data)
        assert offset == 0.0  # moderate weather, no wind chill bonus

    def test_wind_below_threshold_no_bonus(self, engine):
        """Wind below threshold → no wind chill bonus."""
        data_low_wind = {"temperature": -5.0, "wind_speed": WIND_THRESHOLD_MS - 1}
        data_high_wind = {"temperature": -5.0, "wind_speed": WIND_THRESHOLD_MS + 1}
        offset_low = engine.compute_temperature_offset(data_low_wind)
        offset_high = engine.compute_temperature_offset(data_high_wind)
        assert offset_high > offset_low

    def test_high_solar_reduces_offset_when_conditions_met(self, engine):
        """Solar above threshold + outdoor above min → reduces offset.

        Uses outdoor=8°C (mild, offset=0.5) so the solar condition is met (>5°C).
        """
        data_no_solar = {"temperature": 8.0}
        data_with_solar = {"temperature": 8.0, "solar_radiation": SOLAR_OFFSET_THRESHOLD_W + 600}
        offset_base = engine.compute_temperature_offset(data_no_solar)
        offset_solar = engine.compute_temperature_offset(data_with_solar)
        assert offset_solar < offset_base

    def test_solar_below_threshold_no_effect(self, engine):
        """Solar below SOLAR_OFFSET_THRESHOLD_W → no solar reduction."""
        data_low = {"temperature": 8.0, "solar_radiation": SOLAR_OFFSET_THRESHOLD_W - 100}
        data_high = {"temperature": 8.0, "solar_radiation": SOLAR_OFFSET_THRESHOLD_W + 600}
        assert engine.compute_temperature_offset(data_low) >= engine.compute_temperature_offset(data_high)

    def test_solar_ignored_below_outdoor_minimum(self, engine):
        """Solar has no effect when outdoor < SOLAR_OFFSET_MIN_OUTDOOR_C."""
        # outdoor=-5°C is below the minimum (5°C) → solar has no effect
        data_no_solar = {"temperature": -5.0}
        data_with_solar = {"temperature": -5.0, "solar_radiation": 1000.0}
        assert engine.compute_temperature_offset(data_no_solar) == engine.compute_temperature_offset(data_with_solar)

    def test_result_is_multiple_of_half_degree(self, engine):
        """Offset is always rounded to the nearest 0.5°C step."""
        for temp in [-10.0, -0.1, 0.0, 5.0, 10.0, 15.0, 18.0, 25.0]:
            offset = engine.compute_temperature_offset({"temperature": temp})
            assert abs(offset * 2 - round(offset * 2)) < 1e-9, (
                f"Offset {offset} for outdoor={temp} is not a 0.5°C multiple"
            )

    def test_returns_float(self, engine):
        result = engine.compute_temperature_offset({"temperature": 10.0})
        assert isinstance(result, float)


# ── compute_forecast_suppression ─────────────────────────────────────────────

class TestComputeForecastSuppression:
    def test_no_forecast_data_returns_full_heating(self, engine):
        """Without forecast_high, always return 1.0 (heat fully)."""
        factor = engine.compute_forecast_suppression(
            {"temperature": 5.0}, target_temp=21.0, night_temp=18.0
        )
        assert factor == 1.0

    def test_forecast_above_target_suppresses_fully(self, engine):
        """Forecast high ≥ target → house warms itself → factor 0.0."""
        factor = engine.compute_forecast_suppression(
            {"temperature": 5.0, "forecast_high": 22.0},
            target_temp=21.0, night_temp=18.0,
        )
        assert factor == 0.0

    def test_forecast_exactly_at_target_suppresses_fully(self, engine):
        """Forecast high == target → also returns 0.0."""
        factor = engine.compute_forecast_suppression(
            {"temperature": 5.0, "forecast_high": 21.0},
            target_temp=21.0, night_temp=18.0,
        )
        assert factor == 0.0

    def test_cold_forecast_returns_full_heating(self, engine):
        """Forecast well below night temp → heat fully → factor 1.0."""
        factor = engine.compute_forecast_suppression(
            {"temperature": 0.0, "forecast_high": 5.0},
            target_temp=21.0, night_temp=18.0,
        )
        assert factor == 1.0

    def test_linear_blend_between_night_and_target(self, engine):
        """Forecast between night_temp and target → linear interpolation.

        target=21, night=18, forecast_high=20:
          remaining = 21 - 20 = 1
          span      = 21 - 18 = 3
          factor    = 1/3 ≈ 0.333
        """
        factor = engine.compute_forecast_suppression(
            {"temperature": 5.0, "forecast_high": 20.0},
            target_temp=21.0, night_temp=18.0,
        )
        assert factor == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_linear_blend_midpoint(self, engine):
        """Forecast at midpoint between night and target → factor ≈ 0.5."""
        # target=21, night=17, midpoint=19
        factor = engine.compute_forecast_suppression(
            {"temperature": 5.0, "forecast_high": 19.0},
            target_temp=21.0, night_temp=17.0,
        )
        assert factor == pytest.approx(0.5, abs=0.01)

    def test_factor_always_between_0_and_1(self, engine):
        """Factor must always be in [0.0, 1.0] regardless of input."""
        test_cases = [
            ({"forecast_high": 30.0}, 21.0, 18.0),
            ({"forecast_high": -10.0}, 21.0, 18.0),
            ({}, 21.0, 18.0),
            ({"forecast_high": 21.0, "solar_radiation": 1000.0}, 21.0, 18.0),
            ({"temperature": 10.0, "forecast_high": 15.0}, 21.0, 18.0),
        ]
        for data, target, night in test_cases:
            factor = engine.compute_forecast_suppression(data, target, night)
            assert 0.0 <= factor <= 1.0, (
                f"Factor {factor} out of [0,1] for data={data}"
            )

    def test_solar_bonus_further_reduces_factor(self, engine):
        """High solar radiation reduces heating demand beyond forecast suppression.

        outdoor=8°C > SOLAR_GAIN_MIN_OUTDOOR_C=3°C and solar=600 > 300 → bonus applies.
        """
        data_no_solar = {"temperature": 8.0, "forecast_high": 19.0}
        data_with_solar = {"temperature": 8.0, "forecast_high": 19.0, "solar_radiation": 600.0}
        f_base = engine.compute_forecast_suppression(data_no_solar, target_temp=21.0, night_temp=18.0)
        f_solar = engine.compute_forecast_suppression(data_with_solar, target_temp=21.0, night_temp=18.0)
        assert f_solar <= f_base

    def test_solar_bonus_ignored_when_factor_already_zero(self, engine):
        """When factor=0 (fully suppressed), solar bonus cannot go below 0."""
        data = {"temperature": 10.0, "forecast_high": 25.0, "solar_radiation": 1000.0}
        factor = engine.compute_forecast_suppression(data, target_temp=21.0, night_temp=18.0)
        assert factor == 0.0

    def test_solar_ignored_in_cold_outdoor(self, engine):
        """Solar reduction requires outdoor > SOLAR_GAIN_MIN_OUTDOOR_C.

        At very cold outdoor (below 3°C), solar has no effect.
        """
        data_cold_no_solar = {"temperature": 1.0, "forecast_high": 19.0}
        data_cold_solar = {"temperature": 1.0, "forecast_high": 19.0, "solar_radiation": 1000.0}
        f1 = engine.compute_forecast_suppression(data_cold_no_solar, target_temp=21.0, night_temp=18.0)
        f2 = engine.compute_forecast_suppression(data_cold_solar, target_temp=21.0, night_temp=18.0)
        assert f1 == f2  # solar does nothing at outdoor=1°C

    def test_returns_float(self, engine):
        result = engine.compute_forecast_suppression(
            {"temperature": 10.0}, target_temp=21.0, night_temp=18.0
        )
        assert isinstance(result, float)

    def test_result_rounded_to_3_decimals(self, engine):
        """Result is rounded to 3 decimal places."""
        result = engine.compute_forecast_suppression(
            {"temperature": 5.0, "forecast_high": 20.0},
            target_temp=21.0, night_temp=18.0,
        )
        assert result == round(result, 3)
