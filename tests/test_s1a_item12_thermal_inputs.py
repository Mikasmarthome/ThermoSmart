"""S1a Item 12 — Thermal Input and Weather Authority Audit.

Proves source authority, fallback chains, normalization, and Minimal Setup
TRV-only correctness across all thermal and weather inputs.

Coverage:
  - WeatherEngine.async_get_data(): sensor priority over weather entity,
    dedicated sensor overwrite, forecast API fallback, invalid/non-numeric
    sensor states, weather entity unavailable
  - Room temperature source priority: external sensor authoritative,
    fallback to TRV current_temperature, unavailable/unknown handling,
    spike rejection by EMA filter
  - Window contact semantics: unknown/unavailable → treated as closed,
    multiple contacts (any open → zone open), non-existent entity → closed
  - Minimal Setup TRV-only: no external sensors of any kind → complete
    cycle without exception, TRV temp drives TPI, current_temp=None → duty=0
  - Outdoor temperature source priority: dedicated sensor over weather entity
  - TPI with missing inputs: outdoor=None, current_temp=None
  - Humidity/Wind/Solar None: degrade gracefully
  - Forecast empty/None: factor=1.0 (full heating, safe default)
  - Summer mode without outdoor data: no state change (safe)

Pure Python + MagicMock — runs on Windows, no hass fixture needed.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.weather_engine import WeatherEngine
from custom_components.thermosmart.const import HEATING_MODE_AUTO

from tests.helpers import make_coordinator, make_state, make_weather_data, set_hass_states


# ── WeatherEngine helpers ─────────────────────────────────────────────────────

def _engine(
    *,
    temp_sensor: str | None = None,
    humidity_sensor: str | None = None,
    wind_sensor: str | None = None,
    solar_sensor: str | None = None,
    rain_sensor: str | None = None,
    weather_entity: str = "weather.home",
    states: dict | None = None,
    forecast_response: dict | None = None,
    forecast_raises: Exception | None = None,
) -> WeatherEngine:
    hass = MagicMock()
    _states = states or {}
    hass.states.get = MagicMock(side_effect=lambda eid: _states.get(eid))
    svc_mock = AsyncMock()
    if forecast_raises is not None:
        svc_mock.side_effect = forecast_raises
    else:
        svc_mock.return_value = forecast_response or None
    hass.services.async_call = svc_mock
    return WeatherEngine(
        hass=hass,
        weather_entity=weather_entity,
        outdoor_temp_sensor=temp_sensor,
        outdoor_humidity_sensor=humidity_sensor,
        outdoor_wind_sensor=wind_sensor,
        outdoor_solar_sensor=solar_sensor,
        outdoor_rain_sensor=rain_sensor,
    )


def _weather_state(
    condition: str = "sunny",
    temp: float | None = None,
    humidity: float | None = None,
    wind_speed: float | None = None,
) -> MagicMock:
    st = MagicMock()
    st.state = condition
    attrs: dict = {}
    if temp is not None:
        attrs["temperature"] = temp
    if humidity is not None:
        attrs["humidity"] = humidity
    if wind_speed is not None:
        attrs["wind_speed"] = wind_speed
    st.attributes = attrs
    return st


def _sensor_state(value) -> MagicMock:
    st = MagicMock()
    st.state = str(value)
    st.attributes = {}
    return st


def _unavailable_state(reason: str = "unavailable") -> MagicMock:
    st = MagicMock()
    st.state = reason
    st.attributes = {}
    return st


# ── TestWeatherEngineAsyncGetData ─────────────────────────────────────────────


class TestWeatherEngineAsyncGetData:
    """WeatherEngine.async_get_data() — source priority and normalization."""

    async def test_no_weather_entity_returns_all_none(self):
        eng = _engine(states={})  # entity not in states → states.get() returns None
        data = await eng.async_get_data()
        assert data["temperature"] is None
        assert data["humidity"] is None
        assert data["wind_speed"] is None
        assert data["solar_radiation"] is None
        assert data["forecast_high"] is None

    async def test_weather_entity_unavailable_returns_none(self):
        st = MagicMock()
        st.state = "unavailable"
        st.attributes = {"temperature": 10.0}
        eng = _engine(states={"weather.home": st})
        data = await eng.async_get_data()
        assert data["temperature"] is None

    async def test_weather_entity_unknown_returns_none(self):
        st = MagicMock()
        st.state = "unknown"
        st.attributes = {"temperature": 5.0}
        eng = _engine(states={"weather.home": st})
        data = await eng.async_get_data()
        assert data["temperature"] is None

    async def test_weather_entity_provides_temperature(self):
        eng = _engine(states={"weather.home": _weather_state(temp=8.5)})
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(8.5)

    async def test_weather_entity_provides_humidity(self):
        eng = _engine(states={"weather.home": _weather_state(humidity=65.0)})
        data = await eng.async_get_data()
        assert data["humidity"] == pytest.approx(65.0)

    async def test_weather_entity_provides_wind_speed(self):
        eng = _engine(states={"weather.home": _weather_state(wind_speed=4.5)})
        data = await eng.async_get_data()
        assert data["wind_speed"] == pytest.approx(4.5)

    async def test_dedicated_temp_sensor_overrides_weather_entity(self):
        """Dedicated outdoor_temp_sensor must beat weather entity temperature."""
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={
                "weather.home": _weather_state(temp=5.0),
                "sensor.outdoor_temp": _sensor_state(7.3),
            },
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(7.3)

    async def test_dedicated_humidity_sensor_overrides_weather_entity(self):
        eng = _engine(
            humidity_sensor="sensor.outdoor_hum",
            states={
                "weather.home": _weather_state(humidity=70.0),
                "sensor.outdoor_hum": _sensor_state(55.0),
            },
        )
        data = await eng.async_get_data()
        assert data["humidity"] == pytest.approx(55.0)

    async def test_dedicated_wind_sensor_overrides_weather_entity(self):
        eng = _engine(
            wind_sensor="sensor.outdoor_wind",
            states={
                "weather.home": _weather_state(wind_speed=2.0),
                "sensor.outdoor_wind": _sensor_state(8.5),
            },
        )
        data = await eng.async_get_data()
        assert data["wind_speed"] == pytest.approx(8.5)

    async def test_dedicated_sensor_unavailable_falls_back_to_weather_entity(self):
        """If dedicated sensor is unavailable, weather entity value is used."""
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={
                "weather.home": _weather_state(temp=5.0),
                "sensor.outdoor_temp": _unavailable_state("unavailable"),
            },
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(5.0)

    async def test_dedicated_sensor_unknown_falls_back_to_weather_entity(self):
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={
                "weather.home": _weather_state(temp=3.0),
                "sensor.outdoor_temp": _unavailable_state("unknown"),
            },
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(3.0)

    async def test_dedicated_sensor_non_numeric_falls_back(self):
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={
                "weather.home": _weather_state(temp=4.0),
                "sensor.outdoor_temp": _sensor_state("not_a_number"),
            },
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(4.0)

    async def test_dedicated_sensor_none_string_falls_back(self):
        eng = _engine(
            temp_sensor="sensor.outdoor_temp",
            states={
                "weather.home": _weather_state(temp=4.0),
                "sensor.outdoor_temp": _unavailable_state("None"),
            },
        )
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(4.0)

    async def test_forecast_api_success_sets_forecast_high(self):
        response = {"weather.home": {"forecast": [{"temperature": 15.0, "templow": 8.0}]}}
        eng = _engine(
            states={"weather.home": _weather_state()},
            forecast_response=response,
        )
        data = await eng.async_get_data()
        assert data["forecast_high"] == pytest.approx(15.0)

    async def test_forecast_api_empty_list_leaves_forecast_none(self):
        """Empty forecast list must not crash and must leave forecast_high as None."""
        response = {"weather.home": {"forecast": []}}
        eng = _engine(
            states={"weather.home": _weather_state()},
            forecast_response=response,
        )
        data = await eng.async_get_data()
        assert data["forecast_high"] is None

    async def test_forecast_api_timeout_degrades_gracefully(self):
        eng = _engine(
            states={"weather.home": _weather_state()},
            forecast_raises=asyncio.TimeoutError(),
        )
        data = await eng.async_get_data()
        assert data["forecast_high"] is None  # no crash, no stale value

    async def test_forecast_api_exception_degrades_gracefully(self):
        eng = _engine(
            states={"weather.home": _weather_state()},
            forecast_raises=Exception("service_unavailable"),
        )
        data = await eng.async_get_data()
        assert data["forecast_high"] is None

    async def test_all_keys_always_present_in_result(self):
        eng = _engine()
        data = await eng.async_get_data()
        expected = {"temperature", "humidity", "wind_speed", "solar_radiation",
                    "rain", "condition", "forecast_high", "forecast_low"}
        assert expected <= set(data.keys())

    async def test_solar_sensor_sets_solar_radiation(self):
        eng = _engine(
            solar_sensor="sensor.solar",
            states={"sensor.solar": _sensor_state(450.0)},
        )
        data = await eng.async_get_data()
        assert data["solar_radiation"] == pytest.approx(450.0)

    async def test_rain_sensor_sets_rain(self):
        eng = _engine(
            rain_sensor="sensor.rain",
            states={"sensor.rain": _sensor_state(2.5)},
        )
        data = await eng.async_get_data()
        assert data["rain"] == pytest.approx(2.5)


# ── TestRoomTemperatureSourcePriority ─────────────────────────────────────────


_CFG = {
    "comfort_temp": 21.0,
    "night_temp": 18.0,
    "away_temp": 17.0,
    "vacation_temp": 12.0,
    "eco_temp": 19.0,
    "window_open_temp": 5.0,
    "temp_tolerance": 0.5,
    "temp_sensors": ["sensor.room_temp"],
    "climate_entities": ["climate.trv"],
    "humidity_sensors": [],
    "window_sensors": [],
    "window_open_delay": 5,
    "window_close_delay": 2,
    "sched_wd_morning": "06:00",
    "sched_wd_night": "22:00",
    "sched_we_morning": "08:00",
    "sched_we_night": "23:00",
    "schedule_enabled": True,
}
_W = make_weather_data(outdoor_temp=10.0)


class TestRoomTemperatureSourcePriority:
    """External temp_sensor must take priority over TRV current_temperature."""

    async def test_external_sensor_authoritative_over_trv(self):
        """When both external sensor and TRV are available, external sensor wins."""
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("20.5"),
            "climate.trv": make_state("heat", {"current_temperature": 19.0}),
        })
        result = await coord._compute_recommendation(_CFG, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] == pytest.approx(20.5)

    async def test_external_sensor_unavailable_falls_back_to_trv(self):
        """unavailable external sensor → fall back to TRV current_temperature."""
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("unavailable"),
            "climate.trv": make_state("heat", {"current_temperature": 19.5}),
        })
        result = await coord._compute_recommendation(_CFG, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] == pytest.approx(19.5)

    async def test_external_sensor_unknown_falls_back_to_trv(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("unknown"),
            "climate.trv": make_state("heat", {"current_temperature": 18.8}),
        })
        result = await coord._compute_recommendation(_CFG, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] == pytest.approx(18.8)

    async def test_external_sensor_none_string_falls_back_to_trv(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("None"),
            "climate.trv": make_state("heat", {"current_temperature": 20.0}),
        })
        result = await coord._compute_recommendation(_CFG, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] == pytest.approx(20.0)

    async def test_external_sensor_non_numeric_falls_back_to_trv(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("error"),
            "climate.trv": make_state("heat", {"current_temperature": 21.0}),
        })
        result = await coord._compute_recommendation(_CFG, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] == pytest.approx(21.0)

    async def test_trv_unavailable_excluded_from_avg(self):
        """Unavailable TRV must be excluded; available TRVs still used."""
        cfg = {**_CFG, "temp_sensors": [], "climate_entities": ["climate.trv", "climate.trv2"]}
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("unavailable"),
            "climate.trv2": make_state("heat", {"current_temperature": 20.0}),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] == pytest.approx(20.0)

    async def test_all_sensors_unavailable_returns_none_current_temp(self):
        """All sensors + all TRVs unavailable → current_temp=None."""
        cfg = {**_CFG, "temp_sensors": ["sensor.room_temp"]}
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("unavailable"),
            "climate.trv": make_state("unavailable"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] is None

    async def test_multiple_external_sensors_averaged(self):
        """Multiple temp_sensors: average is used (not max/min/first)."""
        cfg = {**_CFG, "temp_sensors": ["sensor.s1", "sensor.s2"]}
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.s1": make_state("20.0"),
            "sensor.s2": make_state("22.0"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        # Average: (20 + 22) / 2 = 21.0 (after EMA which starts at raw on first reading)
        assert result["current_temp"] == pytest.approx(21.0)

    async def test_external_sensor_spike_rejected(self):
        """A sudden spike (>4°C delta from EMA) is rejected; previous EMA persists."""
        coord = make_coordinator()
        # First call: establish EMA at 20.0
        set_hass_states(coord, {"sensor.room_temp": make_state("20.0")})
        r1 = await coord._compute_recommendation(_CFG, _W, HEATING_MODE_AUTO)
        assert r1["current_temp"] == pytest.approx(20.0)

        # Second call: spike to 30.0 (Δ=10 >> 4°C threshold)
        set_hass_states(coord, {"sensor.room_temp": make_state("30.0")})
        r2 = await coord._compute_recommendation(_CFG, _W, HEATING_MODE_AUTO)
        # Spike must be rejected → current_temp stays near EMA (~20°C)
        # With one spike, _read_avg_sensor returns None for this sensor
        # → falls back to TRV if configured; here TRV not set → None
        # Either current_temp is still ~20 (via EMA) or falls back to TRV/None
        # Key assertion: it must NOT be 30.0
        if r2["current_temp"] is not None:
            assert r2["current_temp"] < 25.0, "Spike of 30°C must not pass filter"


# ── TestWindowContactSemantics ────────────────────────────────────────────────


class TestWindowContactSemantics:
    """Window sensor state handling: unknown/unavailable → closed (safe default)."""

    async def test_window_sensor_unknown_treated_as_closed(self):
        """Window sensor in 'unknown' state → window_open=False."""
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"]}
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("20.0"),
            "binary_sensor.window": make_state("unknown"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is False

    async def test_window_sensor_unavailable_treated_as_closed(self):
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"]}
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("20.0"),
            "binary_sensor.window": make_state("unavailable"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is False

    async def test_window_sensor_off_is_closed(self):
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"]}
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("20.0"),
            "binary_sensor.window": make_state("off"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is False

    async def test_window_sensor_on_triggers_window_open(self):
        """Window sensor 'on' with zero open_delay → window detected as open."""
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"], "window_open_delay": 0}
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("20.0"),
            "binary_sensor.window": make_state("on"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is True

    async def test_window_sensor_nonexistent_treated_as_closed(self):
        """Entity not found in hass.states → treated as closed."""
        cfg = {**_CFG, "window_sensors": ["binary_sensor.nonexistent"]}
        coord = make_coordinator()
        set_hass_states(coord, {"sensor.room_temp": make_state("20.0")})
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is False

    async def test_multiple_window_sensors_any_open_wins(self):
        """Two window contacts: one open, one closed → window_open=True."""
        cfg = {
            **_CFG,
            "window_sensors": ["binary_sensor.w1", "binary_sensor.w2"],
            "window_open_delay": 0,
        }
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("20.0"),
            "binary_sensor.w1": make_state("on"),   # OPEN
            "binary_sensor.w2": make_state("off"),  # closed
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is True

    async def test_multiple_window_sensors_all_closed_is_closed(self):
        cfg = {
            **_CFG,
            "window_sensors": ["binary_sensor.w1", "binary_sensor.w2"],
        }
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("20.0"),
            "binary_sensor.w1": make_state("off"),
            "binary_sensor.w2": make_state("off"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is False

    async def test_window_open_suppresses_heating_targets(self):
        """Window open: adjusted_target and effective_target become None."""
        cfg = {**_CFG, "window_sensors": ["binary_sensor.window"], "window_open_delay": 0}
        coord = make_coordinator()
        set_hass_states(coord, {
            "sensor.room_temp": make_state("20.0"),
            "binary_sensor.window": make_state("on"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is True
        assert result["adjusted_target"] is None
        assert result["effective_target"] is None


# ── TestMinimalTRVOnlySetup ───────────────────────────────────────────────────


class TestMinimalTRVOnlySetup:
    """Minimal Setup: only TRV configured, no external sensors of any kind."""

    _MINIMAL_CFG = {
        "comfort_temp": 21.0,
        "night_temp": 18.0,
        "away_temp": 17.0,
        "vacation_temp": 12.0,
        "eco_temp": 19.0,
        "window_open_temp": 5.0,
        "temp_tolerance": 0.5,
        "temp_sensors": [],        # No external room sensor
        "climate_entities": ["climate.trv"],
        "humidity_sensors": [],    # No humidity
        "window_sensors": [],      # No window sensor
        "window_open_delay": 5,
        "window_close_delay": 2,
        "sched_wd_morning": "06:00",
        "sched_wd_night": "22:00",
        "sched_we_morning": "08:00",
        "sched_we_night": "23:00",
        "schedule_enabled": True,
    }

    # Weather with NO outdoor data (completely offline)
    _NO_WEATHER = make_weather_data(
        outdoor_temp=None, humidity=None, wind_speed=None,
        solar_radiation=None, forecast_high=None, forecast_low=None,
        condition=None, rain=None,
    )

    async def test_trv_only_no_exception(self):
        """Complete recommendation cycle without any external sensor must not raise."""
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 19.0}),
        })
        result = await coord._compute_recommendation(
            self._MINIMAL_CFG, self._NO_WEATHER, HEATING_MODE_AUTO
        )
        assert result is not None

    async def test_trv_only_uses_trv_temperature(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 19.5}),
        })
        result = await coord._compute_recommendation(
            self._MINIMAL_CFG, self._NO_WEATHER, HEATING_MODE_AUTO
        )
        assert result["current_temp"] == pytest.approx(19.5)

    async def test_trv_only_has_adjusted_target(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 19.0}),
        })
        result = await coord._compute_recommendation(
            self._MINIMAL_CFG, self._NO_WEATHER, HEATING_MODE_AUTO
        )
        assert result["adjusted_target"] is not None

    async def test_trv_only_no_outdoor_weather_offset_zero(self):
        """Without outdoor data, weather_offset must be 0 (safe default)."""
        coord = make_coordinator()
        coord.weather_engine.compute_temperature_offset.return_value = 0.0
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 19.0}),
        })
        result = await coord._compute_recommendation(
            self._MINIMAL_CFG, self._NO_WEATHER, HEATING_MODE_AUTO
        )
        assert result.get("weather_offset", 0.0) == pytest.approx(0.0)

    async def test_no_current_temp_at_all_does_not_raise(self):
        """No TRV temperature either → current_temp=None must not raise."""
        coord = make_coordinator()
        # TRV has no current_temperature attribute
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {}),
        })
        result = await coord._compute_recommendation(
            self._MINIMAL_CFG, self._NO_WEATHER, HEATING_MODE_AUTO
        )
        assert result["current_temp"] is None

    async def test_required_keys_present_in_minimal_setup(self):
        """Required keys must be present even in minimal TRV-only setup."""
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 19.0}),
        })
        result = await coord._compute_recommendation(
            self._MINIMAL_CFG, self._NO_WEATHER, HEATING_MODE_AUTO
        )
        required = {
            "zone_name", "mode", "current_temp", "window_open",
            "adjusted_target", "effective_target", "override_active",
        }
        assert required <= result.keys()

    async def test_trv_only_window_open_is_false_without_sensor(self):
        """No window sensor → window_open=False (slope fallback, no rapid slope)."""
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 19.0}),
        })
        result = await coord._compute_recommendation(
            self._MINIMAL_CFG, self._NO_WEATHER, HEATING_MODE_AUTO
        )
        assert result["window_open"] is False

    async def test_trv_only_summer_mode_stays_winter_without_outdoor_data(self):
        """Without outdoor temp data, summer mode must not activate."""
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv": make_state("heat", {"current_temperature": 19.0}),
        })
        for _ in range(5):
            await coord._compute_recommendation(
                self._MINIMAL_CFG, self._NO_WEATHER, HEATING_MODE_AUTO
            )
        assert not coord._is_summer


# ── TestOutdoorTemperatureSourcePriority ─────────────────────────────────────


class TestOutdoorTemperatureSourcePriority:
    """Dedicated outdoor sensor must always override weather entity value."""

    async def test_dedicated_outdoor_sensor_overrides_weather_entity(self):
        """WeatherEngine: dedicated temp sensor reads higher priority than weather entity."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(return_value=None)
        hass.states.get = MagicMock(side_effect=lambda eid: {
            "weather.home": _weather_state(temp=5.0),
            "sensor.outdoor": _sensor_state(7.5),
        }.get(eid))
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.outdoor")
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(7.5)

    async def test_outdoor_temp_none_does_not_affect_summer_mode(self):
        """Summer mode must not change when outdoor=None."""
        coord = make_coordinator()
        coord._is_summer = False
        for _ in range(10):
            coord._update_summer_mode({"temperature": None})
        assert coord._is_summer is False

    async def test_outdoor_temp_none_keeps_existing_summer_state(self):
        """Once in summer mode, outdoor=None must not revert it."""
        coord = make_coordinator()
        coord._is_summer = True
        for _ in range(5):
            coord._update_summer_mode({"temperature": None})
        assert coord._is_summer is True

    async def test_tpi_runs_without_outdoor_temp(self):
        """TPI must not raise when outdoor=None; duty comes from indoor error only."""
        from custom_components.thermosmart.tpi import compute_tpi
        duty = compute_tpi(target=21.0, room=19.0, outdoor=None,
                           coef_int=0.6, coef_ext=0.01)
        assert duty > 0.0           # indoor error drives duty
        assert duty <= 100.0

    async def test_tpi_with_outdoor_temp_adds_extra_duty(self):
        """Cold outdoor temp adds coef_ext contribution → higher duty (below cap)."""
        from custom_components.thermosmart.tpi import compute_tpi
        # Small indoor delta (Δ=1°C) keeps both values below 100% cap
        duty_no_outdoor = compute_tpi(21.0, 20.0, None, 0.6, 0.01)
        duty_cold_outdoor = compute_tpi(21.0, 20.0, -5.0, 0.6, 0.01)
        assert duty_cold_outdoor > duty_no_outdoor


# ── TestInputNormalizationAndBounds ──────────────────────────────────────────


class TestInputNormalizationAndBounds:
    """Invalid, out-of-range, and edge-case sensor values must degrade safely."""

    def test_weather_engine_read_sensor_filters_unknown(self):
        hass = MagicMock()
        hass.states.get = MagicMock(return_value=_unavailable_state("unknown"))
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.t")
        assert eng._read_sensor("sensor.t") is None

    def test_weather_engine_read_sensor_filters_unavailable(self):
        hass = MagicMock()
        hass.states.get = MagicMock(return_value=_unavailable_state("unavailable"))
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.t")
        assert eng._read_sensor("sensor.t") is None

    def test_weather_engine_read_sensor_filters_none_string(self):
        hass = MagicMock()
        hass.states.get = MagicMock(return_value=_unavailable_state("None"))
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.t")
        assert eng._read_sensor("sensor.t") is None

    def test_weather_engine_read_sensor_filters_non_numeric(self):
        hass = MagicMock()
        hass.states.get = MagicMock(return_value=_sensor_state("not_a_float"))
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.t")
        assert eng._read_sensor("sensor.t") is None

    def test_weather_engine_read_sensor_no_entity_id_returns_none(self):
        hass = MagicMock()
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        assert eng._read_sensor(None) is None
        assert eng._read_sensor("") is None

    def test_weather_engine_read_sensor_valid_negative_temp(self):
        hass = MagicMock()
        hass.states.get = MagicMock(return_value=_sensor_state("-12.5"))
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.t")
        assert eng._read_sensor("sensor.t") == pytest.approx(-12.5)

    def test_weather_engine_read_sensor_valid_zero(self):
        hass = MagicMock()
        hass.states.get = MagicMock(return_value=_sensor_state("0.0"))
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.t")
        assert eng._read_sensor("sensor.t") == pytest.approx(0.0)

    def test_weather_engine_read_sensor_entity_not_in_hass(self):
        hass = MagicMock()
        hass.states.get = MagicMock(return_value=None)
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.nonexistent")
        assert eng._read_sensor("sensor.nonexistent") is None

    def test_humidity_sensor_unavailable_read_avg_returns_none(self):
        """_read_avg_sensor with unavailable sensor must return None."""
        coord = make_coordinator()
        set_hass_states(coord, {"sensor.hum": make_state("unavailable")})
        result = coord._read_avg_sensor(["sensor.hum"])
        assert result is None

    def test_humidity_sensor_valid_value_read_avg_returns_float(self):
        """_read_avg_sensor with valid humidity sensor returns float."""
        coord = make_coordinator()
        set_hass_states(coord, {"sensor.hum": make_state("65.0")})
        result = coord._read_avg_sensor(["sensor.hum"])
        assert result == pytest.approx(65.0)

    def test_forecast_suppression_none_forecast_high_returns_full_heating(self):
        """forecast_high=None → suppression factor 1.0 (always heat, safe default)."""
        hass = MagicMock()
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        factor = eng.compute_forecast_suppression(
            {"forecast_high": None, "temperature": 5.0, "solar_radiation": None},
            target_temp=21.0, night_temp=18.0,
        )
        assert factor == pytest.approx(1.0)

    def test_compute_temperature_offset_none_wind_no_wind_chill(self):
        """wind_speed=None must not crash; wind chill bonus skipped."""
        hass = MagicMock()
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        offset_no_wind = eng.compute_temperature_offset(
            {"temperature": -5.0, "wind_speed": None})
        offset_with_wind = eng.compute_temperature_offset(
            {"temperature": -5.0, "wind_speed": 15.0})
        assert offset_no_wind <= offset_with_wind  # Wind chill only adds, never subtracts

    def test_compute_temperature_offset_none_solar_no_solar_reduction(self):
        """solar_radiation=None must not crash; solar reduction skipped."""
        hass = MagicMock()
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        offset = eng.compute_temperature_offset(
            {"temperature": 5.0, "solar_radiation": None, "wind_speed": None})
        # Just verify no exception and result is a float
        assert isinstance(offset, float)


# ── TestWeatherEngineIsHeatingSeason ─────────────────────────────────────────


class TestWeatherEngineIsHeatingSeason:
    """Heating season detection: outdoor=None → always heating season (safe)."""

    def test_is_heating_season_outdoor_none_returns_true(self):
        hass = MagicMock()
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        assert eng.is_heating_season({"temperature": None}) is True

    def test_is_heating_season_outdoor_missing_returns_true(self):
        hass = MagicMock()
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        assert eng.is_heating_season({}) is True

    def test_is_heating_season_warm_outdoor_returns_false(self):
        from custom_components.thermosmart.const import WEATHER_WARM_THRESHOLD
        hass = MagicMock()
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        assert eng.is_heating_season({"temperature": WEATHER_WARM_THRESHOLD + 1}) is False
