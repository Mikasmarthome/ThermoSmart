"""WeatherEngine – fetches outdoor conditions and computes heating offsets."""
from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant

from .const import (
    WEATHER_COLD_THRESHOLD,
    WEATHER_MILD_THRESHOLD,
    WEATHER_WARM_THRESHOLD,
    WEATHER_BOOST_OFFSET,
    WEATHER_REDUCE_OFFSET,
    WIND_THRESHOLD_MS,
    WIND_CHILL_BOOST,
)

_LOGGER = logging.getLogger(__name__)


class WeatherEngine:
    """
    Reads a Home Assistant weather entity (or a standalone outdoor temperature sensor)
    and exposes helper methods used by the coordinator and learning engine.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        weather_entity: str,
        outdoor_temp_sensor: str | None = None,
    ) -> None:
        self._hass = hass
        self._weather_entity = weather_entity
        self._outdoor_temp_sensor = outdoor_temp_sensor

    async def async_get_data(self) -> dict:
        """Return a normalised weather snapshot."""
        data: dict = {
            "temperature": None,
            "condition": None,
            "wind_speed": None,
            "humidity": None,
            "forecast_high": None,
            "forecast_low": None,
        }

        # Try dedicated outdoor temperature sensor first (more accurate)
        if self._outdoor_temp_sensor:
            state = self._hass.states.get(self._outdoor_temp_sensor)
            if state and state.state not in ("unknown", "unavailable"):
                try:
                    data["temperature"] = float(state.state)
                except ValueError:
                    pass

        # Weather entity
        weather_state = self._hass.states.get(self._weather_entity)
        if weather_state and weather_state.state not in ("unknown", "unavailable"):
            attrs = weather_state.attributes
            data["condition"] = weather_state.state

            if data["temperature"] is None:
                raw = attrs.get("temperature")
                if raw is not None:
                    try:
                        data["temperature"] = float(raw)
                    except (TypeError, ValueError):
                        pass

            raw_wind = attrs.get("wind_speed")
            if raw_wind is not None:
                try:
                    data["wind_speed"] = float(raw_wind)
                except (TypeError, ValueError):
                    pass

            raw_hum = attrs.get("humidity")
            if raw_hum is not None:
                try:
                    data["humidity"] = float(raw_hum)
                except (TypeError, ValueError):
                    pass

            # Grab first forecast entry if available
            forecast = attrs.get("forecast", [])
            if forecast:
                first = forecast[0]
                try:
                    data["forecast_high"] = float(first.get("temperature", 0))
                except (TypeError, ValueError):
                    pass
                try:
                    data["forecast_low"] = float(first.get("templow", data["forecast_high"] or 0))
                except (TypeError, ValueError):
                    pass

        return data

    def compute_temperature_offset(self, weather_data: dict) -> float:
        """
        Return a °C offset to add to the base target temperature based on
        outdoor conditions (temperature + wind chill).

        Rules:
          outdoor < 0 °C              → +1.5 °C boost
          outdoor 0–10 °C             → +0.5 °C slight boost
          outdoor 10–18 °C            → no change
          outdoor >= 18 °C            → −1.0 °C reduction
          additionally windy + cold   → extra +0.5 °C
        """
        outdoor = weather_data.get("temperature")
        if outdoor is None:
            return 0.0

        if outdoor < WEATHER_COLD_THRESHOLD:
            offset = WEATHER_BOOST_OFFSET
        elif outdoor < WEATHER_MILD_THRESHOLD:
            offset = WEATHER_BOOST_OFFSET / 3.0
        elif outdoor < WEATHER_WARM_THRESHOLD:
            offset = 0.0
        else:
            offset = WEATHER_REDUCE_OFFSET

        # Wind-chill bonus
        wind = weather_data.get("wind_speed") or 0.0
        if outdoor < WEATHER_MILD_THRESHOLD and wind > WIND_THRESHOLD_MS:
            offset += WIND_CHILL_BOOST
            _LOGGER.debug(
                "Wind chill boost applied: wind=%.1f m/s, outdoor=%.1f °C", wind, outdoor
            )

        return round(offset, 2)

    def is_heating_season(self, weather_data: dict) -> bool:
        """Return True when heating is likely needed (outdoor temp below warm threshold)."""
        outdoor = weather_data.get("temperature")
        if outdoor is None:
            return True  # assume heating needed when unknown
        return outdoor < WEATHER_WARM_THRESHOLD

    def get_condition_category(self, weather_data: dict) -> str:
        """Classify condition into cold / mild / warm / hot for the learning engine."""
        outdoor = weather_data.get("temperature")
        if outdoor is None:
            return "unknown"
        if outdoor < WEATHER_COLD_THRESHOLD:
            return "cold"
        if outdoor < WEATHER_MILD_THRESHOLD:
            return "mild"
        if outdoor < WEATHER_WARM_THRESHOLD:
            return "warm"
        return "hot"
