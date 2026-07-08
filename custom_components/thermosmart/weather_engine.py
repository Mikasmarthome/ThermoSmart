"""WeatherEngine – Außenbedingungen aus Sensoren und/oder Wetter-Entity."""
from __future__ import annotations

import asyncio
import logging

from homeassistant.const import UnitOfSpeed
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.unit_conversion import SpeedConverter

from .temperature_units import to_internal_temperature_c
from .const import (
    WEATHER_COLD_THRESHOLD,
    WEATHER_MILD_THRESHOLD,
    WEATHER_WARM_THRESHOLD,
    WEATHER_BOOST_OFFSET,
    WEATHER_REDUCE_OFFSET,
    WIND_THRESHOLD_MS,
    WIND_CHILL_BOOST,
    SOLAR_GAIN_THRESHOLD_W,
    SOLAR_GAIN_MIN_OUTDOOR_C,
    SOLAR_GAIN_MAX_REDUCTION,
    SOLAR_OFFSET_THRESHOLD_W,
    SOLAR_OFFSET_MIN_OUTDOOR_C,
)

_LOGGER = logging.getLogger(__name__)

_VALID_SPEED_UNITS = {
    UnitOfSpeed.METERS_PER_SECOND,
    UnitOfSpeed.KILOMETERS_PER_HOUR,
    UnitOfSpeed.MILES_PER_HOUR,
    UnitOfSpeed.KNOTS,
    UnitOfSpeed.FEET_PER_SECOND,
}


def _to_wind_speed_ms(value, unit: str | None) -> float | None:
    """Normalize a wind-speed reading to m/s.

    Mirrors to_internal_temperature_c()'s unit-aware normalization. Wind
    sources report their own unit (a sensor's unit_of_measurement, or a
    weather entity's dedicated wind_speed_unit attribute) — km/h is a common
    convention for European weather stations, and the config UI explicitly
    advertises "m/s or km/h" as accepted, so WIND_THRESHOLD_MS must compare
    against an actually-normalized value, not a raw passthrough.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if unit not in _VALID_SPEED_UNITS:
        return numeric  # unrecognized/missing unit → assume already m/s
    try:
        return SpeedConverter.convert(numeric, unit, UnitOfSpeed.METERS_PER_SECOND)
    except HomeAssistantError:
        return None


class WeatherEngine:
    """Liest Außenbedingungen aus konfigurierten Sensoren (bevorzugt)
    oder aus einem HA-Wetter-Entity (Fallback).

    Priorität pro Wert:
      1. Dedizierter Sensor (eigene Wetterstation)
      2. Wetter-Entity Attribut
      3. None (unbekannt)
    """

    def __init__(
        self,
        hass: HomeAssistant,
        weather_entity: str,
        outdoor_temp_sensor: str | None = None,
        outdoor_humidity_sensor: str | None = None,
        outdoor_wind_sensor: str | None = None,
        outdoor_solar_sensor: str | None = None,
        outdoor_rain_sensor: str | None = None,
    ) -> None:
        self._hass = hass
        self._weather_entity = weather_entity
        self._temp_sensor = outdoor_temp_sensor
        self._humidity_sensor = outdoor_humidity_sensor
        self._wind_sensor = outdoor_wind_sensor
        self._solar_sensor = outdoor_solar_sensor
        self._rain_sensor = outdoor_rain_sensor

    def _read_sensor(
        self, entity_id: str | None, *, is_temperature: bool = False, is_wind: bool = False,
    ) -> float | None:
        """Einzelnen Sensor lesen. Gibt None zurück wenn unavailable.

        ``is_temperature=True`` normalizes the sensor's own unit_of_measurement
        to °C. ``is_wind=True`` normalizes it to m/s — humidity/solar/rain
        sensors are never unit-converted.
        """
        if not entity_id:
            return None
        state = self._hass.states.get(entity_id)
        if state and state.state not in ("unknown", "unavailable", "None"):
            if is_temperature:
                return to_internal_temperature_c(
                    self._hass, state.state, unit=state.attributes.get("unit_of_measurement"))
            if is_wind:
                return _to_wind_speed_ms(state.state, state.attributes.get("unit_of_measurement"))
            try:
                return float(state.state)
            except (ValueError, TypeError):
                pass
        return None

    async def async_get_data(self) -> dict:
        """Alle Außenbedingungen als normalisiertes Dict zurückgeben."""
        data: dict = {
            "temperature": None,
            "humidity": None,
            "wind_speed": None,
            "solar_radiation": None,
            "rain": None,
            "condition": None,
            "forecast_high": None,
            "forecast_low": None,
        }

        # ── Wetter-Entity (Basis) ─────────────────────────────────────
        weather_state = self._hass.states.get(self._weather_entity)
        weather_temp_unit: str | None = None
        if weather_state and weather_state.state not in ("unknown", "unavailable"):
            attrs = weather_state.attributes
            data["condition"] = weather_state.state
            # weather.* entities expose their own display unit as a dedicated
            # "temperature_unit" attribute (distinct from a sensor's
            # unit_of_measurement) — the entity's temperature/forecast
            # values are already expressed in this unit.
            weather_temp_unit = attrs.get("temperature_unit")

            raw_humidity = attrs.get("humidity")
            if raw_humidity is not None:
                try:
                    data["humidity"] = float(raw_humidity)
                except (TypeError, ValueError):
                    pass
            data["wind_speed"] = _to_wind_speed_ms(
                attrs.get("wind_speed"), attrs.get("wind_speed_unit"))
            data["temperature"] = to_internal_temperature_c(
                self._hass, attrs.get("temperature"), unit=weather_temp_unit)

            # Letzter Fallback: Legacy-Forecast aus Entity-Attributen
            # (für Integrationen die den neuen Service nicht unterstützen)
            legacy_forecast = attrs.get("forecast")
            if isinstance(legacy_forecast, list) and legacy_forecast:
                try:
                    _high = to_internal_temperature_c(
                        self._hass, legacy_forecast[0].get("temperature"), unit=weather_temp_unit)
                    data["forecast_high"] = _high if _high is not None else None
                    _low_raw = legacy_forecast[0].get("templow")
                    data["forecast_low"] = (
                        to_internal_temperature_c(self._hass, _low_raw, unit=weather_temp_unit)
                        if _low_raw is not None else data["forecast_high"])
                except (TypeError, ValueError, AttributeError):
                    pass
        elif self._weather_entity:
            _LOGGER.debug(
                "WeatherEngine: Wetter-Entity '%s' nicht verfügbar – nur Sensordaten",
                self._weather_entity,
            )

        # ── Forecast über neue HA-API (ab 2024.3) ────────────────────
        # Neue Methode hat Vorrang vor Legacy-Attributen.
        # Timeout 10s verhindert, dass ein hängender Service den Update-Loop blockiert.
        try:
            async with asyncio.timeout(10):
                result = await self._hass.services.async_call(
                    "weather",
                    "get_forecasts",
                    {"entity_id": self._weather_entity, "type": "daily"},
                    blocking=True,
                    return_response=True,
                )
            if result:
                forecast_list = result.get(self._weather_entity, {}).get("forecast", [])
                if forecast_list:
                    first = forecast_list[0]
                    # get_forecasts() returns values already converted to the
                    # entity's own display unit (same weather_temp_unit as above).
                    _high = to_internal_temperature_c(
                        self._hass, first.get("temperature"), unit=weather_temp_unit)
                    if _high is not None:
                        data["forecast_high"] = _high
                    _low_raw = first.get("templow")
                    if _low_raw is not None:
                        data["forecast_low"] = to_internal_temperature_c(
                            self._hass, _low_raw, unit=weather_temp_unit)
                    elif data["forecast_high"] is not None:
                        data["forecast_low"] = data["forecast_high"]
        except TimeoutError:
            _LOGGER.warning(
                "WeatherEngine: Forecast-Abruf Timeout (>10s) – nutze Legacy-Daten falls verfügbar"
            )
        except Exception:
            # Service nicht verfügbar oder Entity unterstützt keinen Forecast → kein Problem
            pass

        # ── Dedizierte Sensoren überschreiben Wetter-Entity-Werte ─────
        sensor_map = {
            "temperature":    self._temp_sensor,
            "humidity":       self._humidity_sensor,
            "wind_speed":     self._wind_sensor,
            "solar_radiation": self._solar_sensor,
            "rain":           self._rain_sensor,
        }
        for key, sensor_id in sensor_map.items():
            value = self._read_sensor(
                sensor_id, is_temperature=(key == "temperature"), is_wind=(key == "wind_speed"))
            if value is not None:
                data[key] = value
                if key == "temperature":
                    _LOGGER.debug(
                        "WeatherEngine: Außentemp %.1f°C von %s", value, sensor_id
                    )

        return data

    def compute_temperature_offset(self, weather_data: dict) -> float:
        """Temperatur-Offset basierend auf Außentemperatur + Wind berechnen."""
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

        # Windkälte-Bonus
        wind = weather_data.get("wind_speed") or 0.0
        if outdoor < WEATHER_MILD_THRESHOLD and wind > WIND_THRESHOLD_MS:
            offset += WIND_CHILL_BOOST
            _LOGGER.debug(
                "Windkälte-Boost: wind=%.1f m/s, outdoor=%.1f°C", wind, outdoor
            )

        # Sonneneinstrahlung reduziert Heizbedarf
        solar = weather_data.get("solar_radiation")
        if solar is not None and solar > SOLAR_OFFSET_THRESHOLD_W and outdoor > SOLAR_OFFSET_MIN_OUTDOOR_C:
            solar_reduction = min((solar - SOLAR_OFFSET_THRESHOLD_W) / 600 * 0.5, 0.5)
            offset -= solar_reduction
            _LOGGER.debug(
                "Solarbonus: %.0f W/m² → Offset −%.2f°C", solar, solar_reduction
            )

        # Round to nearest 0.5 °C – matches the config-step for all temperatures.
        # Prevents sub-0.5 °C micro-adjustments (e.g. -0.1 °C from solar reduction)
        # from showing a confusing "14.9 °C" when the user configured exactly 15.0 °C.
        return round(offset * 2) / 2

    def compute_forecast_suppression(
        self, weather_data: dict, target_temp: float, night_temp: float
    ) -> float:
        """Prognose-basierte Heizunterdrückung – Faktor 0.0 bis 1.0.

        Fragt: 'Muss ich jetzt heizen oder wird es tagsüber von selbst warm?'

        Rückgabe:
          1.0 = vollständig heizen (kalt heute)
          0.5 = halb heizen (wird mild)
          0.0 = nicht heizen (wird warm genug, Sonne heizt das Haus)

        Logik:
          - Tages-Hochtemperatur (Prognose) ≥ Zieltemperatur
            → Haus wird sich selbst erwärmen, kein Heizen nötig
          - Tages-Hochtemperatur nahe Zieltemperatur
            → Teilweise heizen (Sonne übernimmt den Rest)
          - Sonneneinstrahlung hoch + Außentemp > 5°C
            → Solarer Wärmeeintrag reduziert Heizbedarf zusätzlich
          - Kalter Tag (Prognose weit unter Ziel)
            → Vollständig heizen
        """
        forecast_high = weather_data.get("forecast_high")
        outdoor = weather_data.get("temperature") or 0.0
        solar = weather_data.get("solar_radiation") or 0.0

        factor = 1.0  # Standard: vollständig heizen

        # ── Prognose-basierte Unterdrückung ──────────────────────────
        if forecast_high is not None:
            if forecast_high >= target_temp:
                # Tages-Maximum erreicht oder überschreitet Zieltemperatur
                # → Haus wird sich selbst auf Zieltemperatur erwärmen
                factor = 0.0
            elif forecast_high >= night_temp:
                # Zwischen Nacht- und Komforttemperatur: lineare Reduktion
                # z.B. Ziel=21°C, Nacht=18°C, Prognose=20°C
                # → factor = (21-20)/(21-18) = 0.33 → 33% heizen
                span = target_temp - night_temp
                if span > 0:
                    remaining = target_temp - forecast_high
                    factor = max(0.0, min(1.0, remaining / span))

        # ── Sonneneinstrahlungs-Bonus ─────────────────────────────────
        # Sonne heizt den Raum zusätzlich → weitere Reduktion
        if solar > SOLAR_GAIN_THRESHOLD_W and outdoor > SOLAR_GAIN_MIN_OUTDOOR_C and factor > 0:
            solar_reduction = min((solar - SOLAR_GAIN_THRESHOLD_W) / 1000.0, SOLAR_GAIN_MAX_REDUCTION)
            factor = max(0.0, factor - solar_reduction)
            _LOGGER.debug(
                "Solarunterdrückung: %.0f W/m² → −%.0f%% Heizfaktor",
                solar, solar_reduction * 100
            )

        result = round(factor, 3)
        _LOGGER.debug(
            "Prognose-Unterdrückung: forecast_high=%.1f°C, target=%.1f°C, "
            "solar=%.0f W/m² → Heizfaktor %.0f%%",
            forecast_high or -99, target_temp, solar, result * 100
        )
        return result

    def is_heating_season(self, weather_data: dict) -> bool:
        """True wenn Heizung wahrscheinlich benötigt wird."""
        outdoor = weather_data.get("temperature")
        if outdoor is None:
            return True
        return outdoor < WEATHER_WARM_THRESHOLD

    def get_condition_category(self, weather_data: dict) -> str:
        """Wetterkategorie: cold / mild / warm / hot."""
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
