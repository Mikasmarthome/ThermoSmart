"""ThermoSmart – AI-powered, weather-aware heating control for Home Assistant.

Architektur: Ein Config-Eintrag = Eine Heizzone.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    DOMAIN,
    DEFAULT_SCAN_INTERVAL,
    PLATFORMS,
    CONF_WEATHER_ENTITY,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR,
    CONF_OUTDOOR_SOLAR_SENSOR,
    CONF_OUTDOOR_RAIN_SENSOR,
    CONF_LEARNING_ENABLED,
    TEMP_NIGHT,
    PRESENCE_PERSONS,
    PRESENCE_PREHEAT_ZONE,
    VACATION_BOOLEAN,
    HEATING_MODE_AUTO,
    HEATING_MODE_AWAY,
    HEATING_MODE_VACATION,
    HEATING_MODE_COMFORT,
    HEATING_MODE_NIGHT,
)
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    hass.data.setdefault(DOMAIN, {})

    cfg = {**entry.data, **entry.options}

    def _sensor(key: str) -> str | None:
        return cfg.get(key) or None

    weather_engine = WeatherEngine(
        hass,
        cfg.get(CONF_WEATHER_ENTITY, "weather.home"),
        outdoor_temp_sensor=_sensor(CONF_OUTDOOR_TEMP_SENSOR),
        outdoor_humidity_sensor=_sensor(CONF_OUTDOOR_HUMIDITY_SENSOR),
        outdoor_wind_sensor=_sensor(CONF_OUTDOOR_WIND_SENSOR),
        outdoor_solar_sensor=_sensor(CONF_OUTDOOR_SOLAR_SENSOR),
        outdoor_rain_sensor=_sensor(CONF_OUTDOOR_RAIN_SENSOR),
    )

    learning_enabled = cfg.get(CONF_LEARNING_ENABLED, True)
    learning_engine = LearningEngine(hass, learning_enabled)
    await learning_engine.async_load()

    coordinator = ThermoSmartCoordinator(
        hass, entry,
        weather_engine=weather_engine,
        learning_engine=learning_engine,
    )
    await coordinator.async_config_entry_first_refresh()

    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
        "weather_engine": weather_engine,
        "learning_engine": learning_engine,
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    _LOGGER.info(
        "ThermoSmart Zone '%s' geladen – Beobachtungsmodus (Aktive Steuerung AUS)",
        cfg.get("name", entry.entry_id),
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        data = hass.data[DOMAIN].pop(entry.entry_id, {})
        if le := data.get("learning_engine"):
            await le.async_save()
    return unload_ok


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


class ThermoSmartCoordinator(DataUpdateCoordinator):
    """Koordinator für eine einzelne Heizzone."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        weather_engine: WeatherEngine,
        learning_engine: LearningEngine,
    ) -> None:
        super().__init__(
            hass, _LOGGER,
            name=f"{DOMAIN}_{entry.entry_id}",
            update_interval=timedelta(seconds=DEFAULT_SCAN_INTERVAL),
        )
        self.entry = entry
        self.weather_engine = weather_engine
        self.learning_engine = learning_engine
        self._active_control: bool = False
        self._mode: str = HEATING_MODE_AUTO
        self._override: float | None = None

    # ── Eigenschaften ────────────────────────────────────────────────

    @property
    def zone_id(self) -> str:
        return self.entry.entry_id

    @property
    def zone_name(self) -> str:
        return self.entry.data.get("name", "Zone")

    @property
    def zone_cfg(self) -> dict:
        return {**self.entry.data, **self.entry.options}

    # ── Setter ──────────────────────────────────────────────────────

    def set_active_control(self, active: bool) -> None:
        self._active_control = active
        _LOGGER.warning(
            "ThermoSmart '%s': Aktive Steuerung %s",
            self.zone_name,
            "AN – Thermostat wird gesteuert" if active else "AUS – Beobachtungsmodus",
        )

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_override(self, value: float) -> None:
        self._override = value if value >= 5.0 else None

    def get_override(self) -> float | None:
        return self._override

    # ── Hauptschleife ────────────────────────────────────────────────

    async def _async_update_data(self) -> dict:
        try:
            cfg = self.zone_cfg
            weather_data = await self.weather_engine.async_get_data()
            presence = self._get_presence_state()
            mode = self._effective_mode(presence)
            recommendation = await self._compute_recommendation(cfg, weather_data, mode)

            indoor_humidity = self._read_avg_sensor(cfg.get("humidity_sensors", []))
            await self.learning_engine.async_observe(
                zone_id=self.zone_id,
                recommendation=recommendation,
                weather_data=weather_data,
                indoor_humidity=indoor_humidity,
            )

            if self._active_control:
                await self._apply_temperature(cfg, recommendation)

            return {
                "weather": weather_data,
                "zone": recommendation,
                "active_control": self._active_control,
                "presence": presence,
            }
        except Exception as err:
            raise UpdateFailed(f"ThermoSmart '{self.zone_name}': {err}") from err

    # ── Präsenz ──────────────────────────────────────────────────────

    def _get_presence_state(self) -> dict:
        persons_home, persons_away = [], []
        for person in PRESENCE_PERSONS:
            state = self.hass.states.get(person)
            (persons_home if (state and state.state == "home") else persons_away).append(person)

        vacation = False
        vs = self.hass.states.get(VACATION_BOOLEAN)
        if vs and vs.state == "on":
            vacation = True

        return {
            "persons_home": persons_home,
            "all_away": len(persons_home) == 0,
            "vacation": vacation,
        }

    def _effective_mode(self, presence: dict) -> str:
        if presence["vacation"]:
            return HEATING_MODE_VACATION
        if self._mode != HEATING_MODE_AUTO:
            return self._mode
        if presence["all_away"]:
            return HEATING_MODE_AWAY
        return HEATING_MODE_AUTO

    # ── Berechnung ───────────────────────────────────────────────────

    async def _compute_recommendation(
        self, cfg: dict, weather_data: dict, mode: str
    ) -> dict:
        # Innentemperatur (Durchschnitt mehrerer Sensoren)
        current_temp = self._read_avg_sensor(cfg.get("temp_sensors", []))

        # Fenster offen?
        window_open = self._check_window_open(cfg)

        # Basis-Zieltemperatur
        base_target = await self.learning_engine.async_get_base_target(
            self.zone_id, mode,
            comfort_temp=cfg.get("comfort_temp", 21.0),
            night_temp=cfg.get("night_temp", 18.0),
            away_temp=cfg.get("away_temp", 17.0),
        )

        # Wetterkorrektur (nur Auto-Modus)
        weather_offset = 0.0
        if mode == HEATING_MODE_AUTO:
            weather_offset = self.weather_engine.compute_temperature_offset(weather_data)

        # Vorheizzeit
        preheat_minutes = await self.learning_engine.async_get_preheat_minutes(
            self.zone_id, base_target, current_temp, weather_data
        )

        # Zieltemperatur berechnen
        override = self.get_override()
        if override is not None and not window_open:
            adjusted_target = override
            override_active = True
        elif not window_open:
            raw_target = round(base_target + weather_offset, 1)
            if mode == HEATING_MODE_AUTO:
                night_temp = cfg.get("night_temp", TEMP_NIGHT)
                suppression = self.weather_engine.compute_forecast_suppression(
                    weather_data, raw_target, night_temp
                )
                adjusted_target = round(
                    night_temp + suppression * (raw_target - night_temp), 1
                ) if suppression < 1.0 else raw_target
            else:
                adjusted_target = raw_target
            override_active = False
        else:
            adjusted_target = None
            override_active = False

        # Prognose-Unterdrückung für Sensor
        if mode == HEATING_MODE_AUTO and not window_open and override is None:
            night_temp = cfg.get("night_temp", TEMP_NIGHT)
            suppression_pct = round(
                (1 - self.weather_engine.compute_forecast_suppression(
                    weather_data, round(base_target + weather_offset, 1), night_temp
                )) * 100
            )
        else:
            suppression_pct = 0

        return {
            "zone_name": self.zone_name,
            "mode": mode,
            "current_temp": current_temp,
            "window_open": window_open,
            "base_target": base_target,
            "weather_offset": weather_offset,
            "adjusted_target": adjusted_target,
            "override_active": override_active,
            "override_temp": override,
            "preheat_minutes": preheat_minutes,
            "forecast_suppression": suppression_pct,
            "learning_confidence": self.learning_engine.get_confidence(self.zone_id),
            "outdoor_temp": weather_data.get("temperature"),
            "forecast_high": weather_data.get("forecast_high"),
            "weather_condition": weather_data.get("condition"),
        }

    def _check_window_open(self, cfg: dict) -> bool:
        for ws_id in cfg.get("window_sensors", []):
            if not ws_id:
                continue
            ws = self.hass.states.get(ws_id)
            if ws and ws.state == "on":
                return True
        return False

    def _read_avg_sensor(self, sensor_ids: list[str]) -> float | None:
        values = []
        for sid in sensor_ids:
            if not sid:
                continue
            state = self.hass.states.get(sid)
            if state and state.state not in ("unknown", "unavailable", "None"):
                try:
                    values.append(float(state.state))
                except ValueError:
                    pass
        return round(sum(values) / len(values), 1) if values else None

    # ── Thermostat schreiben ─────────────────────────────────────────

    async def _apply_temperature(self, cfg: dict, recommendation: dict) -> None:
        target = recommendation.get("adjusted_target")
        if target is None:
            return
        if not (5.0 <= target <= 30.0):
            _LOGGER.warning(
                "ThermoSmart '%s': Ziel %.1f°C außerhalb Sicherheitsbereich",
                self.zone_name, target,
            )
            return

        tolerance = cfg.get("temp_tolerance", 0.5)
        for entity_id in cfg.get("climate_entities", []):
            state = self.hass.states.get(entity_id)
            if not state or state.state in ("unavailable", "unknown"):
                continue
            current_setpoint = state.attributes.get("temperature")
            if current_setpoint is not None:
                try:
                    if abs(float(current_setpoint) - target) < tolerance:
                        continue
                except (TypeError, ValueError):
                    pass
            await self.hass.services.async_call(
                "climate", "set_temperature",
                {"entity_id": entity_id, "temperature": target},
                blocking=False,
            )
            _LOGGER.debug("ThermoSmart '%s' → %s: %.1f°C", self.zone_name, entity_id, target)
