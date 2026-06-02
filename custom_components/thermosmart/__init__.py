"""ThermoSmart – AI-powered, weather-aware heating control for Home Assistant."""
from __future__ import annotations

import logging
from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    DOMAIN,
    DEFAULT_SCAN_INTERVAL,
    PLATFORMS,
    CONF_WEATHER_ENTITY,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_LEARNING_ENABLED,
    ZONES,
)
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up ThermoSmart from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    weather_entity = entry.data.get(CONF_WEATHER_ENTITY, "weather.home")
    outdoor_sensor = entry.data.get(CONF_OUTDOOR_TEMP_SENSOR)
    learning_enabled = entry.data.get(CONF_LEARNING_ENABLED, True)

    weather_engine = WeatherEngine(hass, weather_entity, outdoor_sensor)
    learning_engine = LearningEngine(hass, learning_enabled)
    await learning_engine.async_load()

    coordinator = ThermoSmartCoordinator(
        hass,
        entry,
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

    _LOGGER.info("ThermoSmart integration loaded successfully")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        data = hass.data[DOMAIN].pop(entry.entry_id, {})
        learning_engine: LearningEngine = data.get("learning_engine")
        if learning_engine:
            await learning_engine.async_save()
    return unload_ok


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


class ThermoSmartCoordinator(DataUpdateCoordinator):
    """Coordinator that fetches all ThermoSmart data and runs the heating logic."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        weather_engine: WeatherEngine,
        learning_engine: LearningEngine,
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=DEFAULT_SCAN_INTERVAL),
        )
        self.entry = entry
        self.weather_engine = weather_engine
        self.learning_engine = learning_engine
        self._zone_states: dict = {}
        self._overrides: dict[str, float] = {}  # zone_id → override °C (0 = kein Override)

    def set_override(self, zone_id: str, value: float) -> None:
        """Override für eine Zone setzen. value=0 → kein Override."""
        if value >= 5.0:
            self._overrides[zone_id] = value
            _LOGGER.debug("Override gesetzt: %s → %.1f°C", zone_id, value)
        else:
            self._overrides.pop(zone_id, None)
            _LOGGER.debug("Override entfernt: %s", zone_id)

    def get_override(self, zone_id: str) -> float | None:
        """Aktiven Override zurückgeben, oder None wenn kein Override."""
        return self._overrides.get(zone_id)

    async def _async_update_data(self) -> dict:
        """Fetch data and compute recommendations for every zone."""
        try:
            weather_data = await self.weather_engine.async_get_data()
            zone_recommendations: dict = {}

            for zone_id, zone_cfg in ZONES.items():
                recommendation = await self._compute_zone_recommendation(
                    zone_id, zone_cfg, weather_data
                )
                zone_recommendations[zone_id] = recommendation

                # Let the learning engine observe the outcome
                await self.learning_engine.async_observe(
                    zone_id=zone_id,
                    recommendation=recommendation,
                    weather_data=weather_data,
                )

            return {
                "weather": weather_data,
                "zones": zone_recommendations,
            }
        except Exception as err:
            raise UpdateFailed(f"ThermoSmart update failed: {err}") from err

    async def _compute_zone_recommendation(
        self, zone_id: str, zone_cfg: dict, weather_data: dict
    ) -> dict:
        """
        Core logic: combine schedule, weather offsets, window state,
        and learning-engine preheat suggestion into a single recommendation.
        """
        # Current indoor temperature
        temp_sensor = zone_cfg.get("temp_sensor")
        current_temp: float | None = None
        if temp_sensor:
            state = self.hass.states.get(temp_sensor)
            if state and state.state not in ("unknown", "unavailable"):
                try:
                    current_temp = float(state.state)
                except ValueError:
                    pass

        # Window open?
        window_sensor = zone_cfg.get("window_sensor")
        window_open = False
        if window_sensor:
            ws = self.hass.states.get(window_sensor)
            window_open = ws is not None and ws.state == "on"

        # Base target from the learning engine (falls back to schedule)
        base_target = await self.learning_engine.async_get_base_target(zone_id)

        # Weather adjustment
        weather_offset = self.weather_engine.compute_temperature_offset(weather_data)

        # Preheat suggestion
        preheat_minutes = await self.learning_engine.async_get_preheat_minutes(
            zone_id, base_target, current_temp, weather_data
        )

        # Manueller Override hat höchste Priorität
        override = self.get_override(zone_id)
        if override is not None and not window_open:
            adjusted_target = override
            override_active = True
        elif not window_open:
            adjusted_target = round(base_target + weather_offset, 1)
            override_active = False
        else:
            adjusted_target = None
            override_active = False

        return {
            "zone_id": zone_id,
            "zone_name": zone_cfg["name"],
            "current_temp": current_temp,
            "window_open": window_open,
            "base_target": base_target,
            "weather_offset": weather_offset,
            "adjusted_target": adjusted_target,
            "override_active": override_active,
            "override_temp": override,
            "preheat_minutes": preheat_minutes,
            "learning_confidence": self.learning_engine.get_confidence(zone_id),
            "outdoor_temp": weather_data.get("temperature"),
            "weather_condition": weather_data.get("condition"),
        }
