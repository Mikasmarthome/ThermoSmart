"""ThermoSmart – AI-powered, weather-aware heating control for Home Assistant."""
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
    CONF_LEARNING_ENABLED,
    ZONES,
    ZONE_TEMPS,
    TEMP_COMFORT,
    TEMP_NIGHT,
    TEMP_AWAY,
    PRESENCE_PERSONS,
    PRESENCE_PREHEAT_ZONE,
    VACATION_BOOLEAN,
    HEATING_MODE_AUTO,
    HEATING_MODE_AWAY,
    HEATING_MODE_VACATION,
)
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """ThermoSmart aus einem Config Entry einrichten."""
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

    _LOGGER.info(
        "ThermoSmart geladen – Beobachtungsmodus aktiv "
        "(Aktive Steuerung ist AUS, Thermostate werden NICHT verändert)"
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Config Entry entladen."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        data = hass.data[DOMAIN].pop(entry.entry_id, {})
        learning_engine: LearningEngine = data.get("learning_engine")
        if learning_engine:
            await learning_engine.async_save()
    return unload_ok


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


class ThermoSmartCoordinator(DataUpdateCoordinator):
    """Koordiniert alle Berechnungen und – wenn aktiv – die Thermostat-Steuerung."""

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
        self._overrides: dict[str, float] = {}
        self._zone_modes: dict[str, str] = {}

        # ═══════════════════════════════════════════════════════════════
        # BEOBACHTUNGSMODUS: Standard AUS
        # Thermostate werden erst beschrieben wenn _active_control = True
        # ═══════════════════════════════════════════════════════════════
        self._active_control: bool = False
        self._zones_cache: dict | None = None  # wird bei Options-Änderung geleert

    # ------------------------------------------------------------------
    # Setter (von switch.py / select.py / number.py aufgerufen)
    # ------------------------------------------------------------------

    @property
    def zones(self) -> dict:
        """Aktive Zonen – aus UI-Konfiguration oder const.py-Fallback.

        Gibt immer ein normalisiertes Dict zurück:
          zone_id → {name, climate_entities, temp_sensors,
                     humidity_sensors, window_sensor,
                     comfort_temp, night_temp, away_temp}
        """
        options_zones: list = self.entry.options.get("zones", [])
        if options_zones:
            result = {}
            for z in options_zones:
                zid = z["zone_id"]
                result[zid] = {
                    "name": z["name"],
                    "climate_entities": z.get("climate_entities", []),
                    "temp_sensors": z.get("temp_sensors", []),
                    "humidity_sensors": z.get("humidity_sensors", []),
                    "window_sensor": z.get("window_sensor"),
                    "comfort_temp": z.get("comfort_temp", TEMP_COMFORT),
                    "night_temp": z.get("night_temp", TEMP_NIGHT),
                    "away_temp": z.get("away_temp", TEMP_AWAY),
                }
            return result

        # Fallback: hardcodierte Zonen aus const.py
        result = {}
        for zone_id, zone_cfg in ZONES.items():
            temps = ZONE_TEMPS.get(zone_id, {})
            climate = zone_cfg.get("climate_entity", [])
            result[zone_id] = {
                "name": zone_cfg["name"],
                "climate_entities": [climate] if isinstance(climate, str) else list(climate),
                "temp_sensors": [s for s in [zone_cfg.get("temp_sensor")] if s],
                "humidity_sensors": [s for s in [zone_cfg.get("humidity_sensor")] if s],
                "window_sensor": zone_cfg.get("window_sensor"),
                "comfort_temp": temps.get("comfort", TEMP_COMFORT),
                "night_temp": temps.get("night", TEMP_NIGHT),
                "away_temp": temps.get("away", TEMP_AWAY),
            }
        return result

    def _read_avg_sensor(self, sensor_ids: list[str]) -> float | None:
        """Durchschnittswert mehrerer Sensoren berechnen (unavailable wird übersprungen)."""
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

    def set_active_control(self, active: bool) -> None:
        """Aktive Steuerung ein-/ausschalten.
        False = Beobachtungsmodus (Thermostate werden NICHT verändert).
        True  = ThermoSmart übernimmt die Steuerung.
        """
        self._active_control = active
        _LOGGER.warning(
            "ThermoSmart Aktive Steuerung: %s",
            "AN – Thermostate werden gesteuert" if active
            else "AUS – Beobachtungsmodus, keine Änderungen an Thermostaten",
        )

    def set_override(self, zone_id: str, value: float) -> None:
        if value >= 5.0:
            self._overrides[zone_id] = value
        else:
            self._overrides.pop(zone_id, None)

    def get_override(self, zone_id: str) -> float | None:
        return self._overrides.get(zone_id)

    def set_zone_mode(self, zone_id: str, mode: str) -> None:
        self._zone_modes[zone_id] = mode

    def get_zone_mode(self, zone_id: str) -> str:
        return self._zone_modes.get(zone_id, HEATING_MODE_AUTO)

    # ------------------------------------------------------------------
    # Haupt-Update-Schleife
    # ------------------------------------------------------------------

    async def _async_update_data(self) -> dict:
        """Alle 5 Minuten: Daten holen, Empfehlungen berechnen, ggf. anwenden."""
        try:
            weather_data = await self.weather_engine.async_get_data()
            presence = self._get_presence_state()
            zone_recommendations: dict = {}

            for zone_id, zone_cfg in self.zones.items():
                mode = self._effective_mode(zone_id, presence)
                recommendation = await self._compute_zone_recommendation(
                    zone_id, zone_cfg, weather_data, mode
                )
                zone_recommendations[zone_id] = recommendation

                # Lernalgorithmus beobachtet
                await self.learning_engine.async_observe(
                    zone_id=zone_id,
                    recommendation=recommendation,
                    weather_data=weather_data,
                )

                # ── NUR wenn Aktive Steuerung AN ──────────────────────
                if self._active_control:
                    await self._apply_zone_temperature(zone_id, zone_cfg, recommendation)

            return {
                "weather": weather_data,
                "zones": zone_recommendations,
                "presence": presence,
                "active_control": self._active_control,
            }
        except Exception as err:
            raise UpdateFailed(f"ThermoSmart Update fehlgeschlagen: {err}") from err

    # ------------------------------------------------------------------
    # Präsenz-Erkennung
    # ------------------------------------------------------------------

    def _get_presence_state(self) -> dict:
        """Gibt zurück wer zuhause ist und ob Urlaub aktiv ist."""
        persons_home = []
        persons_away = []

        for person in PRESENCE_PERSONS:
            state = self.hass.states.get(person)
            if state and state.state == "home":
                persons_home.append(person)
            else:
                persons_away.append(person)

        # Urlaub prüfen
        vacation = False
        vac_state = self.hass.states.get(VACATION_BOOLEAN)
        if vac_state and vac_state.state == "on":
            vacation = True

        # Jemand nähert sich der Heizungszone?
        someone_approaching = False
        if PRESENCE_PREHEAT_ZONE:
            for person in PRESENCE_PERSONS:
                state = self.hass.states.get(person)
                if state and state.state == PRESENCE_PREHEAT_ZONE.replace("zone.", ""):
                    someone_approaching = True

        return {
            "persons_home": persons_home,
            "persons_away": persons_away,
            "anyone_home": len(persons_home) > 0,
            "all_away": len(persons_home) == 0,
            "vacation": vacation,
            "someone_approaching": someone_approaching,
        }

    def _effective_mode(self, zone_id: str, presence: dict) -> str:
        """Effektiven Heizmodus für eine Zone bestimmen.

        Priorität:
          1. Urlaubsmodus (höchste Priorität)
          2. Manuell gesetzter Modus (via select.py)
          3. Präsenz-basierter Auto-Modus
        """
        # 1. Urlaub
        if presence["vacation"]:
            return HEATING_MODE_VACATION

        # 2. Manuell gesetzt und nicht Auto
        manual_mode = self._zone_modes.get(zone_id, HEATING_MODE_AUTO)
        if manual_mode != HEATING_MODE_AUTO:
            return manual_mode

        # 3. Auto: Präsenz-basiert
        if presence["all_away"]:
            return HEATING_MODE_AWAY

        return HEATING_MODE_AUTO

    # ------------------------------------------------------------------
    # Zonenberechnung
    # ------------------------------------------------------------------

    async def _compute_zone_recommendation(
        self, zone_id: str, zone_cfg: dict, weather_data: dict, mode: str
    ) -> dict:
        """Empfehlung für eine Zone berechnen."""
        # Ist-Temperatur – Durchschnitt aller konfigurierten Sensoren
        current_temp = self._read_avg_sensor(zone_cfg.get("temp_sensors", []))

        # Fenster offen?
        window_open = False
        window_sensor = zone_cfg.get("window_sensor")
        if window_sensor:
            ws = self.hass.states.get(window_sensor)
            window_open = ws is not None and ws.state == "on"

        # Basis-Zieltemperatur (Lernalgorithmus + Modus)
        base_target = await self.learning_engine.async_get_base_target(zone_id, mode)

        # Wetterkorrektur (nur im Auto-Modus)
        weather_offset = 0.0
        if mode == HEATING_MODE_AUTO:
            weather_offset = self.weather_engine.compute_temperature_offset(weather_data)

        # Vorheizzeit
        preheat_minutes = await self.learning_engine.async_get_preheat_minutes(
            zone_id, base_target, current_temp, weather_data
        )

        # Override hat höchste Priorität (außer Fenster offen)
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
            "mode": mode,
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

    # ------------------------------------------------------------------
    # Thermostat-Steuerung (NUR wenn _active_control = True)
    # ------------------------------------------------------------------

    async def _apply_zone_temperature(
        self, zone_id: str, zone_cfg: dict, recommendation: dict
    ) -> None:
        """Zieltemperatur ans Thermostat schicken.

        Wird NUR aufgerufen wenn _active_control = True.
        Sicherheitsprüfungen:
          - Kein Schreiben wenn Zieltemperatur None (Fenster offen)
          - Kein Schreiben wenn Temperatur außerhalb 5–30°C
        """
        target = recommendation.get("adjusted_target")
        if target is None:
            return
        if not (5.0 <= target <= 30.0):
            _LOGGER.warning(
                "ThermoSmart [%s]: Zieltemperatur %.1f°C außerhalb Sicherheitsbereich, übersprungen",
                zone_id, target,
            )
            return

        climate_entities = zone_cfg.get("climate_entities", [])

        for entity_id in climate_entities:
            state = self.hass.states.get(entity_id)
            if state is None or state.state in ("unavailable", "unknown"):
                continue

            # Nur schreiben wenn Änderung > 0.4°C (Unnötige Befehle vermeiden)
            current_setpoint = state.attributes.get("temperature")
            if current_setpoint is not None:
                try:
                    if abs(float(current_setpoint) - target) < 0.4:
                        continue
                except (TypeError, ValueError):
                    pass

            await self.hass.services.async_call(
                "climate",
                "set_temperature",
                {"entity_id": entity_id, "temperature": target},
                blocking=False,
            )
            _LOGGER.debug(
                "ThermoSmart [%s] → %s: %.1f°C gesetzt", zone_id, entity_id, target
            )
