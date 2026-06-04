"""ThermoSmart – AI-powered, weather-aware heating control for Home Assistant.

Architektur:
  - Ein "system" Config-Eintrag: Globale Schalter (Sommer, Urlaub) – kein TRV nötig
  - Je ein "zone" Config-Eintrag pro Heizzone mit TRV-Steuerung und Lernalgorithmus

Modulstruktur:
  coordinator.py  – Hauptkoordinator (Update-Zyklus, Berechnung, Präsenz)
  trv_control.py  – TRV-Steuerung, Kalibrierung, Quirk-Management
  maintenance.py  – Wöchentliche Ventilübung
  season.py       – Sommer-Erkennung, Frostschutz
  window.py       – Fenstererkennung
  weather_engine.py / learning_engine.py – eigenständige Engines
"""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, Event

from .const import (
    DOMAIN,
    PLATFORMS,
    CONF_WEATHER_ENTITY,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR,
    CONF_OUTDOOR_SOLAR_SENSOR,
    CONF_OUTDOOR_RAIN_SENSOR,
    CONF_LEARNING_ENABLED,
)
from .coordinator import ThermoSmartCoordinator
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine

_LOGGER = logging.getLogger(__name__)

ZONE_PLATFORMS = PLATFORMS          # ["climate", "sensor", "switch", "select"]
SYSTEM_PLATFORMS = ["switch"]       # System-Entry: nur globale Schalter


def _validate_sensors(hass: HomeAssistant, cfg: dict) -> None:
    """Prüft ob alle konfigurierten Sensor-Entities beim Start existieren.

    Loggt eine einmalige Warning für jede fehlende Entity damit Fehlkonfigurationen
    sofort erkennbar sind – verhindert stille None-Werte im Betrieb.
    """
    zone_name = cfg.get("name", "?")

    # Einzelne Sensor-Entities
    sensor_keys = [
        (CONF_OUTDOOR_TEMP_SENSOR,     "Außentemperatur-Sensor"),
        (CONF_OUTDOOR_HUMIDITY_SENSOR, "Außenfeuchte-Sensor"),
        (CONF_OUTDOOR_WIND_SENSOR,     "Wind-Sensor"),
        (CONF_OUTDOOR_SOLAR_SENSOR,    "Solar-Sensor"),
        (CONF_OUTDOOR_RAIN_SENSOR,     "Regen-Sensor"),
    ]
    for key, label in sensor_keys:
        entity_id = cfg.get(key)
        if entity_id and hass.states.get(entity_id) is None:
            _LOGGER.warning(
                "ThermoSmart '%s': %s '%s' nicht gefunden – Wert wird ignoriert",
                zone_name, label, entity_id,
            )

    # Listen-Entities (Temperatur- und Feuchte-Sensoren der Zone)
    for key, label in (
        ("temp_sensors",     "Raumtemperatur-Sensor"),
        ("humidity_sensors", "Raumfeuchte-Sensor"),
        ("window_sensors",   "Fenstersensor"),
        ("climate_entities", "Klima-Entity (TRV)"),
    ):
        for entity_id in cfg.get(key, []):
            if entity_id and hass.states.get(entity_id) is None:
                _LOGGER.warning(
                    "ThermoSmart '%s': %s '%s' nicht gefunden",
                    zone_name, label, entity_id,
                )


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    hass.data.setdefault(DOMAIN, {})
    cfg = {**entry.data, **entry.options}

    # ── System-Entry: nur globale Schalter, kein Coordinator ────────
    if cfg.get("entry_type") == "system":
        hass.data[DOMAIN][entry.entry_id] = {"type": "system"}
        await hass.config_entries.async_forward_entry_setups(entry, SYSTEM_PLATFORMS)
        _LOGGER.info("ThermoSmart System geladen (globale Schalter)")
        return True

    # ── Zone-Entry: vollständiger Coordinator ───────────────────────
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

    if "learning_engine" not in hass.data[DOMAIN]:
        learning_engine = LearningEngine(hass)
        await learning_engine.async_load()

        # Veraltete Zonen bereinigen (nicht mehr in Config-Entries vorhanden)
        active_zone_ids = {
            e.entry_id
            for e in hass.config_entries.async_entries(DOMAIN)
            if e.data.get("entry_type") != "system"
        }
        learning_engine.prune_orphaned_zones(active_zone_ids)

        hass.data[DOMAIN]["learning_engine"] = learning_engine
        _LOGGER.debug("ThermoSmart: Gemeinsame LearningEngine erstellt")

        async def _save_on_stop(event: Event) -> None:
            """Lerndaten vor HA-Shutdown sicher auf Disk schreiben."""
            if le := hass.data.get(DOMAIN, {}).get("learning_engine"):
                await le.async_save()
                _LOGGER.info("ThermoSmart: Lerndaten vor Shutdown gespeichert")

        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _save_on_stop)
    else:
        learning_engine = hass.data[DOMAIN]["learning_engine"]
        _LOGGER.debug("ThermoSmart: Gemeinsame LearningEngine wiederverwendet")

    learning_enabled = cfg.get(CONF_LEARNING_ENABLED, True)
    learning_engine.set_zone_enabled(entry.entry_id, learning_enabled)

    # Sensor-Existenz prüfen – einmalig beim Start, damit Fehlkonfigurationen
    # sofort im Log sichtbar sind statt erst nach 5 Minuten stillem None
    _validate_sensors(hass, cfg)

    coordinator = ThermoSmartCoordinator(
        hass, entry,
        weather_engine=weather_engine,
        learning_engine=learning_engine,
    )
    await coordinator.async_config_entry_first_refresh()
    await coordinator.async_detect_device_entities()
    coordinator.setup_event_listeners()
    entry.async_on_unload(coordinator.cleanup_event_listeners)

    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
        "weather_engine": weather_engine,
    }

    await hass.config_entries.async_forward_entry_setups(entry, ZONE_PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    _LOGGER.info(
        "ThermoSmart Zone '%s' geladen – Beobachtungsmodus (Aktive Steuerung AUS)",
        cfg.get("name", entry.entry_id),
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    cfg = {**entry.data, **entry.options}
    platforms = SYSTEM_PLATFORMS if cfg.get("entry_type") == "system" else ZONE_PLATFORMS

    if unload_ok := await hass.config_entries.async_unload_platforms(entry, platforms):
        hass.data[DOMAIN].pop(entry.entry_id, {})
        if cfg.get("entry_type") != "system":
            remaining = [
                k for k in hass.data[DOMAIN]
                if k not in ("learning_engine", "global_switches_created")
                and not (isinstance(hass.data[DOMAIN].get(k), dict)
                         and hass.data[DOMAIN][k].get("type") == "system")
            ]
            if not remaining:
                if le := hass.data[DOMAIN].pop("learning_engine", None):
                    await le.async_save()
                    _LOGGER.debug("ThermoSmart: LearningEngine gespeichert und entfernt")
            else:
                if le := hass.data[DOMAIN].get("learning_engine"):
                    await le.async_save()
    return unload_ok


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    if {**entry.data, **entry.options}.get("entry_type") != "system":
        await hass.config_entries.async_reload(entry.entry_id)
