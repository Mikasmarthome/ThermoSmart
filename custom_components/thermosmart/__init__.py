"""ThermoSmart – Self-learning, weather-aware heating control for Home Assistant.

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
import os

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, Event, ServiceCall
from homeassistant.components.persistent_notification import async_create as _pn_create

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
    DOMAIN_GLOBAL_SUMMER,
)
from .coordinator import ThermoSmartCoordinator
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine
from .export import async_export_learning_data, build_export_notification_message

_LOGGER = logging.getLogger(__name__)

ZONE_PLATFORMS = PLATFORMS               # ["climate", "sensor", "switch", "select"]
SYSTEM_PLATFORMS = ["button", "switch", "select"]  # System-Entry: globale Schalter + Sommer-Select


def _migrate_old_summer_switch(hass: HomeAssistant) -> None:
    """Entfernt veraltete Summer-Switch-Entity aus der HA Entity-Registry (Migration beta.23).

    In beta.23 wurde der globale Summer-Schalter (switch) durch einen dreistufigen
    Select (Automatic / On / Off) ersetzt. Die alte Switch-Entity mit unique_id
    DOMAIN_GLOBAL_SUMMER wird einmalig pro Session automatisch aus der Registry entfernt,
    damit keine verwaiste "unavailable"-Entity zurückbleibt.
    """
    from homeassistant.helpers import entity_registry as er  # lokaler Import – vermeidet Zyklen
    entity_reg = er.async_get(hass)
    old_entity_id = entity_reg.async_get_entity_id("switch", DOMAIN, DOMAIN_GLOBAL_SUMMER)
    if old_entity_id:
        entity_reg.async_remove(old_entity_id)
        _LOGGER.info(
            "ThermoSmart: Migration beta.23 – "
            "Alte Summer-Switch-Entity '%s' aus der Registry entfernt",
            old_entity_id,
        )


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

    # Einmalige Migration pro HA-Session: alte Summer-Switch-Entity entfernen (beta.23)
    if "summer_switch_migrated" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["summer_switch_migrated"] = True
        _migrate_old_summer_switch(hass)

    # ── System-Entry: nur globale Schalter, kein Coordinator ────────
    if cfg.get("entry_type") == "system":
        hass.data[DOMAIN][entry.entry_id] = {"type": "system"}
        await hass.config_entries.async_forward_entry_setups(entry, SYSTEM_PLATFORMS)

        if not hass.services.has_service(DOMAIN, "export_learning_data"):
            async def _handle_export(call: ServiceCall) -> None:
                filepath = await async_export_learning_data(hass)
                filename = os.path.basename(filepath)
                _pn_create(
                    hass,
                    message=build_export_notification_message(filename),
                    title="ThermoSmart — Learning Data Export",
                    notification_id="thermosmart_export",
                )

            hass.services.async_register(DOMAIN, "export_learning_data", _handle_export)
            _LOGGER.debug("ThermoSmart: export_learning_data service registered")

        _LOGGER.info("ThermoSmart System geladen (globale Schalter)")
        return True

    # ── Zone-Entry: vollständiger Coordinator ───────────────────────
    def _sensor(key: str) -> str | None:
        return cfg.get(key) or None

    weather_engine = WeatherEngine(
        hass,
        cfg.get(CONF_WEATHER_ENTITY) or "weather.home",
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

    # Apply global overrides that were already restored by the system entry
    # before this zone loaded (race condition fix: system entry may load first).
    _SUMMER_OPT_MAP: dict[str, bool | None] = {"automatic": None, "on": True, "off": False}
    _needs_refresh = False
    if hass.data[DOMAIN].get("global_vacation_override"):
        coordinator.set_vacation_override(True)
        _needs_refresh = True
    _summer_opt = hass.data[DOMAIN].get("global_summer_override")
    if _summer_opt is not None:
        coordinator.set_summer_override(_SUMMER_OPT_MAP.get(_summer_opt))
        _needs_refresh = True
    if _needs_refresh:
        await coordinator.async_request_refresh()

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
                if k not in ("learning_engine", "global_switches_created",
                             "global_summer_select_created", "summer_switch_migrated",
                             "global_vacation_override", "global_summer_override")
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


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Lerndaten der entfernten Zone sofort bereinigen.

    Wird von HA aufgerufen NACHDEM die Zone vollständig aus config_entries entfernt
    wurde – zu diesem Zeitpunkt ist die zone_id bereits aus dem Entry-Register verschwunden,
    sodass prune_orphaned_zones sie korrekt als verwaist erkennt.
    """
    if entry.data.get("entry_type") == "system":
        return
    le = hass.data.get(DOMAIN, {}).get("learning_engine")
    if le is None:
        return
    active_zone_ids = {
        e.entry_id
        for e in hass.config_entries.async_entries(DOMAIN)
        if e.data.get("entry_type") != "system"
    }
    le.prune_orphaned_zones(active_zone_ids)
    _LOGGER.info(
        "ThermoSmart: Zone '%s' entfernt – Lerndaten sofort bereinigt",
        entry.data.get("name", entry.entry_id),
    )


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    if {**entry.data, **entry.options}.get("entry_type") != "system":
        await hass.config_entries.async_reload(entry.entry_id)
