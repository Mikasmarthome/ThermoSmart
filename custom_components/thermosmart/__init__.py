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
    CARD_FILENAME,
    CARD_URL_PATH,
    VERSION,
)
from .coordinator import ThermoSmartCoordinator
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine
from .export import (
    ThermoSmartExportDownloadView,
    async_build_export_notification,
    async_cleanup_expired_exports,
    async_export_learning_data,
)

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


async def _async_register_card(hass: HomeAssistant) -> None:
    """Serve the bundled Lovelace card and register it as a frontend module URL.

    Runs at most once per HA session (guarded by the caller via
    hass.data[DOMAIN]["card_registered"]). Two independent, non-fatal steps:

    1. Static path: custom_components/thermosmart/www/thermosmart-card.js is
       exposed at CARD_URL_PATH via the async, non-blocking HA 2024.x static
       path API (StaticPathConfig) — no synchronous file I/O on the event loop.
    2. Frontend module URL: add_extra_js_url() registers the same URL so every
       dashboard loads it automatically, in both storage-mode and YAML-mode —
       no .storage/lovelace_resources entry is written or managed. The helper
       stores URLs in a set, so calling it again (e.g. a second HA session)
       never creates a duplicate <script> tag.

    A `?v=<VERSION>` query string on the frontend URL only busts the browser
    cache after a ThermoSmart update; it does not affect static-path routing
    (aiohttp matches on path, not query string), so it is safe to combine
    with cache_headers=True on the static route itself.
    """
    www_dir = os.path.join(os.path.dirname(__file__), "www")
    card_path = os.path.join(www_dir, CARD_FILENAME)

    try:
        from homeassistant.components.http import StaticPathConfig
        await hass.http.async_register_static_paths([
            StaticPathConfig(CARD_URL_PATH, card_path, True)
        ])
    except Exception as err:  # http component missing/unavailable is non-fatal
        _LOGGER.warning("ThermoSmart: card static path registration skipped: %s", err)
        return

    try:
        from homeassistant.components.frontend import add_extra_js_url
        add_extra_js_url(hass, f"{CARD_URL_PATH}?v={VERSION}")
    except Exception as err:  # frontend component missing/unavailable is non-fatal
        _LOGGER.warning("ThermoSmart: card frontend registration skipped: %s", err)


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

    # Einmalig pro HA-Session, unabhängig von Entry-Typ und -Anzahl: die
    # gebündelte Lovelace-Card servieren und als Frontend-Modul registrieren.
    # Kein separates HACS-Plugin mehr nötig (Frenck-Review #9081/#9082).
    if "card_registered" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["card_registered"] = True
        await _async_register_card(hass)

    # ── System-Entry: nur globale Schalter, kein Coordinator ────────
    if cfg.get("entry_type") == "system":
        hass.data[DOMAIN][entry.entry_id] = {"type": "system"}
        await hass.config_entries.async_forward_entry_setups(entry, SYSTEM_PLATFORMS)

        if not hass.services.has_service(DOMAIN, "export_learning_data"):
            async def _handle_export(call: ServiceCall) -> None:
                filepath = await async_export_learning_data(hass)
                filename = os.path.basename(filepath)
                title, message = await async_build_export_notification(hass, filename)
                _pn_create(
                    hass, message=message, title=title,
                    notification_id="thermosmart_export",
                )

            hass.services.async_register(DOMAIN, "export_learning_data", _handle_export)
            _LOGGER.debug("ThermoSmart: export_learning_data service registered")

        # Einmalig pro HA-Session: Download-View registrieren und abgelaufene
        # Export-Dateien aufräumen (restart-sicher, siehe export.py-Docstring).
        if "export_view_registered" not in hass.data[DOMAIN]:
            hass.data[DOMAIN]["export_view_registered"] = True
            try:
                hass.http.register_view(ThermoSmartExportDownloadView(hass))
            except Exception as err:  # http component missing is non-fatal
                _LOGGER.warning("ThermoSmart: export download view registration skipped: %s", err)
            hass.async_create_task(async_cleanup_expired_exports(hass))

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
        # Phase 19A-B: Learning is the single active learning engine. The legacy engine
        # is frozen immediately after load — read-only, no further state mutation and no
        # store write (store is never deleted or migrated). Its loaded values may still
        # be READ by not-yet-transferred truths until each read is transferred to Learning.
        learning_engine.freeze()

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

    entry_store: dict = {
        "coordinator": coordinator,
        "weather_engine": weather_engine,
    }
    hass.data[DOMAIN][entry.entry_id] = entry_store

    # ── Learning learning runtime (supports adaptive control) ─────────
    # The runtime is started with CONTROL capability, but its EFFECTIVE
    # behavior each cycle is gated by the two user-visible switches (Learning,
    # Active Control) via ControlAdaptationMode.derive() in coordinator.py:
    # Active Control OFF -> shadow-only (observe/learn, no service calls);
    # Active Control ON + Learning OFF -> deterministic TPI-only control;
    # Active Control ON + Learning ON -> adaptive control.
    # Setup failure here must never fail the zone or affect heating.
    try:
        from .learning.runtime.ha_integration import LearningShadowController
        from .learning.runtime.lifecycle import LearningRuntimeMode

        learning_shadow = LearningShadowController(
            hass, entry.entry_id,
            clock=coordinator._clock,
            mode=LearningRuntimeMode.CONTROL,
        )
        await learning_shadow.async_setup()
        coordinator.attach_learning_shadow(learning_shadow)
        entry_store["learning_shadow"] = learning_shadow
        # NOTE: the flush/unload happens via the awaited call in async_unload_entry
        # (block-on-finish), never as a fire-and-forget task, so pending state is
        # guaranteed written by the time the unload completes.
        _LOGGER.debug(
            "ThermoSmart: Learning runtime attached; effective adaptation is "
            "gated by Learning and Active Control switches"
        )
    except Exception as err:  # learning is strictly optional; setup failure is non-fatal
        _LOGGER.warning("ThermoSmart: Learning Engine setup skipped: %s", type(err).__name__)

    await hass.config_entries.async_forward_entry_setups(entry, ZONE_PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    _LOGGER.info(
        "ThermoSmart Zone '%s' geladen – effektiver Adaptionsmodus wird pro Zyklus "
        "aus den Schaltern Lernen und Aktive Steuerung abgeleitet",
        cfg.get("name", entry.entry_id),
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    cfg = {**entry.data, **entry.options}
    platforms = SYSTEM_PLATFORMS if cfg.get("entry_type") == "system" else ZONE_PLATFORMS

    # Best-effort restore of temperature_sensor selects before entity teardown
    if cfg.get("entry_type") != "system":
        entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
        if coordinator := entry_data.get("coordinator"):
            await coordinator._async_restore_temp_source()
            # cancel any in-flight valve-maintenance task (guarded — must never
            # block unload, and a maintenance cycle must never outlive its entry)
            try:
                await coordinator.async_cancel_maintenance()
            except Exception:
                pass
        # flush + unload the Learning learning runtime (guarded)
        if learning_shadow := entry_data.get("learning_shadow"):
            try:
                await learning_shadow.async_unload()
            except Exception:  # never block unload on learning
                pass

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


async def _async_purge_learning_zone_storage(
    hass: HomeAssistant, entry_id: str, *, last_zone: bool
) -> None:
    """Delete all Learning storage for one removed zone.

    On the last remaining zone, also clears the shared legacy learning-engine
    store and the Learning global index — both are zone-independent, so they
    are only ever removed once no ThermoSmart zone remains at all. Only ever
    called from ``async_remove_entry`` (real removal) — never from
    ``async_unload_entry`` (reload/restart/unload must never lose data).
    Best-effort: a failure here is logged but never raised further up.
    """
    from homeassistant.helpers.storage import Store as _HAStore

    from .const import STORAGE_KEY, STORAGE_VERSION
    from .learning.reset import async_purge_zone_storage
    from .learning.runtime.ha_store import HomeAssistantStoreAdapter
    from .learning.storage.capture_registry import build_raw_track_registry
    from .learning.storage.stores import GlobalIndexStore, HomeAssistantStoreFactory

    factory = HomeAssistantStoreFactory(hass)
    result = await async_purge_zone_storage(
        factory, entry_id, raw_registry=build_raw_track_registry(),
    )
    if result.errors:
        _LOGGER.warning(
            "ThermoSmart: Lern-Storage-Bereinigung für Zone %s teilweise fehlgeschlagen: %s",
            entry_id, result.errors,
        )

    # Same hash the shadow controller uses to key its store (ha_integration.py's
    # _zone_segment(), which now delegates to this same consolidated helper).
    # naming.py is already pulled in transitively via storage.stores above —
    # importing it directly here still avoids any dependency on the heavier
    # live runtime module (ha_integration.py), matching the original intent.
    from .learning.storage.naming import zone_segment as _zone_segment_fn
    zone_segment = _zone_segment_fn(entry_id)
    await HomeAssistantStoreAdapter(hass, zone_segment).async_delete()

    if last_zone:
        await _HAStore(hass, STORAGE_VERSION, STORAGE_KEY).async_remove()
        await GlobalIndexStore(factory).delete()


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Lerndaten der entfernten Zone sofort bereinigen.

    Wird von HA aufgerufen NACHDEM die Zone vollständig aus config_entries entfernt
    wurde – zu diesem Zeitpunkt ist die zone_id bereits aus dem Entry-Register verschwunden,
    sodass prune_orphaned_zones sie korrekt als verwaist erkennt.

    Only ever called on REAL removal — never on unload/reload/restart, which
    must never delete learning/support/research storage.
    """
    if entry.data.get("entry_type") == "system":
        return

    active_zone_ids = {
        e.entry_id
        for e in hass.config_entries.async_entries(DOMAIN)
        if e.data.get("entry_type") != "system"
    }

    if le := hass.data.get(DOMAIN, {}).get("learning_engine"):
        le.prune_orphaned_zones(active_zone_ids)

    try:
        await _async_purge_learning_zone_storage(
            hass, entry.entry_id, last_zone=not active_zone_ids
        )
    except Exception as err:
        _LOGGER.warning(
            "ThermoSmart: Zone '%s' Lern-Storage-Bereinigung fehlgeschlagen: %s",
            entry.data.get("name", entry.entry_id), type(err).__name__,
        )

    _LOGGER.info(
        "ThermoSmart: Zone '%s' entfernt – Lerndaten sofort bereinigt",
        entry.data.get("name", entry.entry_id),
    )


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    if {**entry.data, **entry.options}.get("entry_type") != "system":
        _LOGGER.info(
            "ThermoSmart '%s': Options updated — reloading entry",
            entry.data.get("name", entry.entry_id),
        )
        try:
            await hass.config_entries.async_reload(entry.entry_id)
        except Exception as err:
            _LOGGER.error(
                "ThermoSmart '%s': Reload after options update failed: %s",
                entry.data.get("name", entry.entry_id), err,
            )
