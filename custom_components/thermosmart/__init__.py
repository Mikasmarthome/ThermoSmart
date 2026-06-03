"""ThermoSmart – AI-powered, weather-aware heating control for Home Assistant.

Architektur: Ein Config-Eintrag = Eine Heizzone.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

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
    CONF_PRESENCE_PERSONS,
    CONF_HOME_ZONE,
    CONF_VACATION_BOOLEAN,
    CONF_CALIBRATION_ENTITIES,
    CONF_QUIRK_ENTITIES,
    CONF_VALVE_MAINTENANCE,
    AUTO_QUIRK_PATTERNS,
    AUTO_CALIBRATION_PATTERN,
    CONF_VACATION_TEMP,
    CONF_BOOST_TEMP,
    CONF_ECO_TEMP,
    CONF_NO_OFF_FALLBACK,
    NOISE_FILTER_SPIKE_THRESHOLD,
    NOISE_FILTER_EMA_ALPHA,
    TEMP_NIGHT,
    TEMP_FROST_PROTECTION,
    TEMP_BOOST,
    TEMP_ECO,
    FROST_FALLBACK_TEMP,
    SUMMER_THRESHOLD,
    WINTER_THRESHOLD,
    SEASON_HOURS,
    VALVE_MAINTENANCE_HOUR,
    VALVE_MAINTENANCE_WEEKDAY,
    VALVE_MAINTENANCE_BOOST_TEMP,
    VALVE_MAINTENANCE_DURATION_SEC,
    RESIDUAL_HEAT_ZONE,
    EMA_1H_ALPHA,
    HEATING_MODE_AUTO,
    HEATING_MODE_AWAY,
    HEATING_MODE_VACATION,
    HEATING_MODE_COMFORT,
    HEATING_MODE_NIGHT,
    HEATING_MODE_BOOST,
    HEATING_MODE_ECO,
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
    await coordinator.async_detect_device_entities()
    coordinator.setup_event_listeners()
    entry.async_on_unload(coordinator.cleanup_event_listeners)

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
        self._override_schedule_period: str | None = None  # Zeitplan-Slot bei Override-Setzen
        self._event_unsub: list = []
        self._window_open_at: dict[str, datetime] = {}
        self._window_close_at: dict[str, datetime] = {}
        self._boost_active: dict[str, dict] = {}
        self._calibration_offsets: dict[str, float] = {}
        self._outdoor_temp_history: deque = deque(maxlen=int(SEASON_HOURS * 3600 / DEFAULT_SCAN_INTERVAL))
        self._is_summer: bool = False
        self._last_maintenance: datetime | None = None
        self._maintenance_running: bool = False
        self._trv_offline: set[str] = set()
        self._auto_quirk_entities: list[str] = []
        self._auto_calibration_map: dict[str, str] = {}   # climate_id → cal_entity_id
        self._sensor_ema: dict[str, float] = {}
        self._sensor_noise_count: dict[str, int] = {}
        self._indoor_temp_prev: tuple[datetime, float] | None = None
        self._indoor_temp_slope: float = 0.0
        self._ema_1h: float | None = None
        self._last_written_setpoints: dict[str, float] = {}

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
        # Zeitplan-Slot beim Setzen merken → wird beim nächsten Übergang zurückgesetzt
        self._override_schedule_period = self._current_schedule_period() if self._override else None

    def _current_schedule_period(self) -> str:
        """Gibt einen String zurück der den aktuellen Zeitplan-Slot eindeutig identifiziert."""
        now = dt_util.now()
        cfg = self.zone_cfg
        is_weekend = now.weekday() >= 5
        cur = now.hour * 60 + now.minute

        def t(key: str, fallback: str) -> int:
            raw = cfg.get(key, fallback)
            try:
                h, m = str(raw).split(":")
                return int(h) * 60 + int(m)
            except (ValueError, AttributeError):
                h2, m2 = fallback.split(":")
                return int(h2) * 60 + int(m2)

        if is_weekend:
            morning = t("sched_we_morning", "08:00")
            night   = t("sched_we_night",   "23:00")
        else:
            morning = t("sched_wd_morning", "06:00")
            night   = t("sched_wd_night",   "22:00")

        day_type = "we" if is_weekend else "wd"
        if cur < morning or cur >= night:
            return f"{day_type}_night"
        return f"{day_type}_comfort"

    def get_override(self) -> float | None:
        return self._override

    # ── Event-Listener ───────────────────────────────────────────────

    def setup_event_listeners(self) -> None:
        """Sofort-Reaktion auf Fenster-, Personen-, Urlaubs- und TRV-Änderungen."""
        cfg = self.zone_cfg
        window_sensors: set[str] = set(s for s in cfg.get("window_sensors", []) if s)
        presence: set[str] = set(p for p in cfg.get(CONF_PRESENCE_PERSONS, []) if p)
        vacation_entity = cfg.get(CONF_VACATION_BOOLEAN, "")
        if vacation_entity:
            presence.add(vacation_entity)
        climate_entities: set[str] = set(e for e in cfg.get("climate_entities", []) if e)

        # Fenster + Personen + Urlaub → Sofort-Refresh
        all_tracked = window_sensors | presence
        if all_tracked:
            @callback
            def _handle_state_change(event) -> None:
                old = event.data.get("old_state")
                new = event.data.get("new_state")
                if old is None or new is None or old.state == new.state:
                    return
                entity_id = event.data["entity_id"]
                now = dt_util.now()

                if entity_id in window_sensors:
                    if new.state == "on":
                        self._window_open_at[entity_id] = now
                        self._window_close_at.pop(entity_id, None)
                    else:
                        if entity_id in self._window_open_at:
                            self._window_close_at[entity_id] = now
                        self._window_open_at.pop(entity_id, None)

                _LOGGER.info(
                    "ThermoSmart '%s': %s geändert (%s → %s) – Sofort-Update",
                    self.zone_name, entity_id, old.state, new.state,
                )
                self.hass.async_create_task(self.async_request_refresh())

            cancel = async_track_state_change_event(
                self.hass, list(all_tracked), _handle_state_change
            )
            self._event_unsub.append(cancel)

        # TRV-Entities → manuelle Setpoint-Änderungen sofort korrigieren
        if climate_entities and self._active_control:
            @callback
            def _handle_trv_change(event) -> None:
                if not self._active_control:
                    return
                new = event.data.get("new_state")
                if new is None:
                    return
                entity_id = event.data["entity_id"]
                new_setpoint = new.attributes.get("temperature")
                last_written = self._last_written_setpoints.get(entity_id)
                if new_setpoint is None or last_written is None:
                    return
                try:
                    if abs(float(new_setpoint) - last_written) > 0.4:
                        _LOGGER.info(
                            "ThermoSmart '%s': Manuelle TRV-Änderung erkannt %s "
                            "(%.1f°C statt %.1f°C) – Sofort-Korrektur",
                            self.zone_name, entity_id, float(new_setpoint), last_written,
                        )
                        self.hass.async_create_task(self.async_request_refresh())
                except (TypeError, ValueError):
                    pass

            cancel_trv = async_track_state_change_event(
                self.hass, list(climate_entities), _handle_trv_change
            )
            self._event_unsub.append(cancel_trv)

        _LOGGER.debug(
            "ThermoSmart '%s': Event-Listener registriert "
            "(%d Presenz/Fenster, %d TRVs)",
            self.zone_name, len(all_tracked), len(climate_entities),
        )

    def cleanup_event_listeners(self) -> None:
        for cancel in self._event_unsub:
            cancel()
        self._event_unsub.clear()

    # ── Quirk-Autodetect ─────────────────────────────────────────────

    async def async_detect_device_entities(self) -> None:
        """Erkennt Quirk-Switches und Kalibrierungs-Entities via Device Registry.

        Ein Durchlauf über alle Entities der TRV-Geräte:
          - switch.*  → Quirk-Muster (AUTO_QUIRK_PATTERNS)
          - number.*  → local_temperature_calibration
        Ergebnis wird in _auto_quirk_entities und _auto_calibration_map gespeichert.
        """
        cfg = self.zone_cfg
        climate_entities = cfg.get("climate_entities", [])
        if not climate_entities:
            return

        ent_reg = er.async_get(self.hass)

        # device_id → climate_entity_id (für Kalibrierungs-Mapping)
        device_to_climate: dict[str, str] = {}
        for entity_id in climate_entities:
            entry = ent_reg.async_get(entity_id)
            if entry and entry.device_id:
                device_to_climate[entry.device_id] = entity_id

        if not device_to_climate:
            return

        quirks: list[str] = []
        cal_map: dict[str, str] = {}

        for device_id, climate_id in device_to_climate.items():
            for entry in er.async_entries_for_device(ent_reg, device_id):
                domain = entry.entity_id.split(".")[0]

                if domain == "switch":
                    for pattern in AUTO_QUIRK_PATTERNS:
                        if pattern in entry.entity_id:
                            quirks.append(entry.entity_id)
                            break

                elif domain == "number":
                    if AUTO_CALIBRATION_PATTERN in entry.entity_id:
                        cal_map[climate_id] = entry.entity_id

        manual_quirks = set(cfg.get(CONF_QUIRK_ENTITIES, []))
        new_quirks = [e for e in quirks if e not in manual_quirks]
        if new_quirks:
            _LOGGER.info("ThermoSmart '%s': Quirk-Autodetect: %s", self.zone_name, new_quirks)
        if cal_map:
            _LOGGER.info("ThermoSmart '%s': Kalibrierungs-Autodetect: %s", self.zone_name, cal_map)

        self._auto_quirk_entities = quirks
        self._auto_calibration_map = cal_map

    # ── TRV-Offline-Tracking ─────────────────────────────────────────

    def _get_trv_state(self, entity_id: str):
        """Gibt TRV-State zurück; loggt Offline/Online-Übergänge."""
        state = self.hass.states.get(entity_id)
        if state is None or state.state in ("unavailable", "unknown"):
            if entity_id not in self._trv_offline:
                self._trv_offline.add(entity_id)
                _LOGGER.warning(
                    "ThermoSmart '%s': TRV %s nicht erreichbar – Steuerung ausgesetzt",
                    self.zone_name, entity_id,
                )
            return None
        if entity_id in self._trv_offline:
            self._trv_offline.discard(entity_id)
            _LOGGER.info(
                "ThermoSmart '%s': TRV %s wieder erreichbar",
                self.zone_name, entity_id,
            )
        return state

    # ── Hauptschleife ────────────────────────────────────────────────

    async def _async_update_data(self) -> dict:
        try:
            cfg = self.zone_cfg
            self._check_boost_outcome(cfg)

            weather_data = await self.weather_engine.async_get_data()
            self._update_summer_mode(weather_data)

            presence = self._get_presence_state()
            mode = self._effective_mode(presence)
            recommendation = await self._compute_recommendation(cfg, weather_data, mode)
            recommendation["is_summer"] = self._is_summer

            # TRV-Boost-Setpoint berechnen und zur Recommendation hinzufügen
            target = recommendation.get("adjusted_target")
            if target is not None:
                boost_factor = self.learning_engine.get_boost_factor(self.zone_id)
                trv_setpoint = self._compute_trv_setpoint(
                    target,
                    recommendation.get("current_temp"),
                    weather_data,
                    boost_factor,
                )
                recommendation["trv_setpoint"] = trv_setpoint
                recommendation["boost_factor"] = round(boost_factor, 3)

            indoor_humidity = self._read_avg_sensor(cfg.get("humidity_sensors", []))
            await self.learning_engine.async_observe(
                zone_id=self.zone_id,
                recommendation=recommendation,
                weather_data=weather_data,
                indoor_humidity=indoor_humidity,
            )

            if self._active_control and not self._is_summer:
                await self._async_apply_quirks(cfg)
                await self._watchdog_hvac(cfg, recommendation)
                await self._async_calibrate_trvs(cfg, recommendation)
                await self._apply_temperature(cfg, recommendation)
                await self._async_valve_maintenance(cfg, recommendation)
            elif self._active_control and self._is_summer:
                await self._async_apply_quirks(cfg)
                await self._apply_frost_protection(cfg)

            return {
                "weather": weather_data,
                "zone": recommendation,
                "active_control": self._active_control,
                "presence": presence,
            }
        except Exception as err:
            raise UpdateFailed(f"ThermoSmart '{self.zone_name}': {err}") from err

    # ── Sommer-Erkennung ─────────────────────────────────────────────

    def _update_summer_mode(self, weather_data: dict) -> None:
        """72h-Rollmittelwert der Außentemperatur → automatische Sommer/Winter-Erkennung.

        Verhindert Fehlauslösung durch einzelne Ausreißer-Tage:
          - Erst nach SEASON_HOURS Stunden konsistent >SUMMER_THRESHOLD → Sommer
          - Erst nach SEASON_HOURS Stunden konsistent <WINTER_THRESHOLD → Winter
        """
        outdoor = weather_data.get("temperature")
        if outdoor is None:
            return
        self._outdoor_temp_history.append(outdoor)

        if len(self._outdoor_temp_history) < 3:
            return

        avg = sum(self._outdoor_temp_history) / len(self._outdoor_temp_history)

        prev_summer = self._is_summer
        if avg >= SUMMER_THRESHOLD:
            self._is_summer = True
        elif avg <= WINTER_THRESHOLD:
            self._is_summer = False

        if self._is_summer != prev_summer:
            _LOGGER.warning(
                "ThermoSmart '%s': Saisonwechsel – %s (72h-Ø %.1f°C)",
                self.zone_name,
                "SOMMER – Heizung deaktiviert" if self._is_summer else "WINTER – Heizung aktiv",
                avg,
            )

    async def _apply_frost_protection(self, cfg: dict) -> None:
        """Im Sommer: TRVs auf Frostschutztemperatur setzen (12°C)."""
        tasks = []
        for entity_id in cfg.get("climate_entities", []):
            state = self.hass.states.get(entity_id)
            if not state or state.state in ("unavailable", "unknown"):
                continue
            try:
                if abs(float(state.attributes.get("temperature", 0)) - TEMP_FROST_PROTECTION) < 0.5:
                    continue
            except (TypeError, ValueError):
                pass
            tasks.append(self.hass.services.async_call(
                "climate", "set_temperature",
                {"entity_id": entity_id, "temperature": TEMP_FROST_PROTECTION},
                blocking=True,
            ))
        if tasks:
            await asyncio.gather(*tasks)

    # ── Präsenz ──────────────────────────────────────────────────────

    def _is_person_home(self, person_state: str, zone_entity: str) -> bool:
        """Prüft ob eine Person in der konfigurierten Zone ist.

        HA setzt person.state auf:
          - "home"           wenn in der Standard-Heimzone (zone.home)
          - "<zone_name>"    wenn in einer anderen benannten Zone
          - "not_home"       wenn nirgends
        """
        if not zone_entity or zone_entity in ("zone.home", "home"):
            return person_state == "home"

        # Entity-ID-Suffix vergleichen (z.B. "zone.heizungszone" → "heizungszone")
        zone_slug = zone_entity.replace("zone.", "").lower()
        if person_state.lower() == zone_slug:
            return True

        # Friendly-Name der Zone vergleichen (z.B. "Heizungszone")
        zone_state = self.hass.states.get(zone_entity)
        if zone_state:
            friendly = zone_state.attributes.get("friendly_name", "")
            if person_state.lower() == friendly.lower():
                return True

        return False

    def _get_presence_state(self) -> dict:
        cfg = self.zone_cfg
        home_zone = cfg.get(CONF_HOME_ZONE, "zone.home") or "zone.home"
        persons_home, persons_away = [], []

        for person in cfg.get(CONF_PRESENCE_PERSONS, []):
            state = self.hass.states.get(person)
            if state and self._is_person_home(state.state, home_zone):
                persons_home.append(person)
            else:
                persons_away.append(person)

        vacation = False
        vacation_entity = cfg.get(CONF_VACATION_BOOLEAN, "")
        if vacation_entity:
            vs = self.hass.states.get(vacation_entity)
            if vs and vs.state == "on":
                vacation = True

        return {
            "persons_home": persons_home,
            "all_away": len(persons_home) == 0 and bool(cfg.get(CONF_PRESENCE_PERSONS)),
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

        # Temperatur-Slope (K/min) und EMA 1h aus aufeinanderfolgenden Messungen
        now = dt_util.now()
        if current_temp is not None:
            if self._indoor_temp_prev is not None:
                prev_time, prev_temp = self._indoor_temp_prev
                elapsed_min = (now - prev_time).total_seconds() / 60
                if 1.0 <= elapsed_min <= 30.0:
                    self._indoor_temp_slope = round((current_temp - prev_temp) / elapsed_min, 4)
            self._indoor_temp_prev = (now, current_temp)
            if self._ema_1h is None:
                self._ema_1h = current_temp
            else:
                self._ema_1h = round(
                    EMA_1H_ALPHA * current_temp + (1 - EMA_1H_ALPHA) * self._ema_1h, 2
                )

        # Fenster offen?
        window_open = self._check_window_open(cfg)

        # Basis-Zieltemperatur
        base_target = await self.learning_engine.async_get_base_target(
            self.zone_id, mode,
            comfort_temp=cfg.get("comfort_temp", 21.0),
            night_temp=cfg.get("night_temp", 18.0),
            away_temp=cfg.get("away_temp", 17.0),
            vacation_temp=cfg.get(CONF_VACATION_TEMP, 12.0),
            boost_temp=cfg.get(CONF_BOOST_TEMP, TEMP_BOOST),
            eco_temp=cfg.get(CONF_ECO_TEMP, TEMP_ECO),
            schedule_cfg=cfg,
        )

        # Wetterkorrektur (nur Auto-Modus)
        weather_offset = 0.0
        if mode == HEATING_MODE_AUTO:
            weather_offset = self.weather_engine.compute_temperature_offset(weather_data)

        # Vorheizzeit
        preheat_minutes = await self.learning_engine.async_get_preheat_minutes(
            self.zone_id, base_target, current_temp, weather_data
        )

        # Override auto-reset: wenn Zeitplan-Slot gewechselt hat → zurück auf Automatik
        if self._override is not None and self._override_schedule_period is not None:
            current_period = self._current_schedule_period()
            if current_period != self._override_schedule_period:
                _LOGGER.info(
                    "ThermoSmart '%s': Manueller Override abgelaufen (Slot %s → %s) – zurück auf Automatik",
                    self.zone_name, self._override_schedule_period, current_period,
                )
                self._override = None
                self._override_schedule_period = None

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
            "temp_slope": self._indoor_temp_slope,
            "temp_ema_1h": self._ema_1h,
        }

    def _check_window_open(self, cfg: dict) -> bool:
        now = dt_util.now()
        open_delay = timedelta(minutes=cfg.get("window_open_delay", 5))
        close_delay = timedelta(minutes=cfg.get("window_close_delay", 2))

        for ws_id in cfg.get("window_sensors", []):
            if not ws_id:
                continue
            ws = self.hass.states.get(ws_id)
            if ws is None:
                continue

            if ws.state == "on":
                if ws_id not in self._window_open_at:
                    # Beim Start bereits offen → sofort effektiv (Delay bereits abgelaufen)
                    self._window_open_at[ws_id] = now - open_delay
                if now - self._window_open_at[ws_id] >= open_delay:
                    return True
            else:
                # Fenster zu – ggf. noch in Schließ-Toleranzzeit
                if ws_id in self._window_close_at:
                    if now - self._window_close_at[ws_id] < close_delay:
                        return True
                    self._window_close_at.pop(ws_id, None)
                self._window_open_at.pop(ws_id, None)

        return False

    def _read_avg_sensor(self, sensor_ids: list[str]) -> float | None:
        values = []
        for sid in sensor_ids:
            if not sid:
                continue
            state = self.hass.states.get(sid)
            if state and state.state not in ("unknown", "unavailable", "None"):
                try:
                    filtered = self._filter_sensor_value(sid, float(state.state))
                    if filtered is not None:
                        values.append(filtered)
                except ValueError:
                    pass
        return round(sum(values) / len(values), 1) if values else None

    def _filter_sensor_value(self, sensor_id: str, raw: float) -> float | None:
        """EMA-Glättung mit Spike-Erkennung für Temperatursensoren.

        Kurze Ausreißer (defekter Sensor, Zigbee-Übertragungsfehler) werden
        ignoriert. Nachhaltige Temperaturänderungen folgen dem EMA graduell.
        """
        if sensor_id not in self._sensor_ema:
            self._sensor_ema[sensor_id] = raw
            return raw

        ema = self._sensor_ema[sensor_id]
        if abs(raw - ema) > NOISE_FILTER_SPIKE_THRESHOLD:
            count = self._sensor_noise_count.get(sensor_id, 0) + 1
            self._sensor_noise_count[sensor_id] = count
            if count == 1:
                _LOGGER.debug(
                    "ThermoSmart '%s': Sensor %s Spike ignoriert %.1f°C (EMA=%.1f°C)",
                    self.zone_name, sensor_id, raw, ema,
                )
            return None

        self._sensor_noise_count[sensor_id] = 0
        self._sensor_ema[sensor_id] = NOISE_FILTER_EMA_ALPHA * raw + (1 - NOISE_FILTER_EMA_ALPHA) * ema
        return round(self._sensor_ema[sensor_id], 1)

    def _compute_trv_setpoint(
        self,
        target: float,
        current_temp: float | None,
        weather_data: dict,
        boost_factor: float,
    ) -> float:
        """Multi-Faktor Boost-Setpoint für TRV.

        Idee (angelehnt an Better Thermostat):
          setpoint = current + (max_temp - current) × valve_fraction
        valve_fraction wird aus Außenbedingungen + delta berechnet –
        nicht nur Außentemp, sondern auch Wind und Feuchte.

        Faktoren:
          • delta (Ziel – Ist)             → primärer Treiber
          • Außentemperatur                → kälter = mehr Boost nötig
          • Windgeschwindigkeit            → Wind erhöht Wärmeverlust
          • Außenluftfeuchtigkeit          → feuchte Kälte kühlt stärker
          • boost_factor (gelernt)         → passt sich nach Überschießen an
        """
        if current_temp is None:
            return target

        delta = target - current_temp

        # Restwärme-Kompensation: nur relevant wenn Raum sich von unten nähert.
        # Bei delta < -RESIDUAL_HEAT_ZONE war der Heizkörper lange aus → kein Setpoint-Abzug.
        if delta <= -RESIDUAL_HEAT_ZONE:
            # Raum deutlich wärmer als Ziel → Ventil trotzdem zu, Setpoint = Target
            return target
        if delta <= 0:
            # Raum knapp über Ziel → Restwärme vorhanden, Setpoint leicht unter Ziel
            fraction = (delta + RESIDUAL_HEAT_ZONE) / RESIDUAL_HEAT_ZONE
            reduction = (1.0 - fraction) * (RESIDUAL_HEAT_ZONE * 0.5)
            return round(max(target - reduction, 5.0), 1)
        if delta < RESIDUAL_HEAT_ZONE:
            # Annäherungszone: Setpoint linear von target bis (target - zone/2) reduzieren
            fraction = delta / RESIDUAL_HEAT_ZONE
            reduction = (1.0 - fraction) * (RESIDUAL_HEAT_ZONE * 0.5)
            return round(target - reduction, 1)

        outdoor = weather_data.get("temperature") or 15.0
        wind = weather_data.get("wind_speed") or 0.0
        humidity_out = weather_data.get("humidity") or 60.0

        # Kälte-Faktor: min. 0.1 (auch im Sommer kleine Nachtabsenkungen boosten)
        cold_factor = max(0.1, min(1.0, (18.0 - outdoor) / 18.0))

        # Wind erhöht Wärmeverlust durch Wände → bis +30% mehr Boost
        wind_factor = 1.0 + min(wind / 20.0, 0.3)

        # Feuchte Kälte kühlt gefühlt stärker → bis +10% bei sehr feuchter Luft
        humidity_factor = 1.0 + max(0.0, (humidity_out - 70.0) / 300.0)

        combined = cold_factor * wind_factor * humidity_factor * boost_factor

        # Valve-Fraction-Ansatz (wie Better Thermostat):
        # Anteil des Wegs von current_temp bis max_setpoint den wir öffnen
        max_setpoint = 28.0
        valve_fraction = min(delta * 0.7 * combined / (max_setpoint - current_temp), 1.0)
        trv_setpoint = current_temp + (max_setpoint - current_temp) * valve_fraction

        # Nie mehr als target + 3°C und nie über 28°C
        trv_setpoint = min(trv_setpoint, target + 3.0, 28.0)

        # Weiche Annäherung: Boost blendet zwischen delta=1 und delta=2 ein
        if delta < 2.0:
            blend = delta - 1.0
            trv_setpoint = target + (trv_setpoint - target) * blend

        return round(trv_setpoint, 1)

    def _check_boost_outcome(self, cfg: dict) -> None:
        """Erkennt Überschießen und zu langsames Heizen → passt Boost-Faktor an."""
        if self.zone_id not in self._boost_active:
            return
        current = self._read_avg_sensor(cfg.get("temp_sensors", []))
        if current is None:
            return
        entry = self._boost_active[self.zone_id]
        prev_target = entry["target"]
        tolerance = cfg.get("temp_tolerance", 0.5)
        if current > prev_target + tolerance:
            self.learning_engine.update_boost_factor(self.zone_id, overshot=True)
            _LOGGER.info(
                "ThermoSmart '%s': Boost-Überschießen %.1f°C > %.1f°C – Faktor reduziert",
                self.zone_name, current, prev_target,
            )
            self._boost_active.pop(self.zone_id)
        elif current >= prev_target - tolerance * 0.5:
            # Ziel sauber erreicht – kein Überschießen
            self._boost_active.pop(self.zone_id)
        else:
            # Zu langsames Heizen: nach 30 min noch deutlich unter Ziel → Faktor erhöhen
            started = entry.get("started")
            if started and (dt_util.now() - started).total_seconds() > 1800:
                if current < prev_target - 1.0:
                    self.learning_engine.update_boost_factor(self.zone_id, overshot=False, slow=True)
                    _LOGGER.info(
                        "ThermoSmart '%s': Langsames Heizen nach 30 min %.1f°C < %.1f°C – Faktor erhöht",
                        self.zone_name, current, prev_target,
                    )
                self._boost_active.pop(self.zone_id)

    # ── Thermostat schreiben ─────────────────────────────────────────

    async def _watchdog_hvac(self, cfg: dict, recommendation: dict) -> None:
        """TRV Watchdog: stellt Thermostate die auf 'off' gefallen sind wieder her.

        Ersetzt die manuellen BT-Watchdog Automationen.
        Läuft nur wenn:
          - Aktive Steuerung AN
          - Kein Fenster offen (dann ist 'off' gewollt)
          - Nicht 100% Prognose-Unterdrückung (Sommer → Heizung aus ist korrekt)
        """
        if recommendation.get("window_open"):
            return
        # Im Sommer (volle Unterdrückung) ist 'off' gewollt → kein Watchdog
        if recommendation.get("forecast_suppression", 0) >= 100:
            return

        for entity_id in cfg.get("climate_entities", []):
            state = self._get_trv_state(entity_id)
            if state is None:
                continue
            if state.state == "off":
                _LOGGER.info(
                    "ThermoSmart Watchdog '%s': %s ist 'off' → stelle auf 'heat' zurück",
                    self.zone_name, entity_id,
                )
                await self.hass.services.async_call(
                    "climate", "set_hvac_mode",
                    {"entity_id": entity_id, "hvac_mode": "heat"},
                    blocking=False,
                )

    async def _async_calibrate_trvs(self, cfg: dict, recommendation: dict) -> None:
        """Lokale TRV-Kalibrierung: Offset zwischen Raumsensor und TRV-Sensor glätten und schreiben.

        Logik (angelehnt an Better Thermostat Type 3):
          raw_offset = room_temp - trv_eigene_temp
          smoothed   = EMA(raw_offset, α=0.25)   → langsame Anpassung, kein Jitter
          → schreiben wenn |smoothed - aktuelle_kalibrierung| > 0.5°C

        Nicht kalibrieren wenn TRV deutlich wärmer als Raum ist (Heizkörper heizt Sensor).
        Kalibrierungs-Entity wird automatisch via Device Registry erkannt (Vorrang)
        oder aus der manuellen Fallback-Liste gelesen (rückwärtskompatibel).
        """
        room_temp = recommendation.get("current_temp")
        if room_temp is None:
            return

        climate_entities = cfg.get("climate_entities", [])
        manual_cal = cfg.get(CONF_CALIBRATION_ENTITIES, [])

        # Kein Auto-Mapping und keine manuelle Liste → nichts zu tun
        if not self._auto_calibration_map and not manual_cal:
            return

        for i, climate_id in enumerate(climate_entities):
            # Auto-erkannte Kalibrierungs-Entity hat Vorrang
            cal_entity = self._auto_calibration_map.get(climate_id)
            if not cal_entity:
                cal_entity = manual_cal[i] if i < len(manual_cal) else None
            if not cal_entity:
                continue

            # TRV-eigene Temperatur direkt aus der Climate-Entity lesen
            trv_temp: float | None = None
            trv_state = self._get_trv_state(climate_id)
            if trv_state:
                try:
                    trv_temp = float(trv_state.attributes.get("current_temperature", 0))
                except (TypeError, ValueError):
                    pass

            if trv_temp is None:
                continue

            # Heizkörper-Einfluss: TRV ist >3°C wärmer als Raum → Sensor durch Heizkörper verfälscht
            if trv_temp - room_temp > 3.0:
                _LOGGER.debug(
                    "ThermoSmart '%s': Kalibrierung %s übersprungen – TRV %.1f°C > Raum %.1f°C (Heizkörperwärme)",
                    self.zone_name, cal_entity, trv_temp, room_temp,
                )
                continue

            raw_offset = room_temp - trv_temp

            # Plausibilitätsprüfung: Offset > ±7°C ist verdächtig
            if abs(raw_offset) > 7.0:
                _LOGGER.warning(
                    "ThermoSmart '%s': Kalibrierungs-Offset %.1f°C für %s unplausibel – übersprungen",
                    self.zone_name, raw_offset, cal_entity,
                )
                continue

            # EMA-Glättung (α=0.25 → langsam, stabil)
            prev = self._calibration_offsets.get(cal_entity, raw_offset)
            smoothed = round(0.25 * raw_offset + 0.75 * prev, 1)
            self._calibration_offsets[cal_entity] = smoothed

            # Aktuellen Wert der Kalibrierungs-Entity lesen
            cal_state = self.hass.states.get(cal_entity)
            if cal_state is None:
                continue
            try:
                current_cal = float(cal_state.state)
            except (TypeError, ValueError):
                current_cal = 0.0

            # Nur schreiben wenn Änderung > 0.5°C (Zigbee-Traffic sparen)
            if abs(smoothed - current_cal) < 0.5:
                continue

            # TRV-Limit: typisch ±5°C
            clamped = max(-5.0, min(5.0, smoothed))

            _LOGGER.info(
                "ThermoSmart '%s': TRV-Kalibrierung %s → %.1f°C (Raum=%.1f°C, TRV-Sensor=%.1f°C)",
                self.zone_name, cal_entity, clamped, room_temp, trv_temp,
            )
            # Kalibrierungsaufrufe werden unten gesammelt und parallel gesendet
            # (return-Wert nicht benötigt, fire-and-forget via gather)
            self.hass.async_create_task(self.hass.services.async_call(
                "number", "set_value",
                {"entity_id": cal_entity, "value": clamped},
                blocking=False,
            ))

    async def _async_apply_quirks(self, cfg: dict) -> None:
        """TRV-Eigenlogik deaktivieren die mit ThermoSmart konkurriert.

        Typisch beim Sonoff TRVZB via Zigbee2MQTT:
          - switch.*_window_detection  → TRV erkennt Fenster selbst (Konflikt mit unseren Sensoren)
          - switch.*_child_lock        → sperrt externe Setpoint-Änderungen
          - Jeder switch in quirk_entities wird auf OFF gehalten.
        """
        manual = cfg.get(CONF_QUIRK_ENTITIES, [])
        # Auto-erkannte und manuell konfigurierte Quirks zusammenführen (dedupliziert)
        quirk_entities = list(dict.fromkeys(manual + self._auto_quirk_entities))
        if not quirk_entities:
            return
        tasks = []
        for entity_id in quirk_entities:
            state = self.hass.states.get(entity_id)
            if state is None or state.state in ("unavailable", "unknown"):
                continue
            if state.state == "on":
                _LOGGER.info(
                    "ThermoSmart '%s': Quirk deaktivieren %s (war 'on')",
                    self.zone_name, entity_id,
                )
                tasks.append(self.hass.services.async_call(
                    "switch", "turn_off", {"entity_id": entity_id}, blocking=True,
                ))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    _LOGGER.debug("ThermoSmart '%s': Quirk-Deaktivierung fehlgeschlagen: %s", self.zone_name, result)

    async def _async_valve_maintenance(self, cfg: dict, recommendation: dict) -> None:
        """Ventil-Wartung: einmal pro Woche Ventil vollständig auf und zu fahren.

        Verhindert Festklemmen nach dem Sommer (Kalk, Gummi klebt).
        Ablauf:
          1. Sonntag 03:00 Uhr
          2. Alle TRVs auf VALVE_MAINTENANCE_BOOST_TEMP (28°C) → Ventil fährt voll auf
          3. VALVE_MAINTENANCE_DURATION_SEC warten (async, blockiert nicht HA)
          4. Zurück auf adjustierten Zielwert
        """
        if not cfg.get(CONF_VALVE_MAINTENANCE, True):
            return
        if self._maintenance_running:
            return

        now = dt_util.now()
        if now.weekday() != VALVE_MAINTENANCE_WEEKDAY or now.hour != VALVE_MAINTENANCE_HOUR:
            return

        # Nur einmal pro Woche auslösen (nicht jede 5-Min-Runde)
        if self._last_maintenance is not None:
            if (now - self._last_maintenance).days < 6:
                return

        self._last_maintenance = now
        self._maintenance_running = True
        target_after = recommendation.get("adjusted_target") or cfg.get("comfort_temp", 21.0)
        climate_entities = cfg.get("climate_entities", [])

        _LOGGER.info(
            "ThermoSmart '%s': Ventil-Wartung gestartet – fahre auf %.0f°C",
            self.zone_name, VALVE_MAINTENANCE_BOOST_TEMP,
        )

        async def _run_maintenance() -> None:
            try:
                # Vollständig öffnen
                tasks = [
                    self.hass.services.async_call(
                        "climate", "set_temperature",
                        {"entity_id": eid, "temperature": VALVE_MAINTENANCE_BOOST_TEMP},
                        blocking=True,
                    )
                    for eid in climate_entities
                    if self.hass.states.get(eid) and
                    self.hass.states.get(eid).state not in ("unavailable", "unknown")
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Warten
                await asyncio.sleep(VALVE_MAINTENANCE_DURATION_SEC)

                # Zurück auf Zieltemperatur
                tasks = [
                    self.hass.services.async_call(
                        "climate", "set_temperature",
                        {"entity_id": eid, "temperature": target_after},
                        blocking=True,
                    )
                    for eid in climate_entities
                    if self.hass.states.get(eid) and
                    self.hass.states.get(eid).state not in ("unavailable", "unknown")
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                _LOGGER.info(
                    "ThermoSmart '%s': Ventil-Wartung abgeschlossen → zurück auf %.1f°C",
                    self.zone_name, target_after,
                )
            finally:
                self._maintenance_running = False

        # Als Hintergrund-Task starten – blockiert nicht den normalen Update-Zyklus
        self.hass.async_create_task(_run_maintenance())

    async def _apply_temperature(self, cfg: dict, recommendation: dict) -> None:
        target = recommendation.get("adjusted_target")

        # Fenster offen → Frostschutz-Fallback (statt TRV auf letztem Wert lassen)
        if target is None:
            if recommendation.get("window_open") and cfg.get(CONF_NO_OFF_FALLBACK, True):
                frost_temp = FROST_FALLBACK_TEMP
                tasks = []
                for entity_id in cfg.get("climate_entities", []):
                    state = self._get_trv_state(entity_id)
                    if state is None:
                        continue
                    try:
                        if abs(float(state.attributes.get("temperature", 0)) - frost_temp) < 0.3:
                            continue
                    except (TypeError, ValueError):
                        pass
                    self._last_written_setpoints[entity_id] = frost_temp
                    tasks.append(self.hass.services.async_call(
                        "climate", "set_temperature",
                        {"entity_id": entity_id, "temperature": frost_temp},
                        blocking=True,
                    ))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            return

        if not (5.0 <= target <= 30.0):
            _LOGGER.warning(
                "ThermoSmart '%s': Ziel %.1f°C außerhalb Sicherheitsbereich",
                self.zone_name, target,
            )
            return

        # Boost-Setpoint verwenden wenn berechnet, sonst Zieltemperatur
        trv_setpoint = recommendation.get("trv_setpoint", target)
        tolerance = cfg.get("temp_tolerance", 0.5)

        # Parallel alle TRVs gleichzeitig ansteuern (wie Better Thermostat)
        tasks = []
        for entity_id in cfg.get("climate_entities", []):
            state = self._get_trv_state(entity_id)
            if state is None:
                continue
            current_setpoint = state.attributes.get("temperature")
            if current_setpoint is not None:
                try:
                    if abs(float(current_setpoint) - trv_setpoint) < tolerance:
                        continue
                except (TypeError, ValueError):
                    pass
            _LOGGER.debug(
                "ThermoSmart '%s' → %s: %.1f°C (Ziel=%.1f°C, Boost+%.1f°C)",
                self.zone_name, entity_id, trv_setpoint, target, trv_setpoint - target,
            )
            self._last_written_setpoints[entity_id] = trv_setpoint
            tasks.append(self.hass.services.async_call(
                "climate", "set_temperature",
                {"entity_id": entity_id, "temperature": trv_setpoint},
                blocking=True,
            ))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    _LOGGER.debug("ThermoSmart '%s': Temperatur-Setpoint fehlgeschlagen: %s", self.zone_name, result)

        # Boost-Tracking für Überschieß-Erkennung
        if trv_setpoint > target:
            if self.zone_id not in self._boost_active:
                self._boost_active[self.zone_id] = {
                    "target": target,
                    "setpoint": trv_setpoint,
                    "started": dt_util.now(),
                }
            else:
                self._boost_active[self.zone_id]["target"] = target
                self._boost_active[self.zone_id]["setpoint"] = trv_setpoint
