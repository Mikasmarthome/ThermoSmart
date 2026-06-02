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
    CONF_VACATION_BOOLEAN,
    CONF_CALIBRATION_ENTITIES,
    TEMP_NIGHT,
    TEMP_FROST_PROTECTION,
    SUMMER_THRESHOLD,
    WINTER_THRESHOLD,
    SEASON_HOURS,
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
        self._listeners: list = []
        self._window_open_at: dict[str, datetime] = {}
        self._window_close_at: dict[str, datetime] = {}
        self._boost_active: dict[str, dict] = {}
        self._calibration_offsets: dict[str, float] = {}
        # Sommer-Erkennung: rollendes Fenster der letzten SEASON_HOURS Außentemperaturen
        self._outdoor_temp_history: deque = deque(maxlen=int(SEASON_HOURS * 3600 / DEFAULT_SCAN_INTERVAL))
        self._is_summer: bool = False

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

    # ── Event-Listener ───────────────────────────────────────────────

    def setup_event_listeners(self) -> None:
        """Sofort-Reaktion auf Fenster-, Personen- und Urlaubsänderungen."""
        cfg = self.zone_cfg
        window_sensors: set[str] = set(s for s in cfg.get("window_sensors", []) if s)
        presence: set[str] = set(p for p in cfg.get(CONF_PRESENCE_PERSONS, []) if p)
        vacation_entity = cfg.get(CONF_VACATION_BOOLEAN, "")
        if vacation_entity:
            presence.add(vacation_entity)

        all_tracked = window_sensors | presence
        if not all_tracked:
            return

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
        self._listeners.append(cancel)
        _LOGGER.debug(
            "ThermoSmart '%s': Event-Listener für %d Entities registriert",
            self.zone_name, len(all_tracked),
        )

    def cleanup_event_listeners(self) -> None:
        for cancel in self._listeners:
            cancel()
        self._listeners.clear()

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
                await self._watchdog_hvac(cfg, recommendation)
                await self._async_calibrate_trvs(cfg, recommendation)
                await self._apply_temperature(cfg, recommendation)
            elif self._active_control and self._is_summer:
                # Sommer: alle TRVs auf Frostschutz
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

    def _get_presence_state(self) -> dict:
        cfg = self.zone_cfg
        persons_home, persons_away = [], []
        for person in cfg.get(CONF_PRESENCE_PERSONS, []):
            state = self.hass.states.get(person)
            (persons_home if (state and state.state == "home") else persons_away).append(person)

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
                    values.append(float(state.state))
                except ValueError:
                    pass
        return round(sum(values) / len(values), 1) if values else None

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
        if delta < 1.0:
            return target  # Nah am Ziel → kein Boost nötig

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
        """Erkennt Überschießen nach Boost → reduziert Lernfaktor."""
        if self.zone_id not in self._boost_active:
            return
        current = self._read_avg_sensor(cfg.get("temp_sensors", []))
        if current is None:
            return
        prev_target = self._boost_active[self.zone_id]["target"]
        tolerance = cfg.get("temp_tolerance", 0.5)
        if current > prev_target + tolerance:
            self.learning_engine.update_boost_factor(self.zone_id, overshot=True)
            _LOGGER.info(
                "ThermoSmart '%s': Boost-Überschießen %.1f°C > %.1f°C – Faktor angepasst",
                self.zone_name, current, prev_target,
            )
            self._boost_active.pop(self.zone_id)
        elif current >= prev_target - tolerance * 0.5:
            # Ziel sauber erreicht – kein Überschießen
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
            state = self.hass.states.get(entity_id)
            if state is None or state.state in ("unavailable", "unknown"):
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
        """
        calibration_entities = cfg.get(CONF_CALIBRATION_ENTITIES, [])
        if not calibration_entities:
            return

        room_temp = recommendation.get("current_temp")
        if room_temp is None:
            return

        climate_entities = cfg.get("climate_entities", [])

        for i, cal_entity in enumerate(calibration_entities):
            if not cal_entity:
                continue

            # TRV-eigene Temperaturmessung aus zugehöriger Climate-Entity (gleicher Index)
            trv_temp: float | None = None
            if i < len(climate_entities):
                trv_state = self.hass.states.get(climate_entities[i])
                if trv_state and trv_state.state not in ("unavailable", "unknown"):
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

        # Boost-Setpoint verwenden wenn berechnet, sonst Zieltemperatur
        trv_setpoint = recommendation.get("trv_setpoint", target)
        tolerance = cfg.get("temp_tolerance", 0.5)

        # Parallel alle TRVs gleichzeitig ansteuern (wie Better Thermostat)
        tasks = []
        for entity_id in cfg.get("climate_entities", []):
            state = self.hass.states.get(entity_id)
            if not state or state.state in ("unavailable", "unknown"):
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
            tasks.append(self.hass.services.async_call(
                "climate", "set_temperature",
                {"entity_id": entity_id, "temperature": trv_setpoint},
                blocking=True,
            ))

        if tasks:
            await asyncio.gather(*tasks)

        # Boost-Tracking für Überschieß-Erkennung
        if trv_setpoint > target:
            self._boost_active[self.zone_id] = {"target": target, "setpoint": trv_setpoint}
