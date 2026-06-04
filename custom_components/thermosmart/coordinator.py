"""ThermoSmart Coordinator – Hauptkoordinator für eine Heizzone."""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .tpi import compute_tpi, duty_to_setpoint
from .const import (
    DOMAIN,
    DEFAULT_SCAN_INTERVAL,
    CONF_PRESENCE_PERSONS,
    CONF_HOME_ZONE,
    CONF_VACATION_BOOLEAN,
    CONF_VACATION_TEMP,
    CONF_ECO_TEMP,
    NOISE_FILTER_SPIKE_THRESHOLD,
    NOISE_FILTER_EMA_ALPHA,
    TEMP_NIGHT,
    TEMP_ECO,
    EMA_1H_ALPHA,
    FORECAST_DELTA_FULL_HEAT,
    FORECAST_DELTA_BLEND,
    SEASON_HOURS,
    HEATING_MODE_AUTO,
    HEATING_MODE_AWAY,
    HEATING_MODE_VACATION,
    HEATING_FAILURE_DELAY_MIN,
    HEATING_FAILURE_SLOPE_THRESH,
    HEATING_FAILURE_CMD_DELTA,
    TPI_MAX_BOOST_CELSIUS,
)
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine
from .window import WindowMixin
from .season import SeasonMixin
from .trv_control import TRVControlMixin
from .maintenance import MaintenanceMixin

_LOGGER = logging.getLogger(__name__)


class ThermoSmartCoordinator(
    DataUpdateCoordinator,
    TRVControlMixin,
    MaintenanceMixin,
    SeasonMixin,
    WindowMixin,
):
    """Koordinator für eine einzelne Heizzone.

    Erbt von:
      TRVControlMixin    – Geräteerkennung, Kalibrierung, Quirks, Setpoint-Schreiben
      MaintenanceMixin   – wöchentliche Ventilübung
      SeasonMixin        – Sommer-Erkennung und Frostschutz
      WindowMixin        – Fenstererkennung mit Verzögerungslogik
    """

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

        # Steuerungszustand
        self._active_control: bool = False
        self._mode: str = HEATING_MODE_AUTO
        self._override: float | None = None
        self._override_schedule_period: str | None = None
        self._pre_vacation_mode: str | None = None
        self._summer_override: bool | None = None

        # Event-Listener
        self._event_unsub: list = []

        # Fenster-Tracking (genutzt von WindowMixin)
        self._window_open_at: dict[str, datetime] = {}
        self._window_close_at: dict[str, datetime] = {}
        self._window_open_temp: dict[str, float] = {}

        # Boost-Tracking (genutzt von TRVControlMixin)
        self._boost_active: dict[str, dict] = {}
        self._last_written_setpoints: dict[str, float] = {}

        # Kalibrierung (genutzt von TRVControlMixin)
        self._calibration_offsets: dict[str, float] = {}
        self._auto_quirk_entities: list[str] = []
        self._auto_calibration_map: dict[str, str] = {}
        self._auto_ext_temp_map: dict[str, str] = {}
        self._auto_valve_map: dict[str, str] = {}
        self._trv_offline: set[str] = set()

        # Sommer (genutzt von SeasonMixin)
        self._outdoor_temp_history: deque = deque(
            maxlen=int(SEASON_HOURS * 3600 / DEFAULT_SCAN_INTERVAL)
        )
        self._is_summer: bool = False

        # Wartung (genutzt von MaintenanceMixin)
        self._last_maintenance: datetime | None = None
        self._maintenance_running: bool = False

        # Sensor-Filterung
        self._sensor_ema: dict[str, float] = {}
        self._sensor_noise_count: dict[str, int] = {}

        # Live-Cache für sofortige Card-Aktualisierung (unabhängig vom 5-min-Zyklus)
        self._live_temp: float | None = None
        self._live_humidity: float | None = None

        # Temperatur-Trend
        self._indoor_temp_prev: tuple[datetime, float] | None = None
        self._indoor_temp_slope: float = 0.0
        self._ema_1h: float | None = None

        # Slope-basierte Fenstererkennung (genutzt von WindowMixin._check_window_slope)
        self._slope_win_ema: float | None = None
        self._slope_win_consec: int = 0
        self._slope_window_active: bool = False

        # Heizungsausfall-Erkennung
        self._heating_failure_since: datetime | None = None
        self._heating_failure_notified: bool = False

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

    def set_vacation_override(self, active: bool) -> None:
        """Globaler Urlaubs-Override – speichert/restauriert den bisherigen Modus."""
        if active:
            if self._mode != HEATING_MODE_VACATION:
                self._pre_vacation_mode = self._mode
            self._mode = HEATING_MODE_VACATION
        else:
            self._mode = self._pre_vacation_mode or HEATING_MODE_AUTO
            self._pre_vacation_mode = None
        _LOGGER.info(
            "ThermoSmart '%s': Urlaubs-Override %s (Modus: %s)",
            self.zone_name, "AN" if active else "AUS", self._mode,
        )

    def set_summer_override(self, value: bool | None) -> None:
        """Globaler Sommer-Override vom domain-weiten Schalter."""
        self._summer_override = value
        if value is True:
            self._is_summer = True
        _LOGGER.info(
            "ThermoSmart '%s': Sommer-Override %s",
            self.zone_name, "AN" if value else "AUS – automatische Erkennung aktiv",
        )

    def set_override(self, value: float) -> None:
        self._override = value if value >= 5.0 else None
        self._override_schedule_period = self._current_schedule_period() if self._override else None

    def get_override(self) -> float | None:
        return self._override

    # ── Zeitplan-Helfer ──────────────────────────────────────────────

    def _current_schedule_period(self) -> str:
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

    def _minutes_until_next_comfort(self, cfg: dict) -> int | None:
        """Minuten bis zur nächsten Komfortphase – None wenn bereits aktiv."""
        now = dt_util.now()
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

        if morning <= cur < night:
            return None
        if cur < morning:
            return morning - cur
        return (24 * 60 - cur) + morning

    # ── Event-Listener ───────────────────────────────────────────────

    def setup_event_listeners(self) -> None:
        cfg = self.zone_cfg
        window_sensors: set[str] = set(s for s in cfg.get("window_sensors", []) if s)
        presence: set[str] = set(p for p in cfg.get(CONF_PRESENCE_PERSONS, []) if p)
        vacation_entity = cfg.get(CONF_VACATION_BOOLEAN, "")
        if vacation_entity:
            presence.add(vacation_entity)
        climate_entities: set[str] = set(e for e in cfg.get("climate_entities", []) if e)

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
                        current = self._read_avg_sensor(cfg.get("temp_sensors", []))
                        if current is not None:
                            self._window_open_temp[entity_id] = current
                    else:
                        if entity_id in self._window_open_at:
                            self._window_close_at[entity_id] = now
                            opened_at = self._window_open_at[entity_id]
                            duration_min = (now - opened_at).total_seconds() / 60
                            temp_at_open = self._window_open_temp.pop(entity_id, None)
                            current = self._read_avg_sensor(cfg.get("temp_sensors", []))
                            if temp_at_open is not None and current is not None and duration_min >= 1.0:
                                last_weather = (self.data or {}).get("weather", {})
                                self.hass.async_create_task(
                                    self.learning_engine.async_observe_window_cooling(
                                        self.zone_id, duration_min, temp_at_open, current,
                                        last_weather,
                                    )
                                )
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

        if climate_entities:
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

        # Temp- und Feuchtigkeitssensoren: Live-Cache für sofortige Card-Aktualisierung
        temp_sensors: set[str] = set(s for s in cfg.get("temp_sensors", []) if s)
        humidity_sensors: set[str] = set(s for s in cfg.get("humidity_sensors", []) if s)
        all_indoor_sensors = temp_sensors | humidity_sensors
        if all_indoor_sensors:
            @callback
            def _handle_sensor_change(event) -> None:
                new = event.data.get("new_state")
                if new is None or new.state in ("unknown", "unavailable"):
                    return
                if temp_sensors:
                    val = self._read_avg_sensor(list(temp_sensors))
                    if val is not None:
                        self._live_temp = val
                if humidity_sensors:
                    val = self._read_avg_sensor(list(humidity_sensors))
                    if val is not None:
                        self._live_humidity = val
                self.async_update_listeners()

            cancel_sensors = async_track_state_change_event(
                self.hass, list(all_indoor_sensors), _handle_sensor_change
            )
            self._event_unsub.append(cancel_sensors)

        _LOGGER.debug(
            "ThermoSmart '%s': Event-Listener registriert (%d Präsenz/Fenster, %d TRVs)",
            self.zone_name, len(all_tracked), len(climate_entities),
        )

    def cleanup_event_listeners(self) -> None:
        for cancel in self._event_unsub:
            cancel()
        self._event_unsub.clear()

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

            self.learning_engine.evaluate_forecast_decisions(
                self.zone_id,
                recommendation.get("current_temp"),
                weather_data.get("temperature"),
            )

            target = recommendation.get("adjusted_target")
            current_temp = recommendation.get("current_temp")
            if target is not None:
                # TPI-Koeffizienten aus Lerndaten ableiten
                coef_int, coef_ext = self.learning_engine.get_tpi_coefficients(
                    self.zone_id, weather_data
                )
                if current_temp is not None:
                    duty_cycle = compute_tpi(
                        target, current_temp,
                        weather_data.get("temperature"),
                        coef_int, coef_ext,
                    )
                else:
                    duty_cycle = 0.0

                has_valve = any(
                    self._auto_valve_map.get(eid)
                    for eid in cfg.get("climate_entities", [])
                )

                if has_valve:
                    # TRV mit Ventilsteuerung: Setpoint = Ziel (kein Boost nötig)
                    trv_setpoint = target
                else:
                    # TRV ohne Ventilsteuerung: Duty-Cycle → Boost-Setpoint
                    trv_setpoint = duty_to_setpoint(target, duty_cycle, TPI_MAX_BOOST_CELSIUS)

                boost_factor = self.learning_engine.get_boost_factor(self.zone_id)
                recommendation["trv_setpoint"] = trv_setpoint
                recommendation["boost_factor"] = round(boost_factor, 3)
                recommendation["tpi_duty_cycle"] = round(duty_cycle, 1)
                recommendation["tpi_coef_int"] = round(coef_int, 3)
                recommendation["tpi_coef_ext"] = round(coef_ext, 4)
                recommendation["tpi_valve_direct"] = has_valve

            # Nach trv_setpoint-Berechnung: Heizungsausfall-Erkennung
            self._check_heating_failure(recommendation)

            await self._async_observe_trv_setpoints(cfg, recommendation, weather_data)

            # Outcome-Scoring: Heizsitzung tracken und bewerten
            if not self._is_summer:
                self.learning_engine.update_heating_session(
                    zone_id=self.zone_id,
                    current_temp=current_temp,
                    target=target,
                    is_active_control=self._active_control,
                    weather_data=weather_data,
                    expected_minutes=recommendation.get("preheat_minutes", 0),
                )

            indoor_humidity = self._read_avg_sensor(cfg.get("humidity_sensors", []))
            await self.learning_engine.async_observe(
                zone_id=self.zone_id,
                recommendation=recommendation,
                weather_data=weather_data,
                indoor_humidity=indoor_humidity,
            )

            # External Temperature Input immer schreiben (auch im Beobachtungsmodus)
            # – verbessert die TRV-interne Regelung unabhängig von der aktiven Steuerung
            await self._async_write_external_temp(cfg, recommendation)

            if self._active_control and not self._is_summer:
                await self._async_apply_quirks(cfg)
                await self._watchdog_hvac(cfg, recommendation)
                await self._async_calibrate_trvs(cfg, recommendation)
                await self._apply_temperature(cfg, recommendation)
                # Direkte Ventilsteuerung nach Setpoint-Schreiben
                duty = recommendation.get("tpi_duty_cycle", 0.0)
                if recommendation.get("window_open"):
                    duty = 0.0
                await self._async_set_valve_percent(cfg, duty)
                await self._async_valve_maintenance(cfg, recommendation)
            elif self._active_control and self._is_summer:
                await self._async_apply_quirks(cfg)
                await self._apply_frost_protection(cfg)

            recommendation["heating_failure"] = self._heating_failure_notified
            recommendation["indoor_humidity"] = indoor_humidity

            # Live-Cache aktuell halten (wird auch von Sensor-Listener aktualisiert)
            if recommendation.get("current_temp") is not None:
                self._live_temp = recommendation["current_temp"]
            if indoor_humidity is not None:
                self._live_humidity = indoor_humidity

            return {
                "weather": weather_data,
                "zone": recommendation,
                "active_control": self._active_control,
                "presence": presence,
            }
        except Exception as err:
            raise UpdateFailed(f"ThermoSmart '{self.zone_name}': {err}") from err

    # ── Präsenz ──────────────────────────────────────────────────────

    def _is_person_home(self, person_state: str, zone_entity: str) -> bool:
        if not zone_entity or zone_entity in ("zone.home", "home"):
            return person_state == "home"

        zone_slug = zone_entity.replace("zone.", "").lower()
        if person_state.lower() == zone_slug:
            return True

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

    async def _compute_recommendation(self, cfg: dict, weather_data: dict, mode: str) -> dict:
        current_temp = self._read_avg_sensor(cfg.get("temp_sensors", []))

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

        window_open = self._check_window_open(cfg, current_temp=current_temp)

        comfort_temp = cfg.get("comfort_temp", 21.0)
        night_temp_cfg = cfg.get("night_temp", TEMP_NIGHT)
        base_target = await self.learning_engine.async_get_base_target(
            self.zone_id, mode,
            comfort_temp=comfort_temp,
            night_temp=night_temp_cfg,
            away_temp=cfg.get("away_temp", 17.0),
            vacation_temp=cfg.get(CONF_VACATION_TEMP, 12.0),
            eco_temp=cfg.get(CONF_ECO_TEMP, TEMP_ECO),
            schedule_cfg=cfg,
        )

        weather_offset = 0.0
        if mode == HEATING_MODE_AUTO:
            weather_offset = self.weather_engine.compute_temperature_offset(weather_data)

        preheat_minutes = await self.learning_engine.async_get_preheat_minutes(
            self.zone_id, comfort_temp, current_temp, weather_data
        )

        preheat_active = False
        if mode == HEATING_MODE_AUTO and preheat_minutes > 0 and base_target < comfort_temp:
            mins_until_comfort = self._minutes_until_next_comfort(cfg)
            if mins_until_comfort is not None and 0 < mins_until_comfort <= preheat_minutes:
                base_target = comfort_temp
                preheat_active = True
                _LOGGER.debug(
                    "ThermoSmart '%s': Vorheizen – %d min bis Komfortphase, %d min Aufheizzeit",
                    self.zone_name, mins_until_comfort, preheat_minutes,
                )

        if self._override is not None and self._override_schedule_period is not None:
            current_period = self._current_schedule_period()
            if current_period != self._override_schedule_period:
                _LOGGER.info(
                    "ThermoSmart '%s': Manueller Override abgelaufen (Slot %s → %s)",
                    self.zone_name, self._override_schedule_period, current_period,
                )
                self._override = None
                self._override_schedule_period = None

        override = self.get_override()
        biased_suppression = 1.0

        if override is not None and not window_open:
            adjusted_target = override
            override_active = True
        elif not window_open:
            raw_target = round(base_target + weather_offset, 1)
            if mode == HEATING_MODE_AUTO:
                if preheat_active:
                    biased_suppression = 1.0
                    adjusted_target = raw_target
                else:
                    raw_suppression = self.weather_engine.compute_forecast_suppression(
                        weather_data, raw_target, night_temp_cfg
                    )
                    forecast_bias = self.learning_engine.get_forecast_bias(self.zone_id)
                    biased_suppression = 1.0 - (1.0 - raw_suppression) * forecast_bias

                    if current_temp is not None:
                        delta = raw_target - current_temp
                        if delta >= FORECAST_DELTA_FULL_HEAT:
                            biased_suppression = 1.0
                        elif delta >= FORECAST_DELTA_BLEND:
                            blend = (delta - FORECAST_DELTA_BLEND) / (FORECAST_DELTA_FULL_HEAT - FORECAST_DELTA_BLEND)
                            biased_suppression = min(1.0, biased_suppression + blend * (1.0 - biased_suppression))

                    adjusted_target = round(
                        night_temp_cfg + biased_suppression * (raw_target - night_temp_cfg), 1
                    )

                    if current_temp is not None and biased_suppression < 0.95:
                        comfort_floor = round(current_temp - 0.5, 1)
                        if adjusted_target < comfort_floor:
                            _LOGGER.debug(
                                "ThermoSmart '%s': Komfort-Boden greift (%.1f°C → %.1f°C)",
                                self.zone_name, adjusted_target, comfort_floor,
                            )
                            adjusted_target = comfort_floor

                    forecast_high = weather_data.get("forecast_high")
                    if biased_suppression < 0.95 and forecast_high is not None and current_temp is not None:
                        self.learning_engine.record_forecast_decision(
                            self.zone_id, adjusted_target, raw_target, forecast_high,
                            current_temp, biased_suppression, weather_data.get("temperature"),
                        )
            else:
                adjusted_target = raw_target
            override_active = False
        else:
            adjusted_target = None
            override_active = False

        if mode == HEATING_MODE_AUTO and not window_open and override is None:
            suppression_pct = round((1.0 - biased_suppression) * 100)
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
            "preheat_active": preheat_active,
            "forecast_suppression": suppression_pct,
            "learning_confidence": self.learning_engine.get_confidence(self.zone_id),
            "outdoor_temp": weather_data.get("temperature"),
            "forecast_high": weather_data.get("forecast_high"),
            "weather_condition": weather_data.get("condition"),
            "temp_slope": self._indoor_temp_slope,
            "temp_ema_1h": self._ema_1h,
        }

    # ── Sensor-Lesen ─────────────────────────────────────────────────

    def _read_avg_sensor(self, sensor_ids: list[str]) -> float | None:
        """Durchschnitt über alle verfügbaren Sensoren – ignoriert Ausfälle automatisch."""
        values = []
        seen: set[str] = set()
        for sid in sensor_ids:
            if not sid or sid in seen:
                continue
            seen.add(sid)
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
        """EMA-Glättung mit Spike-Erkennung – Ausreißer werden ignoriert."""
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

    # ── Boost-Tracking ───────────────────────────────────────────────

    def _check_boost_outcome(self, cfg: dict) -> None:
        """Überschießen oder zu langsames Heizen erkennen und Boost-Faktor anpassen."""
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
            self._boost_active.pop(self.zone_id)
        else:
            started = entry.get("started")
            if started and (dt_util.now() - started).total_seconds() > 1800:
                if current < prev_target - 1.0:
                    self.learning_engine.update_boost_factor(self.zone_id, overshot=False, slow=True)
                    _LOGGER.info(
                        "ThermoSmart '%s': Langsames Heizen nach 30 min %.1f°C < %.1f°C – Faktor erhöht",
                        self.zone_name, current, prev_target,
                    )
                self._boost_active.pop(self.zone_id)

    # ── Heizungsausfall-Erkennung ─────────────────────────────────────

    def _check_heating_failure(self, recommendation: dict) -> None:
        """Erkennt Heizungsausfall: TRV soll heizen, aber Raumtemperatur fällt.

        Bedingungen für Alarm:
          - Aktive Steuerung an (Beobachtungsmodus → kein Alarm, da BT steuert)
          - TRV-Setpoint > aktuelle Temp + HEATING_FAILURE_CMD_DELTA (echter Heizbefehl)
          - Temperatur-Slope < –HEATING_FAILURE_SLOPE_THRESH (°C/min) für HEATING_FAILURE_DELAY_MIN
          - Kein offenes Fenster (das wäre erklärter Temperaturabfall)

        Alarm: HA Persistent Notification + Log-Warning.
        Alarm wird automatisch quittiert wenn Zieltemperatur erreicht ist.
        """
        if not self._active_control:
            self._heating_failure_since = None
            return

        trv_setpoint = recommendation.get("trv_setpoint")
        current_temp = recommendation.get("current_temp")
        target = recommendation.get("adjusted_target")
        window_open = recommendation.get("window_open", False)

        if window_open or trv_setpoint is None or current_temp is None or target is None:
            self._heating_failure_since = None
            return

        is_heating_commanded = (trv_setpoint - current_temp) >= HEATING_FAILURE_CMD_DELTA
        slope = self._indoor_temp_slope

        if is_heating_commanded and slope < HEATING_FAILURE_SLOPE_THRESH:
            if self._heating_failure_since is None:
                self._heating_failure_since = dt_util.now()
                _LOGGER.debug(
                    "ThermoSmart '%s': Möglicher Heizungsausfall – "
                    "Temp fällt trotz Heizbefehl (SP=%.1f°C, Ist=%.1f°C, Slope=%.4f°C/min)",
                    self.zone_name, trv_setpoint, current_temp, slope,
                )

            elapsed_min = (dt_util.now() - self._heating_failure_since).total_seconds() / 60
            if elapsed_min >= HEATING_FAILURE_DELAY_MIN and not self._heating_failure_notified:
                self._heating_failure_notified = True
                _LOGGER.warning(
                    "ThermoSmart '%s': HEIZUNGSAUSFALL – SP=%.1f°C, Ist=%.1f°C, "
                    "Slope=%.4f°C/min, seit %.0f min. TRV oder Wärmequelle prüfen!",
                    self.zone_name, trv_setpoint, current_temp, slope, elapsed_min,
                )
                self.hass.async_create_task(
                    self.hass.services.async_call(
                        "persistent_notification", "create",
                        {
                            "title": f"ThermoSmart: Heizungsausfall – {self.zone_name}",
                            "message": (
                                f"**{self.zone_name}**: TRV soll auf **{trv_setpoint:.1f}°C** heizen, "
                                f"aber die Raumtemperatur fällt seit **{elapsed_min:.0f} Minuten** "
                                f"(aktuell **{current_temp:.1f}°C**, Ziel **{target:.1f}°C**).\n\n"
                                f"Bitte TRV, Heizkörper und Wärmequelle prüfen."
                            ),
                            "notification_id": f"thermosmart_heating_failure_{self.zone_id}",
                        },
                        blocking=False,
                    )
                )
        else:
            self._heating_failure_since = None
            # Alarm quittieren sobald Zieltemperatur erreicht
            if self._heating_failure_notified and current_temp >= target - 0.5:
                self._heating_failure_notified = False
                self.hass.async_create_task(
                    self.hass.services.async_call(
                        "persistent_notification", "dismiss",
                        {"notification_id": f"thermosmart_heating_failure_{self.zone_id}"},
                        blocking=False,
                    )
                )

    # ── TRV-Setpoint-Beobachtung ──────────────────────────────────────

    async def _async_observe_trv_setpoints(
        self, cfg: dict, recommendation: dict, weather_data: dict
    ) -> None:
        """TRV-Setpoints im Beobachtungsmodus oder der aktiven Steuerung erfassen und lernen."""
        current_temp = recommendation.get("current_temp")
        target = recommendation.get("adjusted_target")
        if current_temp is None or target is None:
            return
        if target <= current_temp:
            return

        heat_rate = None
        now = dt_util.now()
        if self.zone_id in self.learning_engine._last_temp:
            last_time, last_temp = self.learning_engine._last_temp[self.zone_id]
            elapsed_min = (now - last_time).total_seconds() / 60
            if 3 <= elapsed_min <= 15 and current_temp > last_temp:
                heat_rate = round((current_temp - last_temp) / elapsed_min, 5)

        if self._active_control:
            written = [v for v in self._last_written_setpoints.values() if v >= current_temp]
            if not written:
                return
            trv_setpoint = round(sum(written) / len(written), 1)
            await self.learning_engine.async_observe_trv_setpoint(
                zone_id=self.zone_id,
                trv_setpoint=trv_setpoint,
                indoor_temp=current_temp,
                target=target,
                weather_data=weather_data,
                heat_rate=heat_rate,
            )
        else:
            for entity_id in cfg.get("climate_entities", []):
                state = self.hass.states.get(entity_id)
                if state is None or state.state in ("unavailable", "unknown", "off"):
                    continue
                try:
                    trv_setpoint = float(state.attributes.get("temperature", 0))
                except (TypeError, ValueError):
                    continue
                if trv_setpoint < current_temp:
                    continue
                await self.learning_engine.async_observe_trv_setpoint(
                    zone_id=self.zone_id,
                    trv_setpoint=trv_setpoint,
                    indoor_temp=current_temp,
                    target=target,
                    weather_data=weather_data,
                    heat_rate=heat_rate,
                )
