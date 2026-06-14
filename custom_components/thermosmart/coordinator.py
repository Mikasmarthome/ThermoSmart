"""ThermoSmart Coordinator – Hauptkoordinator für eine Heizzone."""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta

from collections.abc import Callable

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_call_later, async_track_state_change_event
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
    TEMP_COMFORT,
    TEMP_AWAY,
    TEMP_FROST_PROTECTION,
    EMA_1H_ALPHA,
    FORECAST_DELTA_FULL_HEAT,
    FORECAST_DELTA_BLEND,
    SEASON_HOURS,
    HEATING_MODE_AUTO,
    HEATING_MODE_COMFORT,
    HEATING_MODE_NIGHT,
    HEATING_MODE_ECO,
    HEATING_MODE_AWAY,
    HEATING_MODE_VACATION,
    HEATING_FAILURE_DELAY_MIN,
    HEATING_FAILURE_SLOPE_THRESH,
    HEATING_FAILURE_CMD_DELTA,
    TPI_MAX_BOOST_CELSIUS,
)
from .device_profiles import DeviceProfile
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine
from .window import WindowMixin
from .season import SeasonMixin
from .trv_control import TRVControlMixin
from .maintenance import MaintenanceMixin

_LOGGER = logging.getLogger(__name__)

# valve_opening_degree recovery is retried across the first N coordinator refreshes
# to handle slow Zigbee2MQTT startup on full HA restart.
_VALVE_RESET_MAX_ATTEMPTS = 3


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
        # Cancel functions for scheduled delay-expiry refreshes (one slot per sensor)
        self._window_delay_cancel: dict[str, Callable[[], None]] = {}

        # Boost-Tracking (genutzt von TRVControlMixin)
        self._boost_active: dict[str, dict] = {}
        self._last_written_setpoints: dict[str, float] = {}

        # Kalibrierung (genutzt von TRVControlMixin)
        self._calibration_offsets: dict[str, float] = {}
        self._auto_quirk_entities: list[str] = []
        self._auto_calibration_map: dict[str, str] = {}
        self._auto_ext_temp_map: dict[str, str] = {}
        self._auto_valve_map: dict[str, str] = {}
        self._auto_temp_source_map: dict[str, str] = {}
        self._trv_offline: set[str] = set()
        self._valve_reset_done: bool = False
        self._valve_reset_attempts: int = 0

        # Device profiles — populated by async_detect_device_entities (TRVControlMixin)
        self._device_profiles: dict[str, DeviceProfile] = {}

        # Temperature source management (genutzt von TRVControlMixin)
        self._temp_source_owned: dict[str, str] = {}   # select_entity_id → "external"
        self._sensor_unavail_since: datetime | None = None
        self._temp_source_warn_ts: datetime | None = None

        # Sommer (genutzt von SeasonMixin)
        self._outdoor_temp_history: deque = deque(
            maxlen=int(SEASON_HOURS * 3600 / DEFAULT_SCAN_INTERVAL)
        )
        self._is_summer: bool = False
        self._indoor_safety_active: bool = False

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
        _LOGGER.info(
            "ThermoSmart '%s': Aktive Steuerung %s",
            self.zone_name,
            "AN – Thermostat wird gesteuert" if active else "AUS – Beobachtungsmodus",
        )

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        # Immediately patch adjusted_target with the preset base temperature so
        # that async_write_ha_state() in climate.py can expose the new target at
        # once – before async_request_refresh() runs the full weather/learning
        # recalculation.  The refresh will overwrite this value with the properly
        # adjusted figure shortly after.
        if self.data is not None and isinstance(self.data.get("zone"), dict):
            cfg = self.zone_cfg
            quick_target = {
                HEATING_MODE_COMFORT:  cfg.get("comfort_temp", TEMP_COMFORT),
                HEATING_MODE_NIGHT:    cfg.get("night_temp", TEMP_NIGHT),
                HEATING_MODE_ECO:      cfg.get(CONF_ECO_TEMP, TEMP_ECO),
                HEATING_MODE_AWAY:     cfg.get("away_temp", TEMP_AWAY),
                HEATING_MODE_VACATION: cfg.get(CONF_VACATION_TEMP, TEMP_FROST_PROTECTION),
                HEATING_MODE_AUTO:     cfg.get("comfort_temp", TEMP_COMFORT),
            }.get(mode)
            if quick_target is not None:
                self.data["zone"]["adjusted_target"] = quick_target

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
        """Globaler Sommer-Override vom domain-weiten Select.

        value=True  → Sommer erzwungen  (_is_summer = True)
        value=False → Winter erzwungen   (_is_summer = False)
        value=None  → Automatik          (_is_summer wird durch _update_summer_mode neu berechnet)
        """
        self._summer_override = value
        if value is True:
            self._is_summer = True
        elif value is False:
            self._is_summer = False
        # value is None: _is_summer unverändert, nächster _update_summer_mode-Zyklus übernimmt
        _LOGGER.info(
            "ThermoSmart '%s': Sommer-Override %s",
            self.zone_name,
            "EIN (erzwungen)" if value is True
            else ("AUS (erzwungen)" if value is False else "Automatik – 72h-Erkennung aktiv"),
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
                parts = str(raw).split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except (ValueError, AttributeError, IndexError):
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
                parts = str(raw).split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except (ValueError, AttributeError, IndexError):
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
        # CONF_VACATION_BOOLEAN: planned per-zone vacation boolean input —
        # implemented here (tracks state changes) but NOT exposed in config_flow UI.
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
                        # Cancel any pending timer (leftover close-delay or open-delay)
                        _cancel = self._window_delay_cancel.pop(entity_id, None)
                        if _cancel is not None:
                            _cancel()
                        self._window_close_at.pop(entity_id, None)
                        current = self._read_avg_sensor(cfg.get("temp_sensors", []))
                        if current is not None:
                            self._window_open_temp[entity_id] = current
                        # Schedule a coordinator refresh exactly when the open delay expires
                        # so window-open mode activates at the configured time, not at the
                        # next regular 5-minute coordinator cycle.
                        open_delay_secs = cfg.get("window_open_delay", 5) * 60

                        @callback
                        def _open_delay_expired(_now, _eid=entity_id):
                            self.hass.async_create_task(self.async_request_refresh())

                        self._window_delay_cancel[entity_id] = async_call_later(
                            self.hass, open_delay_secs, _open_delay_expired
                        )
                    else:
                        # Cancel any pending open-delay timer (sensor closed before delay elapsed)
                        _cancel = self._window_delay_cancel.pop(entity_id, None)
                        if _cancel is not None:
                            _cancel()
                        if entity_id in self._window_open_at:
                            opened_at = self._window_open_at[entity_id]
                            open_delay_td = timedelta(minutes=cfg.get("window_open_delay", 5))

                            # Close-Timer nur setzen wenn open_delay tatsächlich abgelaufen war.
                            # Schließt das Fenster VOR Ablauf des open_delay, darf weder
                            # eine Heizpause noch ein close_delay ausgelöst werden.
                            if (now - opened_at) >= open_delay_td:
                                self._window_close_at[entity_id] = now
                                # Schedule refresh when close delay expires so the TRV
                                # returns to normal at the configured time, not the next cycle.
                                close_delay_secs = cfg.get("window_close_delay", 2) * 60

                                @callback
                                def _close_delay_expired(_now, _eid=entity_id):
                                    self.hass.async_create_task(self.async_request_refresh())

                                self._window_delay_cancel[entity_id] = async_call_later(
                                    self.hass, close_delay_secs, _close_delay_expired
                                )

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
                new = event.data.get("new_state")
                if new is None:
                    return
                # TRV-only zone: keep display cache current when no external sensor.
                if not temp_sensors:
                    trv_t = self._read_trv_avg_temp(list(climate_entities))
                    if trv_t is not None:
                        self._live_temp = trv_t
                        self.async_update_listeners()
                if not self._active_control:
                    return
                # During window-open, the TRV may echo its previous setpoint before
                # confirming the frost temp — suppress manual-change detection to
                # prevent ping-pong between frost temp and normal setpoint.
                if (self.data or {}).get("zone", {}).get("window_open", False):
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
                    val = self._read_raw_avg_sensor(list(temp_sensors))
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
        for cancel in self._window_delay_cancel.values():
            cancel()
        self._window_delay_cancel.clear()

    # ── Hauptschleife ────────────────────────────────────────────────

    async def _async_update_data(self) -> dict:
        try:
            cfg = self.zone_cfg

            # Safety-net for full HA restart: valve_opening_degree recovery is also
            # called from async_detect_device_entities, but at startup Z2M entities
            # may not be available yet.  Retry across up to _VALVE_RESET_MAX_ATTEMPTS
            # refreshes; stop as soon as recovery reports success (all relevant entities
            # were reachable) or after the attempt limit is exhausted.
            if not self._valve_reset_done:
                done = await self._reset_valve_opening_degree()
                self._valve_reset_attempts += 1
                if done or self._valve_reset_attempts >= _VALVE_RESET_MAX_ATTEMPTS:
                    if not done:
                        _LOGGER.warning(
                            "ThermoSmart '%s': valve_opening_degree recovery gave up after "
                            "%d attempts — some TRV entities were still unavailable",
                            self.zone_name, self._valve_reset_attempts,
                        )
                    self._valve_reset_done = True

            self._check_boost_outcome(cfg)

            weather_data = await self.weather_engine.async_get_data()
            self._update_summer_mode(weather_data)

            presence = self._get_presence_state()
            mode = self._effective_mode(presence)
            recommendation = await self._compute_recommendation(cfg, weather_data, mode)

            effective_summer = self._is_summer
            if self._is_summer and self._summer_override is None:
                current_indoor = recommendation.get("current_temp")
                safety_threshold = cfg.get("night_temp", TEMP_NIGHT)
                if current_indoor is not None and current_indoor < safety_threshold:
                    effective_summer = False
                    if not self._indoor_safety_active:
                        self._indoor_safety_active = True
                        _LOGGER.warning(
                            "ThermoSmart '%s': Automatic summer mode temporarily bypassed "
                            "because indoor temperature %.1f°C is below night temperature %.1f°C",
                            self.zone_name,
                            current_indoor,
                            safety_threshold,
                        )
                elif self._indoor_safety_active and (
                    current_indoor is None or current_indoor >= safety_threshold + 0.5
                ):
                    self._indoor_safety_active = False
                    _LOGGER.info(
                        "ThermoSmart '%s': Automatic summer mode indoor safety cleared "
                        "(indoor %.1f°C ≥ night temp %.1f°C + 0.5°C hysteresis)",
                        self.zone_name,
                        current_indoor if current_indoor is not None else float("nan"),
                        safety_threshold,
                    )
            else:
                if self._indoor_safety_active:
                    self._indoor_safety_active = False

            recommendation["is_summer"] = effective_summer

            self.learning_engine.evaluate_forecast_decisions(
                self.zone_id,
                recommendation.get("current_temp"),
                weather_data.get("temperature"),
            )

            # Use effective_target (weather-adjusted) for TPI/TRV calculations.
            # adjusted_target is only the display value.
            target = recommendation.get("effective_target")
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

            # Display fallback: when no active heating target exists (summer, window-open,
            # away with adjusted_target=None), populate trv_setpoint for sensor display so
            # the sensor does not show Unknown after restart. Does not affect control logic,
            # TPI, learning, or boost tracking.
            if "trv_setpoint" not in recommendation:
                for _eid in cfg.get("climate_entities", []):
                    _last = self._last_written_setpoints.get(_eid)
                    if _last is not None:
                        recommendation["trv_setpoint"] = _last
                        break
                if "trv_setpoint" not in recommendation:
                    for _eid in cfg.get("climate_entities", []):
                        _st = self.hass.states.get(_eid)
                        if _st is not None:
                            try:
                                _t = float(_st.attributes.get("temperature", 0))
                                if _t > 0:
                                    recommendation["trv_setpoint"] = _t
                                    break
                            except (TypeError, ValueError):
                                pass

            # Nach trv_setpoint-Berechnung: Heizungsausfall-Erkennung
            self._check_heating_failure(recommendation)

            await self._async_observe_trv_setpoints(cfg, recommendation, weather_data)

            # Outcome-Scoring: Heizsitzung tracken und bewerten
            if not effective_summer:
                self.learning_engine.update_heating_session(
                    zone_id=self.zone_id,
                    current_temp=current_temp,
                    target=target,   # effective_target – actual heating goal
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
                is_active_control=self._active_control,
                window_open=recommendation.get("window_open", False),
                control_reason=self._control_reason(recommendation),
                preheat_active=recommendation.get("preheat_active", False),
                heating_failure=bool(recommendation.get("heating_failure")),
                vacation=recommendation.get("mode") == HEATING_MODE_VACATION,
                summer_mode=bool(recommendation.get("is_summer", False)),
                schedule_period=self._schedule_period(recommendation, cfg),
            )

            # External Temperature Input immer schreiben (auch im Beobachtungsmodus)
            # – verbessert die TRV-interne Regelung unabhängig von der aktiven Steuerung
            await self._async_write_external_temp(cfg, recommendation)

            # Temperature source select management (external/internal)
            # Must run after effective_summer is resolved so recommendation["is_summer"] is set.
            await self._async_manage_temp_source(cfg, recommendation)

            if self._active_control and not effective_summer:
                await self._async_apply_quirks(cfg)
                await self._watchdog_hvac(cfg, recommendation)
                await self._async_calibrate_trvs(cfg, recommendation)
                await self._apply_temperature(cfg, recommendation)
                # Direkte Ventilsteuerung nach Setpoint-Schreiben
                duty = recommendation.get("tpi_duty_cycle", 0.0)
                if recommendation.get("window_open"):
                    duty = 0.0
                await self._async_set_valve_percent(cfg, duty)
            elif self._active_control and effective_summer:
                await self._async_apply_quirks(cfg)
                await self._apply_frost_protection(cfg)

            # Valve maintenance runs in summer mode or observation mode.
            # Must be called outside the active-control blocks — the function
            # self-guards via valve_likely_idle = is_summer or not active_control.
            await self._async_valve_maintenance(cfg, recommendation)

            recommendation["heating_failure"] = self._heating_failure_notified
            recommendation["indoor_humidity"] = indoor_humidity

            # Display cache: raw sensor average so climate.current_temperature
            # reflects the actual reading, not the EMA-smoothed control value.
            # Fallback: use TRV current_temperature when no external sensor.
            raw_temp = self._read_raw_avg_sensor(cfg.get("temp_sensors", []))
            if raw_temp is None:
                raw_temp = self._read_trv_avg_temp(cfg.get("climate_entities", []))
            if raw_temp is not None:
                self._live_temp = raw_temp
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
        # CONF_VACATION_BOOLEAN: planned per-zone vacation boolean input —
        # implemented here (reads state) but NOT exposed in config_flow UI.
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

    def _control_reason(self, recommendation: dict) -> str:
        """Determine the primary reason that governs the current heating target.

        Priority order ensures only one unambiguous reason is recorded:
          1. window_open    – heating suppressed due to open window
          2. manual_override – user set an explicit target temperature
          3. vacation       – vacation/frost-protection mode active
          4. summer_mode    – season-based heating suppression active
          5. presence       – away mode triggered by absence of all occupants
          6. schedule       – default: auto mode driven by time schedule
        Note: frost_protection is not a separate state; it is covered by summer_mode.
        Note: explicit eco/night/comfort mode selections are covered by schedule
              (they are schedule-like user preferences, not presence or override events).
        """
        if recommendation.get("window_open"):
            return "window_open"
        if recommendation.get("override_active"):
            return "manual_override"
        if recommendation.get("mode") == HEATING_MODE_VACATION:
            return "vacation"
        if recommendation.get("is_summer"):
            return "summer_mode"
        if recommendation.get("mode") == HEATING_MODE_AWAY:
            return "presence"
        return "schedule"

    def _schedule_period(self, recommendation: dict, cfg: dict) -> str | None:
        """Determine the active schedule period for observation context.

        Returns one of: "comfort", "night", "eco", "away", or None
        (vacation and summer_mode are captured by their own boolean fields).
        """
        mode = recommendation.get("mode", HEATING_MODE_AUTO)
        if recommendation.get("is_summer") or mode == HEATING_MODE_VACATION:
            return None
        if mode in (HEATING_MODE_COMFORT, HEATING_MODE_NIGHT, HEATING_MODE_ECO, HEATING_MODE_AWAY):
            return mode
        # AUTO mode: derive from the active schedule slot
        if not cfg.get("schedule_enabled", True):
            return "comfort"
        raw = self._current_schedule_period()
        return raw.split("_", 1)[1] if "_" in raw else "comfort"

    # ── Berechnung ───────────────────────────────────────────────────

    async def _compute_recommendation(self, cfg: dict, weather_data: dict, mode: str) -> dict:
        current_temp = self._read_avg_sensor(cfg.get("temp_sensors", []))
        if current_temp is None:
            current_temp = self._read_trv_avg_temp(cfg.get("climate_entities", []))

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

        # Presence-aware override clearing: when presence detection switches to
        # all-away (AUTO mode only), a manual override must not block the away
        # temperature from taking effect.
        # VACATION mode has higher priority (set via set_vacation_override) and is
        # never reached as HEATING_MODE_AWAY here.
        # Non-AUTO modes (_mode != HEATING_MODE_AUTO) return their own mode via
        # _effective_mode(), never HEATING_MODE_AWAY, so this block is inert.
        if mode == HEATING_MODE_AWAY and self._override is not None:
            _LOGGER.info(
                "ThermoSmart '%s': Manual override cleared – all persons away",
                self.zone_name,
            )
            self._override = None
            self._override_schedule_period = None

        override = self.get_override()
        biased_suppression = 1.0

        # ── Effective target (internal) ────────────────────────────────────────
        # Incorporates weather offset + forecast suppression.
        # Used for TPI calculation and TRV setpoint – NOT shown to the user.
        #
        # ── Adjusted target (display) ──────────────────────────────────────────
        # Always the configured schedule / mode temperature, or the manual override.
        # This is what the climate entity and all UI elements show – no hidden offsets.

        if override is not None and not window_open:
            # Manual override: user set an explicit temp → show AND use it as-is
            effective_target = override
            adjusted_target = override
            override_active = True
        elif not window_open:
            raw_target = round(base_target + weather_offset, 1)
            if mode == HEATING_MODE_AUTO:
                if preheat_active:
                    # Pre-heat window: always heat to full comfort target
                    biased_suppression = 1.0
                    effective_target = raw_target
                elif base_target >= comfort_temp:
                    # Active comfort window: always heat to configured comfort temperature.
                    # Forecast may influence WHEN we pre-heat, but not the comfort target itself.
                    biased_suppression = 1.0
                    effective_target = raw_target
                    _LOGGER.debug(
                        "ThermoSmart '%s': Komfortzeit – Prognose-Unterdrückung deaktiviert "
                        "(Ziel: %.1f°C)",
                        self.zone_name, raw_target,
                    )
                else:
                    # Night/transition window: forecast suppression applies
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

                    effective_target = round(
                        night_temp_cfg + biased_suppression * (raw_target - night_temp_cfg), 1
                    )

                    if current_temp is not None and biased_suppression < 0.95:
                        comfort_floor = round(current_temp - 0.5, 1)
                        if effective_target < comfort_floor:
                            _LOGGER.debug(
                                "ThermoSmart '%s': Komfort-Boden greift (%.1f°C → %.1f°C)",
                                self.zone_name, effective_target, comfort_floor,
                            )
                            effective_target = comfort_floor

                    forecast_high = weather_data.get("forecast_high")
                    if biased_suppression < 0.95 and forecast_high is not None and current_temp is not None:
                        self.learning_engine.record_forecast_decision(
                            self.zone_id, effective_target, raw_target, forecast_high,
                            current_temp, biased_suppression, weather_data.get("temperature"),
                        )
            else:
                effective_target = raw_target
            # Display: always show the configured base temperature (no offsets)
            adjusted_target = base_target
            override_active = False
        else:
            effective_target = None
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
            "adjusted_target": adjusted_target,    # display value – no offsets
            "effective_target": effective_target,  # internal value – weather + suppression
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

    def _read_raw_avg_sensor(self, sensor_ids: list[str]) -> float | None:
        """Raw average of available sensors — no EMA smoothing, for display only.

        Filters out invalid states (unknown, unavailable, None) but does not
        apply EMA or spike detection. Used exclusively for _live_temp so that
        climate.current_temperature reflects the actual sensor reading instead
        of the EMA-smoothed value used by the control path.
        """
        values = []
        seen: set[str] = set()
        for sid in sensor_ids:
            if not sid or sid in seen:
                continue
            seen.add(sid)
            state = self.hass.states.get(sid)
            if state and state.state not in ("unknown", "unavailable", "None"):
                try:
                    values.append(float(state.state))
                except ValueError:
                    pass
        return round(sum(values) / len(values), 1) if values else None

    def _read_trv_avg_temp(self, climate_entity_ids: list[str]) -> float | None:
        """Average current_temperature from TRV entities — fallback when no external sensor.

        Reads the current_temperature attribute directly; no EMA, no spike filter.
        Used when temp_sensors is empty so TPI and the display cache have a valid value.
        External temp sensors always take priority — this method is only called as fallback.
        """
        values = []
        seen: set[str] = set()
        for eid in climate_entity_ids:
            if not eid or eid in seen:
                continue
            seen.add(eid)
            state = self.hass.states.get(eid)
            if state and state.state not in ("unknown", "unavailable"):
                trv_t = state.attributes.get("current_temperature")
                if trv_t is not None:
                    try:
                        values.append(float(trv_t))
                    except (TypeError, ValueError):
                        pass
        return round(sum(values) / len(values), 1) if values else None

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
          - Aktive Steuerung an (Beobachtungsmodus → kein Alarm, da ein externer Regler steuert)
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
        target = recommendation.get("effective_target")   # use actual heating goal
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
        # Use effective_target (weather-adjusted) as the actual heating goal for observations
        target = recommendation.get("effective_target")
        if current_temp is None or target is None:
            return

        # In active control: only observe when ThermoSmart itself needs to heat.
        # In observation mode: do NOT filter on our own target – the external controller
        # may heat to a different (higher) target. The actual TRV setpoint check below
        # (trv_setpoint < current_temp → skip) is sufficient as the quality gate.
        if self._active_control and target <= current_temp:
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
