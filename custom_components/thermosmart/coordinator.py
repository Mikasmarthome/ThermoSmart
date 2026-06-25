"""ThermoSmart Coordinator – Hauptkoordinator für eine Heizzone."""
from __future__ import annotations

import dataclasses
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
    TPI_COEF_INT_DEFAULT,
    TPI_COEF_EXT_DEFAULT,
)
from .device_profiles import DeviceProfile
from .learning.clock import Clock, SystemClock
from .weather_engine import WeatherEngine
from .learning_engine import LearningEngine
from .window import WindowMixin
from .season import SeasonMixin
from .trv_control import TRVControlMixin, _DispatchStats
from .maintenance import MaintenanceMixin

_LOGGER = logging.getLogger(__name__)

# valve_opening_degree recovery is retried across the first N coordinator refreshes
# to handle slow Zigbee2MQTT startup on full HA restart.
_VALVE_RESET_MAX_ATTEMPTS = 3

# LE 2.0 forecast bias application gates
_FORECAST_BIAS_MIN_TRUST = 0.35        # minimum trust to apply the °C bias correction
_FORECAST_BIAS_MIN_APPLY_C = 0.05     # noise floor – ignore sub-threshold corrections
_FORECAST_BIAS_CORRECTION_MAX_C = 2.0  # hard cap matching ForecastParameters.bias_correction_max_c

# TPI coef_int step-limiting (Phase 19D) — max change per coordinator cycle (~5 min).
# Derivation: at max 3 °C deficit, 0.10 step → 30 % duty change per cycle.  The
# full default→learned transition (|0.60 − 0.20| = 0.40) takes 4 cycles ≈ 20 min,
# which is well within a room's thermal time constant (30-120 min).
# Symmetric up/down: stepping up (more heating) is equally rate-limited to prevent
# abrupt duty increases on stale→valid recovery.
_TPI_COEF_INT_MAX_STEP_UP: float = 0.10
_TPI_COEF_INT_MAX_STEP_DOWN: float = 0.10
_TPI_COEF_INT_CLAMP_MIN: float = 0.15   # mirrors estimate_coefficients lower bound
_TPI_COEF_INT_CLAMP_MAX: float = 1.2    # mirrors estimate_coefficients upper bound

# LE 2.0 early cutoff application gates
_EARLY_CUTOFF_MIN_RESIDUAL_C = 0.15   # noise floor for residual rise (below → no cutoff)
_EARLY_CUTOFF_MAX_C = 3.0             # hard maximum (mirrors AfterheatParameters.early_cutoff_max_c)
_EARLY_CUTOFF_HOLD_TIMEOUT_SECS: float = 1800.0  # 30-min max afterheat window before forced release


class ControlAdaptationMode:
    """Derived control mode based on the two user-visible switches.

    Invariants (computed each cycle, never stored as state):
      INACTIVE           — Learning OFF + Active Control OFF: no ThermoSmart influence
      SHADOW_ONLY        — Learning ON  + Active Control OFF: observe and learn, no service calls
      DETERMINISTIC      — Learning OFF + Active Control ON:  TPI control, static defaults only
      ADAPTIVE           — Learning ON  + Active Control ON:  full adaptive LE2 control
    """
    INACTIVE      = "inactive"
    SHADOW_ONLY   = "shadow_only"
    DETERMINISTIC = "deterministic"
    ADAPTIVE      = "adaptive"

    @staticmethod
    def derive(learning_mode_on: bool, active_control_on: bool) -> str:
        if learning_mode_on and active_control_on:
            return ControlAdaptationMode.ADAPTIVE
        if learning_mode_on:
            return ControlAdaptationMode.SHADOW_ONLY
        if active_control_on:
            return ControlAdaptationMode.DETERMINISTIC
        return ControlAdaptationMode.INACTIVE


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
        *,
        clock: "Clock | None" = None,
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

        self._last_written_setpoints: dict[str, float] = {}
        # Boost tracking per zone (keyed by zone_id; used by TRVControlMixin._apply_temperature).
        # Must be initialized before the first coordinator refresh to avoid AttributeError.
        self._boost_active: dict = {}

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
        self._indoor_temp_slope: float | None = None
        self._ema_1h: float | None = None

        # Slope-basierte Fenstererkennung (genutzt von WindowMixin._check_window_slope)
        self._slope_win_ema: float | None = None
        self._slope_win_consec: int = 0
        self._slope_window_active: bool = False

        # Heizungsausfall-Erkennung
        self._heating_failure_since: datetime | None = None
        self._heating_failure_notified: bool = False

        # Debug-Logging: Modus-Tracking für Wechsel-Erkennung
        self._last_effective_mode: str | None = None

        # Restore initialization barrier: set True the first time Active Control
        # RestoreEntity calls its setter after restore.  Until then, boost is blocked.
        # Learning mode is read from config (not a RestoreEntity) → not in barrier.
        self._active_control_initialized: bool = False

        # LE 2.0 passive shadow controller (attached at setup; never controls)
        self._le2_shadow = None
        # Authoritative live decision record for the last completed cycle (Phase B1).
        # Built pre-dispatch and completed post-dispatch; consumed by compute_decision_trace_safe().
        self._last_live_decision: object = None

        # Coordinator-level decision identity (baseline, LE2-independent).
        # Stable across cycles as long as the decision type and effective target don't
        # change materially (> 0.05 °C); resets on Reload/Unload (correct: new session).
        # Uses make_decision_id() from capture.py as the single ID authority.
        self._coord_decision_id: object = None      # Optional[DecisionId]
        self._coord_decision_target: object = None   # Optional[float]
        self._coord_decision_type: object = None     # Optional[str]

        # TPI coef_int step-limiting state (transient, per zone, resets on restart).
        # _tpi_coef_int_smoothed: last used coef_int; initialized to deterministic default
        # so the first cycle starts from a known safe state regardless of LE2 predictions.
        # _tpi_duty_previous: last duty cycle for trace/audit (None before first cycle).
        self._tpi_coef_int_smoothed: float = TPI_COEF_INT_DEFAULT
        self._tpi_duty_previous: float | None = None

        # LE 2.0 Early Cutoff hold state (transient, never persisted; safe on HA restart)
        # Once early cutoff is applied, the hold maintains the reduced effective_target
        # through the coasting phase so afterheat is not disrupted by a re-heat impulse.
        self._ec_hold_active: bool = False
        self._ec_hold_cut_target: float = 0.0
        self._ec_hold_comfort_temp: float = 0.0
        self._ec_hold_period: str = ""
        self._ec_hold_started: float = 0.0   # _clock.monotonic() when hold was started
        self._ec_state: str = "inactive"
        self._ec_episode_failed: bool = False  # True after temp-falling/timeout release
        self._ec_failed_cut_target: float = 0.0  # cut_target of the failed episode

        # Injectable clock — replaced by FakeClock in simulation/tests.
        # All fachliche Zeitaufrufe in this coordinator must go through self._clock.
        # Production callers pass nothing → SystemClock; tests/simulation pass FakeClock.
        self._clock: Clock = clock if clock is not None else SystemClock()

    # ── Eigenschaften ────────────────────────────────────────────────

    def attach_le2_shadow(self, shadow) -> None:
        """Attach the passive LE 2.0 shadow controller (diagnostics only)."""
        self._le2_shadow = shadow

    def _now_local(self):
        """Return current local time derived from the injected clock.

        Uses dt_util.as_local() for timezone conversion only — not as a time source.
        All fachliche local-time calls in coordinator and its mixins must use this.
        """
        return dt_util.as_local(self._clock.now_utc())

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
        self._active_control_initialized = True
        _LOGGER.info(
            "ThermoSmart '%s': Aktive Steuerung %s",
            self.zone_name,
            "AN – Thermostat wird gesteuert" if active else "AUS – Beobachtungsmodus",
        )

    def set_mode(self, mode: str) -> None:
        if mode != self._mode and self._override is not None:
            _LOGGER.debug(
                "ThermoSmart '%s': clearing manual override due to mode change %s → %s",
                self.zone_name, self._mode, mode,
            )
            self._override = None
            self._override_schedule_period = None
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
            if self._override is not None:
                _LOGGER.debug(
                    "ThermoSmart '%s': clearing manual override due to vacation mode",
                    self.zone_name,
                )
                self._override = None
                self._override_schedule_period = None
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
            if self._override is not None:
                _LOGGER.debug(
                    "ThermoSmart '%s': clearing manual override due to summer mode",
                    self.zone_name,
                )
                self._override = None
                self._override_schedule_period = None
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
        now = self._now_local()
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
        now = self._now_local()
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
                now = self._now_local()

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
                            self._window_delay_cancel.pop(_eid, None)
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
                                    self._window_delay_cancel.pop(_eid, None)
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

    # ── Authoritative Live Decision Builder ──────────────────────────

    def _get_coord_decision_id(self, recommendation, *, ts: str) -> object:
        """Return a stable coordinator-level decision_id for the current cycle.

        Uses the canonical ``make_decision_id()`` from the capture module — the same
        function that LE2's CaptureCoordinator uses — so there is exactly one ID
        authority.  The ID is stable across successive cycles as long as the decision
        type and effective target don't change materially (>0.05 °C threshold mirrors
        CaptureCoordinator._decision_changed).  It resets on Reload/Unload, which is
        correct: a new HA session starts a new decision lineage.

        When LE2 is present, enrich_live_decision_pre_dispatch() will replace this ID
        with the LE2-generated one (which may be the same or a continuation).  Without
        LE2, this ID is authoritative for the Record, Trace and any future Outcome binding.
        """
        try:
            from .learning.runtime.capture import make_decision_id, DecisionType
        except ImportError:
            return None
        target = recommendation.get("effective_target")
        # Map coordinator state to DecisionType for ID stability domain
        if recommendation.get("preheat_status") in ("heating", "preheating"):
            dec_type = DecisionType.PREHEAT
            dtype_str = "preheat"
        elif recommendation.get("le2_boost_adjusted"):
            dec_type = DecisionType.BOOST
            dtype_str = "boost"
        else:
            dec_type = DecisionType.NORMAL
            dtype_str = "normal"
        target_f = float(target) if target is not None else None
        # A new decision opens when type or target changes materially.
        # For no-target decisions (summer, window-open, absence) the type change is enough.
        type_changed = (dtype_str != self._coord_decision_type)
        target_changed = (
            self._coord_decision_id is None
            or target_f is None
            or self._coord_decision_target is None
            or abs(target_f - self._coord_decision_target) > 0.05
        )
        if self._coord_decision_id is None or type_changed or target_changed:
            self._coord_decision_id = make_decision_id(self.zone_id, dec_type, ts)
            self._coord_decision_target = target_f
            self._coord_decision_type = dtype_str
        return self._coord_decision_id

    def _build_baseline_live_decision(self, recommendation, *, ts, mode_str):
        """Create authoritative LiveDecisionRecord from coordinator recommendation dict.

        Always called regardless of LE2 shadow presence.  LE2-specific fields
        (decision_id, onset_delay) remain None and are added by
        ``LearningShadowController.enrich_live_decision_pre_dispatch()`` when a shadow
        is attached.  Never raises — returns None on unexpected error.
        """
        try:
            from .learning.decision.contracts import LiveDecisionRecord
            boost_applied = bool(recommendation.get("le2_boost_adjusted"))
            baseline_sp = (
                recommendation.get("tpi_baseline_setpoint")
                if boost_applied
                else recommendation.get("trv_setpoint")
            )
            _tpi_diag = recommendation.get("tpi_coef_diag") or {}
            return LiveDecisionRecord(
                decision_id=self._get_coord_decision_id(recommendation, ts=ts),
                zone_id=self.zone_id,
                ts=ts,
                mode=mode_str,
                baseline_setpoint_c=(float(baseline_sp) if baseline_sp is not None else None),
                final_setpoint_c=(float(recommendation["trv_setpoint"])
                                  if recommendation.get("trv_setpoint") is not None else None),
                boost_applied=boost_applied,
                boost_candidate_c=(float(recommendation["_boost_candidate_c"])
                                   if recommendation.get("_boost_candidate_c") is not None
                                   else None),
                boost_applied_c=(float(recommendation["_boost_applied_c"])
                                 if recommendation.get("_boost_applied_c") is not None
                                 else None),
                boost_rejected_reason=recommendation.get("_boost_rejection_reason"),
                boost_evaluation_status=recommendation.get(
                    "_boost_evaluation_status", "unknown"),
                preheat_minutes=float(recommendation.get("preheat_minutes") or 0.0),
                preheat_status=recommendation.get("preheat_status"),
                early_cutoff_state=recommendation.get("early_cutoff_state"),
                heat_rate_c_per_h=_tpi_diag.get("heat_rate_c_per_h"),
                heat_rate_confidence=_tpi_diag.get("hr_confidence"),
                heat_rate_source=recommendation.get("tpi_coef_source"),
                onset_delay_min=None,       # enriched by shadow if available
                onset_delay_source=None,    # enriched by shadow if available
            )
        except Exception:
            return None

    # ── Hauptschleife ────────────────────────────────────────────────

    async def _async_update_data(self) -> dict:
        try:
            # Reset per-cycle live decision record — prevents cross-cycle staleness
            # when the shadow is absent or removed between cycles.
            self._last_live_decision = None
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

            weather_data = await self.weather_engine.async_get_data()
            self._update_summer_mode(weather_data)

            presence = self._get_presence_state()
            mode = self._effective_mode(presence)
            if self._last_effective_mode is not None and mode != self._last_effective_mode:
                if presence.get("vacation"):
                    _reason = "vacation"
                elif self._mode != HEATING_MODE_AUTO:
                    _reason = "manual"
                elif presence.get("all_away"):
                    _reason = "presence"
                else:
                    _reason = "schedule"
                _LOGGER.debug(
                    "ThermoSmart '%s': effective mode changed %s → %s reason=%s",
                    self.zone_name, self._last_effective_mode, mode, _reason,
                )
            self._last_effective_mode = mode
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

            # Derive adaptation mode once per cycle from the two user-visible switches.
            # All LE2 adaptive reads below are gated on _learning_mode_on so that
            # Learning OFF → only deterministic defaults influence real control.
            _learning_mode_on = bool(cfg.get("learning_enabled", True))
            _active_control_on = self._active_control
            _adaptation_mode = ControlAdaptationMode.derive(_learning_mode_on, _active_control_on)
            recommendation["adaptation_mode"] = _adaptation_mode

            # B2b-bootstrap: pass zone-configured device prior to the boost model so
            # it can run initial controlled trials before any adaptive evidence exists.
            _bsp = cfg.get("boost_bootstrap_prior_c")
            if _bsp is not None:
                try:
                    recommendation["boost_bootstrap_prior_c"] = float(_bsp)
                except (TypeError, ValueError):
                    pass

            # Use effective_target (weather-adjusted) for TPI/TRV calculations.
            # adjusted_target is only the display value.
            target = recommendation.get("effective_target")
            current_temp = recommendation.get("current_temp")
            if target is not None:
                # TPI-Koeffizienten: LE 2.0 authoritative (Phase 19D).
                # LE v1 path (learning_engine.get_tpi_coefficients) is kept only as
                # emergency fallback when the LE2 shadow is not yet attached.
                if _learning_mode_on and self._le2_shadow is not None:
                    # LE 2.0 is authoritative for TPI coefficients (Phase 19D).
                    # read_tpi_coefficients_safe() returns the blended *candidate* coef_int.
                    # The coordinator applies step-limiting to produce the *used* value so
                    # valid→stale and stale→valid transitions never cause abrupt duty jumps.
                    _coef_cand, coef_ext, _tpi_hl_rate, _tpi_src, _tpi_diag = (
                        self._le2_shadow.read_tpi_coefficients_safe()
                    )
                    # Step-limiting: used approaches candidate by at most one step per cycle.
                    _prev = self._tpi_coef_int_smoothed
                    _delta = _coef_cand - _prev
                    _step = max(-_TPI_COEF_INT_MAX_STEP_DOWN,
                                min(_TPI_COEF_INT_MAX_STEP_UP, _delta))
                    coef_int = max(_TPI_COEF_INT_CLAMP_MIN,
                                   min(_TPI_COEF_INT_CLAMP_MAX, _prev + _step))
                    self._tpi_coef_int_smoothed = coef_int
                    # Enrich diag with candidate/previous/transition for trace
                    _tpi_diag = dict(_tpi_diag)
                    _tpi_diag.update({
                        "coef_int_candidate": round(_coef_cand, 4),
                        "coef_int_previous": round(_prev, 4),
                        "transition_applied": abs(_step) > 1e-5,
                        "transition_reason": (
                            "step_limited" if abs(_step) < abs(_delta) - 1e-5 else "converged"
                        ),
                        "coef_int_max_step_up": _TPI_COEF_INT_MAX_STEP_UP,
                        "coef_int_max_step_down": _TPI_COEF_INT_MAX_STEP_DOWN,
                    })
                    recommendation["tpi_coef_source"] = _tpi_src
                    recommendation["tpi_hl_rate"] = _tpi_hl_rate
                    recommendation["tpi_coef_int"] = coef_int
                    recommendation["tpi_coef_ext"] = coef_ext
                    recommendation["tpi_coef_diag"] = _tpi_diag
                else:
                    # LE2 shadow not attached (disabled). Deterministic defaults only —
                    # LE v1 _heat_loss_ema must never be read for control.
                    coef_int, coef_ext = TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT
                    self._tpi_coef_int_smoothed = TPI_COEF_INT_DEFAULT  # keep in sync
                    recommendation["tpi_coef_source"] = "deterministic_baseline"
                    recommendation["tpi_hl_rate"] = None
                    recommendation["tpi_coef_int"] = coef_int
                    recommendation["tpi_coef_ext"] = coef_ext
                    recommendation["tpi_coef_diag"] = {
                        "coef_int_default": TPI_COEF_INT_DEFAULT,
                        "coef_int_candidate": TPI_COEF_INT_DEFAULT,
                        "coef_int_previous": TPI_COEF_INT_DEFAULT,
                        "coef_int_learned": None, "blend_weight": 0.0,
                        "relative_only": False, "outdoor_context_available": False,
                        "heat_rate_c_per_h": None, "heat_loss_c_per_h": None,
                        "hr_confidence": None, "hl_confidence": None,
                        "transition_applied": False,
                        "transition_reason": "no_shadow",
                        "coef_int_max_step_up": _TPI_COEF_INT_MAX_STEP_UP,
                        "coef_int_max_step_down": _TPI_COEF_INT_MAX_STEP_DOWN,
                    }
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

                recommendation["trv_setpoint"] = trv_setpoint
                # boost_offset_c / boost_factor: prediction semantics — set below from
                # _boost_candidate_c (raw LE2 proposal, pre-guard) after resolver runs.
                # applied_boost_offset_c: confirmed dispatch truth — updated post-dispatch.
                recommendation["boost_offset_c"] = 0.0
                recommendation["boost_factor"] = 1.0
                recommendation["applied_boost_offset_c"] = 0.0
                recommendation["tpi_duty_cycle"] = round(duty_cycle, 1)
                recommendation["tpi_coef_int"] = round(coef_int, 3)
                recommendation["tpi_coef_ext"] = round(coef_ext, 4)
                recommendation["tpi_valve_direct"] = has_valve
                # Mixed-zone detection: a zone combining setpoint-TRVs and direct-valve-TRVs
                # cannot safely receive a per-device boost split → boost blocked conservatively.
                _climate_eids = [e for e in cfg.get("climate_entities", []) if e]
                _has_valve_ctrl = has_valve
                _has_setpoint_ctrl = any(
                    not self._auto_valve_map.get(eid) for eid in _climate_eids)
                recommendation["zone_control_type"] = (
                    "mixed" if (_has_valve_ctrl and _has_setpoint_ctrl) else
                    "direct_valve" if _has_valve_ctrl else
                    "setpoint"
                )
                # Enrich tpi_coef_diag with duty-level trace for transition audit.
                # duty_candidate: duty that would result from using candidate coef_int directly.
                # duty_previous: duty from the previous cycle (for delta/stability verification).
                # duty_used: actual duty applied this cycle (with step-limited coef_int).
                _tpi_d = recommendation.get("tpi_coef_diag")
                if _tpi_d is not None and current_temp is not None:
                    _cand_ci = _tpi_d.get("coef_int_candidate", coef_int)
                    _duty_cand = compute_tpi(
                        target, current_temp, weather_data.get("temperature"),
                        _cand_ci, coef_ext)
                    _tpi_d["duty_previous"] = (round(self._tpi_duty_previous, 1)
                                               if self._tpi_duty_previous is not None else None)
                    _tpi_d["duty_candidate"] = round(_duty_cand, 1)
                    _tpi_d["duty_used"] = round(duty_cycle, 1)
                self._tpi_duty_previous = duty_cycle if current_temp is not None else (
                    self._tpi_duty_previous)

            # Display fallback: when no active heating target exists (summer, window-open,
            # away with adjusted_target=None), populate trv_setpoint for sensor display so
            # the sensor does not show Unknown after restart. Does not affect control logic,
            # TPI, learning, or boost tracking.
            if "trv_setpoint" not in recommendation:
                if recommendation.get("is_summer"):
                    # In summer mode the frost-protection value is always sent to the TRV.
                    # Use it directly so the sensor reflects the actually applied setpoint
                    # rather than a stale heating value from _last_written_setpoints.
                    recommendation["trv_setpoint"] = TEMP_FROST_PROTECTION
                else:
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

            # ── LE 2.0 Adaptive Boost Control ────────────────────────────
            # resolve_adaptive_boost_control() is the single authority for real boost.
            # Gate order: learning_mode_on → active_control_on
            #   → mixed_zone_check → device_unavailable_check → readiness → inner gates.
            # Runs BEFORE dispatch so the approved offset flows through device guards and
            # the single setpoint dispatch path.  Never raises into heating.
            if self._le2_shadow is not None and not effective_summer:
                # Pre-dispatch device availability: if any setpoint TRV is unavailable
                # right now, block boost for this cycle to prevent partial dispatch.
                # (Mixed/valve zones are caught by zone_control_type in the resolver.)
                _setpoint_device_unavailable = False
                if recommendation.get("zone_control_type") == "setpoint":
                    for _eid in cfg.get("climate_entities", []):
                        if not _eid:
                            continue
                        _dev_st = self.hass.states.get(_eid)
                        if _dev_st is None or _dev_st.state in ("unavailable", "unknown"):
                            _setpoint_device_unavailable = True
                            break
                recommendation["_setpoint_device_unavailable"] = _setpoint_device_unavailable
                _restore_pending = not self._active_control_initialized
                try:
                    self._le2_shadow.resolve_adaptive_boost_control(
                        recommendation,
                        learning_mode_on=_learning_mode_on,
                        active_control_on=self._active_control,
                        restore_pending=_restore_pending,
                        boost_runtime_limit=TPI_MAX_BOOST_CELSIUS,
                    )
                except Exception:
                    pass
                # B2b-3c: authoritative three-state boost emission (runs every cycle);
                # 0.0 (evaluated, no boost) is distinct from None (not evaluated).
                try:
                    self._le2_shadow.emit_boost_authority_status_safe(recommendation)
                except Exception:
                    pass
                # boost_offset_c: raw LE2 prediction (historical semantics, pre-guard).
                # boost_factor: backward-compat multiplier derived from the same prediction.
                # Neither is updated post-dispatch; applied_boost_offset_c carries dispatch truth.
                _candidate_c = float(recommendation.get("_boost_candidate_c") or 0.0)
                recommendation["boost_offset_c"] = round(_candidate_c, 3)
                try:
                    from custom_components.thermosmart.learning.models.boost import (
                        boost_offset_c_to_compat_factor)
                    recommendation["boost_factor"] = boost_offset_c_to_compat_factor(
                        _candidate_c, TPI_MAX_BOOST_CELSIUS)
                except Exception:
                    recommendation["boost_factor"] = 1.0
            # ── Authoritative Live Decision Record (Phase B1b) ────────────────────
            # Baseline record is always built from coordinator-owned data, regardless of
            # whether LE2 is attached.  The shadow optionally enriches with decision_id
            # and onset_delay from LE2 predictions.  Neither build nor enrichment may
            # raise into the heating cycle.
            _cycle_ts = self._clock.now_utc().isoformat()
            _mode_str = "control" if self._active_control else "shadow"
            _live_dec_pre = self._build_baseline_live_decision(
                recommendation, ts=_cycle_ts, mode_str=_mode_str)
            if self._le2_shadow is not None and _live_dec_pre is not None:
                try:
                    _enriched = self._le2_shadow.enrich_live_decision_pre_dispatch(
                        _live_dec_pre, recommendation)
                    if _enriched is not None:
                        _live_dec_pre = _enriched
                except Exception:
                    pass  # baseline record remains authoritative

            # ── Dispatch ──────────────────────────────────────────────────────────
            # Actor/service failures are captured in _DispatchStats and do NOT
            # propagate as UpdateFailed — other control/diagnostic data must stay
            # available even when individual TRV service calls fail.
            duty = 0.0
            _sp_stats = _DispatchStats()   # setpoint/frost dispatch stats
            _vv_stats = _DispatchStats()   # direct valve dispatch stats
            if self._active_control and not effective_summer:
                await self._async_apply_quirks(cfg)
                await self._watchdog_hvac(cfg, recommendation)
                await self._async_calibrate_trvs(cfg, recommendation)
                _sp_stats = await self._apply_temperature(cfg, recommendation)
                # Direkte Ventilsteuerung nach Setpoint-Schreiben
                duty = recommendation.get("tpi_duty_cycle", 0.0)
                if recommendation.get("window_open"):
                    duty = 0.0
                _vv_stats = await self._async_set_valve_percent(cfg, duty)
            elif self._active_control and effective_summer:
                await self._async_apply_quirks(cfg)
                await self._apply_frost_protection(cfg)
                recommendation["trv_setpoint"] = TEMP_FROST_PROTECTION

            # ── Post-dispatch dispatch truth (before LiveDecisionRecord) ─────────────
            # Compute dispatch status now so both the per-device applied truth and the
            # LiveDecisionRecord completion can use the same authoritative values.
            _disp_attempted = self._active_control and not effective_summer
            _combined = _sp_stats.merge(_vv_stats) if _disp_attempted else _DispatchStats()
            _disp_status = _combined.status if _disp_attempted else "not_attempted"
            _eff_sps = _sp_stats.effective_setpoints

            # Per-device applied boost offset — updated BEFORE LiveDecisionRecord so
            # LDR.boost_applied_c reflects real dispatch truth, not the pre-dispatch
            # approved value.
            #
            # Counterfactual formula (per device):
            #   device_applied_i = effective_with_boost_i - effective_baseline_i
            # Both sides run through the same per-device transform path (min-clamp,
            # step-snap, re-clamp) — only the input differs (baseline vs baseline+boost).
            # This correctly accounts for device-specific priors (min_temp), step
            # quantization, and profile offsets without double service calls.
            #
            # Semantics:
            #   fully_succeeded  → min(device_applied_i) across written devices ≥ 0
            #                      (conservative zone truth; absorbs clamp/step reduction).
            #   partial/failed   → 0.0 public; per-device offsets preserved for provenance.
            #   not_attempted    → 0.0 (approved-but-not-dispatched must not appear as applied; lifecycle not touched).
            if self._le2_shadow is not None and bool(recommendation.get("le2_boost_adjusted")):
                _baseline_c = float(recommendation.get("tpi_baseline_setpoint") or 0.0)
                _eff_by_eid = _sp_stats.effective_setpoints_by_entity

                def _counterfactual_offsets(eff_by_eid: dict) -> list:
                    """Compute per-device applied offset via counterfactual baseline resolution."""
                    offsets = []
                    for _eid, _eff_sp in eff_by_eid.items():
                        _st = self.hass.states.get(_eid)
                        if _st is None:
                            # State unavailable post-dispatch: fall back to zone baseline.
                            offsets.append(_eff_sp - _baseline_c)
                            continue
                        _prof = self._device_profiles.get(_eid)
                        _eff_bl = self.resolve_device_effective_setpoint(
                            _eid, _st, _baseline_c, _prof)
                        offsets.append(_eff_sp - _eff_bl)
                    return offsets

                if _disp_status == "fully_succeeded" and _eff_by_eid:
                    _dev_offsets = _counterfactual_offsets(_eff_by_eid)
                    _real_applied = max(0.0, min(_dev_offsets))
                    recommendation["_boost_applied_c"] = round(_real_applied, 4)
                    recommendation["_boost_per_device_applied_c"] = [
                        round(o, 4) for o in _dev_offsets]
                    recommendation["applied_boost_offset_c"] = round(_real_applied, 3)
                    # boost_offset_c (prediction) and boost_factor are not updated post-dispatch.
                elif _disp_status == "not_attempted":
                    # Dispatch path skipped entirely (summer mode / active-control exclusion):
                    # approved-but-not-dispatched must not surface as applied truth.
                    recommendation["_boost_applied_c"] = 0.0
                    recommendation["applied_boost_offset_c"] = 0.0
                    recommendation["_boost_per_device_applied_c"] = []
                else:
                    # Partial or failed dispatch: zero applied truth, preserve per-device data.
                    if _eff_by_eid and _disp_status == "partially_succeeded":
                        _dev_offsets = _counterfactual_offsets(_eff_by_eid)
                        recommendation["_boost_per_device_applied_c"] = [
                            round(o, 4) for o in _dev_offsets]
                    recommendation["_boost_applied_c"] = 0.0
                    recommendation["applied_boost_offset_c"] = 0.0
                    # Hard-release lifecycle so next cycle re-evaluates from baseline.
                    try:
                        self._le2_shadow.invalidate_boost_after_failed_dispatch_safe(
                            _disp_status)
                    except Exception:
                        pass

            # ── Complete LiveDecisionRecord with dispatch truth ────────────────────
            if _live_dec_pre is not None:
                _disp_path = (
                    ("direct_valve" if recommendation.get("tpi_valve_direct") else "setpoint")
                    if _disp_attempted else "none"
                )
                _outcome_elig = _disp_status == "fully_succeeded"
                _outcome_rel = (
                    "full" if _disp_status == "fully_succeeded" else
                    "partial" if _disp_status == "partially_succeeded" else
                    "none"
                )
                _eff_min = float(min(_eff_sps)) if _eff_sps else None
                _eff_max = float(max(_eff_sps)) if _eff_sps else None
                try:
                    self._last_live_decision = dataclasses.replace(
                        _live_dec_pre,
                        dispatch_attempted=_disp_attempted,
                        dispatch_path=_disp_path,
                        dispatch_setpoint_c=(
                            float(recommendation["trv_setpoint"])
                            if recommendation.get("trv_setpoint") is not None else None),
                        dispatch_effective_setpoint_min_c=_eff_min,
                        dispatch_effective_setpoint_max_c=_eff_max,
                        dispatch_effective_setpoints=tuple(float(x) for x in _eff_sps),
                        dispatch_duty_pct=(float(duty) if _disp_path == "direct_valve" else None),
                        dispatch_succeeded=_outcome_elig,
                        dispatch_status=_disp_status,
                        dispatch_targets_total=_combined.targets_total,
                        dispatch_targets_succeeded=_combined.targets_succeeded,
                        dispatch_targets_failed=_combined.targets_failed,
                        dispatch_failure_reasons=tuple(_combined.failure_reasons),
                        dispatch_ts=_cycle_ts if _disp_attempted else None,
                        outcome_eligible=_outcome_elig,
                        outcome_reliability=_outcome_rel,
                    )
                except Exception:
                    self._last_live_decision = None
            else:
                self._last_live_decision = None

            # ── Derive trace from live record ─────────────────────────────────────
            # Production coordinator always passes live_record; legacy resolver path
            # is never triggered.  When no live record, clear shadow's stale trace.
            if self._le2_shadow is not None:
                if self._last_live_decision is not None:
                    try:
                        self._le2_shadow.compute_decision_trace_safe(
                            recommendation,
                            active_control=self._active_control,
                            live_record=self._last_live_decision)
                    except Exception:
                        pass
                else:
                    self._le2_shadow.clear_last_trace()

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

            _ct = recommendation.get("current_temp")
            _et = recommendation.get("effective_target")
            _at = recommendation.get("adjusted_target")
            _ts = recommendation.get("trv_setpoint")
            _LOGGER.debug(
                "ThermoSmart '%s': cycle summary mode=%s current=%s°C"
                " target=%s°C adjusted=%s°C trv=%s°C duty=%s%% preheat=%dmin"
                " window=%s summer=%s active=%s",
                self.zone_name,
                mode,
                f"{_ct:.1f}" if _ct is not None else "n/a",
                f"{_et:.1f}" if _et is not None else "n/a",
                f"{_at:.1f}" if _at is not None else "n/a",
                f"{_ts:.1f}" if _ts is not None else "n/a",
                f"{recommendation.get('tpi_duty_cycle', 0.0):.0f}",
                recommendation.get("preheat_minutes", 0),
                recommendation.get("window_open", False),
                recommendation.get("is_summer", False),
                self._active_control,
            )

            # ── LE 2.0 passive shadow observation ───────────────────────
            # Runs AFTER the control decision is fully determined and applied.
            # Only when Learning Mode is ON — Learning OFF → no shadow observation.
            # Purely passive (diagnostics only); doubly guarded so a learning
            # failure can never become a heating failure or an UpdateFailed.
            if _learning_mode_on and self._le2_shadow is not None:
                try:
                    _sched_time = _sched_temp = None
                    _comfort_min = self._minutes_until_next_comfort(cfg)
                    if _comfort_min is not None:
                        _sched_time = (self._clock.now_utc()
                                       + timedelta(minutes=_comfort_min)).isoformat()
                        _sched_temp = cfg.get("comfort_temp", 21.0)
                    # B2b-4e: live per-device availability for the post-reach boost observer
                    # (binding/entity -> available|unavailable|unknown). No entity ids leave
                    # the learning runtime; they are hashed into anonymous slots internally.
                    _dev_states: dict[str, str] = {}
                    for _eid in cfg.get("climate_entities", []):
                        if not _eid:
                            continue
                        _st = self.hass.states.get(_eid)
                        if _st is None or _st.state == "unavailable":
                            _dev_states[_eid] = "unavailable"
                        elif _st.state == "unknown":
                            _dev_states[_eid] = "unknown"
                        else:
                            _dev_states[_eid] = "available"
                    recommendation["device_states"] = _dev_states
                    self._le2_shadow.observe_safe(
                        recommendation, weather=weather_data,
                        schedule_comfort_time_utc=_sched_time,
                        schedule_comfort_temperature_c=_sched_temp,
                        heating_failure=self._heating_failure_since is not None,
                        live_record=self._last_live_decision)
                except Exception:  # never let LE 2.0 affect the heating path
                    pass

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
        _learning_mode_on = bool(cfg.get("learning_enabled", True))
        current_temp = self._read_avg_sensor(cfg.get("temp_sensors", []))
        if current_temp is None:
            current_temp = self._read_trv_avg_temp(cfg.get("climate_entities", []))

        now = self._now_local()
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

        # ── Preheat duration: LE 2.0 is the sole adaptive source ────────────
        # LE v1 is NEVER read for preheat; the frozen LE v1 store is historical.
        # When LE 2.0 has no evidence, or Learning Mode is OFF, the deterministic
        # baseline is used. Learning OFF → no adaptive preheat, no onset delay.
        # Onset delay is separate from HeatRate learning (never folded in).
        from .learning.runtime.ha_integration import LearningShadowController
        if _learning_mode_on and self._le2_shadow is not None:
            preheat_minutes, preheat_status = self._le2_shadow.read_preheat_minutes_safe(
                current_temp, comfort_temp,
                outdoor_temp=weather_data.get("temperature") if weather_data else None,
            )
            _onset_delay_min, _onset_delay_status = self._le2_shadow.read_onset_delay_safe()
        else:
            # No shadow: deterministic baseline (1.5 °C/h prior; onset prior 5 min).
            preheat_minutes, preheat_status = (
                LearningShadowController.compute_deterministic_preheat_baseline(
                    current_temp, comfort_temp
                )
            )
            _onset_delay_min = LearningShadowController._ONSET_DELAY_PRIOR_MIN
            _onset_delay_status = "cold_start_prior"

        preheat_active = False
        if mode == HEATING_MODE_AUTO and preheat_minutes > 0 and base_target < comfort_temp:
            mins_until_comfort = self._minutes_until_next_comfort(cfg)
            if mins_until_comfort is not None and 0 < mins_until_comfort <= preheat_minutes:
                base_target = comfort_temp
                preheat_active = True
                _LOGGER.debug(
                    "ThermoSmart '%s': Vorheizen – %d min bis Komfortphase, %.1f min Aufheizzeit [%s]",
                    self.zone_name, mins_until_comfort, preheat_minutes, preheat_status,
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

        # Hard-release any active Early Cutoff hold on window open or manual override:
        # both are structural safety events that take absolute priority.
        if self._ec_hold_active and (window_open or override is not None):
            self._ec_hold_active = False
            self._ec_episode_failed = False
            self._ec_state = (
                "released_window_open" if window_open else "released_manual_override"
            )

        biased_suppression = 1.0
        # LE 2.0 forecast state – defaults represent "no evidence" (not "full trust")
        le2_forecast_trust = 0.0
        le2_forecast_bias_c = 0.0
        le2_forecast_trust_status = "not_available"
        le2_forecast_used = False
        # LE 2.0 early cutoff state
        le2_early_cutoff_c = 0.0
        le2_early_cutoff_status = "not_available"
        le2_early_cutoff_applied = False

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
            # Guard: a negative weather offset must not pull the effective target
            # below the configured comfort temperature while the room hasn't
            # reached it yet — otherwise the corrected target can converge with
            # current_temp and stall active heating during an active comfort
            # window. Positive offsets (boost) remain unaffected; negative
            # offsets are still allowed once current_temp >= base_target.
            if (
                mode == HEATING_MODE_AUTO
                and base_target >= comfort_temp
                and weather_offset < 0
                and current_temp is not None
                and current_temp < base_target
            ):
                _LOGGER.debug(
                    "ThermoSmart '%s': negative weather offset %.1f°C suppressed – "
                    "comfort target %.1f°C not yet reached (current=%.1f°C)",
                    self.zone_name, weather_offset, base_target, current_temp,
                )
                weather_offset = 0.0
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
                    # LE 2.0: read FORECAST_TRUST and FORECAST_BIAS separately via Read Gate.
                    # Trust gates suppression strength; Bias corrects the °C target independently.
                    # Only bias reads when trust is "valid" (real evidence, not cold-start/stale).
                    # Learning OFF → static trust=1.0 / bias=0.0 (no LE2 values in control).
                    if _learning_mode_on and self._le2_shadow is not None:
                        le2_forecast_trust, le2_forecast_trust_status = (
                            self._le2_shadow.read_forecast_trust_safe())
                        if le2_forecast_trust_status == "valid":
                            le2_forecast_bias_c = self._le2_shadow.read_forecast_bias_safe()
                            le2_forecast_used = True
                    biased_suppression = 1.0 - (1.0 - raw_suppression) * le2_forecast_trust
                    # Bias correction: only when trust is sufficient and correction clears noise floor
                    if (le2_forecast_trust >= _FORECAST_BIAS_MIN_TRUST
                            and abs(le2_forecast_bias_c) >= _FORECAST_BIAS_MIN_APPLY_C):
                        _correction = max(-_FORECAST_BIAS_CORRECTION_MAX_C,
                                          min(_FORECAST_BIAS_CORRECTION_MAX_C, le2_forecast_bias_c))
                        raw_target = round(raw_target + _correction, 1)

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

                    if biased_suppression < 0.95:
                        _LOGGER.debug(
                            "ThermoSmart '%s': forecast/weather correction active "
                            "raw_target=%.1f°C → effective=%.1f°C correction=%.1f°C suppression=%.0f%%",
                            self.zone_name, raw_target, effective_target,
                            effective_target - raw_target,
                            (1.0 - biased_suppression) * 100,
                        )

            else:
                effective_target = raw_target

            # ── LE 2.0 Early Cutoff Lifecycle ──────────────────────────────────
            # Applies in BOTH preheat and active comfort-window branches; never in
            # the night/transition branch and never in non-AUTO modes.
            # adjusted_target (user-visible schedule temp) is NEVER changed here;
            # only effective_target (internal heating drive) is reduced.
            #
            # Hold semantics: once a cutoff is applied, _ec_hold_active=True maintains
            # the reduced effective_target through the coasting phase. Without a hold,
            # effective_target would revert to comfort_temp as soon as current_temp
            # reached the cut threshold, causing a re-heat impulse that defeats the
            # purpose of the cutoff and leads to overshoot.
            #
            # Structural safety guards (guaranteed by outer if/elif structure):
            #   • window_open: released above; we are inside `elif not window_open:`
            #   • override: released above; override=None guaranteed here
            # Additional guards in _ec_eligible and the state machine below.
            if mode == HEATING_MODE_AUTO:
                _heating_to_target = preheat_active or base_target >= comfort_temp
                _current_period = self._current_schedule_period()
                # Require explicit trend evidence: slope=None (no measurement) must NOT
                # allow a new hold — unavailable trend means unavailable evidence, not
                # "neutral".  Flat-or-rising (>= 0.0) with a real measurement is fine.
                _temp_rising = (self._indoor_temp_slope is not None
                                and self._indoor_temp_slope >= 0.0)
                _ec_eligible = (
                    _learning_mode_on
                    and _heating_to_target
                    and current_temp is not None
                    and self._le2_shadow is not None
                )
                # Regime gate: new hold only when the LE 2.0 regime classifier confirms
                # active heating.  UNKNOWN / AFTERHEAT / COOLING / None → no new hold.
                # "keine zweite parallele Regime-Heuristik": regime comes exclusively
                # from the shadow runtime pipeline (ThermalRegimeClassifier), never from
                # a coordinator-local heuristic.
                # Learning OFF → regime is never used (ec_eligible is already False).
                _ec_regime = (self._le2_shadow.read_regime_safe()
                              if (_learning_mode_on and self._le2_shadow is not None) else None)

                # Episode change: different comfort target or schedule period → reset.
                if (self._ec_hold_active or self._ec_episode_failed) and (
                    self._ec_hold_comfort_temp != comfort_temp
                    or self._ec_hold_period != _current_period
                ):
                    self._ec_hold_active = False
                    self._ec_episode_failed = False
                    self._ec_failed_cut_target = 0.0
                    self._ec_state = "inactive"

                if self._ec_hold_active:
                    # ── Validate ongoing hold ─────────────────────────────────
                    # Hold is governed by episode binding, temperature evidence and
                    # safety gates.  A stale/missing prediction after hold-start must
                    # NOT trigger immediate reheat (Prediction quality is a pre-start
                    # gate only; the applied cut threshold is frozen for the duration).
                    _hold_age = self._clock.monotonic() - self._ec_hold_started
                    _release = None

                    if not _ec_eligible:
                        _release = "released_ineligible"
                    elif current_temp >= comfort_temp:
                        _release = "released_target_reached"
                    elif _hold_age > _EARLY_CUTOFF_HOLD_TIMEOUT_SECS:
                        _release = "released_timeout"
                    elif (current_temp < self._ec_hold_cut_target
                          and self._indoor_temp_slope is not None
                          and self._indoor_temp_slope < 0.0):
                        # Below cut threshold AND actively falling → re-heat needed
                        _release = "released_temperature_falling"
                    # No prediction re-validation here: frozen threshold, real evidence.

                    if _release is not None:
                        self._ec_hold_active = False
                        self._ec_state = _release
                        if _release in ("released_temperature_falling", "released_timeout"):
                            self._ec_episode_failed = True
                            self._ec_failed_cut_target = self._ec_hold_cut_target
                        le2_early_cutoff_applied = False
                    else:
                        # Hold continues: maintain frozen cut target
                        effective_target = max(self._ec_hold_cut_target, night_temp_cfg)
                        le2_early_cutoff_applied = True
                        le2_early_cutoff_c = round(
                            comfort_temp - self._ec_hold_cut_target, 1)
                        self._ec_state = (
                            "coasting_hold"
                            if current_temp >= self._ec_hold_cut_target
                            else "cutoff_applied"
                        )

                elif (_ec_eligible and not self._ec_episode_failed and _temp_rising
                      and _ec_regime == "active_heating"):
                    # ── Evaluate new cutoff (regime gate: active_heating only) ─
                    le2_early_cutoff_c, le2_early_cutoff_status = (
                        self._le2_shadow.read_early_cutoff_safe())
                    if le2_early_cutoff_c >= _EARLY_CUTOFF_MIN_RESIDUAL_C:
                        capped_c = min(le2_early_cutoff_c, _EARLY_CUTOFF_MAX_C)
                        cut_target = round(effective_target - capped_c, 1)
                        if cut_target > current_temp:
                            effective_target = max(cut_target, night_temp_cfg)
                            le2_early_cutoff_applied = True
                            self._ec_hold_active = True
                            self._ec_hold_cut_target = effective_target
                            self._ec_hold_comfort_temp = comfort_temp
                            self._ec_hold_period = _current_period
                            self._ec_hold_started = self._clock.monotonic()
                            self._ec_state = "cutoff_applied"
                    else:
                        self._ec_state = "eligible"
                else:
                    # Recovery: room dropped well below the failed cut target, is now
                    # rising, AND the ThermalRegimeClassifier confirms "active_heating".
                    # Passive warming (solar, neighbours, internal loads) keeps the regime
                    # at UNKNOWN / AFTERHEAT / COOLING and must NOT reset episode_failed.
                    if (self._ec_episode_failed
                            and self._indoor_temp_slope is not None
                            and self._indoor_temp_slope >= 0.0
                            and current_temp is not None
                            and current_temp < self._ec_failed_cut_target - 1.0
                            and _ec_regime == "active_heating"):
                        self._ec_episode_failed = False
                        self._ec_failed_cut_target = 0.0
                    self._ec_state = (
                        "inactive" if not self._ec_episode_failed else "episode_failed"
                    )
            else:
                # Non-AUTO mode: release any active hold
                if self._ec_hold_active:
                    self._ec_hold_active = False
                    self._ec_state = "released_mode_change"
                    self._ec_episode_failed = False

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

        # Phase 19A-B: combined learning confidence from LE 2.0 only (display value).
        # Guarded: a learning failure must never become a heating failure -> neutral 0.0.
        learning_confidence = 0.0
        if self._le2_shadow is not None:
            try:
                learning_confidence = float(self._le2_shadow.confidence_display())
            except Exception:
                learning_confidence = 0.0

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
            "preheat_status": preheat_status,
            # Onset delay and baseline for DecisionTrace (computed once above)
            "onset_delay_min": _onset_delay_min,
            "onset_delay_status": _onset_delay_status,
            "preheat_baseline_minutes": LearningShadowController.compute_deterministic_preheat_baseline(
                current_temp, comfort_temp)[0],
            "temperature_gap_c": ((comfort_temp - current_temp)
                                  if current_temp is not None else None),
            "forecast_suppression": suppression_pct,
            "forecast_trust": le2_forecast_trust,
            "forecast_bias_c": le2_forecast_bias_c,
            "forecast_used": le2_forecast_used,
            "forecast_trust_status": le2_forecast_trust_status,
            "early_cutoff_applied": le2_early_cutoff_applied,
            "early_cutoff_c": le2_early_cutoff_c,
            "early_cutoff_status": le2_early_cutoff_status,
            "early_cutoff_state": self._ec_state,
            "early_cutoff_hold_active": self._ec_hold_active,
            "learning_confidence": learning_confidence,
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

        if is_heating_commanded and (slope is None or slope < HEATING_FAILURE_SLOPE_THRESH):
            if self._heating_failure_since is None:
                self._heating_failure_since = self._clock.now_utc()
                _LOGGER.debug(
                    "ThermoSmart '%s': Möglicher Heizungsausfall – "
                    "Temp fällt trotz Heizbefehl (SP=%.1f°C, Ist=%.1f°C, Slope=%.4f°C/min)",
                    self.zone_name, trv_setpoint, current_temp, slope,
                )

            elapsed_min = (self._clock.now_utc() - self._heating_failure_since).total_seconds() / 60
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
        now = self._now_local()
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
