"""Window open/close detection – ThermoSmart WindowMixin.

Zwei Erkennungsstrategien:
  1. Sensor-basiert:  Fenstersensor (binary_sensor) mit konfigurierbaren Delays.
  2. Slope-basiert:   Automatische Erkennung aus Temperaturverlauf – kein Sensor nötig.
     Aktiviert wenn keine window_sensors konfiguriert sind.
     Logik: EMA-geglätteter Slope < –WINDOW_SLOPE_THRESHOLD für MIN_POINTS Messpunkte
     → Fenster offen; Slope ≥ 0 → Fenster geschlossen.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from homeassistant.util import dt as dt_util

from .const import (
    WINDOW_SLOPE_EMA_ALPHA,
    WINDOW_SLOPE_MIN_POINTS,
    WINDOW_SLOPE_THRESHOLD,
)

_LOGGER = logging.getLogger(__name__)


class WindowMixin:
    """Fenstererkennung mit konfigurierbaren Verzögerungen + Slope-Erkennung."""

    def _check_window_open(self, cfg: dict, current_temp: float | None = None) -> bool:
        now = dt_util.now()
        open_delay = timedelta(minutes=cfg.get("window_open_delay", 5))
        close_delay = timedelta(minutes=cfg.get("window_close_delay", 2))

        sensor_detected = False
        has_sensors = bool(cfg.get("window_sensors"))

        for ws_id in cfg.get("window_sensors", []):
            if not ws_id:
                continue
            ws = self.hass.states.get(ws_id)
            if ws is None:
                continue

            if ws.state == "on":
                if ws_id not in self._window_open_at:
                    self._window_open_at[ws_id] = now - open_delay
                if now - self._window_open_at[ws_id] >= open_delay:
                    sensor_detected = True
            else:
                if ws_id in self._window_close_at:
                    if now - self._window_close_at[ws_id] < close_delay:
                        sensor_detected = True
                    else:
                        self._window_close_at.pop(ws_id, None)
                self._window_open_at.pop(ws_id, None)

        # Slope-basierte Erkennung nur wenn kein Sensor konfiguriert
        if not has_sensors:
            slope_detected = self._check_window_slope(current_temp)
            return slope_detected

        return sensor_detected

    def _check_window_slope(self, current_temp: float | None) -> bool:
        """Slope-basierte Fenstererkennung ohne Sensor.

        Erkennt einen rapiden Temperaturabfall der typischerweise durch ein
        geöffnetes Fenster verursacht wird. Nutzt den bereits im Coordinator
        berechneten _indoor_temp_slope (°C/min).

        Grenzwert: WINDOW_SLOPE_THRESHOLD = 0.06°C/min = 3.6°C/h
        Normal-Abkühlung (Heizung aus): 0.01–0.03°C/min
        Fenster offen: 0.05–0.15°C/min → klar erkennbar
        """
        if current_temp is None:
            return getattr(self, "_slope_window_active", False)

        slope_now = getattr(self, "_indoor_temp_slope", 0.0)

        # EMA der Slope aktualisieren
        if self._slope_win_ema is None:
            self._slope_win_ema = slope_now
        else:
            self._slope_win_ema = round(
                WINDOW_SLOPE_EMA_ALPHA * slope_now
                + (1.0 - WINDOW_SLOPE_EMA_ALPHA) * self._slope_win_ema,
                5,
            )

        ema = self._slope_win_ema

        # Zähler erhöhen wenn Slope unterhalb Schwellwert
        if ema < -WINDOW_SLOPE_THRESHOLD:
            self._slope_win_consec += 1
        else:
            self._slope_win_consec = max(0, self._slope_win_consec - 1)

        # Fenster öffnen: genug konsekutive Detektionen
        if self._slope_win_consec >= WINDOW_SLOPE_MIN_POINTS and not self._slope_window_active:
            _LOGGER.info(
                "ThermoSmart '%s': Fenster-Slope-Erkennung ausgelöst "
                "(EMA-Slope=%.4f°C/min = %.1f°C/h, Schwelle=%.3f°C/min)",
                self.zone_name,
                ema,
                ema * 60.0,
                -WINDOW_SLOPE_THRESHOLD,
            )
            self._slope_window_active = True

        # Fenster schließen: Slope positiv oder neutral
        if ema >= 0.0 and self._slope_window_active:
            _LOGGER.info(
                "ThermoSmart '%s': Slope-Fenstererkennung – Fenster wieder geschlossen "
                "(EMA-Slope=%.4f°C/min)",
                self.zone_name,
                ema,
            )
            self._slope_window_active = False
            self._slope_win_consec = 0

        return self._slope_window_active
