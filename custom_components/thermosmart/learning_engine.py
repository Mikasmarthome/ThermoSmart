"""LearningEngine – Multi-Faktor Thermalmodell für ThermoSmart.

Das System lernt wie sich das Haus verhält unter Berücksichtigung von:
  - Innentemperatur & Zieltemperatur
  - Außentemperatur (primärer Faktor)
  - Windgeschwindigkeit (erhöht Wärmeverlust)
  - Sonneneinstrahlung (reduziert Heizbedarf)
  - Innenluftfeuchtigkeit (beeinflusst Wärmewahrnehmung)
  - Außenluftfeuchtigkeit (Feuchte erhöht gefühlte Kälte)
  - Tageszeit & Wochentag (Nutzungsgewohnheiten)

Zwei getrennte Lernspuren:
  1. Zeitplan-Lernen: Was für eine Temperatur will man wann?
     → zeitbasierte Gewichtung, gilt täglich
  2. Thermisches Lernen: Wie schnell heizt das Haus auf?
     → Multi-Faktor-Ähnlichkeit der Außenbedingungen
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    STORAGE_VERSION,
    STORAGE_KEY,
    LEARNING_MIN_SAMPLES,
    PREHEAT_MAX_MINUTES,
    PREHEAT_MIN_DELTA,
    HEATING_MODE_AUTO,
    CONF_VACATION_TEMP,
    CONF_ECO_TEMP,
    TEMP_ECO,
    CONF_SCHEDULE_ENABLED,
    CONF_SCHED_WD_MORNING, CONF_SCHED_WD_NIGHT,
    CONF_SCHED_WE_MORNING, CONF_SCHED_WE_NIGHT,
    FORECAST_EVAL_HOURS,
    FORECAST_BIAS_MIN,
    FORECAST_BIAS_MAX,
    FORECAST_BIAS_LEARNING_RATE,
)

_LOGGER = logging.getLogger(__name__)


# ── Gewichtungsfunktionen ───────────────────────────────────────────────────

def _time_weight(obs_ts: str, now: datetime, halflife_days: float = 180) -> float:
    """Einfache Zeitgewichtung für Zeitplan-Lernen.
    Halbwertszeit 180 Tage – alte Gewohnheiten zählen weiter.
    """
    try:
        age_days = max((now - datetime.fromisoformat(obs_ts)).total_seconds() / 86400, 0)
    except (ValueError, TypeError):
        return 0.0
    return math.exp(-age_days / halflife_days)



def _thermal_weight(obs: dict, conditions: dict, now: datetime) -> float:
    """Multi-Faktor Gewichtung für thermisches Lernen.

    Fragt: 'Wie ähnlich waren die Außenbedingungen?'
    Berücksichtigt alle verfügbaren Faktoren:
      - Außentemperatur (σ=5°C)    – wichtigster Faktor
      - Windgeschwindigkeit (σ=4)  – erhöht Wärmeverlust
      - Sonneneinstrahlung (σ=200) – reduziert Heizbedarf
      - Außenluftfeuchtigkeit (σ=15%) – feuchte Kälte kühlt stärker
    """
    try:
        obs_dt = datetime.fromisoformat(obs["ts"])
    except (ValueError, TypeError, KeyError):
        return 0.0

    age_days = max((now - obs_dt).total_seconds() / 86400, 0)

    # ── Außentemperatur (Pflichtfeld) ──────────────────────────────────
    curr_outdoor = conditions.get("outdoor_temp") or 10.0
    obs_outdoor = obs.get("outdoor_temp") or curr_outdoor
    temp_sim = math.exp(-((obs_outdoor - curr_outdoor) / 5.0) ** 2)
    if temp_sim < 0.02:
        return 0.0  # Zu unterschiedliche Temperatur → irrelevant

    # ── Wind (optional) ──────────────────────────────────────────────
    curr_wind = conditions.get("wind_speed") or 0.0
    obs_wind = obs.get("wind_speed") or 0.0
    if curr_wind > 0 or obs_wind > 0:
        wind_sim = math.exp(-((obs_wind - curr_wind) / 4.0) ** 2)
    else:
        wind_sim = 1.0

    # ── Sonneneinstrahlung (optional) ─────────────────────────────────
    curr_solar = conditions.get("solar_radiation") or 0.0
    obs_solar = obs.get("solar_radiation") or 0.0
    if curr_solar > 50 or obs_solar > 50:
        solar_sim = math.exp(-((obs_solar - curr_solar) / 200.0) ** 2)
    else:
        solar_sim = 1.0

    # ── Außenluftfeuchtigkeit (optional) ─────────────────────────────
    curr_hum_out = conditions.get("outdoor_humidity") or 0.0
    obs_hum_out = obs.get("outdoor_humidity") or 0.0
    if curr_hum_out > 0 or obs_hum_out > 0:
        hum_out_sim = math.exp(-((obs_hum_out - curr_hum_out) / 15.0) ** 2)
    else:
        hum_out_sim = 1.0

    # ── Aktualität (sehr sanft, 2 Jahre Halbwertszeit) ────────────────
    recency = math.exp(-age_days / 730)

    weight = temp_sim * wind_sim * solar_sim * hum_out_sim * recency
    return round(weight, 6)


# ── Persistenz mit Migrationspfad ───────────────────────────────────────────

class ThermoSmartStore(Store):
    """Store mit Migrationspfad für die Lerndaten.

    Die Lerndaten sind über eine ganze Heizsaison gewachsen und dürfen bei einer
    Formatänderung nicht verloren gehen. Wird STORAGE_VERSION erhöht, ruft HA
    diese Funktion mit den alten Daten auf – hier wird Schritt für Schritt auf
    das aktuelle Format migriert statt die Datei zu verwerfen.
    """

    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: dict
    ) -> dict:
        # Version 1 → 2: künftige Migration hier ergänzen, z.B.
        #   if old_major_version < 2:
        #       old_data = _migrate_v1_to_v2(old_data)
        return old_data


# ── Hauptklasse ─────────────────────────────────────────────────────────────

class LearningEngine:
    """Multi-Faktor Lern-Engine für ThermoSmart.

    Beobachtungs-Format (alles optional außer ts/target/indoor_temp):
    {
      "ts": ISO-Zeitstempel,
      "hour": int, "minute": int, "weekday": int,
      "target": float,           # Zieltemperatur
      "indoor_temp": float,      # Innentemperatur
      "indoor_humidity": float,  # Innenluftfeuchte (%)
      "outdoor_temp": float,     # Außentemperatur
      "outdoor_humidity": float, # Außenluftfeuchte (%)
      "wind_speed": float,       # Windgeschwindigkeit
      "solar_radiation": float,  # Sonneneinstrahlung (W/m²)
      "delta": float,            # Ziel - Ist
      "heat_rate": float         # °C/min Aufheizrate (wenn messbar)
    }
    """

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        self._zone_enabled: dict[str, bool] = {}  # zone_id → Lernmodus an/aus
        self._store: Store = ThermoSmartStore(hass, STORAGE_VERSION, STORAGE_KEY)
        self._observations: dict[str, list[dict]] = defaultdict(list)
        self._trv_observations: dict[str, list[dict]] = defaultdict(list)
        self._window_cooling_obs: dict[str, list[dict]] = defaultdict(list)
        self._last_temp: dict[str, tuple[datetime, float]] = {}
        self._last_idle_temp: dict[str, tuple[datetime, float]] = {}
        self._last_trv_snapshot: dict[str, dict] = {}
        self._confidence: dict[str, float] = {}
        self._boost_factors: dict[str, float] = {}
        self._forecast_bias: dict[str, float] = {}
        self._forecast_decisions: dict[str, list[dict]] = defaultdict(list)
        # EMA der Wärmeverlustrate (°C/min) pro Zone – schnellere Aktualisierung als simple avg
        self._heat_loss_ema: dict[str, float] = {}
        # Outcome-Scoring: aktive Heizsitzungen + abgeschlossene Ergebnisse
        self._heating_sessions: dict[str, dict] = {}
        self._outcome_log: dict[str, list[dict]] = defaultdict(list)

    # ── Persistenz ──────────────────────────────────────────────────────

    async def async_load(self) -> None:
        stored = await self._store.async_load()
        if stored and isinstance(stored, dict):
            for zone_id, obs_list in stored.get("observations", {}).items():
                self._observations[zone_id] = obs_list
            for zone_id, obs_list in stored.get("trv_observations", {}).items():
                self._trv_observations[zone_id] = obs_list
            for zone_id, obs_list in stored.get("window_cooling_obs", {}).items():
                self._window_cooling_obs[zone_id] = obs_list
            self._boost_factors = stored.get("boost_factors", {})
            self._forecast_bias = stored.get("forecast_bias", {})
            self._heat_loss_ema = stored.get("heat_loss_ema", {})
            for zone_id, entries in stored.get("outcome_log", {}).items():
                self._outcome_log[zone_id] = entries
            total = sum(len(v) for v in self._observations.values())
            trv_total = sum(len(v) for v in self._trv_observations.values())
            _LOGGER.info(
                "LearningEngine: %d Zonen, %d Temp-Beobachtungen, %d TRV-Beobachtungen geladen",
                len(self._observations), total, trv_total,
            )
        self._rebuild_confidence()

    async def async_save(self) -> None:
        self._prune_old_observations()
        await self._store.async_save({
            "observations": dict(self._observations),
            "trv_observations": dict(self._trv_observations),
            "window_cooling_obs": dict(self._window_cooling_obs),
            "boost_factors": self._boost_factors,
            "forecast_bias": self._forecast_bias,
            "heat_loss_ema": self._heat_loss_ema,
            "outcome_log": dict(self._outcome_log),
        })

    def _prune_old_observations(
        self, max_age_days: int = 1095, max_per_zone: int = 5000
    ) -> None:
        """Entfernt Beobachtungen die älter als max_age_days sind (Standard: 3 Jahre).

        Beobachtungen älter als 3 Jahre haben durch Zeitgewichtung (HWZ 180 Tage)
        weniger als 0.3% Einfluss und können sicher entfernt werden.
        Zusätzlich werden pro Zone und Spur max. max_per_zone Einträge behalten –
        ein reines Sicherheitsnetz, das im Normalbetrieb nicht greift.
        Läuft nur beim Speichern – kein Performance-Einfluss auf den Update-Zyklus.
        """
        cutoff = dt_util.now() - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat()

        for store in (self._observations, self._trv_observations, self._window_cooling_obs):
            for zone_id in list(store.keys()):
                before = len(store[zone_id])
                store[zone_id] = [
                    o for o in store[zone_id]
                    if o.get("ts", "") >= cutoff_iso
                ]
                if len(store[zone_id]) > max_per_zone:
                    store[zone_id] = store[zone_id][-max_per_zone:]
                removed = before - len(store[zone_id])
                if removed > 0:
                    _LOGGER.debug(
                        "LearningEngine [%s]: %d Beobachtungen entfernt (Alter>%d Tage oder Cap>%d)",
                        zone_id, removed, max_age_days, max_per_zone,
                    )

    # ── Bereinigung ──────────────────────────────────────────────────────

    def prune_orphaned_zones(self, active_zone_ids: set[str]) -> None:
        """Entfernt Lerndaten für Zonen die nicht mehr in den Config-Entries existieren.

        Wird beim Start einmalig aufgerufen – bereinigt alte Test-Zonen und
        umbenannte Einträge automatisch und speichert die Datei sofort.
        """
        if not active_zone_ids:
            return  # Sicherheit: nie alles löschen wenn keine aktiven Zonen

        removed: list[str] = []
        for store in (
            self._observations,
            self._trv_observations,
            self._window_cooling_obs,
            self._outcome_log,
            self._forecast_decisions,
        ):
            for zid in list(store.keys()):
                if zid not in active_zone_ids:
                    del store[zid]
                    removed.append(zid)

        for d in (self._boost_factors, self._forecast_bias, self._heat_loss_ema):
            for zid in list(d.keys()):
                if zid not in active_zone_ids:
                    del d[zid]

        if removed:
            unique = sorted(set(removed))
            _LOGGER.info(
                "LearningEngine: %d veraltete Zone(n) bereinigt: %s",
                len(unique), unique,
            )
            self._hass.async_create_task(self.async_save())
        else:
            _LOGGER.debug("LearningEngine: Keine veralteten Zonen gefunden")

    # ── API ─────────────────────────────────────────────────────────────

    def set_zone_enabled(self, zone_id: str, enabled: bool) -> None:
        """Lernmodus für eine einzelne Zone setzen."""
        self._zone_enabled[zone_id] = enabled

    def is_zone_enabled(self, zone_id: str) -> bool:
        return self._zone_enabled.get(zone_id, True)

    def _is_enabled(self, zone_id: str) -> bool:
        return self.is_zone_enabled(zone_id)

    # ── Deduplizierung ───────────────────────────────────────────────────

    def _is_duplicate(
        self,
        obs_list: list[dict],
        now: datetime,
        hour: int,
        is_weekend: bool,
        outdoor_temp: float,
        delta: float,
        window_min: int = 30,
        temp_tol: float = 2.0,
        delta_tol: float = 0.8,
    ) -> bool:
        """Prüft ob eine ähnliche Beobachtung in den letzten window_min Minuten existiert.

        Ähnlich = selbe Stunde, selber Tag-Typ, ähnliche Außentemp, ähnliches Delta.
        Wenn ja → Duplikat überspringen, Originalwerte bleiben unverändert.
        """
        cutoff = now - timedelta(minutes=window_min)
        for obs in reversed(obs_list[-20:]):   # Nur letzte 20 prüfen → schnell
            try:
                if datetime.fromisoformat(obs["ts"]) < cutoff:
                    break
            except (ValueError, KeyError):
                continue
            if (obs.get("weekday", 0) >= 5) != is_weekend:
                continue
            if obs.get("hour", -99) != hour:
                continue
            if abs((obs.get("outdoor_temp") or outdoor_temp) - outdoor_temp) > temp_tol:
                continue
            if abs((obs.get("delta") or delta) - delta) > delta_tol:
                continue
            return True
        return False

    async def async_observe(
        self,
        zone_id: str,
        recommendation: dict,
        weather_data: dict,
        indoor_humidity: float | None = None,
    ) -> None:
        """Beobachtung aufzeichnen – alle verfügbaren Bedingungen speichern."""
        if not self._is_enabled(zone_id):
            return

        current_temp = recommendation.get("current_temp")
        # Record effective_target (weather-adjusted) – the actual temperature we aimed for.
        # Fall back to adjusted_target for backward compatibility.
        adjusted_target = (
            recommendation.get("effective_target")
            or recommendation.get("adjusted_target")
        )
        if current_temp is None or adjusted_target is None:
            return

        now = dt_util.now()

        # Heizrate aus aufeinanderfolgenden Messungen
        heat_rate: float | None = None
        if zone_id in self._last_temp:
            last_time, last_temp = self._last_temp[zone_id]
            elapsed_min = (now - last_time).total_seconds() / 60
            if 3 <= elapsed_min <= 15 and current_temp > last_temp:
                heat_rate = round((current_temp - last_temp) / elapsed_min, 5)
        self._last_temp[zone_id] = (now, current_temp)

        # Abkühlrate messen wenn Raum unter Ziel liegt und sich abkühlt
        # → Wärmeverlustrate des Gebäudes (°C/min)
        cool_rate: float | None = None
        is_cooling = current_temp < adjusted_target - 0.5
        if is_cooling and zone_id in self._last_idle_temp:
            idle_time, idle_temp = self._last_idle_temp[zone_id]
            elapsed_min = (now - idle_time).total_seconds() / 60
            if 10 <= elapsed_min <= 60 and current_temp < idle_temp:
                raw_cool = (idle_temp - current_temp) / elapsed_min
                # Plausibilitäts-Guard: max. 0.5°C/min Abkühlung (= 30°C/h) ist physikalisch
                # unrealistisch für ein Gebäude; Ausreißer durch Sensor-Fehler abfangen
                if raw_cool <= 0.5:
                    cool_rate = round(raw_cool, 5)
                    # EMA der Wärmeverlustrate aktualisieren (α=0.15 – träge, stabil)
                    prev_ema = self._heat_loss_ema.get(zone_id, cool_rate)
                    self._heat_loss_ema[zone_id] = round(0.15 * cool_rate + 0.85 * prev_ema, 6)
        if is_cooling:
            self._last_idle_temp[zone_id] = (now, current_temp)
        else:
            self._last_idle_temp.pop(zone_id, None)

        obs: dict[str, Any] = {
            "ts": now.isoformat(),
            "hour": now.hour,
            "minute": now.minute,
            "weekday": now.weekday(),
            "target": adjusted_target,
            "indoor_temp": current_temp,
            "delta": round(adjusted_target - current_temp, 2),
        }

        # Alle verfügbaren Außenbedingungen speichern
        outdoor_temp_val: float | None = None
        for key in ("temperature", "humidity", "wind_speed", "solar_radiation"):
            val = weather_data.get(key)
            if val is not None:
                map_key = {
                    "temperature": "outdoor_temp",
                    "humidity": "outdoor_humidity",
                    "wind_speed": "wind_speed",
                    "solar_radiation": "solar_radiation",
                }[key]
                obs[map_key] = val
                if key == "temperature":
                    outdoor_temp_val = val

        if indoor_humidity is not None:
            obs["indoor_humidity"] = indoor_humidity
        if heat_rate is not None:
            obs["heat_rate"] = heat_rate
            # Normalisierte Heizrate: heat_rate / (target - outdoor)
            # Macht Heizraten bei verschiedenen Außentemperaturen direkt vergleichbar.
            # Beschreibt die "thermische Effizienz" des Heizkörpers unabhängig von
            # Außenbedingungen → bessere saisonübergreifende Vorhersage.
            outdoor_delta = max(adjusted_target - (outdoor_temp_val or 10.0), 1.0)
            obs["norm_heat_rate"] = round(heat_rate / outdoor_delta, 6)
        if cool_rate is not None:
            obs["cool_rate"] = cool_rate
        if recommendation.get("forecast_high") is not None:
            obs["forecast_high"] = recommendation["forecast_high"]

        outdoor = weather_data.get("temperature") or 10.0
        is_weekend = now.weekday() >= 5
        if not self._is_duplicate(
            self._observations[zone_id], now,
            now.hour, is_weekend, outdoor, obs["delta"],
            window_min=30, temp_tol=2.0, delta_tol=0.8,
        ):
            self._observations[zone_id].append(obs)
        self._rebuild_confidence(zone_id)

        # Alle 50 Beobachtungen speichern
        if sum(len(v) for v in self._observations.values()) % 50 == 0:
            await self.async_save()

    async def async_observe_trv_setpoint(
        self,
        zone_id: str,
        trv_setpoint: float,
        indoor_temp: float,
        target: float,
        weather_data: dict,
        heat_rate: float | None,
    ) -> None:
        """Beobachtet welchen TRV-Setpoint der aktive Regler verwendet
        und welche Heizrate dabei resultiert.

        Lernt: setpoint_efficiency = heat_rate / (trv_setpoint - indoor_temp)
        Ermöglicht: beim Übernehmen direkt den richtigen Setpoint zu verwenden.
        """
        if not self._is_enabled(zone_id):
            return
        excess = trv_setpoint - indoor_temp
        if excess <= 0.3:
            return
        # Require a positive heat_rate only when heat_rate is actually available.
        # heat_rate is None on the very first call (no previous temp reading yet).
        # Without heat_rate we still record the observation for the count sensor,
        # but omit the efficiency field (which needs actual heating data).
        if heat_rate is not None and heat_rate <= 0:
            return

        now = dt_util.now()
        obs: dict = {
            "ts": now.isoformat(),
            "trv_setpoint": round(trv_setpoint, 1),
            "indoor_temp": round(indoor_temp, 1),
            "target": round(target, 1),
            "delta": round(target - indoor_temp, 2),
            "setpoint_excess": round(excess, 2),
        }
        if heat_rate is not None and heat_rate > 0:
            obs["heat_rate"] = round(heat_rate, 5)
            obs["efficiency"] = round(heat_rate / excess, 6)
        for key, map_key in (
            ("temperature", "outdoor_temp"),
            ("wind_speed", "wind_speed"),
            ("solar_radiation", "solar_radiation"),
        ):
            val = weather_data.get(key)
            if val is not None:
                obs[map_key] = val

        is_weekend = now.weekday() >= 5
        outdoor = weather_data.get("temperature") or 10.0
        if not self._is_duplicate(
            self._trv_observations[zone_id], now,
            now.hour, is_weekend, outdoor, obs["delta"],
            window_min=60, temp_tol=2.0, delta_tol=0.8,
        ):
            self._trv_observations[zone_id].append(obs)
        # Kein hartes Limit – Zeit-basiertes Ausdünnen in async_save()

        if sum(len(v) for v in self._trv_observations.values()) % 20 == 0:
            await self.async_save()

    async def async_observe_window_cooling(
        self,
        zone_id: str,
        duration_min: float,
        temp_at_open: float,
        temp_at_close: float,
        weather_data: dict,
    ) -> None:
        """Beobachtet wie stark der Raum bei geöffnetem Fenster abkühlt.

        Lernt: Abkühlrate (°C/min) in Abhängigkeit von Außenbedingungen.
        """
        if duration_min < 1.0:
            return
        temp_drop = temp_at_open - temp_at_close
        if temp_drop <= 0:
            return

        now = dt_util.now()
        obs: dict = {
            "ts": now.isoformat(),
            "duration_min": round(duration_min, 1),
            "temp_drop": round(temp_drop, 2),
            "cooling_rate_per_min": round(temp_drop / duration_min, 4),
            "indoor_start_temp": round(temp_at_open, 1),
        }
        for key, map_key in (
            ("temperature", "outdoor_temp"),
            ("wind_speed", "wind_speed"),
        ):
            val = weather_data.get(key)
            if val is not None:
                obs[map_key] = val

        self._window_cooling_obs[zone_id].append(obs)
        # Kein hartes Limit – Zeit-basiertes Ausdünnen in async_save()
        _LOGGER.info(
            "LearningEngine [%s]: Fenster-Abkühlrate %.3f°C/min gelernt "
            "(%.1f°C in %.0f min, Outdoor: %s°C)",
            zone_id, obs["cooling_rate_per_min"], temp_drop, duration_min,
            obs.get("outdoor_temp", "?"),
        )
        await self.async_save()

    # ── Outcome-Scoring ──────────────────────────────────────────────────

    def update_heating_session(
        self,
        zone_id: str,
        current_temp: float | None,
        target: float | None,
        is_active_control: bool,
        weather_data: dict,
        expected_minutes: int = 0,
    ) -> None:
        """Verfolgt Heizsitzungen und bewertet deren Ergebnis (Outcome-Score 0–1).

        Wird jeden Update-Zyklus aufgerufen. Erkennt automatisch:
          - Sitzungsstart: Ziel > Ist + 0.5°C
          - Sitzungsende:  Ziel erreicht (Score berechnen) oder Timeout 90 min
          - Unterbrechung: Fenster offen (target=None) oder Modus geändert
        """
        now = dt_util.now()
        session = self._heating_sessions.get(zone_id)
        controller = "ts" if is_active_control else "obs"

        # Keine Temp → laufende Sitzung unterbrechen
        if current_temp is None or target is None:
            if session is not None:
                self._finalize_session(zone_id, current_temp or session.get("start_temp", 0), "interrupted")
            return

        delta = target - current_temp
        tolerance = 0.5

        if delta > tolerance:
            if session is None:
                # Neue Sitzung starten
                self._heating_sessions[zone_id] = {
                    "start_time": now.isoformat(),
                    "start_temp": current_temp,
                    "target": target,
                    "controller": controller,
                    "weather": {
                        "outdoor_temp": weather_data.get("temperature"),
                        "wind_speed": weather_data.get("wind_speed"),
                        "solar_radiation": weather_data.get("solar_radiation"),
                        "outdoor_humidity": weather_data.get("humidity"),
                        "rain": weather_data.get("rain"),
                    },
                    "peak_temp": current_temp,
                    "expected_minutes": expected_minutes,
                }
                _LOGGER.debug(
                    "LearningEngine [%s]: Heizsitzung gestartet "
                    "(%.1f°C → %.1f°C, Regler: %s, erwartet: %d min)",
                    zone_id, current_temp, target, controller, expected_minutes,
                )
            else:
                # Sitzung fortführen: Peak tracken, Timeout prüfen
                session["peak_temp"] = max(session["peak_temp"], current_temp)
                try:
                    start = datetime.fromisoformat(session["start_time"])
                    if (now - start).total_seconds() / 60 >= 90:
                        self._finalize_session(zone_id, current_temp, "timeout")
                except (ValueError, KeyError):
                    pass
        else:
            # Ziel erreicht
            if session is not None:
                session["peak_temp"] = max(session["peak_temp"], current_temp)
                self._finalize_session(zone_id, current_temp, "reached")

    def _finalize_session(self, zone_id: str, current_temp: float, reason: str) -> None:
        """Heizsitzung abschließen, Score berechnen und persistieren."""
        session = self._heating_sessions.pop(zone_id, None)
        if session is None:
            return

        now = dt_util.now()
        try:
            start = datetime.fromisoformat(session["start_time"])
            elapsed_min = (now - start).total_seconds() / 60
        except (ValueError, KeyError):
            return

        if elapsed_min < 3.0:
            return  # Zu kurze Sitzung – nicht aussagekräftig

        score = self._calc_outcome_score(
            minutes_taken=elapsed_min,
            expected_minutes=session.get("expected_minutes", 0),
            peak_temp=session["peak_temp"],
            current_temp=current_temp,
            target=session["target"],
            start_temp=session["start_temp"],
            solar_radiation=session["weather"].get("solar_radiation") or 0.0,
            outdoor_temp=session["weather"].get("outdoor_temp"),
            wind_speed=session["weather"].get("wind_speed") or 0.0,
            outdoor_humidity=session["weather"].get("outdoor_humidity") or 0.0,
            rain=session["weather"].get("rain") or 0.0,
            reason=reason,
        )

        outcome: dict = {
            "ts": now.isoformat(),
            "start_temp": session["start_temp"],
            "target": session["target"],
            "peak_temp": round(session["peak_temp"], 1),
            "end_temp": round(current_temp, 1),
            "minutes_taken": round(elapsed_min, 1),
            "expected_minutes": session.get("expected_minutes", 0),
            "controller": session["controller"],
            "reason": reason,
            "outcome_score": score,
        }
        for k, v in session["weather"].items():
            if v is not None:
                outcome[k] = v

        self._outcome_log[zone_id].append(outcome)
        if len(self._outcome_log[zone_id]) > 200:
            self._outcome_log[zone_id] = self._outcome_log[zone_id][-200:]

        # Score in letzte TRV-Beobachtung einarbeiten damit sie korrekt gewichtet wird
        trv_obs = self._trv_observations.get(zone_id, [])
        if trv_obs:
            trv_obs[-1]["outcome_score"] = score

        _LOGGER.info(
            "LearningEngine [%s]: Heizsitzung beendet "
            "(%.1f→%.1f°C peak=%.1f°C, %.0f min, Regler=%s, Score=%.0f%%, Grund=%s)",
            zone_id, session["start_temp"], current_temp, session["peak_temp"],
            elapsed_min, session["controller"], score * 100, reason,
        )
        self._hass.async_create_task(self.async_save())

    @staticmethod
    def _calc_outcome_score(
        minutes_taken: float,
        expected_minutes: float,
        peak_temp: float,
        current_temp: float,
        target: float,
        start_temp: float,
        solar_radiation: float = 0.0,
        outdoor_temp: float | None = None,
        wind_speed: float = 0.0,
        outdoor_humidity: float = 0.0,
        rain: float = 0.0,
        reason: str = "reached",
    ) -> float:
        """Outcome-Score 0.0–1.0 für eine abgeschlossene Heizsitzung.

        Drei Komponenten:
          Reached  (40%): Wurde das Ziel überhaupt erreicht?
          Speed    (35%): Wie schnell vs. Erwartung? (korrigiert für Schwierigkeit)
          Accuracy (25%): Wie präzise – kein Überschießen?

        Schwierigkeits-Korrektoren (Speed-Score nachsichtiger):
          Kälte       (< −5°C):   Heizung arbeitet schwerer    → erwartete Zeit ×1.4 max.
          Wind        (> 10 m/s): Wärmeverlust steigt          → erwartete Zeit ×1.25 max.
          Hohe Feuchte(> 80%):    Gefühlte Kälte steigt        → erwartete Zeit ×1.1 max.
          Regen       (> 0):      Nasse Wände leiten Wärme ab  → erwartete Zeit ×1.1 max.

        Umgebungs-Reliability-Discount (Score abwerten wenn externe Faktoren helfen):
          Solar       (> 400 W/m²): Sonne heizt Raum mit       → bis −40% Gewicht
          Warm        (> 15°C):     Außenwärme hilft mit        → bis −30% Gewicht
        """
        outdoor = outdoor_temp if outdoor_temp is not None else 10.0
        delta_total = target - start_temp

        # ── Reached Score (40%) ──────────────────────────────────────
        if delta_total <= 0:
            reached_score = 1.0
        elif reason == "timeout":
            reached_score = max(0.0, min(1.0, (current_temp - start_temp) / delta_total))
        elif reason == "interrupted":
            reached_score = max(0.0, min(1.0, (peak_temp - start_temp) / max(delta_total, 0.1)))
        else:
            reached_score = 1.0

        # ── Speed Score (35%) – alle Schwierigkeitsfaktoren ──────────
        if expected_minutes > 5:
            difficulty = 1.0
            if outdoor < -5:
                difficulty += min((abs(outdoor) - 5) / 30, 0.40)   # max ×1.40 Extremkälte
            if wind_speed > 10:
                difficulty += min((wind_speed - 10) / 40, 0.25)    # max ×1.25 Sturm
            if outdoor_humidity > 80 and outdoor < 10:
                difficulty += min((outdoor_humidity - 80) / 200, 0.10)  # max ×1.10 feuchte Kälte
            if rain > 0:
                difficulty += min(rain / 10.0, 0.10)               # max ×1.10 Regen

            adjusted_expected = expected_minutes * difficulty
            ratio = minutes_taken / adjusted_expected

            if ratio <= 1.0:
                speed_score = 1.0
            elif ratio <= 2.0:
                speed_score = 1.0 - (ratio - 1.0) * 0.7
            else:
                speed_score = max(0.05, 0.3 - (ratio - 2.0) * 0.1)
        else:
            speed_score = 0.6  # Kein Erwartungswert → neutral

        # ── Accuracy Score (25%) ─────────────────────────────────────
        overshoot = max(0.0, peak_temp - target)
        accuracy_score = max(0.0, 1.0 - overshoot / 2.0)

        # ── Umgebungs-Reliability-Discount ────────────────────────────
        # Externe Faktoren helfen beim Heizen → Score als Lernquelle weniger verlässlich
        solar_discount = max(0.60, 1.0 - max(0, solar_radiation - 400) / 2000) \
            if solar_radiation > 400 else 1.0
        warm_discount = max(0.70, 1.0 - max(0, outdoor - 15.0) / 25.0) \
            if outdoor > 15.0 else 1.0
        env_discount = solar_discount * warm_discount

        raw = reached_score * 0.40 + speed_score * 0.35 + accuracy_score * 0.25
        return round(min(1.0, max(0.0, raw * env_discount)), 3)

    def get_outcome_stats(self, zone_id: str) -> dict:
        """Statistiken über abgeschlossene Heizsitzungen."""
        log = self._outcome_log.get(zone_id, [])
        if not log:
            return {"sessions": 0, "avg_score": None, "ts_avg": None, "obs_avg": None}

        scores_ts  = [e["outcome_score"] for e in log if e.get("controller") == "ts"]
        scores_obs = [e["outcome_score"] for e in log if e.get("controller") == "obs"]
        all_scores = [e["outcome_score"] for e in log]

        def _avg(lst):
            return round(sum(lst) / len(lst) * 100, 1) if lst else None

        return {
            "sessions": len(log),
            "avg_score_%": _avg(all_scores),
            "ts_avg_%": _avg(scores_ts),
            "obs_avg_%": _avg(scores_obs),
            "last_score_%": round(log[-1]["outcome_score"] * 100, 1) if log else None,
            "last_controller": log[-1].get("controller") if log else None,
        }

    def get_window_cooling_rate(self, zone_id: str, weather_data: dict) -> float | None:
        """Gibt gelernte Fenster-Abkühlrate (°C/min) für aktuelle Außenbedingungen zurück."""
        obs_list = self._window_cooling_obs.get(zone_id, [])
        if len(obs_list) < 2:
            return None

        now = dt_util.now()
        outdoor = weather_data.get("temperature") or 10.0
        wind = weather_data.get("wind_speed") or 0.0

        rates = []
        weights = []
        for obs in obs_list:
            obs_outdoor = obs.get("outdoor_temp") or outdoor
            obs_wind = obs.get("wind_speed") or 0.0
            temp_sim = math.exp(-((obs_outdoor - outdoor) / 5.0) ** 2)
            wind_sim = math.exp(-((obs_wind - wind) / 3.0) ** 2)
            try:
                age_days = (now - datetime.fromisoformat(obs["ts"])).total_seconds() / 86400
            except (ValueError, KeyError):
                age_days = 0
            recency = math.exp(-age_days / 180)
            w = temp_sim * wind_sim * recency
            if w < 0.05:
                continue
            rates.append(obs["cooling_rate_per_min"])
            weights.append(w)

        if not rates:
            return None
        return round(sum(r * w for r, w in zip(rates, weights)) / sum(weights), 4)

    def get_trv_stats(self, zone_id: str) -> dict:
        """Statistiken über TRV-Setpoint-Beobachtungen."""
        trv_obs = self._trv_observations.get(zone_id, [])
        win_obs = self._window_cooling_obs.get(zone_id, [])
        avg_efficiency = None
        if trv_obs:
            effs = [o["efficiency"] for o in trv_obs if o.get("efficiency")]
            if effs:
                avg_efficiency = round(sum(effs) / len(effs), 6)
        avg_window_cooling = None
        if win_obs:
            rates = [o["cooling_rate_per_min"] for o in win_obs if o.get("cooling_rate_per_min")]
            if rates:
                avg_window_cooling = round(sum(rates) / len(rates), 4)
        return {
            "trv_observations": len(trv_obs),
            "window_observations": len(win_obs),
            "avg_setpoint_efficiency": avg_efficiency,
            "avg_window_cooling_rate": avg_window_cooling,
        }

    async def async_get_base_target(
        self,
        zone_id: str,
        mode: str = HEATING_MODE_AUTO,
        comfort_temp: float = 21.0,
        night_temp: float = 18.0,
        away_temp: float = 17.0,
        vacation_temp: float = 12.0,
        eco_temp: float = TEMP_ECO,
        schedule_cfg: dict | None = None,
    ) -> float:
        """Empfohlene Zieltemperatur – Zeitplan oder Modus-Temperatur."""
        mode_temps = {
            "comfort": comfort_temp,
            "night": night_temp,
            "away": away_temp,
            "vacation": vacation_temp,
            "eco": eco_temp,
        }
        if mode != HEATING_MODE_AUTO:
            return mode_temps.get(mode, night_temp)

        if (schedule_cfg or {}).get(CONF_SCHEDULE_ENABLED, True):
            schedule_temp = self._schedule_target(
                zone_id, comfort_temp, night_temp, away_temp, schedule_cfg
            )
        else:
            schedule_temp = comfort_temp

        # The learning engine always returns exactly the configured schedule temperature.
        # Learning is used for heating rates, preheat timing, and TPI coefficients –
        # NOT for overriding the user's explicitly configured target temperature.
        return schedule_temp

    async def async_get_preheat_minutes(
        self,
        zone_id: str,
        target: float,
        current_temp: float | None,
        weather_data: dict,
    ) -> int:
        """Vorheizzeit berechnen – Multi-Faktor-Heizrate nutzen."""
        if current_temp is None:
            return 0
        delta = target - current_temp
        if delta < PREHEAT_MIN_DELTA:
            return 0

        rate = self._get_heat_rate(zone_id, weather_data, target=target)
        if rate <= 0:
            return 0

        # Effektive Heizrate = Heizrate minus Abkühlrate (Haus kühlt während Vorheizen weiter ab)
        cool_rate = self._get_avg_cool_rate(zone_id)
        effective_rate = max(rate - cool_rate, rate * 0.3)

        minutes = int(min(delta / effective_rate, PREHEAT_MAX_MINUTES))
        _LOGGER.debug(
            "LearningEngine [%s] Vorheizzeit=%d min (Δ%.1f°C, Heiz=%.4f°C/min, Kühl=%.4f°C/min)",
            zone_id, minutes, delta, rate, cool_rate,
        )
        return minutes

    def get_confidence(self, zone_id: str) -> float:
        return self._confidence.get(zone_id, 0.0)

    def get_confidence_breakdown(self, zone_id: str) -> dict:
        """Aufschlüsselung der Konfidenz nach Lernspuren."""
        now = dt_util.now()
        obs = self._observations.get(zone_id, [])
        heating_obs = [o for o in obs if o.get("heat_rate") is not None]
        trv_n = len(self._trv_observations.get(zone_id, []))
        win_n = len(self._window_cooling_obs.get(zone_id, []))

        weighted_n = sum(_time_weight(o["ts"], now) for o in heating_obs) if heating_obs else 0
        base_conf = round(min(weighted_n / 50, 1.0) * 100, 1)
        trv_conf  = round(min(trv_n / 30, 1.0) * 100, 1)
        win_conf  = round(min(win_n / 5, 1.0) * 100, 1)

        return {
            "total_%": round(self.get_confidence(zone_id) * 100, 1),
            "room_patterns_%": base_conf,
            "trv_efficiency_%": trv_conf,
            "window_cooling_%": win_conf,
            "total_observations": len(obs),
            "heating_observations": len(heating_obs),
            "trv_observations": trv_n,
            "window_events": win_n,
            "forecast_confidence_%": round(self.get_forecast_bias(zone_id) * 100, 1),
        }

    def get_boost_factor(self, zone_id: str) -> float:
        return self._boost_factors.get(zone_id, 1.0)

    def get_forecast_bias(self, zone_id: str) -> float:
        """Gelerntes Vertrauen in Wetterprognosen (1.0 = voll vertrauen, 0.3 = skeptisch)."""
        return self._forecast_bias.get(zone_id, FORECAST_BIAS_MAX)

    def record_forecast_decision(
        self,
        zone_id: str,
        decision_target: float,
        raw_target: float,
        forecast_high: float,
        current_temp: float,
        suppression: float,
        outdoor_temp: float | None,
    ) -> None:
        """Prognose-Entscheidung aufzeichnen – wird nach FORECAST_EVAL_HOURS ausgewertet."""
        if not self._is_enabled(zone_id):
            return
        now = dt_util.now()
        # Duplikat-Schutz: kein neuer Eintrag wenn eine noch unausgewertete Entscheidung
        # aus den letzten 2 Stunden existiert
        recent = self._forecast_decisions[zone_id]
        if recent:
            try:
                last_ts = datetime.fromisoformat(recent[-1]["ts"])
                if (now - last_ts).total_seconds() < 7200:
                    return
            except (ValueError, KeyError):
                pass
        self._forecast_decisions[zone_id].append({
            "ts": now.isoformat(),
            "decision_target": decision_target,
            "raw_target": raw_target,
            "forecast_high": forecast_high,
            "current_temp_at_decision": current_temp,
            "suppression": suppression,
            "outdoor_temp": outdoor_temp,
            "evaluated": False,
        })
        _LOGGER.debug(
            "LearningEngine [%s]: Prognose-Entscheidung aufgezeichnet "
            "(Ziel=%.1f°C statt %.1f°C, Prognose=%.1f°C, Suppression=%.0f%%)",
            zone_id, decision_target, raw_target, forecast_high, (1 - suppression) * 100,
        )

    def evaluate_forecast_decisions(
        self,
        zone_id: str,
        current_temp: float | None,
        outdoor_temp: float | None,
    ) -> None:
        """Ausstehende Prognose-Entscheidungen auswerten und Forecast-Bias anpassen.

        Logik:
          - Ziel nicht erreicht (Raum kalt) → Prognose war zu optimistisch → Bias senken
          - Ziel deutlich überschritten     → Prognose war konservativ     → Bias leicht erhöhen
          - Ziel im Toleranzbereich         → Prognose war korrekt          → Bias leicht erhöhen
        """
        if current_temp is None:
            return
        now = dt_util.now()
        decisions = self._forecast_decisions.get(zone_id, [])
        changed = False

        for dec in decisions:
            if dec.get("evaluated"):
                continue
            try:
                decision_time = datetime.fromisoformat(dec["ts"])
            except (ValueError, KeyError):
                dec["evaluated"] = True
                continue
            if (now - decision_time).total_seconds() / 3600 < FORECAST_EVAL_HOURS:
                continue

            shortfall = dec["decision_target"] - current_temp  # positiv = Ziel verfehlt
            bias = self._forecast_bias.get(zone_id, FORECAST_BIAS_MAX)

            if shortfall > 0.5:
                # Raum zu kalt → Prognose war zu optimistisch
                reduction = FORECAST_BIAS_LEARNING_RATE * min(shortfall / 2.0, 1.0)
                bias = max(FORECAST_BIAS_MIN, round(bias - reduction, 3))
                _LOGGER.info(
                    "LearningEngine [%s]: Prognose-Auswertung – Ziel %.1f°C verfehlt "
                    "(Ist: %.1f°C, Δ%.1f°C) → Bias %.3f",
                    zone_id, dec["decision_target"], current_temp, shortfall, bias,
                )
            elif shortfall < -1.5:
                # Raum deutlich über Ziel → Prognose war eher konservativ oder Heizung lief trotzdem
                bias = min(FORECAST_BIAS_MAX, round(bias + FORECAST_BIAS_LEARNING_RATE * 0.3, 3))
                _LOGGER.debug(
                    "LearningEngine [%s]: Prognose-Auswertung – Ziel überschritten "
                    "(Ist: %.1f°C) → Bias %.3f", zone_id, current_temp, bias,
                )
            else:
                # Ziel erreicht – Prognose war korrekt
                bias = min(FORECAST_BIAS_MAX, round(bias + FORECAST_BIAS_LEARNING_RATE * 0.15, 3))
                _LOGGER.debug(
                    "LearningEngine [%s]: Prognose-Auswertung – Ziel erreicht → Bias %.3f",
                    zone_id, bias,
                )

            self._forecast_bias[zone_id] = bias
            dec["evaluated"] = True
            changed = True

        # Alte ausgewertete Einträge bereinigen (> 7 Tage)
        cutoff_iso = (now - timedelta(days=7)).isoformat()
        self._forecast_decisions[zone_id] = [
            d for d in decisions
            if not d.get("evaluated") or d.get("ts", "") >= cutoff_iso
        ]

        if changed:
            self._hass.async_create_task(self.async_save())

    def update_boost_factor(self, zone_id: str, overshot: bool, slow: bool = False) -> None:
        """Boost-Faktor nach Heizzyklus anpassen.

        Überschießen → Faktor reduzieren (Ventil war zu weit auf).
        Zu langsam    → Faktor erhöhen (Ventil öffnet zu wenig).
        """
        if not self._is_enabled(zone_id):
            return
        factor = self._boost_factors.get(zone_id, 1.0)
        if overshot:
            factor = max(0.5, round(factor * 0.92, 3))
            _LOGGER.info(
                "LearningEngine [%s] Boost-Faktor reduziert → %.3f (Überschießen)",
                zone_id, factor,
            )
        elif slow:
            factor = min(2.0, round(factor * 1.05, 3))
            _LOGGER.info(
                "LearningEngine [%s] Boost-Faktor erhöht → %.3f (langsames Heizen)",
                zone_id, factor,
            )
        self._boost_factors[zone_id] = factor

    def get_stats(self, zone_id: str) -> dict:
        obs = self._observations[zone_id]
        with_wind = sum(1 for o in obs if o.get("wind_speed") is not None)
        with_solar = sum(1 for o in obs if o.get("solar_radiation") is not None)
        with_humidity = sum(1 for o in obs if o.get("indoor_humidity") is not None)
        with_heat_rate = sum(1 for o in obs if o.get("heat_rate") is not None)
        with_cool_rate = sum(1 for o in obs if o.get("cool_rate") is not None)
        avg_cool_rate = None
        cool_samples = [o["cool_rate"] for o in obs if o.get("cool_rate")]
        if cool_samples:
            avg_cool_rate = round(sum(cool_samples) / len(cool_samples), 5)
        avg_heat_rate = None
        heat_samples = [o["heat_rate"] for o in obs if o.get("heat_rate")]
        if heat_samples:
            avg_heat_rate = round(sum(heat_samples[-50:]) / len(heat_samples[-50:]), 5)
        return {
            "total_observations": len(obs),
            "confidence": self.get_confidence(zone_id),
            "with_wind_data": with_wind,
            "with_solar_data": with_solar,
            "with_humidity_data": with_humidity,
            "with_heat_rate": with_heat_rate,
            "with_cool_rate": with_cool_rate,
            "avg_heat_rate_per_min": avg_heat_rate,
            "avg_cool_rate_per_min": avg_cool_rate,
            "oldest": obs[0]["ts"] if obs else None,
        }

    # ── Interne Helfer ───────────────────────────────────────────────────

    def _schedule_target(
        self,
        zone_id: str,
        comfort_temp: float = 21.0,
        night_temp: float = 18.0,
        away_temp: float = 17.0,
        schedule_cfg: dict | None = None,
    ) -> float:
        """Konfigurierbarer Zeitplan: Werktag 4 Blöcke, Wochenende 2 Blöcke."""
        now = dt_util.now()
        is_weekend = now.weekday() >= 5
        cur = now.hour * 60 + now.minute

        def t(key: str, fallback: str) -> int:
            raw = (schedule_cfg or {}).get(key, fallback)
            try:
                parts = str(raw).split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except (ValueError, AttributeError, IndexError):
                h2, m2 = fallback.split(":")
                return int(h2) * 60 + int(m2)

        # Werktag und Wochenende: gleiches Schema (Nacht → Komfort → Nacht)
        # Abwesenheit wird automatisch via Präsenzerkennung → away_temp gehandelt
        if not is_weekend:
            morning = t(CONF_SCHED_WD_MORNING, "06:00")
            night   = t(CONF_SCHED_WD_NIGHT,   "22:00")
        else:
            morning = t(CONF_SCHED_WE_MORNING, "08:00")
            night   = t(CONF_SCHED_WE_NIGHT,   "23:00")

        if cur < morning or cur >= night:
            return night_temp
        return comfort_temp

    def _weighted_targets_for_hour(
        self, zone_id: str, now: datetime
    ) -> list[tuple[float, float]]:
        """Zeitbasierte Gewichtung – 21°C morgens gilt jeden Tag."""
        hour = now.hour
        is_weekend = now.weekday() >= 5
        result = []
        for obs in self._observations[zone_id]:
            if (obs["weekday"] >= 5) != is_weekend:
                continue
            if abs(obs["hour"] - hour) > 1:
                continue
            weight = _time_weight(obs["ts"], now, halflife_days=180)
            if weight < 0.01:
                continue
            result.append((obs["target"], weight))
        return result

    def _weighted_median(self, weighted: list[tuple[float, float]]) -> float:
        if not weighted:
            return 21.0
        sorted_w = sorted(weighted, key=lambda x: x[0])
        total_weight = sum(w for _, w in sorted_w)
        cumulative = 0.0
        for value, weight in sorted_w:
            cumulative += weight
            if cumulative >= total_weight / 2:
                return value
        return sorted_w[-1][0]

    def _get_avg_cool_rate(self, zone_id: str) -> float:
        """Wärmeverlustrate aus EMA oder Beobachtungs-Durchschnitt."""
        # EMA bevorzugen – wird nach jedem Messzyklus aktualisiert
        if zone_id in self._heat_loss_ema:
            return self._heat_loss_ema[zone_id]
        # Fallback: einfacher Mittelwert der letzten 50 Messungen
        samples = [o["cool_rate"] for o in self._observations[zone_id] if o.get("cool_rate")]
        if not samples:
            return 0.0
        recent = samples[-50:]
        return round(sum(recent) / len(recent), 5)

    def get_tpi_coefficients(
        self, zone_id: str, weather_data: dict
    ) -> tuple[float, float]:
        """Gibt TPI-Koeffizienten (coef_int, coef_ext) für diese Zone zurück.

        Nutzt gelernte Heizrate und Wärmeverlustrate um die Koeffizienten
        physikalisch abzuleiten – kein manuelles Tuning nötig.
        Fällt auf Standardwerte zurück wenn zu wenig Daten vorhanden.
        """
        from .tpi import estimate_coefficients
        heat_rate = self._get_heat_rate(zone_id, weather_data)
        heat_loss = self.get_heat_loss_rate(zone_id)
        return estimate_coefficients(
            heat_rate if heat_rate > 0 else None,
            heat_loss,
        )

    def get_heat_loss_rate(self, zone_id: str) -> float | None:
        """Gibt die gelernte Wärmeverlustrate (°C/min) zurück – None wenn keine Daten."""
        rate = self._heat_loss_ema.get(zone_id)
        if rate:
            return rate
        samples = [o["cool_rate"] for o in self._observations.get(zone_id, []) if o.get("cool_rate")]
        if not samples:
            return None
        return round(sum(samples[-50:]) / len(samples[-50:]), 5)

    def _get_heat_rate(self, zone_id: str, weather_data: dict, target: float | None = None) -> float:
        if self._is_enabled(zone_id) and self.get_confidence(zone_id) >= 0.3:
            learned = self._learned_heat_rate_multifactor(zone_id, weather_data, target=target)
            if learned > 0:
                return learned
        return self._estimate_heat_rate(weather_data)

    def _learned_heat_rate_multifactor(
        self, zone_id: str, weather_data: dict, target: float | None = None
    ) -> float:
        """Multi-Faktor Heizrate: lernt aus allen Außenbedingungen.

        Zwei Lernansätze – der bessere wird bevorzugt:

        1. Normalisierte Heizrate (bevorzugt wenn genug Daten):
           norm_rate = heat_rate / (target - outdoor)
           → saisonunabhängig, benötigt keine Wetterähnlichkeit
           → wenige Beobachtungen reichen für gute Vorhersagen

        2. Multi-Faktor gewichtete Rate (Fallback):
           Ähnlichkeit nach Außentemp, Wind, Solar, Feuchte
           → Oktober bei 8°C + Wind ≈ Januar bei 8°C + Wind
        """
        now = dt_util.now()
        curr_outdoor = weather_data.get("temperature") or 10.0

        # Nur die letzten 500 Beobachtungen auswerten – ältere haben durch
        # Zeitgewichtung (HWZ 180 Tage) ohnehin < 2% Einfluss, sparen aber CPU
        recent_obs = self._observations[zone_id][-500:]

        conditions = {
            "outdoor_temp": curr_outdoor,
            "wind_speed": weather_data.get("wind_speed"),
            "solar_radiation": weather_data.get("solar_radiation"),
            "outdoor_humidity": weather_data.get("humidity"),
        }

        # ── Ansatz 1: Normalisierte Heizrate (bevorzugt) ──────────────
        # Kombiniert Zeit- UND Bedingungs-Gewichtung für maximale Genauigkeit
        if target is not None:
            curr_delta = max(target - curr_outdoor, 1.0)
            norm_obs_weighted: list[tuple[float, float]] = []
            for obs in recent_obs:
                if not obs.get("norm_heat_rate") or obs["norm_heat_rate"] <= 0:
                    continue
                # Zeitgewichtung × thermische Ähnlichkeit
                w_time = _time_weight(obs["ts"], now)
                w_therm = _thermal_weight(obs, conditions, now)
                w = w_time * (0.5 + 0.5 * w_therm)  # Normalisierung bevorzugt Zeit
                if w < 0.005:
                    continue
                norm_obs_weighted.append((obs["norm_heat_rate"], w))

            if len(norm_obs_weighted) >= LEARNING_MIN_SAMPLES:
                total_w = sum(w for _, w in norm_obs_weighted)
                avg_norm = sum(r * w for r, w in norm_obs_weighted) / total_w
                predicted = round(avg_norm * curr_delta, 5)
                _LOGGER.debug(
                    "LearningEngine [%s]: Normalisierte Heizrate %.5f°C/min "
                    "(norm=%.6f × Δ%.1f°C, %d Beob.)",
                    zone_id, predicted, avg_norm, curr_delta, len(norm_obs_weighted),
                )
                return predicted

        # ── Ansatz 2: Multi-Faktor gewichtete Rate (Fallback) ─────────
        rates = []
        weights = []
        for obs in recent_obs:
            if not obs.get("heat_rate") or obs["heat_rate"] <= 0:
                continue
            w = _thermal_weight(obs, conditions, now)
            if w < 0.01:
                continue
            rates.append(obs["heat_rate"])
            weights.append(w)

        if len(rates) < LEARNING_MIN_SAMPLES:
            return 0.0

        weighted_sum = sum(r * w for r, w in zip(rates, weights))
        weight_total = sum(weights)
        return round(weighted_sum / weight_total, 5) if weight_total > 0 else 0.0

    def _estimate_heat_rate(self, weather_data: dict) -> float:
        """Physikalische Schätzung als Cold-Start Fallback.

        Konservative Standardwerte die für die meisten deutschen Wohngebäude passen.
        Werden durch gelerntes Wissen sobald verfügbar ersetzt.
        """
        outdoor = weather_data.get("temperature") or 10.0
        wind = weather_data.get("wind_speed") or 0.0
        solar = weather_data.get("solar_radiation") or 0.0

        # Basis-Rate nach Außentemperatur
        # Kalibiert auf typisches deutsches Wohngebäude (Mehrfamilienhaus, mittlere Dämmung)
        if outdoor < -20:
            base = 0.018   # Extremkälte: Heizkörper kommt kaum hinterher, Rate sehr niedrig
        elif outdoor < -10:
            base = 0.026   # Starker Frost
        elif outdoor < -5:
            base = 0.035   # Sehr kalt: TRV arbeitet hart, Haus kühlt schnell
        elif outdoor < 0:
            base = 0.042
        elif outdoor < 5:
            base = 0.050
        elif outdoor < 10:
            base = 0.058
        elif outdoor < 15:
            base = 0.065
        else:
            base = 0.072   # Mild: Haus heizt sehr schnell

        # Wind erhöht Wärmeverlust → langsamere Aufheizung
        if wind > 5:
            wind_factor = 1.0 - min(wind / 30, 0.20)
            base *= wind_factor

        # Sonne hilft beim Aufheizen → schneller
        if solar > 300:
            solar_boost = min((solar - 300) / 1000, 0.15)
            base *= (1.0 + solar_boost)

        return round(base, 5)

    def get_cold_start_phase(self, zone_id: str) -> str:
        """Gibt die aktuelle Lernphase zurück.

        Phase 1 (0-5 Beob.):   Reine Physik-Formel, kein Lernen aktiv
        Phase 2 (5-50 Beob.):  Erste Muster sichtbar, gemischter Betrieb
        Phase 3 (50+ Beob.):   Lernalgorithmus dominiert
        Phase 4 (150+ Beob.):  Vollständig personalisiert
        """
        n = len(self._observations.get(zone_id, []))
        trv_n = len(self._trv_observations.get(zone_id, []))
        if n < 5:
            return "Initializing – physics formula active"
        if n < 50:
            return f"Phase 1 – First patterns ({n} obs.)"
        if n < 150:
            return f"Phase 2 – Learning ({n} obs., {trv_n} TRV obs.)"
        return f"Phase 3 – Personalized ({n} obs., {trv_n} TRV obs.)"

    def _rebuild_confidence(self, zone_id: str | None = None) -> None:
        """Gesamtkonfidenz aus allen drei Lernspuren.

        Zusammensetzung:
          60% Basis-Lernen   (Raumtemperatur-Muster, Heiz-/Abkühlraten)
          30% TRV-Lernen     (Setpoint-Effizienz aus Beobachtungsmodus)
          10% Fenster-Lernen (Abkühlraten beim Lüften)

        Jede Lernspur wächst unabhängig und verbessert sich mit der Zeit.
        """
        zones = [zone_id] if zone_id else list(self._observations.keys())
        now = dt_util.now()

        for zid in zones:
            # ── Basis-Lernen (60%) ─────────────────────────────────────────
            # Nur Beobachtungen MIT heat_rate zählen – diese entstehen ausschließlich
            # bei echten Heizereignissen. Reine Temperaturlesungen ohne Heizen
            # liefern keine verwertbaren Daten für Heizraten oder TPI-Koeffizienten
            # und würden die Konfidenz sonst unrealistisch schnell ansteigen lassen.
            obs = self._observations[zid]
            if obs:
                recent = obs[-500:]
                heating_obs = [o for o in recent if o.get("heat_rate") is not None]
                weighted_n    = sum(_time_weight(o["ts"], now) for o in heating_obs)
                has_wind      = any(o.get("wind_speed") is not None for o in heating_obs)
                has_solar     = any(o.get("solar_radiation") is not None for o in heating_obs)
                has_humidity  = any(o.get("indoor_humidity") is not None for o in heating_obs)
                diversity     = 1.0 + sum([has_wind, has_solar, has_humidity]) * 0.05
                # 50 Heizbeobachtungen = volle Base-Konfidenz (≈ mehrere Wochen Heizbetrieb)
                base_conf = min(weighted_n / 50 * diversity, 1.0)
            else:
                base_conf = 0.0

            # ── TRV-Lernen (30%) ──────────────────────────────────────────
            trv_n = len(self._trv_observations.get(zid, []))
            trv_conf = min(trv_n / 30, 1.0)   # 30 Beobachtungen = volle TRV-Konfidenz

            # ── Fenster-Lernen (10%) ──────────────────────────────────────
            win_n = len(self._window_cooling_obs.get(zid, []))
            win_conf = min(win_n / 5, 1.0)    # 5 Lüftungsereignisse = volle Fenster-Konfidenz

            # ── Gesamtkonfidenz ───────────────────────────────────────────
            combined = base_conf * 0.6 + trv_conf * 0.3 + win_conf * 0.1
            self._confidence[zid] = round(combined, 4)
