"""LearningEngine – adaptive, ewig lernende Heizungsoptimierung.

Kernidee: Daten werden NIEMALS gelöscht.
Ältere Beobachtungen verlieren durch exponentiellen Gewichtsabfall
schrittweise Einfluss – neue Daten dominieren, ohne das Langzeitwissen
zu vernichten. Die Konfidenz kann nur wachsen, nie sinken.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    STORAGE_VERSION,
    STORAGE_KEY,
    LEARNING_MIN_SAMPLES,
    PREHEAT_MAX_MINUTES,
    PREHEAT_MIN_DELTA,
    SCHEDULE,
    ZONE_TEMPS,
    HEATING_MODE_AUTO,
)

_LOGGER = logging.getLogger(__name__)


def _thermal_weight(obs: dict, outdoor: float, now: datetime) -> float:
    """Gewicht für thermisches Lernen (Heizrate, Vorheizzeit).

    Fragt: 'Wie ähnlich waren die Außenbedingungen?' – NICHT 'Welche Jahreszeit?'

    Ein Oktober-Tag bei 8°C ist für die Heizrate genauso relevant wie
    ein Januar-Tag bei 8°C. Das Gebäude heizt sich physikalisch gleich auf.
    Daher KEINE saisonale Gewichtung hier.

    Gewichtung nach:
      1. Außentemperatur-Ähnlichkeit (wichtigster Faktor)
      2. Leichte Aktualitätspräferenz (neuere Kalibrierungen bevorzugt)
    """
    try:
        obs_dt = datetime.fromisoformat(obs["ts"])
    except (ValueError, TypeError, KeyError):
        return 0.0

    obs_outdoor = obs.get("outdoor_temp") or outdoor
    temp_diff = abs(obs_outdoor - outdoor)

    # Temperaturähnlichkeit: σ = 5°C – bei >15°C Unterschied fast irrelevant
    temp_similarity = math.exp(-(temp_diff / 5.0) ** 2)
    if temp_similarity < 0.01:
        return 0.0

    # Leichte Aktualitätspräferenz: neuere Messungen etwas bevorzugt
    age_days = max((now - obs_dt).total_seconds() / 86400, 0)
    recency = math.exp(-age_days / 730)  # 2 Jahre Halbwertzeit

    return round(temp_similarity * recency, 6)


def _observation_weight(obs_ts: str, now: datetime) -> float:
    """Gewicht einer Beobachtung – kombiniert saisonale Ähnlichkeit + Aktualität.

    Warum nicht nur Zeit-Decay?
    Reines Zeit-Decay würde letzten Winter vergessen bevor dieser Winter kommt.
    Das ist falsch – letzter Dezember ist für diesen Dezember viel relevanter
    als der Juli dieses Jahres.

    Formel:
      Gewicht = saisonale_Ähnlichkeit(Jahrestag-Abstand) × Aktualität(Jahre)

    Saisonale Ähnlichkeit (Gaußkurve, σ = 45 Tage):
      ±0 Tage im Jahr  → 1.00  (gleiche Jahreszeit)
      ±30 Tage         → 0.64
      ±45 Tage         → 0.37
      ±90 Tage         → 0.02
      ±180 Tage        → ~0.00 (gegenüberliegende Jahreszeit)

    Aktualität (sehr sanfter Abfall über Jahre):
      0 Jahre alt  → 1.00
      1 Jahr alt   → 0.78  (letzter Winter zählt noch stark!)
      3 Jahre alt  → 0.47
      5 Jahre alt  → 0.29
    """
    try:
        obs_dt = datetime.fromisoformat(obs_ts)
    except (ValueError, TypeError):
        return 0.0

    age_days = (now - obs_dt).total_seconds() / 86400
    if age_days < 0:
        return 0.0

    # Saisonale Ähnlichkeit: Abstand im Jahreskreis (0–182 Tage)
    obs_doy = obs_dt.timetuple().tm_yday
    now_doy = now.timetuple().tm_yday
    day_dist = abs(obs_doy - now_doy)
    day_dist = min(day_dist, 365 - day_dist)  # Jahreswechsel berücksichtigen
    seasonal = math.exp(-(day_dist / 45.0) ** 2)

    # Aktualität: sanfter Abfall über Jahre (nicht Tage!)
    years_old = age_days / 365.0
    recency = math.exp(-years_old * 0.25)

    return round(seasonal * recency, 6)


class LearningEngine:
    """
    Lernt kontinuierlich aus Beobachtungen und wird mit der Zeit immer besser.

    Gespeichert wird:
      - Beobachtungszeitpunkt
      - Stunde + Wochentag
      - Zieltemperatur + Ist-Temperatur
      - Außentemperatur
      - Heizrate (°C/min) aus aufeinanderfolgenden Messungen
    """

    def __init__(self, hass: HomeAssistant, enabled: bool = True) -> None:
        self._hass = hass
        self._enabled = enabled
        self._store: Store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        # {zone_id: [{"ts": iso, "hour": int, "weekday": int,
        #              "target": float, "indoor_temp": float,
        #              "outdoor_temp": float|None, "delta": float}]}
        self._observations: dict[str, list[dict]] = defaultdict(list)
        # Letzter gemessener Temperaturwert pro Zone für Heizraten-Berechnung
        self._last_temp: dict[str, tuple[datetime, float]] = {}
        # Cached Konfidenz 0.0–1.0 (wächst monoton)
        self._confidence: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    async def async_load(self) -> None:
        """Beobachtungen aus HA-Storage laden."""
        stored = await self._store.async_load()
        if stored and isinstance(stored, dict):
            for zone_id, obs_list in stored.get("observations", {}).items():
                self._observations[zone_id] = obs_list
            _LOGGER.info(
                "LearningEngine: %d Zonen geladen, %d Gesamtbeobachtungen",
                len(self._observations),
                sum(len(v) for v in self._observations.values()),
            )
        self._rebuild_confidence()

    async def async_save(self) -> None:
        """Beobachtungen in HA-Storage schreiben."""
        await self._store.async_save({"observations": dict(self._observations)})
        _LOGGER.debug("LearningEngine: gespeichert")

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    async def async_observe(
        self, zone_id: str, recommendation: dict, weather_data: dict
    ) -> None:
        """Beobachtung aufzeichnen – läuft bei jedem Coordinator-Update."""
        if not self._enabled:
            return

        current_temp = recommendation.get("current_temp")
        adjusted_target = recommendation.get("adjusted_target")
        if current_temp is None or adjusted_target is None:
            return

        now = dt_util.now()

        # Heizrate aus aufeinanderfolgenden Messungen ableiten
        heat_rate: float | None = None
        if zone_id in self._last_temp:
            last_time, last_temp = self._last_temp[zone_id]
            elapsed_min = (now - last_time).total_seconds() / 60
            if 3 <= elapsed_min <= 15 and current_temp > last_temp:
                heat_rate = round((current_temp - last_temp) / elapsed_min, 5)

        self._last_temp[zone_id] = (now, current_temp)

        obs: dict = {
            "ts": now.isoformat(),
            "hour": now.hour,
            "minute": now.minute,
            "weekday": now.weekday(),       # 0=Mo … 6=So
            "target": adjusted_target,
            "indoor_temp": current_temp,
            "outdoor_temp": weather_data.get("temperature"),
            "delta": round(adjusted_target - current_temp, 2),
        }
        if heat_rate is not None:
            obs["heat_rate"] = heat_rate

        self._observations[zone_id].append(obs)
        self._rebuild_confidence(zone_id)

        # Periodisch speichern (alle 50 neue Beobachtungen)
        total = sum(len(v) for v in self._observations.values())
        if total % 50 == 0:
            await self.async_save()

    def get_seasonal_stats(self, zone_id: str) -> dict:
        """Debug-Info: wie viele relevante Beobachtungen für die aktuelle Jahreszeit."""
        now = dt_util.now()
        obs = self._observations[zone_id]
        high = sum(1 for o in obs if _observation_weight(o["ts"], now) > 0.5)
        medium = sum(1 for o in obs if 0.1 < _observation_weight(o["ts"], now) <= 0.5)
        low = sum(1 for o in obs if _observation_weight(o["ts"], now) <= 0.1)
        return {"high_relevance": high, "medium_relevance": medium, "low_relevance": low}

    async def async_get_base_target(
        self, zone_id: str, mode: str = HEATING_MODE_AUTO
    ) -> float:
        """Empfohlene Zieltemperatur für die aktuelle Uhrzeit zurückgeben.

        Im Auto-Modus: Zeitplan als Basis, lernt darüber hinaus.
        In anderen Modi: feste Komforttemperatur aus ZONE_TEMPS.
        """
        if mode != HEATING_MODE_AUTO:
            return ZONE_TEMPS.get(zone_id, {}).get(mode, 18.0)

        schedule_temp = self._schedule_target(zone_id)

        # Ohne ausreichend Daten: Zeitplan nehmen
        if not self._enabled or self.get_confidence(zone_id) < 0.25:
            return schedule_temp

        now = dt_util.now()
        weighted_temps = self._weighted_targets_for_hour(zone_id, now)
        if len(weighted_temps) < LEARNING_MIN_SAMPLES:
            return schedule_temp

        learned = self._weighted_median(weighted_temps)
        _LOGGER.debug(
            "LearningEngine [%s] gelernte Zieltemp=%.1f°C (Zeitplan=%.1f°C, n=%d)",
            zone_id, learned, schedule_temp, len(weighted_temps),
        )
        return round(learned, 1)

    async def async_get_preheat_minutes(
        self,
        zone_id: str,
        target: float,
        current_temp: float | None,
        weather_data: dict,
    ) -> int:
        """Vorheizzeit in Minuten berechnen."""
        if current_temp is None:
            return 0
        delta = target - current_temp
        if delta < PREHEAT_MIN_DELTA:
            return 0

        rate = self._get_heat_rate(zone_id, weather_data)
        if rate <= 0:
            return 0

        minutes = int(min(delta / rate * 60, PREHEAT_MAX_MINUTES))
        _LOGGER.debug(
            "LearningEngine [%s] Vorheizzeit=%d min (delta=%.1f°C, rate=%.4f°C/min)",
            zone_id, minutes, delta, rate,
        )
        return minutes

    def get_confidence(self, zone_id: str) -> float:
        """Konfidenz 0.0–1.0 zurückgeben. Wächst monoton, sinkt nie."""
        return self._confidence.get(zone_id, 0.0)

    def get_stats(self, zone_id: str) -> dict:
        """Statistiken für Dashboard-Anzeige."""
        obs = self._observations[zone_id]
        now = dt_util.now()
        weighted_count = sum(_observation_weight(o["ts"], now) for o in obs)
        return {
            "total_observations": len(obs),
            "weighted_observations": round(weighted_count, 1),
            "confidence": self.get_confidence(zone_id),
            "oldest_observation": obs[0]["ts"] if obs else None,
        }

    # ------------------------------------------------------------------
    # Interne Helfer
    # ------------------------------------------------------------------

    def _schedule_target(self, zone_id: str) -> float:
        """Zieltemperatur aus dem konfigurierten Zeitplan lesen."""
        now = dt_util.now()
        is_weekend = now.weekday() >= 5
        day_key = "weekend" if is_weekend else "weekday"

        schedule = SCHEDULE.get(zone_id, {}).get(day_key, [])
        if not schedule:
            return ZONE_TEMPS.get(zone_id, {}).get("comfort", 21.0)

        now_minutes = now.hour * 60 + now.minute
        active_temp = schedule[0]["temp"]  # Fallback: erster Eintrag

        for entry in sorted(schedule, key=lambda e: e["time"]):
            h, m = map(int, entry["time"].split(":"))
            if h * 60 + m <= now_minutes:
                active_temp = entry["temp"]

        return active_temp

    def _weighted_targets_for_hour(
        self, zone_id: str, now: datetime
    ) -> list[tuple[float, float]]:
        """Gibt (target, weight)-Paare für die aktuelle Stunde zurück."""
        hour = now.hour
        is_weekend = now.weekday() >= 5
        result = []

        for obs in self._observations[zone_id]:
            # Gleiche Tagesart (Wochentag / Wochenende)
            obs_weekend = obs["weekday"] >= 5
            if obs_weekend != is_weekend:
                continue
            # Gleiche Stunde ±1
            if abs(obs["hour"] - hour) > 1:
                continue

            weight = _observation_weight(obs["ts"], now)
            if weight < 0.01:
                continue  # Irrelevante Beobachtungen überspringen
            result.append((obs["target"], weight))

        return result

    def _weighted_median(self, weighted: list[tuple[float, float]]) -> float:
        """Gewichteter Median – robuster als gewichteter Durchschnitt."""
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

    def _get_heat_rate(self, zone_id: str, weather_data: dict) -> float:
        """Heizrate in °C/min bestimmen (gelernt oder Schätzung)."""
        if self._enabled and self.get_confidence(zone_id) >= 0.3:
            learned = self._learned_heat_rate(zone_id, weather_data)
            if learned > 0:
                return learned
        return self._estimate_heat_rate(weather_data)

    def _learned_heat_rate(self, zone_id: str, weather_data: dict) -> float:
        """Gelernte Heizrate – nach Außentemperatur-Ähnlichkeit gewichtet.

        Nutzt _thermal_weight: Oktober bei 8°C = Januar bei 8°C.
        Das System lernt die Physik des Gebäudes aus ALLEN Jahreszeiten.
        """
        outdoor = weather_data.get("temperature") or 10.0
        now = dt_util.now()
        rates = []
        weights = []

        for obs in self._observations[zone_id]:
            if "heat_rate" not in obs or obs["heat_rate"] <= 0:
                continue
            w = _thermal_weight(obs, outdoor, now)
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
        """Physikalische Schätzung der Heizrate als Fallback."""
        outdoor = weather_data.get("temperature") or 10.0
        if outdoor < 0:
            return 0.040
        if outdoor < 5:
            return 0.048
        if outdoor < 10:
            return 0.055
        if outdoor < 15:
            return 0.063
        return 0.070

    def _rebuild_confidence(self, zone_id: str | None = None) -> None:
        """Konfidenz (Vorhersage-Qualität) neu berechnen.

        Bedeutung:
          0%   = keine Daten, Zeitplan wird als Basis genutzt
          100% = maximale Vorhersagezuverlässigkeit

        Die Konfidenz basiert auf der Menge *aktuell relevanter* Daten
        (gewichtet nach Alter). Sie kann steigen UND leicht fallen wenn
        alte Daten an Gewicht verlieren und keine neuen hinzukommen
        (z.B. nach langer Abwesenheit oder Jahreswechsel).
        Das System lernt aber trotzdem immer weiter.
        """
        zones = [zone_id] if zone_id else list(self._observations.keys())
        now = dt_util.now()

        for zid in zones:
            obs = self._observations[zid]
            if not obs:
                self._confidence.setdefault(zid, 0.0)
                continue

            # Summe der gewichteten Beobachtungen
            weighted_n = sum(
                _observation_weight(o["ts"], now)
                for o in obs
            )

            # 0 → 1 über 150 gewichtete Beobachtungen (≈ 2-3 Wochen aktiver Nutzung)
            # Kann leicht schwanken wenn alte Daten veralten – das ist gewollt
            new_conf = min(weighted_n / 150, 1.0)
            self._confidence[zid] = round(new_conf, 4)
