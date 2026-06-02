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
import statistics
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
    CONF_SCHED_WD_MORNING, CONF_SCHED_WD_DAY, CONF_SCHED_WD_DAY_TEMP,
    CONF_SCHED_WD_EVENING, CONF_SCHED_WD_NIGHT,
    CONF_SCHED_WE_MORNING, CONF_SCHED_WE_NIGHT,
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


def _seasonal_weight(obs_ts: str, now: datetime) -> float:
    """Saisonale Gewichtung für Jahreszeit-sensitive Berechnungen.
    Letzter Dezember zählt mehr als der Juli dieses Jahres.
    """
    try:
        obs_dt = datetime.fromisoformat(obs_ts)
    except (ValueError, TypeError):
        return 0.0
    age_days = max((now - obs_dt).total_seconds() / 86400, 0)
    obs_doy = obs_dt.timetuple().tm_yday
    now_doy = now.timetuple().tm_yday
    day_dist = abs(obs_doy - now_doy)
    day_dist = min(day_dist, 365 - day_dist)
    seasonal = math.exp(-(day_dist / 45.0) ** 2)
    recency = math.exp(-age_days / 365 * 0.25)
    return round(seasonal * recency, 6)


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

    def __init__(self, hass: HomeAssistant, enabled: bool = True) -> None:
        self._hass = hass
        self._enabled = enabled
        self._store: Store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._observations: dict[str, list[dict]] = defaultdict(list)
        self._last_temp: dict[str, tuple[datetime, float]] = {}
        self._last_idle_temp: dict[str, tuple[datetime, float]] = {}
        self._confidence: dict[str, float] = {}
        self._boost_factors: dict[str, float] = {}

    # ── Persistenz ──────────────────────────────────────────────────────

    async def async_load(self) -> None:
        stored = await self._store.async_load()
        if stored and isinstance(stored, dict):
            for zone_id, obs_list in stored.get("observations", {}).items():
                self._observations[zone_id] = obs_list
            self._boost_factors = stored.get("boost_factors", {})
            total = sum(len(v) for v in self._observations.values())
            _LOGGER.info(
                "LearningEngine: %d Zonen, %d Beobachtungen geladen",
                len(self._observations), total,
            )
        self._rebuild_confidence()

    async def async_save(self) -> None:
        await self._store.async_save({
            "observations": dict(self._observations),
            "boost_factors": self._boost_factors,
        })

    # ── API ─────────────────────────────────────────────────────────────

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    async def async_observe(
        self,
        zone_id: str,
        recommendation: dict,
        weather_data: dict,
        indoor_humidity: float | None = None,
    ) -> None:
        """Beobachtung aufzeichnen – alle verfügbaren Bedingungen speichern."""
        if not self._enabled:
            return

        current_temp = recommendation.get("current_temp")
        adjusted_target = recommendation.get("adjusted_target")
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
        # → passiert wenn Heizung ausgefallen ist oder Sommer-Nacht
        cool_rate: float | None = None
        is_cooling = current_temp < adjusted_target - 0.5
        if is_cooling and zone_id in self._last_idle_temp:
            idle_time, idle_temp = self._last_idle_temp[zone_id]
            elapsed_min = (now - idle_time).total_seconds() / 60
            if 10 <= elapsed_min <= 60 and current_temp < idle_temp:
                cool_rate = round((idle_temp - current_temp) / elapsed_min, 5)
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

        if indoor_humidity is not None:
            obs["indoor_humidity"] = indoor_humidity
        if heat_rate is not None:
            obs["heat_rate"] = heat_rate
        if cool_rate is not None:
            obs["cool_rate"] = cool_rate

        self._observations[zone_id].append(obs)
        self._rebuild_confidence(zone_id)

        # Alle 50 Beobachtungen speichern
        if sum(len(v) for v in self._observations.values()) % 50 == 0:
            await self.async_save()

    async def async_get_base_target(
        self,
        zone_id: str,
        mode: str = HEATING_MODE_AUTO,
        comfort_temp: float = 21.0,
        night_temp: float = 18.0,
        away_temp: float = 17.0,
        schedule_cfg: dict | None = None,
    ) -> float:
        """Empfohlene Zieltemperatur – Zeitplan oder Modus-Temperatur."""
        mode_temps = {
            "comfort": comfort_temp,
            "night": night_temp,
            "away": away_temp,
            "vacation": 12.0,
        }
        if mode != HEATING_MODE_AUTO:
            return mode_temps.get(mode, night_temp)

        schedule_temp = self._schedule_target(
            zone_id, comfort_temp, night_temp, away_temp, schedule_cfg
        )

        if not self._enabled or self.get_confidence(zone_id) < 0.25:
            return schedule_temp

        now = dt_util.now()
        weighted_temps = self._weighted_targets_for_hour(zone_id, now)
        if len(weighted_temps) < LEARNING_MIN_SAMPLES:
            return schedule_temp

        learned = self._weighted_median(weighted_temps)
        _LOGGER.debug(
            "LearningEngine [%s] Zieltemp=%.1f°C (Zeitplan=%.1f°C, n=%d)",
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
        """Vorheizzeit berechnen – Multi-Faktor-Heizrate nutzen."""
        if current_temp is None:
            return 0
        delta = target - current_temp
        if delta < PREHEAT_MIN_DELTA:
            return 0

        rate = self._get_heat_rate(zone_id, weather_data)
        if rate <= 0:
            return 0

        # Effektive Heizrate = Heizrate minus Abkühlrate (Haus kühlt während Vorheizen weiter ab)
        cool_rate = self._get_avg_cool_rate(zone_id)
        effective_rate = max(rate - cool_rate, rate * 0.3)

        minutes = int(min(delta / effective_rate * 60, PREHEAT_MAX_MINUTES))
        _LOGGER.debug(
            "LearningEngine [%s] Vorheizzeit=%d min (Δ%.1f°C, Heiz=%.4f°C/min, Kühl=%.4f°C/min)",
            zone_id, minutes, delta, rate, cool_rate,
        )
        return minutes

    def get_confidence(self, zone_id: str) -> float:
        return self._confidence.get(zone_id, 0.0)

    def get_boost_factor(self, zone_id: str) -> float:
        return self._boost_factors.get(zone_id, 1.0)

    def update_boost_factor(self, zone_id: str, overshot: bool) -> None:
        """Boost-Faktor nach Heizzyklus anpassen.

        Überschießen → Faktor reduzieren (Ventil war zu weit auf).
        Nur Erhöhen wenn explizit langsam (slow=True) – kommt später.
        """
        if not self._enabled:
            return
        factor = self._boost_factors.get(zone_id, 1.0)
        if overshot:
            factor = max(0.5, round(factor * 0.92, 3))
            _LOGGER.info(
                "LearningEngine [%s] Boost-Faktor reduziert → %.3f (Überschießen)",
                zone_id, factor,
            )
        self._boost_factors[zone_id] = factor

    def get_stats(self, zone_id: str) -> dict:
        obs = self._observations[zone_id]
        now = dt_util.now()
        # Zähle Beobachtungen mit verschiedenen Faktoren
        with_wind = sum(1 for o in obs if o.get("wind_speed") is not None)
        with_solar = sum(1 for o in obs if o.get("solar_radiation") is not None)
        with_humidity = sum(1 for o in obs if o.get("indoor_humidity") is not None)
        with_heat_rate = sum(1 for o in obs if o.get("heat_rate") is not None)
        with_cool_rate = sum(1 for o in obs if o.get("cool_rate") is not None)
        avg_cool_rate = None
        cool_samples = [o["cool_rate"] for o in obs if o.get("cool_rate")]
        if cool_samples:
            avg_cool_rate = round(sum(cool_samples) / len(cool_samples), 5)
        return {
            "total_observations": len(obs),
            "confidence": self.get_confidence(zone_id),
            "with_wind_data": with_wind,
            "with_solar_data": with_solar,
            "with_humidity_data": with_humidity,
            "with_heat_rate": with_heat_rate,
            "with_cool_rate": with_cool_rate,
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
                h, m = str(raw).split(":")
                return int(h) * 60 + int(m)
            except (ValueError, AttributeError):
                h2, m2 = fallback.split(":")
                return int(h2) * 60 + int(m2)

        if not is_weekend:
            morning  = t(CONF_SCHED_WD_MORNING, "06:00")
            day      = t(CONF_SCHED_WD_DAY,     "09:00")
            evening  = t(CONF_SCHED_WD_EVENING, "17:00")
            night    = t(CONF_SCHED_WD_NIGHT,   "22:00")
            day_temp = float((schedule_cfg or {}).get(CONF_SCHED_WD_DAY_TEMP, away_temp))
            if cur < morning or cur >= night:
                return night_temp
            if cur < day:
                return comfort_temp   # Aufwachen
            if cur < evening:
                return day_temp       # Tagsüber (z.B. alle bei der Arbeit)
            return comfort_temp       # Abend
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
        """Durchschnittliche Abkühlrate aus gespeicherten Beobachtungen."""
        samples = [o["cool_rate"] for o in self._observations[zone_id] if o.get("cool_rate")]
        if not samples:
            return 0.0
        # Letzte 50 Messungen – aktuellere Saison gewichtet
        recent = samples[-50:]
        return round(sum(recent) / len(recent), 5)

    def _get_heat_rate(self, zone_id: str, weather_data: dict) -> float:
        if self._enabled and self.get_confidence(zone_id) >= 0.3:
            learned = self._learned_heat_rate_multifactor(zone_id, weather_data)
            if learned > 0:
                return learned
        return self._estimate_heat_rate(weather_data)

    def _learned_heat_rate_multifactor(
        self, zone_id: str, weather_data: dict
    ) -> float:
        """Multi-Faktor Heizrate: lernt aus allen Außenbedingungen.

        Oktober bei 8°C + Wind 5m/s = Januar bei 8°C + Wind 5m/s.
        Sommer bei 8°C + Sonne 600W/m² hebt sich heraus → niedrigere Rate.
        """
        now = dt_util.now()
        conditions = {
            "outdoor_temp": weather_data.get("temperature") or 10.0,
            "wind_speed": weather_data.get("wind_speed"),
            "solar_radiation": weather_data.get("solar_radiation"),
            "outdoor_humidity": weather_data.get("humidity"),
        }

        rates = []
        weights = []
        for obs in self._observations[zone_id]:
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
        """Physikalische Schätzung als Fallback."""
        outdoor = weather_data.get("temperature") or 10.0
        wind = weather_data.get("wind_speed") or 0.0
        solar = weather_data.get("solar_radiation") or 0.0

        # Basis-Rate nach Außentemperatur
        if outdoor < 0:
            base = 0.040
        elif outdoor < 5:
            base = 0.048
        elif outdoor < 10:
            base = 0.055
        elif outdoor < 15:
            base = 0.063
        else:
            base = 0.070

        # Wind erhöht Wärmeverlust → langsamere Aufheizung
        if wind > 5:
            wind_factor = 1.0 - min(wind / 30, 0.20)
            base *= wind_factor

        # Sonne hilft beim Aufheizen → schneller
        if solar > 300:
            solar_boost = min((solar - 300) / 1000, 0.15)
            base *= (1.0 + solar_boost)

        return round(base, 5)

    def _rebuild_confidence(self, zone_id: str | None = None) -> None:
        """Konfidenz = Qualität der aktuell relevanten Datenbasis."""
        zones = [zone_id] if zone_id else list(self._observations.keys())
        now = dt_util.now()

        for zid in zones:
            obs = self._observations[zid]
            if not obs:
                self._confidence.setdefault(zid, 0.0)
                continue

            # Gewichtete effektive Beobachtungen (zeitbasiert)
            weighted_n = sum(_time_weight(o["ts"], now) for o in obs)

            # Bonus für Datenvielfalt (mehr Faktoren = höhere Qualität)
            has_wind = any(o.get("wind_speed") is not None for o in obs)
            has_solar = any(o.get("solar_radiation") is not None for o in obs)
            has_humidity = any(o.get("indoor_humidity") is not None for o in obs)
            has_heat_rate = any(o.get("heat_rate") is not None for o in obs)
            diversity_bonus = 1.0 + sum([has_wind, has_solar, has_humidity, has_heat_rate]) * 0.05

            new_conf = min(weighted_n / 150 * diversity_bonus, 1.0)
            self._confidence[zid] = round(new_conf, 4)
