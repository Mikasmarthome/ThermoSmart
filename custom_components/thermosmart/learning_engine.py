"""LearningEngine – adaptive heating schedule and preheat time learning."""
from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    STORAGE_VERSION,
    STORAGE_KEY,
    LEARNING_WINDOW_DAYS,
    LEARNING_MIN_SAMPLES,
    PREHEAT_MAX_MINUTES,
    PREHEAT_MIN_DELTA,
    TEMP_COMFORT,
    TEMP_NIGHT,
    TEMP_AWAY,
    TEMP_BATHROOM_COMFORT,
    TEMP_GASTE_WC_COMFORT,
    TEMP_KELLER_COMFORT,
)

_LOGGER = logging.getLogger(__name__)

# Fallback schedule: {zone_id: {hour: target_temp}}
DEFAULT_SCHEDULE: dict[str, dict] = {
    "wohnbereich": {
        5: TEMP_COMFORT,   # 05:30 weekday morning
        9: TEMP_AWAY,      # everyone gone
        16: TEMP_COMFORT,  # afternoon
        22: TEMP_NIGHT,    # night setback
    },
    "schlafbereich": {
        5: TEMP_COMFORT,
        9: TEMP_AWAY,
        16: TEMP_COMFORT,
        22: TEMP_NIGHT,
    },
    "badezimmer": {
        5: TEMP_BATHROOM_COMFORT,
        9: TEMP_AWAY,
        16: TEMP_BATHROOM_COMFORT,
        22: TEMP_NIGHT,
    },
    "gaste_wc": {
        6: TEMP_GASTE_WC_COMFORT,
        22: TEMP_NIGHT,
    },
    "keller": {
        0: TEMP_KELLER_COMFORT,  # constant low
    },
}


class LearningEngine:
    """
    Persists zone observations and derives:
      1. An adaptive base target temperature for the current time of day.
      2. A preheat lead-time (in minutes) needed to reach the target.
    """

    def __init__(self, hass: HomeAssistant, enabled: bool = True) -> None:
        self._hass = hass
        self._enabled = enabled
        self._store: Store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        # {zone_id: [{"ts": iso, "hour": int, "weekday": int,
        #              "target": float, "indoor_temp": float,
        #              "outdoor_temp": float, "minutes_to_target": int}]}
        self._observations: dict[str, list[dict]] = defaultdict(list)
        # {zone_id: float} cached confidence 0.0–1.0
        self._confidence: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def async_load(self) -> None:
        """Load persisted observations from HA storage."""
        stored = await self._store.async_load()
        if stored and isinstance(stored, dict):
            for zone_id, obs_list in stored.get("observations", {}).items():
                self._observations[zone_id] = obs_list
            _LOGGER.debug("LearningEngine: loaded %d zones from storage", len(self._observations))
        self._prune_old_observations()
        self._rebuild_confidence()

    async def async_save(self) -> None:
        """Persist current observations to HA storage."""
        await self._store.async_save({"observations": dict(self._observations)})
        _LOGGER.debug("LearningEngine: saved observations")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def async_observe(
        self, zone_id: str, recommendation: dict, weather_data: dict
    ) -> None:
        """Record an observation after the coordinator runs."""
        if not self._enabled:
            return

        current_temp = recommendation.get("current_temp")
        adjusted_target = recommendation.get("adjusted_target")
        if current_temp is None or adjusted_target is None:
            return

        now = dt_util.now()
        obs = {
            "ts": now.isoformat(),
            "hour": now.hour,
            "weekday": now.weekday(),          # 0=Mon … 6=Sun
            "target": adjusted_target,
            "indoor_temp": current_temp,
            "outdoor_temp": weather_data.get("temperature"),
            "delta": round(adjusted_target - current_temp, 2),
        }
        self._observations[zone_id].append(obs)

        # Keep only last N days
        self._prune_old_observations(zone_id)
        self._rebuild_confidence(zone_id)

    async def async_get_base_target(self, zone_id: str) -> float:
        """
        Return the expected target temperature for the current hour.
        Falls back to the hardcoded default schedule when insufficient data.
        """
        if not self._enabled or self.get_confidence(zone_id) < 0.3:
            return self._schedule_target(zone_id)

        now = dt_util.now()
        hour = now.hour
        weekday = now.weekday()
        is_weekend = weekday >= 5

        # Filter observations for same hour ±1h and same day-type
        relevant = [
            o for o in self._observations[zone_id]
            if abs(o["hour"] - hour) <= 1
            and (o["weekday"] >= 5) == is_weekend
        ]
        if len(relevant) < LEARNING_MIN_SAMPLES:
            return self._schedule_target(zone_id)

        targets = [o["target"] for o in relevant]
        learned = round(statistics.median(targets), 1)
        _LOGGER.debug(
            "LearningEngine [%s] h=%d learned target=%.1f (n=%d)",
            zone_id, hour, learned, len(relevant),
        )
        return learned

    async def async_get_preheat_minutes(
        self,
        zone_id: str,
        target: float,
        current_temp: float | None,
        weather_data: dict,
    ) -> int:
        """
        Estimate how many minutes in advance the TRV must start heating
        to reach `target` from `current_temp`.

        Uses stored heat-up rate observations.  Falls back to a simple
        physics-based model when learning data is sparse.
        """
        if current_temp is None:
            return 0

        delta = target - current_temp
        if delta < PREHEAT_MIN_DELTA:
            return 0

        if self._enabled and self.get_confidence(zone_id) >= 0.4:
            rate = self._learned_heat_rate(zone_id, weather_data)
        else:
            rate = self._estimate_heat_rate(weather_data)

        if rate <= 0:
            return 0

        minutes = int(min(delta / rate * 60, PREHEAT_MAX_MINUTES))
        _LOGGER.debug(
            "LearningEngine [%s] preheat %d min (delta=%.1f, rate=%.3f °C/min)",
            zone_id, minutes, delta, rate,
        )
        return minutes

    def get_confidence(self, zone_id: str) -> float:
        """Return confidence score 0.0–1.0 for the given zone."""
        return self._confidence.get(zone_id, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_target(self, zone_id: str) -> float:
        """Look up the default schedule target for the current hour."""
        schedule = DEFAULT_SCHEDULE.get(zone_id, {})
        if not schedule:
            return TEMP_COMFORT

        now_hour = dt_util.now().hour
        # Find the latest scheduled hour that is <= now
        past_hours = [h for h in sorted(schedule.keys()) if h <= now_hour]
        if past_hours:
            return schedule[max(past_hours)]
        # Wrap around to the last entry of the previous day
        return schedule[max(schedule.keys())]

    def _learned_heat_rate(self, zone_id: str, weather_data: dict) -> float:
        """
        Derive °C/min heat-up rate from stored observations where the
        room was warming up, filtered by similar outdoor temperature.
        """
        outdoor = weather_data.get("temperature")
        obs = self._observations[zone_id]

        warming = [
            o for o in obs
            if o["delta"] > 0
            and (outdoor is None or abs((o.get("outdoor_temp") or outdoor) - outdoor) < 5)
        ]
        if len(warming) < LEARNING_MIN_SAMPLES:
            return self._estimate_heat_rate(weather_data)

        # Approximate: assume the coordinator runs every 5 min and tracks delta
        # A proper implementation would store consecutive measurements; for now
        # use the mean delta / 30 min as a proxy rate.
        deltas = [o["delta"] for o in warming]
        return round(statistics.mean(deltas) / 30, 4)

    def _estimate_heat_rate(self, weather_data: dict) -> float:
        """
        Physics-based fallback: typical TRV heat-up rate depends on
        outdoor temperature (cold outside → slower due to heat loss).
        Returns °C/min.
        """
        outdoor = weather_data.get("temperature") or 10.0
        if outdoor < 0:
            return 0.04   # slow in very cold weather
        if outdoor < 10:
            return 0.055
        return 0.07

    def _prune_old_observations(self, zone_id: str | None = None) -> None:
        """Remove observations older than LEARNING_WINDOW_DAYS."""
        cutoff = dt_util.now() - timedelta(days=LEARNING_WINDOW_DAYS)
        zones = [zone_id] if zone_id else list(self._observations.keys())
        for zid in zones:
            before = len(self._observations[zid])
            self._observations[zid] = [
                o for o in self._observations[zid]
                if datetime.fromisoformat(o["ts"]) > cutoff
            ]
            pruned = before - len(self._observations[zid])
            if pruned:
                _LOGGER.debug("LearningEngine [%s] pruned %d old observations", zid, pruned)

    def _rebuild_confidence(self, zone_id: str | None = None) -> None:
        """Recompute confidence score(s) based on observation count."""
        zones = [zone_id] if zone_id else list(self._observations.keys())
        for zid in zones:
            n = len(self._observations[zid])
            # Confidence grows linearly from 0 → 1 over 100 observations
            self._confidence[zid] = min(n / 100, 1.0)
