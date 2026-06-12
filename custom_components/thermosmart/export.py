"""Anonymized learning-data export for ThermoSmart.

Creates a JSON snapshot of per-zone learning data that can be shared voluntarily
to support LE 2.0 development.  Nothing is sent automatically — the user decides
whether and how to share the file.

Privacy contract
----------------
Exported data contains:
  - ThermoSmart version + export format version
  - Export timestamp
  - Per-zone: TRV count, sensor counts, feature flags (booleans only)
  - Per-zone: all numeric learning data (observations, rates, confidence, …)

Exported data does NOT contain:
  - Passwords or authentication tokens of any kind
  - Entity IDs, device names, or integration names
  - Person names or user identifiers
  - Street addresses or geographic coordinates

Timestamps are intentionally retained:
  Observation timestamps (ts, hour, minute, weekday) are required for
  longitudinal learning analysis and are the primary reason this data is
  valuable for LE 2.0.  A series of heating timestamps can reveal usage
  patterns (presence, sleep schedule, away periods).  Users should review
  the exported file before sharing it with anyone.

has_forecast note:
  has_forecast is derived from whether a weather entity is configured.
  It does not guarantee that the entity actually provides forecast data —
  some weather integrations only expose current conditions.

Zone identity: each zone_id (HA entry_id UUID) is replaced with a deterministic
12-char hex digest.  Exports from the same installation share the same digests,
making longitudinal data correlatable without being reversible.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    VERSION,
    CONF_WEATHER_ENTITY,
    CONF_OUTDOOR_SOLAR_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_RAIN_SENSOR,
)

if TYPE_CHECKING:
    from .learning_engine import LearningEngine

_LOGGER = logging.getLogger(__name__)

EXPORT_FORMAT_VERSION = 1
_ANON_SALT = "thermosmart_le_export_v1"


def _zone_hash(zone_id: str) -> str:
    """Deterministic 12-char hex digest — correlatable across exports, not reversible."""
    return hashlib.sha256(f"{_ANON_SALT}:{zone_id}".encode()).hexdigest()[:12]


def _zone_meta(cfg: dict) -> dict:
    """Extract anonymized configuration metadata from a zone config dict."""
    def _count(key: str) -> int:
        return len([e for e in cfg.get(key, []) if e])

    weather = cfg.get(CONF_WEATHER_ENTITY) or None
    return {
        "trv_count": _count("climate_entities"),
        "temp_sensor_count": _count("temp_sensors"),
        "humidity_sensor_count": _count("humidity_sensors"),
        "has_window_sensors": _count("window_sensors") > 0,
        "window_sensor_count": _count("window_sensors"),
        "has_presence": _count("presence_persons") > 0,
        "has_weather": bool(weather),
        "has_forecast": bool(weather),
        "has_solar_sensor": bool(cfg.get(CONF_OUTDOOR_SOLAR_SENSOR)),
        "has_wind_sensor": bool(cfg.get(CONF_OUTDOOR_WIND_SENSOR)),
        "has_outdoor_temp_sensor": bool(cfg.get(CONF_OUTDOOR_TEMP_SENSOR)),
        "has_outdoor_humidity_sensor": bool(cfg.get(CONF_OUTDOOR_HUMIDITY_SENSOR)),
        "has_rain_sensor": bool(cfg.get(CONF_OUTDOOR_RAIN_SENSOR)),
    }


def _compute_analytics(learning: dict) -> dict:
    """Compute per-zone analytics from existing observations at export time.

    Pure calculation — no new fields are written to storage, no counters are
    maintained at runtime.  All inputs come from learning data already present.
    """
    observations: list[dict] = learning.get("observations", [])
    trv_observations: list[dict] = learning.get("trv_observations", [])

    # --- observation_span_days -------------------------------------------
    span_days = 0.0
    if len(observations) >= 2:
        try:
            first_ts = datetime.fromisoformat(observations[0]["ts"])
            last_ts = datetime.fromisoformat(observations[-1]["ts"])
            span_days = round((last_ts - first_ts).total_seconds() / 86400, 2)
        except (KeyError, ValueError):
            span_days = 0.0

    # --- target_changes (transitions) ------------------------------------
    target_changes = 0
    prev_target = None
    for obs in observations:
        t = obs.get("target")
        if prev_target is not None and t != prev_target:
            target_changes += 1
        prev_target = t

    target_changes_per_day = (
        round(target_changes / span_days, 3) if span_days > 0 else 0.0
    )

    # --- delta stats -------------------------------------------------------
    deltas = [obs["delta"] for obs in observations if "delta" in obs]
    avg_delta = round(sum(deltas) / len(deltas), 3) if deltas else 0.0
    max_undershoot = round(min(deltas), 3) if deltas else 0.0
    pct_obs_at_target = (
        round(sum(1 for d in deltas if d >= 0) / len(deltas) * 100, 1)
        if deltas else 0.0
    )

    # --- heat_rate / norm_heat_rate ----------------------------------------
    heat_rates = [obs["heat_rate"] for obs in observations if "heat_rate" in obs]
    heat_rate_obs_count = len(heat_rates)
    heat_rate_mean = (
        round(sum(heat_rates) / heat_rate_obs_count, 5)
        if heat_rate_obs_count else None
    )

    norm_rates = [obs["norm_heat_rate"] for obs in observations if "norm_heat_rate" in obs]
    norm_heat_rate_mean = (
        round(sum(norm_rates) / len(norm_rates), 6) if norm_rates else None
    )

    # --- setpoint_excess from trv_observations ----------------------------
    excesses = [
        o["setpoint_excess"] for o in trv_observations if "setpoint_excess" in o
    ]
    avg_setpoint_excess = (
        round(sum(excesses) / len(excesses), 3) if excesses else None
    )

    return {
        "observation_span_days": span_days,
        "target_changes": target_changes,
        "target_changes_per_day": target_changes_per_day,
        "avg_delta": avg_delta,
        "max_undershoot": max_undershoot,
        "pct_obs_at_target": pct_obs_at_target,
        "heat_rate_obs_count": heat_rate_obs_count,
        "heat_rate_mean": heat_rate_mean,
        "norm_heat_rate_mean": norm_heat_rate_mean,
        "avg_setpoint_excess": avg_setpoint_excess,
    }


async def async_export_learning_data(hass: HomeAssistant) -> str:
    """Build anonymized export, write to /config/www/, return absolute file path."""
    le: LearningEngine | None = hass.data.get(DOMAIN, {}).get("learning_engine")
    now: datetime = dt_util.now()
    ts_str = now.strftime("%Y%m%dT%H%M%S")

    zones: list[dict] = []

    for entry in hass.config_entries.async_entries(DOMAIN):
        cfg = {**entry.data, **entry.options}
        if cfg.get("entry_type") == "system":
            continue

        meta = _zone_meta(cfg)
        learning: dict = le.get_export_data(entry.entry_id) if le is not None else {}
        analytics = _compute_analytics(learning)

        zones.append({
            "zone_hash": _zone_hash(entry.entry_id),
            **meta,
            "learning": learning,
            "analytics": analytics,
        })

    export: dict = {
        "thermosmart_version": VERSION,
        "export_format_version": EXPORT_FORMAT_VERSION,
        "export_timestamp": now.isoformat(),
        "zone_count": len(zones),
        "total_trv_count": sum(z["trv_count"] for z in zones),
        "total_temp_sensor_count": sum(z["temp_sensor_count"] for z in zones),
        "total_humidity_sensor_count": sum(z["humidity_sensor_count"] for z in zones),
        "zones": zones,
    }

    www_dir = hass.config.path("www")
    os.makedirs(www_dir, exist_ok=True)
    random_suffix = os.urandom(3).hex()
    filename = f"thermosmart_export_{ts_str}_{random_suffix}.json"
    filepath = os.path.join(www_dir, filename)

    def _write() -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2, ensure_ascii=False, default=str)

    await hass.async_add_executor_job(_write)
    _LOGGER.info("ThermoSmart: anonymized export written → %s", filepath)
    return filepath


def build_export_notification_message(filename: str) -> str:
    """Return the persistent notification message for a completed export.

    Single source of truth — used by both the service handler and the button entity.
    """
    return (
        f"Export saved to `/config/www/{filename}`.\n\n"
        "To open the export file, append this path to your Home Assistant URL:\n\n"
        f"`/local/{filename}`\n\n"
        "The file will open as JSON in your browser. "
        "To save it, use **Save Page As…** (Ctrl+S / Cmd+S).\n\n"
        "Review the file before sharing. No data has been sent anywhere."
    )
