"""Anonymized learning-data export for ThermoSmart.

Creates a JSON snapshot of per-zone learning data that can be shared voluntarily
to support LE 2.0 development.  Nothing is sent automatically — the user decides
whether and how to share the file.

Anonymization contract
----------------------
Exported data contains:
  - ThermoSmart version + format version
  - Export timestamp
  - Per-zone: TRV count, sensor counts, feature flags (booleans)
  - Per-zone: all numeric learning data (observations, rates, confidence, …)

Exported data does NOT contain:
  - Entity IDs or device names (never stored in learning data)
  - Person names or user identifiers
  - HA configuration, tokens, or credentials
  - Exact geographic location data

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

        zones.append({
            "zone_hash": _zone_hash(entry.entry_id),
            **meta,
            "learning": learning,
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
    filename = f"thermosmart_export_{ts_str}.json"
    filepath = os.path.join(www_dir, filename)

    def _write() -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2, ensure_ascii=False, default=str)

    await hass.async_add_executor_job(_write)
    _LOGGER.info("ThermoSmart: anonymized export written → %s", filepath)
    return filepath
