"""ThermoSmart diagnostics — Home Assistant's standard config-entry diagnostics hook.

Replaces the previous custom export system (private file storage, an
authenticated HTTP download view, persistent-notification links, buttons,
and a service) with HA's built-in "Download diagnostics" action under
Settings -> Devices & services -> ThermoSmart -> (each entry) -> Download
diagnostics. No file is written and no HTTP endpoint is registered — Home
Assistant handles delivery, including authentication, entirely on its own.

Each config entry gets its own diagnostics payload:
  - the "system" entry (global Summer/Vacation controls) returns a small
    overview (versions, configured zone count/hashes) — repeating every
    zone's full diagnostics there too would just be noise.
  - each "zone" entry returns that zone's diagnostic data, built from the
    same privacy-scanned data-shaping helpers already in export.py.

Deliberately NOT surfaced here (see the export -> diagnostics migration
audit this module was introduced for):
  - raw Learning model coefficients / adaptation-candidate internals
    (export.py's former _learning_research_data()) — genuinely internal
    research-engine detail, not actionable for support/debugging. The
    calibrated, human-readable "learning_progress" block already answers
    the support-relevant question ("how well is Learning doing, and why").
  - the always-empty "reserved" application/orchestration-layer placeholder
    blocks — that layer is foundation-only and permanently inactive in this
    version (see export.py's removed _learning_reserved_diagnostics_summary),
    so they would only add noise to a diagnostics download.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import __version__ as HA_VERSION
from homeassistant.core import HomeAssistant

from .const import DOMAIN, VERSION
from .export import (
    _compute_analytics,
    _device_profile_export,
    _historical_learning_snapshot_for_research,
    _learning_adaptation_history_summary,
    _learning_adaptation_summary,
    _learning_critical_events_export,
    _learning_episode_history_export,
    _learning_health_data,
    _learning_pending_data,
    _learning_progress_export,
    _learning_research_daily_export,
    _learning_storage_context_export,
    _resolve_clock,
    _zone_hash,
    _zone_meta,
)

DIAGNOSTICS_SCHEMA_VERSION = 1


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry,
) -> dict[str, Any]:
    """Return ThermoSmart diagnostics for one config entry (HA standard hook)."""
    cfg = {**entry.data, **entry.options}
    now = _resolve_clock(hass) or datetime.now(timezone.utc)

    if cfg.get("entry_type") == "system":
        return _system_diagnostics(hass, now=now)

    return await _zone_diagnostics(hass, entry, cfg, now=now)


def _system_diagnostics(hass: HomeAssistant, *, now: datetime) -> dict[str, Any]:
    """Small overview for the "system" entry (global Summer/Vacation controls,
    no coordinator of its own) — the per-zone data lives on each zone's own
    entry instead of being repeated here."""
    zone_entries = [
        e for e in hass.config_entries.async_entries(DOMAIN)
        if {**e.data, **e.options}.get("entry_type") != "system"
    ]
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "thermosmart_version": VERSION,
        "ha_version": HA_VERSION,
        "generated_at_utc": now.isoformat(),
        "zone_count": len(zone_entries),
        "zones": [{"zone_hash": _zone_hash(e.entry_id)} for e in zone_entries],
    }


async def _zone_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry, cfg: dict, *, now: datetime,
) -> dict[str, Any]:
    """Full diagnostics for one heating zone.

    Every ``_learning_*`` helper below already tolerates ``coord is None``
    (no coordinator attached yet, e.g. right after setup) and returns an
    explicit ``{"available": False, ...}``/``None`` fallback on its own —
    so this function calls them uniformly instead of duplicating that
    None-check per field.
    """
    entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    coord = entry_data.get("coordinator") if isinstance(entry_data, dict) else None

    le = hass.data.get(DOMAIN, {}).get("learning_engine")
    learning: dict = le.get_export_data(entry.entry_id) if le is not None else {}

    runtime_state = None
    if coord is not None:
        runtime_state = {
            "active_control": getattr(coord, "_active_control", None),
            "learning_enabled": getattr(coord, "_learning_enabled", None),
            "current_mode": getattr(coord, "_current_mode", None),
            "confidence": round(
                float(((coord.data or {}).get("learning_confidence") or 0.0)), 3
            ),
        }

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "thermosmart_version": VERSION,
        "generated_at_utc": now.isoformat(),
        "zone_hash": _zone_hash(entry.entry_id),
        "config_flags": _zone_meta(cfg),
        "analytics": _compute_analytics(learning),
        "historical_learning_snapshot": _historical_learning_snapshot_for_research(learning),
        "runtime_state": runtime_state,
        "runtime_health": _learning_health_data(coord),
        "runtime_pending": _learning_pending_data(coord, entry.entry_id),
        "learning_progress": _learning_progress_export(coord),
        "episode_history": _learning_episode_history_export(coord, now=now),
        "research_daily": _learning_research_daily_export(coord, now=now),
        "critical_events": _learning_critical_events_export(coord, now=now),
        "adaptation": _learning_adaptation_summary(coord, entry.entry_id),
        "adaptation_history": _learning_adaptation_history_summary(coord, entry.entry_id),
        "device_profile": _device_profile_export(coord),
        "storage_context": await _learning_storage_context_export(
            hass, entry.entry_id, now=now,
        ),
    }
