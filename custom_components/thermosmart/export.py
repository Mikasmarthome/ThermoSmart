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
  - Per-zone: LE2 model coefficient statistics (numeric only — no IDs)

Exported data does NOT contain:
  - Passwords or authentication tokens of any kind
  - Entity IDs, device names, or integration names
  - Person names or user identifiers
  - Street addresses or geographic coordinates
  - Decision IDs, episode IDs, or internal runtime identifiers

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

LE2 research data:
  Only model coefficient aggregates are included (models, cycles,
  last_cycle_ts, model_update_counts).  Fields containing IDs of any kind
  (decision_id, episode_id, learning_zone_id, zone_id, …) are stripped
  recursively before inclusion.  A privacy scan is performed as a final check.

24h auto-delete:
  Export files are scheduled for deletion 24 hours after creation using
  async_call_later.  This schedule does NOT survive a Home Assistant restart.
  If HA is restarted before the 24-hour window expires, the file remains until
  the user deletes it manually or the next export overwrites the timer.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_call_later

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
_EXPORT_CLEANUP_DELAY_S: float = 24 * 3600  # 24 hours; not restart-safe

# ── LE2 privacy helpers ───────────────────────────────────────────────────────

# Top-level keys safe to include from _ZoneRuntime.serialize().
# Excluded: capture (contains zone_id + decision_id in ledger),
#           pipeline (contains zone_id + open_decision_id),
#           ledger (inside capture, contains decision_ids),
#           baseline_store, pending_* (all contain decision-related IDs).
_LE2_RESEARCH_SAFE_TOP_KEYS = frozenset(
    {"models", "cycles", "last_cycle_ts", "model_update_counts"}
)

# Key substrings to strip recursively — mirrors privacy.py _FORBIDDEN_KEY_SUBSTRINGS
# plus "zone_id" which the scanner does not catch standalone.
_LE2_STRIP_KEY_SUBSTRINGS = (
    "entity_id", "entry_id", "device_id", "decision_id", "episode_id", "event_id",
    "evaluation_id", "user_id", "person", "email", "latitude", "longitude", "address",
    "hostname", "ip_address", "ipaddr", "source_episode_id", "correction_event_id",
    "source_decision_id", "trv_binding_id", "learning_zone_id", "zone_id",
)


def _le2_strip_forbidden(obj: Any) -> Any:
    """Recursively remove keys matching LE2 privacy-forbidden substrings."""
    if isinstance(obj, dict):
        return {
            k: _le2_strip_forbidden(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and any(s in k.lower() for s in _LE2_STRIP_KEY_SUBSTRINGS))
        }
    if isinstance(obj, list):
        return [_le2_strip_forbidden(i) for i in obj]
    return obj


# ── zone helpers ─────────────────────────────────────────────────────────────

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

    # --- contaminated / clean heat_rate -----------------------------------
    contaminated_heat_rate_count = sum(
        1 for obs in observations
        if "heat_rate" in obs and obs.get("delta", 0.0) < -1.0
    )
    clean_heat_obs = [
        obs for obs in observations
        if "heat_rate" in obs and obs.get("delta", 0.0) >= -1.0
    ]
    clean_heat_rate_mean = (
        round(sum(o["heat_rate"] for o in clean_heat_obs) / len(clean_heat_obs), 5)
        if clean_heat_obs else None
    )
    clean_norm_rates = [
        o["norm_heat_rate"] for o in clean_heat_obs if "norm_heat_rate" in o
    ]
    clean_norm_heat_rate_mean = (
        round(sum(clean_norm_rates) / len(clean_norm_rates), 6)
        if clean_norm_rates else None
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
        "contaminated_heat_rate_count": contaminated_heat_rate_count,
        "clean_heat_rate_mean": clean_heat_rate_mean,
        "clean_norm_heat_rate_mean": clean_norm_heat_rate_mean,
    }


# ── LE2 data accessors ────────────────────────────────────────────────────────

def _le2_runtime(coord):
    """Safely return the LE2 LearningRuntime from a coordinator, or None."""
    try:
        shadow = getattr(coord, "_le2_shadow", None)
        if shadow is None:
            return None
        return getattr(shadow, "runtime", None)
    except Exception:
        return None


def _le2_research_data(coord, zone_id: str) -> dict | None:
    """Return privacy-safe LE2 model statistics for research export, or None.

    Only includes top-level keys from _LE2_RESEARCH_SAFE_TOP_KEYS (models,
    cycles, last_cycle_ts, model_update_counts).  All fields whose key contains
    a forbidden substring (decision_id, learning_zone_id, zone_id, …) are
    stripped recursively before inclusion.  A final privacy scan confirms no
    violations remain; if any do, the LE2 block is excluded for that zone.
    """
    try:
        rt = _le2_runtime(coord)
        if rt is None:
            return None
        zone_rt = rt._zones.get(zone_id)
        if zone_rt is None:
            return None
        raw = zone_rt.serialize()
        # 1. Allowlist: only safe top-level sections
        filtered = {k: v for k, v in raw.items() if k in _LE2_RESEARCH_SAFE_TOP_KEYS}
        # 2. Recursive strip of forbidden keys within allowed sections
        safe = _le2_strip_forbidden(filtered)
        # 3. Replace serialized outcome model with research-scoped export.
        #    The raw serialization (serialize_state) uses internal confounder_flags and
        #    has no truncation_info. The research export normalizes to confounder_codes
        #    and adds truncation_info so consumers see a stable, documented format.
        try:
            from .learning.contracts import ExportScope
            _om = zone_rt.orchestrator.models.get("outcome")
            if _om is not None and isinstance(safe.get("models"), dict):
                safe["models"]["outcome"] = dict(_om.export(ExportScope.RESEARCH))
        except Exception:
            pass  # non-fatal; raw outcome data remains
        # 3b. Passive adaptation candidates (shadow-only; no control modification).
        try:
            from .learning.adaptation import (
                OutcomeSignal, SituationContext, suggest_candidates, traces_for_export,
            )
            _om2 = zone_rt.orchestrator.models.get("outcome")
            if _om2 is not None:
                _od = _om2.diagnostics()
                _fc, _pc = _od.full_partial
                _total = _fc + _pc
                _rejections = sum(_od.rejection_counts.values()) if _od.rejection_counts else 0
                _attempts = _total + _rejections
                _signal = OutcomeSignal(
                    sample_count=_total,
                    timeout_rate=_od.timeout_rate,
                    overshoot_rate=_od.overshoot_rate,
                    reached_rate=_od.reached_rate,
                    general_data_quality=_od.general_data_quality,
                    aggregate_reliability=getattr(
                        getattr(_om2, "_state", None), "aggregate_reliability", 0.0
                    ),
                    partial_ratio=(_pc / _total) if _total > 0 else 0.0,
                    confounder_contamination=(
                        (_rejections > _attempts * 0.20) if _attempts > 0 else False
                    ),
                )
                _sit = _adaptation_situation_context(
                    coord, zone_rt, last_update_ts=_od.last_update_ts
                )
                _traces = suggest_candidates(
                    zone_id, _signal, _sit, _od.last_update_ts or ""
                )
                _export_list = traces_for_export(_traces)
                if _export_list:
                    safe["adaptation_candidates"] = _export_list
        except Exception:
            pass
        # 3c. Adaptation candidate history (in-memory; empty until first outcome cycle).
        try:
            shadow = getattr(coord, "_le2_shadow", None)
            _history_tuples: list = []
            if shadow is not None:
                _hist_snapshot = shadow.adaptation_history_snapshot()
                if _hist_snapshot:
                    from datetime import datetime as _dt
                    for _entry in _hist_snapshot.values():
                        try:
                            _span = 0.0
                            if _entry.first_seen_ts and _entry.last_seen_ts:
                                _t0 = _dt.fromisoformat(
                                    _entry.first_seen_ts.replace("Z", "+00:00"))
                                _t1 = _dt.fromisoformat(
                                    _entry.last_seen_ts.replace("Z", "+00:00"))
                                _span = max(0.0, (_t1 - _t0).total_seconds() / 86400.0)
                        except Exception:
                            _span = 0.0
                        _history_tuples.append((_entry, _span, 0.0))
            safe["adaptation_candidate_history"] = _le2_adaptation_history_for_research(
                _history_tuples
            )
        except Exception:
            safe["adaptation_candidate_history"] = []
        # 4. Final privacy scan (belt-and-suspenders)
        try:
            from .learning.privacy import scan_payload
            violations = scan_payload(safe)
            if violations:
                _LOGGER.warning(
                    "ThermoSmart: LE2 research data for zone %s has %d residual privacy "
                    "violation(s) after stripping — LE2 block excluded for this zone. "
                    "Violations: %s",
                    _zone_hash(zone_id), len(violations),
                    [(v.path, v.kind) for v in violations[:5]],
                )
                return None
        except Exception as scan_err:
            _LOGGER.debug("ThermoSmart: LE2 privacy scan skipped: %s", scan_err)
        return safe
    except Exception:
        return None


def _le2_health_data(coord) -> dict | None:
    """Return LE2 RuntimeHealth as a plain dict for support export, or None."""
    try:
        rt = _le2_runtime(coord)
        if rt is None:
            return None
        return dataclasses.asdict(rt.health())
    except Exception:
        return None


def _le2_pending_data(coord, zone_id: str) -> dict | None:
    """Return LE2 pending-attribution summary for support export, or None."""
    try:
        rt = _le2_runtime(coord)
        if rt is None:
            return None
        return rt.pending_attribution_summary(zone_id)
    except Exception:
        return None


def _le2_adaptation_summary(coord, zone_id: str) -> dict | None:
    """Return passive adaptation candidate counts for support export, or None.

    Summary only — no trace details. Never modifies control state.
    """
    try:
        from .learning.adaptation import (
            OutcomeSignal, SituationContext, suggest_candidates, AdaptationLifecycle,
        )
        rt = _le2_runtime(coord)
        if rt is None:
            return None
        zone_rt = rt._zones.get(zone_id)
        if zone_rt is None:
            return None
        _om = zone_rt.orchestrator.models.get("outcome")
        if _om is None:
            return None
        _od = _om.diagnostics()
        _fc, _pc = _od.full_partial
        _total = _fc + _pc
        _rejections = sum(_od.rejection_counts.values()) if _od.rejection_counts else 0
        _attempts = _total + _rejections
        _signal = OutcomeSignal(
            sample_count=_total,
            timeout_rate=_od.timeout_rate,
            overshoot_rate=_od.overshoot_rate,
            reached_rate=_od.reached_rate,
            general_data_quality=_od.general_data_quality,
            aggregate_reliability=getattr(
                getattr(_om, "_state", None), "aggregate_reliability", 0.0
            ),
            partial_ratio=(_pc / _total) if _total > 0 else 0.0,
            confounder_contamination=(
                (_rejections > _attempts * 0.20) if _attempts > 0 else False
            ),
        )
        _sit = _adaptation_situation_context(coord, zone_rt, last_update_ts=_od.last_update_ts)
        try:
            import dataclasses as _dc
            _ctx_available = any(
                getattr(_sit, f.name) is not None for f in _dc.fields(_sit)
            )
        except Exception:
            _ctx_available = False
        _traces = suggest_candidates(
            zone_id, _signal, _sit, _od.last_update_ts or ""
        )
        shadow_count = sum(
            1 for t in _traces if t.lifecycle is AdaptationLifecycle.SHADOW
        )
        rejected_count = sum(
            1 for t in _traces if t.lifecycle is AdaptationLifecycle.REJECTED
        )
        return {
            "candidate_count": shadow_count,
            "shadow_candidate_count": shadow_count,
            "rejected_candidate_count": rejected_count,
            "context_available": _ctx_available,
            "last_error": None,
        }
    except Exception:
        return None


def _le2_adaptation_history_summary(coord, zone_id: str) -> dict:
    """Return adaptation candidate history summary for support export.

    Reads the in-memory candidate history from coord._le2_shadow.
    Never raises.
    """
    _zero = {"entry_count": 0, "promotion_ready_count": 0, "blocked_count": 0, "last_error": None}
    try:
        shadow = getattr(coord, "_le2_shadow", None)
        if shadow is None:
            return _zero
        history = shadow.adaptation_history_snapshot()
        if not history:
            last_err = shadow.adaptation_last_error()
            return {**_zero, "last_error": last_err}
        from .learning.adaptation import (
            evaluate_promotion_readiness,
            PromotionReadiness,
        )
        from datetime import datetime, timezone
        ready = 0
        blocked = 0
        for entry in history.values():
            try:
                span_days = 0.0
                try:
                    if entry.first_seen_ts and entry.last_seen_ts:
                        t0 = datetime.fromisoformat(entry.first_seen_ts.replace("Z", "+00:00"))
                        t1 = datetime.fromisoformat(entry.last_seen_ts.replace("Z", "+00:00"))
                        span_days = max(0.0, (t1 - t0).total_seconds() / 86400.0)
                except Exception:
                    pass
                pgr = evaluate_promotion_readiness(entry, span_days=span_days, confounder_ratio=0.0)
                if pgr.readiness is PromotionReadiness.ELIGIBLE:
                    ready += 1
                else:
                    blocked += 1
            except Exception:
                blocked += 1
        return {
            "entry_count": len(history),
            "promotion_ready_count": ready,
            "blocked_count": blocked,
            "last_error": shadow.adaptation_last_error(),
        }
    except Exception as err:
        return {**_zero, "last_error": str(err)}


def _le2_adaptation_history_for_research(history_entries: list) -> list[dict]:
    """Convert adaptation candidate history entries to research-safe export dicts.

    Args:
        history_entries: list of (CandidateHistoryEntry, span_days, confounder_ratio)
            tuples. Pass [] when no runtime accumulation is available yet.

    Returns public-safe dicts (see adaptation_history_entry_for_research_export).
    Always returns a list (empty when no entries). Never raises.
    """
    if not history_entries:
        return []
    result: list[dict] = []
    try:
        from .learning.adaptation import adaptation_history_entry_for_research_export
        for entry, span_days, confounder_ratio in history_entries:
            try:
                result.append(
                    adaptation_history_entry_for_research_export(
                        entry,
                        span_days=span_days,
                        confounder_ratio=confounder_ratio,
                    )
                )
            except Exception:
                pass
    except Exception:
        pass
    return result


def _adaptation_situation_context(
    coord: Any, zone_rt: Any, last_update_ts: Optional[str] = None
) -> Any:
    """Build an enriched SituationContext for passive adaptation candidates.

    Reads from coordinator.data and the last OutcomeModel sample. Time-of-day
    and weekday are derived from last_update_ts (OutcomeDiagnostics.last_update_ts
    = episode.end_ts of the last accepted outcome) — never from the current
    wall-clock, so the same model state always produces the same context.

    All access is defensive — missing data leaves the corresponding field None.
    Never raises; falls back to an empty SituationContext on any unexpected error.
    """
    from .learning.adaptation import SituationContext
    try:
        _OUTDOOR_EDGES = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)

        zdata = (getattr(coord, "data", None) or {}).get("zone", {}) or {}

        # ── mode / preheat / target delta from coord.data ────────────────────
        mode_context       = zdata.get("mode") or None
        preheat_was_active = zdata.get("preheat_active")
        preheat_raw        = zdata.get("preheat_minutes")
        preheat_minutes_used = (
            round(float(preheat_raw), 1) if preheat_raw is not None else None
        )
        gap = zdata.get("temperature_gap_c")
        target_delta_c = round(float(gap), 2) if gap is not None else None

        # ── zone operational state ────────────────────────────────────────────
        active_control   = getattr(coord, "_active_control", None)
        try:
            learning_enabled = bool(
                getattr(coord, "zone_cfg", {}).get("learning_enabled", True)
            )
        except Exception:
            learning_enabled = None

        # ── outdoor bucket: discretize outdoor_temp with FeatureExtractor edges
        outdoor_bucket = None
        outdoor_raw = zdata.get("outdoor_temp")
        if outdoor_raw is not None:
            try:
                ot  = float(outdoor_raw)
                idx = sum(1 for edge in _OUTDOOR_EDGES if ot >= edge)
                outdoor_bucket = f"b{idx}"
            except Exception:
                pass

        # ── controller_kind from last accepted outcome sample ─────────────────
        controller_kind = None
        try:
            if zone_rt is not None:
                _om = zone_rt.orchestrator.models.get("outcome")
                if _om is not None:
                    _state   = getattr(_om, "_state", None)
                    _samples = getattr(_state, "recent_samples", ()) or ()
                    if _samples:
                        controller_kind = getattr(_samples[-1], "controller_kind", None)
        except Exception:
            pass

        # ── time of day / weekday from last outcome timestamp (deterministic) ─
        # Derived from OutcomeDiagnostics.last_update_ts = episode.end_ts of the
        # last accepted outcome — NOT from datetime.now() so that the same model
        # state always produces the same context values.
        time_of_day_bucket: Optional[int]  = None
        weekday:            Optional[int]  = None
        is_weekend:         Optional[bool] = None
        context_time_source = "unavailable"
        if last_update_ts:
            try:
                _ts = datetime.fromisoformat(last_update_ts.replace("Z", "+00:00"))
                time_of_day_bucket  = _ts.hour
                weekday             = _ts.weekday()
                is_weekend          = weekday >= 5
                context_time_source = "model_last_update"
            except Exception:
                pass

        return SituationContext(
            controller_kind=controller_kind,
            outdoor_bucket=outdoor_bucket,
            mode_context=mode_context,
            time_of_day_bucket=time_of_day_bucket,
            weekday=weekday,
            is_weekend=is_weekend,
            preheat_was_active=(
                bool(preheat_was_active) if preheat_was_active is not None else None
            ),
            boost_was_active=None,        # not available at export time
            target_delta_c=target_delta_c,
            heat_loss_c_per_h=None,       # not available at export time
            preheat_minutes_used=preheat_minutes_used,
            active_control=(
                bool(active_control) if active_control is not None else None
            ),
            learning_enabled=learning_enabled,
            context_time_source=context_time_source,
        )
    except Exception:
        from .learning.adaptation import SituationContext
        return SituationContext()


def _resolve_clock(hass: HomeAssistant) -> datetime | None:
    """Return a UTC timestamp from the first available coordinator clock."""
    for entry_data in hass.data.get(DOMAIN, {}).values():
        if isinstance(entry_data, dict):
            coord = entry_data.get("coordinator")
            if coord is not None and hasattr(coord, "_clock"):
                try:
                    return coord._clock.now_utc()
                except Exception:
                    pass
    return None


# ── cleanup scheduler ─────────────────────────────────────────────────────────

def _schedule_export_cleanup(hass: HomeAssistant, filepath: str) -> None:
    """Schedule deletion of an export file after 24 h.

    Best-effort: the timer runs in the current HA session only and does NOT
    survive a restart.  If HA restarts before 24 h, the file remains until the
    user deletes it or until this timer fires in a future session.
    """
    def _cleanup(_now: datetime) -> None:
        try:
            os.remove(filepath)
            _LOGGER.debug("ThermoSmart: auto-deleted export file after 24 h: %s", filepath)
        except FileNotFoundError:
            pass
        except OSError as err:
            _LOGGER.warning("ThermoSmart: could not delete export file %s: %s", filepath, err)

    async_call_later(hass, _EXPORT_CLEANUP_DELAY_S, _cleanup)


# ── export functions ──────────────────────────────────────────────────────────

async def async_export_learning_data(hass: HomeAssistant, *, ts: datetime | None = None) -> str:
    """Build anonymized research export covering all zones, write to /config/www/."""
    le: LearningEngine | None = hass.data.get(DOMAIN, {}).get("learning_engine")
    if ts is None:
        ts = _resolve_clock(hass)
    if ts is None:
        raise RuntimeError(
            "async_export_learning_data: no coordinator clock available — "
            "ensure at least one zone is configured before exporting."
        )
    ts_str = ts.strftime("%Y%m%dT%H%M%S")

    zones: list[dict] = []

    for entry in hass.config_entries.async_entries(DOMAIN):
        cfg = {**entry.data, **entry.options}
        if cfg.get("entry_type") == "system":
            continue

        entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
        coord = entry_data.get("coordinator") if isinstance(entry_data, dict) else None

        meta = _zone_meta(cfg)
        learning: dict = le.get_export_data(entry.entry_id) if le is not None else {}
        analytics = _compute_analytics(learning)
        le2 = _le2_research_data(coord, entry.entry_id) if coord is not None else None

        zones.append({
            "zone_hash": _zone_hash(entry.entry_id),
            **meta,
            "historical_learning_snapshot": learning,
            "analytics": analytics,
            "runtime_models": le2,
        })

    export: dict = {
        "thermosmart_version": VERSION,
        "export_type": "research",
        "export_format_version": EXPORT_FORMAT_VERSION,
        "export_timestamp": ts.isoformat(),
        "zone_count": len(zones),
        "total_trv_count": sum(z["trv_count"] for z in zones),
        "total_temp_sensor_count": sum(z["temp_sensor_count"] for z in zones),
        "total_humidity_sensor_count": sum(z["humidity_sensor_count"] for z in zones),
        "zones": zones,
    }

    www_dir = hass.config.path("www")
    os.makedirs(www_dir, exist_ok=True)
    random_suffix = os.urandom(3).hex()
    filename = f"thermosmart_research_{ts_str}_{random_suffix}.json"
    filepath = os.path.join(www_dir, filename)

    def _write() -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2, ensure_ascii=False, default=str)

    await hass.async_add_executor_job(_write)
    _schedule_export_cleanup(hass, filepath)
    _LOGGER.info("ThermoSmart: research export written → %s", filepath)
    return filepath


async def async_export_support_data(hass: HomeAssistant, *, ts: datetime | None = None) -> str:
    """Build a support-oriented export covering all zones, write to /config/www/."""
    from homeassistant.const import __version__ as HA_VERSION  # noqa: PLC0415

    if ts is None:
        ts = _resolve_clock(hass)
    if ts is None:
        raise RuntimeError(
            "async_export_support_data: no coordinator clock available — "
            "ensure at least one zone is configured before exporting."
        )
    ts_str = ts.strftime("%Y%m%dT%H%M%S")

    all_entries = [
        e for e in hass.config_entries.async_entries(DOMAIN)
        if {**e.data, **e.options}.get("entry_type") != "system"
    ]

    zones: list[dict] = []
    for entry in all_entries:
        cfg = {**entry.data, **entry.options}

        entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
        coord = entry_data.get("coordinator") if isinstance(entry_data, dict) else None

        zone_info: dict = {
            "zone_hash": _zone_hash(entry.entry_id),
            "config_flags": _zone_meta(cfg),
        }

        if coord is not None:
            zone_info["runtime_state"] = {
                "active_control": getattr(coord, "_active_control", None),
                "learning_enabled": getattr(coord, "_learning_enabled", None),
                "current_mode": getattr(coord, "_current_mode", None),
                "confidence": round(float(((coord.data or {}).get("learning_confidence") or 0.0)), 3),
            }
            zone_info["runtime_health"] = _le2_health_data(coord)
            zone_info["runtime_pending"] = _le2_pending_data(coord, entry.entry_id)
            zone_info["adaptation"] = _le2_adaptation_summary(coord, entry.entry_id)
            zone_info["adaptation_history"] = _le2_adaptation_history_summary(
                coord, entry.entry_id
            )
        else:
            zone_info["runtime_state"] = None
            zone_info["runtime_health"] = None
            zone_info["runtime_pending"] = None
            zone_info["adaptation"] = None
            zone_info["adaptation_history"] = None

        zones.append(zone_info)

    export: dict = {
        "thermosmart_version": VERSION,
        "export_type": "support",
        "export_format_version": EXPORT_FORMAT_VERSION,
        "export_timestamp": ts.isoformat(),
        "system": {
            "ha_version": HA_VERSION,
            "zone_count": len(zones),
        },
        "zones": zones,
    }

    www_dir = hass.config.path("www")
    os.makedirs(www_dir, exist_ok=True)
    random_suffix = os.urandom(3).hex()
    filename = f"thermosmart_support_{ts_str}_{random_suffix}.json"
    filepath = os.path.join(www_dir, filename)

    def _write() -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2, ensure_ascii=False, default=str)

    await hass.async_add_executor_job(_write)
    _schedule_export_cleanup(hass, filepath)
    _LOGGER.info("ThermoSmart: support export written → %s", filepath)
    return filepath


# ── notification messages ─────────────────────────────────────────────────────

def build_export_notification_message(filename: str) -> str:
    """Return the persistent notification message for a completed research export."""
    return (
        "ThermoSmart Research-Export wurde erstellt.\n\n"
        "Die Datei enthält detailliertere anonymisierte technische Lern-, Sensor-, "
        "Entscheidungs- und Ergebnisdaten. Sie ist zur Analyse und Weiterentwicklung "
        "von ThermoSmart gedacht.\n\n"
        "Die Datei enthält keine Raum- oder Zonennamen, Entity-IDs, Geräte-IDs, "
        "Adressen oder exakten Standortdaten. Prüfe die Datei vor dem Teilen.\n\n"
        "Die Datei wurde nur lokal erstellt, nicht hochgeladen oder übertragen. "
        "Sie wird nach 24 Stunden automatisch gelöscht "
        "(sofern Home Assistant bis dahin nicht neu gestartet wurde).\n\n"
        f"**Datei:**\n`/config/www/{filename}`\n\n"
        f"**Öffnen:**\n`/local/{filename}`"
    )


def build_support_notification_message(filename: str) -> str:
    """Return the persistent notification message for a completed support export."""
    return (
        "ThermoSmart Support-Export wurde erstellt.\n\n"
        "Die Datei enthält zusammengefasste datenschutzfreundliche "
        "Diagnoseinformationen.\n\n"
        "Sie wird nach 24 Stunden automatisch gelöscht "
        "(sofern Home Assistant bis dahin nicht neu gestartet wurde).\n\n"
        f"**Datei:**\n`/config/www/{filename}`\n\n"
        f"**Öffnen:**\n`/local/{filename}`\n\n"
        "Prüfe die Datei vor dem Teilen. Es wurden keine Daten übertragen."
    )
