"""Anonymized learning-data export for ThermoSmart.

Creates a JSON snapshot of per-zone learning data that can be shared voluntarily
to support Learning development.  Nothing is sent automatically — the user decides
whether and how to share the file.

Privacy contract
----------------
Exported data contains:
  - ThermoSmart version + export format version
  - Export timestamp
  - Per-zone: TRV count, sensor counts, feature flags (booleans only)
  - Per-zone: all numeric learning data (observations, rates, confidence, …)
  - Per-zone: Learning model coefficient statistics (numeric only — no IDs)

Exported data does NOT contain:
  - Passwords or authentication tokens of any kind
  - Entity IDs, device names, or integration names
  - Person names or user identifiers
  - Street addresses or geographic coordinates
  - Decision IDs, episode IDs, or internal runtime identifiers

Timestamps are intentionally retained:
  Observation timestamps (ts, hour, minute, weekday) are required for
  longitudinal learning analysis and are the primary reason this data is
  valuable for Learning.  A series of heating timestamps can reveal usage
  patterns (presence, sleep schedule, away periods).  Users should review
  the exported file before sharing it with anyone.

has_forecast note:
  has_forecast is derived from whether a weather entity is configured.
  It does not guarantee that the entity actually provides forecast data —
  some weather integrations only expose current conditions.

Zone identity: each zone_id (HA entry_id UUID) is replaced with a deterministic
12-char hex digest.  Exports from the same installation share the same digests,
making longitudinal data correlatable without being reversible.

Historical (frozen legacy) learning snapshot:
  Per-zone "historical_learning_snapshot" block — the frozen legacy
  learning-engine data (learning_engine.freeze() in __init__.py) reshaped
  into a structured Deep-Research view instead of an unfiltered pass-through
  of LearningEngine.get_export_data(). Per-category allow-lists keep genuine
  research value (heat rate, delta, outcome score, ts/weekday/hour/minute
  time context) while only fields explicitly named in those allow-lists can
  ever reach the export — no entity_id/device_id/unique_id/person/presence/
  location/home name. event_count_summary reports the true (uncapped) totals;
  research_events is capped per category (see
  _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY) with any excess reported via
  records_truncated, never silently dropped. A final scan_payload() pass
  (the same second-barrier scanner used by the Learning runtime-models research
  block) excludes any category that unexpectedly fails it. Never crashes the
  export — a missing/malformed source yields available: false instead.

Learning research data:
  Only model coefficient aggregates are included (models, cycles,
  last_cycle_ts, model_update_counts).  Fields containing IDs of any kind
  (decision_id, episode_id, learning_zone_id, zone_id, …) are stripped
  recursively before inclusion.  A privacy scan is performed as a final check.

Learning learning progress:
  Per-zone "learning_progress" block — the same calibrated scores/labels
  LearningShadowController.learning_progress_safe() already exposes to the
  confidence sensor (data volume/coverage/diversity/clean-episode/outcome/
  confidence scores, confounder_penalty, regime_cap_pct, main_blocker,
  next_needed). No episode history, no timestamps, no IDs — just the
  existing numeric/label explanation of why a zone reads at its current
  progress percentage.

Learning episode history:
  Per-zone "episode_history" block — a BOUNDED SUMMARY (counts, ages,
  retention metadata) of LearningShadowController.episode_history_snapshot(),
  never a full episode list. No episode_id/learning_zone_id/decision_id/
  trv_binding_id, no trajectory, no raw per-episode timestamps — only
  aggregate counts_by_type, confounder_count, timeout_count, relative
  oldest/newest ages in hours, and the registry's own retention policy
  (max_records/max_age_days per type) to make boundedness visible.

Learning research daily buckets:
  Per-zone "research_daily" block — a BOUNDED SUMMARY of
  LearningShadowController.research_daily_snapshot() (already-in-memory,
  no new store read). "summary" aggregates counters/averages/progress-
  confidence min/max/last across ALL valid buckets; "daily" is a compact,
  newest-first list capped at 90 days (records_truncated/truncation_reason
  report any excess) — the summary itself is never truncated. No episode/
  event ids, no raw events, no trajectories — every field is already one of
  ResearchDailyBucket's own fixed scalar aggregates.

Learning support critical event timeline (support export only):
  Per-zone "critical_events" block — reads ONLY the already-in-memory
  LearningShadowController.support_critical_events_snapshot(), no store
  read. Each event is rendered via support_event_for_export() (drops
  event_id, bounds "details"). A separate, smaller export cap (200) applies
  on top of the store's own 750-record cap so a single export file stays
  readable; any excess is reported via records_truncated, never silently
  dropped. Coverage/retention metadata (coverage_start/end,
  persistent_store_retention_h, full_window_covered, store_warmup) describes
  the underlying data span, independent of the export cap.

Support export reserved diagnostics:
  Per-zone "reserved_diagnostics" block replaces the previous
  "adaptation_application"/"orchestration_preview" pair — both were always-
  zero placeholders for the application/orchestration layer, which is
  foundation-only in this version (a global kill-switch keeps
  application_enabled False; nothing ever actually applies a candidate). The
  two live, currently-meaningful blocks ("adaptation",
  "adaptation_history" — real passive-candidate/attribution counts, not
  reserved) are unaffected and remain as-is.

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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_call_later
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
_EXPORT_CLEANUP_DELAY_S: float = 24 * 3600  # 24 hours; not restart-safe

# ── Learning privacy helpers ───────────────────────────────────────────────────────

# Top-level keys safe to include from _ZoneRuntime.serialize().
# Excluded: capture (contains zone_id + decision_id in ledger),
#           pipeline (contains zone_id + open_decision_id),
#           ledger (inside capture, contains decision_ids),
#           baseline_store, pending_* (all contain decision-related IDs).
_LEARNING_RESEARCH_SAFE_TOP_KEYS = frozenset(
    {"models", "cycles", "last_cycle_ts", "model_update_counts"}
)

# Key substrings to strip recursively — mirrors privacy.py _FORBIDDEN_KEY_SUBSTRINGS
# plus "zone_id" which the scanner does not catch standalone.
_LEARNING_STRIP_KEY_SUBSTRINGS = (
    "entity_id", "entry_id", "device_id", "decision_id", "episode_id", "event_id",
    "evaluation_id", "user_id", "person", "email", "latitude", "longitude", "address",
    "hostname", "ip_address", "ipaddr", "source_episode_id", "correction_event_id",
    "source_decision_id", "trv_binding_id", "learning_zone_id", "zone_id",
    "radiator_profile_id",
)


def _learning_strip_forbidden(obj: Any) -> Any:
    """Recursively remove keys matching Learning privacy-forbidden substrings."""
    if isinstance(obj, dict):
        return {
            k: _learning_strip_forbidden(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and any(s in k.lower() for s in _LEARNING_STRIP_KEY_SUBSTRINGS))
        }
    if isinstance(obj, list):
        return [_learning_strip_forbidden(i) for i in obj]
    return obj


# ── Historical (frozen legacy) learning snapshot — Deep Research reshaping ──
#
# The legacy learning engine (frozen — LearningEngine.freeze() in __init__.py)
# accumulated raw per-event dicts before Learning existed. These carry genuine
# research value (heat rate, delta, outcome score, time-of-day context) and
# are intentionally NOT reduced to counts/averages — only entity/person/
# location identifiers are stripped. Unlike every other research-export
# block, this snapshot used to be an unfiltered pass-through of
# learning_engine.get_export_data() with no scan_payload() pass; the
# functions below are the fix.
#
# Per-category ALLOW-lists (not a blacklist over the legacy free-form dicts)
# are the primary defense: only fields explicitly named here can ever reach
# the export, so an unexpected legacy key can only be silently excluded,
# never silently leaked. scan_payload() (the same second-barrier scanner used
# by the Learning runtime-models research block, _learning_research_data) still runs as
# a final check.

_HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY = 200

_HIST_OBS_ALLOWED_FIELDS = (
    "target", "indoor_temp", "delta", "active_control", "window_open",
    "control_reason", "preheat_active", "heating_failure", "vacation",
    "summer_mode", "outdoor_temp", "outdoor_humidity", "wind_speed",
    "solar_radiation", "indoor_humidity", "heat_rate", "norm_heat_rate",
    "cool_rate", "schedule_period", "forecast_high",
)
_HIST_TRV_OBS_ALLOWED_FIELDS = (
    "trv_setpoint", "indoor_temp", "target", "delta", "setpoint_excess",
    "heat_rate", "efficiency", "outdoor_temp", "wind_speed", "solar_radiation",
)
_HIST_WINDOW_COOLING_ALLOWED_FIELDS = (
    "duration_min", "temp_drop", "cooling_rate_per_min", "indoor_start_temp",
    "outdoor_temp", "wind_speed",
)
_HIST_OUTCOME_ALLOWED_FIELDS = (
    "start_temp", "target", "peak_temp", "end_temp", "minutes_taken",
    "expected_minutes", "controller", "reason", "outcome_score",
    "outdoor_temp", "outdoor_humidity", "wind_speed", "solar_radiation", "rain",
)


def _hist_event_view(raw: Any, allowed_fields: tuple, *, source: str) -> dict | None:
    """Build one allow-listed research event from a raw legacy learning-engine
    dict. Only technical/numeric fields explicitly named in ``allowed_fields``
    survive, plus a research time context (``ts``/``weekday``/``hour``/
    ``minute``, derived from ``ts`` so all four categories get the same
    context shape even though only "observations" stored hour/minute/weekday
    directly) and a fixed ``source`` marker. Returns None for a malformed
    (non-dict) entry — never raises — so one corrupt legacy row cannot break
    the whole export.
    """
    if not isinstance(raw, dict):
        return None
    ts_raw = raw.get("ts")
    view: dict = {}
    if isinstance(ts_raw, str):
        view["ts"] = ts_raw
        try:
            ts_text = ts_raw[:-1] + "+00:00" if ts_raw.endswith("Z") else ts_raw
            parsed = datetime.fromisoformat(ts_text)
            view["weekday"] = parsed.weekday()
            view["hour"] = parsed.hour
            view["minute"] = parsed.minute
        except Exception:
            pass
    for key in allowed_fields:
        val = raw.get(key)
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            view[key] = val
        # else: unexpected type for a technical field — dropped, not guessed at.
    view["source"] = source
    return view


def _hist_category(
    raw_list: Any, allowed_fields: tuple, *, source: str,
) -> tuple[list, int]:
    """Cap a raw legacy event list to the most recent
    ``_HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY`` entries (already
    chronological in the source), then allow-list each kept entry. Returns
    ``(kept_views, records_truncated)`` — never raises; a missing/malformed
    source list yields an empty category rather than failing the export.
    """
    if not isinstance(raw_list, list):
        return [], 0
    total = len(raw_list)
    capped_raw = raw_list[-_HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY:]
    views = [
        v for v in (_hist_event_view(r, allowed_fields, source=source) for r in capped_raw)
        if v is not None
    ]
    truncated = max(0, total - len(capped_raw))
    return views, truncated


def _historical_learning_snapshot_for_research(learning: dict) -> dict:
    """Convert the frozen legacy learning-engine snapshot into a
    structured, privacy-scanned Deep-Research block (see module-level
    comment above for the rationale).

    Scalar model-state values (confidence/boost_factor/forecast_bias/
    heat_loss_ema) are plain floats already produced by
    ``LearningEngine.get_export_data()`` — no allow-list needed beyond a
    numeric-type check, kept under ``model_state`` in the new structure.
    """
    try:
        obs_kept, obs_trunc = _hist_category(
            learning.get("observations"), _HIST_OBS_ALLOWED_FIELDS, source="historical")
        trv_kept, trv_trunc = _hist_category(
            learning.get("trv_observations"), _HIST_TRV_OBS_ALLOWED_FIELDS, source="historical")
        wc_kept, wc_trunc = _hist_category(
            learning.get("window_cooling_obs"), _HIST_WINDOW_COOLING_ALLOWED_FIELDS,
            source="historical")
        out_kept, out_trunc = _hist_category(
            learning.get("outcome_log"), _HIST_OUTCOME_ALLOWED_FIELDS, source="historical")

        research_events = {
            "observations": obs_kept,
            "trv_observations": trv_kept,
            "window_cooling": wc_kept,
            "outcomes": out_kept,
        }

        # Final belt-and-suspenders scan — same defense used by
        # _learning_research_data(). The allow-lists above should already make
        # this a no-op; if it ever isn't, drop only the offending category
        # so one unexpected field cannot suppress the rest of the snapshot.
        from .learning.privacy import scan_payload
        for cat_name, cat_events in list(research_events.items()):
            try:
                if scan_payload(cat_events):
                    _LOGGER.warning(
                        "ThermoSmart: historical_learning_snapshot.%s failed the "
                        "privacy scan — category excluded from this export.",
                        cat_name,
                    )
                    research_events[cat_name] = []
            except Exception:
                research_events[cat_name] = []

        def _num(key: str):
            val = learning.get(key)
            return val if isinstance(val, (int, float)) else None

        return {
            "available": True,
            "frozen": True,
            "raw_legacy_dump_included": False,
            "historical_learning_dump_included": False,
            "privacy_checked": True,
            "event_count_summary": {
                "observations": int(learning.get("observation_count") or 0),
                "trv_observations": int(learning.get("trv_observation_count") or 0),
                "window_cooling_observations": int(learning.get("window_cooling_obs_count") or 0),
                "outcome_entries": len(learning.get("outcome_log") or []),
            },
            "model_state": {
                "confidence": _num("confidence"),
                "boost_factor": _num("boost_factor"),
                "forecast_bias": _num("forecast_bias"),
                "heat_loss_ema": _num("heat_loss_ema"),
            },
            "research_events": research_events,
            "records_truncated": {
                "observations": obs_trunc,
                "trv_observations": trv_trunc,
                "window_cooling": wc_trunc,
                "outcomes": out_trunc,
            },
        }
    except Exception:
        return {
            "available": False,
            "reason": "historical_snapshot_unavailable",
            "frozen": True,
            "raw_legacy_dump_included": False,
            "historical_learning_dump_included": False,
            "privacy_checked": True,
        }


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


# ── Learning data accessors ────────────────────────────────────────────────────────

def _learning_runtime(coord):
    """Safely return the Learning LearningRuntime from a coordinator, or None."""
    try:
        shadow = getattr(coord, "_learning_shadow", None)
        if shadow is None:
            return None
        return getattr(shadow, "runtime", None)
    except Exception:
        return None


def _learning_research_data(coord, zone_id: str) -> dict | None:
    """Return privacy-safe Learning model statistics for research export, or None.

    Only includes top-level keys from _LEARNING_RESEARCH_SAFE_TOP_KEYS (models,
    cycles, last_cycle_ts, model_update_counts).  All fields whose key contains
    a forbidden substring (decision_id, learning_zone_id, zone_id, …) are
    stripped recursively before inclusion.  A final privacy scan confirms no
    violations remain; if any do, the Learning block is excluded for that zone.
    """
    try:
        rt = _learning_runtime(coord)
        if rt is None:
            return None
        zone_rt = rt._zones.get(zone_id)
        if zone_rt is None:
            return None
        raw = zone_rt.serialize()
        # 1. Allowlist: only safe top-level sections
        filtered = {k: v for k, v in raw.items() if k in _LEARNING_RESEARCH_SAFE_TOP_KEYS}
        # 2. Recursive strip of forbidden keys within allowed sections
        safe = _learning_strip_forbidden(filtered)
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
                    confounder_contamination=(_rejections > 0),
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
            shadow = getattr(coord, "_learning_shadow", None)
            _history_tuples: list = []
            _lifecycle_state = None
            if shadow is not None:
                _lifecycle_state = getattr(shadow, "_application_lifecycle_state", None)
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
                        _history_tuples.append(
                            (_entry, _span, _learning_confounder_ratio(coord, zone_id))
                        )
            safe["adaptation_candidate_history"] = _learning_adaptation_history_for_research(
                _history_tuples, lifecycle_state=_lifecycle_state,
            )
        except Exception:
            safe["adaptation_candidate_history"] = []
        # 3d. Application lifecycle state entries (public-safe; no timestamps).
        try:
            safe["adaptation_application_state"] = (
                _learning_application_lifecycle_research_entries(coord)
            )
        except Exception:
            safe["adaptation_application_state"] = []
        # 4. Final privacy scan (belt-and-suspenders)
        try:
            from .learning.privacy import scan_payload
            violations = scan_payload(safe)
            if violations:
                _LOGGER.warning(
                    "ThermoSmart: Learning Engine research data for zone %s has %d residual "
                    "privacy violation(s) after stripping — block excluded for this zone. "
                    "Violations: %s",
                    _zone_hash(zone_id), len(violations),
                    [(v.path, v.kind) for v in violations[:5]],
                )
                return None
        except Exception as scan_err:
            _LOGGER.debug("ThermoSmart: learning privacy scan skipped: %s", scan_err)
        return safe
    except Exception:
        return None


def _learning_health_data(coord) -> dict | None:
    """Return Learning RuntimeHealth as a plain dict for support export, or None."""
    try:
        rt = _learning_runtime(coord)
        if rt is None:
            return None
        return dataclasses.asdict(rt.health())
    except Exception:
        return None


# Fields taken as-is from the real compute_learning_progress() attrs dict
# (learning/runtime/learning_progress.py) — numeric scores/labels only, no
# IDs, no timestamps, no raw episode data. Nothing here is invented; anything
# not in this list (e.g. sample_count, outcome_quality, per-model episode
# counts, bootstrap_excluded) simply isn't surfaced in this small research
# block yet.
_LEARNING_PROGRESS_RESEARCH_KEYS = (
    "learning_stage",
    "confidence_level",
    "data_volume_score",
    "clean_episode_score",
    "thermal_regime_coverage_score",
    "situation_diversity_score",
    "outcome_validation_score",
    "model_confidence_score",
    "confounder_penalty",
    "regime_cap_pct",
    "main_blocker",
    "next_needed",
)


def _learning_progress_export(coord) -> dict:
    """Return a small, public-safe Learning learning-progress block for research export.

    Sourced directly from LearningShadowController.learning_progress_safe() —
    the SAME method ThermoSmartConfidenceSensor reads — so the exported
    numbers always match what the user already sees. No new store read, no
    episode-history access: this reuses the existing model-diagnostics-based
    calculation that already runs every cycle.

    Never raises and never breaks the export: a missing/unattached Learning shadow
    or any unexpected failure inside learning_progress_safe() (which is
    already designed to never raise, but this stays defensive independent of
    that guarantee) yields an explicit ``{"available": False, ...}`` block
    instead of omitting the zone or failing the whole export.
    """
    try:
        shadow = getattr(coord, "_learning_shadow", None)
        if shadow is None:
            return {"available": False, "reason": "learning_engine_unavailable"}
        progress_pct, attrs = shadow.learning_progress_safe()
        block: dict = {"available": True, "progress_pct": progress_pct}
        for key in _LEARNING_PROGRESS_RESEARCH_KEYS:
            if key in attrs:
                block[key] = attrs[key]
        safe = _learning_strip_forbidden(block)
        try:
            from .learning.privacy import scan_payload
            if scan_payload(safe):
                return {"available": False, "reason": "privacy_scan_failed"}
        except Exception:
            pass  # scan itself failing is non-fatal — block already key-stripped
        return safe
    except Exception as err:
        return {"available": False, "learning_progress_error": str(err)}


# Known episode types (episode_schemas.EpisodeType values) — used both to seed
# counts_by_type at zero (so a zone with no episodes of a type still reports
# it explicitly) and to recognise/skip unrecognised "episode_type" values.
_LEARNING_EPISODE_TYPES = (
    "heating", "afterheat", "passive_cooling", "window_cooling", "outcome",
)


def _learning_episode_history_export(coord, *, now: datetime) -> dict:
    """Return a small, bounded, public-safe episode-history SUMMARY for research export.

    Deliberately a summary, not a per-episode entry list — this reuses
    ``LearningShadowController.episode_history_snapshot()`` (already-in-memory,
    no new store read) and aggregates counts/ages/retention metadata only.
    No ``episode_id``, ``learning_zone_id``, ``decision_id``, ``trv_binding_id``,
    no ``trajectory``, no raw per-episode timestamps — only aggregate counts
    and relative ages (hours before ``now``).

    Never raises and never breaks the export: a missing/unattached Learning shadow,
    a snapshot() failure, or an unexpected error anywhere in the aggregation
    yields an explicit ``{"available": False, ...}`` block instead of omitting
    the zone or failing the whole export. Malformed individual entries
    (not a dict, unrecognised/missing "episode_type") are skipped and counted
    in ``malformed_skipped_count`` — they never abort the summary.
    """
    try:
        shadow = getattr(coord, "_learning_shadow", None)
        if shadow is None:
            return {"available": False, "reason": "learning_engine_unavailable"}
        snapshot = shadow.episode_history_snapshot()
        if not isinstance(snapshot, dict):
            return {"available": False, "reason": "episode_history_unavailable"}

        counts_by_type = {t: 0 for t in _LEARNING_EPISODE_TYPES}
        confounder_count = 0
        timeout_count = 0
        malformed_skipped = 0
        oldest_age_hours = None
        newest_age_hours = None

        for entry in snapshot.values():
            if not isinstance(entry, dict):
                malformed_skipped += 1
                continue
            etype = entry.get("episode_type")
            if etype not in counts_by_type:
                malformed_skipped += 1
                continue
            counts_by_type[etype] += 1

            confounder_flags = entry.get("confounder_flags")
            if isinstance(confounder_flags, list) and len(confounder_flags) > 0:
                confounder_count += 1

            if etype == "outcome" and entry.get("reason") == "timeout":
                timeout_count += 1

            end_ts_raw = entry.get("end_ts")
            if isinstance(end_ts_raw, str):
                try:
                    end_ts = datetime.fromisoformat(end_ts_raw)
                    if end_ts.tzinfo is None:
                        end_ts = end_ts.replace(tzinfo=timezone.utc)
                    age_hours = round((now - end_ts).total_seconds() / 3600.0, 1)
                    if oldest_age_hours is None or age_hours > oldest_age_hours:
                        oldest_age_hours = age_hours
                    if newest_age_hours is None or age_hours < newest_age_hours:
                        newest_age_hours = age_hours
                except Exception:
                    pass  # age is best-effort; a malformed timestamp just skips age tracking

        entry_count = sum(counts_by_type.values())
        available_types = sorted(t for t, n in counts_by_type.items() if n > 0)
        missing_types = sorted(t for t, n in counts_by_type.items() if n == 0)

        retention: dict = {"bounded": False}
        try:
            registry = shadow.capture_stores.episode_registry if shadow.capture_stores else None
            if registry is not None:
                max_records_by_type: dict = {}
                max_age_days_by_type: dict = {}
                for definition in registry.definitions():
                    key = definition.episode_type.value
                    max_records_by_type[key] = definition.retention.max_records
                    max_age_days_by_type[key] = definition.retention.max_age_days
                retention = {
                    "bounded": bool(max_records_by_type) and all(
                        v is not None for v in max_records_by_type.values()
                    ),
                    "max_records_by_type": max_records_by_type,
                    "max_age_days_by_type": max_age_days_by_type,
                }
        except Exception:
            pass  # retention metadata is best-effort; core counts remain valid

        block: dict = {
            "available": True,
            "schema_version": _episode_schema_version(),
            "entry_count": entry_count,
            "counts_by_type": counts_by_type,
            "available_types": available_types,
            "missing_types": missing_types,
            "confounder_count": confounder_count,
            "timeout_count": timeout_count,
            "malformed_skipped_count": malformed_skipped,
            "oldest_age_hours": oldest_age_hours,
            "newest_age_hours": newest_age_hours,
            "retention": retention,
        }
        safe = _learning_strip_forbidden(block)
        try:
            from .learning.privacy import scan_payload
            if scan_payload(safe):
                return {"available": False, "reason": "privacy_scan_failed"}
        except Exception:
            pass  # scan itself failing is non-fatal — block already key-stripped
        return safe
    except Exception as err:
        return {"available": False, "episode_history_error": str(err)}


def _episode_schema_version() -> int:
    """Return the current episode schema version, or None if unavailable."""
    try:
        from .learning.storage.episode_serialization import EPISODE_SCHEMA_VERSION
        return EPISODE_SCHEMA_VERSION
    except Exception:
        return None


# Research Daily Buckets are already bounded in storage (365 days / 400
# buckets — research_daily_persistence.py). This is a SEPARATE, smaller cap
# on the per-day "daily" LIST actually returned in the research export, so a
# single export file stays readable — 400 daily entries in a JSON file a
# user is asked to review before sharing is not "small". 90 days (~3 months)
# is long enough for a meaningful research-diagnosis window without bloating
# the file. Centralised so a later calibration pass has one place to change.
# The "summary" block below is NOT subject to this cap — it aggregates over
# ALL valid buckets regardless, so long-term totals/min/max/last stay
# complete even when the daily list itself is truncated.
_RESEARCH_DAILY_EXPORT_MAX_BUCKETS = 90

# Fixed, pre-defined counter fields eligible for a compact "daily" entry —
# included only when > 0, mirroring ResearchDailyBucket's own field set
# (research_daily_schemas.py). No entity/zone/episode ids, no raw events, no
# trajectories — every one of these is already a scalar aggregate by
# construction.
_RESEARCH_DAILY_COUNTER_FIELDS = (
    "decision_count",
    "heating_allowed_count", "heating_blocked_count",
    "trv_command_sent_count", "trv_command_blocked_count",
    "same_setpoint_block_count", "trv_unavailable_count",
    "window_hold_count", "summer_hold_count", "manual_override_count",
    "boost_started_count", "boost_blocked_count", "boost_ended_count",
    "sensor_unavailable_count", "sensor_restored_count", "fallback_used_count",
    "outcome_resolved_count", "outcome_success_count", "outcome_failed_count",
    "outcome_confounded_count",
)
# (sum_field, count_field) pairs — both included in a compact daily entry
# only when the count is > 0 (a zero count means the sum is meaningless).
_RESEARCH_DAILY_SUM_COUNT_PAIRS = (
    ("overshoot_sum_c", "overshoot_count"),
    ("undershoot_sum_c", "undershoot_count"),
    ("comfort_error_sum_c", "comfort_error_count"),
)
_RESEARCH_DAILY_PROGRESS_CONFIDENCE_FIELDS = (
    "learning_progress_min_pct", "learning_progress_max_pct", "learning_progress_last_pct",
    "confidence_min", "confidence_max", "confidence_last",
)

# Schema-defined but currently UNPRODUCED counters (research_daily_schemas.py
# still carries them — deliberately not removed, since a real Coordinator/
# Decision-Record producer may fill them in later; see that module's
# docstring). Showing them as a flat "0" in the summary would misleadingly
# read as "zero decisions ever made" rather than "not measured yet" — so
# they are omitted from the summary entirely while every bucket's own value
# is 0, and appear automatically the moment any bucket actually carries a
# real (>0) value for one of them (no separate flag/field needed for that).
_RESEARCH_DAILY_UNPRODUCED_SUMMARY_FIELDS = (
    "decision_count", "heating_allowed_count", "heating_blocked_count",
)


def _learning_research_daily_compact_entry(bucket) -> dict:
    """Compact per-day export entry: always ``bucket_date``, counters only
    when > 0, sum/count pairs only when the count is > 0, progress/
    confidence fields only when not None. Keeps the "daily" list readable —
    a day with mostly-zero activity does not carry two dozen zero fields."""
    entry: dict = {"bucket_date": bucket.bucket_date}
    for field_name in _RESEARCH_DAILY_COUNTER_FIELDS:
        value = getattr(bucket, field_name, 0)
        if value:
            entry[field_name] = value
    for sum_field, count_field in _RESEARCH_DAILY_SUM_COUNT_PAIRS:
        count_value = getattr(bucket, count_field, 0)
        if count_value:
            entry[sum_field] = getattr(bucket, sum_field)
            entry[count_field] = count_value
    for field_name in _RESEARCH_DAILY_PROGRESS_CONFIDENCE_FIELDS:
        value = getattr(bucket, field_name, None)
        if value is not None:
            entry[field_name] = value
    return entry


def _learning_research_daily_export(coord, *, now: datetime) -> dict:
    """Return a small, bounded, public-safe Research Daily Bucket long-term
    summary for research export.

    Reads ONLY the already-in-memory
    ``LearningShadowController.research_daily_snapshot()`` — no store read
    here (this module never touches the underlying research-daily storage
    layer directly), no new aggregation, no runtime/coordinator touched. Each
    stored entry is validated with ``deserialize_research_daily_bucket()``
    (research_daily_serialization.py) before being used — a malformed or
    schema-mismatched entry is skipped and counted in
    ``malformed_skipped_count``, never aborts the summary.

    ``summary`` aggregates over ALL valid buckets (counters summed;
    ``avg_overshoot_c``/``avg_undershoot_c``/``avg_comfort_error_c`` from
    each pair's sum/count; ``learning_progress_min_pct``/``confidence_min``
    as the global min across buckets, ``..._max_pct``/``..._max`` as the
    global max, ``..._last_pct``/``confidence_last`` from the most recent
    bucket that actually has a non-None value). ``daily`` is capped to the
    newest ``_RESEARCH_DAILY_EXPORT_MAX_BUCKETS`` buckets (newest-first) —
    the summary stays complete even when the list is truncated; excess is
    reported via ``records_truncated``/``truncation_reason``, never
    silently dropped. Each daily entry is a COMPACT
    ``_learning_research_daily_compact_entry()`` — no zero/None-field bloat.
    Currently-unproduced counters (``_RESEARCH_DAILY_UNPRODUCED_SUMMARY_FIELDS``
    — ``decision_count``/``heating_allowed_count``/``heating_blocked_count``,
    schema-defined but with no live producer yet) are omitted from
    ``summary`` while they are 0, so the export never misleadingly reads as
    "zero decisions ever made"; they reappear automatically the moment a
    real producer starts giving one of them a genuine non-zero value.

    Never raises and never breaks the export: a missing/unattached Learning
    shadow, a snapshot() failure, or an unexpected error anywhere in the
    aggregation yields an explicit ``{"available": False, ...}`` block
    instead of omitting the zone or failing the whole export.
    """
    try:
        shadow = getattr(coord, "_learning_shadow", None)
        if shadow is None:
            return {"available": False, "reason": "learning_engine_unavailable"}
        snapshot = shadow.research_daily_snapshot()
        if not isinstance(snapshot, dict):
            return {"available": False, "reason": "research_daily_unavailable"}

        from .learning.storage.research_daily_serialization import (
            deserialize_research_daily_bucket,
        )
        from .learning.storage.research_daily_persistence import (
            RESEARCH_DAILY_RETENTION_MAX_BUCKETS,
            RESEARCH_DAILY_RETENTION_MAX_DAYS,
        )
        from .learning.research_daily_schemas import RESEARCH_DAILY_SCHEMA_VERSION

        malformed_skipped = 0
        buckets: list = []
        for entry in snapshot.values():
            if not isinstance(entry, dict):
                malformed_skipped += 1
                continue
            bucket = deserialize_research_daily_bucket(entry)
            if bucket is None:
                malformed_skipped += 1
                continue
            buckets.append(bucket)

        retention = {
            "bounded": (
                RESEARCH_DAILY_RETENTION_MAX_DAYS is not None
                and RESEARCH_DAILY_RETENTION_MAX_BUCKETS is not None
            ),
            "max_days": RESEARCH_DAILY_RETENTION_MAX_DAYS,
            "max_buckets": RESEARCH_DAILY_RETENTION_MAX_BUCKETS,
        }

        block: dict = {
            "available": True,
            "schema_version": RESEARCH_DAILY_SCHEMA_VERSION,
            "retention": retention,
            "export_cap_buckets": _RESEARCH_DAILY_EXPORT_MAX_BUCKETS,
            "bucket_count": len(buckets),
            "coverage_start": None,
            "coverage_end": None,
            "records_truncated": 0,
            "truncation_reason": None,
            "malformed_skipped_count": malformed_skipped,
            "summary": {},
            "daily": [],
        }

        if buckets:
            buckets_newest_first = sorted(buckets, key=lambda b: b.bucket_date, reverse=True)
            bucket_dates = [b.bucket_date for b in buckets_newest_first]
            block["coverage_start"] = min(bucket_dates)
            block["coverage_end"] = max(bucket_dates)

            kept = buckets_newest_first[:_RESEARCH_DAILY_EXPORT_MAX_BUCKETS]
            truncated = len(buckets_newest_first) - len(kept)
            block["records_truncated"] = truncated
            block["truncation_reason"] = "export_cap_exceeded" if truncated > 0 else None
            block["daily"] = [_learning_research_daily_compact_entry(b) for b in kept]

            summary: dict = {name: 0 for name in _RESEARCH_DAILY_COUNTER_FIELDS}
            sum_totals = {sum_field: 0.0 for sum_field, _count_field in _RESEARCH_DAILY_SUM_COUNT_PAIRS}
            count_totals = {count_field: 0 for _sum_field, count_field in _RESEARCH_DAILY_SUM_COUNT_PAIRS}
            progress_min = None
            progress_max = None
            confidence_min = None
            confidence_max = None
            for b in buckets:
                for name in _RESEARCH_DAILY_COUNTER_FIELDS:
                    summary[name] += getattr(b, name)
                for sum_field, count_field in _RESEARCH_DAILY_SUM_COUNT_PAIRS:
                    sum_totals[sum_field] += getattr(b, sum_field)
                    count_totals[count_field] += getattr(b, count_field)
                if b.learning_progress_min_pct is not None:
                    progress_min = (
                        b.learning_progress_min_pct if progress_min is None
                        else min(progress_min, b.learning_progress_min_pct)
                    )
                if b.learning_progress_max_pct is not None:
                    progress_max = (
                        b.learning_progress_max_pct if progress_max is None
                        else max(progress_max, b.learning_progress_max_pct)
                    )
                if b.confidence_min is not None:
                    confidence_min = (
                        b.confidence_min if confidence_min is None
                        else min(confidence_min, b.confidence_min)
                    )
                if b.confidence_max is not None:
                    confidence_max = (
                        b.confidence_max if confidence_max is None
                        else max(confidence_max, b.confidence_max)
                    )
            summary["avg_overshoot_c"] = (
                round(sum_totals["overshoot_sum_c"] / count_totals["overshoot_count"], 3)
                if count_totals["overshoot_count"] > 0 else None
            )
            summary["avg_undershoot_c"] = (
                round(sum_totals["undershoot_sum_c"] / count_totals["undershoot_count"], 3)
                if count_totals["undershoot_count"] > 0 else None
            )
            summary["avg_comfort_error_c"] = (
                round(sum_totals["comfort_error_sum_c"] / count_totals["comfort_error_count"], 3)
                if count_totals["comfort_error_count"] > 0 else None
            )
            summary["learning_progress_min_pct"] = progress_min
            summary["learning_progress_max_pct"] = progress_max
            summary["confidence_min"] = confidence_min
            summary["confidence_max"] = confidence_max
            # "last" = from the most recent bucket (by bucket_date) that
            # actually carries a non-None value — a later day with no
            # progress/confidence sample must not shadow an earlier day's
            # real reading with a false None.
            progress_last = None
            confidence_last = None
            for b in buckets_newest_first:
                if progress_last is None and b.learning_progress_last_pct is not None:
                    progress_last = b.learning_progress_last_pct
                if confidence_last is None and b.confidence_last is not None:
                    confidence_last = b.confidence_last
                if progress_last is not None and confidence_last is not None:
                    break
            summary["learning_progress_last_pct"] = progress_last
            summary["confidence_last"] = confidence_last

            for field_name in _RESEARCH_DAILY_UNPRODUCED_SUMMARY_FIELDS:
                if summary.get(field_name) == 0:
                    del summary[field_name]

            block["summary"] = summary

        safe = _learning_strip_forbidden(block)
        try:
            from .learning.privacy import scan_payload
            if scan_payload(safe):
                return {"available": False, "reason": "privacy_scan_failed"}
        except Exception:
            pass  # scan itself failing is non-fatal — block already key-stripped
        return safe
    except Exception as err:
        return {"available": False, "research_daily_error": str(err)}


# Maximum events actually returned in the "events" list, independent of the
# store's own 750-record cap. The store cap protects storage size; this
# separate, smaller cap keeps a single support export file readable — 750
# entries in a JSON file a user is asked to review before sharing is not
# "small". Centralised so a later calibration pass has one place to change.
_SUPPORT_EVENT_EXPORT_MAX_RECORDS = 200

# Same eviction-priority direction as support_event_persistence.py's cap
# eviction (INFO first, CRITICAL last) — re-declared locally rather than
# importing that module's private rank map, since export.py should not reach
# into another module's underscore-prefixed internals for a one-line lookup.
_SUPPORT_EVENT_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def _learning_critical_events_export(coord, *, now: datetime) -> dict:
    """Return a small, bounded, public-safe Support Critical Event timeline
    summary for support export.

    Reads ONLY the already-in-memory
    ``LearningShadowController.support_critical_events_snapshot()`` — no
    store read here, no new events created, no runtime/coordinator touched.
    Uses ``support_event_for_export()`` (support_event_serialization.py) per
    entry, which already drops ``event_id`` and bounds ``details`` — this
    function adds no additional per-event fields beyond what that helper
    already produces.

    The underlying store is already bounded (48h retention, 750-record cap,
    severity-priority eviction — support_event_persistence.py). This export
    applies a SEPARATE, smaller cap
    (``_SUPPORT_EVENT_EXPORT_MAX_RECORDS`` = 200) to keep a single export
    file readable; when more valid events exist than that cap, the kept
    subset is chosen by the same severity-priority-then-recency ordering the
    store itself uses for eviction (critical/newest preferred), and the
    excess is reported via ``records_truncated``/``truncation_reason`` — no
    event is silently dropped without being counted.

    Never raises and never breaks the export: a missing/unattached Learning
    shadow, a snapshot() failure, or an unexpected error anywhere in the
    aggregation yields an explicit ``{"available": False, ...}`` block
    instead of omitting the zone or failing the whole export. Malformed
    individual entries (not a dict, or rejected by
    ``support_event_for_export()``) are skipped and counted in
    ``malformed_skipped_count`` — they never abort the summary.
    """
    try:
        shadow = getattr(coord, "_learning_shadow", None)
        if shadow is None:
            return {"available": False, "reason": "learning_engine_unavailable"}
        snapshot = shadow.support_critical_events_snapshot()
        if not isinstance(snapshot, dict):
            return {"available": False, "reason": "critical_events_unavailable"}

        from .learning.storage.support_event_serialization import support_event_for_export
        from .learning.storage.support_event_persistence import (
            SUPPORT_EVENT_RETENTION_MAX_AGE_DAYS,
        )

        malformed_skipped = 0
        candidates: list[tuple[dict, datetime | None]] = []
        for entry in snapshot.values():
            if not isinstance(entry, dict):
                malformed_skipped += 1
                continue
            view = support_event_for_export(entry)
            if view is None:
                malformed_skipped += 1
                continue
            ts_raw = view.get("ts")
            ts_parsed = None
            if isinstance(ts_raw, str):
                try:
                    ts_text = ts_raw[:-1] + "+00:00" if ts_raw.endswith("Z") else ts_raw
                    ts_parsed = datetime.fromisoformat(ts_text)
                    if ts_parsed.tzinfo is None:
                        ts_parsed = ts_parsed.replace(tzinfo=timezone.utc)
                except Exception:
                    ts_parsed = None
            candidates.append((view, ts_parsed))

        records_available = len(candidates)

        # Selection when over the export cap: severity-priority first
        # (critical kept over info), newest first as tiebreak — unknown-
        # timestamp entries sort last within their severity (never preferred
        # over a known-recent one), mirroring the store's own conservative
        # "unknown -> not preferred" eviction stance.
        def _sort_key(item):
            view, ts_parsed = item
            severity_rank = _SUPPORT_EVENT_SEVERITY_RANK.get(view.get("severity"), 0)
            has_ts = ts_parsed is not None
            return (-severity_rank, not has_ts, -(ts_parsed.timestamp() if has_ts else 0.0))

        ordered = sorted(candidates, key=_sort_key)
        kept = ordered[:_SUPPORT_EVENT_EXPORT_MAX_RECORDS]
        records_truncated = records_available - len(kept)

        # Display order: newest first, independent of the selection order above.
        kept_sorted = sorted(
            kept, key=lambda item: (item[1] is None, -(item[1].timestamp() if item[1] else 0.0)),
        )
        events = [view for view, _ts in kept_sorted]

        known_ts = [ts for _view, ts in candidates if ts is not None]
        coverage_start = min(known_ts).isoformat() if known_ts else None
        coverage_end = max(known_ts).isoformat() if known_ts else None

        retention_h = SUPPORT_EVENT_RETENTION_MAX_AGE_DAYS * 24
        full_window_covered = None
        store_warmup = None
        if known_ts:
            oldest_age_h = (now - min(known_ts)).total_seconds() / 3600.0
            full_window_covered = oldest_age_h >= (retention_h - 1.0)  # 1h tolerance
            store_warmup = not full_window_covered

        block: dict = {
            "available": True,
            "requested_window_h": retention_h,
            "coverage_scope": "persistent_support_critical_events",
            "persistent_store_enabled": getattr(shadow, "capture_stores", None) is not None,
            "persistent_store_retention_h": retention_h,
            "records_available": records_available,
            "records_exported": len(events),
            "records_truncated": records_truncated,
            "truncation_reason": "export_cap_exceeded" if records_truncated > 0 else None,
            "coverage_start": coverage_start,
            "coverage_end": coverage_end,
            "full_window_covered": full_window_covered,
            "store_warmup": store_warmup,
            "malformed_skipped_count": malformed_skipped,
            "events": events,
        }
        safe = _learning_strip_forbidden(block)
        try:
            from .learning.privacy import scan_payload
            if scan_payload(safe):
                return {"available": False, "reason": "privacy_scan_failed"}
        except Exception:
            pass  # scan itself failing is non-fatal — block already key-stripped
        return safe
    except Exception as err:
        return {"available": False, "critical_events_error": str(err)}


def _learning_pending_data(coord, zone_id: str) -> dict | None:
    """Return Learning pending-attribution summary for support export, or None."""
    try:
        rt = _learning_runtime(coord)
        if rt is None:
            return None
        return rt.pending_attribution_summary(zone_id)
    except Exception:
        return None


def _device_profile_export(coord: Any) -> dict | None:
    """Return a small, public-safe device-compatibility summary for support
    export: the first configured TRV's matched profile display name,
    active/observe-only mode, and warning (if any).

    Pure read of the already-computed per-entity profile map from device
    detection (coordinator.py's ``_device_profiles``, populated by
    ``async_detect_device_entities()``) — never re-matches, never changes
    match order, never enforces anything. No entity ids, no device ids: only
    the profile's own display name/flags, which are public source data, not
    per-installation data.

    A zone with multiple TRVs is represented by its first configured
    device's profile only (matches how other zone-level display values in
    this export are already single, effective values, not per-TRV lists).

    Returns None when no coordinator is attached or no profile has been
    detected yet (e.g. right after setup, before device detection has run).
    Never raises.
    """
    try:
        from .device_profiles import device_profile_status
        device_profiles = getattr(coord, "_device_profiles", None) or {}
        profile = next(iter(device_profiles.values()), None)
        if profile is None:
            return None
        return device_profile_status(profile)
    except Exception:
        return None


def _learning_adaptation_summary(coord, zone_id: str) -> dict | None:
    """Return passive adaptation candidate counts for support export, or None.

    Summary only — no trace details. Never modifies control state.
    """
    try:
        from .learning.adaptation import (
            OutcomeSignal, SituationContext, suggest_candidates, AdaptationLifecycle,
        )
        rt = _learning_runtime(coord)
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
            confounder_contamination=(_rejections > 0),
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


def _learning_confounder_ratio(coord: Any, zone_id: str) -> float:
    """Compute rejection-based confounder ratio from the outcome model. Returns 0.0 on error."""
    try:
        rt = _learning_runtime(coord)
        if rt is None:
            return 0.0
        zr = rt._zones.get(zone_id)
        if zr is None:
            return 0.0
        om = zr.orchestrator.models.get("outcome")
        if om is None:
            return 0.0
        d = om.diagnostics()
        fc, pc = d.full_partial
        total = fc + pc
        rej = sum(d.rejection_counts.values()) if d.rejection_counts else 0
        att = total + rej
        return float(rej / att) if att > 0 else 0.0
    except Exception:
        return 0.0


def _learning_adaptation_history_summary(coord, zone_id: str) -> dict:
    """Return adaptation candidate history summary for support export.

    Reads the in-memory candidate history from coord._learning_shadow.
    Never raises.

    ``entry_count``/``promotion_ready_count``/``blocked_count`` are real,
    currently-active passive candidate-tracking counts — not inert. Only
    ``shadow_preview_count`` (mirrors ``promotion_ready_count``) and
    ``application_enabled`` describe the separate, currently-inactive
    application/orchestration layer (see
    _learning_application_lifecycle_summary's docstring); ``application_layer_status``
    marks specifically those two fields as reserved/foundation-only, without
    implying the rest of this block is inactive too.
    """
    _zero = {"entry_count": 0, "promotion_ready_count": 0, "blocked_count": 0,
             "shadow_preview_count": 0, "application_enabled": False,
             "application_layer_status": "reserved", "last_error": None}
    try:
        shadow = getattr(coord, "_learning_shadow", None)
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
                pgr = evaluate_promotion_readiness(
                    entry, span_days=span_days,
                    confounder_ratio=_learning_confounder_ratio(coord, zone_id),
                )
                if pgr.readiness is PromotionReadiness.ELIGIBLE:
                    ready += 1
                else:
                    blocked += 1
            except Exception:
                blocked += 1
        return {
            **_zero,
            "entry_count": len(history),
            "promotion_ready_count": ready,
            "blocked_count": blocked,
            "shadow_preview_count": ready,
            "last_error": shadow.adaptation_last_error(),
        }
    except Exception as err:
        return {**_zero, "last_error": str(err)}


def _learning_adaptation_history_for_research(
    history_entries: list, *, lifecycle_state: Any = None,
) -> list[dict]:
    """Convert adaptation candidate history entries to research-safe export dicts.

    Args:
        history_entries: list of (CandidateHistoryEntry, span_days, confounder_ratio)
            tuples. Pass [] when no runtime accumulation is available yet.
        lifecycle_state: current ApplicationLifecycleState (or None), used to
            attach a read-only "application_orchestration_preview" per entry.

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
                d = adaptation_history_entry_for_research_export(
                    entry,
                    span_days=span_days,
                    confounder_ratio=confounder_ratio,
                )
                d["application_orchestration_preview"] = _learning_orchestration_preview_dict(
                    entry,
                    lifecycle_state=lifecycle_state,
                    span_days=span_days,
                    confounder_ratio=confounder_ratio,
                )
                result.append(d)
            except Exception:
                pass
    except Exception:
        pass
    return result


def _learning_orchestration_preview_dict(
    entry: Any, *, lifecycle_state: Any, span_days: float, confounder_ratio: float,
) -> dict:
    """Build a public-safe, timestamp-free orchestration preview for one
    candidate history entry, for the anonymized research export.

    Runtime context is intentionally always None here: the export path has no
    safe live source for proposed_delta / current_cumulative_delta / safety
    flags, so guessing them would be unsafe. Passing None makes the preview
    block conservatively via "unknown_context" instead — application_enabled
    and would_apply always report False; would_apply_if_enabled reports the
    same conservative False in real export (an actually-live runtime context
    is required for that field to read True, which only test/helper callers
    of evaluate_application_orchestration() supply directly).

    Never raises. Read-only — never mutates history or lifecycle state.
    """
    try:
        from .learning.adaptation.orchestrator import evaluate_application_orchestration
        result = evaluate_application_orchestration(
            history_entry=entry,
            lifecycle_state=lifecycle_state,
            runtime_context=None,
            span_days=span_days,
            confounder_ratio=confounder_ratio,
        )
        return result.to_research_dict()
    except Exception:
        return {
            "application_enabled": False,
            "promotion_readiness": "blocked",
            "would_apply": False,
            "would_apply_if_enabled": False,
            "blocked_by_unknown_context": False,
            "candidate_type": None,
            "direction": None,
            "blocked_reasons": ["orchestration_preview_error"],
            "safety_reasons": [],
            "bounded_delta_min": None,
            "current_cumulative_delta_min": None,
            "next_cumulative_delta_min": None,
            "monitoring_required": True,
            "rollback_supported": True,
            "lifecycle_existing_status": None,
            "cooldown_active": False,
        }


def _learning_orchestration_preview_summary(coord: Any, zone_id: str) -> dict:
    """Aggregate orchestration-preview counts across candidate history for
    support export. Counts/status only — no per-entry detail, never raises.

    Runtime context is intentionally None for every entry (see
    _learning_orchestration_preview_dict) — this keeps would_apply_count always 0
    and would_apply_if_enabled_count conservative (unknown_context blocks
    both). lifecycle_blocked_count remains meaningful regardless, since
    lifecycle-state gating does not depend on runtime_context.

    ``status``/``active``/``reason`` mark this whole block as a preview of a
    currently-inactive application/orchestration layer (see
    _learning_application_lifecycle_summary's docstring) — the per-entry counts
    stay real diagnostic previews ("what would happen if this were on"), not
    a live feature a maintainer should expect to see actually apply anything.
    """
    _zero = {
        "status": "reserved",
        "active": False,
        "reason": "application_layer_not_active_in_this_version",
        "entry_count": 0,
        "would_apply_count": 0,
        "would_apply_if_enabled_count": 0,
        "blocked_count": 0,
        "lifecycle_blocked_count": 0,
        "safety_blocked_count": 0,
        "application_enabled": False,
        "last_error": None,
    }
    try:
        shadow = getattr(coord, "_learning_shadow", None)
        if shadow is None:
            return _zero
        history = shadow.adaptation_history_snapshot()
        if not history:
            return {**_zero, "last_error": shadow.adaptation_last_error()}

        lifecycle_state = getattr(shadow, "_application_lifecycle_state", None)
        from .learning.adaptation.orchestrator import evaluate_application_orchestration
        from datetime import datetime as _dt

        would_apply = 0
        would_apply_if_enabled = 0
        blocked = 0
        lifecycle_blocked = 0
        safety_blocked = 0
        for entry in history.values():
            try:
                span_days = 0.0
                if entry.first_seen_ts and entry.last_seen_ts:
                    t0 = _dt.fromisoformat(entry.first_seen_ts.replace("Z", "+00:00"))
                    t1 = _dt.fromisoformat(entry.last_seen_ts.replace("Z", "+00:00"))
                    span_days = max(0.0, (t1 - t0).total_seconds() / 86400.0)
                result = evaluate_application_orchestration(
                    history_entry=entry,
                    lifecycle_state=lifecycle_state,
                    runtime_context=None,
                    span_days=span_days,
                    confounder_ratio=_learning_confounder_ratio(coord, zone_id),
                )
                if result.would_apply:
                    would_apply += 1
                if result.would_apply_if_enabled:
                    would_apply_if_enabled += 1
                if result.blocked_reasons:
                    blocked += 1
                if any(r.startswith("lifecycle_") for r in result.blocked_reasons):
                    lifecycle_blocked += 1
                if result.safety_reasons:
                    safety_blocked += 1
            except Exception:
                blocked += 1
        return {
            **_zero,
            "entry_count": len(history),
            "would_apply_count": would_apply,
            "would_apply_if_enabled_count": would_apply_if_enabled,
            "blocked_count": blocked,
            "lifecycle_blocked_count": lifecycle_blocked,
            "safety_blocked_count": safety_blocked,
            "last_error": shadow.adaptation_last_error(),
        }
    except Exception as err:
        return {**_zero, "last_error": str(err)}


def _learning_application_lifecycle_summary(coord: Any, zone_id: str) -> dict:
    """Return adaptation application lifecycle summary for support export.

    Read-only, no mutation, no control effect. Never raises.

    The application/orchestration layer is foundation-only in this version —
    ``ApplicationPolicy.application_enabled`` is a global kill-switch that is
    always False (see learning/adaptation/application.py), and nothing in the
    live runtime ever adopts/applies a candidate. Every count below is
    therefore always 0, not a sign of a broken feature — ``status``/``active``/
    ``reason`` make that explicit so a maintainer reading the export does not
    mistake this for a live, currently-inactive-by-configuration feature.
    """
    _zero = {
        "status": "reserved",
        "active": False,
        "reason": "application_layer_not_active_in_this_version",
        "state_entry_count": 0,
        "active_count": 0,
        "rollback_recommended_count": 0,
        "adoption_ready_count": 0,
        "adopted_count": 0,
        "rolled_back_count": 0,
        "expired_count": 0,
        "last_error": None,
    }
    try:
        shadow = getattr(coord, "_learning_shadow", None)
        if shadow is None:
            return _zero
        snapshot = shadow.application_lifecycle_snapshot()
        last_err = getattr(shadow, "_application_lifecycle_last_error", None)
        if snapshot is None:
            return {**_zero, "last_error": last_err}
        return {
            **_zero,
            "state_entry_count": snapshot.get("total", 0),
            "active_count": snapshot.get("active", 0),
            "rollback_recommended_count": snapshot.get("rollback_recommended", 0),
            "adoption_ready_count": snapshot.get("adoption_ready", 0),
            "adopted_count": snapshot.get("adopted", 0),
            "rolled_back_count": snapshot.get("rolled_back", 0),
            "expired_count": snapshot.get("expired", 0),
            "last_error": last_err,
        }
    except Exception as err:
        return {**_zero, "last_error": str(err)}


def _learning_reserved_diagnostics_summary(coord: Any, zone_id: str) -> dict:
    """Collapsed placeholder for the application/orchestration layer in the
    support export.

    That layer is foundation-only in this version — ``ApplicationPolicy.
    application_enabled`` is a global kill-switch that is always False (see
    _learning_application_lifecycle_summary's docstring), and
    _learning_orchestration_preview_summary's ``would_apply_count`` is always 0
    for the same reason. Replaces the two previous always-inert blocks
    (``adaptation_application``, ``orchestration_preview`` — every counter in
    both was always 0) with one compact, clearly-labeled placeholder instead
    of two blocks of always-zero counters that read as noise in a support
    export. Any internal error from either underlying summary still surfaces
    here via ``last_error`` (support-relevant even though the feature itself
    is inactive). Never raises.
    """
    last_error = None
    try:
        app = _learning_application_lifecycle_summary(coord, zone_id)
        if app.get("last_error"):
            last_error = app["last_error"]
    except Exception:
        pass
    if last_error is None:
        try:
            preview = _learning_orchestration_preview_summary(coord, zone_id)
            if preview.get("last_error"):
                last_error = preview["last_error"]
        except Exception:
            pass
    return {
        "available": False,
        "reason": "reserved_inactive",
        "last_error": last_error,
    }


def _learning_application_lifecycle_research_entries(coord: Any) -> list:
    """Return public-safe application lifecycle state entries for research export.

    No timestamps, no entity IDs, no zone names — only boolean / numeric fields.
    Never raises; returns empty list on any error or when no state exists.
    """
    try:
        shadow = getattr(coord, "_learning_shadow", None)
        if shadow is None:
            return []
        state = getattr(shadow, "_application_lifecycle_state", None)
        if state is None or not getattr(state, "entries", None):
            return []
        from datetime import datetime as _dt, timezone as _tz
        now_ts = _dt.now(_tz.utc).isoformat()
        from .learning.adaptation.application_state import (
            application_state_entry_for_research_export,
        )
        result: list = []
        for entry in state.entries.values():
            try:
                result.append(
                    application_state_entry_for_research_export(entry, now_ts=now_ts)
                )
            except Exception:
                pass
        return result
    except Exception:
        return []


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
        # The legacy learning engine is frozen — learning_engine.freeze() in
        # __init__.py — so this is a static historical snapshot, not
        # live-updating data. analytics is computed from the raw learning
        # dict (pure numeric aggregates, already export-safe); the exported
        # historical_learning_snapshot itself is a separate, sanitized,
        # allow-listed Deep-Research view built by
        # _historical_learning_snapshot_for_research() below — see that
        # function's docstring for why the raw dict is never exported as-is.
        learning: dict = le.get_export_data(entry.entry_id) if le is not None else {}
        analytics = _compute_analytics(learning)
        runtime_models = _learning_research_data(coord, entry.entry_id) if coord is not None else None
        learning_progress = (
            _learning_progress_export(coord) if coord is not None
            else {"available": False, "reason": "no_coordinator"}
        )
        episode_history = (
            _learning_episode_history_export(coord, now=ts) if coord is not None
            else {"available": False, "reason": "no_coordinator"}
        )
        research_daily = (
            _learning_research_daily_export(coord, now=ts) if coord is not None
            else {"available": False, "reason": "no_coordinator"}
        )
        # Independent of coordinator liveness — see _learning_storage_context_export()'s
        # docstring: store-level context, never a substitute for the content-time
        # fields already inside episode_history/research_daily/historical_learning_snapshot.
        storage_context = await _learning_storage_context_export(hass, entry.entry_id, now=ts)

        zones.append({
            "zone_hash": _zone_hash(entry.entry_id),
            **meta,
            "historical_learning_snapshot": _historical_learning_snapshot_for_research(learning),
            "analytics": analytics,
            "runtime_models": runtime_models,
            "learning_progress": learning_progress,
            "episode_history": episode_history,
            "research_daily": research_daily,
            "storage_context": storage_context,
        })

    _tz_name = str(hass.config.time_zone) if hass.config.time_zone else None
    export: dict = {
        "thermosmart_version": VERSION,
        "export_type": "research",
        "export_format_version": EXPORT_FORMAT_VERSION,
        "export_timestamp": ts.isoformat(),
        "exported_at_utc": ts.isoformat(),
        "exported_at_local": dt_util.as_local(ts).isoformat(),
        "timezone": _tz_name,
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


def _storage_summary_age_minutes(updated_at_utc: Any, now: datetime) -> float | None:
    """Minutes between ``updated_at_utc`` (ISO-8601, 'Z' or '+00:00' suffix)
    and ``now`` — or ``None`` on anything not a clean, parseable timestamp.
    Never raises."""
    if not isinstance(updated_at_utc, str):
        return None
    try:
        updated = datetime.fromisoformat(updated_at_utc.replace("Z", "+00:00"))
        now_utc = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return round((now_utc.astimezone(timezone.utc) - updated).total_seconds() / 60.0, 1)
    except (ValueError, TypeError, OverflowError):
        return None


async def _storage_metadata_stores_for_export(
    hass: HomeAssistant, learning_zone_id: str, *, now: datetime,
) -> dict:
    """Shared core for the Support Export ``storage_summary`` (Commit C) and
    the Research Export ``storage_context`` (Commit D): reads the zone's
    ``StorageMetadataStore`` (Commit A/B) — never the real data stores it
    describes — and reshapes it into ``{"available": True, "stores": {...}}``
    or ``{"available": False, "reason": ...}``.

    Per store: whether it exists, its created/updated timestamps, an
    ``age_minutes`` derived at export time, the last write reason, and the
    key-migration state. No raw learning data, no entity/device/person ids,
    no real zone/sensor names — only the small store-name labels and enum
    values ``StorageMetadataStore`` itself already restricts to (see
    stores.py's ``StorageMetadataStore`` docstring).

    ``runtime_snapshot``/``raw_segments``/``global_index`` are intentionally
    never shown here (neither ``exists: true`` nor ``exists: false``): none
    of them is wired into ``StorageMetadataStore`` as of Commit B — showing
    them at all would wrongly imply a tracking attempt that never happens
    for these three. Every store name that actually appears below is exactly
    whatever the zone's own index currently contains — no fixed/guessed
    store-name list, so newly tracked raw tracks etc. show up without an
    export.py change.

    Never raises and never breaks the export it's used from: a missing or
    corrupt ``StorageMetadataStore`` (construction failure, ``StoreVersionError``
    on a mismatched schema version) yields ``{"available": False, "reason":
    ...}`` instead of failing or omitting the zone, matching the established
    fallback shape used by ``_learning_critical_events_export`` and friends
    above. A single malformed per-store entry (not a dict) is skipped rather
    than aborting the whole summary; a missing/unparseable timestamp simply
    omits ``age_minutes`` for that store instead of raising.
    """
    try:
        from .learning.storage.stores import HomeAssistantStoreFactory, StorageMetadataStore
        meta_store = StorageMetadataStore(HomeAssistantStoreFactory(hass), learning_zone_id)
        summary = await meta_store.get_store_summary()
    except Exception:
        return {"available": False, "reason": "storage_metadata_unavailable"}

    stores_out: dict = {}
    for store_name, entry in (summary.get("stores") or {}).items():
        if not isinstance(entry, dict):
            continue  # malformed per-store entry -> skip, never abort the summary
        if not entry.get("exists"):
            stores_out[store_name] = {"exists": False}
            continue
        out: dict = {"exists": True}
        created_at_utc = entry.get("created_at_utc")
        updated_at_utc = entry.get("updated_at_utc")
        if isinstance(created_at_utc, str):
            out["created_at_utc"] = created_at_utc
        if isinstance(updated_at_utc, str):
            out["updated_at_utc"] = updated_at_utc
        age_minutes = _storage_summary_age_minutes(updated_at_utc, now)
        if age_minutes is not None:
            out["age_minutes"] = age_minutes
        last_write_reason = entry.get("last_write_reason")
        if isinstance(last_write_reason, str):
            out["last_write_reason"] = last_write_reason
        storage_key_state = entry.get("storage_key_state")
        if isinstance(storage_key_state, str):
            out["storage_key_state"] = storage_key_state
        stores_out[store_name] = out

    return {"available": True, "stores": stores_out}


async def _learning_storage_summary_export(
    hass: HomeAssistant, learning_zone_id: str, *, now: datetime,
) -> dict:
    """Per-zone Storage-Metadata summary for the support export (Commit C).

    Thin wrapper around ``_storage_metadata_stores_for_export()`` — see that
    function's docstring for the full shape/privacy/fallback contract.
    """
    return await _storage_metadata_stores_for_export(hass, learning_zone_id, now=now)


async def _learning_storage_context_export(
    hass: HomeAssistant, learning_zone_id: str, *, now: datetime,
) -> dict:
    """Per-zone Storage-Metadata context for the research export (Commit D).

    Same underlying data as ``_learning_storage_summary_export()`` (both
    delegate to ``_storage_metadata_stores_for_export()``), but framed for a
    different audience: Research Export readers work with *content time*
    (an episode's/bucket's/observation's own ``ts``/``hour``/``minute``/
    ``weekday`` fields — when the underlying heating event happened) and
    could otherwise easily mistake a store's ``updated_at_utc``/
    ``age_minutes`` for another content timestamp. ``granularity``:
    ``"store_level"`` and ``timestamp_semantics``: ``"store_write_time"`` are
    small, constant, machine-readable markers (deliberately not a prose
    ``note`` — this export favors compact fields) making that distinction
    explicit and unambiguous wherever this block is read, independent of
    whether any store data is currently available.

    This block is pure additional context: it never changes, replaces, or
    reads from the Episode/Research-Daily/Observation payloads themselves —
    those keep their own existing content-time fields exactly as before.
    """
    result = await _storage_metadata_stores_for_export(hass, learning_zone_id, now=now)
    return {
        "granularity": "store_level",
        "timestamp_semantics": "store_write_time",
        **result,
    }


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
            zone_info["runtime_health"] = _learning_health_data(coord)
            zone_info["runtime_pending"] = _learning_pending_data(coord, entry.entry_id)
            zone_info["adaptation"] = _learning_adaptation_summary(coord, entry.entry_id)
            zone_info["adaptation_history"] = _learning_adaptation_history_summary(
                coord, entry.entry_id
            )
            zone_info["reserved_diagnostics"] = _learning_reserved_diagnostics_summary(
                coord, entry.entry_id
            )
            zone_info["critical_events"] = _learning_critical_events_export(coord, now=ts)
            zone_info["device_profile"] = _device_profile_export(coord)
        else:
            zone_info["runtime_state"] = None
            zone_info["runtime_health"] = None
            zone_info["runtime_pending"] = None
            zone_info["adaptation"] = None
            zone_info["adaptation_history"] = None
            zone_info["reserved_diagnostics"] = {
                "available": False, "reason": "no_coordinator", "last_error": None,
            }
            zone_info["critical_events"] = {"available": False, "reason": "no_coordinator"}
            zone_info["device_profile"] = None

        # Independent of coordinator liveness — StorageMetadataStore only
        # needs hass + the zone's entry_id (== learning_zone_id), and a
        # zone's stores can outlive a currently-unloaded coordinator.
        zone_info["storage_summary"] = await _learning_storage_summary_export(
            hass, entry.entry_id, now=ts
        )

        zones.append(zone_info)

    _tz_name = str(hass.config.time_zone) if hass.config.time_zone else None
    export: dict = {
        "thermosmart_version": VERSION,
        "export_type": "support",
        "export_format_version": EXPORT_FORMAT_VERSION,
        "export_timestamp": ts.isoformat(),
        "exported_at_utc": ts.isoformat(),
        "exported_at_local": dt_util.as_local(ts).isoformat(),
        "timezone": _tz_name,
        "system": {
            "ha_version": HA_VERSION,
            "zone_count": len(zones),
        },
        "storage_layout": {
            "note": (
                "ThermoSmart stores learning data per zone. The active learning "
                "runtime store uses a hashed zone key: thermosmart_learning__<hash>. "
                "Segmented learning stores use: "
                "thermosmart_learning__<zone_entry_id>__<suffix>. Older "
                "thermosmart_le2__ keys (pre-migration installs) are read once as a "
                "fallback and then mirrored onto the keys above; they are kept as a "
                "safety fallback and not deleted automatically. The separate, older "
                "thermosmart_learning_data store is unrelated legacy/read-only data. "
                "Do not manually delete any thermosmart_learning__<hash> or "
                "thermosmart_le2__<hash> files unless you intentionally want to "
                "reset runtime learning state for that zone."
            ),
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
