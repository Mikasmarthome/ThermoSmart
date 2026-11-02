"""Diagnostics data-shaping helpers for ThermoSmart.

Pure functions that turn live coordinator/Learning state into privacy-safe,
bounded summaries. Consumed by diagnostics.py's async_get_config_entry_
diagnostics() (Home Assistant's standard "Download diagnostics" action under
Settings -> Devices & services -> ThermoSmart) — nothing here writes a file,
serves HTTP, or notifies anyone; delivery is entirely Home Assistant's job.

Privacy contract
----------------
Diagnostics data contains:
  - ThermoSmart version
  - Per-zone: TRV count, sensor counts, feature flags (booleans only)
  - Per-zone: all numeric learning data (observations, rates, confidence, …)

Diagnostics data does NOT contain:
  - Passwords or authentication tokens of any kind
  - Entity IDs, device names, or integration names
  - Person names or user identifiers
  - Street addresses or geographic coordinates
  - Decision IDs, episode IDs, or internal runtime identifiers

Timestamps are intentionally retained:
  Observation timestamps (ts, hour, minute, weekday) are required for
  longitudinal learning analysis and are the primary reason this data is
  valuable for diagnosis.  A series of heating timestamps can reveal usage
  patterns (presence, sleep schedule, away periods).  Diagnostics downloads
  in Home Assistant already require the downloading user to be logged in.

has_forecast note:
  has_forecast is derived from whether a weather entity is configured.
  It does not guarantee that the entity actually provides forecast data —
  some weather integrations only expose current conditions.

Zone identity: each zone_id (HA entry_id UUID) is replaced with a deterministic
12-char hex digest.  Exports from the same installation share the same digests,
making longitudinal data correlatable without being reversible.

Learning progress:
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

Research daily buckets:
  Per-zone "research_daily" block — a BOUNDED SUMMARY of
  LearningShadowController.research_daily_snapshot() (already-in-memory,
  no new store read). "summary" aggregates counters/averages/progress-
  confidence min/max/last across ALL valid buckets; "daily" is a compact,
  newest-first list capped at 90 days (records_truncated/truncation_reason
  report any excess) — the summary itself is never truncated. No episode/
  event ids, no raw events, no trajectories — every field is already one of
  ResearchDailyBucket's own fixed scalar aggregates.

Support critical event timeline:
  Per-zone "critical_events" block — reads ONLY the already-in-memory
  LearningShadowController.support_critical_events_snapshot(), no store
  read. Each event is rendered via support_event_for_export() (drops
  event_id, bounds "details"). A separate, smaller cap (200) applies on top
  of the store's own 750-record cap so a single diagnostics download stays
  readable; any excess is reported via records_truncated, never silently
  dropped. Coverage/retention metadata (coverage_start/end,
  persistent_store_retention_h, full_window_covered, store_warmup) describes
  the underlying data span, independent of that cap.

Delivery:
  Home Assistant's own diagnostics download mechanism (Settings -> Devices &
  services -> ThermoSmart -> Download diagnostics) handles authentication,
  signing, and transport — ThermoSmart has no HTTP endpoint, no file on
  disk, and no cleanup to manage of its own.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from homeassistant.core import HomeAssistant

from .const import (
    DOMAIN,
    CONF_WEATHER_ENTITY,
    CONF_OUTDOOR_SOLAR_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_RAIN_SENSOR,
)

_LOGGER = logging.getLogger(__name__)

_ANON_SALT = "thermosmart_le_export_v1"

# ── Learning privacy helpers ───────────────────────────────────────────────────────

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


def _learning_health_data(coord) -> dict | None:
    """Return Learning RuntimeHealth as a plain dict for support export, or None."""
    try:
        rt = _learning_runtime(coord)
        if rt is None:
            return None
        return dataclasses.asdict(rt.health())
    except Exception:
        return None


def _learning_effectiveness_export(coord) -> dict:
    """Read-only learning-effectiveness diagnostics for one zone.

    Assembles the per-model detail, per-builder episode lifecycle, boost
    effectiveness, open preheat-comparison stats and persistence detail that the
    shadow controller exposes (all pure reads — collection never mutates model
    state). ``boost.prior_configured`` is added here from the zone config, which
    the runtime cannot see.

    Counters are session-scoped unless noted (model/builder tallies reset on
    restart; the models' own rejection_counts come from persisted model state).
    Returns ``{"available": False}`` when no learning shadow is attached. No
    entity/device/person ids, no absolute timestamps — counts, enum reasons,
    confidences and relative ages only.
    """
    shadow = getattr(coord, "_learning_shadow", None)
    if shadow is None:
        return {"available": False, "reason": "learning_engine_unavailable"}
    try:
        boost = shadow.boost_effectiveness_safe()
        try:
            cfg = getattr(coord, "zone_cfg", {}) or {}
            prior = cfg.get("boost_bootstrap_prior_c")
            boost["prior_configured"] = bool(prior is not None and float(prior) > 0.0)
        except Exception:
            boost["prior_configured"] = None
        return {
            "available": True,
            "models": shadow.learning_model_detail_safe(),
            "builders": shadow.builder_lifecycle_safe(),
            "boost": boost,
            "preheat_comparisons": shadow.preheat_comparison_stats_safe(),
            "persistence": shadow.persistence_detail_safe(),
        }
    except Exception:
        return {"available": False}


def _learning_application_export(coord) -> dict:
    """Read-only per-room learned-value APPLICATION diagnostics (Phase A).

    Sourced entirely from already-computed state — the recommendation (zone
    data) and the authoritative, id-free ``LiveDecisionRecord``
    (``coord._last_live_decision``). Pure reads: never computes control, never
    writes a setpoint, never creates runtime state. Each sub-block is
    independently ``available``.

    Preheat — distinguishes learned vs deterministic selection, the candidate /
    baseline / selected minutes, the exact fallback reason, and a *calculated*
    duration (``calculated``) from an *active* preheat window (``active_window``).
    TPI — exposes the authoritative ``coef_source`` (never inferred from the
    coefficients differing from defaults), candidate vs selected (post blend /
    step-limit) coefficients, the fallback gate, and marks that these are the
    coefficient SELECTION, not actuator dispatch. Boost — proposed /
    policy-approved / lifecycle-approved, requested setpoint, dispatch status +
    confirmed service-call result, a NEW write vs a setpoint already within
    tolerance, and the exact block reason; physical heating effectiveness is
    never inferred from a successful service call.

    Privacy: counts, enums, temperatures/offsets and booleans only — the
    ``LiveDecisionRecord`` carries no entity/device ids by construction.
    """
    try:
        zone = {}
        try:
            zone = (getattr(coord, "data", None) or {}).get("zone", {}) or {}
        except Exception:
            zone = {}
        ldr = getattr(coord, "_last_live_decision", None)

        def _g(name, default=None):
            return getattr(ldr, name, default) if ldr is not None else default

        # ── Preheat ──────────────────────────────────────────────────────────
        # Sourced from the recommendation (zone data), with the LiveDecisionRecord
        # as a fallback for status. The deterministic baseline is RECOMPUTED with
        # the pure classmethod for comparison (no side effects, no state change).
        _ph_status = zone.get("preheat_status") or _g("preheat_status")
        _ph_selected = zone.get("preheat_minutes")
        if _ph_selected is None:
            _ph_selected = _g("preheat_minutes")
        _learned = _ph_status in ("valid", "valid_hl_prior")
        _baseline_min = None
        try:
            from .learning.runtime.ha_integration import LearningShadowController as _LSC
            _cur = zone.get("current_temp")
            _comfort = (getattr(coord, "zone_cfg", {}) or {}).get("comfort_temp", 21.0)
            if _cur is not None:
                _baseline_min, _ = _LSC.compute_deterministic_preheat_baseline(_cur, _comfort)
        except Exception:
            _baseline_min = None
        preheat = {
            "available": True,
            "source": "learned" if _learned else (_ph_status or "unknown"),
            "selected_minutes": _ph_selected,
            "baseline_minutes": _baseline_min,
            # The learned candidate equals the selected value only when a learned
            # value was actually chosen; otherwise a baseline/fallback was used.
            "learned_candidate_minutes": _ph_selected if _learned else None,
            "fallback_reason": _ph_status,
            # A duration was COMPUTED this cycle (> 0) — not necessarily heating now.
            "calculated": bool(_ph_selected or 0),
            # The zone is ACTUALLY in a preheat window right now (coordinator state).
            "active_window": bool(zone.get("preheat_active", False)),
        }

        # ── TPI ──────────────────────────────────────────────────────────────
        _diag = zone.get("tpi_coef_diag") or {}
        tpi = {
            "available": True,
            "coef_source": zone.get("tpi_coef_source"),      # authoritative source of truth
            "coef_int_selected": zone.get("tpi_coef_int"),   # after blend + step-limit
            "coef_ext_selected": zone.get("tpi_coef_ext"),
            "coef_int_candidate": _diag.get("coef_int_candidate"),
            "coef_int_previous": _diag.get("coef_int_previous"),
            "coef_int_default": _diag.get("coef_int_default"),
            "blend_weight": _diag.get("blend_weight"),
            "transition_applied": _diag.get("transition_applied"),
            "transition_reason": _diag.get("transition_reason"),
            "relative_only": _diag.get("relative_only"),
            "duty_percent": zone.get("tpi_duty_cycle"),
            # The coefficients/duty are the SELECTION; the actual setpoint/valve
            # write is reported in the boost block's dispatch fields.
            "is_selection_only": True,
        }

        # ── Boost ────────────────────────────────────────────────────────────
        if ldr is None:
            boost = {"available": False, "reason": "no_decision_record"}
        else:
            _boost_applied = bool(_g("boost_applied"))     # authoritative applied-this-cycle flag
            _block = _g("boost_rejected_reason")
            _targets_total = int(_g("dispatch_targets_total", 0) or 0)
            boost = {
                "available": True,
                "proposed_offset_c": _g("boost_candidate_c"),   # raw pre-guard proposal
                # Derived from existing state: boost_applied means the policy AND
                # lifecycle approved it; "lifecycle_blocked" means the policy
                # approved but the lifecycle rejected it.
                "policy_approved": bool(_boost_applied or _block == "lifecycle_blocked"),
                "lifecycle_approved": _boost_applied,
                "applied_offset_c": _g("boost_applied_c"),       # offset actually applied (None = not applied)
                "requested_setpoint_c": _g("final_setpoint_c"),
                "baseline_setpoint_c": _g("baseline_setpoint_c"),
                "block_reason": _block,
                "dispatch_status": _g("dispatch_status", "not_attempted"),
                "dispatch_targets_total": _targets_total,
                "dispatch_targets_failed": int(_g("dispatch_targets_failed", 0) or 0),
                "service_call_succeeded": bool(_g("dispatch_succeeded", False)),
                "outcome_eligible": bool(_g("outcome_eligible", False)),
                # A NEW service call was issued vs the boosted setpoint already
                # being within write tolerance (no new write this cycle).
                "new_write": _targets_total > 0,
                "within_tolerance_no_write": bool(_boost_applied and _targets_total == 0),
                # Never inferred from a successful service call.
                "heating_effect_confirmed": None,
            }

        return {
            "available": True,
            "preheat": preheat,
            "tpi": tpi,
            "boost": boost,
            "notes": {
                "is_selection_only": "TPI coefficients/duty are the selection; "
                                     "actuator dispatch is in boost.dispatch_* fields",
                "within_tolerance_no_write": "boost offset active on the TRV but no new "
                                             "service call issued this cycle (setpoint within tolerance)",
                "heating_effect_confirmed": "always null — a successful dispatch does not "
                                            "confirm the room physically warmed",
            },
        }
    except Exception:
        return {"available": False}


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

    ``runtime_snapshot`` is intentionally never shown here (neither
    ``exists: true`` nor ``exists: false``): it is not wired into
    ``StorageMetadataStore`` — showing it would wrongly imply a tracking
    attempt that never happens. Every store name that actually appears below is
    exactly whatever the zone's own index currently contains — no fixed/guessed
    store-name list, so newly tracked stores show up without an export.py
    change.

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


async def _learning_storage_context_export(
    hass: HomeAssistant, learning_zone_id: str, *, now: datetime,
) -> dict:
    """Per-zone Storage-Metadata context for diagnostics.

    ``granularity``: ``"store_level"`` and ``timestamp_semantics``:
    ``"store_write_time"`` are small, constant, machine-readable markers
    making explicit that a store's ``updated_at_utc``/``age_minutes`` is a
    *storage write time*, not a content timestamp — episode/research-daily/
    observation payloads elsewhere in diagnostics carry their own ``ts``/
    ``hour``/``minute``/``weekday`` content-time fields, unrelated to this
    block, which is pure additional context and never changes/replaces them.
    """
    result = await _storage_metadata_stores_for_export(hass, learning_zone_id, now=now)
    return {
        "granularity": "store_level",
        "timestamp_semantics": "store_write_time",
        **result,
    }
