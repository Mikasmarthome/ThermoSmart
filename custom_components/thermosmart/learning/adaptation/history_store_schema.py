"""Serialization and pruning for adaptation candidate history storage.

Pure functions only. No HA imports. No Store calls. No runtime binding.
No control modification.

Designed for a future _VersionedStore with key:
    thermosmart_le2__{learning_zone_id}__adaptation_history

The store envelope (store_schema_version) is added by _VersionedStore.
These functions operate on the ``data`` payload only.

Privacy: entries use candidate_key (zone_hash-derived, no entity_id),
discretized buckets, and no personal data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from .contracts import AdaptationDirection, AdaptationLifecycle, CandidateType
from .history import CandidateHistoryEntry, evaluate_promotion_readiness

HISTORY_SCHEMA_VERSION = 1
_COMPONENT_NAME = "adaptation_history"

# ── Pruning defaults ───────────────────────────────────────────────────────

_DEFAULT_MAX_ENTRIES = 50       # per zone; 5 types × 2 directions × ~5 situations
_DEFAULT_MAX_AGE_DAYS = 180.0   # entries unseen for N days become candidates for removal
_DEFAULT_MIN_SEEN_FOR_KEEP = 3  # entries below this count lose age protection


# ── State container ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdaptationHistoryState:
    """Versioned, zone-scoped snapshot of all accumulated candidate history.

    This is the ``data`` payload stored by a future _VersionedStore.
    The store_schema_version envelope is managed by _VersionedStore.

    ``learning_zone_id`` is the opaque LE 2.0 zone id (= entry_id),
    not a zone name, entity_id, or human-readable string.
    The stored value is the authoritative identity anchor for recovery.
    """
    learning_zone_id: str
    updated_at: str                         # ISO 8601 UTC of last write
    entries: dict                           # candidate_key -> CandidateHistoryEntry
    schema_version: int = HISTORY_SCHEMA_VERSION
    component: str = _COMPONENT_NAME


# ── Serialization ──────────────────────────────────────────────────────────

def serialize_history_state(state: AdaptationHistoryState) -> dict:
    """Convert AdaptationHistoryState to a JSON-serializable dict.

    Returns the ``data`` payload; the caller (_VersionedStore.save) wraps it
    in ``{"store_schema_version": N, "data": ...}``.
    """
    return {
        "schema_version": state.schema_version,
        "component": state.component,
        "learning_zone_id": state.learning_zone_id,
        "updated_at": state.updated_at,
        "entries": {
            key: entry.to_dict()
            for key, entry in state.entries.items()
        },
    }


def _parse_entry(raw: Mapping[str, Any]) -> Optional[CandidateHistoryEntry]:
    """Reconstruct a CandidateHistoryEntry from a dict. Returns None on error."""
    try:
        return CandidateHistoryEntry(
            candidate_key=raw["candidate_key"],
            candidate_type=CandidateType(raw["candidate_type"]),
            direction=AdaptationDirection(raw["direction"]),
            first_seen_ts=raw.get("first_seen_ts"),
            last_seen_ts=raw.get("last_seen_ts"),
            seen_count=int(raw["seen_count"]),
            supporting_outcome_count=int(raw["supporting_outcome_count"]),
            avg_outcome_quality=float(raw["avg_outcome_quality"]),
            avg_timeout_rate=(
                float(raw["avg_timeout_rate"])
                if raw.get("avg_timeout_rate") is not None else None
            ),
            avg_overshoot_rate=(
                float(raw["avg_overshoot_rate"])
                if raw.get("avg_overshoot_rate") is not None else None
            ),
            last_lifecycle=AdaptationLifecycle(raw["last_lifecycle"]),
            dominant_reason=raw.get("dominant_reason"),
            dominant_reason_ratio=float(raw.get("dominant_reason_ratio", 0.0)),
        )
    except (KeyError, ValueError, TypeError):
        return None


def deserialize_history_state(
    data: Mapping[str, Any],
    learning_zone_id: str,
) -> AdaptationHistoryState:
    """Reconstruct AdaptationHistoryState from a raw dict payload.

    Defensive: unknown keys and malformed entries are silently skipped so that
    a partially-written or forward-compatible store does not block startup.

    The store_schema_version envelope is already validated by _VersionedStore.load()
    before this function is called — no additional version check needed here.

    Args:
        data: the ``data`` payload from _VersionedStore.load().
        learning_zone_id: the authoritative zone id for this load context;
            the stored value is advisory only (identity anchor for recovery).
    """
    if not isinstance(data, Mapping):
        return AdaptationHistoryState(
            learning_zone_id=learning_zone_id,
            updated_at="",
            entries={},
        )

    raw_entries = data.get("entries") or {}
    entries: dict[str, CandidateHistoryEntry] = {}
    if isinstance(raw_entries, Mapping):
        for key, raw in raw_entries.items():
            if not isinstance(raw, Mapping):
                continue
            entry = _parse_entry(raw)
            if entry is not None:
                entries[str(key)] = entry

    return AdaptationHistoryState(
        learning_zone_id=learning_zone_id,
        updated_at=data.get("updated_at") or "",
        entries=entries,
    )


# ── Pruning ────────────────────────────────────────────────────────────────

def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 UTC string to a timezone-aware datetime. None on error."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


def prune_history_entries(
    entries: dict,
    *,
    now_ts: str,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
    min_seen_count_for_keep: int = _DEFAULT_MIN_SEEN_FOR_KEEP,
) -> dict:
    """Remove stale or low-value entries from a candidate history dict.

    Pure function — always returns a new dict; never modifies the input.
    Caller owns storage and passes the result to serialize_history_state.

    Pruning rules (applied in order):
    1. Drop entries whose last_seen_ts is older than max_age_days AND
       whose seen_count < min_seen_count_for_keep (low-confidence and stale).
    2. If the remaining count still exceeds max_entries, retain only the
       most recently seen entries (newest last_seen_ts first).

    Args:
        entries: current {candidate_key: CandidateHistoryEntry} mapping.
        now_ts: ISO 8601 UTC reference timestamp (typically the current
            OutcomeDiagnostics.last_update_ts, not datetime.now()).
        max_entries: hard cap on retained entries per zone (default 50).
        max_age_days: entries last seen before this horizon are eligible
            for removal when their seen_count is below the floor (default 180 d).
        min_seen_count_for_keep: entries at or above this count survive age
            pruning regardless (default 3, matching _MIN_SEEN_COUNT gate).
    """
    if not entries:
        return {}

    now = _parse_ts(now_ts)
    age_cutoff: Optional[datetime] = None
    if now is not None and max_age_days > 0:
        age_cutoff = now - timedelta(days=max_age_days)

    # Step 1: age-gate — drop stale, low-confidence entries
    retained: dict = {}
    for key, entry in entries.items():
        if age_cutoff is not None:
            last_ts = _parse_ts(entry.last_seen_ts)
            if last_ts is not None and last_ts < age_cutoff:
                if entry.seen_count < min_seen_count_for_keep:
                    continue  # stale and not yet meaningful — drop
        retained[key] = entry

    # Step 2: hard cap — keep the most recently observed entries
    if len(retained) > max_entries:
        _epoch = datetime.min.replace(tzinfo=timezone.utc)

        def _sort_key(kv: tuple) -> datetime:
            ts = _parse_ts(kv[1].last_seen_ts)
            return ts if ts is not None else _epoch

        sorted_pairs = sorted(retained.items(), key=_sort_key, reverse=True)
        retained = dict(sorted_pairs[:max_entries])

    return retained


# ── Research export helper ─────────────────────────────────────────────────

def adaptation_history_entry_for_research_export(
    entry: CandidateHistoryEntry,
    *,
    span_days: float,
    confounder_ratio: float,
) -> dict:
    """Convert a CandidateHistoryEntry to a public-safe research export dict.

    Evaluates promotion readiness and includes the result. Excludes raw
    timestamps (first_seen_ts / last_seen_ts) and per-rate averages
    (available separately from the outcome model export).

    Pure function — no HA imports, no control modification, no side effects.
    """
    pgr = evaluate_promotion_readiness(
        entry, span_days=span_days, confounder_ratio=confounder_ratio,
    )
    return {
        "candidate_key":            entry.candidate_key,
        "candidate_type":           entry.candidate_type.value,
        "direction":                entry.direction.value,
        "seen_count":               entry.seen_count,
        "supporting_outcome_count": entry.supporting_outcome_count,
        "avg_outcome_quality":      round(entry.avg_outcome_quality, 3),
        "dominant_reason":          entry.dominant_reason,
        "dominant_reason_ratio":    round(entry.dominant_reason_ratio, 3),
        "last_lifecycle":           entry.last_lifecycle.value,
        "promotion_readiness":      pgr.readiness.value,
        "blocking_reasons":         list(pgr.blocking_reasons),
    }
