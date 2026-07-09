"""Pure retention-enforcement decisions for Learning raw segments and episodes.

Operates only on already-loaded data structures — a :class:`SegmentIndex` plus
a mapping of sealed segment metadata dicts, or an ``EpisodesStore`` payload
dict. Performs no storage I/O itself; see ``retention_service.py`` for the
small HA-store-facing layer that loads/saves via the store layer and calls
these pure functions.

Conservative by design, matching the existing pruning style used for
adaptation history (``history_store_schema.py``) and application lifecycle
state (``application_state.py``):

  - A segment/episode with a missing or malformed timestamp is always kept —
    it can never be safely aged out.
  - The active (unsealed) raw segment is never a deletion candidate, even if
    it is accidentally present in the supplied metadata mapping.
  - When ``RetentionPolicy.max_records`` triggers a hard cap, only entries
    with a known, parseable timestamp/sequence are eligible for eviction —
    unknown entries are excluded from eviction, not assumed oldest.
  - Nothing is pruned when a policy has neither ``max_age_days`` nor
    ``max_records`` set.

No HA imports. No control modification. All functions are pure and never
raise; malformed input is handled conservatively (keep, don't crash).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Mapping, Optional

from ..registry import RetentionPolicy
from .segments import SegmentIndex


def _parse_iso_utc(value: object) -> Optional[datetime]:
    """Best-effort ISO 8601 parse. None on missing/malformed input."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


# ── Raw segment retention ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentRetentionDecision:
    """Result of evaluating raw segment retention for one (zone, track).

    Describes what would be pruned — this object never mutates storage.
    """
    keep_segment_ids: tuple[str, ...]
    delete_segment_ids: tuple[str, ...]
    updated_index: SegmentIndex


def evaluate_segment_retention(
    index: SegmentIndex,
    sealed_segments: Mapping[str, Mapping[str, object]],
    *,
    policy: RetentionPolicy,
    now_utc: datetime,
) -> SegmentRetentionDecision:
    """Decide which SEALED segments may be deleted for one (zone, track).

    Pure function — no storage I/O, no mutation of inputs. Never raises.

    Args:
        index: the loaded SegmentIndex for this (zone, track).
        sealed_segments: segment_id -> raw segment metadata dict, as found
            under the ``"segment"`` key of a payload returned by
            ``RawSegmentStore.load_segment()`` (keys of interest: "sealed_utc"
            as an ISO 8601 UTC string or None, "sequence_number" as int).
            Only ids that also appear in ``index.sealed_segment_ids`` are ever
            considered; the active segment is always excluded even if present.
        policy: RetentionPolicy. ``max_age_days`` gates eligibility by
            ``sealed_utc`` age. ``max_records`` (if set) caps the number of
            sealed segments retained, evicting the lowest ``sequence_number``
            first among segments with known metadata.
        now_utc: reference clock for age comparisons.

    Returns:
        SegmentRetentionDecision. Never raises.
    """
    eligible = [
        sid for sid in index.sealed_segment_ids
        if sid != index.active_segment_id
    ]

    def _is_age_candidate(sid: str) -> bool:
        if policy.max_age_days is None:
            return False
        payload = sealed_segments.get(sid)
        if not isinstance(payload, Mapping):
            return False  # unknown metadata -> conservative keep
        sealed_at = _parse_iso_utc(payload.get("sealed_utc"))
        if sealed_at is None:
            return False  # missing/malformed timestamp -> conservative keep
        age_days = (now_utc - sealed_at).total_seconds() / 86400.0
        return age_days > policy.max_age_days

    to_delete: set[str] = {sid for sid in eligible if _is_age_candidate(sid)}

    if policy.max_records is not None:
        survivors = [sid for sid in eligible if sid not in to_delete]
        if len(survivors) > policy.max_records:
            known_survivors = [
                sid for sid in survivors
                if isinstance(sealed_segments.get(sid), Mapping)
                and isinstance(sealed_segments[sid].get("sequence_number"), int)
            ]
            ordered = sorted(
                known_survivors,
                key=lambda sid: sealed_segments[sid]["sequence_number"],
            )
            overflow = min(len(survivors) - policy.max_records, len(ordered))
            to_delete.update(ordered[:overflow])

    keep = tuple(sid for sid in index.sealed_segment_ids if sid not in to_delete)
    updated_index = replace(index, sealed_segment_ids=keep)

    return SegmentRetentionDecision(
        keep_segment_ids=keep,
        delete_segment_ids=tuple(sorted(to_delete)),
        updated_index=updated_index,
    )


# ── Episode retention ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeRetentionDecision:
    """Result of evaluating episode retention for one zone's EpisodesStore payload.

    Describes what would be pruned — this object never mutates storage.
    """
    keep_episode_ids: tuple[str, ...]
    delete_episode_ids: tuple[str, ...]
    updated_payload: dict


def _episode_timestamp(entry: object) -> Optional[datetime]:
    """Read an episode's age-reference timestamp: end_ts, falling back to start_ts.

    None when the entry is malformed or carries no parseable timestamp —
    callers must treat that as "conservatively keep".
    """
    if not isinstance(entry, Mapping):
        return None
    return _parse_iso_utc(entry.get("end_ts")) or _parse_iso_utc(entry.get("start_ts"))


def evaluate_episode_retention(
    episodes_payload: Mapping[str, object],
    *,
    policy: RetentionPolicy,
    now_utc: datetime,
) -> EpisodeRetentionDecision:
    """Decide which episodes may be pruned from one zone's EpisodesStore payload.

    Pure function — no storage I/O, no mutation of inputs. Never raises.

    Args:
        episodes_payload: the loaded EpisodesStore "data" payload, expected
            shape ``{"episodes": {episode_id: {..., "end_ts": <iso str>, ...}}}``.
            Each entry is judged solely on its own timestamp — episode_type is
            never inspected, so different episode types (heating / afterheat /
            passive_cooling / window_cooling / outcome) never influence one
            another's retention decision.
        policy: RetentionPolicy. ``max_age_days`` gates eligibility by the
            episode's end_ts (or start_ts fallback). ``max_records`` (if set)
            caps the total episode count, evicting the oldest known timestamp
            first.
        now_utc: reference clock for age comparisons.

    Returns:
        EpisodeRetentionDecision. Never raises.
    """
    raw_entries = (
        episodes_payload.get("episodes")
        if isinstance(episodes_payload, Mapping) else None
    )
    if not isinstance(raw_entries, Mapping):
        return EpisodeRetentionDecision(
            keep_episode_ids=(), delete_episode_ids=(),
            updated_payload={"episodes": {}},
        )

    def _is_age_candidate(eid: str) -> bool:
        if policy.max_age_days is None:
            return False
        ts = _episode_timestamp(raw_entries[eid])
        if ts is None:
            return False  # missing/malformed timestamp -> conservative keep
        age_days = (now_utc - ts).total_seconds() / 86400.0
        return age_days > policy.max_age_days

    to_delete: set[str] = {eid for eid in raw_entries if _is_age_candidate(eid)}

    if policy.max_records is not None:
        survivors = [eid for eid in raw_entries if eid not in to_delete]
        if len(survivors) > policy.max_records:
            known_survivors = [
                eid for eid in survivors if _episode_timestamp(raw_entries[eid]) is not None
            ]
            ordered = sorted(
                known_survivors,
                key=lambda eid: _episode_timestamp(raw_entries[eid]),
            )
            overflow = min(len(survivors) - policy.max_records, len(ordered))
            to_delete.update(ordered[:overflow])

    keep_ids = tuple(eid for eid in raw_entries if eid not in to_delete)
    updated_payload = {"episodes": {eid: raw_entries[eid] for eid in keep_ids}}

    return EpisodeRetentionDecision(
        keep_episode_ids=keep_ids,
        delete_episode_ids=tuple(sorted(to_delete)),
        updated_payload=updated_payload,
    )
