"""Support event persistence helpers for bounded critical-event history.

Pure helpers only; runtime producers call higher-level controller methods
(``LearningShadowController.record_support_critical_event_safe()`` in
``ha_integration.py``), never these functions directly. Mirrors
``episode_persistence.py``'s pipeline exactly:

    SupportCriticalEvent dataclass(es)
    -> serialize_support_event()          (support_event_serialization.py)
    -> append into a SupportCriticalEventStore "events" dict, skipping
       unknown/malformed events and duplicate event_ids
    -> evaluate_support_event_retention() applied immediately, using ONE
       central retention policy (unlike episodes, support events do not
       have per-type retention — see the constants below for why).

No HA imports. No runtime imports. No control imports. No storage I/O of any
kind — every function here is a pure transformation of in-memory dicts; the
caller decides if/when to persist the resulting payload through a
SupportCriticalEventStore. Never mutates the ``event`` objects passed in, nor
the input ``payload`` in place (a new dict is always returned).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from ..registry import RetentionPolicy
from ..support_event_schemas import SupportCriticalEvent, SupportEventSeverity
from .support_event_serialization import serialize_support_event

# -- retention constants ------------------------------------------------------
#
# 48h (2 days) rather than a strict 24h window: covers the "gestern Abend hat
# ThermoSmart falsch geheizt, ich schaue heute Morgen nach" scenario with
# margin, without drifting toward a long-term research retention window.
SUPPORT_EVENT_RETENTION_MAX_AGE_DAYS = 2

# Mid-point of the requested 500-1000 bounded-cap range: generous enough for
# a busy multi-zone household within the 48h window without being
# unbounded. Kept as ONE central constant (not scattered across call sites)
# so a later calibration pass has exactly one place to change; covered by
# tests below.
SUPPORT_EVENT_RETENTION_MAX_RECORDS = 750

SUPPORT_EVENT_RETENTION_DEFAULT = RetentionPolicy(
    max_age_days=SUPPORT_EVENT_RETENTION_MAX_AGE_DAYS,
    max_records=SUPPORT_EVENT_RETENTION_MAX_RECORDS,
)

# Eviction priority when the record cap is reached: lower number evicted
# first. INFO-severity events (typically repeated, low-value blockers) make
# room before WARNING, WARNING before CRITICAL — see
# evaluate_support_event_retention().
_SEVERITY_EVICTION_RANK = {
    SupportEventSeverity.INFO.value: 0,
    SupportEventSeverity.WARNING.value: 1,
    SupportEventSeverity.CRITICAL.value: 2,
}


def _parse_ts(value: object) -> Optional[datetime]:
    """Best-effort ISO 8601 parse. None on missing/malformed input — the
    same conservative "unknown -> keep" contract as retention.py's
    ``_parse_iso_utc``."""
    if not isinstance(value, str) or not value:
        return None
    try:
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


@dataclass(frozen=True)
class SupportEventAppendResult:
    """Outcome of an append/prune pass over a SupportCriticalEventStore payload.

    Describes what changed — never mutates the input payload in place.
    """
    updated_payload: dict
    appended_event_ids: tuple[str, ...]
    skipped_count: int          # unknown/malformed events, not appended
    duplicate_count: int        # event_id already present, not re-appended
    pruned_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class SupportEventRetentionDecision:
    """Result of evaluating retention for one zone's event-store payload.

    Describes what would be pruned — this object never mutates storage.
    """
    keep_event_ids: tuple[str, ...]
    delete_event_ids: tuple[str, ...]
    updated_payload: dict


def _events_dict(payload: Any) -> dict:
    if isinstance(payload, Mapping):
        existing = payload.get("events")
        if isinstance(existing, Mapping):
            return dict(existing)
    return {}


def evaluate_support_event_retention(
    events_payload: Mapping[str, object],
    *,
    policy: RetentionPolicy = SUPPORT_EVENT_RETENTION_DEFAULT,
    now_utc: datetime,
) -> SupportEventRetentionDecision:
    """Decide which events may be pruned from one zone's event-store payload.

    Pure function — no storage I/O, no mutation of inputs. Never raises.

    Age (``policy.max_age_days``) is evaluated first, exactly like episode
    retention: an entry with a missing/malformed ``ts`` is always kept — it
    can never be safely aged out. If ``policy.max_records`` still caps the
    survivors, eviction is severity-priority first (INFO before WARNING
    before CRITICAL), then oldest-known-timestamp first within the same
    severity — unlike episodes, a CRITICAL event should outlive many INFO
    events even if it happens to be slightly older, since a support
    conversation about "why didn't it heat" cares more about the one
    critical block than a dozen routine same-setpoint no-ops.
    """
    raw_entries = (
        events_payload.get("events") if isinstance(events_payload, Mapping) else None
    )
    if not isinstance(raw_entries, Mapping):
        return SupportEventRetentionDecision(
            keep_event_ids=(), delete_event_ids=(), updated_payload={"events": {}},
        )

    def _is_age_candidate(eid: str) -> bool:
        if policy.max_age_days is None:
            return False
        entry = raw_entries[eid]
        if not isinstance(entry, Mapping):
            return False  # unknown shape -> conservative keep
        ts = _parse_ts(entry.get("ts"))
        if ts is None:
            return False  # missing/malformed timestamp -> conservative keep
        age_days = (now_utc - ts).total_seconds() / 86400.0
        return age_days > policy.max_age_days

    to_delete: set[str] = {eid for eid in raw_entries if _is_age_candidate(eid)}

    if policy.max_records is not None:
        survivors = [eid for eid in raw_entries if eid not in to_delete]
        if len(survivors) > policy.max_records:
            def _known(eid: str) -> bool:
                entry = raw_entries[eid]
                return (
                    isinstance(entry, Mapping)
                    and entry.get("severity") in _SEVERITY_EVICTION_RANK
                    and _parse_ts(entry.get("ts")) is not None
                )
            known_survivors = [eid for eid in survivors if _known(eid)]
            ordered = sorted(
                known_survivors,
                key=lambda eid: (
                    _SEVERITY_EVICTION_RANK[raw_entries[eid]["severity"]],
                    _parse_ts(raw_entries[eid]["ts"]),
                ),
            )
            overflow = min(len(survivors) - policy.max_records, len(ordered))
            to_delete.update(ordered[:overflow])

    keep_ids = tuple(eid for eid in raw_entries if eid not in to_delete)
    updated_payload = {"events": {eid: raw_entries[eid] for eid in keep_ids}}

    return SupportEventRetentionDecision(
        keep_event_ids=keep_ids,
        delete_event_ids=tuple(sorted(to_delete)),
        updated_payload=updated_payload,
    )


def append_support_event(
    payload: Any,
    event: SupportCriticalEvent,
    *,
    now_utc: datetime,
    policy: RetentionPolicy = SUPPORT_EVENT_RETENTION_DEFAULT,
) -> SupportEventAppendResult:
    """Append ONE SupportCriticalEvent into a SupportCriticalEventStore payload.

    Pure function — no storage I/O, no mutation of the input payload or of
    ``event``. Never raises.
    """
    return append_support_events(
        payload, (event,), now_utc=now_utc, policy=policy,
    )


def append_support_events(
    payload: Any,
    events: Sequence[SupportCriticalEvent],
    *,
    now_utc: datetime,
    policy: RetentionPolicy = SUPPORT_EVENT_RETENTION_DEFAULT,
) -> SupportEventAppendResult:
    """Append a sequence of SupportCriticalEvents; see append_support_event().

    Processes events in order, preserving that order among entries actually
    appended. Each event is serialized independently via
    ``serialize_support_event()`` — a malformed event is skipped
    (``skipped_count`` incremented), never raises. A duplicate ``event_id``
    (already present, or repeated within this same call) is not re-appended
    (``duplicate_count`` incremented). After all appends, retention is
    applied immediately across the whole payload (support events share one
    policy, unlike per-episode-type retention).
    """
    entries = _events_dict(payload)
    appended: list[str] = []
    skipped = 0
    duplicates = 0

    for event in events:
        serialized = serialize_support_event(event)
        if serialized is None:
            skipped += 1
            continue
        eid = serialized.get("event_id")
        if not isinstance(eid, str) or not eid:
            skipped += 1
            continue
        if eid in entries:
            duplicates += 1
            continue
        entries[eid] = serialized
        appended.append(eid)

    try:
        decision = evaluate_support_event_retention(
            {"events": entries}, policy=policy, now_utc=now_utc,
        )
        entries = dict(decision.updated_payload["events"])
        pruned = list(decision.delete_event_ids)
    except Exception:
        pruned = []  # non-fatal: retention failure leaves the appended entries as-is

    return SupportEventAppendResult(
        updated_payload={"events": entries},
        appended_event_ids=tuple(appended),
        skipped_count=skipped,
        duplicate_count=duplicates,
        pruned_event_ids=tuple(pruned),
    )


def prune_support_events(
    payload: Any,
    *,
    now_utc: datetime,
    policy: RetentionPolicy = SUPPORT_EVENT_RETENTION_DEFAULT,
) -> SupportEventAppendResult:
    """Apply retention across ALL entries in a SupportCriticalEventStore
    payload, without appending anything new. A standalone maintenance pass,
    e.g. to re-run retention on an already-loaded payload right before a
    save. Never raises.
    """
    entries = _events_dict(payload)
    decision = evaluate_support_event_retention(
        {"events": entries}, policy=policy, now_utc=now_utc,
    )
    return SupportEventAppendResult(
        updated_payload=decision.updated_payload,
        appended_event_ids=(),
        skipped_count=0,
        duplicate_count=0,
        pruned_event_ids=decision.delete_event_ids,
    )


def is_recent_duplicate(
    event: SupportCriticalEvent,
    recent_entries: Sequence[Mapping[str, Any]],
    *,
    window_s: float,
) -> bool:
    """Best-effort check: does an entry with the same ``event_type`` AND
    ``reason`` already exist within ``window_s`` seconds of this event's
    ``ts``? Pure and side-effect-free — callers MAY use this before
    ``append_support_event()`` to avoid persisting bursts of the same
    repeated blocker (e.g. a ``same_setpoint_block`` firing every cycle).
    ``append_support_event()`` itself never calls this — calling code stays
    in full control of what counts as "the same" burst. Never raises; a
    malformed ``recent_entries`` item is simply skipped.
    """
    try:
        for entry in recent_entries:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("event_type") != event.event_type.value:
                continue
            if entry.get("reason") != event.reason:
                continue
            other_ts = _parse_ts(entry.get("ts"))
            if other_ts is None:
                continue
            if abs((event.ts - other_ts).total_seconds()) <= window_s:
                return True
        return False
    except Exception:
        return False
