"""Versioned, defensive serialization for LE2 Support Critical Events.

Mirrors ``episode_serialization.py``'s pattern: a flat, JSON-safe dict shape,
a schema-version guard, and a separate public-safe export view. Reuses the
same ``encode_utc``/``decode_utc`` primitives so timestamps round-trip
identically to episodes and raw records.

Storage-layer shape (``serialize_support_event``/``deserialize_support_event``):
    A FLAT dict — ``schema_version``, ``event_id``, ``event_type``, ``ts``,
    ``severity``, ``reason``, ``summary``, ``details``. No nested "data"
    wrapper, matching the ``{"events": {event_id: <this dict>}}`` container
    convention used by ``support_event_persistence.py``.

``details`` is defensively bounded here (``DETAILS_MAX_KEYS`` keys,
``DETAILS_VALUE_MAX_LENGTH`` chars per string value, non-JSON-safe values
dropped) — this is the one field a future producer might otherwise use as an
escape hatch for something too large or not public-safe. Everything else in
this schema is already public-safe by construction (see
``support_event_schemas.py``'s module docstring for why ``zone_ref`` is
deliberately absent).

No HA imports. No runtime imports. No storage I/O of any kind. Never raises:
malformed/unknown input is handled defensively (skip or return None).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ..support_event_schemas import (
    DETAILS_MAX_KEYS,
    DETAILS_VALUE_MAX_LENGTH,
    SUMMARY_MAX_LENGTH,
    SupportCriticalEvent,
    SupportEventSeverity,
    SupportEventType,
)
from .serialization import decode_utc, encode_utc

SUPPORT_EVENT_SCHEMA_VERSION = 1

_KNOWN_EVENT_TYPE_VALUES = {t.value for t in SupportEventType}
_KNOWN_SEVERITY_VALUES = {s.value for s in SupportEventSeverity}


def _bounded_details(details: Any) -> dict:
    """Defensively bound an arbitrary ``details`` mapping to small, JSON-safe
    scalars only. Never raises; unrecognised/oversized/non-scalar values are
    dropped rather than causing the whole event to fail to serialize.
    """
    if not isinstance(details, Mapping):
        return {}
    bounded: dict = {}
    for key, value in details.items():
        if len(bounded) >= DETAILS_MAX_KEYS:
            break
        if not isinstance(key, str) or not key:
            continue
        if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
            bounded[key] = value
        elif isinstance(value, str):
            bounded[key] = value[:DETAILS_VALUE_MAX_LENGTH]
        # else: lists/dicts/objects/HA reprs never survive into storage.
    return bounded


def serialize_support_event(event: SupportCriticalEvent) -> Optional[dict]:
    """Serialize one SupportCriticalEvent to a flat, versioned, JSON-safe dict.

    Returns None on any encoding failure (e.g. a malformed ``ts``) — never
    raises. Never mutates ``event``.
    """
    if not isinstance(event, SupportCriticalEvent):
        return None
    try:
        summary = event.summary
        if isinstance(summary, str):
            summary = summary[:SUMMARY_MAX_LENGTH]
        elif summary is not None:
            summary = None
        return {
            "schema_version": SUPPORT_EVENT_SCHEMA_VERSION,
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "ts": encode_utc(event.ts),
            "severity": event.severity.value,
            "reason": event.reason if isinstance(event.reason, str) else None,
            "summary": summary,
            "details": _bounded_details(event.details),
        }
    except Exception:
        return None


def deserialize_support_event(raw: Mapping[str, Any]) -> Optional[SupportCriticalEvent]:
    """Reconstruct one SupportCriticalEvent from a serialize_support_event()
    payload. Returns None on schema-version mismatch, unknown event_type/
    severity, malformed shape, or a missing required field — never raises.
    """
    if not isinstance(raw, Mapping):
        return None
    if raw.get("schema_version") != SUPPORT_EVENT_SCHEMA_VERSION:
        return None
    if raw.get("event_type") not in _KNOWN_EVENT_TYPE_VALUES:
        return None
    if raw.get("severity") not in _KNOWN_SEVERITY_VALUES:
        return None
    try:
        return SupportCriticalEvent(
            schema_version=SUPPORT_EVENT_SCHEMA_VERSION,
            event_id=raw["event_id"],
            event_type=SupportEventType(raw["event_type"]),
            ts=decode_utc(raw["ts"]),
            severity=SupportEventSeverity(raw["severity"]),
            reason=raw.get("reason"),
            summary=raw.get("summary"),
            details=_bounded_details(raw.get("details")),
        )
    except Exception:
        return None


def support_event_for_export(entry: Mapping[str, Any]) -> Optional[dict]:
    """Return a public-safe export view of one already-serialized event entry.

    Deliberately drops ``event_id`` (see support_event_schemas.py's guidance:
    "not later exported unreflectively") and re-applies the same details
    bounds. Operates on the flat serialized dict shape (not a dataclass) —
    this is the shape a future export step would actually read from a
    loaded/snapshotted store. Returns None for malformed/unrecognised input;
    never raises. NOT wired into export.py in this step (foundation only).
    """
    if not isinstance(entry, Mapping):
        return None
    if entry.get("event_type") not in _KNOWN_EVENT_TYPE_VALUES:
        return None
    if entry.get("severity") not in _KNOWN_SEVERITY_VALUES:
        return None
    try:
        summary = entry.get("summary")
        if isinstance(summary, str):
            summary = summary[:SUMMARY_MAX_LENGTH]
        elif summary is not None:
            summary = None
        return {
            "event_type": entry["event_type"],
            "ts": entry.get("ts"),
            "severity": entry["severity"],
            "reason": entry.get("reason") if isinstance(entry.get("reason"), str) else None,
            "summary": summary,
            "details": _bounded_details(entry.get("details")),
        }
    except Exception:
        return None
