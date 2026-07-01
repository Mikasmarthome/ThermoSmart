"""Typed schema for LE2 Support Critical Events (pure Python, no Home Assistant).

Support Critical Events are NOT a log of every coordinator cycle. They are the
small subset of "why did/didn't ThermoSmart act here" moments a support
conversation actually needs — e.g. a blocked TRV command, a boost start/stop,
a window-open hold, a sensor going unavailable. Mirrors the existing
``episode_schemas.py`` pattern: a typed Enum for the closed set of kinds, a
frozen dataclass for one materialised event, no builder/instrumentation logic
here (that is a separate, later, per-call-site step — see
``support_event_persistence.py``'s module docstring for the exact reasoning).

Deliberately excluded from this schema (see the accompanying report for the
full reasoning):
  - ``zone_ref`` / any zone identity: the store this event belongs to is
    already zone-scoped by its own store key (one store per zone, exactly
    like ``EpisodesStore``) — duplicating zone identity inside every event
    would be a second place that could leak it. A future, separate export
    step may add an already-pseudonymized zone reference at export time
    (reusing export.py's own hashing), never a raw zone id stored here.
  - Entity ids, HA object reprs, secrets, paths: never accepted as fields;
    ``details`` is a small, JSON-safe mapping of numeric/string diagnostic
    values only (bounded and stripped further at serialization time).

No HA imports. No control imports. No storage I/O. Never mutates its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional

from .contracts import require_nonempty, require_utc


class SupportEventType(Enum):
    """The closed set of support-relevant event kinds.

    Not every kind needs a live producer yet — this schema step only needs
    to accept all of them cleanly. New kinds should be added here first, so
    ``deserialize_support_event()`` can keep rejecting truly unknown values
    rather than silently accepting typos.
    """
    HEATING_DECISION = "heating_decision"
    TRV_COMMAND_SENT = "trv_command_sent"
    TRV_COMMAND_BLOCKED = "trv_command_blocked"
    BOOST_STARTED = "boost_started"
    BOOST_BLOCKED = "boost_blocked"
    BOOST_ENDED = "boost_ended"
    WINDOW_OPEN_HOLD = "window_open_hold"
    WINDOW_CLOSED_RELEASE = "window_closed_release"
    SUMMER_MODE_HOLD = "summer_mode_hold"
    MANUAL_OVERRIDE_START = "manual_override_start"
    MANUAL_OVERRIDE_END = "manual_override_end"
    SCHEDULE_CHANGE = "schedule_change"
    PRESENCE_HOLD = "presence_hold"
    SENSOR_UNAVAILABLE = "sensor_unavailable"
    SENSOR_RESTORED = "sensor_restored"
    TEMPERATURE_INVALID = "temperature_invalid"
    TRV_UNAVAILABLE = "trv_unavailable"
    MIN_INTERVAL_BLOCK = "min_interval_block"
    SAME_SETPOINT_BLOCK = "same_setpoint_block"
    LEARNING_RECOMMENDATION_ONLY = "learning_recommendation_only"
    LEARNING_ADAPTIVE_APPLIED = "learning_adaptive_applied"
    FALLBACK_USED = "fallback_used"
    RESTART_RESTORE = "restart_restore"
    STORAGE_RESTORE = "storage_restore"
    STORAGE_RESTORE_FAILED = "storage_restore_failed"
    STORAGE_SAVE_FAILED = "storage_save_failed"
    STORAGE_SAVE_RECOVERED = "storage_save_recovered"
    OUTCOME_RESOLVED = "outcome_resolved"


class SupportEventSeverity(Enum):
    """Eviction priority when the store's record cap is reached: INFO is
    evicted first, then WARNING, CRITICAL last (see
    ``support_event_persistence.py``'s ``evaluate_support_event_retention()``).
    """
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Event kinds that are routinely frequent and low-value on their own (repeated
# blockers, not genuine incidents) — a natural default severity of INFO keeps
# them first in line for eviction once the cap is reached, without needing a
# separate "is this noise" concept. Producers may still choose a higher
# severity explicitly (e.g. a MIN_INTERVAL_BLOCK the user is actively asking
# about), this is only the suggested default.
NOISE_PRONE_EVENT_TYPES = frozenset({
    SupportEventType.MIN_INTERVAL_BLOCK,
    SupportEventType.SAME_SETPOINT_BLOCK,
    SupportEventType.LEARNING_RECOMMENDATION_ONLY,
})

# Bounds enforced at serialization time (see support_event_serialization.py) —
# kept here so the schema module documents its own limits in one place.
SUMMARY_MAX_LENGTH = 200
DETAILS_MAX_KEYS = 12
DETAILS_VALUE_MAX_LENGTH = 200


@dataclass(frozen=True)
class SupportCriticalEvent:
    """One materialised, support-relevant event.

    ``details`` must be a small, flat, JSON-safe mapping — only numbers,
    booleans, strings and None. No entity ids, no HA objects, no nested
    dataclasses/objects. Enforced defensively (never raises) at
    serialization time, not here — this constructor only enforces the
    structural invariants (non-empty id, real enum members, UTC timestamp)
    the same way episode dataclasses do, so a genuinely malformed
    construction fails loudly at the call site rather than silently
    producing a bad event.
    """
    schema_version: int
    event_id: str
    event_type: SupportEventType
    ts: datetime
    severity: SupportEventSeverity
    reason: Optional[str] = None
    summary: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.event_id, "event_id")
        if not isinstance(self.event_type, SupportEventType):
            raise TypeError(f"event_type must be a SupportEventType, got {type(self.event_type)!r}")
        if not isinstance(self.severity, SupportEventSeverity):
            raise TypeError(f"severity must be a SupportEventSeverity, got {type(self.severity)!r}")
        require_utc(self.ts, "ts")
        if not isinstance(self.details, Mapping):
            raise TypeError(f"details must be a mapping, got {type(self.details)!r}")
