"""Pure application lifecycle state schema, serialization, and pruning.

Defines the persistable lifecycle state for adaptation applications:

    AppliedAdaptationRecord + MonitoringEvaluation
    → AppliedAdaptationStateEntry  (serializable, versioned)
    → ApplicationLifecycleState    (zone-scoped container)
    → serialize / deserialize / prune / export / summarize

No control modification. No HA imports. No runtime state.
All dataclasses are frozen; all functions are pure and never raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from .monitoring import (
    AppliedAdaptationRecord,
    MonitoringEvaluation,
    MonitoringStatus,
)

APPLICATION_STATE_SCHEMA_VERSION = 1
_COMPONENT_NAME = "application_lifecycle"

_DEFAULT_MAX_ENTRIES = 50
_DEFAULT_MAX_AGE_DAYS = 180.0

# Active statuses — entries that still require monitoring attention
_ACTIVE_STATUSES: frozenset[str] = frozenset({
    MonitoringStatus.PENDING.value,
    MonitoringStatus.IN_PROGRESS.value,
    MonitoringStatus.ADOPTION_CANDIDATE.value,
    MonitoringStatus.ROLLBACK_RECOMMENDED.value,
})

# Terminal statuses — entries that have reached a final (or window-expired) state
_COMPLETED_STATUSES: frozenset[str] = frozenset({
    MonitoringStatus.ADOPTED.value,
    MonitoringStatus.ROLLED_BACK.value,
    MonitoringStatus.WINDOW_EXPIRED.value,
})


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 string to a timezone-aware datetime. None on error."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


# ── State Entry ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AppliedAdaptationStateEntry:
    """Serializable, versioned lifecycle state for one applied adaptation.

    Built from an AppliedAdaptationRecord via record_to_state_entry().
    Updated by update_state_entry_from_evaluation() after each monitoring cycle.

    No control modification is performed by this object.
    adopted_ts and rolled_back_ts are set only by a future Application Layer
    that performs the actual state transition — never by this foundation.
    """
    candidate_key: str
    candidate_type_value: str       # CandidateType.value
    direction_value: str            # AdaptationDirection.value
    status: MonitoringStatus
    applied_delta_min: float
    previous_cumulative_delta_min: float
    new_cumulative_delta_min: float
    applied_ts: str                             # ISO 8601 UTC of (theoretical) application
    monitoring_deadline_ts: Optional[str]
    last_evaluated_ts: Optional[str]
    monitored_outcome_count: int
    rollback_supported: bool
    rollback_recommended: bool
    rollback_reasons: tuple[str, ...]
    adoption_ready: bool
    adopted_ts: Optional[str]       # set by Application Layer when formally adopted
    rolled_back_ts: Optional[str]   # set by Application Layer when formally rolled back
    cooldown_until_ts: Optional[str]  # applied_ts + cooldown_days; no new application before
    last_error: Optional[str]

    def to_dict(self) -> dict:
        """Serialize all fields to a JSON-safe dict. No control fields."""
        return {
            "candidate_key":                 self.candidate_key,
            "candidate_type":                self.candidate_type_value,
            "direction":                     self.direction_value,
            "status":                        self.status.value,
            "applied_delta_min":             self.applied_delta_min,
            "previous_cumulative_delta_min": self.previous_cumulative_delta_min,
            "new_cumulative_delta_min":      self.new_cumulative_delta_min,
            "applied_ts":                    self.applied_ts,
            "monitoring_deadline_ts":        self.monitoring_deadline_ts,
            "last_evaluated_ts":             self.last_evaluated_ts,
            "monitored_outcome_count":       self.monitored_outcome_count,
            "rollback_supported":            self.rollback_supported,
            "rollback_recommended":          self.rollback_recommended,
            "rollback_reasons":              list(self.rollback_reasons),
            "adoption_ready":                self.adoption_ready,
            "adopted_ts":                    self.adopted_ts,
            "rolled_back_ts":                self.rolled_back_ts,
            "cooldown_until_ts":             self.cooldown_until_ts,
            "last_error":                    self.last_error,
        }


# ── Zone container ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApplicationLifecycleState:
    """Versioned, zone-scoped container for all adaptation lifecycle entries.

    The store_schema_version envelope is managed by _VersionedStore when later
    wired to persistent storage. These functions operate on the data payload only.
    """
    learning_zone_id: str
    updated_at: str                  # ISO 8601 UTC of last write
    entries: dict                    # candidate_key -> AppliedAdaptationStateEntry
    schema_version: int = APPLICATION_STATE_SCHEMA_VERSION
    component: str = _COMPONENT_NAME


# ── Summary ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApplicationLifecycleSummary:
    """Aggregate counts across all lifecycle entries for one zone."""
    total: int
    active: int              # PENDING + IN_PROGRESS + ADOPTION_CANDIDATE + ROLLBACK_RECOMMENDED
    rollback_recommended: int
    adoption_ready: int      # adoption_ready flag True (may differ from ADOPTION_CANDIDATE count)
    adopted: int
    rolled_back: int
    expired: int

    def to_dict(self) -> dict:
        return {
            "total":                self.total,
            "active":               self.active,
            "rollback_recommended": self.rollback_recommended,
            "adoption_ready":       self.adoption_ready,
            "adopted":              self.adopted,
            "rolled_back":          self.rolled_back,
            "expired":              self.expired,
        }


# ── Conversion helpers ────────────────────────────────────────────────────────

def record_to_state_entry(
    record: AppliedAdaptationRecord,
    *,
    cooldown_days: float = 7.0,
    now_ts: Optional[str] = None,
) -> AppliedAdaptationStateEntry:
    """Convert an AppliedAdaptationRecord to a fresh PENDING state entry.

    Pure function — no HA imports, no control modification, no side effects.
    cooldown_until_ts is derived as applied_ts + cooldown_days. Entry starts
    with monitored_outcome_count=0 and no diagnostic flags set.
    """
    cooldown_until_ts: Optional[str] = None
    try:
        t0 = datetime.fromisoformat(record.applied_ts.replace("Z", "+00:00"))
        cooldown_until_ts = (t0 + timedelta(days=cooldown_days)).isoformat()
    except Exception:
        pass

    return AppliedAdaptationStateEntry(
        candidate_key=record.candidate_key,
        candidate_type_value=record.candidate_type_value,
        direction_value=record.direction_value,
        status=record.status,
        applied_delta_min=record.applied_delta_min,
        previous_cumulative_delta_min=record.previous_cumulative_delta_min,
        new_cumulative_delta_min=record.new_cumulative_delta_min,
        applied_ts=record.applied_ts,
        monitoring_deadline_ts=record.monitoring_deadline_ts,
        last_evaluated_ts=now_ts,
        monitored_outcome_count=0,
        rollback_supported=record.rollback_supported,
        rollback_recommended=False,
        rollback_reasons=(),
        adoption_ready=False,
        adopted_ts=None,
        rolled_back_ts=None,
        cooldown_until_ts=cooldown_until_ts,
        last_error=None,
    )


def update_state_entry_from_evaluation(
    entry: AppliedAdaptationStateEntry,
    evaluation: MonitoringEvaluation,
    *,
    now_ts: Optional[str] = None,
) -> AppliedAdaptationStateEntry:
    """Return a new entry updated with data from a MonitoringEvaluation.

    Pure function — returns a new frozen entry; never modifies the input.
    No real adoption or rollback action is performed. adopted_ts and
    rolled_back_ts are preserved from the input entry (set only by a future
    Application Layer that performs the actual state transition).

    Rollback recommended / adoption_ready are diagnostic flags only.
    """
    return AppliedAdaptationStateEntry(
        candidate_key=entry.candidate_key,
        candidate_type_value=entry.candidate_type_value,
        direction_value=entry.direction_value,
        status=evaluation.status,
        applied_delta_min=entry.applied_delta_min,
        previous_cumulative_delta_min=entry.previous_cumulative_delta_min,
        new_cumulative_delta_min=entry.new_cumulative_delta_min,
        applied_ts=entry.applied_ts,
        monitoring_deadline_ts=entry.monitoring_deadline_ts,
        last_evaluated_ts=now_ts,
        monitored_outcome_count=evaluation.monitored_outcome_count,
        rollback_supported=entry.rollback_supported,
        rollback_recommended=evaluation.rollback_recommended,
        rollback_reasons=evaluation.rollback_reasons,
        adoption_ready=evaluation.adoption_ready,
        adopted_ts=entry.adopted_ts,          # Application Layer sets this
        rolled_back_ts=entry.rolled_back_ts,  # Application Layer sets this
        cooldown_until_ts=entry.cooldown_until_ts,
        last_error=None,
    )


# ── Cooldown helper ───────────────────────────────────────────────────────────

def is_cooldown_active(
    entry: AppliedAdaptationStateEntry,
    *,
    now_ts: Optional[str],
) -> bool:
    """Return True if the cooldown period has not yet elapsed.

    Returns False conservatively when either timestamp is missing or unparseable.
    """
    if not entry.cooldown_until_ts or not now_ts:
        return False
    t_until = _parse_ts(entry.cooldown_until_ts)
    t_now = _parse_ts(now_ts)
    if t_until is None or t_now is None:
        return False
    return t_now < t_until


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_application_lifecycle_state(state: ApplicationLifecycleState) -> dict:
    """Serialize ApplicationLifecycleState to a JSON-safe data payload.

    Returns the ``data`` dict; a future _VersionedStore.save() wraps it
    in ``{"store_schema_version": N, "data": ...}``.
    """
    return {
        "schema_version":   state.schema_version,
        "component":        state.component,
        "learning_zone_id": state.learning_zone_id,
        "updated_at":       state.updated_at,
        "entries": {
            key: entry.to_dict()
            for key, entry in state.entries.items()
        },
    }


def _parse_state_entry(raw: Mapping[str, Any]) -> Optional[AppliedAdaptationStateEntry]:
    """Reconstruct an AppliedAdaptationStateEntry from a raw dict. None on error."""
    try:
        return AppliedAdaptationStateEntry(
            candidate_key=str(raw["candidate_key"]),
            candidate_type_value=str(raw.get("candidate_type", "")),
            direction_value=str(raw.get("direction", "")),
            status=MonitoringStatus(str(raw["status"])),
            applied_delta_min=float(raw["applied_delta_min"]),
            previous_cumulative_delta_min=float(
                raw.get("previous_cumulative_delta_min", 0.0)),
            new_cumulative_delta_min=float(
                raw.get("new_cumulative_delta_min", 0.0)),
            applied_ts=str(raw.get("applied_ts") or ""),
            monitoring_deadline_ts=raw.get("monitoring_deadline_ts") or None,
            last_evaluated_ts=raw.get("last_evaluated_ts") or None,
            monitored_outcome_count=int(raw.get("monitored_outcome_count", 0)),
            rollback_supported=bool(raw.get("rollback_supported", True)),
            rollback_recommended=bool(raw.get("rollback_recommended", False)),
            rollback_reasons=tuple(
                str(r) for r in (raw.get("rollback_reasons") or [])
                if isinstance(r, str)
            ),
            adoption_ready=bool(raw.get("adoption_ready", False)),
            adopted_ts=raw.get("adopted_ts") or None,
            rolled_back_ts=raw.get("rolled_back_ts") or None,
            cooldown_until_ts=raw.get("cooldown_until_ts") or None,
            last_error=raw.get("last_error") or None,
        )
    except (KeyError, ValueError, TypeError):
        return None


def deserialize_application_lifecycle_state(
    data: Mapping[str, Any],
    learning_zone_id: str,
) -> ApplicationLifecycleState:
    """Reconstruct ApplicationLifecycleState from a raw dict payload.

    Defensive: returns an empty safe state on any structural failure.
    Unknown/malformed individual entries are silently skipped.
    Schema version mismatch → empty state (safe forward-compatibility).

    Args:
        data:             the ``data`` payload from _VersionedStore.load().
        learning_zone_id: authoritative zone id for this load context.
    """
    _empty = ApplicationLifecycleState(
        learning_zone_id=learning_zone_id,
        updated_at="",
        entries={},
    )

    if not isinstance(data, Mapping):
        return _empty

    stored_version = data.get("schema_version")
    if stored_version != APPLICATION_STATE_SCHEMA_VERSION:
        return _empty

    raw_entries = data.get("entries") or {}
    entries: dict[str, AppliedAdaptationStateEntry] = {}
    if isinstance(raw_entries, Mapping):
        for key, raw in raw_entries.items():
            if not isinstance(raw, Mapping):
                continue
            entry = _parse_state_entry(raw)
            if entry is not None:
                entries[str(key)] = entry

    return ApplicationLifecycleState(
        learning_zone_id=learning_zone_id,
        updated_at=data.get("updated_at") or "",
        entries=entries,
    )


# ── Pruning ───────────────────────────────────────────────────────────────────

def prune_application_state_entries(
    entries: dict,
    *,
    now_ts: str,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> dict:
    """Remove stale completed entries and enforce the hard entry cap.

    Pure function — always returns a new dict; never modifies the input.

    Pruning rules (applied in order):
    1. Drop completed entries (ADOPTED / ROLLED_BACK / WINDOW_EXPIRED) whose
       applied_ts predates max_age_days. Active entries always survive rule 1.
    2. If count still exceeds max_entries: keep all active entries first, then
       fill remaining slots with the most recently applied completed entries.

    Args:
        entries:      current {candidate_key: AppliedAdaptationStateEntry} mapping.
        now_ts:       ISO 8601 UTC reference timestamp.
        max_entries:  hard per-zone cap (default 50).
        max_age_days: completed entries older than this are eligible for removal.
    """
    if not entries:
        return {}

    now = _parse_ts(now_ts)
    age_cutoff: Optional[datetime] = None
    if now is not None and max_age_days > 0:
        age_cutoff = now - timedelta(days=max_age_days)

    # Step 1: age-gate — drop old completed entries
    retained: dict = {}
    for key, entry in entries.items():
        if entry.status.value in _COMPLETED_STATUSES and age_cutoff is not None:
            applied = _parse_ts(entry.applied_ts)
            if applied is not None and applied < age_cutoff:
                continue
        retained[key] = entry

    # Step 2: hard cap — active entries first, then newest completed
    if len(retained) > max_entries:
        active = {
            k: v for k, v in retained.items()
            if v.status.value in _ACTIVE_STATUSES
        }
        completed = {
            k: v for k, v in retained.items()
            if v.status.value not in _ACTIVE_STATUSES
        }
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        sorted_completed = sorted(
            completed.items(),
            key=lambda kv: (_parse_ts(kv[1].applied_ts) or _epoch),
            reverse=True,
        )
        slots = max(0, max_entries - len(active))
        retained = dict(list(active.items()) + sorted_completed[:slots])

    return retained


# ── Research export ───────────────────────────────────────────────────────────

def application_state_entry_for_research_export(
    entry: AppliedAdaptationStateEntry,
    *,
    now_ts: Optional[str] = None,
) -> dict:
    """Return a public-safe research export dict for one state entry.

    Excludes: entity_id, zone_name, person data, local paths, secrets,
    raw timestamps (conservative — no comparable timestamp export exists in this layer).
    cooldown_active is a boolean relative indicator instead of cooldown_until_ts.
    """
    return {
        "candidate_key":           entry.candidate_key,
        "candidate_type":          entry.candidate_type_value,
        "direction":               entry.direction_value,
        "status":                  entry.status.value,
        "applied_delta_min":       entry.applied_delta_min,
        "new_cumulative_delta_min": entry.new_cumulative_delta_min,
        "monitored_outcome_count": entry.monitored_outcome_count,
        "rollback_recommended":    entry.rollback_recommended,
        "rollback_reasons":        list(entry.rollback_reasons),
        "adoption_ready":          entry.adoption_ready,
        "cooldown_active":         is_cooldown_active(entry, now_ts=now_ts),
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize_application_lifecycle_state(
    state: ApplicationLifecycleState,
) -> ApplicationLifecycleSummary:
    """Return aggregate counts for all lifecycle entries in a zone.

    Pure function — no side effects.
    """
    evs = list(state.entries.values())
    return ApplicationLifecycleSummary(
        total=len(evs),
        active=sum(1 for e in evs if e.status.value in _ACTIVE_STATUSES),
        rollback_recommended=sum(
            1 for e in evs if e.status is MonitoringStatus.ROLLBACK_RECOMMENDED
        ),
        adoption_ready=sum(1 for e in evs if e.adoption_ready),
        adopted=sum(1 for e in evs if e.status is MonitoringStatus.ADOPTED),
        rolled_back=sum(1 for e in evs if e.status is MonitoringStatus.ROLLED_BACK),
        expired=sum(1 for e in evs if e.status is MonitoringStatus.WINDOW_EXPIRED),
    )
