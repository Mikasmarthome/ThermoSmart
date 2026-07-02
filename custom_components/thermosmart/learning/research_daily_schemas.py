"""Typed schema for LE2 Research Daily Buckets (pure Python, no Home Assistant).

Research Daily Buckets are bounded, per-day AGGREGATE counters/summaries —
never a raw event/episode log. They exist to answer longitudinal research
questions ("is ThermoSmart getting better?", "how often was heating
blocked?", "how stable is confidence?") from small, fixed-shape daily
rollups instead of an ever-growing raw history. Mirrors the existing
``episode_schemas.py``/``support_event_schemas.py`` pattern: a frozen
dataclass for one materialised bucket, plus a small, frozen "delta" dataclass
(``ResearchDailyObservation``) a producer merges into it one small increment
at a time — no builder/instrumentation logic here; the actual producers
(Support Critical Event aggregation, Learning Progress/confidence sampling)
live in the runtime shadow controller layer, see
``research_daily_persistence.py``'s module docstring for the pure
merge/persist pipeline they call into. Not every field has a live producer
yet — ``decision_count``/``heating_allowed_count``/``heating_blocked_count``
remain schema-defined but unproduced until a real per-decision runtime
signal exists (deliberately never estimated from Support Events).

``bucket_date`` (``YYYY-MM-DD``, UTC calendar date) is the only "identity"
a bucket carries — no zone id, no entity ids, no episode/decision ids. The
store this bucket will later belong to is expected to be zone-scoped by its
own store key, exactly like ``EpisodesStore``/``SupportCriticalEventStore``
— duplicating zone identity inside every bucket would be a second place that
could leak it. A future, separate export step may add an already-
pseudonymized zone reference at export time, never a raw zone id stored
here.

All counters are additive, bounded, non-negative integers; all sums are
finite floats; min/max/last fields track a running statistic across the
observations merged into that day. No entity ids, no raw episode/event data,
no trajectories, no HA object reprs — every field is a scalar aggregate by
construction, so (unlike Support Critical Events) no free-form ``details``
mapping is needed and no key-substring stripping applies here.

No HA imports. No control imports. No storage I/O. Never mutates its inputs.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

_BUCKET_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class InvalidBucketDateError(ValueError):
    """Raised when a bucket_date is not a well-formed, real calendar date."""


def _validate_bucket_date(value: str) -> str:
    """Validate ``YYYY-MM-DD`` and that it is a real calendar date.

    Raises InvalidBucketDateError on any malformed input — used only at
    dataclass construction time (eager, strict); downstream serialization
    stays defensive (skip/None) rather than raising, matching the rest of
    this schema family (episodes, support events).
    """
    if not isinstance(value, str) or not _BUCKET_DATE_RE.match(value):
        raise InvalidBucketDateError(f"bucket_date must be YYYY-MM-DD, got {value!r}")
    try:
        from datetime import date
        y, m, d = (int(p) for p in value.split("-"))
        date(y, m, d)
    except ValueError as err:
        raise InvalidBucketDateError(f"bucket_date is not a real calendar date: {value!r}") from err
    return value


RESEARCH_DAILY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResearchDailyBucket:
    """One materialised, bounded per-day aggregate for one zone's Research
    Export. Immutable — every update (see
    ``research_daily_persistence.record_research_daily_observation()``)
    returns a NEW bucket, never mutates this one in place, matching the
    immutable-replace style already used for episodes/support events.
    """
    schema_version: int
    bucket_date: str  # YYYY-MM-DD, UTC calendar date — no zone identity

    decision_count: int = 0

    heating_allowed_count: int = 0
    heating_blocked_count: int = 0

    trv_command_sent_count: int = 0
    trv_command_blocked_count: int = 0
    same_setpoint_block_count: int = 0
    trv_unavailable_count: int = 0

    window_hold_count: int = 0
    summer_hold_count: int = 0
    manual_override_count: int = 0

    boost_started_count: int = 0
    boost_blocked_count: int = 0
    boost_ended_count: int = 0

    sensor_unavailable_count: int = 0
    sensor_restored_count: int = 0
    fallback_used_count: int = 0

    outcome_resolved_count: int = 0
    outcome_success_count: int = 0
    outcome_failed_count: int = 0
    outcome_confounded_count: int = 0

    overshoot_sum_c: float = 0.0
    overshoot_count: int = 0
    undershoot_sum_c: float = 0.0
    undershoot_count: int = 0
    comfort_error_sum_c: float = 0.0
    comfort_error_count: int = 0

    learning_progress_min_pct: Optional[float] = None
    learning_progress_max_pct: Optional[float] = None
    learning_progress_last_pct: Optional[float] = None

    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    confidence_last: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_bucket_date(self.bucket_date)
        for name in (
            "decision_count", "heating_allowed_count", "heating_blocked_count",
            "trv_command_sent_count", "trv_command_blocked_count",
            "same_setpoint_block_count", "trv_unavailable_count",
            "window_hold_count", "summer_hold_count", "manual_override_count",
            "boost_started_count", "boost_blocked_count", "boost_ended_count",
            "sensor_unavailable_count", "sensor_restored_count", "fallback_used_count",
            "outcome_resolved_count", "outcome_success_count", "outcome_failed_count",
            "outcome_confounded_count", "overshoot_count", "undershoot_count",
            "comfort_error_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        for name in ("overshoot_sum_c", "undershoot_sum_c", "comfort_error_sum_c"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        for name in (
            "learning_progress_min_pct", "learning_progress_max_pct", "learning_progress_last_pct",
            "confidence_min", "confidence_max", "confidence_last",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be None or a finite number, got {value!r}")


@dataclass(frozen=True)
class ResearchDailyObservation:
    """One small, additive delta to merge into a day's ResearchDailyBucket.

    All counter/sum fields default to zero and all statistic fields default
    to None — a caller only sets what it actually observed this call
    (typically: one Support Critical Event, one outcome resolution, or one
    learning-progress/confidence reading), never a full bucket rewrite.
    Nothing anywhere constructs one of these yet — this is the pure delta
    shape a future, separately-approved producer step would populate from
    Support Events / completed episodes / ``learning_progress_safe()``.
    """
    bucket_date: str

    decision_count: int = 0

    heating_allowed_count: int = 0
    heating_blocked_count: int = 0

    trv_command_sent_count: int = 0
    trv_command_blocked_count: int = 0
    same_setpoint_block_count: int = 0
    trv_unavailable_count: int = 0

    window_hold_count: int = 0
    summer_hold_count: int = 0
    manual_override_count: int = 0

    boost_started_count: int = 0
    boost_blocked_count: int = 0
    boost_ended_count: int = 0

    sensor_unavailable_count: int = 0
    sensor_restored_count: int = 0
    fallback_used_count: int = 0

    outcome_resolved_count: int = 0
    outcome_success_count: int = 0
    outcome_failed_count: int = 0
    outcome_confounded_count: int = 0

    overshoot_c: Optional[float] = None      # one new overshoot sample, if any
    undershoot_c: Optional[float] = None     # one new undershoot sample, if any
    comfort_error_c: Optional[float] = None  # one new comfort-error sample, if any

    learning_progress_pct: Optional[float] = None  # one new progress reading, if any
    confidence: Optional[float] = None              # one new confidence reading, if any

    def __post_init__(self) -> None:
        _validate_bucket_date(self.bucket_date)
