"""Versioned, defensive serialization for LE2 Research Daily Buckets.

Mirrors ``episode_serialization.py``/``support_event_serialization.py``'s
pattern: a flat, JSON-safe dict shape and a schema-version guard.

Storage-layer shape (``serialize_research_daily_bucket``/
``deserialize_research_daily_bucket``):
    A FLAT dict — ``schema_version``, ``bucket_date``, and every counter/sum/
    statistic field of ``ResearchDailyBucket`` at the top level. No nested
    "data" wrapper, matching the ``{"buckets": {bucket_date: <this dict>}}``
    container convention used by ``research_daily_persistence.py``.

No entity ids, no zone ids, no episode/decision ids, no raw trajectories —
every field is already a scalar aggregate by construction (see
``research_daily_schemas.py``'s module docstring), so unlike Support
Critical Events there is no free-form ``details`` mapping here and no
key-substring stripping is needed.

No HA imports. No runtime imports. No storage I/O of any kind. Never raises:
malformed/unknown input is handled defensively (skip fields or return None).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from ..research_daily_schemas import (
    RESEARCH_DAILY_SCHEMA_VERSION,
    InvalidBucketDateError,
    ResearchDailyBucket,
)

_COUNTER_FIELDS = (
    "decision_count",
    "heating_allowed_count", "heating_blocked_count",
    "trv_command_sent_count", "trv_command_blocked_count",
    "same_setpoint_block_count", "trv_unavailable_count",
    "window_hold_count", "summer_hold_count", "manual_override_count",
    "boost_started_count", "boost_blocked_count", "boost_ended_count",
    "sensor_unavailable_count", "sensor_restored_count", "fallback_used_count",
    "outcome_resolved_count", "outcome_success_count", "outcome_failed_count",
    "outcome_confounded_count", "overshoot_count", "undershoot_count",
    "comfort_error_count",
)
_SUM_FIELDS = ("overshoot_sum_c", "undershoot_sum_c", "comfort_error_sum_c")
_STAT_FIELDS = (
    "learning_progress_min_pct", "learning_progress_max_pct", "learning_progress_last_pct",
    "confidence_min", "confidence_max", "confidence_last",
)


def _clean_counter(value: Any) -> int:
    """Best-effort non-negative int; malformed/negative input becomes 0."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n >= 0 else 0


def _clean_sum(value: Any) -> float:
    """Best-effort finite float; malformed/NaN/inf input becomes 0.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _clean_stat(value: Any) -> Optional[float]:
    """Best-effort finite float or None; malformed/NaN/inf input becomes None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def serialize_research_daily_bucket(bucket: ResearchDailyBucket) -> Optional[dict]:
    """Serialize one ResearchDailyBucket to a flat, versioned, JSON-safe dict.

    Returns None for a non-bucket input or any encoding failure — never
    raises. Never mutates ``bucket``.
    """
    if not isinstance(bucket, ResearchDailyBucket):
        return None
    try:
        out: dict = {
            "schema_version": RESEARCH_DAILY_SCHEMA_VERSION,
            "bucket_date": bucket.bucket_date,
        }
        for name in _COUNTER_FIELDS:
            out[name] = _clean_counter(getattr(bucket, name))
        for name in _SUM_FIELDS:
            out[name] = _clean_sum(getattr(bucket, name))
        for name in _STAT_FIELDS:
            out[name] = _clean_stat(getattr(bucket, name))
        return out
    except Exception:
        return None


def deserialize_research_daily_bucket(raw: Mapping[str, Any]) -> Optional[ResearchDailyBucket]:
    """Reconstruct one ResearchDailyBucket from a
    serialize_research_daily_bucket() payload.

    Returns None on schema-version mismatch, malformed shape, an invalid
    ``bucket_date``, or any other reconstruction failure — never raises.
    Unknown/extra fields in ``raw`` are silently ignored. Missing known
    fields default to their schema default (0 / 0.0 / None). Malformed
    numeric values are cleaned (negative counters -> 0, NaN/inf -> 0.0 or
    None) rather than rejecting the whole bucket.
    """
    if not isinstance(raw, Mapping):
        return None
    if raw.get("schema_version") != RESEARCH_DAILY_SCHEMA_VERSION:
        return None
    bucket_date = raw.get("bucket_date")
    if not isinstance(bucket_date, str):
        return None
    try:
        kwargs: dict = {
            "schema_version": RESEARCH_DAILY_SCHEMA_VERSION,
            "bucket_date": bucket_date,
        }
        for name in _COUNTER_FIELDS:
            if name in raw:
                kwargs[name] = _clean_counter(raw[name])
        for name in _SUM_FIELDS:
            if name in raw:
                kwargs[name] = _clean_sum(raw[name])
        for name in _STAT_FIELDS:
            if name in raw:
                kwargs[name] = _clean_stat(raw[name])
        return ResearchDailyBucket(**kwargs)
    except (InvalidBucketDateError, ValueError, TypeError):
        return None
    except Exception:
        return None


def serialize_research_daily_history(history: Mapping[str, ResearchDailyBucket]) -> dict:
    """Serialize a bucket_date -> ResearchDailyBucket mapping to the
    ``{"buckets": {bucket_date: <flat dict>}}`` container shape.

    Skips any entry that fails to serialize (never raises); preserves the
    input's iteration order (callers typically pass an already date-sorted
    mapping — see ``research_daily_persistence.py``).
    """
    buckets: dict = {}
    if isinstance(history, Mapping):
        for key, bucket in history.items():
            serialized = serialize_research_daily_bucket(bucket)
            if serialized is not None:
                buckets[serialized["bucket_date"]] = serialized
    return {"buckets": buckets}


def deserialize_research_daily_history(raw: Any) -> dict:
    """Deserialize a ``{"buckets": {bucket_date: <flat dict>}}`` payload back
    into a bucket_date -> ResearchDailyBucket mapping.

    Malformed top-level shape returns an empty mapping. Each entry is
    validated independently via ``deserialize_research_daily_bucket()`` — a
    malformed individual entry is skipped, never aborts the whole load.
    Never raises.
    """
    if not isinstance(raw, Mapping):
        return {}
    raw_buckets = raw.get("buckets")
    if not isinstance(raw_buckets, Mapping):
        return {}
    result: dict = {}
    for key, entry in raw_buckets.items():
        bucket = deserialize_research_daily_bucket(entry)
        if bucket is not None:
            result[bucket.bucket_date] = bucket
    return result
