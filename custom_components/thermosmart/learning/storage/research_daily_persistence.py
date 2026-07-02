"""Pure, bounded helpers for creating, aggregating, and retaining LE2
Research Daily Buckets.

Foundation only — no runtime producer exists yet anywhere (no call site
constructs a ``ResearchDailyObservation``), no ``LearningShadowController``
wiring, no store wrapper, no export wiring. This module defines the pure
aggregation/retention pipeline a future, separately-approved step would call
from higher-level controller methods, mirroring exactly how
``episode_persistence.py``/``support_event_persistence.py`` were built and
approved as standalone foundations before their own later runtime-hook
steps:

    ResearchDailyObservation (small delta)
    -> record_research_daily_observation()   merges into one day's bucket
    -> append_or_update_research_daily_bucket()
       serializes and stores the updated bucket in a
       {"buckets": {bucket_date: <flat dict>}} payload
    -> prune_research_daily_buckets()        applied immediately, using ONE
       central retention policy (age + a total bucket-count cap)

No HA imports. No runtime imports. No control imports. No storage I/O of any
kind — every function here is a pure transformation of in-memory dicts /
dataclasses; the caller decides if/when to persist the resulting payload
through a future store wrapper. Never mutates the ``bucket``/``observation``
objects passed in (both frozen dataclasses), nor the input ``payload`` in
place (a new dict is always returned).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping, Optional

from ..registry import RetentionPolicy
from ..research_daily_schemas import ResearchDailyBucket, ResearchDailyObservation
from .research_daily_serialization import (
    deserialize_research_daily_bucket,
    serialize_research_daily_bucket,
)

# -- retention constants ------------------------------------------------------
#
# Long-term target: 180-365 days of daily buckets, bounded without an
# external DB/Recorder dependency. 365 covers a full year of longitudinal
# "is ThermoSmart getting better" comparisons (season-over-season).
RESEARCH_DAILY_RETENTION_MAX_DAYS = 365

# A modest margin above 365 (leap years, clock-skew-adjacent duplicate-date
# edge cases, a bucket lingering one extra cycle before its own age-based
# eviction catches up) rather than an exact 365 cap — the age-based rule
# above is the real long-term bound; this is a hard safety net, not the
# primary limit. Kept as ONE central constant so a later calibration pass
# has exactly one place to change; covered by tests below.
RESEARCH_DAILY_RETENTION_MAX_BUCKETS = 400

RESEARCH_DAILY_RETENTION_DEFAULT = RetentionPolicy(
    max_age_days=RESEARCH_DAILY_RETENTION_MAX_DAYS,
    max_records=RESEARCH_DAILY_RETENTION_MAX_BUCKETS,
)


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _nonneg(value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n >= 0 else 0


def create_empty_research_daily_bucket(bucket_date: str) -> ResearchDailyBucket:
    """Return a fresh, all-zero/None ResearchDailyBucket for ``bucket_date``.

    Raises ``InvalidBucketDateError`` (from research_daily_schemas.py) for a
    malformed date — construction is strict, matching episode/support-event
    dataclass style; callers that only have untrusted date strings should
    validate/catch before calling this, or use the persistence-layer
    ``append_or_update_research_daily_bucket()`` below, which stays
    non-fatal end-to-end.
    """
    from ..research_daily_schemas import RESEARCH_DAILY_SCHEMA_VERSION
    return ResearchDailyBucket(schema_version=RESEARCH_DAILY_SCHEMA_VERSION, bucket_date=bucket_date)


def record_research_daily_observation(
    bucket: ResearchDailyBucket, observation: ResearchDailyObservation,
) -> ResearchDailyBucket:
    """Merge one small observation into ``bucket``, returning a NEW bucket.

    Pure function — never mutates ``bucket``/``observation``. Counters are
    added (non-negative, clamped defensively). ``overshoot_c``/
    ``undershoot_c``/``comfort_error_c`` (one new sample each, if given) are
    folded into their running sum+count pair. ``learning_progress_pct``/
    ``confidence`` (one new reading each, if given) update the running
    min/max/last triple. A NaN/inf/non-numeric sample is silently ignored
    (never raises, never corrupts the running statistic) — matching the
    ``math.isfinite()`` numeric-safety idiom already used across this
    codebase's learning modules.

    Does not itself validate that ``observation.bucket_date`` matches
    ``bucket.bucket_date`` — selecting the correct day's bucket for a given
    observation is the caller's responsibility (see
    ``append_or_update_research_daily_bucket()``, which does that lookup).
    """
    def _min(cur: Optional[float], new: Optional[float]) -> Optional[float]:
        if new is None:
            return cur
        return new if cur is None else min(cur, new)

    def _max(cur: Optional[float], new: Optional[float]) -> Optional[float]:
        if new is None:
            return cur
        return new if cur is None else max(cur, new)

    overshoot_sample = _finite_or_none(observation.overshoot_c)
    undershoot_sample = _finite_or_none(observation.undershoot_c)
    comfort_error_sample = _finite_or_none(observation.comfort_error_c)
    progress_sample = _finite_or_none(observation.learning_progress_pct)
    confidence_sample = _finite_or_none(observation.confidence)

    return ResearchDailyBucket(
        schema_version=bucket.schema_version,
        bucket_date=bucket.bucket_date,
        decision_count=bucket.decision_count + _nonneg(observation.decision_count),
        heating_allowed_count=bucket.heating_allowed_count + _nonneg(observation.heating_allowed_count),
        heating_blocked_count=bucket.heating_blocked_count + _nonneg(observation.heating_blocked_count),
        trv_command_sent_count=bucket.trv_command_sent_count + _nonneg(observation.trv_command_sent_count),
        trv_command_blocked_count=(
            bucket.trv_command_blocked_count + _nonneg(observation.trv_command_blocked_count)),
        same_setpoint_block_count=(
            bucket.same_setpoint_block_count + _nonneg(observation.same_setpoint_block_count)),
        trv_unavailable_count=bucket.trv_unavailable_count + _nonneg(observation.trv_unavailable_count),
        window_hold_count=bucket.window_hold_count + _nonneg(observation.window_hold_count),
        summer_hold_count=bucket.summer_hold_count + _nonneg(observation.summer_hold_count),
        manual_override_count=bucket.manual_override_count + _nonneg(observation.manual_override_count),
        boost_started_count=bucket.boost_started_count + _nonneg(observation.boost_started_count),
        boost_blocked_count=bucket.boost_blocked_count + _nonneg(observation.boost_blocked_count),
        boost_ended_count=bucket.boost_ended_count + _nonneg(observation.boost_ended_count),
        sensor_unavailable_count=(
            bucket.sensor_unavailable_count + _nonneg(observation.sensor_unavailable_count)),
        sensor_restored_count=bucket.sensor_restored_count + _nonneg(observation.sensor_restored_count),
        fallback_used_count=bucket.fallback_used_count + _nonneg(observation.fallback_used_count),
        outcome_resolved_count=bucket.outcome_resolved_count + _nonneg(observation.outcome_resolved_count),
        outcome_success_count=bucket.outcome_success_count + _nonneg(observation.outcome_success_count),
        outcome_failed_count=bucket.outcome_failed_count + _nonneg(observation.outcome_failed_count),
        outcome_confounded_count=(
            bucket.outcome_confounded_count + _nonneg(observation.outcome_confounded_count)),
        overshoot_sum_c=bucket.overshoot_sum_c + (overshoot_sample or 0.0),
        overshoot_count=bucket.overshoot_count + (1 if overshoot_sample is not None else 0),
        undershoot_sum_c=bucket.undershoot_sum_c + (undershoot_sample or 0.0),
        undershoot_count=bucket.undershoot_count + (1 if undershoot_sample is not None else 0),
        comfort_error_sum_c=bucket.comfort_error_sum_c + (comfort_error_sample or 0.0),
        comfort_error_count=bucket.comfort_error_count + (1 if comfort_error_sample is not None else 0),
        learning_progress_min_pct=_min(bucket.learning_progress_min_pct, progress_sample),
        learning_progress_max_pct=_max(bucket.learning_progress_max_pct, progress_sample),
        learning_progress_last_pct=(
            progress_sample if progress_sample is not None else bucket.learning_progress_last_pct),
        confidence_min=_min(bucket.confidence_min, confidence_sample),
        confidence_max=_max(bucket.confidence_max, confidence_sample),
        confidence_last=confidence_sample if confidence_sample is not None else bucket.confidence_last,
    )


def merge_research_daily_bucket(
    bucket: ResearchDailyBucket, delta: ResearchDailyBucket,
) -> ResearchDailyBucket:
    """Merge two ALREADY-BUILT buckets for the SAME ``bucket_date`` (e.g. a
    maintenance/consolidation pass over duplicate entries), returning a NEW
    bucket. Pure function — never mutates either input.

    Counters/sums add. For the min/max/last statistics, ``delta`` is treated
    as the chronologically LATER bucket: its ``*_last_*`` value wins when
    present, while min/max combine both sides. Returns ``bucket`` unchanged
    (as a new, structurally-identical instance) if ``delta.bucket_date``
    does not match ``bucket.bucket_date`` — merging across different days
    would silently corrupt a day's aggregate, so this is a safe no-op
    rather than a raise (matching this module's "never raise" contract).
    """
    if delta.bucket_date != bucket.bucket_date:
        return bucket

    def _min(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    def _max(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None:
            return b
        if b is None:
            return a
        return max(a, b)

    return ResearchDailyBucket(
        schema_version=bucket.schema_version,
        bucket_date=bucket.bucket_date,
        decision_count=bucket.decision_count + delta.decision_count,
        heating_allowed_count=bucket.heating_allowed_count + delta.heating_allowed_count,
        heating_blocked_count=bucket.heating_blocked_count + delta.heating_blocked_count,
        trv_command_sent_count=bucket.trv_command_sent_count + delta.trv_command_sent_count,
        trv_command_blocked_count=bucket.trv_command_blocked_count + delta.trv_command_blocked_count,
        same_setpoint_block_count=bucket.same_setpoint_block_count + delta.same_setpoint_block_count,
        trv_unavailable_count=bucket.trv_unavailable_count + delta.trv_unavailable_count,
        window_hold_count=bucket.window_hold_count + delta.window_hold_count,
        summer_hold_count=bucket.summer_hold_count + delta.summer_hold_count,
        manual_override_count=bucket.manual_override_count + delta.manual_override_count,
        boost_started_count=bucket.boost_started_count + delta.boost_started_count,
        boost_blocked_count=bucket.boost_blocked_count + delta.boost_blocked_count,
        boost_ended_count=bucket.boost_ended_count + delta.boost_ended_count,
        sensor_unavailable_count=bucket.sensor_unavailable_count + delta.sensor_unavailable_count,
        sensor_restored_count=bucket.sensor_restored_count + delta.sensor_restored_count,
        fallback_used_count=bucket.fallback_used_count + delta.fallback_used_count,
        outcome_resolved_count=bucket.outcome_resolved_count + delta.outcome_resolved_count,
        outcome_success_count=bucket.outcome_success_count + delta.outcome_success_count,
        outcome_failed_count=bucket.outcome_failed_count + delta.outcome_failed_count,
        outcome_confounded_count=bucket.outcome_confounded_count + delta.outcome_confounded_count,
        overshoot_sum_c=bucket.overshoot_sum_c + delta.overshoot_sum_c,
        overshoot_count=bucket.overshoot_count + delta.overshoot_count,
        undershoot_sum_c=bucket.undershoot_sum_c + delta.undershoot_sum_c,
        undershoot_count=bucket.undershoot_count + delta.undershoot_count,
        comfort_error_sum_c=bucket.comfort_error_sum_c + delta.comfort_error_sum_c,
        comfort_error_count=bucket.comfort_error_count + delta.comfort_error_count,
        learning_progress_min_pct=_min(bucket.learning_progress_min_pct, delta.learning_progress_min_pct),
        learning_progress_max_pct=_max(bucket.learning_progress_max_pct, delta.learning_progress_max_pct),
        learning_progress_last_pct=(
            delta.learning_progress_last_pct
            if delta.learning_progress_last_pct is not None
            else bucket.learning_progress_last_pct
        ),
        confidence_min=_min(bucket.confidence_min, delta.confidence_min),
        confidence_max=_max(bucket.confidence_max, delta.confidence_max),
        confidence_last=(
            delta.confidence_last if delta.confidence_last is not None else bucket.confidence_last
        ),
    )


@dataclass(frozen=True)
class ResearchDailyAppendResult:
    """Outcome of one append/aggregate/prune pass over a Research Daily
    Buckets payload. Describes what changed — never mutates the input
    payload in place.
    """
    updated_payload: dict
    bucket_date: str
    created: bool                       # True if this was a brand-new bucket
    pruned_bucket_dates: tuple[str, ...]


def _buckets_dict(payload) -> dict:
    if isinstance(payload, Mapping):
        existing = payload.get("buckets")
        if isinstance(existing, Mapping):
            return dict(existing)
    return {}


def append_or_update_research_daily_bucket(
    payload,
    bucket_date: str,
    observation: ResearchDailyObservation,
    *,
    now_utc: datetime,
    policy: RetentionPolicy = RESEARCH_DAILY_RETENTION_DEFAULT,
) -> ResearchDailyAppendResult:
    """Merge ONE observation into ``bucket_date``'s bucket within ``payload``,
    creating that day's bucket if it does not exist yet, then applying
    retention immediately. Pure function — no storage I/O. Never raises: a
    malformed existing entry for ``bucket_date`` is treated as missing (a
    fresh bucket is created), matching the conservative "skip malformed,
    keep going" behaviour episode/support-event persistence already use.
    """
    entries = _buckets_dict(payload)
    try:
        existing_raw = entries.get(bucket_date)
        existing_bucket = deserialize_research_daily_bucket(existing_raw) if existing_raw else None
        created = existing_bucket is None
        if existing_bucket is None:
            existing_bucket = create_empty_research_daily_bucket(bucket_date)
        updated_bucket = record_research_daily_observation(existing_bucket, observation)
        serialized = serialize_research_daily_bucket(updated_bucket)
        if serialized is not None:
            entries[bucket_date] = serialized
    except Exception:
        # Never let a malformed date/observation abort the whole payload —
        # the entries dict is left exactly as it was found for this key.
        created = bucket_date not in entries

    prune_result = prune_research_daily_buckets(
        {"buckets": entries}, now_utc=now_utc, policy=policy,
    )
    return ResearchDailyAppendResult(
        updated_payload=prune_result.updated_payload,
        bucket_date=bucket_date,
        created=created,
        pruned_bucket_dates=prune_result.pruned_bucket_dates,
    )


@dataclass(frozen=True)
class ResearchDailyPruneResult:
    """Result of evaluating retention across a Research Daily Buckets
    payload. Describes what was pruned — never mutates storage.
    """
    updated_payload: dict
    keep_bucket_dates: tuple[str, ...]
    pruned_bucket_dates: tuple[str, ...]


def _parse_bucket_date(value: object) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        y, m, d = (int(p) for p in value.split("-"))
        return date(y, m, d)
    except (ValueError, AttributeError):
        return None


def prune_research_daily_buckets(
    payload,
    *,
    now_utc: datetime,
    policy: RetentionPolicy = RESEARCH_DAILY_RETENTION_DEFAULT,
) -> ResearchDailyPruneResult:
    """Decide which buckets may be pruned from a Research Daily Buckets
    payload. Pure function — no storage I/O, no mutation of inputs. Never
    raises. Idempotent: pruning an already-pruned payload changes nothing.

    Age (``policy.max_age_days``) is evaluated first, using each entry's own
    ``bucket_date`` — an entry with a missing/malformed ``bucket_date`` is
    always kept (mirrors ``retention.py``'s conservative "unknown timestamp
    -> keep" contract for episodes/raw segments: it can never be safely aged
    out). If ``policy.max_records`` still caps the survivors, eviction is
    oldest-``bucket_date``-first among entries with a KNOWN date — unknown-
    date entries are excluded from eviction, never assumed oldest (same
    conservative pattern). Output is stable-sorted by ``bucket_date``.
    """
    entries = _buckets_dict(payload)
    now_date = now_utc.date() if isinstance(now_utc, datetime) else now_utc

    def _is_age_candidate(key: str) -> bool:
        if policy.max_age_days is None:
            return False
        entry = entries.get(key)
        bucket_date_value = entry.get("bucket_date") if isinstance(entry, Mapping) else key
        parsed = _parse_bucket_date(bucket_date_value)
        if parsed is None:
            return False  # missing/malformed date -> conservative keep
        age_days = (now_date - parsed).days
        return age_days > policy.max_age_days

    to_delete: set = {key for key in entries if _is_age_candidate(key)}

    if policy.max_records is not None:
        survivors = [key for key in entries if key not in to_delete]
        if len(survivors) > policy.max_records:
            def _known_date(key: str) -> Optional[date]:
                entry = entries.get(key)
                value = entry.get("bucket_date") if isinstance(entry, Mapping) else key
                return _parse_bucket_date(value)

            known_survivors = [key for key in survivors if _known_date(key) is not None]
            ordered = sorted(known_survivors, key=_known_date)
            overflow = min(len(survivors) - policy.max_records, len(ordered))
            to_delete.update(ordered[:overflow])

    keep_keys = sorted(
        (key for key in entries if key not in to_delete),
        key=lambda k: (_parse_bucket_date(k) is None, _parse_bucket_date(k) or date.min, k),
    )
    updated_entries = {key: entries[key] for key in keep_keys}

    return ResearchDailyPruneResult(
        updated_payload={"buckets": updated_entries},
        keep_bucket_dates=tuple(keep_keys),
        pruned_bucket_dates=tuple(sorted(to_delete)),
    )
