"""Shared robust-statistics helpers for Learning models (pure Python).

Only genuinely shared numeric utilities live here — no speculative model
framework. Used now by HeatRateModel; reusable by later rate-style models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

# -- confidence freshness (staleness) -----------------------------------------
#
# Conservative defaults: a 14-day grace window absorbs normal gaps between
# heating cycles/episodes without any decay, a 90-day floor stops the decay
# from ever reaching zero (physical building characteristics don't invalidate
# just because nothing new was observed — this says "unconfirmed", not
# "wrong"), and STALE_THRESHOLD triggers the aggregator's existing stale
# handling (confidence.py already multiplies freshness into effective
# confidence and applies extra caps when status is STALE).
FRESHNESS_GRACE_DAYS = 14.0
FRESHNESS_FLOOR_DAYS = 90.0
FRESHNESS_FLOOR = 0.45
STALE_THRESHOLD = 0.70


def is_finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(frozen=True)
class RobustEmaUpdate:
    value: float
    effective_n: float
    dispersion: float
    alpha: float


def robust_ema_update(
    value: float, effective_n: float, dispersion: float,
    sample: float, weight: float,
) -> RobustEmaUpdate:
    """Conservative weighted EMA whose learning rate shrinks with evidence.

    ``alpha = weight / (effective_n + weight)`` → few new samples cannot
    overwrite a well-evidenced value. ``dispersion`` is an EMA of the absolute
    deviation, a robust spread proxy used for outlier scoring.
    """
    if effective_n <= 0:
        return RobustEmaUpdate(value=sample, effective_n=weight,
                               dispersion=0.0, alpha=1.0)
    alpha = weight / (effective_n + weight)
    new_value = value + alpha * (sample - value)
    deviation = abs(sample - value)
    new_dispersion = dispersion + alpha * (deviation - dispersion)
    return RobustEmaUpdate(value=new_value, effective_n=effective_n + weight,
                           dispersion=new_dispersion, alpha=alpha)


def outlier_z(sample: float, value: float, dispersion: float) -> float:
    """Robust z-like score relative to the model's own dispersion (not a prior)."""
    if dispersion <= 1e-9:
        return 0.0
    return abs(sample - value) / dispersion


# -- persisted conditioned-bucket validation (D2-C1a) -------------------------
#
# HeatRateModel and HeatLossModel share the EXACT conditioned-bucket dict schema
# (key/rate_c_per_h/effective_n/dispersion/sample_count[/last_update_ts]) and the
# same "outdoor:<band>[|profile:<id>]" key grammar, so the structural validation
# genuinely belongs here. Only the plausibility rate band differs, so it is passed
# in per model. These helpers are pure, never raise, and return a short privacy-safe
# reason slug (no stored payload) for the FIRST violation — suitable for restore
# diagnostics. They quarantine a single corrupt conditioned bucket WITHOUT
# discarding the model's valid general/normalized learning.

_BUCKET_REQUIRED_FIELDS = ("key", "rate_c_per_h", "effective_n", "dispersion",
                           "sample_count")


def _is_nonneg_int(x) -> bool:
    # bool is an int subclass; a persisted sample_count must be a real int.
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _finite_num(x) -> bool:
    """Overflow-safe finiteness for a persisted numeric field. Unlike ``is_finite``,
    a huge integer (e.g. ``10**400``) — which ``math.isfinite`` cannot convert to
    float and raises ``OverflowError`` on — is treated as NOT a usable finite number
    (and quarantined) rather than being allowed to escape the per-bucket boundary.
    bool is excluded. Never raises."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(x)
    except (OverflowError, ValueError):
        return False


def _is_finite_nonneg(x, *, tol: float = 0.0) -> bool:
    return _finite_num(x) and x >= -tol


def validate_bucket_key(key, *, max_band: int) -> bool:
    """A conditioned-bucket key must be ``outdoor:<band>`` with ``band`` an integer
    in ``[0, max_band]`` (``max_band == len(outdoor_buckets)``, i.e. the number of
    bands the model can produce), optionally followed by ``|profile:<non-empty>``.
    Mirrors ``Model._bucket_key()`` exactly. Pure; never raises."""
    if not isinstance(key, str) or not key:
        return False
    parts = key.split("|")
    if len(parts) > 2:
        return False
    head = parts[0]
    prefix = "outdoor:"
    if not head.startswith(prefix):
        return False
    band_str = head[len(prefix):]
    if not band_str.isdigit():        # rejects "", "+1", "-1", "1.0", whitespace
        return False
    band = int(band_str)
    if band < 0 or band > max_band:
        return False
    if len(parts) == 2:
        prof = parts[1]
        if not prof.startswith("profile:") or len(prof) <= len("profile:"):
            return False
    return True


def validate_bucket_dict(key, d, *, max_band: int, max_rate: float,
                         rate_tol: float = 1e-6) -> Optional[str]:
    """Validate one persisted conditioned-bucket mapping. Return ``None`` when it is
    safe to load, else a short privacy-safe reason slug for the first violation.

    Checks: structural (mapping + all required fields present), key grammar/range,
    inner ``key`` field consistency, finite non-negative in-range ``rate_c_per_h``
    (``[0, max_rate]`` with fp tolerance — rejects NaN/inf/negative/implausible),
    finite non-negative ``effective_n`` and ``dispersion``, non-negative integer
    ``sample_count``, and an optional string ``last_update_ts``. Pure; never raises."""
    if not isinstance(d, dict):
        return "not_a_mapping"
    for f in _BUCKET_REQUIRED_FIELDS:
        if f not in d:
            return "missing_field"
    if not validate_bucket_key(key, max_band=max_band):
        return "bad_key"
    if d.get("key") != key:
        return "key_mismatch"
    rate = d.get("rate_c_per_h")
    if not _finite_num(rate):          # overflow-safe: a huge int -> non_finite_rate
        return "non_finite_rate"
    if rate < -rate_tol or rate > max_rate + rate_tol:
        return "rate_out_of_range"
    if not _is_finite_nonneg(d.get("effective_n")):
        return "bad_effective_n"
    if not _is_finite_nonneg(d.get("dispersion")):
        return "bad_dispersion"
    if not _is_nonneg_int(d.get("sample_count")):
        return "bad_sample_count"
    lut = d.get("last_update_ts")
    if lut is not None and not isinstance(lut, str):
        return "bad_last_update_ts"
    return None


def quarantine_buckets(raw_buckets, bucket_from, *, max_band: int, max_rate: float):
    """Load conditioned buckets one at a time, dropping (quarantining) any that fail
    validation while keeping the valid ones. ``bucket_from`` is the model's own
    ``_bucket_from`` loader. Returns ``(loaded_dict, quarantine_dict)`` where
    quarantine maps the offending key (stringified) to its reason slug. A corrupt
    bucket therefore never aborts the restore or discards sibling buckets."""
    loaded: dict = {}
    quarantine: dict = {}
    for k, b in dict(raw_buckets or {}).items():
        try:
            reason = validate_bucket_dict(k, b, max_band=max_band, max_rate=max_rate)
        except Exception:               # belt-and-braces at the per-bucket boundary:
            reason = "validation_error"  # a validation raise is itself a quarantine event
        if reason is not None:
            quarantine[str(k)] = reason
            continue
        try:
            loaded[k] = bucket_from(b)
        except Exception:               # belt-and-braces: never let a loader raise
            quarantine[str(k)] = "load_error"
    return loaded, quarantine


def freshness_from_age(
    now: Optional[str], last_update_ts: Optional[str],
) -> tuple[float, Optional[str]]:
    """Derive (freshness, status) from elapsed time since the model's own
    last_update_ts — never reads the wall clock; ``now`` must already be a
    materialised ISO timestamp from the caller's injected Clock.

    Conservative by design: missing/unparseable timestamps return
    ``(1.0, None)`` rather than inventing staleness for data that was never
    there — a genuinely cold-start model already reports low confidence via
    its own evidence-based value, so this must never double-penalize it.
    Linear decay between FRESHNESS_GRACE_DAYS and FRESHNESS_FLOOR_DAYS, never
    below FRESHNESS_FLOOR. status is "stale" once freshness drops below
    STALE_THRESHOLD, else None (healthy).
    """
    if now is None or last_update_ts is None:
        return 1.0, None
    try:
        now_dt = datetime.fromisoformat(now)
        last_dt = datetime.fromisoformat(last_update_ts)
    except (TypeError, ValueError):
        return 1.0, None
    age_days = (now_dt - last_dt).total_seconds() / 86400.0
    if age_days <= FRESHNESS_GRACE_DAYS:
        freshness = 1.0
    elif age_days >= FRESHNESS_FLOOR_DAYS:
        freshness = FRESHNESS_FLOOR
    else:
        span = FRESHNESS_FLOOR_DAYS - FRESHNESS_GRACE_DAYS
        frac = (age_days - FRESHNESS_GRACE_DAYS) / span
        freshness = 1.0 - frac * (1.0 - FRESHNESS_FLOOR)
    freshness = clamp(freshness, FRESHNESS_FLOOR, 1.0)
    status = "stale" if freshness < STALE_THRESHOLD else None
    return freshness, status
