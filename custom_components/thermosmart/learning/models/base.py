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
