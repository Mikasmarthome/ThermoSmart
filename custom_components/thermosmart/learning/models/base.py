"""Shared robust-statistics helpers for LE 2.0 models (pure Python).

Only genuinely shared numeric utilities live here — no speculative model
framework. Used now by HeatRateModel; reusable by later rate-style models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


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
