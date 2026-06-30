"""Shared helpers for Phase 6 episode-builder tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.regime import RegimeResult

UTC = timezone.utc
T0 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def regime(kind: Regime, reliability: float = 0.8, **kw) -> RegimeResult:
    if kind is Regime.DISTURBED and "disturbance_reasons" not in kw:
        kw["disturbance_reasons"] = ("window_open",)
    return RegimeResult(regime=kind, reliability=reliability, **kw)
