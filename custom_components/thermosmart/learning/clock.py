"""Injectable clock abstraction for the LE 2.0 pure-Python core.

The learning core must never call ``homeassistant.util.dt.now()`` (or any other
Home Assistant helper) directly. Instead every component receives a ``Clock``
and uses it for wall-clock time (UTC, timezone-aware) and a monotonic source
for intra-episode durations that survive system clock corrections.

This module has no Home Assistant dependency and is fully testable with a
deterministic ``FakeClock``.
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Time source contract used across the learning core."""

    def now_utc(self) -> datetime:
        """Return the current time as a timezone-aware UTC ``datetime``."""
        ...

    def monotonic(self) -> float:
        """Return a monotonically increasing time in seconds.

        Only differences are meaningful. Used for trajectory offsets and
        durations so NTP / wall-clock jumps cannot corrupt timing.
        """
        ...


class SystemClock:
    """Default production clock backed by the standard library."""

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return _time.monotonic()


class FakeClock:
    """Deterministic clock for tests.

    Wall-clock and monotonic time advance independently and only when the test
    explicitly advances them, making episode timing fully reproducible.
    """

    def __init__(self, start: datetime, monotonic_start: float = 0.0) -> None:
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("FakeClock start time must be timezone-aware")
        self._now = start.astimezone(timezone.utc)
        self._monotonic = float(monotonic_start)

    def now_utc(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    # -- test controls ---------------------------------------------------

    def set_utc(self, value: datetime) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("FakeClock time must be timezone-aware")
        self._now = value.astimezone(timezone.utc)

    def advance(self, seconds: float) -> None:
        """Advance both wall-clock and monotonic time by ``seconds``."""
        if seconds < 0:
            raise ValueError("cannot advance time backwards")
        self._monotonic += float(seconds)
        from datetime import timedelta

        self._now = self._now + timedelta(seconds=seconds)

    def advance_monotonic(self, seconds: float) -> None:
        """Advance only the monotonic source (simulate wall-clock drift)."""
        if seconds < 0:
            raise ValueError("cannot advance monotonic time backwards")
        self._monotonic += float(seconds)
