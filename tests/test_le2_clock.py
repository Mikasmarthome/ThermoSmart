"""Phase 1 tests for the injectable LE 2.0 clock."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.clock import (
    Clock,
    FakeClock,
    SystemClock,
)


class TestSystemClock:
    def test_now_utc_is_timezone_aware_utc(self):
        ts = SystemClock().now_utc()
        assert ts.tzinfo is not None
        assert ts.utcoffset() == timedelta(0)

    def test_monotonic_is_non_decreasing(self):
        clock = SystemClock()
        a = clock.monotonic()
        b = clock.monotonic()
        assert b >= a

    def test_satisfies_protocol(self):
        assert isinstance(SystemClock(), Clock)


class TestFakeClock:
    def test_requires_aware_start(self):
        with pytest.raises(ValueError):
            FakeClock(datetime(2026, 1, 1, 12, 0, 0))

    def test_normalises_start_to_utc(self):
        tz = timezone(timedelta(hours=2))
        clock = FakeClock(datetime(2026, 1, 1, 12, 0, 0, tzinfo=tz))
        assert clock.now_utc().utcoffset() == timedelta(0)
        assert clock.now_utc().hour == 10

    def test_advance_moves_both_clocks(self):
        clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        before_wall = clock.now_utc()
        before_mono = clock.monotonic()
        clock.advance(60)
        assert clock.now_utc() == before_wall + timedelta(seconds=60)
        assert clock.monotonic() == before_mono + 60

    def test_advance_monotonic_only(self):
        clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        wall = clock.now_utc()
        clock.advance_monotonic(30)
        assert clock.now_utc() == wall  # wall clock unaffected
        assert clock.monotonic() == 30

    def test_cannot_advance_backwards(self):
        clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        with pytest.raises(ValueError):
            clock.advance(-1)

    def test_satisfies_protocol(self):
        assert isinstance(FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc)), Clock)
