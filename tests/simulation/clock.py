"""SimulationClock — deterministic fake time for the S1 simulation harness.

Wraps FakeClock and patches homeassistant.util.dt.now / utcnow / as_local
globally so all production code (coordinator.py, learning_engine.py,
ha_integration.py) uses simulation time for the duration of the simulation loop.

``dt_util.as_local`` is patched to apply the simulation timezone (tz_offset_hours)
instead of HA's configured local timezone. This is required because coordinator.py
and its mixins now derive local time via ``dt_util.as_local(self._clock.now_utc())``.
Without the patch, ``as_local`` would convert simulation UTC to the test host's
local timezone, breaking schedule-period calculations in simulation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch


class SimulationClock:
    """Deterministic time controller that patches homeassistant.util.dt.

    Usage::

        clock = SimulationClock(start_utc=datetime(2024, 1, 8, tzinfo=timezone.utc))
        with clock:
            clock.advance(300)   # advance 5 minutes
            result = await coord._async_update_data()
    """

    def __init__(
        self,
        start_utc: datetime,
        tz_offset_hours: float = 1.0,
    ) -> None:
        from custom_components.thermosmart.learning.clock import FakeClock

        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)
        self._fake = FakeClock(start=start_utc)
        self._tz = timezone(timedelta(hours=tz_offset_hours))
        self._patches: list = []

    # ── Time accessors ──────────────────────────────────────────────────

    def now_utc(self) -> datetime:
        return self._fake.now_utc()

    def now_local(self) -> datetime:
        return self._fake.now_utc().astimezone(self._tz)

    def now_utc_iso(self) -> str:
        return self._fake.now_utc().isoformat()

    def monotonic(self) -> float:
        return self._fake.monotonic()

    def advance(self, seconds: float) -> None:
        self._fake.advance(seconds)

    # ── Patch context ───────────────────────────────────────────────────

    def start_patches(self) -> None:
        """Install dt_util patches. Must be called before simulation loop."""
        self._patches = [
            patch("homeassistant.util.dt.now", side_effect=lambda: self.now_local()),
            patch("homeassistant.util.dt.utcnow", side_effect=lambda: self.now_utc()),
            # Patch as_local to apply the simulation timezone instead of the host's
            # local timezone. Required because production code now uses
            # dt_util.as_local(clock.now_utc()) for all local-time conversions.
            patch("homeassistant.util.dt.as_local",
                  side_effect=lambda utc_dt: utc_dt.astimezone(self._tz)),
        ]
        for p in self._patches:
            p.start()

    def stop_patches(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()

    def __enter__(self) -> "SimulationClock":
        self.start_patches()
        return self

    def __exit__(self, *_) -> None:
        self.stop_patches()
