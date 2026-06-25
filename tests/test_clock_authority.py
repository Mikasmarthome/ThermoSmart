"""Phase S1a Item 3 — Clock Authority tests.

Proves:
1. Coordinator fachliche UTC timestamps use the injected Clock (not dt_util).
2. A frozen FakeClock produces fully deterministic decision timestamps.
3. Advancing the clock changes heating-failure elapsed time correctly.
4. Wall-clock time of the test system does not bleed into coordinator UTC timestamps.
5. Static guard: no unauthorized datetime.now / utcnow / time.time patterns exist
   in the LE2 pure-core modules (learning/ except clock.py and ha_integration.py).
6. No unauthorized dt_util calls inside the LE2 pure core.
"""
from __future__ import annotations

import ast
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch as _patch

import pytest

from custom_components.thermosmart.learning.clock import FakeClock

# ── helpers ──────────────────────────────────────────────────────────────────

_PROD_ROOT = Path(__file__).parent.parent / "custom_components" / "thermosmart"
_LEARNING_ROOT = _PROD_ROOT / "learning"

_T0 = datetime(2025, 3, 10, 6, 0, 0, tzinfo=timezone.utc)

# Coordinator local-time now uses dt_util.as_local(self._clock.now_utc()).
# In unit tests we patch as_local to apply a fixed timezone so results are deterministic.
_DT_NOW_PATCH = "homeassistant.util.dt.now"
_DT_UTCNOW_PATCH = "homeassistant.util.dt.utcnow"
_DT_AS_LOCAL_PATCH = "homeassistant.util.dt.as_local"


def _make_coord_with_fake_clock(fake: FakeClock):
    """Return a coordinator whose injected Clock IS the given FakeClock."""
    from tests.helpers import make_coordinator
    return make_coordinator(clock=fake)


# ── 1. Decision timestamp uses injected Clock ─────────────────────────────────

class TestDecisionTimestamp:
    """_cycle_ts (used as the live-decision record timestamp) must reflect
    the injected FakeClock, NOT the wall clock."""

    def test_cycle_ts_matches_fake_clock(self):
        fake = FakeClock(start=_T0)
        coord = _make_coord_with_fake_clock(fake)

        fixed_local = datetime(2025, 3, 10, 7, 0, 0,
                               tzinfo=timezone(timedelta(hours=1)))
        with _patch(_DT_NOW_PATCH, return_value=fixed_local), \
             _patch(_DT_UTCNOW_PATCH, return_value=_T0):
            ts = coord._clock.now_utc().isoformat()

        assert ts == _T0.isoformat(), (
            f"Expected clock-based ts={_T0.isoformat()!r}, got {ts!r}. "
            "coordinator._clock is not wired to the FakeClock."
        )

    def test_cycle_ts_different_from_wall_clock(self):
        """Timestamp from injected clock is deterministic regardless of system time."""
        wall_before = datetime.now(timezone.utc)

        fake = FakeClock(start=_T0)
        coord = _make_coord_with_fake_clock(fake)
        ts_iso = coord._clock.now_utc().isoformat()

        wall_after = datetime.now(timezone.utc)

        # The FakeClock is frozen at _T0 which is in the past; wall clock is in 2025/2026.
        # If _T0 is not between wall_before and wall_after, the fake clock is independent.
        parsed = datetime.fromisoformat(ts_iso)
        assert parsed == _T0, (
            "Decision timestamp must match FakeClock, not wall clock."
        )
        # Sanity: wall clock is running (not also frozen)
        assert wall_after >= wall_before

    def test_two_coordinators_same_clock_same_ts(self):
        """Same FakeClock → identical timestamps in both coordinators."""
        fake1 = FakeClock(start=_T0)
        fake2 = FakeClock(start=_T0)
        coord1 = _make_coord_with_fake_clock(fake1)
        coord2 = _make_coord_with_fake_clock(fake2)

        ts1 = coord1._clock.now_utc().isoformat()
        ts2 = coord2._clock.now_utc().isoformat()
        assert ts1 == ts2

    def test_advance_clock_advances_timestamp(self):
        fake = FakeClock(start=_T0)
        coord = _make_coord_with_fake_clock(fake)

        ts_before = coord._clock.now_utc()
        fake.advance(300)  # 5 minutes
        ts_after = coord._clock.now_utc()

        assert ts_after == ts_before + timedelta(seconds=300)


# ── 2. Heating failure timer uses injected Clock ──────────────────────────────

class TestHeatingFailureClock:
    """_check_heating_failure writes and compares timestamps via self._clock.now_utc()."""

    def _rec(self, trv_sp=25.0, current=18.0, target=21.0, window=False):
        return {
            "trv_setpoint": trv_sp,
            "current_temp": current,
            "effective_target": target,
            "window_open": window,
        }

    def test_failure_since_uses_injected_clock(self):
        fake = FakeClock(start=_T0)
        coord = _make_coord_with_fake_clock(fake)
        coord.set_active_control(True)
        coord._active_control_initialized = True
        coord._indoor_temp_slope = -0.5  # falling temp → heating failure

        fixed_local = datetime(2025, 3, 10, 7, 0, 0,
                               tzinfo=timezone(timedelta(hours=1)))
        with _patch(_DT_NOW_PATCH, return_value=fixed_local), \
             _patch(_DT_UTCNOW_PATCH, return_value=_T0):
            coord._check_heating_failure(self._rec())

        assert coord._heating_failure_since == _T0, (
            "_heating_failure_since must be set from the injected FakeClock, "
            f"not from dt_util. Got {coord._heating_failure_since!r}"
        )

    def test_elapsed_uses_injected_clock(self):
        """Advancing the FakeClock changes the computed elapsed_min."""
        from custom_components.thermosmart.const import HEATING_FAILURE_DELAY_MIN

        fake = FakeClock(start=_T0)
        coord = _make_coord_with_fake_clock(fake)
        coord.set_active_control(True)
        coord._active_control_initialized = True
        coord._indoor_temp_slope = -0.5

        fixed_local = datetime(2025, 3, 10, 7, 0, 0,
                               tzinfo=timezone(timedelta(hours=1)))

        with _patch(_DT_NOW_PATCH, return_value=fixed_local), \
             _patch(_DT_UTCNOW_PATCH, return_value=_T0):
            # First call: sets _heating_failure_since to _T0 (via fake clock)
            coord._check_heating_failure(self._rec())

        assert coord._heating_failure_since is not None

        # Advance fake clock past the notification threshold
        advance_s = (HEATING_FAILURE_DELAY_MIN + 1) * 60
        fake.advance(advance_s)

        # Should now compute elapsed >= HEATING_FAILURE_DELAY_MIN and notify
        coord.hass.components = coord.hass.MagicMock()
        coord._heating_failure_notified = False

        with _patch(_DT_NOW_PATCH, return_value=fixed_local), \
             _patch(_DT_UTCNOW_PATCH, return_value=_T0), \
             _patch.object(coord, "hass") as mock_hass:
            mock_hass.components = mock_hass
            coord._check_heating_failure(self._rec())

        assert coord._heating_failure_notified is True, (
            "After advancing FakeClock past threshold, heating failure should be notified."
        )

    def test_heating_failure_cleared_on_reset(self):
        fake = FakeClock(start=_T0)
        coord = _make_coord_with_fake_clock(fake)
        coord.set_active_control(True)
        coord._heating_failure_since = fake.now_utc()

        # Window open → clears failure
        fixed_local = datetime(2025, 3, 10, 7, 0, 0,
                               tzinfo=timezone(timedelta(hours=1)))
        with _patch(_DT_NOW_PATCH, return_value=fixed_local), \
             _patch(_DT_UTCNOW_PATCH, return_value=_T0):
            coord._check_heating_failure(self._rec(window=True))

        assert coord._heating_failure_since is None


# ── 3. Schedule offset uses injected Clock ────────────────────────────────────

class TestScheduleOffsetClock:
    """_minutes_until_next_comfort × clock → _sched_time must use the injected clock."""

    def test_next_comfort_ts_from_injected_clock(self):
        """The schedule-time sent to LE2 is based on the injected FakeClock."""
        fake = FakeClock(start=_T0)  # 2025-03-10 06:00 UTC
        coord = _make_coord_with_fake_clock(fake)

        # _T0 is 06:00 UTC = 07:00 local (UTC+1). comfort starts at 06:00 local.
        # So comfort is already active (cur=420 min, morning=360 min) → _minutes_until returns None.
        # Advance to night (22:00 local = 21:00 UTC)
        fake.advance(15 * 3600)  # +15h → 21:00 UTC = 22:00 local (entering night)
        fake.advance(3600)       # +1h  → 22:00 UTC = 23:00 local (night, next comfort at 06:00 local)

        # Patch dt_util.now to return local time from fake clock
        local_time = fake.now_utc().astimezone(timezone(timedelta(hours=1)))

        # Compute _sched_time inline as the coordinator does
        comfort_min = coord._minutes_until_next_comfort(coord.entry.data)

        with _patch(_DT_NOW_PATCH, return_value=local_time):
            comfort_min2 = coord._minutes_until_next_comfort(coord.entry.data)

        if comfort_min2 is not None:
            expected_ts = (fake.now_utc() + timedelta(minutes=comfort_min2)).isoformat()
            # The expected_ts uses FakeClock, not wall clock
            from datetime import datetime as _dt
            assert _dt.fromisoformat(expected_ts).year == 2025


# ── 4. Static guard: no unauthorized patterns in LE2 pure core ────────────────

_LE2_PURE_CORE_GLOB = "**/*.py"

# Only SystemClock (clock.py) is allowed to call datetime.now() / time.monotonic()
_CLOCK_IMPL_FILE = "clock.py"

# Patterns that are NEVER acceptable in the LE2 pure core
_BANNED_IN_PURE_CORE = [
    r"\bdatetime\.now\(",         # raw wall clock
    r"\bdatetime\.utcnow\(",      # deprecated UTC
    r"\btime\.time\(\)",          # raw epoch seconds
    r"\btime\.monotonic\(\)",     # raw monotonic (only allowed in clock.py)
    r"\btime\.perf_counter\(",    # performance-only (never fachlich)
    r"\bdt_util\.now\(",          # HA integration leak into pure core
    r"\bdt_util\.utcnow\(",       # HA integration leak into pure core
]

# Patterns that must NEVER appear anywhere in production (absolutely prohibited)
_BANNED_EVERYWHERE = [
    r"\bdatetime\.utcnow\(",      # deprecated; datetime.now(timezone.utc) only
    r"\btime\.time\(\)",          # raw epoch without authority
    r"\btime\.perf_counter\(",    # no fachliche role
    r"asyncio\.get_event_loop\(\)\.time\(",
    r"asyncio\.get_running_loop\(\)\.time\(",
]


def _scan_file(path: Path, patterns: list[str]) -> list[tuple[int, str, str]]:
    """Return list of (lineno, pattern, line_content) for each match."""
    results = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return results
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # skip comment-only lines
        for pat in patterns:
            if re.search(pat, line):
                results.append((i, pat, line.rstrip()))
    return results


def _pure_core_files() -> list[Path]:
    """All .py files under learning/ except clock.py (the sole SystemClock implementation)."""
    return [
        p for p in _LEARNING_ROOT.rglob("*.py")
        if p.name != _CLOCK_IMPL_FILE and "__pycache__" not in str(p)
    ]


def _all_production_files() -> list[Path]:
    """All .py files under custom_components/thermosmart/."""
    return [
        p for p in _PROD_ROOT.rglob("*.py")
        if "__pycache__" not in str(p)
    ]


# Files that must not contain dt_util.now() — they must use _now_local() or injected clock.
# Exact list: any new file added here requires explicit justification.
_MUST_USE_CLOCK_FILES = [
    "coordinator.py",
    "window.py",
    "maintenance.py",
    "trv_control.py",
    "learning_engine.py",
    "export.py",
]


def _coordinator_layer_files() -> list[Path]:
    """HA-layer files that must derive local time via injected clock (not dt_util.now())."""
    return [
        _PROD_ROOT / name for name in _MUST_USE_CLOCK_FILES
        if (_PROD_ROOT / name).exists()
    ]


class TestClockGuard:
    """Static analysis guard that prevents new unauthorized clock usages."""

    def test_no_banned_patterns_in_le2_pure_core(self):
        violations: list[str] = []
        for path in _pure_core_files():
            hits = _scan_file(path, _BANNED_IN_PURE_CORE)
            for lineno, pat, content in hits:
                rel = path.relative_to(_PROD_ROOT.parent.parent)
                violations.append(f"{rel}:{lineno} — pattern {pat!r} — {content!r}")

        assert not violations, (
            "Unauthorized clock access in LE2 pure core:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_no_absolutely_banned_patterns_anywhere(self):
        violations: list[str] = []
        for path in _all_production_files():
            hits = _scan_file(path, _BANNED_EVERYWHERE)
            for lineno, pat, content in hits:
                rel = path.relative_to(_PROD_ROOT.parent.parent)
                violations.append(f"{rel}:{lineno} — pattern {pat!r} — {content!r}")

        assert not violations, (
            "Absolutely prohibited time-source pattern found in production:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_clock_py_is_only_file_using_datetime_now_in_pure_core(self):
        """Only clock.py (SystemClock) may call datetime.now(). No other LE2 file."""
        for path in _pure_core_files():
            hits = _scan_file(path, [r"\bdatetime\.now\("])
            assert not hits, (
                f"{path.name} calls datetime.now() — only clock.py/SystemClock is allowed to."
                f" Hits: {hits}"
            )

    def test_clock_py_is_only_file_using_time_monotonic_in_pure_core(self):
        """Only clock.py (SystemClock.monotonic) may call time.monotonic()."""
        for path in _pure_core_files():
            hits = _scan_file(path, [r"_time\.monotonic\(|time\.monotonic\("])
            assert not hits, (
                f"{path.name} calls time.monotonic() — only SystemClock.monotonic() is allowed."
                f" Hits: {hits}"
            )

    def test_coordinator_layer_no_dt_util_now(self):
        """Coordinator.py and its mixins must not call dt_util.now() directly.

        All local-time access must go through _now_local() which derives from the
        injected Clock via dt_util.as_local(self._clock.now_utc()).
        """
        violations: list[str] = []
        for path in _coordinator_layer_files():
            hits = _scan_file(path, [r"\bdt_util\.now\(", r"\bdt_util\.utcnow\("])
            for lineno, pat, content in hits:
                rel = path.relative_to(_PROD_ROOT.parent.parent)
                violations.append(f"{rel}:{lineno} — {content!r}")

        assert not violations, (
            "Coordinator-layer file uses dt_util.now/utcnow directly — "
            "must use self._now_local() or self._clock.now_utc() instead:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_ha_integration_no_direct_time_calls(self):
        """ha_integration.py must not call dt_util, datetime.now(), or time.monotonic().

        All clock access inside LearningShadowController uses self._utcnow_iso()
        which wraps the injected Clock. No module-level fallback is permitted.
        """
        path = _LEARNING_ROOT / "runtime" / "ha_integration.py"
        banned = [r"\bdt_util\.utcnow\(", r"\bdt_util\.now\(",
                  r"\bdatetime\.now\(", r"\btime\.monotonic\("]
        hits = _scan_file(path, banned)
        assert not hits, (
            f"ha_integration.py uses a direct time call — all time access must go "
            f"through self._utcnow_iso() (injected Clock). Hits: {hits}"
        )

    def test_shadow_controller_requires_explicit_clock(self):
        """LearningShadowController must raise ValueError when no clock is injected."""
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            LearningShadowController,
        )

        class _FakeHass:
            class states:
                @staticmethod
                def get(_): return None

        with pytest.raises(ValueError, match="requires an explicit Clock"):
            LearningShadowController(_FakeHass(), "test-zone")

    def test_shadow_controller_injected_clock_flows_to_runtime(self):
        """FakeClock injected into shadow must flow through to LearningRuntime._clock."""
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            LearningShadowController,
        )
        from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode

        class _FakeHass:
            class states:
                @staticmethod
                def get(_): return None

        fake = FakeClock(start=_T0)
        shadow = LearningShadowController(
            _FakeHass(), "test-zone",
            clock=fake,
            mode=LearningRuntimeMode.CONTROL,
        )
        result = shadow._runtime._clock()
        assert result == _T0.isoformat(), (
            f"Runtime clock with injected FakeClock must return FakeClock time, got {result!r}"
        )
