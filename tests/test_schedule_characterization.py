"""Schedule semantics across the three call sites.

Originally the S7.0 characterization of current behaviour; the schedule-off,
midnight-wrap and midnight-continuity assertions (marked S7.8 A/B/C) now assert
the *fixed* behaviour from the S7.8 schedule behaviour-fix commit, while the
rest still pin the long-standing semantics that the S7.1 extraction preserved:

  * coordinator._current_schedule_period()     – wd/we comfort-vs-night label
  * coordinator._minutes_until_next_comfort()  – minutes to next comfort window
  * schedule.resolve_base_target()             – mode mapping + schedule target

Weekday/weekend slot defaults: wd 06:00/22:00, we 08:00/23:00. In the test
environment HA's as_local() returns UTC, so a UTC-aware FakeClock time maps
directly to the local weekday/hour the schedule reads.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.const import (
    HEATING_MODE_AUTO,
    HEATING_MODE_AWAY,
    HEATING_MODE_COMFORT,
    HEATING_MODE_ECO,
    HEATING_MODE_NIGHT,
    HEATING_MODE_VACATION,
)
from custom_components.thermosmart import schedule
from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_coordinator, make_zone_config

# 2025-06-16 Monday (weekday 0); 2025-06-21 Saturday (weekday 5);
# 2025-06-20 Friday (weekday 4); 2025-06-22 Sunday (weekday 6).
MON = datetime(2025, 6, 16, tzinfo=timezone.utc)
SAT = datetime(2025, 6, 21, tzinfo=timezone.utc)
FRI = datetime(2025, 6, 20, tzinfo=timezone.utc)
SUN = datetime(2025, 6, 22, tzinfo=timezone.utc)


def _coord_at(when: datetime, **cfg):
    return make_coordinator(cfg, clock=FakeClock(start=when))


# ── _current_schedule_period ────────────────────────────────────────────────

class TestCurrentSchedulePeriod:
    def test_weekday_morning_is_comfort(self):
        assert _coord_at(MON.replace(hour=9))._current_schedule_period() == "wd_comfort"

    def test_weekday_evening_night_is_night(self):
        assert _coord_at(MON.replace(hour=23))._current_schedule_period() == "wd_night"

    def test_weekday_early_morning_night_inherits_previous_day(self):
        # S7.8(C): Monday 05:00 is the tail of Sunday's (weekend) night → we_night,
        # not a fresh wd_night that would wrongly expire an evening override.
        assert _coord_at(MON.replace(hour=5))._current_schedule_period() == "we_night"

    def test_weekend_morning_is_comfort(self):
        assert _coord_at(SAT.replace(hour=9))._current_schedule_period() == "we_comfort"

    def test_weekend_early_morning_night_inherits_previous_day(self):
        # S7.8(C): Saturday 07:00 is the tail of Friday's (weekday) night → wd_night.
        assert _coord_at(SAT.replace(hour=7))._current_schedule_period() == "wd_night"

    def test_exact_morning_boundary_weekday_is_comfort(self):
        # cur == morning (06:00): not < morning → comfort
        assert _coord_at(MON.replace(hour=6, minute=0))._current_schedule_period() == "wd_comfort"

    def test_exact_night_boundary_weekday_is_night(self):
        # cur == night (22:00): cur >= night → night
        assert _coord_at(MON.replace(hour=22, minute=0))._current_schedule_period() == "wd_night"

    def test_schedule_disabled_is_always_comfort(self):
        # S7.8(A): with the schedule off the zone is always in comfort.
        c = _coord_at(MON.replace(hour=5), schedule_enabled=False)
        assert c._current_schedule_period() == "always_comfort"

    def test_night_label_is_stable_across_midnight(self):
        # S7.8(C): the night Fri 22:00 → Sat 08:00 keeps one label across midnight,
        # so an evening override does not expire at 00:00.
        assert _coord_at(FRI.replace(hour=23))._current_schedule_period() == "wd_night"
        assert _coord_at(SAT.replace(hour=0, minute=30))._current_schedule_period() == "wd_night"


# ── _minutes_until_next_comfort ─────────────────────────────────────────────

class TestMinutesUntilNextComfort:
    def _m(self, coord):
        return coord._minutes_until_next_comfort(coord.zone_cfg)

    def test_inside_comfort_window_returns_none(self):
        assert self._m(_coord_at(MON.replace(hour=12))) is None

    def test_before_morning_returns_delta(self):
        # 05:00 → 06:00 morning = 60 min (today's morning, same day-type)
        assert self._m(_coord_at(MON.replace(hour=5, minute=0))) == 60

    def test_after_night_wraps_to_next_weekday_morning(self):
        # Monday 23:00 → Tuesday (weekday) 06:00 = (1440-1380)+360 = 420
        assert self._m(_coord_at(MON.replace(hour=23, minute=0))) == 420

    def test_sunday_night_wraps_to_monday_weekday_morning(self):
        # S7.8(B): Sunday 23:30 wraps to MONDAY's weekday morning (06:00), not the
        # weekend morning. 1410 → (1440-1410)+360 = 390.
        assert self._m(_coord_at(SUN.replace(hour=23, minute=30))) == 390

    def test_friday_night_wraps_to_saturday_weekend_morning(self):
        # S7.8(B): Friday 22:30 wraps to SATURDAY's weekend morning (08:00), not the
        # weekday morning. 1350 → (1440-1350)+480 = 570.
        assert self._m(_coord_at(FRI.replace(hour=22, minute=30))) == 570

    def test_schedule_disabled_returns_none(self):
        # S7.8(A): schedule off → comfort always active → no upcoming transition.
        c = _coord_at(MON.replace(hour=5, minute=0), schedule_enabled=False)
        assert self._m(c) is None


# ── schedule.resolve_base_target: mode mapping ──────────────────────────────

class TestBaseTargetModeMapping:
    def _target(self, when, mode, **cfg):
        # In the test environment HA's as_local() is identity for UTC, so the
        # coordinator's now_local equals this UTC-aware `when`.
        z = make_zone_config(**cfg)
        return schedule.resolve_base_target(
            when, mode,
            comfort_temp=21.0, night_temp=18.0, away_temp=17.0,
            vacation_temp=12.0, eco_temp=19.0, cfg=z,
        )

    def test_comfort_mode(self):
        assert self._target(MON.replace(hour=3), HEATING_MODE_COMFORT) == 21.0

    def test_night_mode(self):
        assert self._target(MON.replace(hour=12), HEATING_MODE_NIGHT) == 18.0

    def test_away_mode(self):
        assert self._target(MON.replace(hour=12), HEATING_MODE_AWAY) == 17.0

    def test_vacation_mode(self):
        assert self._target(MON.replace(hour=12), HEATING_MODE_VACATION) == 12.0

    def test_eco_mode(self):
        assert self._target(MON.replace(hour=12), HEATING_MODE_ECO) == 19.0

    def test_unknown_non_auto_mode_falls_back_to_night(self):
        assert self._target(MON.replace(hour=12), "frobnicate") == 18.0

    def test_auto_comfort_window_returns_comfort(self):
        assert self._target(MON.replace(hour=12), HEATING_MODE_AUTO) == 21.0

    def test_auto_night_window_returns_night(self):
        assert self._target(MON.replace(hour=5), HEATING_MODE_AUTO) == 18.0

    def test_auto_weekend_morning_is_later(self):
        # weekend morning default 08:00 → 07:00 is still night
        assert self._target(SAT.replace(hour=7), HEATING_MODE_AUTO) == 18.0

    def test_auto_schedule_disabled_returns_comfort(self):
        # schedule_enabled=False → base target is always comfort in AUTO
        assert self._target(
            MON.replace(hour=5), HEATING_MODE_AUTO, schedule_enabled=False
        ) == 21.0
