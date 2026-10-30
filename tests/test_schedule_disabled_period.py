"""With the schedule disabled there are no slots, so no slot change either.

The period label is what a manual override compares against to know it has
outlived the slot it was set in. A disabled schedule used to report a
day-type dependent comfort label (weekday vs weekend), so an override expired
at the Friday -> Saturday and Sunday -> Monday boundaries although no schedule
slot had changed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart import schedule
from custom_components.thermosmart.const import HEATING_MODE_AUTO
from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_coordinator

# 2025-06-19 Thursday ... 2025-06-23 Monday
THU = datetime(2025, 6, 19, 12, 0, tzinfo=timezone.utc)
_WEATHER = {"temperature": 10.0}


def _coord(start, **cfg):
    clock = FakeClock(start=start)
    return make_coordinator(cfg, clock=clock), clock


def _label(when, **cfg):
    return schedule.current_schedule_period(when, cfg)


class TestDisabledScheduleHasOneStableLabel:
    @pytest.mark.parametrize("day", range(0, 7))
    @pytest.mark.parametrize("hour", [0, 5, 7, 12, 21, 23])
    def test_same_label_on_every_day_and_hour(self, day, hour):
        when = datetime(2025, 6, 16 + day, hour, 30, tzinfo=timezone.utc)
        assert _label(when, schedule_enabled=False) == "always_comfort"

    def test_thursday_friday_saturday_do_not_change_it(self):
        labels = {_label(THU + timedelta(days=d), schedule_enabled=False) for d in range(3)}
        assert labels == {"always_comfort"}

    def test_sunday_to_monday_does_not_change_it(self):
        sunday = datetime(2025, 6, 22, 23, 59, tzinfo=timezone.utc)
        monday = sunday + timedelta(minutes=2)
        assert _label(sunday, schedule_enabled=False) == _label(monday, schedule_enabled=False)

    def test_a_week_of_hourly_samples_has_exactly_one_label(self):
        labels = {
            _label(THU + timedelta(hours=h), schedule_enabled=False) for h in range(24 * 7)
        }
        assert labels == {"always_comfort"}


class TestEnabledScheduleKeepsItsLabels:
    def test_weekday_and_weekend_labels_are_unchanged(self):
        assert _label(datetime(2025, 6, 19, 12, tzinfo=timezone.utc)) == "wd_comfort"
        assert _label(datetime(2025, 6, 21, 12, tzinfo=timezone.utc)) == "we_comfort"
        assert _label(datetime(2025, 6, 19, 23, tzinfo=timezone.utc)) == "wd_night"
        assert _label(datetime(2025, 6, 22, 23, 30, tzinfo=timezone.utc)) == "we_night"

    def test_enabled_schedule_still_changes_label_at_slot_boundaries(self):
        before = _label(datetime(2025, 6, 19, 5, 59, tzinfo=timezone.utc))
        after = _label(datetime(2025, 6, 19, 6, 0, tzinfo=timezone.utc))
        assert before != after

    def test_an_explicit_true_flag_behaves_like_the_default(self):
        when = datetime(2025, 6, 21, 12, tzinfo=timezone.utc)
        assert _label(when, schedule_enabled=True) == _label(when)

    def test_the_observation_context_label_is_still_comfort_when_disabled(self):
        coord, _ = _coord(THU, schedule_enabled=False)
        assert coord._schedule_period({"mode": HEATING_MODE_AUTO}, coord.zone_cfg) == "comfort"


class TestOverrideExpiry:
    async def _step(self, coord, clock, to):
        clock.set_utc(to)
        return await coord._compute_recommendation(coord.zone_cfg, _WEATHER, HEATING_MODE_AUTO)

    async def test_override_survives_every_day_boundary_with_the_schedule_off(self):
        coord, clock = _coord(THU, schedule_enabled=False)
        coord.set_override(23.0)
        for hours in (12, 36, 60, 84, 108, 132):                 # Fri, Sat, Sun, Mon, ...
            result = await self._step(coord, clock, THU + timedelta(hours=hours))
            assert coord.get_override() == 23.0, hours
            assert result["override_active"] is True

    async def test_override_expires_at_the_real_slot_change_with_the_schedule_on(self):
        start = datetime(2025, 6, 19, 18, 0, tzinfo=timezone.utc)    # Thursday evening, comfort
        coord, clock = _coord(start)
        coord.set_override(23.0)
        await self._step(coord, clock, datetime(2025, 6, 19, 21, 0, tzinfo=timezone.utc))
        assert coord.get_override() == 23.0                          # same comfort slot
        await self._step(coord, clock, datetime(2025, 6, 19, 22, 30, tzinfo=timezone.utc))
        assert coord.get_override() is None                          # night began

    async def test_an_override_set_in_the_evening_lasts_through_midnight_on_a_day_type_change(self):
        start = datetime(2025, 6, 20, 22, 30, tzinfo=timezone.utc)   # Friday night
        coord, clock = _coord(start)
        coord.set_override(19.0)
        await self._step(coord, clock, datetime(2025, 6, 21, 0, 30, tzinfo=timezone.utc))
        assert coord.get_override() == 19.0                          # Saturday 00:30, same night
        await self._step(coord, clock, datetime(2025, 6, 21, 8, 30, tzinfo=timezone.utc))
        assert coord.get_override() is None                          # weekend morning: new slot
