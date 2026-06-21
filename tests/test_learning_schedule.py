"""Characterization tests for LearningEngine schedule logic (Welle 4a).

Covers:
  _schedule_target          – weekday/weekend comfort-vs-night blocks
  _weighted_median          – weighted median over (value, weight) pairs
  _weighted_targets_for_hour – time-weighted target observations near an hour

These freeze the current behaviour. `_schedule_target` reads dt_util.now(),
so it is monkeypatched to fixed datetimes.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from custom_components.thermosmart import learning_engine as le
from custom_components.thermosmart.const import (
    CONF_SCHED_WD_MORNING,
    CONF_SCHED_WD_NIGHT,
    CONF_SCHED_WE_MORNING,
)
from tests.helpers import make_learning_engine


# 2025-06-16 is a Monday (weekday 0); 2025-06-21 is a Saturday (weekday 5).
MONDAY = datetime(2025, 6, 16)
SATURDAY = datetime(2025, 6, 21)


def set_now(monkeypatch, when):
    monkeypatch.setattr(le.dt_util, "now", lambda: when)


# ── _schedule_target ─────────────────────────────────────────────────────────

class TestScheduleTargetWeekday:
    def test_midday_returns_comfort(self, monkeypatch):
        set_now(monkeypatch, MONDAY.replace(hour=12))
        e = make_learning_engine()
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0) == 21.0

    def test_before_morning_returns_night(self, monkeypatch):
        set_now(monkeypatch, MONDAY.replace(hour=5))
        e = make_learning_engine()
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0) == 18.0

    def test_at_night_boundary_returns_night(self, monkeypatch):
        # 22:00 is the night boundary → cur >= night → night
        set_now(monkeypatch, MONDAY.replace(hour=22, minute=0))
        e = make_learning_engine()
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0) == 18.0

    def test_just_before_night_returns_comfort(self, monkeypatch):
        set_now(monkeypatch, MONDAY.replace(hour=21, minute=59))
        e = make_learning_engine()
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0) == 21.0

    def test_at_morning_boundary_returns_comfort(self, monkeypatch):
        # 06:00 → cur == morning → not < morning → comfort
        set_now(monkeypatch, MONDAY.replace(hour=6, minute=0))
        e = make_learning_engine()
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0) == 21.0


class TestScheduleTargetWeekend:
    def test_uses_later_weekend_morning(self, monkeypatch):
        # weekend default morning is 08:00 → 07:00 is still night
        set_now(monkeypatch, SATURDAY.replace(hour=7))
        e = make_learning_engine()
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0) == 18.0

    def test_weekend_midday_returns_comfort(self, monkeypatch):
        set_now(monkeypatch, SATURDAY.replace(hour=9))
        e = make_learning_engine()
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0) == 21.0


class TestScheduleTargetConfig:
    def test_custom_morning_time_respected(self, monkeypatch):
        set_now(monkeypatch, MONDAY.replace(hour=5, minute=30))
        e = make_learning_engine()
        cfg = {CONF_SCHED_WD_MORNING: "05:00", CONF_SCHED_WD_NIGHT: "22:00"}
        # with custom 05:00 morning, 05:30 is now comfort
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0,
                                  schedule_cfg=cfg) == 21.0

    def test_invalid_time_string_falls_back_to_default(self, monkeypatch):
        set_now(monkeypatch, MONDAY.replace(hour=5, minute=0))
        e = make_learning_engine()
        cfg = {CONF_SCHED_WD_MORNING: "not-a-time"}
        # fallback default 06:00 → 05:00 is night
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0,
                                  schedule_cfg=cfg) == 18.0

    def test_weekend_config_key_used_on_weekend(self, monkeypatch):
        set_now(monkeypatch, SATURDAY.replace(hour=7, minute=30))
        e = make_learning_engine()
        cfg = {CONF_SCHED_WE_MORNING: "07:00"}
        # custom weekend morning 07:00 → 07:30 is comfort
        assert e._schedule_target("z", comfort_temp=21.0, night_temp=18.0,
                                  schedule_cfg=cfg) == 21.0


# ── _weighted_median ─────────────────────────────────────────────────────────

class TestWeightedMedian:
    def setup_method(self):
        self.e = make_learning_engine()

    def test_empty_returns_default(self):
        assert self.e._weighted_median([]) == 21.0

    def test_single_value(self):
        assert self.e._weighted_median([(19.5, 1.0)]) == 19.5

    def test_equal_weights_returns_lower_middle(self):
        # cumulative >= total/2 on the first (lower) value due to >= test
        assert self.e._weighted_median([(20.0, 1.0), (22.0, 1.0)]) == 20.0

    def test_weight_shifts_median_up(self):
        assert self.e._weighted_median([(18.0, 1.0), (22.0, 3.0)]) == 22.0

    def test_weight_shifts_median_down(self):
        assert self.e._weighted_median([(18.0, 3.0), (22.0, 1.0)]) == 18.0

    def test_unsorted_input_is_sorted_internally(self):
        # same data, different order → same result
        a = self.e._weighted_median([(22.0, 1.0), (18.0, 3.0)])
        b = self.e._weighted_median([(18.0, 3.0), (22.0, 1.0)])
        assert a == b == 18.0


# ── _weighted_targets_for_hour ───────────────────────────────────────────────

class TestWeightedTargetsForHour:
    def _obs(self, *, target, hour, weekday, days_ago=0, now=MONDAY):
        ts = (now.replace(hour=hour) - timedelta(days=days_ago)).isoformat()
        return {"ts": ts, "target": target, "hour": hour, "weekday": weekday}

    def test_matching_hour_and_daytype_included(self):
        e = make_learning_engine()
        now = MONDAY.replace(hour=12)
        e._observations["z"] = [self._obs(target=21.0, hour=12, weekday=0)]
        result = e._weighted_targets_for_hour("z", now)
        assert len(result) == 1
        assert result[0][0] == 21.0
        assert result[0][1] > 0.0

    def test_adjacent_hour_included(self):
        e = make_learning_engine()
        now = MONDAY.replace(hour=12)
        e._observations["z"] = [self._obs(target=20.0, hour=11, weekday=0)]
        result = e._weighted_targets_for_hour("z", now)
        assert len(result) == 1

    def test_distant_hour_excluded(self):
        e = make_learning_engine()
        now = MONDAY.replace(hour=12)
        e._observations["z"] = [self._obs(target=20.0, hour=15, weekday=0)]
        assert e._weighted_targets_for_hour("z", now) == []

    def test_weekend_observation_excluded_on_weekday(self):
        e = make_learning_engine()
        now = MONDAY.replace(hour=12)  # weekday
        e._observations["z"] = [self._obs(target=20.0, hour=12, weekday=6)]
        assert e._weighted_targets_for_hour("z", now) == []

    def test_weekday_observation_excluded_on_weekend(self):
        e = make_learning_engine()
        now = SATURDAY.replace(hour=12)  # weekend
        e._observations["z"] = [self._obs(target=20.0, hour=12, weekday=2, now=SATURDAY)]
        assert e._weighted_targets_for_hour("z", now) == []

    def test_very_old_observation_excluded_by_weight_floor(self):
        e = make_learning_engine()
        now = MONDAY.replace(hour=12)
        # ~1000 days old → time weight < 0.01 → excluded
        e._observations["z"] = [self._obs(target=21.0, hour=12, weekday=0, days_ago=1000)]
        assert e._weighted_targets_for_hour("z", now) == []
