"""Characterization tests for LearningEngine heating-session outcome logic (Welle 4b).

Covers:
  update_heating_session  – start / continue / reached / timeout / interrupted
  _finalize_session       – 3-min floor, score write-back into trv_observations,
                            outcome_log cap (200)
  get_outcome_stats       – ts/obs averages

Clock is frozen via FakeClock injection; session start times are injected as
UTC-aware ISO strings so elapsed durations are deterministic. Persistence is mocked.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_learning_engine


FIXED_NOW = datetime(2025, 6, 16, 12, 0, 0, tzinfo=timezone.utc)
WEATHER = {"temperature": 5.0, "wind_speed": 2.0, "solar_radiation": 0.0,
           "humidity": 60.0, "rain": 0.0}


def make_engine(monkeypatch=None, now=FIXED_NOW):
    e = make_learning_engine(clock=FakeClock(start=now))
    # _finalize_session fires save via hass.async_create_task(self.async_save()).
    # A sync MagicMock for async_save avoids creating an un-awaited coroutine.
    e.async_save = MagicMock()
    e._hass.async_create_task = lambda *a, **k: None
    return e


def seed_session(e, *, start_temp, target, minutes_ago, controller="ts",
                 expected_minutes=0, peak=None, weather=None, now=FIXED_NOW):
    """Inject an active heating session that started `minutes_ago` minutes ago."""
    e._heating_sessions["z"] = {
        "start_time": (now - timedelta(minutes=minutes_ago)).isoformat(),
        "start_temp": start_temp,
        "target": target,
        "controller": controller,
        "weather": weather or {
            "outdoor_temp": WEATHER["temperature"],
            "wind_speed": WEATHER["wind_speed"],
            "solar_radiation": WEATHER["solar_radiation"],
            "outdoor_humidity": WEATHER["humidity"],
            "rain": WEATHER["rain"],
        },
        "peak_temp": peak if peak is not None else start_temp,
        "expected_minutes": expected_minutes,
    }


# ── Session start ────────────────────────────────────────────────────────────

class TestSessionStart:
    def test_session_starts_when_below_target(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.update_heating_session("z", current_temp=18.0, target=21.0,
                                 is_active_control=True, weather_data=WEATHER,
                                 expected_minutes=40)
        s = e._heating_sessions["z"]
        assert s["start_temp"] == 18.0
        assert s["target"] == 21.0
        assert s["controller"] == "ts"
        assert s["expected_minutes"] == 40

    def test_controller_obs_when_not_active_control(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.update_heating_session("z", 18.0, 21.0, is_active_control=False,
                                 weather_data=WEATHER)
        assert e._heating_sessions["z"]["controller"] == "obs"

    def test_no_session_when_within_tolerance(self, monkeypatch):
        e = make_engine(monkeypatch)
        # delta 0.3 <= tolerance 0.5 → no session
        e.update_heating_session("z", 20.7, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        assert "z" not in e._heating_sessions

    def test_peak_tracked_on_continuation(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_session(e, start_temp=18.0, target=21.0, minutes_ago=10, peak=18.0)
        e.update_heating_session("z", 19.5, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        assert e._heating_sessions["z"]["peak_temp"] == 19.5


# ── Reached / timeout / interrupted finalization ─────────────────────────────

class TestSessionFinalize:
    def test_reached_logs_outcome(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_session(e, start_temp=18.0, target=21.0, minutes_ago=30,
                     expected_minutes=40, peak=21.0)
        # current within tolerance → reached
        e.update_heating_session("z", 21.0, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        assert "z" not in e._heating_sessions
        log = e._outcome_log["z"]
        assert len(log) == 1
        assert log[0]["reason"] == "reached"
        assert 0.0 <= log[0]["outcome_score"] <= 1.0

    def test_timeout_after_90_minutes(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_session(e, start_temp=18.0, target=21.0, minutes_ago=95,
                     expected_minutes=40, peak=20.0)
        # still below target → continuation path detects timeout
        e.update_heating_session("z", 20.0, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        assert "z" not in e._heating_sessions
        assert e._outcome_log["z"][0]["reason"] == "timeout"

    def test_interrupted_when_target_none(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_session(e, start_temp=18.0, target=21.0, minutes_ago=20, peak=19.5)
        # target None (e.g. window opened) → interrupted
        e.update_heating_session("z", 19.0, None, is_active_control=True,
                                 weather_data=WEATHER)
        assert "z" not in e._heating_sessions
        assert e._outcome_log["z"][0]["reason"] == "interrupted"

    def test_interrupted_when_current_temp_none(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_session(e, start_temp=18.0, target=21.0, minutes_ago=20, peak=19.5)
        e.update_heating_session("z", None, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        assert e._outcome_log["z"][0]["reason"] == "interrupted"

    def test_no_session_no_finalize_on_none(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.update_heating_session("z", None, None, is_active_control=True,
                                 weather_data=WEATHER)
        assert e._outcome_log["z"] == []


# ── 3-minute floor ───────────────────────────────────────────────────────────

class TestThreeMinuteFloor:
    def test_session_below_three_minutes_dropped(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_session(e, start_temp=20.0, target=21.0, minutes_ago=2, peak=21.0)
        e.update_heating_session("z", 21.0, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        # too short → no outcome logged, session still removed
        assert "z" not in e._heating_sessions
        assert e._outcome_log["z"] == []

    def test_session_at_three_minutes_kept(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_session(e, start_temp=20.0, target=21.0, minutes_ago=4, peak=21.0)
        e.update_heating_session("z", 21.0, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        assert len(e._outcome_log["z"]) == 1


# ── Outcome details & score write-back ───────────────────────────────────────

class TestOutcomeDetails:
    def test_outcome_carries_weather_and_meta(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_session(e, start_temp=18.0, target=21.0, minutes_ago=30,
                     expected_minutes=40, peak=21.2, controller="ts")
        e.update_heating_session("z", 21.0, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        o = e._outcome_log["z"][0]
        assert o["start_temp"] == 18.0
        assert o["target"] == 21.0
        assert o["peak_temp"] == 21.2
        assert o["controller"] == "ts"
        assert o["expected_minutes"] == 40
        assert o["outdoor_temp"] == 5.0  # weather flattened into outcome

    def test_score_written_back_into_last_trv_obs(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._trv_observations["z"] = [{"ts": FIXED_NOW.isoformat(), "trv_setpoint": 22.0}]
        seed_session(e, start_temp=18.0, target=21.0, minutes_ago=30,
                     expected_minutes=40, peak=21.0)
        e.update_heating_session("z", 21.0, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        assert "outcome_score" in e._trv_observations["z"][-1]
        assert e._trv_observations["z"][-1]["outcome_score"] == e._outcome_log["z"][0]["outcome_score"]

    def test_outcome_log_capped_at_200(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._outcome_log["z"] = [{"outcome_score": 0.5, "controller": "ts"} for _ in range(200)]
        seed_session(e, start_temp=18.0, target=21.0, minutes_ago=30,
                     expected_minutes=40, peak=21.0)
        e.update_heating_session("z", 21.0, 21.0, is_active_control=True,
                                 weather_data=WEATHER)
        assert len(e._outcome_log["z"]) == 200  # capped, oldest dropped


# ── get_outcome_stats ────────────────────────────────────────────────────────

class TestOutcomeStats:
    def test_empty_returns_zero_sessions(self, monkeypatch):
        e = make_engine(monkeypatch)
        stats = e.get_outcome_stats("z")
        assert stats["sessions"] == 0
        assert stats["avg_score"] is None

    def test_stats_split_by_controller(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._outcome_log["z"] = [
            {"outcome_score": 1.0, "controller": "ts"},
            {"outcome_score": 0.5, "controller": "ts"},
            {"outcome_score": 0.8, "controller": "obs"},
        ]
        stats = e.get_outcome_stats("z")
        assert stats["sessions"] == 3
        assert stats["avg_score_%"] == pytest.approx((1.0 + 0.5 + 0.8) / 3 * 100, abs=0.1)
        assert stats["ts_avg_%"] == pytest.approx(75.0, abs=0.1)
        assert stats["obs_avg_%"] == pytest.approx(80.0, abs=0.1)
        assert stats["last_score_%"] == pytest.approx(80.0, abs=0.1)
        assert stats["last_controller"] == "obs"
