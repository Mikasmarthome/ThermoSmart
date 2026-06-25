"""Characterization tests for LearningEngine TRV / window / boost paths (Welle 4b).

Covers:
  async_observe_trv_setpoint  – excess>0.3 gate, heat_rate/efficiency, burst guard
  async_observe_window_cooling – noise floor 0.5, duration guard, recording
  update_boost_factor          – overshoot/slow adjustment + documented NO-SAVE quirk
  get_trv_stats / get_window_cooling_rate

The debounced TRV save (async_call_later) is a scheduling side-effect, not under
test, so _schedule_debounced_save is stubbed. async_save is mocked.
Clock is frozen via FakeClock injection (no dt_util.now patching).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_learning_engine


FIXED_NOW = datetime(2025, 6, 16, 12, 0, 0, tzinfo=timezone.utc)
WEATHER = {"temperature": 8.0, "wind_speed": 3.0, "solar_radiation": 100.0}


def make_engine(monkeypatch=None, now=FIXED_NOW):
    e = make_learning_engine(clock=FakeClock(start=now))
    e.async_save = AsyncMock()
    e._schedule_debounced_save = MagicMock()  # stub scheduling side-effect
    return e


# ── async_observe_trv_setpoint ───────────────────────────────────────────────

class TestObserveTrvSetpoint:
    async def test_recorded_with_efficiency(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_trv_setpoint(
            "z", trv_setpoint=22.0, indoor_temp=20.0, target=21.0,
            weather_data=WEATHER, heat_rate=0.1)
        obs = e._trv_observations["z"][0]
        assert obs["trv_setpoint"] == 22.0
        assert obs["indoor_temp"] == 20.0
        assert obs["setpoint_excess"] == pytest.approx(2.0)
        assert obs["heat_rate"] == pytest.approx(0.1)
        # efficiency = heat_rate / excess = 0.1 / 2.0
        assert obs["efficiency"] == pytest.approx(0.05, abs=1e-6)

    async def test_small_excess_rejected(self, monkeypatch):
        e = make_engine(monkeypatch)
        # excess 0.2 <= 0.3 → not recorded. (The exact 0.3 boundary is avoided
        # here: 20.3-20.0 is float 0.30000000000000071, just above the cutoff.)
        await e.async_observe_trv_setpoint(
            "z", trv_setpoint=20.2, indoor_temp=20.0, target=21.0,
            weather_data=WEATHER, heat_rate=0.1)
        assert e._trv_observations["z"] == []

    async def test_excess_just_above_threshold_recorded(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_trv_setpoint(
            "z", trv_setpoint=20.4, indoor_temp=20.0, target=21.0,
            weather_data=WEATHER, heat_rate=0.1)
        assert len(e._trv_observations["z"]) == 1

    async def test_none_heat_rate_recorded_without_efficiency(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_trv_setpoint(
            "z", trv_setpoint=22.0, indoor_temp=20.0, target=21.0,
            weather_data=WEATHER, heat_rate=None)
        obs = e._trv_observations["z"][0]
        assert "efficiency" not in obs
        assert "heat_rate" not in obs

    async def test_non_positive_heat_rate_rejected(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_trv_setpoint(
            "z", trv_setpoint=22.0, indoor_temp=20.0, target=21.0,
            weather_data=WEATHER, heat_rate=0.0)
        assert e._trv_observations["z"] == []

    async def test_disabled_zone_records_nothing(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.set_zone_enabled("z", False)
        await e.async_observe_trv_setpoint(
            "z", 22.0, 20.0, 21.0, WEATHER, heat_rate=0.1)
        assert e._trv_observations["z"] == []

    async def test_weather_fields_mapped(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_trv_setpoint(
            "z", 22.0, 20.0, 21.0, WEATHER, heat_rate=0.1)
        obs = e._trv_observations["z"][0]
        assert obs["outdoor_temp"] == 8.0
        assert obs["wind_speed"] == 3.0
        assert obs["solar_radiation"] == 100.0

    async def test_burst_guard_rejects_identical_rapid_observation(self, monkeypatch):
        e = make_engine(monkeypatch)
        # seed an existing obs with the same frozen timestamp (delta < 5s)
        e._trv_observations["z"].append({
            "ts": FIXED_NOW.isoformat(),
            "trv_setpoint": 22.0, "indoor_temp": 20.0, "target": 21.0,
        })
        await e.async_observe_trv_setpoint(
            "z", 22.0, 20.0, 21.0, WEATHER, heat_rate=0.1)
        assert len(e._trv_observations["z"]) == 1  # rejected by burst guard

    async def test_schedule_save_called(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_trv_setpoint(
            "z", 22.0, 20.0, 21.0, WEATHER, heat_rate=0.1)
        e._schedule_debounced_save.assert_called_once()


# ── async_observe_window_cooling ─────────────────────────────────────────────

class TestObserveWindowCooling:
    async def test_recorded(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_window_cooling(
            "z", duration_min=10.0, temp_at_open=21.0, temp_at_close=19.0,
            weather_data=WEATHER)
        obs = e._window_cooling_obs["z"][0]
        assert obs["duration_min"] == 10.0
        assert obs["temp_drop"] == pytest.approx(2.0)
        # cooling_rate = 2.0 / 10.0
        assert obs["cooling_rate_per_min"] == pytest.approx(0.2, abs=1e-4)
        assert obs["indoor_start_temp"] == 21.0
        e.async_save.assert_awaited_once()

    async def test_short_duration_rejected(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_window_cooling(
            "z", duration_min=0.5, temp_at_open=21.0, temp_at_close=19.0,
            weather_data=WEATHER)
        assert e._window_cooling_obs["z"] == []

    async def test_non_positive_drop_rejected(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_window_cooling(
            "z", duration_min=10.0, temp_at_open=20.0, temp_at_close=20.5,
            weather_data=WEATHER)
        assert e._window_cooling_obs["z"] == []

    async def test_noise_floor_rejects_small_drop(self, monkeypatch):
        e = make_engine(monkeypatch)
        # drop 0.4 < 0.5 noise floor → rejected
        await e.async_observe_window_cooling(
            "z", duration_min=10.0, temp_at_open=21.0, temp_at_close=20.6,
            weather_data=WEATHER)
        assert e._window_cooling_obs["z"] == []

    async def test_drop_at_noise_floor_recorded(self, monkeypatch):
        e = make_engine(monkeypatch)
        # drop exactly 0.5 → not < 0.5 → recorded
        await e.async_observe_window_cooling(
            "z", duration_min=10.0, temp_at_open=21.0, temp_at_close=20.5,
            weather_data=WEATHER)
        assert len(e._window_cooling_obs["z"]) == 1

    async def test_weather_fields_mapped(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe_window_cooling(
            "z", duration_min=10.0, temp_at_open=21.0, temp_at_close=19.0,
            weather_data=WEATHER)
        obs = e._window_cooling_obs["z"][0]
        assert obs["outdoor_temp"] == 8.0
        assert obs["wind_speed"] == 3.0


# ── update_boost_factor ──────────────────────────────────────────────────────

class TestUpdateBoostFactor:
    def test_update_boost_factor_is_noop(self, monkeypatch):
        # update_boost_factor superseded by LE 2.0 BoostModel — now a no-op
        e = make_engine(monkeypatch)
        e.update_boost_factor("z", overshot=True)
        assert e.get_boost_factor("z") == pytest.approx(1.0)  # unchanged

    def test_update_noop_slow_too(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.update_boost_factor("z", overshot=False, slow=True)
        assert e.get_boost_factor("z") == pytest.approx(1.0)  # unchanged

    def test_noop_does_not_modify_stored_value(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._boost_factors["z"] = 0.52
        e.update_boost_factor("z", overshot=True)
        # no-op: stored value unchanged
        assert e.get_boost_factor("z") == pytest.approx(0.52)

    def test_noop_stored_high_value_unchanged(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._boost_factors["z"] = 1.96
        e.update_boost_factor("z", overshot=False, slow=True)
        # no-op: stored value unchanged
        assert e.get_boost_factor("z") == pytest.approx(1.96)

    def test_neither_flag_leaves_factor(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._boost_factors["z"] = 1.1
        e.update_boost_factor("z", overshot=False, slow=False)
        assert e.get_boost_factor("z") == 1.1

    def test_disabled_zone_no_change(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.set_zone_enabled("z", False)
        e.update_boost_factor("z", overshot=True)
        assert e.get_boost_factor("z") == 1.0  # default, untouched

    def test_no_save_quirk(self, monkeypatch):
        # DOCUMENTED QUIRK: update_boost_factor mutates in-memory state but never
        # triggers a save → the new factor is lost on restart until another path
        # persists. Frozen here as current behaviour, NOT fixed.
        e = make_engine(monkeypatch)
        e.update_boost_factor("z", overshot=True)
        e.async_save.assert_not_called()


# ── get_trv_stats / get_window_cooling_rate ──────────────────────────────────

class TestTrvStats:
    def test_empty_stats(self, monkeypatch):
        e = make_engine(monkeypatch)
        stats = e.get_trv_stats("z")
        assert stats["trv_observations"] == 0
        assert stats["window_observations"] == 0
        assert stats["avg_setpoint_efficiency"] is None
        assert stats["avg_window_cooling_rate"] is None

    def test_avg_efficiency_computed(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._trv_observations["z"] = [
            {"efficiency": 0.04},
            {"efficiency": 0.06},
            {"trv_setpoint": 22.0},  # no efficiency → ignored
        ]
        stats = e.get_trv_stats("z")
        assert stats["trv_observations"] == 3
        assert stats["avg_setpoint_efficiency"] == pytest.approx(0.05, abs=1e-6)

    def test_avg_window_cooling_rate(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._window_cooling_obs["z"] = [
            {"cooling_rate_per_min": 0.10},
            {"cooling_rate_per_min": 0.20},
        ]
        stats = e.get_trv_stats("z")
        assert stats["avg_window_cooling_rate"] == pytest.approx(0.15, abs=1e-4)


class TestGetWindowCoolingRate:
    def test_none_below_two_observations(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._window_cooling_obs["z"] = [
            {"ts": FIXED_NOW.isoformat(), "cooling_rate_per_min": 0.2,
             "outdoor_temp": 8.0, "wind_speed": 3.0},
        ]
        assert e.get_window_cooling_rate("z", WEATHER) is None

    def test_weighted_rate_for_similar_conditions(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._window_cooling_obs["z"] = [
            {"ts": FIXED_NOW.isoformat(), "cooling_rate_per_min": 0.10,
             "outdoor_temp": 8.0, "wind_speed": 3.0},
            {"ts": FIXED_NOW.isoformat(), "cooling_rate_per_min": 0.20,
             "outdoor_temp": 8.0, "wind_speed": 3.0},
        ]
        # identical conditions → equal weights → ~0.15
        rate = e.get_window_cooling_rate("z", WEATHER)
        assert rate == pytest.approx(0.15, abs=1e-3)

    def test_none_when_all_dissimilar(self, monkeypatch):
        e = make_engine(monkeypatch)
        # obs at outdoor 40 vs current 8 → weight below floor → filtered out
        e._window_cooling_obs["z"] = [
            {"ts": FIXED_NOW.isoformat(), "cooling_rate_per_min": 0.10,
             "outdoor_temp": 40.0, "wind_speed": 3.0},
            {"ts": FIXED_NOW.isoformat(), "cooling_rate_per_min": 0.20,
             "outdoor_temp": 40.0, "wind_speed": 3.0},
        ]
        assert e.get_window_cooling_rate("z", WEATHER) is None
