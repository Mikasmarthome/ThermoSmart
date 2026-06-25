"""Tests for Learning Engine weight functions.

_time_weight() and _thermal_weight() are module-level pure functions.
They take Python dicts and datetime objects — no HA fixtures required.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning_engine import (
    _time_weight,
    _thermal_weight,
)


# ── _time_weight ──────────────────────────────────────────────────────────────

class TestTimeWeight:
    _NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_fresh_observation_weight_is_one(self):
        """Observation from 'now' → weight = 1.0."""
        ts = self._NOW.isoformat()
        weight = _time_weight(ts, self._NOW, halflife_days=180)
        assert weight == pytest.approx(1.0, abs=0.001)

    def test_weight_at_one_time_constant(self):
        """At age == time-constant (180 days), weight = exp(-1) ≈ 0.368.

        The function uses exp(-age/τ) with a time-constant τ, not a true
        mathematical half-life. At age=τ, weight = 1/e ≈ 0.368.
        """
        past = self._NOW - timedelta(days=180)
        weight = _time_weight(past.isoformat(), self._NOW, halflife_days=180)
        assert weight == pytest.approx(math.exp(-1), abs=0.001)

    def test_exp_decay_formula(self):
        """Verify weight = exp(-age/τ) at two different ages."""
        for age_days in [30, 90, 360]:
            past = self._NOW - timedelta(days=age_days)
            weight = _time_weight(past.isoformat(), self._NOW, halflife_days=180)
            expected = math.exp(-age_days / 180)
            assert weight == pytest.approx(expected, abs=0.001), (
                f"Wrong weight at age={age_days}d: got {weight}, expected {expected}"
            )

    def test_older_observation_has_lower_weight(self):
        """Weight is strictly monotonically decreasing with age."""
        w_recent = _time_weight((self._NOW - timedelta(days=30)).isoformat(), self._NOW)
        w_old = _time_weight((self._NOW - timedelta(days=300)).isoformat(), self._NOW)
        assert w_recent > w_old

    def test_invalid_timestamp_string_returns_zero(self):
        assert _time_weight("not-a-date", self._NOW) == 0.0

    def test_empty_string_timestamp_returns_zero(self):
        assert _time_weight("", self._NOW) == 0.0

    def test_none_timestamp_returns_zero(self):
        assert _time_weight(None, self._NOW) == 0.0  # type: ignore[arg-type]

    def test_weight_is_always_positive(self):
        """Even ancient observations produce a positive weight (never zero or negative)."""
        ancient = datetime(2010, 1, 1, tzinfo=timezone.utc).isoformat()
        weight = _time_weight(ancient, self._NOW)
        assert weight > 0.0

    def test_custom_halflife_shorter_decays_faster(self):
        """Shorter halflife → faster decay at the same age."""
        past = self._NOW - timedelta(days=30)
        w_short = _time_weight(past.isoformat(), self._NOW, halflife_days=30)
        w_long = _time_weight(past.isoformat(), self._NOW, halflife_days=180)
        assert w_short < w_long

    def test_custom_halflife_longer_decays_slower(self):
        """Weight at 30-day halflife < weight at 360-day halflife for the same age."""
        past = self._NOW - timedelta(days=60)
        w_slow = _time_weight(past.isoformat(), self._NOW, halflife_days=360)
        w_fast = _time_weight(past.isoformat(), self._NOW, halflife_days=60)
        assert w_slow > w_fast

    def test_returns_float(self):
        ts = self._NOW.isoformat()
        result = _time_weight(ts, self._NOW)
        assert isinstance(result, float)


# ── _thermal_weight ───────────────────────────────────────────────────────────

class TestThermalWeight:
    """Tests for multi-factor thermal similarity weighting.

    _thermal_weight(obs, conditions, now) compares an observation's
    outdoor conditions to the current conditions and returns a 0–1 weight.
    """

    _NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def _obs(
        self,
        outdoor_temp: float = 10.0,
        wind: float = 0.0,
        solar: float = 0.0,
        humidity: float = 0.0,
        days_ago: int = 0,
    ) -> dict:
        ts = (self._NOW - timedelta(days=days_ago)).isoformat()
        return {
            "ts": ts,
            "outdoor_temp": outdoor_temp,
            "wind_speed": wind,
            "solar_radiation": solar,
            "outdoor_humidity": humidity,
        }

    def _cond(
        self,
        outdoor_temp: float = 10.0,
        wind: float = 0.0,
        solar: float = 0.0,
        humidity: float = 0.0,
    ) -> dict:
        return {
            "outdoor_temp": outdoor_temp,
            "wind_speed": wind,
            "solar_radiation": solar,
            "outdoor_humidity": humidity,
        }

    def test_identical_conditions_returns_near_one(self):
        """Obs from today with identical conditions → weight close to 1.0."""
        obs = self._obs(outdoor_temp=10.0, days_ago=0)
        weight = _thermal_weight(obs, self._cond(outdoor_temp=10.0), self._NOW)
        assert weight > 0.95

    def test_very_different_outdoor_temp_returns_zero(self):
        """25°C outdoor difference exceeds the similarity cut-off → weight = 0.0."""
        obs = self._obs(outdoor_temp=10.0)
        weight = _thermal_weight(obs, self._cond(outdoor_temp=35.0), self._NOW)
        assert weight == pytest.approx(0.0, abs=0.01)

    def test_closer_outdoor_temp_has_higher_weight(self):
        """Closer outdoor temperature → higher similarity weight."""
        obs_close = self._obs(outdoor_temp=10.0)
        obs_far = self._obs(outdoor_temp=20.0)
        cond = self._cond(outdoor_temp=10.0)
        w_close = _thermal_weight(obs_close, cond, self._NOW)
        w_far = _thermal_weight(obs_far, cond, self._NOW)
        assert w_close > w_far

    def test_older_observation_has_lower_weight(self):
        """More recent observations (smaller age) receive higher weight."""
        cond = self._cond(outdoor_temp=10.0)
        obs_recent = self._obs(outdoor_temp=10.0, days_ago=10)
        obs_old = self._obs(outdoor_temp=10.0, days_ago=365)
        w_recent = _thermal_weight(obs_recent, cond, self._NOW)
        w_old = _thermal_weight(obs_old, cond, self._NOW)
        assert w_recent > w_old

    def test_invalid_timestamp_returns_zero(self):
        obs = {"ts": "not-a-date", "outdoor_temp": 10.0}
        result = _thermal_weight(obs, self._cond(), self._NOW)
        assert result == 0.0

    def test_missing_ts_key_returns_zero(self):
        obs = {"outdoor_temp": 10.0}  # no 'ts' key at all
        result = _thermal_weight(obs, self._cond(), self._NOW)
        assert result == 0.0

    def test_weight_always_between_0_and_1(self):
        """Weight must always be in [0.0, 1.0] for any sane input."""
        cases = [
            (self._obs(outdoor_temp=10.0), self._cond(outdoor_temp=10.0)),
            (self._obs(outdoor_temp=10.0, days_ago=500), self._cond(outdoor_temp=10.0)),
            (self._obs(outdoor_temp=10.0, wind=15.0), self._cond(outdoor_temp=10.0, wind=0.0)),
            (self._obs(outdoor_temp=10.0, solar=500.0), self._cond(outdoor_temp=10.0, solar=100.0)),
        ]
        for obs, cond in cases:
            w = _thermal_weight(obs, cond, self._NOW)
            assert 0.0 <= w <= 1.0, f"Weight {w} out of range for obs={obs}"

    def test_wind_difference_reduces_weight(self):
        """Higher wind difference → lower similarity."""
        obs_no_wind = self._obs(outdoor_temp=10.0, wind=0.0)
        obs_high_wind = self._obs(outdoor_temp=10.0, wind=15.0)
        cond_no_wind = self._cond(outdoor_temp=10.0, wind=0.0)
        w_match = _thermal_weight(obs_no_wind, cond_no_wind, self._NOW)
        w_mismatch = _thermal_weight(obs_high_wind, cond_no_wind, self._NOW)
        assert w_match > w_mismatch

    def test_returns_float(self):
        obs = self._obs(outdoor_temp=10.0)
        result = _thermal_weight(obs, self._cond(outdoor_temp=10.0), self._NOW)
        assert isinstance(result, float)

    def test_result_is_rounded_to_6_decimals(self):
        """Result is rounded to 6 decimal places."""
        obs = self._obs(outdoor_temp=10.0)
        result = _thermal_weight(obs, self._cond(outdoor_temp=10.0), self._NOW)
        assert result == round(result, 6)
