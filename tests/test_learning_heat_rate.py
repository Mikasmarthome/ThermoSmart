"""Characterization tests for LearningEngine heat-rate & heat-loss logic.

Covers (Welle 4a):
  _estimate_heat_rate             – cold-start physics fallback
  _learned_heat_rate_multifactor  – approach 1 (normalised) vs. approach 2
  _get_heat_rate                  – confidence gate & fallback selection
  get_heat_loss_rate / _get_avg_cool_rate – heat-loss getters

These freeze current behaviour, including the `x or default` falsy quirks
(e.g. an outdoor temperature reported as exactly 0.0 is treated as 10.0).
Clock frozen via FakeClock injection for methods that use _now_local().
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning_engine import LearningEngine

from tests.helpers import make_learning_engine


FIXED_NOW = datetime(2025, 6, 16, 12, 0, 0, tzinfo=timezone.utc)


def engine_at(now=FIXED_NOW):
    """Return a LearningEngine with clock frozen at *now*."""
    return make_learning_engine(clock=FakeClock(start=now))


def make_obs(
    *,
    heat_rate=None,
    norm_heat_rate=None,
    cool_rate=None,
    outdoor_temp=10.0,
    wind=0.0,
    solar=0.0,
    humidity=0.0,
    days_ago=0,
    now=FIXED_NOW,
) -> dict:
    obs = {
        "ts": (now - timedelta(days=days_ago)).isoformat(),
        "outdoor_temp": outdoor_temp,
        "wind_speed": wind,
        "solar_radiation": solar,
        "outdoor_humidity": humidity,
    }
    if heat_rate is not None:
        obs["heat_rate"] = heat_rate
    if norm_heat_rate is not None:
        obs["norm_heat_rate"] = norm_heat_rate
    if cool_rate is not None:
        obs["cool_rate"] = cool_rate
    return obs


# ── _estimate_heat_rate: physics base table ──────────────────────────────────

class TestEstimateHeatRateBaseTable:
    def setup_method(self):
        self.e = make_learning_engine()

    @pytest.mark.parametrize(
        "outdoor,expected_base",
        [
            (-25.0, 0.018),   # < -20
            (-15.0, 0.026),   # < -10
            (-7.0, 0.035),    # < -5
            (-2.0, 0.042),    # < 0
            (2.0, 0.050),     # < 5
            (7.0, 0.058),     # < 10
            (12.0, 0.065),    # < 15
            (20.0, 0.072),    # >= 15
        ],
    )
    def test_base_rate_by_outdoor_temp(self, outdoor, expected_base):
        rate = self.e._estimate_heat_rate({"temperature": outdoor})
        assert rate == pytest.approx(expected_base, abs=1e-6)

    @pytest.mark.parametrize(
        "outdoor,expected_base",
        [
            (-20.0, 0.026),   # boundary: -20 is NOT < -20 → next bucket
            (-10.0, 0.035),
            (-5.0, 0.042),
            (0.0001, 0.050),  # tiny positive avoids the falsy-zero quirk
            (5.0, 0.058),
            (10.0, 0.065),
            (15.0, 0.072),
        ],
    )
    def test_base_rate_boundaries(self, outdoor, expected_base):
        rate = self.e._estimate_heat_rate({"temperature": outdoor})
        assert rate == pytest.approx(expected_base, abs=1e-6)


class TestEstimateHeatRateQuirks:
    def setup_method(self):
        self.e = make_learning_engine()

    def test_outdoor_exactly_zero_is_treated_as_ten(self):
        # `weather_data.get("temperature") or 10.0` → 0.0 is falsy → 10.0 → 0.065
        assert self.e._estimate_heat_rate({"temperature": 0.0}) == pytest.approx(0.065, abs=1e-6)

    def test_missing_temperature_defaults_to_ten(self):
        assert self.e._estimate_heat_rate({}) == pytest.approx(0.065, abs=1e-6)

    def test_none_temperature_defaults_to_ten(self):
        assert self.e._estimate_heat_rate({"temperature": None}) == pytest.approx(0.065, abs=1e-6)


class TestEstimateHeatRateWindSolar:
    def setup_method(self):
        self.e = make_learning_engine()

    def test_wind_below_five_no_effect(self):
        base = self.e._estimate_heat_rate({"temperature": 2.0})
        windy = self.e._estimate_heat_rate({"temperature": 2.0, "wind_speed": 5.0})
        assert windy == pytest.approx(base, abs=1e-6)

    def test_wind_reduces_rate(self):
        # outdoor 2 → base 0.050; wind 15 → factor 1-min(0.5,0.2)=0.8 → 0.040
        rate = self.e._estimate_heat_rate({"temperature": 2.0, "wind_speed": 15.0})
        assert rate == pytest.approx(0.040, abs=1e-6)

    def test_wind_factor_capped_at_twenty_percent(self):
        # very high wind saturates the 0.20 reduction cap
        r_30 = self.e._estimate_heat_rate({"temperature": 2.0, "wind_speed": 30.0})
        r_100 = self.e._estimate_heat_rate({"temperature": 2.0, "wind_speed": 100.0})
        assert r_30 == pytest.approx(r_100, abs=1e-6)
        assert r_30 == pytest.approx(0.050 * 0.80, abs=1e-6)

    def test_solar_below_threshold_no_effect(self):
        base = self.e._estimate_heat_rate({"temperature": 20.0})
        sunny = self.e._estimate_heat_rate({"temperature": 20.0, "solar_radiation": 300.0})
        assert sunny == pytest.approx(base, abs=1e-6)

    def test_solar_boosts_rate(self):
        # outdoor 20 → base 0.072; solar 1300 → boost min(1.0,0.15)=0.15 → *1.15
        rate = self.e._estimate_heat_rate({"temperature": 20.0, "solar_radiation": 1300.0})
        assert rate == pytest.approx(0.072 * 1.15, abs=1e-6)

    def test_solar_boost_capped_at_fifteen_percent(self):
        r_1300 = self.e._estimate_heat_rate({"temperature": 20.0, "solar_radiation": 1300.0})
        r_5000 = self.e._estimate_heat_rate({"temperature": 20.0, "solar_radiation": 5000.0})
        assert r_1300 == pytest.approx(r_5000, abs=1e-6)


# ── _learned_heat_rate_multifactor ───────────────────────────────────────────

class TestLearnedHeatRateApproach1Normalised:
    def test_normalised_rate_used_with_enough_samples(self):
        e = engine_at()
        # 6 fresh, identical-condition obs with norm_heat_rate 0.005
        e._observations["z"] = [
            make_obs(norm_heat_rate=0.005, outdoor_temp=10.0) for _ in range(6)
        ]
        # target 21, outdoor 10 → curr_delta = 11 → predicted = 0.005 * 11
        rate = e._learned_heat_rate_multifactor("z", {"temperature": 10.0}, target=21.0)
        assert rate == pytest.approx(0.055, abs=1e-4)

    def test_below_min_samples_falls_through_to_approach2(self):
        e = engine_at()
        # only 3 norm obs (< LEARNING_MIN_SAMPLES) but 6 heat_rate obs
        e._observations["z"] = (
            [make_obs(norm_heat_rate=0.005) for _ in range(3)]
            + [make_obs(heat_rate=0.05) for _ in range(6)]
        )
        rate = e._learned_heat_rate_multifactor("z", {"temperature": 10.0}, target=21.0)
        # approach 2 weighted average of heat_rate = 0.05
        assert rate == pytest.approx(0.05, abs=1e-4)


class TestLearnedHeatRateApproach2Fallback:
    def test_no_target_uses_approach2(self):
        e = engine_at()
        e._observations["z"] = [make_obs(heat_rate=0.06) for _ in range(6)]
        rate = e._learned_heat_rate_multifactor("z", {"temperature": 10.0}, target=None)
        assert rate == pytest.approx(0.06, abs=1e-4)

    def test_insufficient_heat_rate_samples_returns_zero(self):
        e = engine_at()
        e._observations["z"] = [make_obs(heat_rate=0.06) for _ in range(3)]
        rate = e._learned_heat_rate_multifactor("z", {"temperature": 10.0}, target=None)
        assert rate == 0.0

    def test_dissimilar_conditions_filtered_out(self):
        e = engine_at()
        # obs at outdoor 40, current 10 → thermal weight ~0 → all filtered
        e._observations["z"] = [
            make_obs(heat_rate=0.06, outdoor_temp=40.0) for _ in range(6)
        ]
        rate = e._learned_heat_rate_multifactor("z", {"temperature": 10.0}, target=None)
        assert rate == 0.0

    def test_empty_observations_returns_zero(self, monkeypatch):
        e = engine_at()
        rate = e._learned_heat_rate_multifactor("z", {"temperature": 10.0}, target=None)
        assert rate == 0.0


# ── _get_heat_rate: confidence gate + fallback ───────────────────────────────

class TestGetHeatRateGate:
    def test_low_confidence_uses_physics_estimate(self, monkeypatch):
        e = engine_at()
        e._observations["z"] = [make_obs(heat_rate=0.99) for _ in range(10)]
        e._confidence["z"] = 0.1  # below 0.3 gate
        rate = e._get_heat_rate("z", {"temperature": 2.0})
        # falls back to physics estimate for outdoor 2 → 0.050, NOT the learned 0.99
        assert rate == pytest.approx(0.050, abs=1e-6)

    def test_high_confidence_uses_learned_rate(self):
        e = engine_at()
        e._observations["z"] = [make_obs(heat_rate=0.08) for _ in range(8)]
        e._confidence["z"] = 0.9
        rate = e._get_heat_rate("z", {"temperature": 10.0}, target=None)
        assert rate == pytest.approx(0.08, abs=1e-4)

    def test_disabled_zone_uses_physics_estimate(self, monkeypatch):
        e = engine_at()
        e._observations["z"] = [make_obs(heat_rate=0.08) for _ in range(8)]
        e._confidence["z"] = 0.9
        e.set_zone_enabled("z", False)
        rate = e._get_heat_rate("z", {"temperature": 2.0})
        assert rate == pytest.approx(0.050, abs=1e-6)

    def test_learned_zero_falls_back_to_estimate(self, monkeypatch):
        e = engine_at()
        # high confidence but no usable obs → learned 0 → estimate
        e._confidence["z"] = 0.9
        rate = e._get_heat_rate("z", {"temperature": 2.0}, target=None)
        assert rate == pytest.approx(0.050, abs=1e-6)


# ── Heat-loss getters ────────────────────────────────────────────────────────

class TestHeatLossGetters:
    def test_get_heat_loss_rate_prefers_ema(self):
        e = make_learning_engine()
        e._heat_loss_ema["z"] = 0.012
        assert e.get_heat_loss_rate("z") == 0.012

    def test_get_heat_loss_rate_none_when_empty(self):
        e = make_learning_engine()
        assert e.get_heat_loss_rate("z") is None

    def test_get_heat_loss_rate_uses_cool_rate_samples(self):
        e = make_learning_engine()
        e._observations["z"] = [
            make_obs(cool_rate=0.01),
            make_obs(cool_rate=0.03),
        ]
        assert e.get_heat_loss_rate("z") == pytest.approx(0.02, abs=1e-6)

    def test_get_heat_loss_rate_zero_ema_falls_through_to_samples(self):
        # QUIRK: get_heat_loss_rate uses `if rate:` → ema 0.0 is falsy → samples
        e = make_learning_engine()
        e._heat_loss_ema["z"] = 0.0
        e._observations["z"] = [make_obs(cool_rate=0.04)]
        assert e.get_heat_loss_rate("z") == pytest.approx(0.04, abs=1e-6)

    def test_avg_cool_rate_prefers_ema_even_when_zero(self):
        # QUIRK: _get_avg_cool_rate uses `in` → returns ema 0.0 (diverges from getter)
        e = make_learning_engine()
        e._heat_loss_ema["z"] = 0.0
        e._observations["z"] = [make_obs(cool_rate=0.04)]
        assert e._get_avg_cool_rate("z") == 0.0

    def test_avg_cool_rate_zero_when_nothing_learned(self):
        e = make_learning_engine()
        assert e._get_avg_cool_rate("z") == 0.0

    def test_avg_cool_rate_uses_samples_without_ema(self):
        e = make_learning_engine()
        e._observations["z"] = [
            make_obs(cool_rate=0.02),
            make_obs(cool_rate=0.04),
        ]
        assert e._get_avg_cool_rate("z") == pytest.approx(0.03, abs=1e-6)
