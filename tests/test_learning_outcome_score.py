"""Characterization tests for LearningEngine._calc_outcome_score.

`_calc_outcome_score` is a @staticmethod — fully pure, no hass / no Store.
These tests FREEZE the current scoring behaviour (Welle 4a) so a later LE 2.0
refactor can be proven behaviour-equivalent.  They assert *what the code does
today*, including quirks — not what it "should" do.

Score composition (current):
    Reached  40%  – did we reach target? (reason-dependent)
    Speed    35%  – minutes_taken vs. difficulty-adjusted expectation
    Accuracy 25%  – overshoot penalty
    × env_discount (solar + warm) at the very end, then clamp [0,1], round 3dp.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning_engine import LearningEngine


def score(**kw) -> float:
    """Call _calc_outcome_score with neutral defaults → baseline = 1.0.

    Baseline: reason='reached' (reached=1.0), ratio 0.5 (speed=1.0),
    no overshoot (accuracy=1.0), mild outdoor & no solar (discount=1.0).
    """
    params = dict(
        minutes_taken=30.0,
        expected_minutes=60.0,
        peak_temp=21.0,
        current_temp=21.0,
        target=21.0,
        start_temp=18.0,
        solar_radiation=0.0,
        outdoor_temp=5.0,
        wind_speed=0.0,
        outdoor_humidity=0.0,
        rain=0.0,
        reason="reached",
    )
    params.update(kw)
    return LearningEngine._calc_outcome_score(**params)


# ── Baseline & invariants ────────────────────────────────────────────────────

class TestBaselineAndInvariants:
    def test_perfect_session_scores_one(self):
        assert score() == 1.0

    def test_result_is_float(self):
        assert isinstance(score(), float)

    def test_result_rounded_to_3_decimals(self):
        s = score(peak_temp=21.7, minutes_taken=95.0, expected_minutes=60.0)
        assert s == round(s, 3)

    @pytest.mark.parametrize(
        "kw",
        [
            {},
            {"reason": "timeout", "current_temp": 17.0},
            {"reason": "interrupted", "peak_temp": 30.0},
            {"solar_radiation": 5000.0, "outdoor_temp": 50.0},
            {"peak_temp": 40.0, "minutes_taken": 5000.0, "expected_minutes": 30.0},
            {"start_temp": 25.0, "target": 21.0},
        ],
    )
    def test_score_always_within_unit_interval(self, kw):
        s = score(**kw)
        assert 0.0 <= s <= 1.0


# ── Reached component (40%) ──────────────────────────────────────────────────

class TestReachedComponent:
    def test_reason_reached_full_credit(self):
        # reached=1, speed=1, accuracy=1, discount=1 → 1.0
        assert score(reason="reached") == 1.0

    def test_delta_total_non_positive_is_always_reached(self):
        # start >= target → first branch wins regardless of reason
        assert score(reason="timeout", start_temp=22.0, target=21.0) == 1.0

    def test_timeout_partial_progress_half_credit(self):
        # delta_total=3, progressed 1.5 → reached=0.5
        # speed=1, accuracy=1 → raw = 0.5*0.4 + 0.35 + 0.25 = 0.8
        s = score(reason="timeout", start_temp=18.0, target=21.0,
                  current_temp=19.5, peak_temp=21.0)
        assert s == pytest.approx(0.8, abs=0.001)

    def test_timeout_progress_beyond_target_clamped_to_one(self):
        s = score(reason="timeout", start_temp=18.0, target=21.0,
                  current_temp=22.0, peak_temp=22.0)
        # reached clamps to 1.0; overshoot 1.0 → accuracy 0.5
        # raw = 0.4 + 0.35 + 0.5*0.25 = 0.875
        assert s == pytest.approx(0.875, abs=0.001)

    def test_timeout_negative_progress_clamped_to_zero(self):
        # current below start → reached = 0
        s = score(reason="timeout", start_temp=18.0, target=21.0,
                  current_temp=17.0, peak_temp=18.0)
        # accuracy: peak 18 < target → 1.0; speed 1 → raw = 0.35 + 0.25 = 0.6
        assert s == pytest.approx(0.6, abs=0.001)

    def test_interrupted_uses_peak_temp_not_current(self):
        # interrupted reached = (peak - start)/delta_total
        s = score(reason="interrupted", start_temp=18.0, target=21.0,
                  peak_temp=19.5, current_temp=10.0)
        # reached=0.5, speed=1, accuracy(peak 19.5<21)=1 → 0.8
        assert s == pytest.approx(0.8, abs=0.001)

    def test_interrupted_tiny_delta_uses_0_1_guard(self):
        # delta_total = 0.05 but divisor guarded to 0.1
        # reached = (peak - start)/0.1 = (21.0-20.95)/0.1 = 0.5
        s = score(reason="interrupted", start_temp=20.95, target=21.0,
                  peak_temp=21.0, current_temp=21.0)
        # accuracy: peak == target → 1.0; speed 1 → raw 0.8
        assert s == pytest.approx(0.8, abs=0.001)


# ── Speed component (35%) ────────────────────────────────────────────────────

class TestSpeedComponent:
    def test_expected_minutes_at_threshold_is_neutral(self):
        # expected <= 5 → speed = 0.6 (neutral), not difficulty-adjusted
        # raw = 0.4 + 0.6*0.35 + 0.25 = 0.86
        assert score(expected_minutes=5.0) == pytest.approx(0.86, abs=0.001)

    def test_expected_minutes_zero_is_neutral(self):
        assert score(expected_minutes=0.0) == pytest.approx(0.86, abs=0.001)

    def test_faster_than_expected_full_speed(self):
        # ratio <= 1 → speed 1.0
        assert score(expected_minutes=60.0, minutes_taken=30.0) == 1.0

    def test_ratio_exactly_one_full_speed(self):
        assert score(expected_minutes=60.0, minutes_taken=60.0) == 1.0

    def test_ratio_one_point_five_linear_falloff(self):
        # ratio 1.5 → speed = 1 - 0.5*0.7 = 0.65
        # raw = 0.4 + 0.65*0.35 + 0.25 = 0.8775 → rounds to 0.878 / 0.877
        s = score(expected_minutes=100.0, minutes_taken=150.0)
        assert s == pytest.approx(0.8775, abs=0.001)

    def test_ratio_two_lower_edge_of_linear(self):
        # ratio 2.0 → speed = 1 - 1.0*0.7 = 0.3
        # raw = 0.4 + 0.3*0.35 + 0.25 = 0.755
        s = score(expected_minutes=50.0, minutes_taken=100.0)
        assert s == pytest.approx(0.755, abs=0.001)

    def test_ratio_above_two_steep_falloff(self):
        # ratio 3.0 → speed = max(0.05, 0.3 - 1.0*0.1) = 0.2
        # raw = 0.4 + 0.2*0.35 + 0.25 = 0.72
        s = score(expected_minutes=50.0, minutes_taken=150.0)
        assert s == pytest.approx(0.72, abs=0.001)

    def test_extreme_ratio_hits_speed_floor(self):
        # ratio huge → speed floor 0.05
        # raw = 0.4 + 0.05*0.35 + 0.25 = 0.6675
        s = score(expected_minutes=10.0, minutes_taken=10000.0)
        assert s == pytest.approx(0.6675, abs=0.001)


# ── Difficulty correctors (raise speed by inflating expected time) ───────────

class TestDifficultyCorrectors:
    # Reference scenario: ratio 1.5 without difficulty → speed 0.65.
    _REF = dict(expected_minutes=100.0, minutes_taken=150.0)

    def test_extreme_cold_increases_score(self):
        baseline = score(outdoor_temp=5.0, **self._REF)
        cold = score(outdoor_temp=-35.0, **self._REF)
        assert cold > baseline

    def test_cold_corrector_capped(self):
        # outdoor -35 already saturates the +0.40 cap; -100 must give same speed
        at_cap = score(outdoor_temp=-35.0, **self._REF)
        beyond = score(outdoor_temp=-100.0, **self._REF)
        assert beyond == pytest.approx(at_cap, abs=0.001)

    def test_cold_only_applies_below_minus_five(self):
        # outdoor -5 is NOT < -5 → no cold corrector
        no_cold = score(outdoor_temp=-5.0, **self._REF)
        baseline = score(outdoor_temp=5.0, **self._REF)
        assert no_cold == pytest.approx(baseline, abs=0.001)

    def test_strong_wind_increases_score(self):
        baseline = score(wind_speed=0.0, **self._REF)
        windy = score(wind_speed=50.0, **self._REF)
        assert windy > baseline

    def test_wind_only_applies_above_ten(self):
        no_wind = score(wind_speed=10.0, **self._REF)
        baseline = score(wind_speed=0.0, **self._REF)
        assert no_wind == pytest.approx(baseline, abs=0.001)

    def test_humidity_corrector_requires_cold_outdoor(self):
        # humidity > 80 AND outdoor < 10 → applies
        applies = score(outdoor_humidity=100.0, outdoor_temp=5.0, **self._REF)
        # humidity > 80 but outdoor >= 10 → does NOT apply
        ignored = score(outdoor_humidity=100.0, outdoor_temp=12.0, **self._REF)
        baseline_cold = score(outdoor_humidity=0.0, outdoor_temp=5.0, **self._REF)
        baseline_mild = score(outdoor_humidity=0.0, outdoor_temp=12.0, **self._REF)
        assert applies > baseline_cold
        assert ignored == pytest.approx(baseline_mild, abs=0.001)

    def test_rain_increases_score(self):
        baseline = score(rain=0.0, **self._REF)
        rainy = score(rain=10.0, **self._REF)
        assert rainy > baseline

    def test_combined_correctors_stack(self):
        baseline = score(**self._REF)
        stacked = score(outdoor_temp=-35.0, wind_speed=50.0,
                        outdoor_humidity=100.0, rain=10.0, **self._REF)
        assert stacked > baseline


# ── Accuracy component (25%) ─────────────────────────────────────────────────

class TestAccuracyComponent:
    def test_no_overshoot_full_accuracy(self):
        assert score(peak_temp=21.0, target=21.0) == 1.0

    def test_peak_below_target_full_accuracy(self):
        assert score(peak_temp=20.0, target=21.0) == 1.0

    def test_one_degree_overshoot_half_accuracy(self):
        # overshoot 1 → accuracy 0.5 → raw = 0.4 + 0.35 + 0.5*0.25 = 0.875
        assert score(peak_temp=22.0, target=21.0) == pytest.approx(0.875, abs=0.001)

    def test_two_degree_overshoot_zero_accuracy(self):
        # overshoot 2 → accuracy 0 → raw = 0.4 + 0.35 = 0.75
        assert score(peak_temp=23.0, target=21.0) == pytest.approx(0.75, abs=0.001)

    def test_large_overshoot_clamped_at_zero_accuracy(self):
        # overshoot 5 → accuracy clamps to 0 → same as 2° overshoot
        s5 = score(peak_temp=26.0, target=21.0)
        s2 = score(peak_temp=23.0, target=21.0)
        assert s5 == pytest.approx(s2, abs=0.001)


# ── Environmental reliability discount ───────────────────────────────────────

class TestEnvDiscount:
    def test_solar_at_threshold_no_discount(self):
        # solar == 400 is NOT > 400 → no discount
        assert score(solar_radiation=400.0) == 1.0

    def test_high_solar_applies_discount(self):
        # solar 2400 → discount floor 0.60 → 1.0 * 0.60
        assert score(solar_radiation=2400.0) == pytest.approx(0.60, abs=0.001)

    def test_solar_discount_partial(self):
        # solar 900 → 1 - 500/2000 = 0.75
        assert score(solar_radiation=900.0) == pytest.approx(0.75, abs=0.001)

    def test_warm_at_threshold_no_discount(self):
        # outdoor == 15 is NOT > 15 → no warm discount
        assert score(outdoor_temp=15.0) == 1.0

    def test_warm_outdoor_applies_discount(self):
        # outdoor 40 → 1 - 25/25 = 0 → floor 0.70
        assert score(outdoor_temp=40.0) == pytest.approx(0.70, abs=0.001)

    def test_warm_discount_partial(self):
        # outdoor 20 → 1 - 5/25 = 0.8
        assert score(outdoor_temp=20.0) == pytest.approx(0.80, abs=0.001)

    def test_solar_and_warm_discounts_multiply(self):
        # 0.60 * 0.70 = 0.42
        s = score(solar_radiation=2400.0, outdoor_temp=40.0)
        assert s == pytest.approx(0.42, abs=0.001)


# ── Defaults / signature ─────────────────────────────────────────────────────

class TestDefaults:
    def test_outdoor_temp_none_defaults_to_ten(self):
        # outdoor_temp=None → treated as 10.0 (no cold, no warm discount)
        s_none = LearningEngine._calc_outcome_score(
            minutes_taken=30.0, expected_minutes=60.0, peak_temp=21.0,
            current_temp=21.0, target=21.0, start_temp=18.0,
            outdoor_temp=None,
        )
        s_ten = LearningEngine._calc_outcome_score(
            minutes_taken=30.0, expected_minutes=60.0, peak_temp=21.0,
            current_temp=21.0, target=21.0, start_temp=18.0,
            outdoor_temp=10.0,
        )
        assert s_none == s_ten == 1.0

    def test_default_reason_is_reached(self):
        # omitting reason → 'reached' → full reached credit
        s = LearningEngine._calc_outcome_score(
            minutes_taken=30.0, expected_minutes=60.0, peak_temp=21.0,
            current_temp=21.0, target=21.0, start_temp=18.0,
        )
        assert s == 1.0
