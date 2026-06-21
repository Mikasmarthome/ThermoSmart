"""Tests for the TPI (Time Proportional Integrator) algorithm.

compute_tpi, duty_to_setpoint and estimate_coefficients are pure Python
functions with no HA dependencies — no fixtures needed.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.tpi import (
    compute_tpi,
    duty_to_setpoint,
    estimate_coefficients,
)
from custom_components.thermosmart.const import (
    TPI_COEF_INT_DEFAULT,
    TPI_COEF_EXT_DEFAULT,
)


# ── compute_tpi ──────────────────────────────────────────────────────────────

class TestComputeTPI:
    def test_zero_when_at_target_no_outdoor(self):
        """room == target, outdoor=None → only internal term → 0.0."""
        assert compute_tpi(21.0, 21.0, None, 0.6, 0.0) == 0.0

    def test_zero_when_at_target_and_outdoor_equals_target(self):
        """Both error_int and error_ext are 0 → 0.0."""
        assert compute_tpi(21.0, 21.0, 21.0, 0.6, 0.01) == 0.0

    def test_zero_when_room_above_target(self):
        """room > target → negative duty clamped to 0.0."""
        assert compute_tpi(21.0, 23.0, None, 0.6, 0.0) == 0.0

    def test_clamped_at_100_very_cold(self):
        """Very large temperature error clamps output to 100%."""
        result = compute_tpi(30.0, 5.0, -20.0, 0.6, 0.01)
        assert result == 100.0

    def test_proportional_to_internal_error(self):
        """Duty scales linearly with room error when coef_ext=0."""
        # error_int=1.0, coef_int=0.6, outdoor ignored → duty_pct = 60.0
        result = compute_tpi(21.0, 20.0, None, 0.6, 0.0)
        assert result == 60.0

    def test_outdoor_term_adds_to_duty(self):
        """Outdoor error adds a positive contribution to duty."""
        result_no_outdoor = compute_tpi(21.0, 20.0, None, 0.6, 0.01)
        result_with_outdoor = compute_tpi(21.0, 20.0, 0.0, 0.6, 0.01)
        # error_ext = 21 - 0 = 21, coef_ext=0.01 → +21 duty_pct added
        assert result_with_outdoor > result_no_outdoor

    def test_outdoor_none_omits_external_term(self):
        """outdoor=None → only coef_int * error_int drives the duty."""
        result = compute_tpi(21.0, 20.0, None, 0.6, 0.01)
        # Only internal term: 0.6 * 1.0 = 0.6 → 60.0%
        assert result == 60.0

    def test_manual_calculation_full_formula(self):
        """Verify the complete TPI formula manually."""
        target, room, outdoor = 22.0, 20.0, 5.0
        coef_int, coef_ext = 0.5, 0.02
        duty = coef_int * (target - room) + coef_ext * (target - outdoor)
        expected = min(100.0, max(0.0, round(duty * 100.0, 1)))
        assert compute_tpi(target, room, outdoor, coef_int, coef_ext) == expected

    def test_result_rounded_to_one_decimal(self):
        """Output is always rounded to 1 decimal place."""
        result = compute_tpi(21.5, 20.0, None, 0.33, 0.0)
        assert result == round(result, 1)

    def test_coef_int_zero_only_outdoor_drives(self):
        """coef_int=0 → only the outdoor term contributes."""
        # error_ext = 21.0 - 0.0 = 21.0, duty = 0.01 * 21.0 = 0.21 → 21.0%
        result = compute_tpi(21.0, 21.0, 0.0, 0.0, 0.01)
        assert result == 21.0

    def test_returns_float(self):
        """Return type is always float."""
        result = compute_tpi(21.0, 20.0, 10.0, 0.6, 0.01)
        assert isinstance(result, float)


# ── duty_to_setpoint ─────────────────────────────────────────────────────────

class TestDutyToSetpoint:
    def test_zero_duty_returns_target(self):
        """0% duty → setpoint equals the target temperature."""
        assert duty_to_setpoint(21.0, 0.0) == 21.0

    def test_50_percent_duty_half_boost(self):
        """50% duty with default 8°C max_boost → target + 4°C."""
        assert duty_to_setpoint(21.0, 50.0, max_boost=8.0) == 25.0

    def test_100_percent_duty_capped_at_28(self):
        """100% duty is always capped at 28°C (TRV safety limit)."""
        assert duty_to_setpoint(21.0, 100.0, max_boost=8.0) == 28.0

    def test_high_target_still_capped_at_28(self):
        """High base target + full duty cannot exceed 28°C."""
        assert duty_to_setpoint(25.0, 100.0, max_boost=8.0) == 28.0

    def test_custom_max_boost_scales_result(self):
        """Custom max_boost parameter is respected."""
        # 100% duty, max_boost=4 → target + 4 = 25.0 (< 28)
        assert duty_to_setpoint(21.0, 100.0, max_boost=4.0) == 25.0

    def test_result_rounded_to_one_decimal(self):
        """Setpoint is always rounded to 1 decimal place."""
        result = duty_to_setpoint(21.0, 33.3, max_boost=8.0)
        assert result == round(result, 1)

    def test_proportional_boost(self):
        """Boost is proportional: 25% duty → target + 2°C."""
        assert duty_to_setpoint(21.0, 25.0, max_boost=8.0) == 23.0

    def test_returns_float(self):
        assert isinstance(duty_to_setpoint(21.0, 50.0), float)


# ── estimate_coefficients ────────────────────────────────────────────────────

class TestEstimateCoefficients:
    def test_typical_learned_data(self):
        """heat_rate=0.05, heat_loss=0.02 → coef_int=0.4, coef_ext≈0.008."""
        coef_int, coef_ext = estimate_coefficients(0.05, 0.02)
        assert coef_int == pytest.approx(0.4, abs=0.001)
        assert coef_ext == pytest.approx(0.008, abs=0.001)

    def test_fallback_when_heat_rate_is_none(self):
        """None heat_rate → returns default coefficients."""
        coef_int, coef_ext = estimate_coefficients(None, 0.02)
        assert coef_int == TPI_COEF_INT_DEFAULT
        assert coef_ext == TPI_COEF_EXT_DEFAULT

    def test_fallback_when_heat_loss_is_none(self):
        """None heat_loss_rate → returns default coefficients."""
        coef_int, coef_ext = estimate_coefficients(0.05, None)
        assert coef_int == TPI_COEF_INT_DEFAULT
        assert coef_ext == TPI_COEF_EXT_DEFAULT

    def test_fallback_when_heat_rate_is_zero(self):
        """Zero heat_rate (division-by-zero guard) → returns defaults."""
        coef_int, coef_ext = estimate_coefficients(0.0, 0.02)
        assert coef_int == TPI_COEF_INT_DEFAULT
        assert coef_ext == TPI_COEF_EXT_DEFAULT

    def test_fallback_when_both_are_none(self):
        coef_int, coef_ext = estimate_coefficients(None, None)
        assert coef_int == TPI_COEF_INT_DEFAULT
        assert coef_ext == TPI_COEF_EXT_DEFAULT

    def test_clamp_coef_int_minimum(self):
        """Ratio below 0.15 is clamped to the minimum (0.15)."""
        # 0.001 / 0.1 = 0.01 → clamped to 0.15
        coef_int, _ = estimate_coefficients(0.1, 0.001)
        assert coef_int == 0.15

    def test_clamp_coef_int_maximum(self):
        """Very high ratio (inefficient heating) clamped to 1.2."""
        # 0.1 / 0.001 = 100 → clamped to 1.2
        coef_int, _ = estimate_coefficients(0.001, 0.1)
        assert coef_int == 1.2

    def test_coef_ext_is_derived_from_coef_int(self):
        """coef_ext should be approximately coef_int / 50."""
        coef_int, coef_ext = estimate_coefficients(0.05, 0.02)
        assert coef_ext == pytest.approx(coef_int / 50.0, abs=0.0001)

    def test_coef_ext_never_below_minimum(self):
        """coef_ext is always >= 0.002 regardless of input."""
        # Even if coef_int is at its minimum (0.15), coef_ext = 0.003 ≥ 0.002
        _, coef_ext = estimate_coefficients(0.1, 0.001)
        assert coef_ext >= 0.002

    def test_coef_ext_never_above_maximum(self):
        """coef_ext is always <= 0.06 regardless of input."""
        _, coef_ext = estimate_coefficients(0.001, 0.1)
        assert coef_ext <= 0.06

    def test_returns_two_floats(self):
        coef_int, coef_ext = estimate_coefficients(0.05, 0.02)
        assert isinstance(coef_int, float)
        assert isinstance(coef_ext, float)
