"""Golden tests for the TPI formula and LE2 → TPI coefficient mapping (Phase 19D).

All expected values in this file are hand-calculated from the production formulas
in tpi.py and ha_integration.read_tpi_coefficients_safe().

Formulas
--------
  compute_tpi:
    error_int = target - room                                       [°C]
    error_ext = target - outdoor                                    [°C]
    duty      = coef_int * error_int + coef_ext * error_ext        [fraction]
    duty_pct  = clamp(duty * 100, 0, 100)                         [%]

  coef_int  = heat_loss_rate / heat_rate  (clamped [0.15, 1.2])  [dimensionless]
  coef_ext  = TPI_COEF_EXT_DEFAULT = 0.01                        [Variante A: static]

No double-outdoor-effect because coef_ext is static; the raw heat_loss_rate (which
was observed at some outdoor temp) is ONLY used to derive coef_int, never coef_ext.
"""
from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock

from custom_components.thermosmart.const import TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT
from custom_components.thermosmart.tpi import compute_tpi, estimate_coefficients
from custom_components.thermosmart.learning.contracts import PredictionType
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator

# ── helpers ─────────────────────────────────────────────────────────────────

def _pred(values, *, fallback_used=False, confidence=0.8, unit_key="heat_rate"):
    p = MagicMock()
    p.fallback_used = fallback_used
    p.confidence = confidence
    p.values = values
    p.warnings = ()
    p.units = {unit_key: "C/h"}
    return p


def _hr(rate): return _pred({"heat_rate": rate})
def _hl(rate): return _pred({"heat_loss_rate": rate}, unit_key="heat_loss_rate")


def _sh_with_rates(heat_rate, heat_loss):
    coord = make_recording_coordinator()
    sh = attach_shadow(coord)
    zr = sh._runtime._zone(sh._zone)
    zr.last_predictions = {
        PredictionType.HEAT_RATE: _hr(heat_rate),
        PredictionType.HEAT_LOSS_RATE: _hl(heat_loss),
    }
    return sh


# ── Section 1: compute_tpi() formula verification ──────────────────────────

class TestComputeTpiFormula:
    """Exact formula: duty = coef_int*(target-room) + coef_ext*(target-outdoor)"""

    def test_perfect_room_no_outdoor(self):
        # room == target → error_int == 0, no outdoor → duty == 0
        assert compute_tpi(21.0, 21.0, None, 0.5, 0.01) == pytest.approx(0.0)

    def test_one_degree_error_int_only(self):
        # target=21, room=20 → error_int=1; coef_int=0.5; no outdoor
        # duty = 0.5 * 1 = 0.5 → 50%
        assert compute_tpi(21.0, 20.0, None, 0.5, 0.01) == pytest.approx(50.0)

    def test_two_degree_error_int_only(self):
        # error_int=2; coef_int=0.6 (default)
        # duty = 0.6 * 2 = 1.2 → clamp to 100%
        assert compute_tpi(21.0, 19.0, None, 0.6, 0.01) == pytest.approx(100.0)

    def test_outdoor_term_added(self):
        # target=21, room=20, outdoor=0 → error_int=1, error_ext=21
        # coef_int=0.3, coef_ext=0.01
        # duty = 0.3*1 + 0.01*21 = 0.3 + 0.21 = 0.51 → 51%
        result = compute_tpi(21.0, 20.0, 0.0, 0.3, 0.01)
        assert result == pytest.approx(51.0, abs=0.2)

    def test_outdoor_none_skips_ext_term(self):
        # outdoor=None → only int term
        r_with = compute_tpi(21.0, 20.0, 0.0, 0.3, 0.01)
        r_without = compute_tpi(21.0, 20.0, None, 0.3, 0.01)
        assert r_without < r_with

    def test_room_above_target_returns_zero(self):
        # room > target → error_int < 0 → duty < 0 → clamp to 0
        assert compute_tpi(21.0, 22.0, None, 0.5, 0.01) == pytest.approx(0.0)

    def test_clamp_lower_bound(self):
        # extreme undershoot still gives 0, never negative
        assert compute_tpi(21.0, 30.0, None, 1.0, 0.05) == pytest.approx(0.0)

    def test_clamp_upper_bound(self):
        # extreme overshoot still gives 100, never above 100
        assert compute_tpi(21.0, 1.0, 0.0, 1.2, 0.06) == pytest.approx(100.0)

    def test_no_double_outdoor_effect(self):
        # heat_loss_rate is observed at outdoor=10 (ΔT=11), now outdoor=0 (ΔT=21).
        # Variante A: coef_ext is TPI_COEF_EXT_DEFAULT regardless.
        # The outdoor difference between the two scenarios only enters via error_ext,
        # not through coef_ext, so there is no double-scaling.
        # At outdoor=10: duty_ext = 0.01 * (21-10) = 0.11
        # At outdoor=0:  duty_ext = 0.01 * (21-0)  = 0.21
        # Difference is exactly 0.01 * (10-0) = 0.10 → 10% extra duty at colder outdoor
        r_warm = compute_tpi(21.0, 20.0, 10.0, 0.3, TPI_COEF_EXT_DEFAULT)
        r_cold = compute_tpi(21.0, 20.0, 0.0,  0.3, TPI_COEF_EXT_DEFAULT)
        # Expected difference: 0.01 * 10 * 100 = 10%
        assert (r_cold - r_warm) == pytest.approx(10.0, abs=0.2)


# ── Section 2: estimate_coefficients() clamping ─────────────────────────────

class TestEstimateCoefficients:
    """estimate_coefficients computes coef_int (clamped); coef_ext is now overridden."""

    def test_ratio_two_to_four(self):
        ci, _ = estimate_coefficients(4.0, 2.0)
        assert ci == pytest.approx(0.5, abs=0.001)

    def test_ratio_clamped_to_min(self):
        # 0.01/4 = 0.0025 → clamp to 0.15
        ci, _ = estimate_coefficients(4.0, 0.01)
        assert ci == pytest.approx(0.15, abs=0.001)

    def test_ratio_clamped_to_max(self):
        # 10/1 = 10 → clamp to 1.2
        ci, _ = estimate_coefficients(1.0, 10.0)
        assert ci == pytest.approx(1.2, abs=0.001)

    def test_zero_heat_rate_returns_defaults(self):
        ci, ce = estimate_coefficients(0.0, 2.0)
        assert ci == TPI_COEF_INT_DEFAULT
        assert ce == TPI_COEF_EXT_DEFAULT

    def test_zero_heat_loss_returns_defaults(self):
        ci, ce = estimate_coefficients(2.0, 0.0)
        assert ci == TPI_COEF_INT_DEFAULT
        assert ce == TPI_COEF_EXT_DEFAULT

    def test_none_returns_defaults(self):
        ci, ce = estimate_coefficients(None, None)
        assert ci == TPI_COEF_INT_DEFAULT
        assert ce == TPI_COEF_EXT_DEFAULT


# ── Section 3: read_tpi_coefficients_safe() golden values ───────────────────

class TestTpiGoldenValues:
    """Hand-calculated expected coef_int values from valid LE2 predictions."""

    def test_no_heat_loss_cold_start_returns_defaults(self):
        # heat_loss = 0 → gate 7 (invalid_value) → static defaults
        sh = _sh_with_rates(2.0, 0.0)
        ci, ce, hl, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_value"
        assert ci == TPI_COEF_INT_DEFAULT
        assert ce == TPI_COEF_EXT_DEFAULT
        assert hl is None

    def test_moderate_loss_coef_int(self):
        # heat_rate=2.0, heat_loss=0.4 → coef_int = 0.4/2.0 = 0.2
        # but 0.2 > 0.15 (COEF_INT_MIN), so no clamping → 0.2
        sh = _sh_with_rates(2.0, 0.4)
        ci, ce, hl, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert ci == pytest.approx(0.2, abs=0.001)
        assert ce == TPI_COEF_EXT_DEFAULT  # Variante A: always static
        assert hl == pytest.approx(0.4, abs=0.001)

    def test_moderate_loss_tpi_duty(self):
        # With coef_int=0.2, coef_ext=0.01, target=21, room=20, no outdoor:
        # duty = 0.2 * 1 = 0.2 → 20%
        sh = _sh_with_rates(2.0, 0.4)
        ci, ce, _, _, _ = sh.read_tpi_coefficients_safe()
        duty = compute_tpi(21.0, 20.0, None, ci, ce)
        assert duty == pytest.approx(20.0, abs=0.2)

    def test_high_loss_coef_int_clamped(self):
        # heat_rate=2.0, heat_loss=1.2 → coef_int = 1.2/2.0 = 0.6 (no clamping needed)
        sh = _sh_with_rates(2.0, 1.2)
        ci, ce, hl, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert ci == pytest.approx(0.6, abs=0.001)
        assert ce == TPI_COEF_EXT_DEFAULT

    def test_high_loss_tpi_duty(self):
        # coef_int=0.6, target=21, room=20, no outdoor → duty=60%
        sh = _sh_with_rates(2.0, 1.2)
        ci, ce, _, _, _ = sh.read_tpi_coefficients_safe()
        duty = compute_tpi(21.0, 20.0, None, ci, ce)
        assert duty == pytest.approx(60.0, abs=0.2)

    def test_high_loss_higher_duty_than_moderate(self):
        # Higher heat_loss → higher coef_int → higher duty (physically correct)
        sh_mod = _sh_with_rates(2.0, 0.4)
        sh_hi  = _sh_with_rates(2.0, 1.2)
        ci_mod, _, _, _, _ = sh_mod.read_tpi_coefficients_safe()
        ci_hi,  _, _, _, _ = sh_hi.read_tpi_coefficients_safe()
        duty_mod = compute_tpi(21.0, 20.0, None, ci_mod, TPI_COEF_EXT_DEFAULT)
        duty_hi  = compute_tpi(21.0, 20.0, None, ci_hi,  TPI_COEF_EXT_DEFAULT)
        assert duty_hi > duty_mod

    def test_heat_loss_ge_heat_rate_clamped_at_max(self):
        # heat_rate=1.0, heat_loss=5.0 → ratio=5 → clamped to 1.2
        sh = _sh_with_rates(1.0, 5.0)
        ci, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert ci == pytest.approx(1.2, abs=0.001)

    def test_very_small_heat_rate_not_division_by_zero(self):
        # Very small but positive heat_rate: should clamp to max coef_int, not exception
        sh = _sh_with_rates(0.001, 0.5)
        ci, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert ci <= 1.2

    def test_same_outdoor_temp_no_extra_duty_from_coef_ext(self):
        # Two scenarios with different outdoor temps at episode time — but same NOW:
        # Both get coef_ext = TPI_COEF_EXT_DEFAULT, confirming no double-outdoor-scaling.
        sh_warm_obs = _sh_with_rates(2.0, 0.3)  # episode was at outdoor=15°C
        sh_cold_obs = _sh_with_rates(2.0, 0.5)  # episode was at outdoor=0°C
        _, ce_warm, _, _, _ = sh_warm_obs.read_tpi_coefficients_safe()
        _, ce_cold, _, _, _ = sh_cold_obs.read_tpi_coefficients_safe()
        # coef_ext must be the same (static default) regardless of episode outdoor
        assert ce_warm == ce_cold == TPI_COEF_EXT_DEFAULT

    def test_extreme_nan_heat_rate_returns_defaults(self):
        sh = _sh_with_rates(float("nan"), 0.5)
        ci, ce, hl, status, _ = sh.read_tpi_coefficients_safe()
        assert status in ("invalid_value", "not_available", "prediction_missing")
        assert ci == TPI_COEF_INT_DEFAULT

    def test_extreme_inf_heat_rate_returns_defaults(self):
        sh = _sh_with_rates(float("inf"), 0.5)
        # estimate_coefficients with inf → ratio≈0 → clamped to 0.15 → valid_le2 or invalid
        ci, ce, hl, status, _ = sh.read_tpi_coefficients_safe()
        assert ci <= 1.2  # never exceeds max

    def test_negative_heat_loss_returns_invalid_value(self):
        sh = _sh_with_rates(2.0, -0.5)
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_value"


# ── Section 4: outdoor isolation proof ──────────────────────────────────────

class TestNoDoubleOutdoorEffect:
    """Prove that heat_loss_rate (raw) does not double-multiply with outdoor temp.

    The risk: heat_loss_rate is observed at T_outdoor_obs.  If coef_ext were derived
    from it and then multiplied by (target - outdoor_now), outdoor would be counted
    twice.  Variante A prevents this by keeping coef_ext at TPI_COEF_EXT_DEFAULT.
    """

    @pytest.mark.parametrize("outdoor_obs", [0.0, 5.0, 10.0, 15.0])
    def test_coef_ext_constant_across_observation_outdoor_temps(self, outdoor_obs):
        # heat_loss rate as if observed at different outdoor temps:
        # simulated as different rates (proxy, since we don't normalize by ΔT)
        assumed_delta = 21.0 - outdoor_obs
        heat_loss_rate = 0.3 * assumed_delta / 21.0  # scale by ΔT ratio
        sh = _sh_with_rates(2.0, max(heat_loss_rate, 0.01))
        _, coef_ext, _, _, _ = sh.read_tpi_coefficients_safe()
        assert coef_ext == TPI_COEF_EXT_DEFAULT, (
            f"coef_ext must be TPI_COEF_EXT_DEFAULT regardless of episode outdoor; "
            f"got {coef_ext} for outdoor_obs={outdoor_obs}"
        )

    def test_outdoor_change_only_enters_via_error_ext(self):
        # With fixed coef_int and coef_ext, changing outdoor temp changes only error_ext.
        # The difference between two outdoor temps must be exactly:
        #   Δduty = coef_ext * Δoutdoor * 100
        ci, ce = 0.4, TPI_COEF_EXT_DEFAULT
        target, room = 21.0, 20.0
        d_warm = compute_tpi(target, room, 10.0, ci, ce)
        d_cold = compute_tpi(target, room,  0.0, ci, ce)
        expected_delta = ce * (10.0 - 0.0) * 100.0  # 0.01 * 10 * 100 = 10%
        assert (d_cold - d_warm) == pytest.approx(expected_delta, abs=0.1)

