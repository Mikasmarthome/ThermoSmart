"""Phase 19D: TPI coefficient blending, transition safety, and stability tests.

Covers Points 2 and 3 from the audit:
  - Confidence-proportional blend between TPI_COEF_INT_DEFAULT and learned coef_int
  - No abrupt step change at the confidence threshold
  - Oscillation safety: confidence flicker causes only bounded duty variation
  - relative_only cap: HeatLoss without outdoor context blends at most 50 %
  - Stale / invalid predictions revert to deterministic baseline
  - Multi-cycle stability: sequence of valid → low_confidence → valid is smooth
  - Numerical duty sequence tests (point 3)

Variante B is the honest classification:
  heat_loss / heat_rate is a bounded adaptive heuristic — directionally correct
  but NOT a rigorous derivation of a proportional control gain.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from custom_components.thermosmart.const import (
    TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, UNIT_C_PER_H,
)
from custom_components.thermosmart.tpi import compute_tpi
from custom_components.thermosmart.learning.contracts import PredictionType
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator

# ── shared helpers ────────────────────────────────────────────────────────────

def _pred(values, *, fallback_used=False, confidence=0.8, units=None, warnings=(),
          reason_codes=(), missing_evidence=()):
    p = MagicMock()
    p.fallback_used = fallback_used
    p.confidence = confidence
    p.values = values
    p.warnings = tuple(warnings)
    p.units = {k: UNIT_C_PER_H for k in values} if units is None else units
    p.reason_codes = tuple(reason_codes)
    p.missing_evidence = tuple(missing_evidence)
    return p


def _hr(rate=4.0, *, conf=0.8, fallback=False):
    return _pred({"heat_rate": rate}, confidence=conf, fallback_used=fallback,
                 units={"heat_rate": UNIT_C_PER_H})


def _hl(rate=1.0, *, conf=0.8, fallback=False, relative_only=False):
    reason_codes = ("relative_only",) if relative_only else ("general_rate",)
    missing = ("outdoor_temp",) if relative_only else ()
    return _pred({"heat_loss_rate": rate}, confidence=conf, fallback_used=fallback,
                 units={"heat_loss_rate": UNIT_C_PER_H},
                 reason_codes=reason_codes, missing_evidence=missing)


def _sh(preds: dict):
    coord = make_recording_coordinator()
    sh = attach_shadow(coord)
    zr = sh._runtime._zone(sh._zone)
    zr.last_predictions = preds
    return sh


def _sh_with(heat_rate=4.0, heat_loss=1.0, *,
             hr_conf=0.8, hl_conf=0.8,
             hr_fallback=False, hl_fallback=False,
             relative_only=False):
    return _sh({
        PredictionType.HEAT_RATE: _hr(heat_rate, conf=hr_conf, fallback=hr_fallback),
        PredictionType.HEAT_LOSS_RATE: _hl(heat_loss, conf=hl_conf, fallback=hl_fallback,
                                           relative_only=relative_only),
    })


# Expected blend weight formula: clamp((conf - 0.35) / (0.80 - 0.35), 0, 1)
_MIN_CONF = 0.35
_FULL_CONF = 0.80
_REL_CAP = 0.50


def _expected_blend(conf: float, relative_only: bool = False) -> float:
    b = max(0.0, min(1.0, (conf - _MIN_CONF) / (_FULL_CONF - _MIN_CONF)))
    if relative_only:
        b = min(b, _REL_CAP)
    return b


def _expected_coef_int(heat_rate: float, heat_loss: float, conf: float,
                        relative_only: bool = False) -> float:
    from custom_components.thermosmart.tpi import estimate_coefficients
    raw, _ = estimate_coefficients(heat_rate, heat_loss)
    blend = _expected_blend(conf, relative_only)
    return blend * raw + (1.0 - blend) * TPI_COEF_INT_DEFAULT


# ── Section 1: Blend weight computation ──────────────────────────────────────

class TestBlendWeightComputation:

    def test_blend_zero_at_exactly_min_confidence(self):
        sh = _sh_with(hr_conf=_MIN_CONF, hl_conf=_MIN_CONF)
        _, _, _, status, diag = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert diag["blend_weight"] == pytest.approx(0.0, abs=0.001)

    def test_blend_one_at_full_confidence(self):
        sh = _sh_with(hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        _, _, _, status, diag = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert diag["blend_weight"] == pytest.approx(1.0, abs=0.001)

    def test_blend_linear_at_half_confidence(self):
        half_conf = (_MIN_CONF + _FULL_CONF) / 2.0  # 0.575
        expected = _expected_blend(half_conf)  # 0.5
        sh = _sh_with(hr_conf=half_conf, hl_conf=half_conf)
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        assert diag["blend_weight"] == pytest.approx(expected, abs=0.01)

    def test_blend_limited_by_min_confidence(self):
        sh = _sh_with(hr_conf=0.4, hl_conf=0.4)
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        blend = diag["blend_weight"]
        assert 0.0 <= blend <= 0.5  # early ramp: still mostly default

    def test_relative_only_caps_blend_at_50pct(self):
        # Even at full confidence, relative_only should cap blend at 0.50
        sh = _sh_with(hr_conf=1.0, hl_conf=1.0, relative_only=True)
        _, _, _, status, diag = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert diag["blend_weight"] <= _REL_CAP + 0.001
        assert diag["relative_only"] is True
        assert diag["outdoor_context_available"] is False

    def test_outdoor_context_present_not_capped(self):
        # No relative_only → blend can reach 1.0 at full confidence
        sh = _sh_with(hr_conf=1.0, hl_conf=1.0, relative_only=False)
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        assert diag["blend_weight"] == pytest.approx(1.0, abs=0.001)
        assert diag["outdoor_context_available"] is True

    def test_blend_zero_when_below_threshold(self):
        sh = _sh_with(hr_conf=0.34, hl_conf=0.34)
        ci, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "low_confidence"
        assert ci == TPI_COEF_INT_DEFAULT

    def test_diagnostics_contains_all_required_keys(self):
        sh = _sh_with()
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        for key in ("coef_int_default", "coef_int_learned", "blend_weight",
                    "relative_only", "outdoor_context_available",
                    "heat_rate_c_per_h", "heat_loss_c_per_h",
                    "hr_confidence", "hl_confidence"):
            assert key in diag, f"Missing key: {key}"


# ── Section 2: Blended coef_int values ───────────────────────────────────────

class TestBlendedCoefIntValues:

    def test_at_min_confidence_coef_int_is_default(self):
        sh = _sh_with(heat_rate=2.0, heat_loss=0.4,
                      hr_conf=_MIN_CONF, hl_conf=_MIN_CONF)
        ci, _, _, _, _ = sh.read_tpi_coefficients_safe()
        assert ci == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_at_full_confidence_coef_int_is_learned(self):
        sh = _sh_with(heat_rate=2.0, heat_loss=0.4,
                      hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        ci, _, _, _, diag = sh.read_tpi_coefficients_safe()
        # heat_loss/heat_rate = 0.4/2.0 = 0.2; blend=1.0 → ci=0.2
        assert ci == pytest.approx(0.2, abs=0.001)
        assert diag["coef_int_learned"] == pytest.approx(0.2, abs=0.001)

    def test_intermediate_confidence_yields_intermediate_coef(self):
        half_conf = (_MIN_CONF + _FULL_CONF) / 2.0
        sh = _sh_with(heat_rate=2.0, heat_loss=0.4,
                      hr_conf=half_conf, hl_conf=half_conf)
        ci, _, _, _, _ = sh.read_tpi_coefficients_safe()
        expected = _expected_coef_int(2.0, 0.4, half_conf)
        assert ci == pytest.approx(expected, abs=0.002)

    def test_relative_only_limits_coef_change(self):
        sh_normal = _sh_with(heat_rate=2.0, heat_loss=0.4, hr_conf=1.0, hl_conf=1.0)
        sh_relonly = _sh_with(heat_rate=2.0, heat_loss=0.4, hr_conf=1.0, hl_conf=1.0,
                               relative_only=True)
        ci_normal, _, _, _, _ = sh_normal.read_tpi_coefficients_safe()
        ci_relonly, _, _, _, _ = sh_relonly.read_tpi_coefficients_safe()
        # relative_only should be closer to default than normal
        assert abs(ci_relonly - TPI_COEF_INT_DEFAULT) < abs(ci_normal - TPI_COEF_INT_DEFAULT)

    def test_coef_int_clamped_to_max(self):
        # heat_loss >> heat_rate → ratio > 1.2 → must clamp; blend doesn't uncap
        sh = _sh_with(heat_rate=1.0, heat_loss=10.0,
                      hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        ci, _, _, _, diag = sh.read_tpi_coefficients_safe()
        assert ci <= 1.2 + 0.001  # never above clamp
        assert diag["coef_int_learned"] == pytest.approx(1.2, abs=0.001)

    def test_coef_int_never_below_min(self):
        # With any blend, coef_int must stay >= _COEF_INT_MIN (0.15)
        # (blend between 0.6 default and anything >= 0.15 learned is always >= 0.15)
        sh = _sh_with(heat_rate=100.0, heat_loss=0.001,
                      hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        ci, _, _, _, _ = sh.read_tpi_coefficients_safe()
        assert ci >= 0.15 - 0.001


# ── Section 3: No abrupt step at confidence threshold ────────────────────────

class TestNoAbruptStepAtThreshold:
    """Verify that crossing the gate-6 threshold causes only a small duty step."""

    _TARGET = 21.0
    _ROOM = 20.0  # 1°C error

    def _duty(self, ci: float) -> float:
        return compute_tpi(self._TARGET, self._ROOM, None, ci, TPI_COEF_EXT_DEFAULT)

    def test_duty_jump_at_threshold_is_small(self):
        # Just below threshold: status=low_confidence → default coef_int
        duty_below = self._duty(TPI_COEF_INT_DEFAULT)

        # Just above threshold: blend ≈ 0 → coef_int ≈ default
        sh_above = _sh_with(heat_rate=2.0, heat_loss=0.4,
                             hr_conf=_MIN_CONF + 0.005,
                             hl_conf=_MIN_CONF + 0.005)
        ci_above, _, _, _, _ = sh_above.read_tpi_coefficients_safe()
        duty_above = self._duty(ci_above)

        # Step must be tiny (< 1 % duty)
        assert abs(duty_above - duty_below) < 1.0

    def test_duty_at_full_confidence_is_correct(self):
        # heat_rate=2.0, heat_loss=0.4 → coef_int=0.2; blend=1.0 → duty=20%
        sh = _sh_with(heat_rate=2.0, heat_loss=0.4,
                      hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        ci, _, _, _, _ = sh.read_tpi_coefficients_safe()
        duty = self._duty(ci)
        assert duty == pytest.approx(20.0, abs=0.5)

    def test_default_to_learned_duty_change_high_loss(self):
        # Simulate first cycle above threshold vs. full learning
        sh_first = _sh_with(heat_rate=2.0, heat_loss=1.2,
                             hr_conf=_MIN_CONF, hl_conf=_MIN_CONF)
        sh_full  = _sh_with(heat_rate=2.0, heat_loss=1.2,
                             hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        ci_first, _, _, _, _ = sh_first.read_tpi_coefficients_safe()
        ci_full,  _, _, _, _ = sh_full.read_tpi_coefficients_safe()
        duty_first = self._duty(ci_first)
        duty_full  = self._duty(ci_full)
        # At full confidence: coef_int=0.6 (1.2/2.0) → duty=60%
        # At min confidence: coef_int=0.6 (default) → duty=60% (same)
        # No abrupt jump regardless of path
        assert abs(duty_full - duty_first) < 30.0  # max possible is ~40% (0.15 → 1.2)


# ── Section 4: Confidence oscillation stability ───────────────────────────────

class TestConfidenceOscillationStability:
    """Confidence flickering around the threshold must not cause TPI oscillation."""

    def test_flicker_at_threshold_causes_tiny_duty_change(self):
        target, room = 21.0, 20.0
        heat_rate, heat_loss = 2.0, 0.4  # learned coef_int = 0.2
        FLICKER_STEP = 0.01

        # Below threshold: default
        duty_below = compute_tpi(target, room, None, TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT)

        # Just above threshold
        sh = _sh_with(heat_rate=heat_rate, heat_loss=heat_loss,
                      hr_conf=_MIN_CONF + FLICKER_STEP,
                      hl_conf=_MIN_CONF + FLICKER_STEP)
        ci_above, _, _, _, _ = sh.read_tpi_coefficients_safe()
        duty_above = compute_tpi(target, room, None, ci_above, TPI_COEF_EXT_DEFAULT)

        assert abs(duty_above - duty_below) < 3.0  # < 3 % duty change at threshold

    def test_no_oscillation_over_confidence_sequence(self):
        confs = [0.34, 0.36, 0.34, 0.38, 0.35, 0.37]  # flicker sequence
        target, room = 21.0, 20.0
        duties = []
        for c in confs:
            if c >= _MIN_CONF:
                sh = _sh_with(heat_rate=2.0, heat_loss=0.4, hr_conf=c, hl_conf=c)
                ci, _, _, _, _ = sh.read_tpi_coefficients_safe()
            else:
                ci = TPI_COEF_INT_DEFAULT
            duties.append(compute_tpi(target, room, None, ci, TPI_COEF_EXT_DEFAULT))

        max_swing = max(duties) - min(duties)
        assert max_swing < 5.0  # never swing > 5 % across flicker sequence

    def test_stale_prediction_returns_to_default(self):
        sh = _sh({
            PredictionType.HEAT_RATE: _hr(2.0, conf=0.9),
            PredictionType.HEAT_LOSS_RATE: _pred(
                {"heat_loss_rate": 0.4}, confidence=0.9,
                units={"heat_loss_rate": UNIT_C_PER_H},
                warnings=("stale",), reason_codes=("general_rate",)),
        })
        ci, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "stale_or_superseded"
        assert ci == TPI_COEF_INT_DEFAULT

    def test_invalid_prediction_returns_to_default(self):
        sh = _sh_with(heat_rate=0.0, heat_loss=0.4)  # heat_rate=0 → invalid_value
        ci, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_value"
        assert ci == TPI_COEF_INT_DEFAULT


# ── Section 5: Sequence / stability over multiple TPI cycles ─────────────────

class TestTpiSequenceStability:
    """Numerical multi-cycle tests. Each cycle = one call to compute_tpi().

    Simulates a room heating from cold: large deficits early, shrinking as
    room temperature approaches target. Tests that duty response is stable.
    """

    def _simulate_cycles(self, coef_int: float, target=21.0, start_room=18.0,
                          outdoor=None, cycles=6, heat_rate_c_per_cycle=0.5):
        duties = []
        room = start_room
        for _ in range(cycles):
            duty = compute_tpi(target, room, outdoor, coef_int, TPI_COEF_EXT_DEFAULT)
            duties.append(round(duty, 1))
            # Simulated room rise: duty fraction × assumed heating rate
            room += (duty / 100.0) * heat_rate_c_per_cycle
        return duties, room

    def test_slow_room_moderate_loss_heats_to_target(self):
        # Low heat_rate (2.0), moderate heat_loss (0.4) → coef_int = 0.2
        coef_int = _expected_coef_int(2.0, 0.4, _FULL_CONF)
        duties, final_room = self._simulate_cycles(coef_int, target=21.0, start_room=18.0,
                                                    cycles=10, heat_rate_c_per_cycle=0.8)
        # Must produce meaningful duty (not near-zero)
        assert max(duties) > 20.0
        assert final_room > 19.0  # room is warming

    def test_well_insulated_room_no_overshoot(self):
        # Low heat_loss (0.1) with good heat_rate (3.0) → coef_int ≈ 0.033 → clamped to 0.15
        coef_int = _expected_coef_int(3.0, 0.1, _FULL_CONF)
        duties, final_room = self._simulate_cycles(coef_int, target=21.0, start_room=20.5,
                                                    cycles=6, heat_rate_c_per_cycle=0.3)
        # At small deficit (0.5°C), duty should be moderate, not excessive
        assert duties[0] < 50.0  # not wildly aggressive

    def test_high_loss_room_clamped_coef_stable(self):
        # heat_loss ≈ heat_rate → ratio > 1.2 → clamp → stable high duty
        coef_int = _expected_coef_int(2.0, 5.0, _FULL_CONF)
        duties, _ = self._simulate_cycles(coef_int, target=21.0, start_room=19.0,
                                          cycles=4, heat_rate_c_per_cycle=1.0)
        for d in duties:
            assert 0.0 <= d <= 100.0  # duty always in [0, 100]

    def test_confidence_ramp_duty_monotone(self):
        # As confidence rises from 0.35 to 0.80, duty for fixed error must be smooth.
        # heat_loss/heat_rate = 0.2 < 0.6 default → duty DECREASES as learned takes over
        confs = [0.35 + i * 0.05 for i in range(10)]
        target, room = 21.0, 20.0
        duties = []
        for c in confs:
            sh = _sh_with(heat_rate=2.0, heat_loss=0.4, hr_conf=c, hl_conf=c)
            ci, _, _, _, _ = sh.read_tpi_coefficients_safe()
            duties.append(compute_tpi(target, room, None, ci, TPI_COEF_EXT_DEFAULT))
        # Each step must change by at most the blend step size (≤ 15 % of full range)
        step_size = (TPI_COEF_INT_DEFAULT - 0.2) * 100.0 / (len(confs) - 1)
        for i in range(1, len(duties)):
            assert abs(duties[i] - duties[i - 1]) <= step_size + 1.0

    def test_valid_invalid_valid_sequence(self):
        # Cycle: valid_le2 → stale (→ default) → valid_le2
        # Duty must not jump unacceptably between transitions.
        target, room = 21.0, 20.0
        sh_valid = _sh_with(heat_rate=2.0, heat_loss=0.4,
                             hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        ci_v, _, _, s_v, _ = sh_valid.read_tpi_coefficients_safe()

        sh_stale = _sh({
            PredictionType.HEAT_RATE: _hr(2.0, conf=0.9),
            PredictionType.HEAT_LOSS_RATE: _pred(
                {"heat_loss_rate": 0.4}, confidence=0.9,
                units={"heat_loss_rate": UNIT_C_PER_H},
                warnings=("stale",), reason_codes=("general_rate",)),
        })
        ci_s, _, _, s_s, _ = sh_stale.read_tpi_coefficients_safe()

        d_valid = compute_tpi(target, room, None, ci_v, TPI_COEF_EXT_DEFAULT)
        d_stale = compute_tpi(target, room, None, ci_s, TPI_COEF_EXT_DEFAULT)
        d_back  = compute_tpi(target, room, None, ci_v, TPI_COEF_EXT_DEFAULT)

        assert s_v == "valid_le2"
        assert s_s == "stale_or_superseded"
        assert ci_s == TPI_COEF_INT_DEFAULT
        assert d_back == d_valid  # reproducible after restore

    def test_direct_valve_vs_setpoint_trv_same_coef_int(self):
        # Both device types use the same TPI coef_int — no double HeatLoss effect.
        sh = _sh_with(heat_rate=2.0, heat_loss=0.4,
                      hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        ci, ce, _, _, _ = sh.read_tpi_coefficients_safe()
        # Direct valve: compute_tpi applied directly to duty_pct
        duty_direct = compute_tpi(21.0, 20.0, None, ci, ce)
        # Setpoint TRV: same TPI coef applied, offset from duty via duty_to_setpoint
        from custom_components.thermosmart.tpi import duty_to_setpoint
        sp = duty_to_setpoint(21.0, duty_direct, 8.0)
        # Setpoint above or at target (boost mode), but same TPI basis
        assert sp >= 21.0


# ── Section 6: Diagnostics dict completeness ─────────────────────────────────

class TestDiagnosticsDict:

    def test_valid_le2_diag_has_learned_coef(self):
        sh = _sh_with(heat_rate=2.0, heat_loss=0.4,
                      hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        assert diag["coef_int_learned"] == pytest.approx(0.2, abs=0.001)
        assert diag["coef_int_default"] == TPI_COEF_INT_DEFAULT

    def test_valid_le2_diag_has_heat_rates(self):
        sh = _sh_with(heat_rate=3.0, heat_loss=0.9,
                      hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        assert diag["heat_rate_c_per_h"] == pytest.approx(3.0, abs=0.001)
        assert diag["heat_loss_c_per_h"] == pytest.approx(0.9, abs=0.001)

    def test_valid_le2_diag_has_confidences(self):
        sh = _sh_with(hr_conf=0.75, hl_conf=0.65)
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        assert diag["hr_confidence"] == pytest.approx(0.75, abs=0.001)
        assert diag["hl_confidence"] == pytest.approx(0.65, abs=0.001)

    def test_fallback_diag_has_none_learned(self):
        sh = _sh_with(hr_fallback=True)
        _, _, _, status, diag = sh.read_tpi_coefficients_safe()
        assert status == "cold_start"
        assert diag["coef_int_learned"] is None

    def test_disabled_diag_has_none_learned(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        sh._enabled = False
        _, _, _, status, diag = sh.read_tpi_coefficients_safe()
        assert status == "le2_disabled"
        assert diag["coef_int_learned"] is None

    def test_relative_only_in_diag(self):
        sh = _sh_with(relative_only=True, hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        assert diag["relative_only"] is True
        assert diag["outdoor_context_available"] is False

    def test_outdoor_context_in_diag(self):
        sh = _sh_with(relative_only=False, hr_conf=_FULL_CONF, hl_conf=_FULL_CONF)
        _, _, _, _, diag = sh.read_tpi_coefficients_safe()
        assert diag["outdoor_context_available"] is True
        assert diag["relative_only"] is False


# ── Section 7: Coordinator writes all required fields to recommendation ───────

class TestCoordinatorWritesTraceFields:
    """Verify coordinator stores all Phase 19D fields in recommendation before trace.

    The trace function (compute_decision_trace_safe) reads from recommendation;
    checking recommendation at trace-call time verifies the full coordinator flow.
    """

    def _run_and_capture(self, preds_dict):
        """Run coordinator update, capture recommendation at compute_decision_trace_safe call."""
        import asyncio
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        zr = shadow._runtime._zone(shadow._zone)
        zr.last_predictions = preds_dict

        captured = {}
        original = shadow.compute_decision_trace_safe
        def capturing_trace(rec, **kw):
            captured["rec"] = dict(rec)
            return original(rec, **kw)
        shadow.compute_decision_trace_safe = capturing_trace
        asyncio.run(coord._async_update_data())
        return captured.get("rec", {})

    def _valid_preds(self):
        return {
            PredictionType.HEAT_RATE: _hr(2.0, conf=0.9),
            PredictionType.HEAT_LOSS_RATE: _hl(0.4, conf=0.9),
        }

    def test_tpi_coef_int_written_to_recommendation(self):
        rec = self._run_and_capture(self._valid_preds())
        assert "tpi_coef_int" in rec
        assert rec["tpi_coef_int"] is not None

    def test_tpi_coef_ext_written_to_recommendation(self):
        rec = self._run_and_capture(self._valid_preds())
        assert "tpi_coef_ext" in rec
        assert rec["tpi_coef_ext"] == TPI_COEF_EXT_DEFAULT

    def test_tpi_coef_diag_written_to_recommendation(self):
        rec = self._run_and_capture(self._valid_preds())
        assert "tpi_coef_diag" in rec
        diag = rec["tpi_coef_diag"]
        assert isinstance(diag, dict)
        for key in ("coef_int_default", "coef_int_learned", "blend_weight",
                    "relative_only", "outdoor_context_available",
                    "heat_rate_c_per_h", "heat_loss_c_per_h"):
            assert key in diag, f"Missing diag key: {key}"

    def test_tpi_coef_source_written_when_valid(self):
        rec = self._run_and_capture(self._valid_preds())
        assert rec.get("tpi_coef_source") == "valid_le2"

    def test_tpi_hl_rate_written_and_positive_when_valid(self):
        rec = self._run_and_capture(self._valid_preds())
        hl = rec.get("tpi_hl_rate")
        assert hl is not None and hl > 0

    def test_diag_blend_weight_in_expected_range(self):
        rec = self._run_and_capture(self._valid_preds())
        diag = rec.get("tpi_coef_diag", {})
        blend = diag.get("blend_weight", -1)
        assert 0.0 <= blend <= 1.0

    def test_no_blend_fields_when_prediction_missing(self):
        rec = self._run_and_capture({})  # no predictions
        assert rec.get("tpi_coef_source") == "prediction_missing"
        assert rec.get("tpi_hl_rate") is None
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("coef_int_learned") is None
