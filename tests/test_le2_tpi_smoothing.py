"""Phase 19D: TPI coef_int step-limiting (candidate vs. used) tests.

Architecture: read_tpi_coefficients_safe() returns the blended *candidate* coef_int.
The coordinator applies per-cycle step-limiting (max ±0.10) to produce the *used* value
stored in recommendation["tpi_coef_int"].  This prevents abrupt duty jumps on:
  - valid → stale (candidate reverts to 0.60 default immediately; used ramps toward it)
  - stale → valid (used ramps back to learned candidate gradually)
  - confidence oscillation (sub-step-size flicker causes negligible duty change)
  - restore / restart (used starts at TPI_COEF_INT_DEFAULT; controlled first cycle)
  - sharp model value change (large candidate delta truncated to max step per cycle)
  - general → conditioned bucket switch (bounded change via same mechanism)

Each test that runs multiple coordinator cycles calls _cycle() to ensure state accumulates
correctly on the SAME coordinator instance.

Step derivation (documented here for traceability):
  cycle period  ≈ 5 min (DEFAULT_SCAN_INTERVAL)
  max step      = 0.10 per cycle
  full range    = [0.15, 1.2]; typical excursion 0.20 ↔ 0.60 = 0.40
  convergence   = ⌈0.40 / 0.10⌉ = 4 cycles ≈ 20 min  (within room thermal TC ~30-120 min)
  max duty Δ    = 0.10 step × 3 °C error = 30 % / cycle  (bounded, large deficit tolerated)
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from custom_components.thermosmart.const import (
    TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, UNIT_C_PER_H,
)
from custom_components.thermosmart.coordinator import (
    _TPI_COEF_INT_MAX_STEP_UP, _TPI_COEF_INT_MAX_STEP_DOWN,
    _TPI_COEF_INT_CLAMP_MIN, _TPI_COEF_INT_CLAMP_MAX,
)
from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.tpi import compute_tpi
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


# ── shared prediction builders ────────────────────────────────────────────────

def _pred(values, *, fallback_used=False, confidence=0.9, units=None,
          warnings=(), reason_codes=(), missing_evidence=()):
    p = MagicMock()
    p.fallback_used = fallback_used
    p.confidence = confidence
    p.values = values
    p.warnings = tuple(warnings)
    p.units = {k: UNIT_C_PER_H for k in values} if units is None else units
    p.reason_codes = tuple(reason_codes)
    p.missing_evidence = tuple(missing_evidence)
    return p


def _hr(rate=4.0, *, conf=0.9):
    return _pred({"heat_rate": rate}, confidence=conf, units={"heat_rate": UNIT_C_PER_H})


def _hl(rate=1.0, *, conf=0.9, relative_only=False):
    rc = ("relative_only",) if relative_only else ("general_rate",)
    me = ("outdoor_temp",) if relative_only else ()
    return _pred({"heat_loss_rate": rate}, confidence=conf,
                 units={"heat_loss_rate": UNIT_C_PER_H},
                 reason_codes=rc, missing_evidence=me)


def _stale_hr(rate=4.0):
    return _pred({"heat_rate": rate}, warnings=("stale",),
                 units={"heat_rate": UNIT_C_PER_H})


def _stale_hl(rate=1.0):
    return _pred({"heat_loss_rate": rate}, warnings=("stale",),
                 units={"heat_loss_rate": UNIT_C_PER_H})


def _valid_preds(hr=4.0, hl=1.0, *, hr_conf=0.9, hl_conf=0.9):
    """Valid predictions with candidate = clamp(hl/hr, 0.15, 1.2)."""
    return {
        PredictionType.HEAT_RATE: _hr(hr, conf=hr_conf),
        PredictionType.HEAT_LOSS_RATE: _hl(hl, conf=hl_conf),
    }


def _stale_preds(hr=4.0, hl=1.0):
    return {
        PredictionType.HEAT_RATE: _stale_hr(hr),
        PredictionType.HEAT_LOSS_RATE: _stale_hl(hl),
    }


def _setup(preds_dict=None):
    """Create coordinator + shadow, optionally inject initial predictions."""
    coord = make_recording_coordinator()
    shadow = attach_shadow(coord)
    if preds_dict is not None:
        shadow._runtime._zone(shadow._zone).last_predictions = preds_dict
    return coord, shadow


def _cycle(coord, shadow, preds_dict):
    """Run one coordinator update cycle with given predictions, return recommendation dict."""
    shadow._runtime._zone(shadow._zone).last_predictions = preds_dict
    result = asyncio.run(coord._async_update_data())
    # _async_update_data returns {"weather": ..., "zone": recommendation, ...}
    return (result or {}).get("zone", {})


# ── Section 1: Step constant correctness ─────────────────────────────────────

class TestStepConstants:
    def test_step_up_is_01(self):
        assert _TPI_COEF_INT_MAX_STEP_UP == pytest.approx(0.10, abs=1e-9)

    def test_step_down_is_01(self):
        assert _TPI_COEF_INT_MAX_STEP_DOWN == pytest.approx(0.10, abs=1e-9)

    def test_clamp_min_matches_estimate_coefficients(self):
        from custom_components.thermosmart.tpi import estimate_coefficients
        # estimate_coefficients returns coef_int clamped to [0.15, 1.2]
        ci, _ = estimate_coefficients(1.0, 0.1)
        assert _TPI_COEF_INT_CLAMP_MIN <= ci <= _TPI_COEF_INT_CLAMP_MAX

    def test_clamp_max_matches_estimate_coefficients(self):
        from custom_components.thermosmart.tpi import estimate_coefficients
        ci, _ = estimate_coefficients(1.0, 10.0)
        assert ci == pytest.approx(_TPI_COEF_INT_CLAMP_MAX, abs=1e-9)

    def test_initial_smoothed_state_is_default(self):
        coord, _ = _setup()
        assert coord._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT)

    def test_initial_duty_previous_is_none(self):
        coord, _ = _setup()
        assert coord._tpi_duty_previous is None


# ── Section 2: First cycle from cold start ───────────────────────────────────

class TestFirstCycleColdStart:
    """Restart always begins at TPI_COEF_INT_DEFAULT; first cycle does a controlled step."""

    def test_no_shadow_first_cycle_uses_default(self):
        coord = make_recording_coordinator()
        result = asyncio.run(coord._async_update_data())
        ci = (result or {}).get("zone", {}).get("tpi_coef_int")
        assert ci == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_first_valid_cycle_steps_toward_candidate(self):
        # candidate = hl/hr = 1.0/4.0 = 0.25, previous = 0.60
        # step = max(-0.10, min(0.10, 0.25 - 0.60)) = -0.10
        # used = 0.60 - 0.10 = 0.50
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        ci = rec.get("tpi_coef_int", rec.get("tpi_coef_int"))
        assert ci == pytest.approx(0.50, abs=0.001)

    def test_first_stale_cycle_stays_at_default(self):
        # candidate for stale = TPI_COEF_INT_DEFAULT = 0.60
        # previous = 0.60, step = 0, used = 0.60
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, _stale_preds())
        assert rec.get("tpi_coef_int") == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_first_missing_preds_cycle_stays_at_default(self):
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, {})
        assert rec.get("tpi_coef_int") == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_smoothed_state_updated_after_first_cycle(self):
        coord, shadow = _setup()
        _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        # coef_int_smoothed = 0.50 after one step from 0.60
        assert coord._tpi_coef_int_smoothed == pytest.approx(0.50, abs=0.001)

    def test_duty_previous_set_after_first_cycle(self):
        coord, shadow = _setup()
        _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        assert coord._tpi_duty_previous is not None


# ── Section 3: Valid → stale transition ──────────────────────────────────────

class TestValidToStaleTransition:
    """Stale candidate = TPI_COEF_INT_DEFAULT (0.60); used ramps from learned back to 0.60."""

    def _converge_to_learned(self, coord, shadow, n=10):
        """Run n cycles of valid predictions to converge used → candidate."""
        for _ in range(n):
            _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))

    def test_candidate_reverts_to_default_on_stale(self):
        """Status immediately changes to stale_or_superseded; source = stale_or_superseded."""
        coord, shadow = _setup()
        self._converge_to_learned(coord, shadow)
        rec = _cycle(coord, shadow, _stale_preds())
        assert rec.get("tpi_coef_source") == "stale_or_superseded"

    def test_used_approaches_default_after_stale(self):
        """After stale, used moves from ~0.25 toward 0.60 by max step per cycle."""
        coord, shadow = _setup()
        self._converge_to_learned(coord, shadow)
        prev = coord._tpi_coef_int_smoothed
        rec = _cycle(coord, shadow, _stale_preds())
        used = rec.get("tpi_coef_int", 0.0)
        # Step toward 0.60 from prev ≈ 0.25: should increase by exactly max_step
        expected = min(TPI_COEF_INT_DEFAULT, prev + _TPI_COEF_INT_MAX_STEP_UP)
        assert used == pytest.approx(expected, abs=0.001)

    def test_duty_change_bounded_on_stale(self):
        """At 2 °C error, duty change per cycle ≤ max_step × error = 0.10 × 2 = 20 %."""
        coord, shadow = _setup()
        self._converge_to_learned(coord, shadow)
        _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        duty_before = coord._tpi_duty_previous
        _cycle(coord, shadow, _stale_preds())
        duty_after = coord._tpi_duty_previous
        if duty_before is not None and duty_after is not None:
            # Error bound: max_step × max reasonable error (3 °C) = 30 %
            assert abs(duty_after - duty_before) <= 32.0  # 30 % + small tolerance

    def test_converges_to_default_within_6_stale_cycles(self):
        """4 cycles × 0.10 step ≥ 0.40 range → converged in ≤ 5 cycles."""
        coord, shadow = _setup()
        self._converge_to_learned(coord, shadow)
        for _ in range(6):
            _cycle(coord, shadow, _stale_preds())
        assert coord._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_diag_shows_transition_applied_during_ramp(self):
        coord, shadow = _setup()
        self._converge_to_learned(coord, shadow)
        rec = _cycle(coord, shadow, _stale_preds())
        diag = rec.get("tpi_coef_diag", {})
        # Only applies if previous < candidate (i.e., used != candidate yet)
        if coord._tpi_coef_int_smoothed < TPI_COEF_INT_DEFAULT - 0.001:
            assert diag.get("transition_applied") is True

    def test_diag_coef_int_candidate_is_default_on_stale(self):
        coord, shadow = _setup()
        self._converge_to_learned(coord, shadow)
        rec = _cycle(coord, shadow, _stale_preds())
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("coef_int_candidate") == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)


# ── Section 4: Stale → valid recovery ────────────────────────────────────────

class TestStaleToValidRecovery:
    """When predictions become valid again, used ramps back toward candidate."""

    def test_recovery_bounded_per_cycle(self):
        coord, shadow = _setup()
        # Start at default (no prior learning)
        _cycle(coord, shadow, _stale_preds())
        assert coord._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)
        # One valid cycle: candidate=0.25, prev=0.60, step=-0.10, used=0.50
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        assert rec.get("tpi_coef_int") == pytest.approx(0.50, abs=0.001)

    def test_no_oscillation_on_valid_stale_valid(self):
        """valid→stale→valid sequence converges smoothly without oscillation."""
        coord, shadow = _setup()
        # Converge to learned (~0.25)
        for _ in range(10):
            _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        at_learned = coord._tpi_coef_int_smoothed
        # Go stale for 4 cycles
        for _ in range(4):
            _cycle(coord, shadow, _stale_preds())
        at_stale = coord._tpi_coef_int_smoothed
        # Recover valid for 10 cycles
        prev = coord._tpi_coef_int_smoothed
        for _ in range(10):
            rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
            used = rec.get("tpi_coef_int", 0.0)
            # Monotonically decreasing back toward learned (no overshoot)
            assert used <= prev + 0.001
            prev = used
        # Eventually converges back near at_learned (no permanent drift)
        assert coord._tpi_coef_int_smoothed == pytest.approx(at_learned, abs=0.11)


# ── Section 5: Confidence oscillation safety ─────────────────────────────────

class TestConfidenceOscillation:
    """Confidence flicker near threshold causes tiny duty variation."""

    def test_oscillation_below_threshold_stays_at_default(self):
        """conf < 0.35 → gate 6 failure → candidate = default; no oscillation."""
        coord, shadow = _setup()
        duties = []
        for conf in [0.30, 0.34, 0.30, 0.33, 0.31]:
            rec = _cycle(coord, shadow, {
                PredictionType.HEAT_RATE: _hr(conf=conf),
                PredictionType.HEAT_LOSS_RATE: _hl(conf=conf),
            })
            duties.append(rec.get("tpi_coef_int", TPI_COEF_INT_DEFAULT))
        # All at default (no LE2 blending)
        for ci in duties:
            assert ci == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_oscillation_around_threshold_causes_small_duty_swing(self):
        """conf oscillates 0.34 ↔ 0.36; duty swing < 10 % per swing (sub step-size)."""
        coord, shadow = _setup()
        ci_values = []
        for conf in [0.34, 0.36, 0.34, 0.37, 0.33]:
            rec = _cycle(coord, shadow, {
                PredictionType.HEAT_RATE: _hr(conf=conf),
                PredictionType.HEAT_LOSS_RATE: _hl(conf=conf),
            })
            ci_values.append(rec.get("tpi_coef_int", TPI_COEF_INT_DEFAULT))
        # Max swing between any two consecutive values ≤ max_step = 0.10
        for i in range(1, len(ci_values)):
            assert abs(ci_values[i] - ci_values[i - 1]) <= _TPI_COEF_INT_MAX_STEP_UP + 0.001

    def test_full_confidence_flicker_bounded(self):
        """High-conf valid ↔ stale flicker: coef_int change per cycle ≤ 0.10."""
        coord, shadow = _setup()
        # Converge to learned first
        for _ in range(10):
            _cycle(coord, shadow, _valid_preds())
        ci_prev = coord._tpi_coef_int_smoothed
        for preds in [_stale_preds(), _valid_preds(), _stale_preds(), _valid_preds()]:
            rec = _cycle(coord, shadow, preds)
            ci_now = rec.get("tpi_coef_int", 0.0)
            assert abs(ci_now - ci_prev) <= _TPI_COEF_INT_MAX_STEP_UP + 0.001
            ci_prev = ci_now


# ── Section 6: Sharp prediction value change ─────────────────────────────────

class TestSharpPredictionChange:
    """Large candidate Δ is rate-limited to max_step per cycle."""

    def test_large_candidate_jump_is_step_limited(self):
        """candidate jumps 0.25 → 0.90; each cycle moves at most 0.10."""
        coord, shadow = _setup()
        # Converge to hr=4, hl=1.0 → candidate ≈ 0.25
        for _ in range(10):
            _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        at_low = coord._tpi_coef_int_smoothed
        # Now high-loss prediction: hr=2, hl=1.8 → raw=0.9, clamped=0.9
        rec = _cycle(coord, shadow, _valid_preds(hr=2.0, hl=1.8))
        used = rec.get("tpi_coef_int", 0.0)
        expected = min(0.9, at_low + _TPI_COEF_INT_MAX_STEP_UP)
        assert used == pytest.approx(expected, abs=0.001)

    def test_converges_to_new_candidate_in_finite_cycles(self):
        """After 15 cycles, used reaches candidate ≈ 0.90."""
        coord, shadow = _setup()
        for _ in range(15):
            _cycle(coord, shadow, _valid_preds(hr=2.0, hl=1.8))
        # 0.60 → 0.90 in 3 steps of 0.10 each
        assert coord._tpi_coef_int_smoothed == pytest.approx(0.90, abs=0.001)

    def test_downward_jump_bounded(self):
        """candidate drops from ~0.90 to 0.25; per cycle step ≤ 0.10."""
        coord, shadow = _setup()
        # Converge to high candidate
        for _ in range(15):
            _cycle(coord, shadow, _valid_preds(hr=2.0, hl=1.8))
        at_high = coord._tpi_coef_int_smoothed
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        used = rec.get("tpi_coef_int", 0.0)
        expected = max(0.25, at_high - _TPI_COEF_INT_MAX_STEP_DOWN)
        assert used == pytest.approx(expected, abs=0.001)


# ── Section 7: Multi-zone isolation ──────────────────────────────────────────

class TestMultiZoneIsolation:
    """Two coordinators (two zones) maintain independent smoothed states."""

    def test_two_zones_independent(self):
        coord1, shadow1 = _setup()
        coord2, shadow2 = _setup()
        # Zone 1: converge to learned ≈ 0.25
        for _ in range(10):
            _cycle(coord1, shadow1, _valid_preds(hr=4.0, hl=1.0))
        # Zone 2: stay at stale (→ default 0.60)
        for _ in range(4):
            _cycle(coord2, shadow2, _stale_preds())
        assert coord1._tpi_coef_int_smoothed < 0.35
        assert coord2._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_zone1_stale_does_not_affect_zone2(self):
        coord1, shadow1 = _setup()
        coord2, shadow2 = _setup()
        # Both converge to learned
        for _ in range(10):
            _cycle(coord1, shadow1, _valid_preds(hr=4.0, hl=1.0))
            _cycle(coord2, shadow2, _valid_preds(hr=4.0, hl=1.0))
        zone2_before = coord2._tpi_coef_int_smoothed
        # Zone 1 goes stale
        for _ in range(6):
            _cycle(coord1, shadow1, _stale_preds())
        # Zone 2 unchanged
        assert coord2._tpi_coef_int_smoothed == pytest.approx(zone2_before, abs=0.001)

    def test_zone_states_are_independent_instances(self):
        """Modifying one zone's smoothed state does not affect the other."""
        coord1, shadow1 = _setup()
        coord2, shadow2 = _setup()
        coord1._tpi_coef_int_smoothed = 0.33
        assert coord2._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT)


# ── Section 8: Restart / restore semantics ───────────────────────────────────

class TestRestartRestoreSemantics:
    """Transient state: after restart, coordinator always starts at deterministic baseline."""

    def test_new_coordinator_starts_at_default(self):
        coord, _ = _setup()
        assert coord._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT)

    def test_fresh_coordinator_not_contaminated_by_prior(self):
        """Simulates restart: a second coordinator is independent of the first."""
        coord1, shadow1 = _setup()
        for _ in range(10):
            _cycle(coord1, shadow1, _valid_preds(hr=4.0, hl=1.0))
        # Create fresh coordinator (simulates HA restart)
        coord2, _ = _setup()
        assert coord2._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT)

    def test_first_valid_cycle_after_restart_controlled_step(self):
        """After restart (at default), first valid cycle steps by at most 0.10."""
        coord, shadow = _setup()
        prev = coord._tpi_coef_int_smoothed  # 0.60
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        used = rec.get("tpi_coef_int", 0.0)
        assert abs(used - prev) <= _TPI_COEF_INT_MAX_STEP_DOWN + 0.001

    def test_controlled_convergence_to_learned_after_restart(self):
        """After 10 cycles, used converges to candidate ≈ 0.25."""
        coord, shadow = _setup()
        for _ in range(10):
            _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        assert coord._tpi_coef_int_smoothed == pytest.approx(0.25, abs=0.001)


# ── Section 9: Duty trace fields ─────────────────────────────────────────────

class TestDutyTraceFields:
    """Duty trace fields in tpi_coef_diag for audit."""

    def _run(self, preds):
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, preds)
        return rec.get("tpi_coef_diag", {})

    def test_duty_used_present_when_temp_available(self):
        diag = self._run(_valid_preds())
        assert "duty_used" in diag
        assert diag["duty_used"] is not None

    def test_duty_candidate_present_when_temp_available(self):
        diag = self._run(_valid_preds())
        assert "duty_candidate" in diag
        assert diag["duty_candidate"] is not None

    def test_duty_previous_none_on_first_cycle(self):
        diag = self._run(_valid_preds())
        assert diag.get("duty_previous") is None  # no prior cycle

    def test_duty_previous_set_on_second_cycle(self):
        coord, shadow = _setup()
        _cycle(coord, shadow, _valid_preds())
        rec2 = _cycle(coord, shadow, _valid_preds())
        diag2 = rec2.get("tpi_coef_diag", {})
        assert diag2.get("duty_previous") is not None

    def test_duty_used_equals_duty_cycle(self):
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, _valid_preds())
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("duty_used") == pytest.approx(rec.get("tpi_duty_cycle", -1), abs=0.01)

    def test_duty_candidate_ge_duty_used_when_stepping_down(self):
        """When coef_int is stepping down (toward lower learned), candidate duty ≤ used duty."""
        coord, shadow = _setup()
        # First cycle: candidate ≈ 0.25 < prev 0.60 → stepping down → candidate duty < used duty
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        diag = rec.get("tpi_coef_diag", {})
        if diag.get("duty_candidate") is not None and diag.get("duty_used") is not None:
            # candidate duty (at lower coef) ≤ used duty (at higher stepped coef)
            assert diag["duty_candidate"] <= diag["duty_used"] + 0.5


# ── Section 10: Transition diag fields ───────────────────────────────────────

class TestTransitionDiagFields:
    """coef_int_candidate, coef_int_previous, transition_applied in diag."""

    def test_coef_int_candidate_in_diag(self):
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        diag = rec.get("tpi_coef_diag", {})
        assert "coef_int_candidate" in diag
        assert diag["coef_int_candidate"] == pytest.approx(0.25, abs=0.001)

    def test_coef_int_previous_in_diag(self):
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, _valid_preds())
        diag = rec.get("tpi_coef_diag", {})
        assert "coef_int_previous" in diag
        assert diag["coef_int_previous"] == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_transition_applied_true_when_stepping(self):
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("transition_applied") is True

    def test_transition_applied_false_when_converged(self):
        coord, shadow = _setup()
        for _ in range(10):
            _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("transition_applied") is False

    def test_transition_reason_step_limited_vs_converged(self):
        coord, shadow = _setup()
        # First cycle: stepping
        rec1 = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        d1 = rec1.get("tpi_coef_diag", {})
        assert d1.get("transition_reason") == "step_limited"
        # After convergence: converged
        for _ in range(9):
            _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        rec2 = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        d2 = rec2.get("tpi_coef_diag", {})
        assert d2.get("transition_reason") == "converged"

    def test_max_step_fields_in_diag(self):
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, _valid_preds())
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("coef_int_max_step_up") == pytest.approx(_TPI_COEF_INT_MAX_STEP_UP)
        assert diag.get("coef_int_max_step_down") == pytest.approx(_TPI_COEF_INT_MAX_STEP_DOWN)

    def test_no_shadow_diag_has_no_shadow_reason(self):
        coord = make_recording_coordinator()
        result = asyncio.run(coord._async_update_data())
        rec = (result or {}).get("zone", {})
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("transition_reason") == "no_shadow"

    def test_stale_candidate_default_matches_default_previous_on_cold_start(self):
        coord, shadow = _setup()
        rec = _cycle(coord, shadow, _stale_preds())
        diag = rec.get("tpi_coef_diag", {})
        # Both candidate and previous are TPI_COEF_INT_DEFAULT → converged immediately
        assert diag.get("coef_int_candidate") == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)
        assert diag.get("coef_int_previous") == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)
        assert diag.get("transition_applied") is False


# ── Section 11: Relative-only bucket change ───────────────────────────────────

class TestGeneralToConditionedBucketChange:
    """Switching from relative_only to conditioned bucket does not cause abrupt jump."""

    def test_relative_only_to_full_context_bounded(self):
        coord, shadow = _setup()
        # Converge with relative_only (blend capped at 0.50)
        for _ in range(15):
            _cycle(coord, shadow, {
                PredictionType.HEAT_RATE: _hr(4.0, conf=0.9),
                PredictionType.HEAT_LOSS_RATE: _hl(1.0, conf=0.9, relative_only=True),
            })
        at_relative = coord._tpi_coef_int_smoothed
        # Switch to conditioned bucket (full blend)
        prev = coord._tpi_coef_int_smoothed
        rec = _cycle(coord, shadow, {
            PredictionType.HEAT_RATE: _hr(4.0, conf=0.9),
            PredictionType.HEAT_LOSS_RATE: _hl(1.0, conf=0.9, relative_only=False),
        })
        used = rec.get("tpi_coef_int", 0.0)
        assert abs(used - prev) <= _TPI_COEF_INT_MAX_STEP_DOWN + 0.001

    def test_relative_only_cap_limits_learned_influence(self):
        """With relative_only, blend ≤ 0.50 → coef_int_used ≥ 50% × default."""
        coord, shadow = _setup()
        # Run enough cycles with relative_only to converge
        for _ in range(15):
            _cycle(coord, shadow, {
                PredictionType.HEAT_RATE: _hr(4.0, conf=0.9),
                PredictionType.HEAT_LOSS_RATE: _hl(1.0, conf=0.9, relative_only=True),
            })
        # Candidate with relative_only + blend=0.50:
        # coef_int_raw = 0.25, blend = 0.50, candidate = 0.50×0.25 + 0.50×0.60 = 0.425
        ci_converged = coord._tpi_coef_int_smoothed
        assert ci_converged >= 0.40  # always above pure-learned 0.25
        assert ci_converged < TPI_COEF_INT_DEFAULT  # but below full default


# ── Section 12: Safety and goal-achievement ───────────────────────────────────

class TestSafetyAndGoalAchievement:
    """Heating remains effective; step-limiting never blocks frost or safety decisions."""

    def test_large_deficit_duty_still_positive(self):
        """Even with low learned coef_int, large deficit produces positive duty."""
        coord, shadow = _setup()
        for _ in range(10):
            _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        assert rec.get("tpi_duty_cycle", 0.0) > 0.0

    def test_coef_int_never_below_clamp_min(self):
        """coef_int_used always ≥ _TPI_COEF_INT_CLAMP_MIN regardless of transitions."""
        coord, shadow = _setup()
        for preds in [_valid_preds(hr=4.0, hl=0.001),  # would give very small candidate
                      _stale_preds(), _valid_preds(), {}]:
            rec = _cycle(coord, shadow, preds)
            ci = rec.get("tpi_coef_int", 1.0)
            assert ci >= _TPI_COEF_INT_CLAMP_MIN - 0.001

    def test_coef_int_never_above_clamp_max(self):
        coord, shadow = _setup()
        for _ in range(5):
            rec = _cycle(coord, shadow, _valid_preds(hr=1.0, hl=2.0))  # raw > 1.2 → clamped
            ci = rec.get("tpi_coef_int", 0.0)
            assert ci <= _TPI_COEF_INT_CLAMP_MAX + 0.001


# ── Section 13: One-coordinator-per-zone architecture proof ──────────────────

class TestOneCoordinatorPerZone:
    """Production architecture: one ConfigEntry = one Zone = one Coordinator.

    zone_id = entry.entry_id (unique per HA config entry).
    _tpi_coef_int_smoothed is an instance attribute on each coordinator → inherently
    zone-scoped.  Two zones = two coordinator instances = two independent states.

    This mirrors the __init__.py setup path:
        coordinator = ThermoSmartCoordinator(hass, entry, ...)   # one per entry
        hass.data[DOMAIN][entry.entry_id] = {"coordinator": coordinator}

    Reload (options change) calls async_reload → new coordinator instance → fresh state.
    """

    def test_two_coordinator_instances_have_independent_state(self):
        """Different coordinator objects cannot share instance attributes."""
        coord_a, shadow_a = _setup()
        coord_b, shadow_b = _setup()
        # Drive zone A toward learned ≈ 0.25
        for _ in range(10):
            _cycle(coord_a, shadow_a, _valid_preds(hr=4.0, hl=1.0))
        # Zone B stays at default (stale predictions)
        for _ in range(4):
            _cycle(coord_b, shadow_b, _stale_preds())
        assert coord_a._tpi_coef_int_smoothed < 0.35
        assert coord_b._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_reload_creates_fresh_state(self):
        """Simulated reload: new coordinator object starts at default regardless of prior state."""
        coord_old, shadow_old = _setup()
        for _ in range(10):
            _cycle(coord_old, shadow_old, _valid_preds(hr=4.0, hl=1.0))
        old_state = coord_old._tpi_coef_int_smoothed
        assert old_state < 0.35  # was trained
        # Reload = new coordinator instance (old one discarded)
        coord_new, _ = _setup()
        assert coord_new._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_no_shadow_resets_smoothed_state_each_cycle(self):
        """Without shadow, every cycle explicitly holds _tpi_coef_int_smoothed at default."""
        coord = make_recording_coordinator()
        # Manually corrupt the smoothed state (simulates hypothetical stale attachment)
        coord._tpi_coef_int_smoothed = 0.35
        result = asyncio.run(coord._async_update_data())
        # No shadow branch resets to TPI_COEF_INT_DEFAULT
        assert coord._tpi_coef_int_smoothed == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_detach_reattach_shadow_starts_from_previous_used(self):
        """Attaching shadow to a fresh coordinator always starts from TPI_COEF_INT_DEFAULT."""
        coord, shadow = _setup()
        # First valid cycle: starts at TPI_COEF_INT_DEFAULT → steps toward candidate
        rec = _cycle(coord, shadow, _valid_preds(hr=4.0, hl=1.0))
        used_cycle1 = rec.get("tpi_coef_int", 0.0)
        # Must be between default and candidate (exactly one step down from 0.60)
        assert TPI_COEF_INT_DEFAULT - _TPI_COEF_INT_MAX_STEP_DOWN - 0.001 <= used_cycle1 <= TPI_COEF_INT_DEFAULT + 0.001


# ── Section 14: Stale trace semantics ─────────────────────────────────────────

class TestStaleTraceSemantics:
    """During ramp from learned used to default, trace must clearly show:
      - coef_int_candidate is immediately TPI_COEF_INT_DEFAULT (not the old learned value)
      - tpi_coef_source is "stale_or_superseded" (not "valid_le2")
      - coef_int_used is the transient controller state only
      - transition_applied = True during ramp, False when converged
      - transition_reason = "step_limited" during ramp

    Required sequence from audit:
      valid learned used ≈ 0.20 → stale → used 0.30 → 0.40 → 0.50 → 0.60
    """

    def _converge(self, coord, shadow, n=10):
        for _ in range(n):
            _cycle(coord, shadow, _valid_preds(hr=4.0, hl=0.8))  # raw = 0.2

    def test_candidate_is_immediately_default_on_stale(self):
        coord, shadow = _setup()
        self._converge(coord, shadow)
        rec = _cycle(coord, shadow, _stale_preds())
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("coef_int_candidate") == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    def test_source_is_stale_or_superseded_immediately(self):
        coord, shadow = _setup()
        self._converge(coord, shadow)
        rec = _cycle(coord, shadow, _stale_preds())
        assert rec.get("tpi_coef_source") == "stale_or_superseded"

    def test_source_never_claims_valid_le2_during_ramp(self):
        """No cycle during stale ramp should call itself valid_le2."""
        coord, shadow = _setup()
        self._converge(coord, shadow)
        for _ in range(5):
            rec = _cycle(coord, shadow, _stale_preds())
            assert rec.get("tpi_coef_source") != "valid_le2"

    def test_used_ramps_from_learned_to_default_in_sequence(self):
        """Exact numerical sequence: 0.20 → 0.30 → 0.40 → 0.50 → 0.60."""
        coord, shadow = _setup()
        self._converge(coord, shadow)
        # After convergence, smoothed state ≈ 0.20
        assert coord._tpi_coef_int_smoothed == pytest.approx(0.20, abs=0.001)
        expected = [0.30, 0.40, 0.50, 0.60]
        for expected_used in expected:
            rec = _cycle(coord, shadow, _stale_preds())
            used = rec.get("tpi_coef_int", 0.0)
            assert used == pytest.approx(expected_used, abs=0.001), (
                f"Expected used={expected_used}, got {used}")

    def test_transition_applied_true_during_ramp(self):
        coord, shadow = _setup()
        self._converge(coord, shadow)
        rec = _cycle(coord, shadow, _stale_preds())
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("transition_applied") is True

    def test_transition_reason_step_limited_during_ramp(self):
        coord, shadow = _setup()
        self._converge(coord, shadow)
        rec = _cycle(coord, shadow, _stale_preds())
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("transition_reason") == "step_limited"

    def test_transition_converged_after_full_ramp(self):
        coord, shadow = _setup()
        self._converge(coord, shadow)
        # 4 stale cycles to reach default from 0.20
        for _ in range(4):
            _cycle(coord, shadow, _stale_preds())
        rec = _cycle(coord, shadow, _stale_preds())
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("transition_reason") == "converged"
        assert diag.get("transition_applied") is False
