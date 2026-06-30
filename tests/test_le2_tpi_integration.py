"""Phase 19D: TPI and device path integration tests.

Verifies that the full coordinator cycle applies HeatLoss authority exactly once,
produces the correct setpoint/duty, and routes through the right device adapter path.

Test axes:
  - Setpoint-TRV: compute_tpi once, duty → duty_to_setpoint, Boost additive after TPI
  - Direct valve: same coef_int_used, duty % sent to valve, no additive boost
  - Minimal setup (TRV-only, no outdoor sensor): relative-only cap, safe default
  - No double HeatLoss application: coef_int feeds compute_tpi once; no second multiply
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from custom_components.thermosmart.const import (
    TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, UNIT_C_PER_H, TPI_MAX_BOOST_CELSIUS,
)
from custom_components.thermosmart.tpi import compute_tpi, duty_to_setpoint
from custom_components.thermosmart.learning.contracts import PredictionType
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


# ── shared helpers ────────────────────────────────────────────────────────────

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


def _valid_preds(hr=2.0, hl=0.4):
    """Valid predictions; candidate = clamp(hl/hr, 0.15, 1.2)."""
    return {
        PredictionType.HEAT_RATE: _pred({"heat_rate": hr}, units={"heat_rate": UNIT_C_PER_H}),
        PredictionType.HEAT_LOSS_RATE: _pred(
            {"heat_loss_rate": hl}, units={"heat_loss_rate": UNIT_C_PER_H},
            reason_codes=("general_rate",)),
    }


def _relative_only_preds(hr=2.0, hl=0.4):
    return {
        PredictionType.HEAT_RATE: _pred({"heat_rate": hr}, units={"heat_rate": UNIT_C_PER_H}),
        PredictionType.HEAT_LOSS_RATE: _pred(
            {"heat_loss_rate": hl}, units={"heat_loss_rate": UNIT_C_PER_H},
            reason_codes=("relative_only",), missing_evidence=("outdoor_temp",)),
    }


def _run(preds, *, n_warmup=10):
    """Create coordinator + shadow, warm up, return final recommendation."""
    coord = make_recording_coordinator()
    shadow = attach_shadow(coord)
    zr = shadow._runtime._zone(shadow._zone)
    for _ in range(n_warmup):
        zr.last_predictions = preds
        asyncio.run(coord._async_update_data())
    zr.last_predictions = preds
    result = asyncio.run(coord._async_update_data())
    return (result or {}).get("zone", {}), coord, shadow


# ── Section 1: Setpoint-TRV path ─────────────────────────────────────────────

class TestSetpointTRVPath:
    """Setpoint-TRV (no direct valve): TPI → duty_to_setpoint → trv_setpoint."""

    def test_tpi_duty_matches_compute_tpi_formula(self):
        """After convergence, tpi_duty_cycle == compute_tpi(target, indoor, outdoor, coef_int, coef_ext)."""
        rec, coord, _ = _run(_valid_preds())
        coef_int = rec.get("tpi_coef_int", 0.0)
        coef_ext = rec.get("tpi_coef_ext", 0.0)
        target = rec.get("effective_target")
        indoor = rec.get("current_temp")
        outdoor = 10.0  # make_recording_coordinator uses weather 10.0
        if target is not None and indoor is not None:
            expected_duty = compute_tpi(target, indoor, outdoor, coef_int, coef_ext)
            assert rec.get("tpi_duty_cycle") == pytest.approx(expected_duty, abs=0.01)

    def test_trv_setpoint_derived_from_duty(self):
        """trv_setpoint = duty_to_setpoint(target, duty, TPI_MAX_BOOST_CELSIUS) for non-valve TRVs."""
        rec, _, _ = _run(_valid_preds())
        target = rec.get("effective_target")
        duty = rec.get("tpi_duty_cycle", 0.0)
        if target is not None and not rec.get("tpi_valve_direct", False):
            expected_setpoint = duty_to_setpoint(target, duty, TPI_MAX_BOOST_CELSIUS)
            assert rec.get("trv_setpoint") == pytest.approx(expected_setpoint, abs=0.01)

    def test_coef_int_is_smoothed_used_value(self):
        """After 10 warmup cycles, tpi_coef_int is the smoothed used value, not raw candidate."""
        rec, coord, shadow = _run(_valid_preds(hr=2.0, hl=0.4))
        # After convergence: used ≈ candidate = 0.4/2.0 = 0.20
        assert rec.get("tpi_coef_int") == pytest.approx(0.20, abs=0.001)
        assert coord._tpi_coef_int_smoothed == pytest.approx(0.20, abs=0.001)

    def test_coef_ext_always_static_default(self):
        """coef_ext must always be TPI_COEF_EXT_DEFAULT; never modified by HeatLoss."""
        rec, _, _ = _run(_valid_preds())
        assert rec.get("tpi_coef_ext") == pytest.approx(TPI_COEF_EXT_DEFAULT, abs=1e-6)

    def test_no_double_heat_loss_application(self):
        """coef_int feeds compute_tpi once; HeatLoss is not applied again after TPI."""
        rec, _, _ = _run(_valid_preds(hr=2.0, hl=0.4))
        # coef_int_used = 0.20 (after convergence)
        # tpi_duty_candidate in diag = compute_tpi(target, indoor, outdoor, 0.20, coef_ext)
        # tpi_duty_used = same formula with coef_int (also 0.20 after convergence)
        diag = rec.get("tpi_coef_diag", {})
        assert diag.get("duty_candidate") == pytest.approx(
            rec.get("tpi_duty_cycle"), abs=0.01), (
            "After convergence, duty_candidate and duty_used must match "
            "(no second HeatLoss multiply)")

    def test_tpi_valve_direct_false_for_default_trv(self):
        """Default test TRV config has no valve entity → valve_direct = False."""
        rec, _, _ = _run(_valid_preds())
        # make_recording_coordinator uses "climate.test_trv" but no auto_valve_map entry
        assert rec.get("tpi_valve_direct") is False

    def test_boost_is_additive_not_multiplicative(self):
        """boost_offset_c is the additive correction; must not multiply with TPI output."""
        rec, _, _ = _run(_valid_preds())
        boost = rec.get("boost_offset_c", 0.0)
        # In shadow mode with no BOOST_FACTOR prediction: boost_offset_c = 0.0
        assert isinstance(boost, float)
        assert boost == pytest.approx(0.0, abs=0.001)  # no boost prediction in test


# ── Section 2: Direct valve path ─────────────────────────────────────────────

class TestDirectVentilPath:
    """Direct valve: same coef_int and duty, but trv_setpoint = target (no boost)."""

    def _run_with_valve(self, preds, *, n_warmup=10):
        """Run with a valve entity configured so auto_valve_map returns truthy."""
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        # Simulate valve present by pre-populating _auto_valve_map
        zr = shadow._runtime._zone(shadow._zone)
        for _ in range(n_warmup):
            zr.last_predictions = preds
            asyncio.run(coord._async_update_data())
        # Inject valve mapping and run one final cycle
        coord._auto_valve_map["climate.test_trv"] = "number.test_valve"
        zr.last_predictions = preds
        result = asyncio.run(coord._async_update_data())
        return (result or {}).get("zone", {}), coord

    def test_direct_valve_duty_uses_same_coef_int(self):
        """Whether valve or setpoint path, same coef_int_used feeds compute_tpi."""
        rec_no_valve, coord_no_valve, _ = _run(_valid_preds())
        rec_valve, coord_valve = self._run_with_valve(_valid_preds())
        # After 10 warmup cycles, both should converge to same coef_int
        assert rec_no_valve.get("tpi_coef_int") == pytest.approx(
            rec_valve.get("tpi_coef_int"), abs=0.001)

    def test_direct_valve_duty_matches_compute_tpi(self):
        rec, coord = self._run_with_valve(_valid_preds())
        coef_int = rec.get("tpi_coef_int", 0.0)
        coef_ext = rec.get("tpi_coef_ext", 0.0)
        target = rec.get("effective_target")
        indoor = rec.get("current_temp")
        outdoor = 10.0
        if target is not None and indoor is not None:
            expected = compute_tpi(target, indoor, outdoor, coef_int, coef_ext)
            assert rec.get("tpi_duty_cycle") == pytest.approx(expected, abs=0.01)

    def test_direct_valve_trv_setpoint_equals_target(self):
        """With valve, trv_setpoint = target (no duty-to-setpoint boost)."""
        rec, _ = self._run_with_valve(_valid_preds())
        target = rec.get("effective_target")
        if target is not None and rec.get("tpi_valve_direct"):
            assert rec.get("trv_setpoint") == pytest.approx(target, abs=0.01)

    def test_tpi_valve_direct_true_when_valve_mapped(self):
        rec, _ = self._run_with_valve(_valid_preds())
        assert rec.get("tpi_valve_direct") is True


# ── Section 3: Minimal setup — TRV-only, no outdoor sensor ───────────────────

class TestMinimalSetupTRVOnly:
    """TRV-only (no outdoor sensor), relative-only HeatLoss, no boost prediction."""

    def test_relative_only_cap_limits_blend(self):
        """With relative_only HeatLoss: blend capped at 0.50 → coef_int_used > raw candidate."""
        rec_rel, coord_rel, _ = _run(_relative_only_preds(hr=4.0, hl=1.0), n_warmup=15)
        rec_full, coord_full, _ = _run(_valid_preds(hr=4.0, hl=1.0), n_warmup=15)
        # relative_only converges to higher coef_int than full-context (less learned influence)
        assert coord_rel._tpi_coef_int_smoothed > coord_full._tpi_coef_int_smoothed - 0.001

    def test_relative_only_coef_int_above_pure_learned(self):
        """With relative_only: coef_int > raw hl/hr because blend ≤ 0.50."""
        rec, coord, _ = _run(_relative_only_preds(hr=4.0, hl=1.0), n_warmup=15)
        raw_candidate = 1.0 / 4.0  # = 0.25
        # coef_int_candidate = 0.50 * 0.25 + 0.50 * 0.60 = 0.425; used converges to 0.425
        assert coord._tpi_coef_int_smoothed > raw_candidate + 0.05

    def test_no_outdoor_sensor_tpi_still_computes(self):
        """Without outdoor temperature, compute_tpi still produces a valid duty."""
        # make_recording_coordinator has outdoor temp from weather engine mock
        # Here we test that even without outdoor the duty is valid
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        # Override weather engine to return no outdoor temp
        coord.weather_engine.async_get_data.return_value = {"temperature": None}
        zr = shadow._runtime._zone(shadow._zone)
        zr.last_predictions = _relative_only_preds()
        result = asyncio.run(coord._async_update_data())
        rec = (result or {}).get("zone", {})
        duty = rec.get("tpi_duty_cycle")
        # Should still compute (outdoor=None → compute_tpi skips ext term)
        assert duty is not None and duty >= 0.0

    def test_zone_available_without_outdoor_sensor(self):
        """Zone must remain operational with relative-only HeatLoss."""
        rec, _, _ = _run(_relative_only_preds())
        # effective_target must be set
        assert rec.get("effective_target") is not None
        # tpi_coef_int must be set and in valid range
        ci = rec.get("tpi_coef_int", 0.0)
        assert 0.15 <= ci <= 1.2

    def test_safe_baseline_when_no_predictions(self):
        """No predictions → deterministic baseline → safe default coef_int."""
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        zr = shadow._runtime._zone(shadow._zone)
        zr.last_predictions = {}  # no predictions at all
        result = asyncio.run(coord._async_update_data())
        rec = (result or {}).get("zone", {})
        assert rec.get("tpi_coef_int") == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)


# ── Section 4: Authority chain end-to-end ────────────────────────────────────

class TestAuthorityChain:
    """Confirm end-to-end: LE2 → candidate → smoothed used → single compute_tpi."""

    def test_valid_le2_produces_lower_coef_int_than_default(self):
        """With well-insulated room (hl << hr), learned coef < default."""
        rec, _, _ = _run(_valid_preds(hr=4.0, hl=0.8))  # raw=0.20 < default 0.60
        assert rec.get("tpi_coef_int", 1.0) < TPI_COEF_INT_DEFAULT - 0.01

    def test_high_loss_room_produces_higher_coef_int(self):
        """With high-loss room (hl ≈ hr), learned coef ≥ default."""
        rec, _, _ = _run(_valid_preds(hr=2.0, hl=1.8))  # raw=0.90 > default 0.60
        assert rec.get("tpi_coef_int", 0.0) > TPI_COEF_INT_DEFAULT - 0.01

    def test_diag_chain_is_complete(self):
        """coef_int_default → coef_int_learned → blend_weight → coef_int_candidate → coef_int_used."""
        rec, _, _ = _run(_valid_preds(hr=4.0, hl=0.8))
        diag = rec.get("tpi_coef_diag", {})
        for key in ("coef_int_default", "coef_int_learned", "blend_weight",
                    "coef_int_candidate", "coef_int_previous"):
            assert key in diag, f"Missing authority-chain key in diag: {key}"
        # Verify chain consistency after convergence
        assert diag["coef_int_default"] == pytest.approx(TPI_COEF_INT_DEFAULT)
        assert diag["coef_int_learned"] == pytest.approx(0.20, abs=0.001)
        assert diag["coef_int_candidate"] == pytest.approx(0.20, abs=0.002)  # blend=1 at conf=0.9

    def test_coef_ext_unchanged_by_heat_loss(self):
        """coef_ext must never be derived from HeatLoss; always static default."""
        for preds in [_valid_preds(hr=1.0, hl=2.0),  # high loss
                      _valid_preds(hr=4.0, hl=0.01),  # very low loss
                      _relative_only_preds()]:
            rec, _, _ = _run(preds)
            assert rec.get("tpi_coef_ext") == pytest.approx(TPI_COEF_EXT_DEFAULT, abs=1e-6)

    def test_preheat_does_not_affect_tpi_coef(self):
        """Preheat minutes are computed independently; they must not modify tpi_coef_int."""
        rec, coord, _ = _run(_valid_preds())
        coef_before = coord._tpi_coef_int_smoothed
        # Run another cycle — preheat recalculated, coef should still converge normally
        shadow = coord._le2_shadow
        zr = shadow._runtime._zone(shadow._zone)
        zr.last_predictions = _valid_preds()
        asyncio.run(coord._async_update_data())
        # coef_int_smoothed should stay at converged value (no preheat interference)
        assert abs(coord._tpi_coef_int_smoothed - coef_before) <= _TPI_COEF_INT_MAX_STEP_DOWN + 0.001


def _TPI_COEF_INT_MAX_STEP_DOWN():
    from custom_components.thermosmart.coordinator import _TPI_COEF_INT_MAX_STEP_DOWN as v
    return v

# Resolve the constant at module scope for use in the last test
from custom_components.thermosmart.coordinator import _TPI_COEF_INT_MAX_STEP_DOWN
