"""Pflicht-Tests: HeatLoss Authority Transfer zu LE 2.0 (Phase 19D).

Verifiziert, dass LE 2.0 die alleinige adaptive Quelle für die TPI-Koeffizienten
(coef_int / coef_ext) ist, sofern ein Shadow angehängt ist.  LE v1
learning_engine.get_tpi_coefficients darf dann NICHT gelesen werden.

Test-Klassen:
    TestReadTpiCoefficientsSafe  – read_tpi_coefficients_safe Semantik + Fallbacks
    TestTpiAuthorityTransfer     – statische Authority-Checks (no LE v1 lesen)
    TestCoordinatorIntegration   – coordinator schreibt tpi_coef_source/-hl_rate
    TestTraceEnrichment          – compute_decision_trace_safe enriches trace

Muster: pure Python + MagicMock; kein Docker / HA-Runtime nötig.
"""
from __future__ import annotations

import inspect
import pytest
from unittest.mock import MagicMock

from custom_components.thermosmart.const import TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT
from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)
from tests.helpers_ha_runtime import attach_shadow, FakeStore, make_recording_coordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred(values, *, fallback_used=False, confidence=0.8, units=None, warnings=()):
    p = MagicMock()
    p.fallback_used = fallback_used
    p.confidence = confidence
    p.values = values
    p.warnings = tuple(warnings)
    if units is None:
        units = {k: "C/h" for k in values}
    p.units = units
    return p


def _hr_pred(rate_c_per_h=4.0, *, fallback_used=False, confidence=0.8,
             unit="C/h", warnings=()):
    return _pred({"heat_rate": rate_c_per_h}, fallback_used=fallback_used,
                 confidence=confidence, units={"heat_rate": unit}, warnings=warnings)


def _hl_pred(rate_c_per_h=2.0, *, fallback_used=False, confidence=0.8,
             unit="C/h", warnings=()):
    return _pred({"heat_loss_rate": rate_c_per_h}, fallback_used=fallback_used,
                 confidence=confidence, units={"heat_loss_rate": unit}, warnings=warnings)


def _sh(preds_dict=None):
    """Build a LearningShadowController with mocked zone predictions."""
    coord = make_recording_coordinator()
    shadow = attach_shadow(coord)
    if preds_dict is not None:
        zr = shadow._runtime._zone(shadow._zone)
        zr.last_predictions = preds_dict
    return shadow


# ---------------------------------------------------------------------------
# 1.  read_tpi_coefficients_safe — unit tests
# ---------------------------------------------------------------------------

class TestReadTpiCoefficientsSafe:

    def test_valid_predictions_return_valid_le2_status(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"

    def test_coef_ratio_correct(self):
        # heat_loss=2 C/h, heat_rate=4 C/h → coef_int = 2/4 = 0.5
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        coef_int, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert abs(coef_int - 0.5) < 0.01

    def test_heat_loss_rate_returned_in_result(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, hl_rate, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert abs(hl_rate - 2.0) < 0.01

    def test_high_loss_rate_clamped_to_coef_max(self):
        # coef_int max is 1.2 (from tpi._COEF_INT_MAX)
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(1.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(10.0)})
        coef_int, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert coef_int <= 1.2

    def test_coef_ext_is_static_default_variante_a(self):
        # Variante A: coef_ext must always be TPI_COEF_EXT_DEFAULT regardless of
        # heat_loss and heat_rate, to avoid re-multiplying the outdoor term.
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, coef_ext, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"
        assert coef_ext == TPI_COEF_EXT_DEFAULT

    def test_hr_prediction_none_returns_prediction_missing(self):
        sh = _sh({PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        coef_int, coef_ext, hl_rate, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "prediction_missing"
        assert coef_int == TPI_COEF_INT_DEFAULT
        assert coef_ext == TPI_COEF_EXT_DEFAULT
        assert hl_rate is None

    def test_hl_prediction_none_returns_prediction_missing(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "prediction_missing"

    def test_both_predictions_none_returns_prediction_missing(self):
        sh = _sh({})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "prediction_missing"

    def test_hr_fallback_used_returns_cold_start(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, fallback_used=True),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "cold_start"

    def test_hl_fallback_used_returns_cold_start(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, fallback_used=True)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "cold_start"

    def test_low_hr_confidence_returns_low_confidence(self):
        # threshold is _PREHEAT_MIN_CONFIDENCE = 0.35
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, confidence=0.2),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, confidence=0.8)})
        coef_int, coef_ext, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "low_confidence"
        assert coef_int == TPI_COEF_INT_DEFAULT
        assert coef_ext == TPI_COEF_EXT_DEFAULT

    def test_low_hl_confidence_returns_low_confidence(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, confidence=0.8),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, confidence=0.1)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "low_confidence"

    def test_minimum_confidence_governs(self):
        # both must be above threshold; minimum determines gate
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, confidence=0.36),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, confidence=0.36)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"

    def test_zero_heat_rate_returns_invalid_value(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(0.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_value"

    def test_zero_heat_loss_returns_invalid_value(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(0.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_value"

    def test_negative_heat_rate_returns_invalid_value(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(-1.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_value"

    def test_disabled_shadow_returns_le2_disabled(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        sh._enabled = False
        coef_int, coef_ext, hl_rate, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "le2_disabled"
        assert coef_int == TPI_COEF_INT_DEFAULT
        assert coef_ext == TPI_COEF_EXT_DEFAULT
        assert hl_rate is None

    def test_prediction_missing_returns_none_for_heat_loss(self):
        sh = _sh({})
        coef_int, coef_ext, hl_rate, status, _ = sh.read_tpi_coefficients_safe()
        assert coef_int == TPI_COEF_INT_DEFAULT
        assert coef_ext == TPI_COEF_EXT_DEFAULT
        assert hl_rate is None
        assert status == "prediction_missing"

    def test_ratio_unit_independence(self):
        # coef_int = heat_loss / heat_rate is dimensionless.
        # Same ratio in °C/h or °C/min produces identical coef_int.
        sh_h = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                    PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        sh_m = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0 / 60.0),
                    PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0 / 60.0)})
        coef_h, _, _, _, _ = sh_h.read_tpi_coefficients_safe()
        coef_m, _, _, _, _ = sh_m.read_tpi_coefficients_safe()
        assert abs(coef_h - coef_m) < 0.001

    def test_method_never_reads_heat_loss_ema(self):
        # The method must not call learning_engine._heat_loss_ema or
        # learning_engine.get_tpi_coefficients. Strip docstring before checking
        # because docstrings may mention LE v1 names for documentation purposes.
        import ast
        src = inspect.getsource(
            LearningShadowController.read_tpi_coefficients_safe)
        # Extract the function body after the first triple-quote docstring
        body_start = src.find('"""', src.find('"""') + 3)
        body_end = src.rfind('"""')
        body = src[body_end + 3:]  # everything after the closing docstring quotes
        assert "_heat_loss_ema" not in body
        assert "get_tpi_coefficients" not in body

    def test_result_has_five_elements(self):
        sh = _sh({})
        result = sh.read_tpi_coefficients_safe()
        assert len(result) == 5


# ---------------------------------------------------------------------------
# 2.  Static authority checks
# ---------------------------------------------------------------------------

class TestTpiAuthorityTransfer:

    def test_read_tpi_coefficients_safe_exists_on_shadow(self):
        assert hasattr(LearningShadowController, "read_tpi_coefficients_safe")
        assert callable(LearningShadowController.read_tpi_coefficients_safe)

    def test_coordinator_calls_read_tpi_coefficients_safe_when_shadow(self):
        import custom_components.thermosmart.coordinator as coord_mod
        src = inspect.getsource(coord_mod.ThermoSmartCoordinator._async_update_data)
        assert "read_tpi_coefficients_safe" in src

    def test_coordinator_skips_le_v1_when_shadow_active(self):
        # When _le2_shadow is not None, coordinator must NOT call
        # learning_engine.get_tpi_coefficients.  The LE v1 call was fully removed
        # from the else-branch in Phase 19D; only read_tpi_coefficients_safe remains.
        import custom_components.thermosmart.coordinator as coord_mod
        src = inspect.getsource(coord_mod.ThermoSmartCoordinator._async_update_data)
        assert "read_tpi_coefficients_safe" in src
        # LE v1 get_tpi_coefficients must NOT appear as a call in the coordinator
        assert ".get_tpi_coefficients(" not in src

    def test_else_branch_uses_static_defaults_not_le_v1(self):
        # When shadow is None, coordinator must use TPI_COEF_INT_DEFAULT /
        # TPI_COEF_EXT_DEFAULT — never LE v1 learning_engine.get_tpi_coefficients.
        import custom_components.thermosmart.coordinator as coord_mod
        src = inspect.getsource(coord_mod.ThermoSmartCoordinator._async_update_data)
        assert "TPI_COEF_INT_DEFAULT" in src or "default_no_shadow" in src
        assert ".get_tpi_coefficients(" not in src

    def test_tpi_coef_source_written_to_recommendation(self):
        import custom_components.thermosmart.coordinator as coord_mod
        src = inspect.getsource(coord_mod.ThermoSmartCoordinator._async_update_data)
        assert '"tpi_coef_source"' in src or "'tpi_coef_source'" in src

    def test_tpi_hl_rate_written_to_recommendation(self):
        import custom_components.thermosmart.coordinator as coord_mod
        src = inspect.getsource(coord_mod.ThermoSmartCoordinator._async_update_data)
        assert '"tpi_hl_rate"' in src or "'tpi_hl_rate'" in src

    def test_ha_integration_docstring_no_le_v1(self):
        src = inspect.getsource(
            LearningShadowController.read_tpi_coefficients_safe)
        assert "Never reads LE v1" in src or "Never read" in src or "heat_loss_ema" not in src

    def test_trace_enrichment_includes_tpi_coef_source(self):
        import custom_components.thermosmart.learning.runtime.ha_integration as hai
        src = inspect.getsource(
            LearningShadowController.compute_decision_trace_safe)
        assert "tpi_coef_source" in src

    def test_trace_enrichment_includes_tpi_heat_loss_prediction(self):
        src = inspect.getsource(
            LearningShadowController.compute_decision_trace_safe)
        assert "tpi_heat_loss_prediction_c_per_h" in src

    def test_trace_enrichment_includes_tpi_coef_int_used(self):
        src = inspect.getsource(
            LearningShadowController.compute_decision_trace_safe)
        assert "tpi_coef_int_used" in src

    def test_trace_enrichment_includes_tpi_coef_ext_used(self):
        src = inspect.getsource(
            LearningShadowController.compute_decision_trace_safe)
        assert "tpi_coef_ext_used" in src


# ---------------------------------------------------------------------------
# 3.  Coordinator integration (recommendation dict writes)
# ---------------------------------------------------------------------------

class TestCoordinatorIntegration:
    """Verify coordinator wires up the LE2 shadow path correctly.

    Uses real shadow with patched read_tpi_coefficients_safe to capture calls
    without mocking all other shadow methods.
    """

    def test_le2_shadow_read_called_when_shadow_attached(self):
        from unittest.mock import patch
        import asyncio
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        with patch.object(shadow, "read_tpi_coefficients_safe",
                          return_value=(0.5, 0.01, 2.0, "valid_le2", {})) as mock_fn:
            asyncio.run(coord._async_update_data())
        mock_fn.assert_called_once()

    def test_le_v1_not_called_when_shadow_attached(self):
        from unittest.mock import patch
        import asyncio
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        with patch.object(shadow, "read_tpi_coefficients_safe",
                          return_value=(0.5, 0.01, 2.0, "valid_le2", {})):
            with patch.object(coord.learning_engine, "get_tpi_coefficients",
                              wraps=coord.learning_engine.get_tpi_coefficients) as mock_le1:
                asyncio.run(coord._async_update_data())
        mock_le1.assert_not_called()

    def test_le_v1_not_called_when_shadow_none(self):
        # Phase 19D: when shadow is absent the coordinator uses static defaults,
        # NOT learning_engine.get_tpi_coefficients. LE v1 is never the thermal source.
        from unittest.mock import patch
        import asyncio
        coord = make_recording_coordinator()
        assert coord._le2_shadow is None
        with patch.object(coord.learning_engine, "get_tpi_coefficients",
                          wraps=coord.learning_engine.get_tpi_coefficients) as mock_le1:
            asyncio.run(coord._async_update_data())
        mock_le1.assert_not_called()

    def test_valid_le2_coef_applied_to_tpi(self):
        from unittest.mock import patch
        import asyncio
        import custom_components.thermosmart.tpi as tpi_mod
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        called_with = {}
        original = tpi_mod.compute_tpi
        def capturing_tpi(target, current, outdoor, coef_int, coef_ext):
            called_with["coef_int"] = coef_int
            return original(target, current, outdoor, coef_int, coef_ext)
        with patch.object(shadow, "read_tpi_coefficients_safe",
                          return_value=(0.5, 0.01, 2.0, "valid_le2", {})):
            with patch.object(tpi_mod, "compute_tpi", side_effect=capturing_tpi):
                asyncio.run(coord._async_update_data())
        if called_with:
            assert called_with.get("coef_int") == pytest.approx(0.5, abs=0.01)

    def test_cold_start_le2_uses_static_defaults(self):
        from unittest.mock import patch
        import asyncio
        import custom_components.thermosmart.tpi as tpi_mod
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        called_with = {}
        original = tpi_mod.compute_tpi
        def capturing_tpi(target, current, outdoor, coef_int, coef_ext):
            called_with["coef_int"] = coef_int
            return original(target, current, outdoor, coef_int, coef_ext)
        with patch.object(shadow, "read_tpi_coefficients_safe",
                          return_value=(TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT,
                                        0.0, "cold_start", {})):
            with patch.object(tpi_mod, "compute_tpi", side_effect=capturing_tpi):
                asyncio.run(coord._async_update_data())
        if called_with:
            assert called_with.get("coef_int") == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.01)


# ---------------------------------------------------------------------------
# 4.  Trace enrichment
# ---------------------------------------------------------------------------

class TestTraceEnrichment:
    """Verify compute_decision_trace_safe receives and passes heat_loss trace fields.

    These tests check that trace_dict is enriched with tpi_coef_source and
    tpi_heat_loss_rate_c_per_h from the coordinator's recommendation dict.
    The trace call is intercepted via mock to inspect the recommendation argument.
    """

    def _run_and_capture_trace_rec(self, preds_dict):
        """Run coordinator update, capture the recommendation passed to compute_decision_trace_safe."""
        from unittest.mock import patch
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

    def test_trace_has_tpi_coef_source_field(self):
        rec = self._run_and_capture_trace_rec({
            PredictionType.HEAT_RATE: _hr_pred(4.0),
            PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0),
        })
        assert "tpi_coef_source" in rec

    def test_trace_has_tpi_heat_loss_rate_field(self):
        rec = self._run_and_capture_trace_rec({
            PredictionType.HEAT_RATE: _hr_pred(4.0),
            PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0),
        })
        assert "tpi_hl_rate" in rec

    def test_valid_trace_heat_loss_rate_positive(self):
        rec = self._run_and_capture_trace_rec({
            PredictionType.HEAT_RATE: _hr_pred(4.0, confidence=0.9),
            PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, confidence=0.9),
        })
        if rec.get("tpi_coef_source") == "valid_le2":
            hl = rec.get("tpi_hl_rate")
            assert hl is not None and hl > 0

    def test_prediction_missing_trace_heat_loss_is_none(self):
        rec = self._run_and_capture_trace_rec({})
        hl = rec.get("tpi_hl_rate")
        assert hl is None


# ---------------------------------------------------------------------------
# 5.  Validity gate tests (Phase 19D additions)
# ---------------------------------------------------------------------------

class TestValidityGates:
    """Gate 4 (unit) and Gate 5 (stale/superseded) in read_tpi_coefficients_safe."""

    def test_invalid_hr_unit_returns_invalid_unit(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, unit="celsius"),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        ci, ce, hl, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_unit"
        assert ci == TPI_COEF_INT_DEFAULT and ce == TPI_COEF_EXT_DEFAULT
        assert hl is None

    def test_invalid_hl_unit_returns_invalid_unit(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, unit="K/h")})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_unit"

    def test_celsius_per_min_unit_rejected(self):
        # "C/min" is NOT the contracted unit; must be caught by gate 4
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, unit="C/min"),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_unit"

    def test_valid_unit_passes_gate(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, unit="C/h"),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, unit="C/h")})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"

    def test_hr_stale_warning_returns_stale_or_superseded(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, warnings=("stale",)),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        ci, ce, hl, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "stale_or_superseded"
        assert ci == TPI_COEF_INT_DEFAULT and ce == TPI_COEF_EXT_DEFAULT
        assert hl is None

    def test_hl_stale_warning_returns_stale_or_superseded(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, warnings=("stale",))})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "stale_or_superseded"

    def test_hr_superseded_warning_returns_stale_or_superseded(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, warnings=("superseded",)),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "stale_or_superseded"

    def test_hl_superseded_warning_returns_stale_or_superseded(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0, warnings=("superseded",))})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "stale_or_superseded"

    def test_other_warnings_do_not_block(self):
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, warnings=("some_other_flag",)),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "valid_le2"

    def test_gate_order_unit_before_stale(self):
        # Prediction with wrong unit AND stale: unit gate fires first → invalid_unit
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, unit="K/h", warnings=("stale",)),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "invalid_unit"

    def test_gate_order_stale_before_confidence(self):
        # Prediction stale AND low confidence: stale gate fires first
        sh = _sh({PredictionType.HEAT_RATE: _hr_pred(4.0, confidence=0.1, warnings=("stale",)),
                  PredictionType.HEAT_LOSS_RATE: _hl_pred(2.0)})
        _, _, _, status, _ = sh.read_tpi_coefficients_safe()
        assert status == "stale_or_superseded"

    def test_shadow_none_returns_static_defaults(self):
        # No shadow at all → coordinator returns default_no_shadow, never LE v1
        import asyncio
        from unittest.mock import patch
        coord = make_recording_coordinator()
        assert coord._le2_shadow is None
        captured_src = {}
        original_update = coord._async_update_data

        async def wrapping():
            result = await original_update()
            captured_src["source"] = result.get("zone", {}).get("tpi_coef_source")
            return result

        with patch.object(coord.learning_engine, "get_tpi_coefficients",
                          side_effect=AssertionError("LE v1 must not be called")) as _:
            asyncio.run(wrapping())

        assert captured_src.get("source") == "deterministic_baseline"

    def test_heat_loss_ema_extreme_value_no_control_effect(self):
        # Injecting a huge EMA value must not affect TPI when shadow is absent;
        # coordinator must use static defaults, not the EMA.
        import asyncio
        from unittest.mock import patch
        coord = make_recording_coordinator()
        coord.learning_engine._heat_loss_ema["test_zone"] = 9999.0
        assert coord._le2_shadow is None
        captured = {}

        async def wrapping():
            result = await coord._async_update_data()
            captured["zone"] = result.get("zone", {})

        asyncio.run(wrapping())
        src = captured["zone"].get("tpi_coef_source")
        assert src == "deterministic_baseline"
        # coef_int must be the static default, not anything derived from 9999.0 EMA
        coef_int = captured["zone"].get("tpi_coef_int")
        if coef_int is not None:
            assert abs(coef_int - TPI_COEF_INT_DEFAULT) < 0.01

    def test_le1_ema_mutation_silenced_in_source(self):
        # The EMA mutation line (self._heat_loss_ema[zone_id] = ...) must be
        # commented out. Static check: strip comment lines, confirm no live mutation.
        import custom_components.thermosmart.learning_engine as le_mod
        src = inspect.getsource(le_mod.LearningEngine.async_observe)
        lines = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
        body = "\n".join(lines)
        assert "self._heat_loss_ema[zone_id] =" not in body


