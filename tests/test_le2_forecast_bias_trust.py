"""Phase 19A-C: Forecast Bias and Forecast Trust transfer to LE 2.0.

Semantik-Invarianten (verbindlich):
  FORECAST_TRUST = 1.0  darf niemals als Fallback erscheinen.
  Fehlende Evidenz → trust=0.0, status="not_available" (kein erfundenes Vertrauen).
  Cold-Start (fallback_used=True) → trust=0.0, status="cold_start".
  Gültiger Trust-Wert 0.0 und "not_available" sind über status unterscheidbar.
  Bias-Anwendung nur bei status="valid" und ausreichendem Trust.
  Fehlender Forecast darf lokales Preheat nicht begrenzen.
  Keine LE-v1-Forecast-Reads oder -Writes.

Test-Patterns:
  Pattern A  – _compute_recommendation() direkt via make_coordinator()
  Pattern B  – _async_update_data() via make_recording_coordinator()
  Pattern C  – Shadow-Unit-Tests (attach_shadow / FakeStore)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from custom_components.thermosmart.const import HEATING_MODE_AUTO
from custom_components.thermosmart.learning.contracts import Prediction, PredictionType
from custom_components.thermosmart.learning.decision.runtime_adapter import (
    validated_prediction_read,
    READ_OK, READ_MISSING, READ_STALE, READ_SUPERSEDED, READ_UNIT_MISMATCH, READ_WRONG_TYPE,
)
from tests.helpers import make_coordinator, make_state, set_hass_states
from tests.helpers_ha_runtime import attach_shadow, FakeStore, make_recording_coordinator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _trust_pred(trust: float, *, fallback: bool = False) -> Prediction:
    if fallback:
        return Prediction(
            prediction_type=PredictionType.FORECAST_TRUST,
            values={"forecast_trust": trust},
            units={"forecast_trust": "score"},
            confidence=0.15, reliability=0.5,
            model_version=1, parameter_version=1,
            prior_contribution=1.0, learned_contribution=0.0,
            fallback_used=True, evidence_count=0,
        )
    return Prediction(
        prediction_type=PredictionType.FORECAST_TRUST,
        values={"forecast_trust": trust},
        units={"forecast_trust": "score"},
        confidence=0.75, reliability=0.9,
        model_version=1, parameter_version=1,
        prior_contribution=0.0, learned_contribution=1.0,
        fallback_used=False, evidence_count=30,
    )


def _bias_pred(bias_c: float, *, fallback: bool = False) -> Prediction:
    if fallback:
        return Prediction(
            prediction_type=PredictionType.FORECAST_BIAS,
            values={"forecast_bias": 0.0},
            units={"forecast_bias": "celsius"},
            confidence=0.15, reliability=0.5,
            model_version=1, parameter_version=1,
            prior_contribution=1.0, learned_contribution=0.0,
            fallback_used=True, evidence_count=0,
        )
    return Prediction(
        prediction_type=PredictionType.FORECAST_BIAS,
        values={"forecast_bias": bias_c},
        units={"forecast_bias": "celsius"},
        confidence=0.75, reliability=0.9,
        model_version=1, parameter_version=1,
        prior_contribution=0.0, learned_contribution=1.0,
        fallback_used=False, evidence_count=30,
    )


def _shadow_with_preds(coord, trust_pred=None, bias_pred=None):
    """Attach a real shadow to coord and inject last_predictions directly."""
    sh = attach_shadow(coord, store=FakeStore())
    zr = sh.runtime._zone(coord.zone_id)
    preds = {}
    if trust_pred is not None:
        preds[PredictionType.FORECAST_TRUST] = trust_pred
    if bias_pred is not None:
        preds[PredictionType.FORECAST_BIAS] = bias_pred
    zr.last_predictions = preds
    return sh


def _mock_shadow(trust=1.0, bias_c=0.0, *, trust_status="valid") -> MagicMock:
    """Return a MagicMock shadow that returns the given trust/bias values.

    read_forecast_trust_safe returns a (float, str) tuple.
    When trust_status != 'valid', trust is forced to 0.0 as the shadow would do.
    """
    sh = MagicMock()
    actual_trust = float(trust) if trust_status == "valid" else 0.0
    sh.read_forecast_trust_safe.return_value = (actual_trust, trust_status)
    sh.read_forecast_bias_safe.return_value = bias_c
    # Early cutoff / preheat / regime are not relevant to forecast tests.
    sh.read_early_cutoff_safe.return_value = (0.0, "not_available")
    sh.read_preheat_minutes_safe.return_value = (0.0, "deterministic_baseline")
    sh.read_regime_safe.return_value = None
    sh.read_onset_delay_safe.return_value = (5.0, "cold_start_prior")
    sh.confidence_display.return_value = 0.0
    sh.compute_decision_trace_safe = MagicMock()
    return sh


def _make_coord_night(*, trust=1.0, bias_c=0.0, raw_suppression=0.5,
                      base_target=18.5, trust_status="valid"):
    """Coordinator primed for the AUTO night/transition branch (Pattern A).

    base_target (18.5) < comfort_temp (21.0) → else-branch where LE 2.0 reads happen.
    """
    coord = make_coordinator()
    coord.attach_le2_shadow(_mock_shadow(trust=trust, bias_c=bias_c,
                                        trust_status=trust_status))
    coord.weather_engine.compute_forecast_suppression.return_value = raw_suppression
    coord.learning_engine.async_get_base_target = AsyncMock(return_value=base_target)
    coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=0)
    return coord


async def _rec(coord, *, indoor="17.0", weather=None) -> dict:
    """Call _compute_recommendation directly (Pattern A)."""
    set_hass_states(coord, {"sensor.test_temp": make_state(indoor)})
    return await coord._compute_recommendation(
        coord.entry.data, weather or {"temperature": 10.0}, HEATING_MODE_AUTO)


# ---------------------------------------------------------------------------
# 1. Read Gate – FORECAST_TRUST
# ---------------------------------------------------------------------------

class TestForecastTrustReadGate:
    def test_ok_mid_value(self):
        val, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: _trust_pred(0.75)}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score")
        assert st == READ_OK and val == pytest.approx(0.75)

    def test_ok_zero(self):
        val, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: _trust_pred(0.0)}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score")
        assert st == READ_OK and val == pytest.approx(0.0)

    def test_ok_one(self):
        val, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: _trust_pred(1.0)}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score")
        assert st == READ_OK and val == pytest.approx(1.0)

    def test_wrong_unit_celsius_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: _trust_pred(0.8)}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_UNIT_MISMATCH

    def test_missing_rejected(self):
        _, st = validated_prediction_read(
            {}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score")
        assert st == READ_MISSING

    def test_unknown_feature_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: _trust_pred(0.8)}, "no_such_feature",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score")
        assert st == READ_WRONG_TYPE

    def test_stale_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: _trust_pred(0.8)}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score",
            age_seconds=7201.0, max_age_seconds=7200.0)
        assert st == READ_STALE

    def test_decision_id_mismatch_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: _trust_pred(0.8)}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score",
            prediction_decision_id="old-id", current_decision_id="new-id")
        assert st == READ_SUPERSEDED


# ---------------------------------------------------------------------------
# 2. Read Gate – FORECAST_BIAS
# ---------------------------------------------------------------------------

class TestForecastBiasReadGate:
    def test_ok_positive(self):
        val, st = validated_prediction_read(
            {PredictionType.FORECAST_BIAS: _bias_pred(1.2)}, "forecast_bias",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_OK and val == pytest.approx(1.2)

    def test_ok_negative(self):
        val, st = validated_prediction_read(
            {PredictionType.FORECAST_BIAS: _bias_pred(-0.8)}, "forecast_bias",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_OK and val == pytest.approx(-0.8)

    def test_ok_zero(self):
        val, st = validated_prediction_read(
            {PredictionType.FORECAST_BIAS: _bias_pred(0.0)}, "forecast_bias",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_OK and val == pytest.approx(0.0)

    def test_wrong_unit_score_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_BIAS: _bias_pred(1.0)}, "forecast_bias",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score")
        assert st == READ_UNIT_MISMATCH

    def test_stale_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_BIAS: _bias_pred(1.0)}, "forecast_bias",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius",
            age_seconds=3601.0, max_age_seconds=3600.0)
        assert st == READ_STALE


# ---------------------------------------------------------------------------
# 3. Bias and Trust must not be transposed
# ---------------------------------------------------------------------------

class TestBiasTrustNotTransposed:
    def test_trust_feature_returns_trust_value_not_bias(self):
        preds = {
            PredictionType.FORECAST_TRUST: _trust_pred(0.8),
            PredictionType.FORECAST_BIAS: _bias_pred(1.5),
        }
        val, st = validated_prediction_read(
            preds, "forecast_trust", zone_id="z1", prediction_zone_id="z1",
            expected_unit="score")
        assert st == READ_OK and val == pytest.approx(0.8)  # not 1.5

    def test_bias_feature_returns_bias_value_not_trust(self):
        preds = {
            PredictionType.FORECAST_TRUST: _trust_pred(0.8),
            PredictionType.FORECAST_BIAS: _bias_pred(1.5),
        }
        val, st = validated_prediction_read(
            preds, "forecast_bias", zone_id="z1", prediction_zone_id="z1",
            expected_unit="celsius")
        assert st == READ_OK and val == pytest.approx(1.5)  # not 0.8

    def test_trust_unit_is_score_not_celsius(self):
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: _trust_pred(0.7)}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_UNIT_MISMATCH

    def test_bias_unit_is_celsius_not_score(self):
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_BIAS: _bias_pred(0.5)}, "forecast_bias",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score")
        assert st == READ_UNIT_MISMATCH


# ---------------------------------------------------------------------------
# 4. Shadow: read_forecast_trust_safe() returns (float, str) tuple
# ---------------------------------------------------------------------------

class TestShadowForecastTrustReads:
    def test_valid_prediction_returns_value_and_valid_status(self):
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, trust_pred=_trust_pred(0.65))
        val, st = sh.read_forecast_trust_safe()
        assert st == "valid" and val == pytest.approx(0.65)

    def test_valid_zero_is_genuinely_valid(self):
        """Trust = 0.0 with real evidence is "valid", not "not_available"."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, trust_pred=_trust_pred(0.0))
        val, st = sh.read_forecast_trust_safe()
        assert st == "valid" and val == pytest.approx(0.0)

    def test_valid_one_is_genuinely_valid(self):
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, trust_pred=_trust_pred(1.0))
        val, st = sh.read_forecast_trust_safe()
        assert st == "valid" and val == pytest.approx(1.0)

    def test_no_prediction_returns_zero_missing(self):
        """No prediction at all → (0.0, 'missing'), NOT (1.0, ...)."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        val, st = sh.read_forecast_trust_safe()
        assert val == pytest.approx(0.0)
        assert st == "missing"

    def test_cold_start_fallback_used_returns_zero_cold_start(self):
        """Cold-start prior (fallback_used=True) → (0.0, 'cold_start').

        The neutral prior (e.g., 0.5) must NOT be returned as a trust value
        because it is not derived from evidence.
        """
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, trust_pred=_trust_pred(0.5, fallback=True))
        val, st = sh.read_forecast_trust_safe()
        assert val == pytest.approx(0.0)
        assert st == "cold_start"

    def test_disabled_shadow_returns_zero_not_available(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._enabled = False
        val, st = sh.read_forecast_trust_safe()
        assert val == pytest.approx(0.0) and st == "not_available"

    def test_internal_error_returns_zero_not_available(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime = None  # triggers AttributeError inside method
        val, st = sh.read_forecast_trust_safe()
        assert val == pytest.approx(0.0) and st == "not_available"

    def test_stale_prediction_returns_zero_stale(self):
        """Stale prediction must be rejected with status='stale' and trust=0.0."""
        # We inject a prediction but the gate sees it as stale via age_seconds.
        # We test this via the Read Gate directly (shadow age is not injectable here).
        pred = _trust_pred(0.8)
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: pred}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score",
            age_seconds=7201.0, max_age_seconds=7200.0)
        assert st == READ_STALE

    def test_superseded_returns_zero(self):
        pred = _trust_pred(0.8)
        _, st = validated_prediction_read(
            {PredictionType.FORECAST_TRUST: pred}, "forecast_trust",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score",
            prediction_decision_id="old", current_decision_id="new")
        assert st == READ_SUPERSEDED

    def test_not_available_and_valid_zero_are_distinguishable_by_status(self):
        """(0.0, 'not_available') and (0.0, 'valid') must carry different status strings."""
        coord = make_coordinator()
        # valid 0.0
        sh_valid = _shadow_with_preds(coord, trust_pred=_trust_pred(0.0))
        val_v, st_v = sh_valid.read_forecast_trust_safe()
        # not_available
        coord2 = make_coordinator()
        sh_na = attach_shadow(coord2, store=FakeStore())
        val_na, st_na = sh_na.read_forecast_trust_safe()
        assert val_v == pytest.approx(0.0)
        assert val_na == pytest.approx(0.0)
        assert st_v == "valid"
        assert st_na != "valid"   # "missing", never "valid"


# ---------------------------------------------------------------------------
# 5. Shadow: read_forecast_bias_safe() returns float 0.0 when unavailable
# ---------------------------------------------------------------------------

class TestShadowForecastBiasReads:
    def test_reads_positive_prediction(self):
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, bias_pred=_bias_pred(0.8))
        assert sh.read_forecast_bias_safe() == pytest.approx(0.8)

    def test_reads_negative_prediction(self):
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, bias_pred=_bias_pred(-0.6))
        assert sh.read_forecast_bias_safe() == pytest.approx(-0.6)

    def test_reads_exact_zero(self):
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, bias_pred=_bias_pred(0.0))
        assert sh.read_forecast_bias_safe() == pytest.approx(0.0)

    def test_no_prediction_returns_zero(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        assert sh.read_forecast_bias_safe() == pytest.approx(0.0)

    def test_cold_start_returns_zero(self):
        """Cold-start BIAS (fallback_used=True) must return 0.0°C."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, bias_pred=_bias_pred(0.0, fallback=True))
        assert sh.read_forecast_bias_safe() == pytest.approx(0.0)

    def test_disabled_returns_zero(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._enabled = False
        assert sh.read_forecast_bias_safe() == pytest.approx(0.0)

    def test_internal_error_returns_zero(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime = None
        assert sh.read_forecast_bias_safe() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Coordinator: LE v1 forecast methods no longer called (Pattern B)
# ---------------------------------------------------------------------------

class TestNoLEV1ForecastCalls:
    async def test_get_forecast_bias_not_called(self):
        coord = make_recording_coordinator(indoor="18.5")
        coord._active_control = True
        coord.weather_engine.compute_forecast_suppression.return_value = 0.5
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        await coord._async_update_data()
        coord.learning_engine.get_forecast_bias.assert_not_called()

    async def test_record_forecast_decision_not_called(self):
        coord = make_recording_coordinator(indoor="18.5")
        coord._active_control = True
        coord.weather_engine.compute_forecast_suppression.return_value = 0.5
        coord.weather_engine.async_get_data = AsyncMock(
            return_value={"temperature": 10.0, "forecast_high": 15.0})
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        await coord._async_update_data()
        coord.learning_engine.record_forecast_decision.assert_not_called()

    async def test_evaluate_forecast_decisions_not_called(self):
        coord = make_recording_coordinator(indoor="18.5")
        coord._active_control = True
        await coord._async_update_data()
        coord.learning_engine.evaluate_forecast_decisions.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Recommendation dict: forecast_trust, forecast_bias_c, forecast_used,
#    forecast_trust_status all present (Pattern A)
# ---------------------------------------------------------------------------

class TestRecommendationForecastAttributes:
    async def test_valid_trust_in_recommendation(self):
        coord = _make_coord_night(trust=0.7)
        rec = await _rec(coord)
        assert rec["forecast_trust"] == pytest.approx(0.7)
        assert rec["forecast_trust_status"] == "valid"
        assert rec["forecast_used"] is True

    async def test_valid_bias_c_in_recommendation(self):
        coord = _make_coord_night(trust=0.8, bias_c=0.3)
        rec = await _rec(coord)
        assert rec["forecast_bias_c"] == pytest.approx(0.3)

    async def test_negative_bias_c_in_recommendation(self):
        coord = _make_coord_night(trust=0.8, bias_c=-0.5)
        rec = await _rec(coord)
        assert rec["forecast_bias_c"] == pytest.approx(-0.5)

    async def test_no_shadow_yields_not_available_status(self):
        """Without shadow: trust=0.0, status='not_available', forecast_used=False."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        rec = await _rec(coord)
        assert rec["forecast_trust"] == pytest.approx(0.0)
        assert rec["forecast_trust_status"] == "not_available"
        assert rec["forecast_used"] is False
        assert rec["forecast_bias_c"] == pytest.approx(0.0)

    async def test_cold_start_trust_yields_zero_and_cold_start_status(self):
        """Cold-start shadow: trust=0.0, status='cold_start', forecast_used=False."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, trust_pred=_trust_pred(0.5, fallback=True))
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        # Prevent preheat activation so coordinator stays in the night branch
        # where read_forecast_trust_safe() is called.
        coord._minutes_until_next_comfort = MagicMock(return_value=None)
        rec = await _rec(coord)
        assert rec["forecast_trust"] == pytest.approx(0.0)
        assert rec["forecast_trust_status"] == "cold_start"
        assert rec["forecast_used"] is False

    async def test_forecast_used_true_only_when_valid(self):
        """forecast_used is True only when trust status is 'valid'."""
        coord_valid = _make_coord_night(trust=0.6)
        rec_valid = await _rec(coord_valid)
        assert rec_valid["forecast_used"] is True

        coord_na = make_coordinator()
        coord_na.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        rec_na = await _rec(coord_na)
        assert rec_na["forecast_used"] is False


# ---------------------------------------------------------------------------
# 8. No suppression when forecast_used=False (fehlender Forecast)
# ---------------------------------------------------------------------------

class TestNoSuppressionWithoutForecast:
    async def test_no_shadow_no_forecast_suppression(self):
        """Without shadow: trust=0.0 → biased_suppression=1.0 → no suppression."""
        coord = make_coordinator()
        coord.weather_engine.compute_forecast_suppression.return_value = 0.6
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        rec = await _rec(coord)
        assert rec["forecast_suppression"] == 0  # trust=0.0 → biased_suppression=1.0
        assert rec["forecast_used"] is False

    async def test_cold_start_trust_no_forecast_suppression(self):
        """Cold-start shadow: trust=0.0 → no suppression."""
        coord = make_coordinator()
        _shadow_with_preds(coord, trust_pred=_trust_pred(0.5, fallback=True))
        coord.weather_engine.compute_forecast_suppression.return_value = 0.6
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        rec = await _rec(coord)
        assert rec["forecast_suppression"] == 0
        assert rec["forecast_used"] is False

    async def test_missing_trust_no_suppression(self):
        """Shadow present but no trust prediction: trust=0.0 → no suppression."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())  # no predictions injected
        coord.weather_engine.compute_forecast_suppression.return_value = 0.6
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        # Prevent preheat activation so coordinator stays in the night branch.
        coord._minutes_until_next_comfort = MagicMock(return_value=None)
        rec = await _rec(coord)
        assert rec["forecast_suppression"] == 0
        assert rec["forecast_trust_status"] == "missing"

    async def test_valid_zero_trust_no_suppression(self):
        """Valid trust = 0.0 (real evidence, zero reliability) → no suppression."""
        coord = _make_coord_night(trust=0.0, raw_suppression=0.6)
        rec = await _rec(coord)
        assert rec["forecast_suppression"] == 0
        assert rec["forecast_trust_status"] == "valid"
        assert rec["forecast_used"] is True


# ---------------------------------------------------------------------------
# 9. Trust gates suppression strength
# ---------------------------------------------------------------------------

class TestTrustGatesSuppressionStrength:
    async def test_full_trust_applies_full_raw_suppression(self):
        coord = _make_coord_night(trust=1.0, raw_suppression=0.5)
        rec = await _rec(coord)
        # biased_suppression = 1 - (1-0.5)*1.0 = 0.5 → suppression_pct = 50
        assert rec["forecast_suppression"] == 50
        assert rec["forecast_trust_status"] == "valid"

    async def test_low_trust_reduces_suppression_effect(self):
        coord = _make_coord_night(trust=0.1, raw_suppression=0.5)
        rec = await _rec(coord)
        # biased_suppression = 1 - 0.5*0.1 = 0.95 → suppression_pct = 5
        assert rec["forecast_suppression"] == 5

    async def test_zero_trust_no_suppression(self):
        coord = _make_coord_night(trust=0.0, raw_suppression=0.3)
        rec = await _rec(coord)
        assert rec["forecast_suppression"] == 0
        assert rec["forecast_trust_status"] == "valid"  # genuine 0.0, not unavailable

    async def test_mid_trust_partial_suppression(self):
        coord = _make_coord_night(trust=0.5, raw_suppression=0.4)
        rec = await _rec(coord)
        # biased_suppression = 1 - 0.6*0.5 = 0.7 → suppression_pct = 30
        assert rec["forecast_suppression"] == 30


# ---------------------------------------------------------------------------
# 10. Bias correction: trust gate, noise floor, hard cap (Pattern A)
# ---------------------------------------------------------------------------

class TestBiasCorrectionGating:
    _NIGHT = 18.0
    _BASE = 18.5
    _RS = 0.5

    async def _effective(self, *, trust, bias_c, indoor="17.5", trust_status="valid"):
        coord = _make_coord_night(trust=trust, bias_c=bias_c, raw_suppression=self._RS,
                                  base_target=self._BASE, trust_status=trust_status)
        return (await _rec(coord, indoor=indoor))["effective_target"]

    async def test_low_trust_prevents_bias_correction(self):
        """trust (0.2) < 0.35 → bias must not be applied even if non-zero."""
        eff_no_bias = await self._effective(trust=0.2, bias_c=0.0)
        eff_with_bias = await self._effective(trust=0.2, bias_c=1.0)
        assert eff_with_bias == eff_no_bias

    async def test_not_available_trust_prevents_bias_correction(self):
        """trust 'not_available' → forecast_used=False → bias NOT applied."""
        coord = _make_coord_night(trust=0.0, bias_c=1.0, trust_status="not_available")
        rec = await _rec(coord, indoor="17.5")
        assert rec["forecast_used"] is False
        # effective_target is purely local – not shifted by bias
        night = self._NIGHT
        base = self._BASE
        # biased_suppression = 1 - (1-_RS)*0.0 = 1.0 → effective = raw_target
        expected_effective = round(night + 1.0 * (base - night), 1)  # 18.5
        assert rec["effective_target"] == expected_effective

    async def test_sufficient_trust_applies_positive_bias(self):
        """trust (0.6) >= 0.35 and bias (0.3) >= 0.05 → raw_target increased."""
        eff_bias = await self._effective(trust=0.6, bias_c=0.3)
        eff_none = await self._effective(trust=0.6, bias_c=0.0)
        assert eff_bias > eff_none

    async def test_sufficient_trust_applies_negative_bias(self):
        """trust (0.6) >= 0.35 and |bias| (-0.4) >= 0.05 → raw_target decreased."""
        eff_neg = await self._effective(trust=0.6, bias_c=-0.4)
        eff_none = await self._effective(trust=0.6, bias_c=0.0)
        assert eff_neg < eff_none

    async def test_sufficient_trust_positive_bias_exact_target(self):
        """Exact effective_target with trust=0.6, bias=0.3, no blend (delta=1.0<1.5)."""
        eff = await self._effective(trust=0.6, bias_c=0.3)
        corrected = round(self._BASE + 0.3, 1)   # 18.8
        sup = 1.0 - (1.0 - self._RS) * 0.6       # 0.7
        expected = round(self._NIGHT + sup * (corrected - self._NIGHT), 1)
        assert eff == expected

    async def test_bias_below_noise_floor_not_applied(self):
        """|bias| (0.03) < 0.05 noise floor → no correction applied."""
        eff_small = await self._effective(trust=0.8, bias_c=0.03)
        eff_zero = await self._effective(trust=0.8, bias_c=0.0)
        assert eff_small == eff_zero

    async def test_bias_capped_at_positive_max(self):
        """bias_c = 5.0 gives same effective_target as bias_c = 2.0 (cap = 2.0°C)."""
        eff_big = await self._effective(trust=0.9, bias_c=5.0)
        eff_cap = await self._effective(trust=0.9, bias_c=2.0)
        assert eff_big == eff_cap

    async def test_bias_capped_at_negative_max(self):
        """bias_c = -5.0 gives same effective_target as bias_c = -2.0 (cap = -2.0°C)."""
        eff_big = await self._effective(trust=0.9, bias_c=-5.0)
        eff_cap = await self._effective(trust=0.9, bias_c=-2.0)
        assert eff_big == eff_cap

    async def test_bias_not_read_when_trust_invalid(self):
        """When trust status is not 'valid', read_forecast_bias_safe must NOT be called."""
        coord = _make_coord_night(trust=0.0, trust_status="stale", bias_c=1.0)
        await _rec(coord, indoor="17.5")
        coord._le2_shadow.read_forecast_bias_safe.assert_not_called()


# ---------------------------------------------------------------------------
# 11. Forecast unavailable must not block preheat or local thermal path
# ---------------------------------------------------------------------------

class TestForecastUnavailableNoBlockage:
    async def test_preheat_path_bypasses_trust_reads(self):
        """In preheat branch, LE 2.0 reads are skipped – suppression stays 0."""
        coord = make_coordinator()
        sh = _mock_shadow(trust=0.0)
        # Preheat comes from LE2 now; provide valid preheat so preheat_active=True.
        sh.read_preheat_minutes_safe.return_value = (30.0, "valid")
        coord.attach_le2_shadow(sh)
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        coord._minutes_until_next_comfort = MagicMock(return_value=20)
        rec = await _rec(coord)
        assert rec["preheat_active"] is True
        assert rec["forecast_suppression"] == 0

    async def test_comfort_path_bypasses_trust_reads(self):
        """Comfort branch (base_target >= comfort_temp) bypasses trust reads."""
        coord = make_coordinator()
        coord.attach_le2_shadow(_mock_shadow(trust=0.0))
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        rec = await _rec(coord, indoor="19.0")
        assert rec["forecast_suppression"] == 0

    async def test_no_shadow_local_path_active(self):
        """Without shadow, local path computes targets without forecast interference."""
        coord = make_coordinator()
        coord.weather_engine.compute_forecast_suppression.return_value = 0.6
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        rec = await _rec(coord)
        # trust=0.0 (not_available) → no suppression
        assert rec["forecast_suppression"] == 0
        assert rec["forecast_trust_status"] == "not_available"
        assert rec["forecast_used"] is False


# ---------------------------------------------------------------------------
# 12. Weather Engine is unaffected by LE 2.0 forecast values
# ---------------------------------------------------------------------------

class TestWeatherEngineUnaffected:
    async def test_weather_offset_unaffected_by_low_trust(self):
        """weather_engine.compute_temperature_offset() is always called."""
        coord = make_coordinator()
        coord.attach_le2_shadow(_mock_shadow(trust=0.0))
        coord.weather_engine.compute_temperature_offset.return_value = 1.5
        coord.weather_engine.compute_forecast_suppression.return_value = 1.0
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        rec = await _rec(coord, indoor="21.5")
        assert rec["weather_offset"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 13. Learning failure must not affect heating
# ---------------------------------------------------------------------------

class TestLearningFailureIsolation:
    def test_trust_safe_catches_internal_error_returns_zero_not_available(self):
        """Shadow method catches AttributeError on broken runtime; returns (0.0, 'not_available')."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime = None
        val, st = sh.read_forecast_trust_safe()
        assert val == pytest.approx(0.0) and st == "not_available"

    def test_bias_safe_catches_internal_error_returns_zero(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime = None
        assert sh.read_forecast_bias_safe() == pytest.approx(0.0)

    async def test_broken_shadow_leaves_heating_intact_no_suppression(self):
        """Broken shadow: trust=0.0 (not_available) → no suppression, heating continues."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime = None  # both safe methods return fallbacks
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        coord.weather_engine.compute_forecast_suppression.return_value = 0.6
        rec = await _rec(coord)
        assert rec["effective_target"] is not None
        assert rec["forecast_trust"] == pytest.approx(0.0)
        assert rec["forecast_trust_status"] == "not_available"
        assert rec["forecast_used"] is False
        # No suppression when not_available
        assert rec["forecast_suppression"] == 0
