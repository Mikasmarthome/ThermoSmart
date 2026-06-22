"""Phase 13 ForecastModel trust / bias / correction / confidence tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.models import (
    ForecastModel,
    ForecastParameters,
    ForecastPredictionContext,
    ForecastUpdateContext,
)
from tests.helpers_forecast import decision, evaluation, pair


def m(params=None):
    return ForecastModel("lz", params)


def trained(n=10, forecast=10.0, observed=10.3, horizon_h=6.0, source=None):
    model = m()
    for i in range(n):
        off = i * 24.0
        model.register_decision(decision(f"d{i}", forecast, created_offset_h=off))
        ctx = (ForecastUpdateContext(source_decision_id=f"d{i}", forecast_source_key=source,
                                     horizon_s=horizon_h * 3600) if source else None)
        model.update(evaluation(f"e{i}", f"d{i}", observed, created_offset_h=off,
                                horizon_h=horizon_h), ctx)
    return model


def ctx(**kw):
    return ForecastPredictionContext(**kw)


class TestTrust:
    def test_general_trust(self):
        tr = trained().predict_trust(ctx())
        assert 0.0 <= tr.trust <= 1.0 and tr.available and not tr.fallback_used

    def test_horizon_bucket(self):
        tr = trained(horizon_h=6.0).predict_trust(ctx(horizon_s=6 * 3600))
        assert tr.horizon_bucket.startswith("h")
        assert "horizon_bucket" in tr.reason_codes

    def test_source_bucket(self):
        tr = trained(source="metno").predict_trust(ctx(source_key="metno", horizon_s=6 * 3600))
        assert tr.source_bucket == "src:metno"

    def test_general_fallback_when_no_bucket_support(self):
        tr = trained(n=2).predict_trust(ctx(horizon_s=6 * 3600))
        assert tr.horizon_bucket in ("general", "h1")  # falls back to general state

    def test_neutral_prior_when_empty(self):
        tr = m().predict_trust(ctx(horizon_s=6 * 3600))
        assert tr.fallback_used and "neutral_prior" in tr.reason_codes

    def test_high_error_low_trust(self):
        good = trained(observed=10.1).predict_trust(ctx())
        bad = trained(observed=13.0).predict_trust(ctx())
        assert bad.trust < good.trust

    def test_high_bias_reduces_trust(self):
        biased = trained(observed=12.5).predict_trust(ctx())
        assert biased.bias_c > 0 and biased.trust < 0.8

    def test_cold_start_low(self):
        tr = trained(n=1).predict_trust(ctx())
        assert tr.trust <= 0.6

    def test_trust_values_finite(self):
        for model in (m(), trained()):
            tr = model.predict_trust(ctx())
            assert math.isfinite(tr.trust) and math.isfinite(tr.bias_c)


class TestNoForecast:
    def test_unavailable_context(self):
        tr = trained().predict_trust(ctx(forecast_available=False))
        assert not tr.available and tr.fallback_used
        assert "forecast_unavailable" in tr.reason_codes

    def test_empty_model_unavailable_or_prior(self):
        tr = m().predict_trust(ctx())
        assert tr.fallback_used

    def test_predict_returns_prediction(self):
        p = trained().predict(ctx())
        assert p.prediction_type is PredictionType.FORECAST_TRUST
        assert math.isfinite(p.control_value("forecast_trust"))

    def test_predict_no_nan(self):
        p = m().predict(ctx())
        assert math.isfinite(p.values["forecast_trust"])


class TestBiasCorrection:
    def test_bias_prediction(self):
        p = trained(observed=12.0).predict_bias(ctx())
        assert p.prediction_type is PredictionType.FORECAST_BIAS
        assert p.values["forecast_bias"] > 0

    def test_corrected_forecast_applies_bias(self):
        model = trained(forecast=10.0, observed=12.0, n=12)  # +2 bias
        p = model.predict_corrected_forecast(ctx(raw_forecast_temperature_c=10.0))
        assert p.values["corrected_forecast"] > 10.0
        assert "bias_correction_applied" in p.reason_codes

    def test_correction_hard_capped(self):
        model = trained(forecast=10.0, observed=14.0, n=12)  # +4 bias, cap 2.0
        p = model.predict_corrected_forecast(ctx(raw_forecast_temperature_c=10.0))
        assert p.values["applied_bias"] <= ForecastParameters().bias_correction_max_c + 1e-9

    def test_no_correction_low_evidence(self):
        model = trained(forecast=10.0, observed=12.0, n=3)
        p = model.predict_corrected_forecast(ctx(raw_forecast_temperature_c=10.0))
        assert p.values["applied_bias"] == 0.0
        assert "insufficient_evidence_for_correction" in p.reason_codes

    def test_corrected_no_raw(self):
        model = trained(observed=12.0, n=12)
        p = model.predict_corrected_forecast(ctx())
        assert "raw_forecast" not in p.values  # no NaN injected
        assert "no_raw_forecast" in p.warnings


class TestConfidence:
    def test_cold_start_low(self):
        c = m().confidence()
        assert c.value <= 0.35 and "cold_start" in c.reasons

    def test_evidence_domains(self):
        c = trained().confidence()
        assert "forecast" in c.evidence_domains and "weather" in c.evidence_domains
        assert c.component == "forecast"

    def test_low_reality_quality_not_high_confidence(self):
        model = m()
        for i in range(15):
            off = i * 24.0
            model.register_decision(decision(f"d{i}", 10.0, created_offset_h=off))
            model.update(evaluation(f"e{i}", f"d{i}", 10.2, created_offset_h=off),
                         ForecastUpdateContext(source_decision_id=f"d{i}", reality_quality=0.35))
        assert model.confidence().value < 0.6

    def test_in_range(self):
        assert 0.0 <= trained().confidence().value <= 1.0
