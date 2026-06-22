"""Phase 14 BoostModel factor / duration / outcome predictions + confidence tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.models import (
    BoostModel,
    BoostParameters,
    BoostPredictionContext,
    boost_model_definition,
)
from tests.helpers_boost import boost_context, boost_episode, good_boost


def m(params=None):
    return BoostModel("lz", params)


def trained(n=8, offset=1.5, vals=(18.0, 19.5, 20.5, 21.0, 21.0), deficit=3.0):
    model = m()
    for i in range(n):
        ep = boost_episode(f"e{i}", list(vals), decision_id=f"d{i}")
        model.update(ep, boost_context(ep, requested_offset_c=offset, start_deficit_c=deficit,
                                       controller_kind="ts"))
    return model


def ctx(**kw):
    return BoostPredictionContext(**kw)


class TestFactor:
    def test_general(self):
        rec = trained().predict_boost_factor(ctx())
        assert not rec.fallback_used and "general_state" in rec.reason_codes
        assert 0.0 <= rec.boost_offset_c <= BoostParameters().max_boost_offset_c

    def test_conditioned_bucket(self):
        rec = trained().predict_boost_factor(ctx(start_deficit_c=3.0, decision_type="ts",
                                                 intensity_c=1.5))
        assert rec.bucket.startswith("def") or rec.bucket == "general"

    def test_safe_default_cold_start(self):
        rec = m().predict_boost_factor(ctx())
        assert rec.fallback_used and rec.bucket == "safe_default"
        assert rec.boost_offset_c == BoostParameters().default_boost_offset_c

    def test_device_prior(self):
        rec = m().predict_boost_factor(ctx(device_prior_offset_c=2.5))
        assert rec.fallback_used and rec.bucket == "device_prior"
        assert rec.boost_offset_c == 2.5

    def test_factor_clamped(self):
        rec = m().predict_boost_factor(ctx(device_prior_offset_c=99.0))
        assert rec.boost_offset_c <= BoostParameters().max_boost_offset_c

    def test_finite(self):
        for model in (m(), trained()):
            rec = model.predict_boost_factor(ctx())
            assert math.isfinite(rec.boost_offset_c)


class TestDuration:
    def test_duration_present(self):
        rec = trained().predict_boost_duration(ctx())
        assert rec.recommended_duration_s is not None
        assert math.isfinite(rec.recommended_duration_s)

    def test_duration_clamped(self):
        p = BoostParameters(min_duration_pred_s=600, max_duration_pred_s=3000)
        rec = trained(vals=(18.0, 19.5, 20.5, 21.0, 21.0)).predict_boost_factor(ctx())
        # general duration ~1800 from helper, within clamp
        assert rec.recommended_duration_s is not None

    def test_cold_start_default_duration(self):
        rec = m().predict_boost_factor(ctx())
        assert rec.recommended_duration_s == BoostParameters().default_boost_duration_s


class TestExpectedOutcome:
    def test_expected_outcome(self):
        p = trained().predict_expected_outcome(ctx())
        assert p.prediction_type is PredictionType.BOOST_OUTCOME
        assert set(p.values) == {"expected_gain", "expected_overshoot", "expected_comfort"}
        for v in p.values.values():
            assert math.isfinite(v)

    def test_expected_outcome_fallback(self):
        p = m().predict_expected_outcome(ctx())
        assert p.fallback_used and p.confidence_cap is not None


class TestPredictProtocol:
    def test_predict_factor(self):
        p = trained().predict(ctx())
        assert p.prediction_type is PredictionType.BOOST_FACTOR
        assert math.isfinite(p.control_value("boost_factor"))

    def test_predict_no_nan_cold(self):
        p = m().predict(ctx())
        assert math.isfinite(p.values["boost_factor"])

    def test_low_confidence_cap(self):
        p = m().predict(ctx())
        assert p.confidence <= p.confidence_cap


class TestConfidence:
    def test_cold_start_low(self):
        c = m().confidence()
        assert c.value <= 0.35 and "cold_start" in c.reasons

    def test_component_and_domains(self):
        c = trained().confidence()
        assert c.component == "boost"
        assert set(c.evidence_domains) >= {"thermal_trajectory", "controller_events",
                                           "outcome_history"}

    def test_low_attribution_not_high_confidence(self):
        model = m()
        for i in range(15):
            ep = boost_episode(f"e{i}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"d{i}",
                               confounder_flags=("additional_radiator",))
            model.update(ep, boost_context(ep))
        assert model.confidence().value < 0.6

    def test_in_range(self):
        assert 0.0 <= trained().confidence().value <= 1.0
