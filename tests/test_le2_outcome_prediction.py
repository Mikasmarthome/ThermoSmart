"""Phase 11 OutcomeModel prediction / advisory / confidence tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.models import (
    OutcomeModel,
    OutcomePredictionContext,
    OutcomeUpdateContext,
)
from tests.helpers_outcome import no_temp, reached_clean


def learned(n=6, controller_kind=None):
    model = OutcomeModel("lz")
    for i in range(n):
        ep = reached_clean(f"e{i}")
        ctx = (OutcomeUpdateContext(source_episode_id=ep.episode_id,
                                    controller_kind=controller_kind)
               if controller_kind else None)
        model.update(ep, ctx)
    return model


def ctx(**kw):
    return OutcomePredictionContext(**kw)


class TestPrediction:
    def test_general(self):
        p = learned().predict(ctx())
        assert p.prediction_type is PredictionType.OUTCOME_QUALITY
        assert not p.fallback_used and "general_state" in p.reason_codes
        assert 0.0 <= p.values["outcome_quality"] <= 1.0

    def test_conditioned_bucket(self):
        model = learned(n=6, controller_kind="ts")
        p = model.predict(ctx(controller_kind="ts"))
        assert p.bucket.startswith("controller:")
        assert "conditioned_bucket" in p.reason_codes

    def test_prior_fallback(self):
        p = OutcomeModel("lz").predict(ctx())
        assert p.fallback_used and p.prior_contribution == 1.0
        assert p.confidence_cap is not None

    def test_controller_quality_signal(self):
        p = learned().predict_controller_quality(ctx())
        assert p.prediction_type is PredictionType.CONTROLLER_QUALITY
        assert math.isfinite(p.values["controller_quality"])

    def test_no_control_value(self):
        # advisory: prediction carries quality scores, not a setpoint/duration
        p = learned().predict(ctx())
        assert set(p.values.keys()) == {"outcome_quality"}

    def test_all_finite(self):
        for model in (OutcomeModel("lz"), learned()):
            p = model.predict(ctx())
            assert math.isfinite(p.values["outcome_quality"])
            assert math.isfinite(p.confidence)

    def test_low_confidence_cap_on_fallback(self):
        p = OutcomeModel("lz").predict(ctx())
        assert p.confidence <= p.confidence_cap


class TestConfidence:
    def test_cold_start_low(self):
        c = OutcomeModel("lz").confidence()
        assert c.value <= 0.4 and "cold_start" in c.reasons

    def test_many_low_quality_not_high_confidence(self):
        model = OutcomeModel("lz")
        for i in range(20):
            model.update(reached_clean(f"e{i}", reliability=0.3))
        assert model.confidence().value < 0.6
        assert "low_data_quality" in model.confidence().reasons

    def test_mostly_partial_flagged(self):
        model = OutcomeModel("lz")
        from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
        for i in range(6):
            model.update(no_temp(f"p{i}", reason=EpisodeReason.TIMEOUT))
        assert "mostly_partial" in model.confidence().reasons

    def test_in_range(self):
        assert 0.0 <= learned(n=10).confidence().value <= 1.0
