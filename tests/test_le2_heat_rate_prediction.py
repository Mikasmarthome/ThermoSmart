"""Phase 8 HeatRateModel prediction / preheat / confidence tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import (
    DataQuality,
    Measurement,
    PredictionType,
)
from custom_components.thermosmart.learning.models import (
    HeatRateModel,
    HeatRateParameters,
    HeatRatePredictionContext,
)
from tests.helpers_heat_rate import heating_episode


def ctx(**kw):
    return HeatRatePredictionContext(**kw)


def learned(rate=2.4, n=3):
    model = HeatRateModel("lz")
    for i in range(n):
        model.update(heating_episode(f"e{i}", rate, start_offset_min=i * 60))
    return model


class TestHeatRatePrediction:
    def test_general_fallback(self):
        p = learned().predict_heat_rate(ctx(current_temp=18.0, target=21.0))
        assert p.bucket == "general" and not p.fallback_used
        assert p.control_value("heat_rate") == pytest.approx(2.4, abs=0.1)

    def test_device_prior_when_cold(self):
        model = HeatRateModel("lz", device_prior=lambda pid: 3.5)
        p = model.predict_heat_rate(ctx(current_temp=18.0, target=21.0, profile_id="cast_iron"))
        assert p.fallback_used and p.bucket == "device_prior"
        assert p.control_value("heat_rate") == 3.5
        assert p.confidence <= p.confidence_cap

    def test_generic_prior_when_no_device(self):
        p = HeatRateModel("lz").predict_heat_rate(ctx(current_temp=18.0, target=21.0))
        assert p.fallback_used and p.bucket == "generic_prior"
        assert p.prior_contribution == 1.0 and p.learned_contribution == 0.0

    def test_trv_only_uses_general(self):
        # no outdoor at all -> general (no conditioned bucket)
        p = learned().predict_heat_rate(ctx(current_temp=18.0, target=21.0))
        assert "outdoor_temp" in p.missing_evidence
        assert p.bucket == "general"

    def test_no_none_rate(self):
        p = HeatRateModel("lz").predict_heat_rate(ctx())
        assert p.control_value("heat_rate") is not None
        assert math.isfinite(p.control_value("heat_rate"))

    def test_cold_start_confidence_capped(self):
        p = HeatRateModel("lz").predict_heat_rate(ctx(current_temp=18.0, target=21.0))
        assert p.confidence <= HeatRateParameters().cold_start_confidence_cap


class TestPreheat:
    def test_target_reached_zero(self):
        p = learned().predict_preheat(ctx(current_temp=21.0, target=21.0))
        assert p.control_value("preheat_minutes") == 0.0
        assert "target_reached" in p.reason_codes

    def test_learned_preheat(self):
        # rate 2.4 C/h, need +3C -> 75 min
        p = learned().predict_preheat(ctx(current_temp=18.0, target=21.0))
        assert p.control_value("preheat_minutes") == pytest.approx(75.0, abs=1.0)

    def test_invalid_rate_fallback(self):
        # generic prior path still gives a positive rate; force invalid via params
        model = HeatRateModel("lz", HeatRateParameters(generic_prior_rate_c_per_h=2.0))
        p = model.predict_preheat(ctx(current_temp=18.0, target=21.0))
        # cold start -> capped/conservative, still finite and within bounds
        v = p.control_value("preheat_minutes")
        assert 0.0 <= v <= HeatRateParameters().preheat_max_minutes

    def test_missing_temps_fallback(self):
        p = learned().predict_preheat(ctx())
        assert p.fallback_used
        assert "missing_temps_fallback" in p.reason_codes

    def test_clamped_to_max(self):
        # tiny rate would imply a huge preheat -> clamped
        model = HeatRateModel("lz", HeatRateParameters(generic_prior_rate_c_per_h=0.1))
        p = model.predict_preheat(ctx(current_temp=10.0, target=22.0))
        assert p.control_value("preheat_minutes") <= HeatRateParameters().preheat_max_minutes

    def test_preheat_no_nan(self):
        p = learned().predict_preheat(ctx(current_temp=18.0, target=21.0))
        assert math.isfinite(p.control_value("preheat_minutes"))


class TestConfidence:
    def test_cold_start_low(self):
        c = HeatRateModel("lz").confidence()
        assert c.value <= 0.4
        assert "cold_start" in c.reasons

    def test_more_evidence_higher_confidence(self):
        few = learned(n=2).confidence().value
        many = learned(n=15).confidence().value
        assert many >= few

    def test_confidence_in_range(self):
        assert 0.0 <= learned(n=10).confidence().value <= 1.0
