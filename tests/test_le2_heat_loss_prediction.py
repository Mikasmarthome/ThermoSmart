"""Phase 9 HeatLossModel prediction / cooling-duration / confidence tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.models import (
    HeatLossModel,
    HeatLossParameters,
    HeatLossPredictionContext,
    HeatLossUpdateContext,
)
from tests.helpers_heat_loss import cooling_episode


def ctx(**kw):
    return HeatLossPredictionContext(**kw)


def learned(loss=1.0, n=3, outdoor=None):
    model = HeatLossModel("lz")
    for i in range(n):
        ep = cooling_episode(f"e{i}", loss, start_offset_min=i * 60)
        c = (HeatLossUpdateContext(source_episode_id=ep.episode_id, outdoor_temperature_c=outdoor)
             if outdoor is not None else None)
        model.update(ep, c)
    return model


class TestHeatLossPrediction:
    def test_general_fallback(self):
        p = learned().predict_heat_loss(ctx(current_temp=20.0))
        assert p.bucket == "general" and not p.fallback_used
        assert p.control_value("heat_loss_rate") == pytest.approx(1.0, abs=0.1)

    def test_relative_only_flag(self):
        p = learned().predict_heat_loss(ctx(current_temp=20.0))  # no outdoor
        assert "relative_only" in p.reason_codes
        assert "outdoor_temp" in p.missing_evidence

    def test_conditioned_bucket(self):
        model = learned(n=6, outdoor=2.0)
        p = model.predict_heat_loss(ctx(current_temp=20.0,
                                        outdoor_temp=Measurement(2.0, DataQuality.OK)))
        assert p.bucket.startswith("outdoor:")
        assert "conditioned_bucket" in p.reason_codes

    def test_device_prior(self):
        model = HeatLossModel("lz", device_prior=lambda pid: 0.6)
        p = model.predict_heat_loss(ctx(current_temp=20.0, profile_id="brick"))
        assert p.fallback_used and p.bucket == "device_prior"
        assert p.control_value("heat_loss_rate") == 0.6

    def test_generic_prior(self):
        p = HeatLossModel("lz").predict_heat_loss(ctx(current_temp=20.0))
        assert p.fallback_used and p.bucket == "generic_prior"
        assert p.prior_contribution == 1.0

    def test_no_none_or_nan(self):
        p = HeatLossModel("lz").predict_heat_loss(ctx())
        assert math.isfinite(p.control_value("heat_loss_rate"))

    def test_cold_start_confidence_capped(self):
        p = HeatLossModel("lz").predict_heat_loss(ctx(current_temp=20.0))
        assert p.confidence <= HeatLossParameters().cold_start_confidence_cap


class TestCoolingDuration:
    def test_floor_reached_zero(self):
        p = learned().predict_cooling_duration(ctx(current_temp=18.0, floor_temp=18.0))
        assert p.control_value("cooling_minutes") == 0.0
        assert "floor_reached" in p.reason_codes

    def test_learned_duration(self):
        # loss 1.0 C/h, drop 3C -> 180 min
        p = learned().predict_cooling_duration(ctx(current_temp=21.0, floor_temp=18.0))
        assert p.control_value("cooling_minutes") == pytest.approx(180.0, abs=2.0)

    def test_missing_temps_fallback(self):
        p = learned().predict_cooling_duration(ctx())
        assert p.fallback_used and "missing_temps_fallback" in p.reason_codes

    def test_clamped_to_max(self):
        model = HeatLossModel("lz", HeatLossParameters(generic_prior_loss_c_per_h=0.01))
        p = model.predict_cooling_duration(ctx(current_temp=22.0, floor_temp=10.0))
        assert p.control_value("cooling_minutes") <= HeatLossParameters().cooling_max_minutes

    def test_no_nan(self):
        p = learned().predict_cooling_duration(ctx(current_temp=21.0, floor_temp=18.0))
        assert math.isfinite(p.control_value("cooling_minutes"))


class TestConfidence:
    def test_cold_start_low(self):
        c = HeatLossModel("lz").confidence()
        assert c.value <= 0.4 and "cold_start" in c.reasons

    def test_relative_only_reason(self):
        c = learned().confidence()  # no outdoor buckets
        assert "relative_only" in c.reasons

    def test_in_range(self):
        assert 0.0 <= learned(n=10).confidence().value <= 1.0
