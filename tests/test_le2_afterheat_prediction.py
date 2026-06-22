"""Phase 10 AfterheatModel prediction / early-cutoff / confidence tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.models import (
    AfterheatModel,
    AfterheatParameters,
    AfterheatPredictionContext,
    AfterheatUpdateContext,
)
from tests.helpers_afterheat import afterheat_episode


def ctx(**kw):
    return AfterheatPredictionContext(**kw)


def learned(rise=0.5, n=3, heating_duration_s=None):
    model = AfterheatModel("lz")
    for i in range(n):
        ep = afterheat_episode(f"e{i}", rise, start_offset_min=i * 30)
        c = (AfterheatUpdateContext(source_episode_id=ep.episode_id,
                                    heating_duration_s=heating_duration_s)
             if heating_duration_s is not None else None)
        model.update(ep, c)
    return model


class TestAfterheatPrediction:
    def test_general(self):
        a = learned().predict_afterheat(ctx())
        assert a.bucket == "general" and not a.fallback_used
        assert "general_state" in a.reason_codes
        assert a.residual_rise_c == pytest.approx(0.5, abs=0.1)

    def test_conditioned_bucket(self):
        model = learned(n=6, heating_duration_s=2400.0)
        a = model.predict_afterheat(ctx(heating_duration_s=2400.0))
        assert a.bucket.startswith("heating_dur:")
        assert "conditioned_bucket" in a.reason_codes

    def test_device_prior(self):
        model = AfterheatModel("lz", device_prior=lambda pid: 0.8)
        a = model.predict_afterheat(ctx(profile_id="cast_iron"))
        assert a.fallback_used and a.bucket == "device_prior"
        assert a.residual_rise_c == 0.8

    def test_generic_prior(self):
        a = AfterheatModel("lz").predict_afterheat(ctx())
        assert a.fallback_used and a.bucket == "generic_prior"
        assert a.prior_contribution == 1.0

    def test_all_values_finite(self):
        a = AfterheatModel("lz").predict_afterheat(ctx())
        for v in (a.residual_rise_c, a.time_to_peak_min, a.afterheat_duration_min, a.decay_indicator):
            assert math.isfinite(v)

    def test_predict_protocol_returns_prediction(self):
        p = learned().predict(ctx())
        assert math.isfinite(p.control_value("expected_overshoot"))


class TestEarlyCutoff:
    def test_learned_cutoff(self):
        ec = learned().predict_early_cutoff(ctx(current_temp=20.0, target=21.0))
        assert ec.control_value("expected_residual_rise") == pytest.approx(0.5, abs=0.1)

    def test_near_zero_cutoff_zero(self):
        nz = AfterheatModel("lz")
        for i in range(5):
            nz.update(afterheat_episode(f"z{i}", 0.0, start_offset_min=i * 30))
        ec = nz.predict_early_cutoff(ctx(current_temp=20.0, target=21.0))
        assert ec.control_value("expected_residual_rise") == 0.0
        assert "near_zero" in ec.reason_codes

    def test_clamped_to_remaining_delta(self):
        model = AfterheatModel("lz", device_prior=lambda pid: 2.0)
        ec = model.predict_early_cutoff(ctx(current_temp=20.7, target=21.0, profile_id="x"))
        # residual 2.0 capped to remaining 0.3
        assert ec.control_value("expected_residual_rise") == pytest.approx(0.3, abs=1e-6)
        assert "clamped_to_remaining_delta" in ec.reason_codes

    def test_max_cap(self):
        model = AfterheatModel("lz", device_prior=lambda pid: 99.0)
        ec = model.predict_early_cutoff(ctx(profile_id="x"))
        assert ec.control_value("expected_residual_rise") <= AfterheatParameters().early_cutoff_max_c

    def test_no_nan(self):
        ec = learned().predict_early_cutoff(ctx(current_temp=20.0, target=21.0))
        assert math.isfinite(ec.control_value("expected_residual_rise"))


class TestConfidence:
    def test_cold_start_low(self):
        c = AfterheatModel("lz").confidence()
        assert c.value <= 0.4 and "cold_start" in c.reasons

    def test_many_near_zero_high_confidence_possible(self):
        nz = AfterheatModel("lz")
        for i in range(15):
            nz.update(afterheat_episode(f"z{i}", 0.0, start_offset_min=i * 30))
        c = nz.confidence()
        assert c.value > 0.4  # confident the zone has little afterheat
        assert "near_zero_zone" in c.reasons

    def test_in_range(self):
        assert 0.0 <= learned(n=10).confidence().value <= 1.0
