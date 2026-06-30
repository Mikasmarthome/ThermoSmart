"""Phase 8 HeatRateModel TRV-only, numeric guards, diagnostics/export tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, ExportScope, Measurement
from custom_components.thermosmart.learning.models import (
    HeatRateModel,
    HeatRatePredictionContext,
)
from tests.helpers_heat_rate import heating_episode


def m():
    return HeatRateModel("lz")


def trained(n=3):
    model = m()
    for i in range(n):
        model.update(heating_episode(f"e{i}", 2.4, start_offset_min=i * 60))
    return model


class TestTrvOnly:
    def test_learns_without_any_optional_data(self):
        model = m()
        r = model.update(heating_episode("e", 2.4))
        assert r.accepted
        p = model.predict_heat_rate(HeatRatePredictionContext(current_temp=18.0, target=21.0))
        assert not p.fallback_used and p.bucket == "general"

    def test_missing_outdoor_does_not_block(self):
        model = trained()
        # no outdoor: conditioned buckets never created, general still predicts
        assert model._state.buckets == {}
        p = model.predict_heat_rate(HeatRatePredictionContext(current_temp=18.0, target=21.0))
        assert "outdoor_temp" in p.missing_evidence


class TestNumericGuards:
    def test_constant_temperature_not_learned(self):
        ep = heating_episode("flat", 0.0)  # delta ~ 0
        r = m().update_eligibility(ep)
        assert not r.accepted

    def test_extreme_rate_rejected(self):
        ep = heating_episode("hot", 30.0, duration_min=5.0, points=3)
        r = m().update_eligibility(ep)
        assert not r.accepted

    def test_prediction_never_nan(self):
        for model in (m(), trained()):
            p = model.predict_heat_rate(HeatRatePredictionContext(current_temp=0.0, target=0.0))
            assert math.isfinite(p.control_value("heat_rate"))

    def test_zero_delta_target(self):
        p = trained().predict_preheat(HeatRatePredictionContext(current_temp=21.0, target=21.0))
        assert p.control_value("preheat_minutes") == 0.0


class TestDiagnostics:
    def test_diagnostics_fields(self):
        d = trained().diagnostics()
        assert d.general_rate_c_per_h > 0
        assert d.model_version == 1
        assert "general" in d.sample_counts

    def test_rejection_counts_tracked(self):
        model = m()
        model.update(heating_episode("bad", -2.0))  # rejected
        d = model.diagnostics()
        assert sum(d.rejection_counts.values()) >= 1

    def test_no_raw_trajectory_in_diagnostics(self):
        d = trained().diagnostics()
        # diagnostics is a typed object without trajectory points
        assert not hasattr(d, "trajectory")


class TestExport:
    def test_support_export_no_raw_or_ids(self):
        exp = trained().export(ExportScope.SUPPORT)
        assert "general_rate_c_per_h" in exp
        assert "samples" not in exp  # no raw samples/trajectories
        assert "learning_zone_id" not in exp

    def test_research_export_anonymized(self):
        exp = trained().export(ExportScope.RESEARCH)
        assert "samples" in exp
        for s in exp["samples"]:
            assert "source_episode_id" not in s
            assert "learning_zone_id" not in s
            assert "rate_c_per_h" in s

    def test_raw_export_not_provided_by_model(self):
        exp = trained().export(ExportScope.RAW)
        assert "unsupported" in exp


class TestPurityBasics:
    def test_input_not_mutated(self):
        ep = heating_episode("e", 2.4)
        before = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        m().update(ep)
        after = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        assert before == after

    def test_deterministic(self):
        a = trained(); b = trained()
        assert a._state.general.rate_c_per_h == b._state.general.rate_c_per_h
