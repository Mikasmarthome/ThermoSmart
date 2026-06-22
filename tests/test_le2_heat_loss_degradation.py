"""Phase 9 HeatLossModel TRV-only / relative-only / numeric / diagnostics / export."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, ExportScope, Measurement
from custom_components.thermosmart.learning.models import (
    HeatLossModel,
    HeatLossPredictionContext,
    HeatLossUpdateContext,
)
from tests.helpers_heat_loss import cooling_episode


def m():
    return HeatLossModel("lz")


def trained(n=3):
    model = m()
    for i in range(n):
        model.update(cooling_episode(f"e{i}", 1.0, start_offset_min=i * 60))
    return model


class TestTrvOnlyRelativeOnly:
    def test_learns_without_optional_data(self):
        model = m()
        assert model.update(cooling_episode("e", 1.0)).accepted
        p = model.predict_heat_loss(HeatLossPredictionContext(current_temp=20.0))
        assert not p.fallback_used and "relative_only" in p.reason_codes

    def test_no_normalized_without_outdoor(self):
        model = trained()
        assert not model._state.normalized.has_evidence
        d = model.diagnostics()
        assert d.normalized_coefficient_per_h is None
        assert d.relative_only is True


class TestNumericGuards:
    def test_no_cooling_not_learned(self):
        assert not m().update_eligibility(cooling_episode("e", 0.0)).accepted

    def test_extreme_loss_rejected(self):
        assert not m().update_eligibility(cooling_episode("e", 30.0)).accepted

    def test_indoor_equals_outdoor_no_normalized(self):
        model = m()
        ep = cooling_episode("e", 1.0)
        model.update(ep, HeatLossUpdateContext(source_episode_id="e", outdoor_temperature_c=20.0,
                                               indoor_outdoor_delta_c=0.0))
        assert not model._state.normalized.has_evidence

    def test_negative_io_delta_no_normalized(self):
        model = m()
        ep = cooling_episode("e", 1.0)
        model.update(ep, HeatLossUpdateContext(source_episode_id="e", outdoor_temperature_c=25.0,
                                               indoor_outdoor_delta_c=-4.0))
        assert not model._state.normalized.has_evidence

    def test_prediction_finite(self):
        for model in (m(), trained()):
            p = model.predict_heat_loss(HeatLossPredictionContext(current_temp=0.0))
            assert math.isfinite(p.control_value("heat_loss_rate"))


class TestDiagnostics:
    def test_fields(self):
        d = trained().diagnostics()
        assert d.general_loss_c_per_h > 0 and d.model_version == 1

    def test_rejection_counts(self):
        model = m()
        model.update(cooling_episode("bad", 0.0))
        assert sum(model.diagnostics().rejection_counts.values()) >= 1


class TestExport:
    def test_support_no_raw_or_ids(self):
        exp = trained().export(ExportScope.SUPPORT)
        assert "general_loss_c_per_h" in exp and "relative_only" in exp
        assert "samples" not in exp and "learning_zone_id" not in exp

    def test_research_anonymized(self):
        exp = trained().export(ExportScope.RESEARCH)
        assert "samples" in exp
        for s in exp["samples"]:
            assert "source_episode_id" not in s and "learning_zone_id" not in s
            assert "loss_rate_c_per_h" in s

    def test_raw_not_provided(self):
        assert "unsupported" in trained().export(ExportScope.RAW)


class TestPurity:
    def test_input_not_mutated(self):
        ep = cooling_episode("e", 1.0)
        before = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        m().update(ep)
        after = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        assert before == after

    def test_deterministic(self):
        assert trained()._state.general.rate_c_per_h == trained()._state.general.rate_c_per_h
