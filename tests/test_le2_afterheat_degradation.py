"""Phase 10 AfterheatModel TRV-only / numeric / diagnostics / export tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.models import (
    AfterheatModel,
    AfterheatPredictionContext,
    AfterheatUpdateContext,
)
from tests.helpers_afterheat import afterheat_episode


def m():
    return AfterheatModel("lz")


def trained(n=3):
    model = m()
    for i in range(n):
        model.update(afterheat_episode(f"e{i}", 0.5, start_offset_min=i * 30))
    return model


class TestTrvOnly:
    def test_learns_with_inferred_drive_end(self):
        model = m()
        assert model.update(afterheat_episode("e", 0.5)).accepted  # no context, setpoint reduction
        assert model._state.inferred_drive_end_count == 1

    def test_no_buckets_without_context(self):
        assert trained()._state.buckets == {}


class TestNumericGuards:
    def test_peak_at_drive_end_is_near_zero(self):
        # flat trajectory: peak == close -> rise 0 (near-zero), accepted
        model = m()
        r = model.update(afterheat_episode("z", 0.0))
        assert r.accepted

    def test_extreme_rise_rejected(self):
        assert not m().update_eligibility(afterheat_episode("e", 10.0)).accepted

    def test_prediction_finite(self):
        for model in (m(), trained()):
            a = model.predict_afterheat(AfterheatPredictionContext())
            assert math.isfinite(a.residual_rise_c)

    def test_constant_temperature(self):
        # constant temp = near-zero afterheat, learnable
        model = m()
        assert model.update(afterheat_episode("c", 0.0)).accepted


class TestDiagnostics:
    def test_fields(self):
        d = trained().diagnostics()
        assert d.general_rise_c > 0 and d.model_version == 1
        assert d.inferred_drive_end_count >= 1

    def test_near_zero_count_tracked(self):
        model = m()
        model.update(afterheat_episode("z", 0.0))
        assert model.diagnostics().near_zero_count == 1

    def test_rejection_counts(self):
        model = m()
        model.update(afterheat_episode("bad", 10.0))  # implausible
        assert sum(model.diagnostics().rejection_counts.values()) >= 1


class TestExport:
    def test_support_no_raw_or_ids(self):
        exp = trained().export(ExportScope.SUPPORT)
        assert "general_rise_c" in exp and "direct_drive_end_count" in exp
        assert "near_zero_count" in exp
        assert "samples" not in exp and "learning_zone_id" not in exp

    def test_research_anonymized(self):
        exp = trained().export(ExportScope.RESEARCH)
        assert "samples" in exp
        for s in exp["samples"]:
            assert "source_episode_id" not in s and "learning_zone_id" not in s
            assert "residual_rise_c" in s

    def test_raw_not_provided(self):
        assert "unsupported" in trained().export(ExportScope.RAW)


class TestPurity:
    def test_input_not_mutated(self):
        ep = afterheat_episode("e", 0.5)
        before = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        m().update(ep)
        after = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        assert before == after

    def test_deterministic(self):
        assert trained()._state.general.rise.value == trained()._state.general.rise.value
