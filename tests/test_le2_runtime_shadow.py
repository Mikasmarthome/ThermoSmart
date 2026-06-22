"""Phase 17 shadow predictions / comparisons / orchestrator tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
    ShadowOrchestrator,
)
from custom_components.thermosmart.learning.runtime.shadow import (
    ComparisonType,
    ShadowStatus,
)
from tests.helpers_runtime import cycle_input


def shadow_runtime():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW))


class TestShadowPredictions:
    def test_predictions_present(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert len(r.shadow_predictions) >= 5

    def test_no_control_effect_on_predictions(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert all(not sp.control_effect for sp in r.shadow_predictions)

    def test_predictions_finite(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        for sp in r.shadow_predictions:
            for v in sp.values.values():
                assert v is None or math.isfinite(v)

    def test_predictions_carry_decision_id(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert all(sp.decision_id == r.decision_id for sp in r.shadow_predictions)


class TestComparisons:
    def test_preheat_comparison(self):
        r = shadow_runtime().run_cycle(
            cycle_input("2026-06-22T05:00:00+00:00",
                        legacy_preheat="2026-06-22T06:00:00+00:00"))
        cmps = [c for c in r.comparisons
                if c.comparison_type == ComparisonType.PREHEAT_START.value]
        assert cmps and cmps[0].difference is not None

    def test_no_baseline_flagged(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00",
                                                   legacy_preheat=None))
        cmps = [c for c in r.comparisons
                if c.comparison_type == ComparisonType.PREHEAT_START.value]
        assert all("no_baseline" in c.reason_codes for c in cmps)

    def test_boost_comparison(self):
        r = shadow_runtime().run_cycle(
            cycle_input("2026-06-22T05:00:00+00:00", legacy_boost=1.5))
        assert any(c.comparison_type == ComparisonType.BOOST_OFFSET.value for c in r.comparisons)

    def test_comparison_open_then_completed(self):
        orch = ShadowOrchestrator()
        snap = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        cmp = snap.comparisons[0]
        assert cmp.status == ShadowStatus.OPEN.value
        completed = orch.complete_comparison(cmp, actual_outcome=20.8, baseline_outcome=20.5)
        assert completed.status == ShadowStatus.EVALUATED.value
        assert completed.actual_outcome == 20.8

    def test_no_assumption_le2_better(self):
        # a difference is just a difference; no field claims superiority
        r = shadow_runtime().run_cycle(
            cycle_input("2026-06-22T05:00:00+00:00",
                        legacy_preheat="2026-06-22T06:00:00+00:00"))
        cmp = r.comparisons[0]
        assert not hasattr(cmp, "le2_better")


class TestShadowOutcome:
    def test_evaluate_reached_on_time(self):
        orch = ShadowOrchestrator()
        o = orch.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=21.0, on_time=True,
                                         overshoot_c=0.0, stability=0.1, data_quality=0.9)
        assert o.reached and o.on_time and o.status == ShadowStatus.EVALUATED.value

    def test_not_evaluable_without_temp(self):
        orch = ShadowOrchestrator()
        o = orch.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=None, on_time=None,
                                         overshoot_c=None, stability=None, data_quality=0.2)
        assert o.status == ShadowStatus.NOT_EVALUABLE.value

    def test_disturbed_not_evaluable(self):
        orch = ShadowOrchestrator()
        o = orch.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=22.0, on_time=True,
                                         overshoot_c=1.0, stability=0.1, data_quality=0.9,
                                         disturbed=True)
        assert o.status == ShadowStatus.NOT_EVALUABLE.value

    def test_metrics_aggregation(self):
        orch = ShadowOrchestrator()
        r = shadow_runtime().run_cycle(
            cycle_input("2026-06-22T05:00:00+00:00",
                        legacy_preheat="2026-06-22T06:00:00+00:00"))
        outcomes = [orch.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=21.0,
                                                 on_time=True, overshoot_c=0.0, stability=0.1,
                                                 data_quality=0.9)]
        m = orch.aggregate_metrics(list(r.comparisons), outcomes)
        assert m.comparison_count >= 1 and m.on_time_rate == 1.0

    def test_no_energy_claim(self):
        orch = ShadowOrchestrator()
        o = orch.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=21.0, on_time=True,
                                         overshoot_c=0.0, stability=0.1, data_quality=0.9)
        assert not hasattr(o, "energy_saved_kwh")


class TestModes:
    def test_inactive_no_work(self):
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.INACTIVE))
        r = rt.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert not r.captured and r.shadow_predictions == ()

    def test_capture_only_no_shadow(self):
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.CAPTURE_ONLY))
        r = rt.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert r.captured and r.shadow_predictions == ()

    def test_shadow_produces_predictions(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert r.captured and r.shadow_predictions
