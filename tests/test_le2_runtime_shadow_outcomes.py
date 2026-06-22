"""Phase 17 shadow outcome evaluation + aggregated metrics + comparison completion."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import ShadowOrchestrator
from custom_components.thermosmart.learning.runtime.shadow import (
    ComparisonType,
    ShadowStatus,
)


def orch():
    return ShadowOrchestrator()


def _comparison(o, baseline=360.0, shadow=350.0):
    class _Snap:
        zone_id = "lr"
        decision_id = "d1"
        ts = "2026-06-22T05:00:00+00:00"
    return o.compare_scalar(_Snap(), ComparisonType.PREHEAT_START, baseline, shadow, "minutes",
                            0.8)


class TestShadowOutcomes:
    def test_reached_on_time(self):
        o = orch()
        r = o.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=21.0, on_time=True,
                                      overshoot_c=0.0, stability=0.1, data_quality=0.9)
        assert r.reached and r.on_time and r.status == ShadowStatus.EVALUATED.value

    def test_not_reached(self):
        o = orch()
        r = o.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=19.5, on_time=False,
                                      overshoot_c=0.0, stability=0.1, data_quality=0.9)
        assert not r.reached and r.late

    def test_overshoot(self):
        o = orch()
        r = o.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=22.5, on_time=True,
                                      overshoot_c=None, stability=0.2, data_quality=0.9)
        assert r.overshoot_c == pytest.approx(1.5)

    def test_no_temp_not_evaluable(self):
        o = orch()
        r = o.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=None, on_time=None,
                                      overshoot_c=None, stability=None, data_quality=0.2)
        assert r.status == ShadowStatus.NOT_EVALUABLE.value

    def test_disturbed_not_evaluable(self):
        o = orch()
        r = o.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=22.0, on_time=True,
                                      overshoot_c=1.0, stability=0.1, data_quality=0.9,
                                      disturbed=True)
        assert r.status == ShadowStatus.NOT_EVALUABLE.value

    def test_complete_comparison(self):
        o = orch()
        cmp = _comparison(o)
        assert cmp.status == ShadowStatus.OPEN.value
        done = o.complete_comparison(cmp, actual_outcome=20.8, baseline_outcome=20.4)
        assert done.status == ShadowStatus.EVALUATED.value and done.actual_outcome == 20.8

    def test_metrics(self):
        o = orch()
        comparisons = [o.complete_comparison(_comparison(o, 360.0, 350.0), actual_outcome=20.9)]
        outcomes = [
            o.evaluate_shadow_outcome("d1", target_c=21.0, achieved_c=21.0, on_time=True,
                                      overshoot_c=0.0, stability=0.1, data_quality=0.9),
            o.evaluate_shadow_outcome("d2", target_c=21.0, achieved_c=None, on_time=None,
                                      overshoot_c=None, stability=None, data_quality=0.2),
        ]
        m = o.aggregate_metrics(comparisons, outcomes)
        assert m.comparison_count == 1 and m.evaluated_count == 1
        assert m.mean_preheat_start_diff_min == pytest.approx(-10.0)
        assert m.on_time_rate == 1.0 and m.not_evaluable_rate == 0.5

    def test_metrics_empty(self):
        m = orch().aggregate_metrics([], [])
        assert m.comparison_count == 0 and m.on_time_rate is None

    def test_no_energy_metric(self):
        m = orch().aggregate_metrics([], [])
        assert not hasattr(m, "energy_saved")
