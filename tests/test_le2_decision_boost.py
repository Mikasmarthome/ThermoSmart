"""Phase 19A: Boost source resolution (single combined offset, reduce-only, limit guard)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import pipeline, preds, rec, run


class TestBoostSourceResolution:
    def test_baseline_boost_is_combined_offset(self):
        # the baseline trv_setpoint already combines TPI + weather + defaults into one offset
        from custom_components.thermosmart.learning.decision.baseline import (
            baseline_from_recommendation)
        base = baseline_from_recommendation("z", rec(target=21.0, setpoint=24.0))
        assert base.boost_offset_c == 3.0  # one authoritative offset, not multiple adders

    def test_le2_can_only_reduce(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=23.0),
                   predictions=preds(boost=5.0))
        assert t.final_setpoint_c <= 23.0

    def test_le2_reduction_applied(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=23.0),
                   predictions=preds(boost=1.5))
        assert t.final_setpoint_c == 22.5

    def test_runtime_limit_mismatch_blocks(self):
        p = pipeline(boost_runtime_limit=99.0)
        t, _ = p.run("z", "2025-01-01T07:00:00+00:00", rec(target=21.0, setpoint=23.0),
                     preds(boost=1.5), mode=DecisionMode.CONTROL, active_control=True)
        assert t.final_setpoint_c == 23.0  # mismatch -> no adjustment

    def test_no_existing_boost_no_change(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=21.0),
                   predictions=preds(boost=1.5))
        assert t.final_setpoint_c == 21.0

    def test_combination_rule_documented_in_trace(self):
        t, _ = run(DecisionMode.CONTROL, predictions=preds(boost=1.5))
        e = next(e for e in t.entries if e.feature == "boost_offset")
        # baseline offset vs le2 proposal both recorded -> auditable combination
        assert e.baseline_value == 2.0 and e.le2_value == 1.5
