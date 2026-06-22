"""Phase 19A: Final Resolver priority + single-authority behaviour."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import preds, rec, run


class TestFinalResolver:
    def test_shadow_never_applies(self):
        t, d = run(DecisionMode.SHADOW)
        assert not t.applied_any and t.final_setpoint_c == 23.0 and d.shadow_only

    def test_control_applies_reduction(self):
        t, d = run(DecisionMode.CONTROL)
        assert t.applied_any and t.final_setpoint_c == 22.5

    def test_safety_frost_blocks(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(frost_protection=True))
        assert not t.applied_any and "frost_protection" in t.reason_codes

    def test_window_blocks(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(window_open=True))
        assert not t.applied_any and "window_open" in t.reason_codes

    def test_summer_blocks(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(is_summer=True))
        assert not t.applied_any

    def test_absence_blocks(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(absence=True))
        assert not t.applied_any

    def test_no_prediction_falls_back(self):
        t, _ = run(DecisionMode.CONTROL, predictions=preds(boost=None))
        assert not t.applied_any and t.final_setpoint_c == 23.0

    def test_reduce_only(self):
        # le2 proposes a larger boost -> never increases the setpoint
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=22.0),
                   predictions=preds(boost=5.0))
        assert t.final_setpoint_c <= 22.0

    def test_trace_has_reason_and_confidence(self):
        t, _ = run(DecisionMode.CONTROL)
        e = next(e for e in t.entries if e.feature == "boost_offset")
        assert e.reason == "le2_applied" and e.confidence is not None

    def test_every_change_has_baseline_and_clamp(self):
        t, _ = run(DecisionMode.CONTROL)
        e = next(e for e in t.entries if e.applied)
        assert e.baseline_value is not None and e.clamp_applied >= 0.0
