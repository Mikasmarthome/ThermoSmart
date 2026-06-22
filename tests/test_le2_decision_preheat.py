"""Phase 19A: Preheat integration (end-to-end computed, not applied to setpoint in 19A)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import preds, rec, run


class TestPreheatIntegration:
    def test_preheat_in_trace(self):
        t, _ = run(DecisionMode.CONTROL, predictions=preds(boost=1.5, preheat=70.0))
        feats = {e.feature for e in t.entries}
        assert "preheat_start" in feats

    def test_preheat_does_not_change_setpoint(self):
        # 19A: preheat is computed end-to-end but never moves the immediate setpoint
        t_no, _ = run(DecisionMode.CONTROL, predictions=preds(boost=None))
        t_pre, _ = run(DecisionMode.CONTROL, predictions=preds(boost=None, preheat=70.0))
        assert t_no.final_setpoint_c == t_pre.final_setpoint_c == 23.0

    def test_preheat_computed_in_shadow(self):
        t, _ = run(DecisionMode.SHADOW, predictions=preds(boost=None, preheat=70.0))
        e = next(e for e in t.entries if e.feature == "preheat_start")
        assert e.le2_value == 70.0 and not e.applied

    def test_preheat_blocked_by_safety(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(window_open=True),
                   predictions=preds(boost=None, preheat=70.0))
        e = next(e for e in t.entries if e.feature == "preheat_start")
        assert not e.applied

    def test_preheat_records_confidence(self):
        t, _ = run(DecisionMode.CONTROL, predictions=preds(boost=None, preheat=70.0))
        e = next(e for e in t.entries if e.feature == "preheat_start")
        assert e.confidence is not None
