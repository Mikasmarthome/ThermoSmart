"""Phase 19A: Early Cutoff integration (computed in pipeline, no direct dispatch)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import preds, rec, run


class TestEarlyCutoffIntegration:
    def test_cutoff_in_trace(self):
        t, _ = run(DecisionMode.CONTROL, predictions=preds(boost=None, cutoff=8.0))
        assert any(e.feature == "early_cutoff" for e in t.entries)

    def test_cutoff_does_not_dispatch_setpoint(self):
        sent = []
        t, d = run(DecisionMode.CONTROL, predictions=preds(boost=None, cutoff=8.0),
                   sink=sent.append)
        assert sent == [] and t.final_setpoint_c == 23.0

    def test_cutoff_blocked_when_window_open(self):
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(window_open=True),
                   predictions=preds(boost=None, cutoff=8.0))
        e = next(e for e in t.entries if e.feature == "early_cutoff")
        assert not e.applied

    def test_cutoff_computed_in_shadow(self):
        t, _ = run(DecisionMode.SHADOW, predictions=preds(boost=None, cutoff=8.0))
        e = next(e for e in t.entries if e.feature == "early_cutoff")
        assert e.le2_value == 8.0 and not e.applied
