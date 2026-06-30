"""Phase 19A: Shadow compatibility — SHADOW pipeline never changes the dispatched value."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import preds, rec, run


class TestShadowCompatibility:
    @pytest.mark.parametrize("setpoint,target,boost", [
        (23.0, 21.0, 1.5), (22.0, 21.0, 0.5), (24.0, 21.0, 2.0), (21.0, 21.0, 0.0)])
    def test_shadow_final_equals_baseline(self, setpoint, target, boost):
        t, d = run(DecisionMode.SHADOW, recommendation=rec(target=target, setpoint=setpoint),
                   predictions=preds(boost=boost))
        assert t.final_setpoint_c == setpoint and not t.applied_any and d.shadow_only

    def test_shadow_computes_full_trace(self):
        # full LE2 decision is still computed (baseline, le2 proposal, resolver, fallback)
        # boost_offset_c=2.0 to set a non-zero existing le2 boost (no reconstruction from setpoint-target)
        t, _ = run(DecisionMode.SHADOW,
                   recommendation=rec(target=21.0, setpoint=23.0, boost_offset_c=2.0),
                   predictions=preds(boost=1.5, preheat=70.0, cutoff=8.0))
        e = next(e for e in t.entries if e.feature == "boost_offset")
        assert e.le2_value == pytest.approx(1.5) and e.baseline_value == pytest.approx(2.0)
        assert e.fallback_reason == "shadow_mode"
        assert {"boost_offset", "preheat_start", "early_cutoff"} <= {x.feature for x in t.entries}

    def test_shadow_never_dispatches(self):
        sent = []
        run(DecisionMode.SHADOW, predictions=preds(boost=1.5), sink=sent.append)
        assert sent == []

    def test_trace_shows_hypothetical_value(self):
        # shadow trace records what WOULD have been applied without applying it
        t, _ = run(DecisionMode.SHADOW, predictions=preds(boost=1.5))
        e = next(e for e in t.entries if e.feature == "boost_offset")
        assert e.le2_value == 1.5 and t.final_setpoint_c == 23.0
