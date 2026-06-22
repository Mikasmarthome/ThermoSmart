"""Phase 19A: multi-zone isolation through the decision pipeline."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import pipeline, preds, rec


class TestMultiZone:
    def test_independent_zones(self):
        p = pipeline()
        ta, _ = p.run("zoneA", "2025-01-01T07:00:00+00:00", rec(target=21.0, setpoint=23.0),
                      preds("zoneA", boost=1.5), mode=DecisionMode.CONTROL, active_control=True)
        tb, _ = p.run("zoneB", "2025-01-01T07:00:00+00:00", rec(target=20.0, setpoint=20.0),
                      preds("zoneB", boost=None), mode=DecisionMode.CONTROL, active_control=True)
        assert ta.zone_id == "zoneA" and ta.final_setpoint_c == 22.5 and ta.applied_any
        assert tb.zone_id == "zoneB" and tb.final_setpoint_c == 20.0 and not tb.applied_any

    def test_zone_id_propagated(self):
        p = pipeline()
        t, d = p.run("kitchen", "2025-01-01T07:00:00+00:00", rec(), preds("kitchen", boost=1.5),
                     mode=DecisionMode.SHADOW)
        assert t.zone_id == "kitchen" and d.zone_id == "kitchen"

    def test_one_zone_safety_does_not_affect_other(self):
        p = pipeline()
        ta, _ = p.run("a", "t", rec(window_open=True), preds("a", boost=1.5),
                      mode=DecisionMode.CONTROL, active_control=True)
        tb, _ = p.run("b", "t", rec(target=21.0, setpoint=23.0), preds("b", boost=1.5),
                      mode=DecisionMode.CONTROL, active_control=True)
        assert not ta.applied_any and tb.applied_any
