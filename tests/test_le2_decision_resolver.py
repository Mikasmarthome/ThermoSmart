"""Phase 19A: Final Resolver priority + single-authority behaviour."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import preds, rec, run


class TestFinalResolver:
    def test_shadow_never_applies(self):
        t, d = run(DecisionMode.SHADOW)
        assert not t.applied_any and t.final_setpoint_c == 23.0 and d.shadow_only

    def test_control_applies_boost_on_tpi_baseline(self):
        # TPI baseline=23°C (from setpoint), no current le2 boost (boost_offset_c=0).
        # LE2 predicts 1.5°C boost; first step from 0→0.5 → final = 23 + 0.5 = 23.5°C.
        t, d = run(DecisionMode.CONTROL)
        assert t.applied_any and t.final_setpoint_c == pytest.approx(23.5)
        # Baseline setpoint in trace must be the TPI baseline, not target
        assert t.tpi_baseline_setpoint_c == pytest.approx(23.0)

    def test_control_applies_boost_reduction(self):
        # Current le2 boost = 2.0°C; LE2 predicts 1.5°C → reduces its own boost.
        # final = tpi_baseline(21) + 1.5 = 22.5°C (step from 2.0 → 1.5, within one step).
        r = rec(target=19.0, setpoint=21.0, boost_offset_c=2.0)
        t, d = run(DecisionMode.CONTROL, recommendation=r, predictions=preds(boost=1.5))
        assert t.applied_any and t.final_setpoint_c == pytest.approx(22.5)

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

    def test_full_authority_increase_step_limited(self):
        # LE 2.0 full authority: boost is additive on TPI baseline, step-limited to 0.5/cycle.
        # TPI baseline=22°C, no current boost (offset_c=0), LE2 requests 3.0°C.
        # First step: 0.0→0.5 → final = 22.0 + 0.5 = 22.5°C
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=22.0),
                   predictions=preds(boost=3.0))
        assert t.final_setpoint_c == pytest.approx(22.5)
        assert t.tpi_baseline_setpoint_c == pytest.approx(22.0)

    def test_trace_has_reason_and_confidence(self):
        t, _ = run(DecisionMode.CONTROL)
        e = next(e for e in t.entries if e.feature == "boost_offset")
        assert e.reason == "le2_applied" and e.confidence is not None

    def test_every_change_has_baseline_and_clamp(self):
        t, _ = run(DecisionMode.CONTROL)
        e = next(e for e in t.entries if e.applied)
        assert e.baseline_value is not None and e.clamp_applied >= 0.0
