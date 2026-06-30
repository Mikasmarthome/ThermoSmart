"""Phase 19B: Boost source resolution — LE 2.0 full authority (not reduce-only)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import pipeline, preds, rec, run


class TestBoostSourceResolution:
    def test_baseline_boost_from_rec_key(self):
        # boost_offset_c is read directly from the rec dict (coordinator sets this).
        # No reconstruction: setpoint-target is NOT the le2 boost.
        from custom_components.thermosmart.learning.decision.baseline import (
            baseline_from_recommendation)
        # With explicit boost_offset_c=3.0: used directly
        base = baseline_from_recommendation("z", rec(target=21.0, setpoint=24.0,
                                                     boost_offset_c=3.0))
        assert base.boost_offset_c == pytest.approx(3.0)
        # Without boost_offset_c key: defaults to 0.0 (no current le2 boost)
        base_no_boost = baseline_from_recommendation("z", rec(target=21.0, setpoint=24.0))
        assert base_no_boost.boost_offset_c == pytest.approx(0.0)
        # Baseline setpoint is always the TPI baseline (not target), regardless of le2 boost
        assert base.trv_setpoint_c == pytest.approx(24.0)

    def test_le2_full_authority_can_increase(self):
        # LE2 proposes more than existing boost: step-limited roll-in (+0.5/cycle max).
        # existing le2 boost=2.0, TPI baseline=23°C, LE2 requests 3.0°C.
        # raw_dev=1.0, step=0.5 → final_offset=2.5, sp=23+2.5=25.5°C
        t, _ = run(DecisionMode.CONTROL,
                   recommendation=rec(target=21.0, setpoint=23.0, boost_offset_c=2.0),
                   predictions=preds(boost=3.0))
        assert t.final_setpoint_c == pytest.approx(25.5)

    def test_le2_full_authority_can_decrease(self):
        # LE2 proposes less than existing boost: step-limited decrease (reduction).
        # existing le2 boost=2.0, TPI baseline=23°C, LE2 requests 1.5°C.
        # raw_dev=-0.5, within step → final_offset=1.5, sp=23+1.5=24.5°C
        t, _ = run(DecisionMode.CONTROL,
                   recommendation=rec(target=21.0, setpoint=23.0, boost_offset_c=2.0),
                   predictions=preds(boost=1.5))
        assert t.final_setpoint_c == pytest.approx(24.5)

    def test_le2_can_propose_boost_from_zero(self):
        # LE2 adds boost when no existing le2 offset (boost_offset_c=0 by default).
        # TPI baseline=21°C, LE2 requests 1.5°C.
        # raw_dev=1.5, step=0.5 → final_offset=0.5, sp=21+0.5=21.5°C
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=21.0),
                   predictions=preds(boost=1.5))
        assert t.final_setpoint_c == pytest.approx(21.5)

    def test_runtime_limit_mismatch_blocks(self):
        p = pipeline(boost_runtime_limit=99.0)
        t, _ = p.run("z", "2025-01-01T07:00:00+00:00", rec(target=21.0, setpoint=23.0),
                     preds(boost=1.5), mode=DecisionMode.CONTROL, active_control=True)
        assert t.final_setpoint_c == 23.0  # mismatch -> no adjustment

    def test_shadow_mode_never_changes_setpoint(self):
        t, _ = run(DecisionMode.SHADOW, predictions=preds(boost=1.5))
        assert t.final_setpoint_c == pytest.approx(23.0)

    def test_combination_rule_documented_in_trace(self):
        # With explicit boost_offset_c=2.0: shows in trace baseline_value=2.0
        t, _ = run(DecisionMode.CONTROL,
                   recommendation=rec(target=21.0, setpoint=23.0, boost_offset_c=2.0),
                   predictions=preds(boost=1.5))
        e = next(e for e in t.entries if e.feature == "boost_offset")
        assert e.baseline_value == pytest.approx(2.0) and e.le2_value == pytest.approx(1.5)

    def test_early_cutoff_hold_blocks_boost(self):
        from custom_components.thermosmart.learning.decision.resolver import FinalResolver
        from custom_components.thermosmart.learning.decision.contracts import (
            ControllerBaselineDecision, ZoneRuntimeInput)
        r = FinalResolver(boost_runtime_limit=8.0)
        zin = ZoneRuntimeInput(zone_id="z", ts="2025-01-01T07:00:00+00:00",
                               early_cutoff_state="coasting_hold")
        base = ControllerBaselineDecision(zone_id="z", target_c=21.0, trv_setpoint_c=21.0,
                                         boost_offset_c=0.0)
        trace, _ = r.resolve(zin, base, preds(boost=2.0), mode=DecisionMode.CONTROL)
        boost_entry = next(e for e in trace.entries if e.feature == "boost_offset")
        assert not boost_entry.applied
        assert boost_entry.reason == "early_cutoff_hold"

    def test_small_deficit_blocks_boost(self):
        from custom_components.thermosmart.learning.decision.resolver import FinalResolver
        from custom_components.thermosmart.learning.decision.contracts import (
            ControllerBaselineDecision, ZoneRuntimeInput)
        r = FinalResolver(boost_runtime_limit=8.0)
        # temp_gap = 0.3°C → very small, below default threshold of 0.5
        zin = ZoneRuntimeInput(zone_id="z", ts="2025-01-01T07:00:00+00:00",
                               temperature_gap_c=0.3)
        base = ControllerBaselineDecision(zone_id="z", target_c=21.0, trv_setpoint_c=21.0,
                                         boost_offset_c=0.0)
        trace, _ = r.resolve(zin, base, preds(boost=2.0), mode=DecisionMode.CONTROL)
        boost_entry = next(e for e in trace.entries if e.feature == "boost_offset")
        assert not boost_entry.applied
        assert boost_entry.reason == "deficit_too_small"

    def test_explicit_ineligible_blocks_boost(self):
        from custom_components.thermosmart.learning.decision.resolver import FinalResolver
        from custom_components.thermosmart.learning.decision.contracts import (
            ControllerBaselineDecision, ZoneRuntimeInput)
        r = FinalResolver(boost_runtime_limit=8.0)
        zin = ZoneRuntimeInput(zone_id="z", ts="2025-01-01T07:00:00+00:00",
                               boost_eligible=False)
        base = ControllerBaselineDecision(zone_id="z", target_c=21.0, trv_setpoint_c=21.0,
                                         boost_offset_c=0.0)
        trace, _ = r.resolve(zin, base, preds(boost=2.0), mode=DecisionMode.CONTROL)
        boost_entry = next(e for e in trace.entries if e.feature == "boost_offset")
        assert not boost_entry.applied
        assert boost_entry.reason == "boost_not_eligible"
