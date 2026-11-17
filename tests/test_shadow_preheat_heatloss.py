"""Phase D3 follow-up — diagnostic shadow preheat heat-loss double-subtraction.

``shadow.compute_preheat_plan`` used ``net_rate = heat_rate - heat_loss`` for the shadow
heating duration. HeatRate is already the OBSERVED NET temperature rise (deg C/h over the
active heating phase, see HeatRateModel) — the rise that remained AFTER the room's
simultaneous passive heat loss. Subtracting HEAT_LOSS_RATE again double-counted the loss
and inflated the shadow duration. The follow-up uses the net rate as-is, mirroring the
active planner corrected in Phase D3 (``read_preheat_minutes_safe``):

    effective_rate    = max(net learned heat rate, min_net_rate floor)   # no HL subtraction
    effective_deficit = max(0, deficit - afterheat)
    active_duration   = effective_deficit / effective_rate * 60

HEAT_LOSS_RATE is retained as the diagnostic ``expected_heat_loss_c_per_h`` field but is no
longer applied to the duration. These are direct golden tests of the pure planner plus the
end-to-end shadow comparison contract, and they demonstrate the OLD defect: changing only
the heat-loss input used to change the duration, and now must not.

The shadow planner intentionally differs from the active planner: it has no confidence gate
or 180-minute cap and uses a fixed start margin rather than a learned onset delay. Those
differences are preserved here on purpose.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.runtime.shadow import (
    ComparisonType,
    PreheatParameters,
    ShadowOrchestrator,
    ShadowStatus,
    compute_preheat_plan,
)

_COMFORT = "2026-01-12T07:00:00+00:00"


def _plan(*, current=17.5, comfort=21.0, heat_rate=4.0, heat_loss=0.8, afterheat=0.0,
          confidence=0.9, fallback=False, params=None):
    return compute_preheat_plan(
        comfort_time_utc=_COMFORT, comfort_temperature_c=comfort,
        current_temperature_c=current, heat_rate_c_per_h=heat_rate,
        heat_loss_c_per_h=heat_loss, afterheat_rise_c=afterheat,
        confidence=confidence, fallback_used=fallback, params=params)


# ── core correction: golden plan uses the net heat rate ─────────────────────────

class TestNetHeatRate:
    def test_golden_plan_fields(self):
        p = _plan(heat_rate=4.0, heat_loss=0.8)
        # deficit 3.5 / net rate 4.0 * 60 = 52.5 min (OLD buggy value: 3.5/(4.0-0.8)*60 = 65.6)
        assert p.active_duration_min == pytest.approx(52.5, abs=0.01)
        assert p.active_duration_min < 65.0            # strictly below the double-subtracted value
        assert p.deficit_c == pytest.approx(3.5, abs=0.001)
        assert p.expected_heat_rate_c_per_h == pytest.approx(4.0, abs=0.0001)
        # HEAT_LOSS_RATE survives as a diagnostic value even though it does not change duration
        assert p.expected_heat_loss_c_per_h == pytest.approx(0.8, abs=0.0001)
        assert p.expected_afterheat_rise_c == pytest.approx(0.0, abs=0.0001)
        assert p.expected_achievement_c == pytest.approx(21.0, abs=0.001)
        # recommended start = comfort - (duration + start_margin 5.0) = 07:00 - 57.5 min
        assert p.recommended_start_utc == "2026-01-12T06:02:30+00:00"
        assert p.confidence == pytest.approx(0.9, abs=0.0001)
        assert p.fallback_used is False
        assert p.reason_codes == ("ok",)
        assert p.control_effect is False

    def test_changing_only_heat_loss_does_not_change_duration(self):
        # OLD-defect detector: pre-fix these differed wildly (net 3.5 -> 60 min vs
        # net max(1.0,0.2) -> 210 min). After the fix heat loss is irrelevant to duration.
        low = _plan(heat_loss=0.5)
        high = _plan(heat_loss=3.0)
        assert low.active_duration_min == high.active_duration_min == pytest.approx(52.5, abs=0.01)
        assert low.recommended_start_utc == high.recommended_start_utc
        # ...but it is still reported as a distinct diagnostic value
        assert low.expected_heat_loss_c_per_h == pytest.approx(0.5, abs=0.0001)
        assert high.expected_heat_loss_c_per_h == pytest.approx(3.0, abs=0.0001)

    def test_valid_vs_fallback_heat_loss_same_duration(self):
        # A learned heat loss (0.8) and the fallback heat-loss prior (0.3) must yield the
        # same duration; only the diagnostic field differs.
        valid = _plan(heat_loss=0.8)
        prior = _plan(heat_loss=0.3)
        assert valid.active_duration_min == prior.active_duration_min
        assert valid.expected_heat_loss_c_per_h != prior.expected_heat_loss_c_per_h


# ── preserved behaviour ─────────────────────────────────────────────────────────

class TestPreservedBehaviour:
    def test_afterheat_shortens_duration(self):
        none = _plan(afterheat=0.0)
        some = _plan(afterheat=1.0)
        # (3.5-1.0)/4.0*60 = 37.5 ; achievement still reaches comfort (21.0)
        assert none.active_duration_min == pytest.approx(52.5, abs=0.01)
        assert some.active_duration_min == pytest.approx(37.5, abs=0.01)
        assert some.expected_achievement_c == pytest.approx(21.0, abs=0.001)
        assert some.active_duration_min < none.active_duration_min

    def test_min_rate_floor_bounds_a_low_heat_rate(self):
        # effective_rate floors at 0.2 C/h: 3.5/0.2*60 = 1050 min for both 0.1 and 0.0.
        slow = _plan(heat_rate=0.1)
        zero = _plan(heat_rate=0.0)
        assert slow.active_duration_min == zero.active_duration_min == pytest.approx(1050.0, abs=0.01)

    def test_start_margin_is_additive_and_leaves_duration_untouched(self):
        # The shadow's fixed start margin plays the active planner's onset-delay role: it
        # shifts the recommended start earlier without changing the heating duration.
        base = _plan()
        wide = _plan(params=PreheatParameters(start_margin_min=15.0))
        assert base.active_duration_min == wide.active_duration_min
        assert wide.recommended_start_utc == "2026-01-12T05:52:30+00:00"   # 10 min earlier

    def test_low_confidence_flagged_but_duration_still_uses_net_rate(self):
        p = _plan(confidence=0.3)                      # below low_confidence_threshold 0.4
        assert "low_confidence" in p.reason_codes
        assert p.active_duration_min == pytest.approx(52.5, abs=0.01)   # no gate, by design

    def test_fallback_flagged_but_duration_still_uses_net_rate(self):
        # build_preheat_plan feeds the fallback heat-rate prior (1.5) when HR falls back.
        p = _plan(heat_rate=1.5, fallback=True)
        assert "fallback_used" in p.reason_codes
        assert p.active_duration_min == pytest.approx(3.5 / 1.5 * 60.0, abs=0.01)

    def test_no_indoor_temp_preserves_diagnostic_heat_loss(self):
        p = _plan(current=None)
        assert p.active_duration_min is None
        assert p.recommended_start_utc is None
        assert p.fallback_used is True
        assert "no_indoor_temp" in p.reason_codes
        assert p.expected_heat_loss_c_per_h == pytest.approx(0.8, abs=0.0001)


# ── end-to-end shadow comparison contract ───────────────────────────────────────

def _pred(values, *, confidence=0.9, fallback=False):
    return SimpleNamespace(values=values, confidence=confidence, fallback_used=fallback)


def _snapshot():
    return SimpleNamespace(
        zone_id="living", decision_id="d1", ts="2026-01-12T06:00:00+00:00",
        schedule_comfort_time_utc=_COMFORT, schedule_comfort_temperature_c=21.0,
        indoor_temp=SimpleNamespace(value=17.5))


def _predictions(*, heat_loss=0.8):
    return {
        PredictionType.HEAT_RATE: _pred({"heat_rate": 4.0}, confidence=0.9, fallback=False),
        PredictionType.HEAT_LOSS_RATE: _pred({"heat_loss_rate": heat_loss}),
        PredictionType.EXPECTED_OVERSHOOT: _pred({"expected_overshoot": 0.0}),
    }


class TestShadowComparisonContract:
    def test_build_and_compare_preheat_uses_corrected_duration(self):
        orch = ShadowOrchestrator()
        snap = _snapshot()
        plan = orch.build_preheat_plan(snap, _predictions())
        assert plan is not None
        assert plan.active_duration_min == pytest.approx(52.5, abs=0.01)
        assert plan.recommended_start_utc == "2026-01-12T06:02:30+00:00"

        cmp = orch.compare_preheat(snap, plan, "2026-01-12T06:00:00+00:00", 0.9)
        assert cmp is not None
        assert cmp.comparison_type == ComparisonType.PREHEAT_START.value
        assert cmp.status == ShadowStatus.OPEN.value
        assert cmp.baseline_value == pytest.approx(360.0, abs=0.001)   # 06:00
        assert cmp.shadow_value == pytest.approx(362.5, abs=0.001)     # 06:02:30
        assert cmp.difference == pytest.approx(2.5, abs=0.01)
        assert cmp.shadow_expected_outcome == pytest.approx(21.0, abs=0.001)

    def test_shadow_duration_is_invariant_to_heat_loss_prediction(self):
        orch = ShadowOrchestrator()
        snap = _snapshot()
        low = orch.build_preheat_plan(snap, _predictions(heat_loss=0.5))
        high = orch.build_preheat_plan(snap, _predictions(heat_loss=3.0))
        assert low.active_duration_min == high.active_duration_min
        assert low.recommended_start_utc == high.recommended_start_utc
        assert low.expected_heat_loss_c_per_h != high.expected_heat_loss_c_per_h
