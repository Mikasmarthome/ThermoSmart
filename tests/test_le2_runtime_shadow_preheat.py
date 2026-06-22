"""Phase 17 shadow preheat orchestration tests (comfort-time = achievement time)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import compute_preheat_plan
from custom_components.thermosmart.learning.runtime.shadow import PreheatParameters


def plan(**kw):
    base = dict(comfort_time_utc="2026-06-22T07:00:00+00:00", comfort_temperature_c=21.0,
                current_temperature_c=18.0, heat_rate_c_per_h=2.0, heat_loss_c_per_h=0.3,
                afterheat_rise_c=0.5, confidence=0.8, fallback_used=False)
    base.update(kw)
    return compute_preheat_plan(**base)


class TestPreheat:
    def test_schedule_time_is_achievement_not_start(self):
        p = plan()
        # heating starts well before the comfort time
        assert p.recommended_start_utc < "2026-06-22T07:00:00+00:00"
        assert p.comfort_time_utc == "2026-06-22T07:00:00+00:00"

    def test_deficit_and_duration(self):
        p = plan(current_temperature_c=18.0, comfort_temperature_c=21.0)
        assert p.deficit_c == pytest.approx(3.0)
        assert p.active_duration_min > 0

    def test_afterheat_reduces_active_duration(self):
        with_af = plan(afterheat_rise_c=1.0)
        without_af = plan(afterheat_rise_c=0.0)
        assert with_af.active_duration_min < without_af.active_duration_min

    def test_heat_loss_extends_duration(self):
        low = plan(heat_loss_c_per_h=0.1)
        high = plan(heat_loss_c_per_h=1.0)
        assert high.active_duration_min > low.active_duration_min

    def test_expected_achievement_reaches_target(self):
        p = plan()
        assert p.expected_achievement_c == pytest.approx(21.0, abs=0.2)

    def test_early_cutoff_before_comfort(self):
        p = plan(afterheat_rise_c=0.8)
        assert p.expected_early_cutoff_utc < "2026-06-22T07:00:00+00:00"

    def test_no_indoor_temp_safe_fallback(self):
        p = plan(current_temperature_c=None)
        assert p.recommended_start_utc is None and p.active_duration_min is None
        assert "no_indoor_temp" in p.reason_codes
        assert p.confidence <= 0.3

    def test_forecast_optional_still_works(self):
        # forecast is not an input to the local plan -> local preheat fully functional
        p = plan()
        assert p.recommended_start_utc is not None

    def test_low_confidence_flag(self):
        p = plan(confidence=0.2)
        assert "low_confidence" in p.reason_codes

    def test_fallback_flag(self):
        p = plan(fallback_used=True)
        assert "fallback_used" in p.reason_codes

    def test_already_warm_zero_deficit(self):
        p = plan(current_temperature_c=21.5, comfort_temperature_c=21.0)
        assert p.deficit_c == 0.0 and p.active_duration_min == pytest.approx(0.0, abs=1e-6)

    def test_net_rate_floor(self):
        # heat loss >= heat rate -> floored net rate, finite duration, no crash
        p = plan(heat_rate_c_per_h=0.3, heat_loss_c_per_h=1.0)
        assert p.active_duration_min is not None and p.active_duration_min >= 0

    def test_no_control_effect(self):
        assert plan().control_effect is False

    def test_dst_safe_iso_handling(self):
        p = compute_preheat_plan(
            comfort_time_utc="2026-03-29T03:00:00+02:00", comfort_temperature_c=21.0,
            current_temperature_c=19.0, heat_rate_c_per_h=2.0, heat_loss_c_per_h=0.3,
            afterheat_rise_c=0.3, confidence=0.7, fallback_used=False)
        assert p.recommended_start_utc is not None
