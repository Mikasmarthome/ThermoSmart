"""Phase 17B preheat hardening tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.runtime import compute_preheat_plan


def plan(**kw):
    base = dict(comfort_time_utc="2026-01-02T07:00:00+00:00", comfort_temperature_c=21.0,
                current_temperature_c=18.0, heat_rate_c_per_h=2.0, heat_loss_c_per_h=0.3,
                afterheat_rise_c=0.3, confidence=0.8, fallback_used=False)
    base.update(kw)
    return compute_preheat_plan(**base)


class TestPreheatHardening:
    def test_small_deficit(self):
        p = plan(current_temperature_c=20.9)
        assert p.active_duration_min < 30 and p.recommended_start_utc is not None

    def test_large_deficit(self):
        p = plan(current_temperature_c=12.0)
        assert p.active_duration_min > 120

    def test_slow_room(self):
        slow = plan(heat_rate_c_per_h=0.6)
        fast = plan(heat_rate_c_per_h=4.0)
        assert slow.active_duration_min > fast.active_duration_min

    def test_strong_afterheat(self):
        strong = plan(afterheat_rise_c=2.0)
        weak = plan(afterheat_rise_c=0.0)
        assert strong.active_duration_min < weak.active_duration_min

    def test_near_zero_afterheat(self):
        p = plan(afterheat_rise_c=0.0)
        assert p.expected_early_cutoff_utc == p.comfort_time_utc or \
            p.expected_early_cutoff_utc <= p.comfort_time_utc

    def test_moderate_heat_loss(self):
        p = plan(heat_loss_c_per_h=0.8)
        assert math.isfinite(p.active_duration_min)

    def test_already_reached(self):
        p = plan(current_temperature_c=21.5)
        assert p.deficit_c == 0.0 and p.active_duration_min == pytest.approx(0.0, abs=1e-6)

    def test_comfort_time_in_past(self):
        p = compute_preheat_plan(comfort_time_utc="2020-01-01T07:00:00+00:00",
                                 comfort_temperature_c=21.0, current_temperature_c=18.0,
                                 heat_rate_c_per_h=2.0, heat_loss_c_per_h=0.3,
                                 afterheat_rise_c=0.3, confidence=0.8, fallback_used=False)
        # start would be in the past; value stays finite and safe
        assert p.recommended_start_utc is not None and math.isfinite(p.active_duration_min)

    def test_no_indoor_temp_no_exact_time(self):
        p = plan(current_temperature_c=None)
        assert p.recommended_start_utc is None and "no_indoor_temp" in p.reason_codes

    def test_dst_forward(self):
        p = compute_preheat_plan(comfort_time_utc="2026-03-29T03:00:00+02:00",
                                 comfort_temperature_c=21.0, current_temperature_c=18.0,
                                 heat_rate_c_per_h=2.0, heat_loss_c_per_h=0.3,
                                 afterheat_rise_c=0.3, confidence=0.7, fallback_used=False)
        assert p.recommended_start_utc is not None

    def test_dst_backward(self):
        p = compute_preheat_plan(comfort_time_utc="2026-10-25T02:30:00+01:00",
                                 comfort_temperature_c=21.0, current_temperature_c=18.0,
                                 heat_rate_c_per_h=2.0, heat_loss_c_per_h=0.3,
                                 afterheat_rise_c=0.3, confidence=0.7, fallback_used=False)
        assert p.recommended_start_utc is not None

    def test_no_nan_or_inf(self):
        for kw in ({}, dict(heat_rate_c_per_h=0.1, heat_loss_c_per_h=1.0),
                   dict(current_temperature_c=21.0), dict(afterheat_rise_c=5.0)):
            p = plan(**kw)
            if p.active_duration_min is not None:
                assert math.isfinite(p.active_duration_min)
            assert math.isfinite(p.expected_achievement_c) if p.expected_achievement_c else True

    def test_start_is_before_comfort(self):
        p = plan()
        assert p.recommended_start_utc < p.comfort_time_utc

    def test_summer_no_deficit(self):
        p = plan(current_temperature_c=24.0, comfort_temperature_c=21.0)
        assert p.deficit_c == 0.0
