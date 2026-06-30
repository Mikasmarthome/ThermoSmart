"""Phase 18 preheat control + clamp/roll-in tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime.control import (
    ControlFeature,
    ControlPolicy,
    ControlPolicyParameters,
    FeatureClamp,
)
from tests.helpers_control import control_decision, strong_preheat


F = ControlFeature.PREHEAT_START


class TestPreheatControl:
    def test_sufficient_confidence_applies(self):
        d = control_decision(F, 60.0, 70.0, strong_preheat(), unit="minutes")
        assert d.applied

    def test_low_confidence_falls_back(self):
        from tests.helpers_control import weak_purpose
        from custom_components.thermosmart.learning.confidence import (
            ComponentKind, ConfidencePurpose)
        d = control_decision(F, 60.0, 70.0,
                             weak_purpose(ConfidencePurpose.PREHEAT, ComponentKind.HEAT_RATE),
                             unit="minutes")
        assert not d.applied and d.final_value == 60.0

    def test_positive_clamp_roll_in(self):
        # proposed far above baseline -> clamped to baseline + max_step (15)
        d = control_decision(F, 60.0, 85.0, strong_preheat(), unit="minutes")
        assert d.applied and d.final_value == 75.0 and d.clamp_applied > 0

    def test_negative_clamp_roll_in(self):
        d = control_decision(F, 60.0, 30.0, strong_preheat(), unit="minutes")
        assert d.applied and d.final_value == 45.0  # 60 - max_step(15)

    def test_min_clamp(self):
        params = ControlPolicyParameters(clamps={F: FeatureClamp(20.0, 180.0, 100.0, 200.0)})
        pol = ControlPolicy(params, runtime_is_control=True)
        d = control_decision(F, 25.0, 5.0, strong_preheat(), unit="minutes", policy=pol)
        assert d.applied and d.final_value == 20.0  # floored at min

    def test_max_clamp(self):
        params = ControlPolicyParameters(clamps={F: FeatureClamp(0.0, 90.0, 100.0, 200.0)})
        pol = ControlPolicy(params, runtime_is_control=True)
        d = control_decision(F, 80.0, 150.0, strong_preheat(), unit="minutes", policy=pol)
        assert d.applied and d.final_value == 90.0

    def test_excessive_deviation_falls_back(self):
        d = control_decision(F, 10.0, 150.0, strong_preheat(), unit="minutes")
        assert not d.applied

    def test_no_change_when_equal(self):
        d = control_decision(F, 60.0, 60.0, strong_preheat(), unit="minutes")
        assert d.applied and d.final_value == 60.0 and d.clamp_applied == 0.0

    def test_baseline_preserved_on_fallback(self):
        d = control_decision(F, 55.0, 70.0, strong_preheat(), runtime_is_control=False)
        assert d.final_value == 55.0 and d.fallback is not None and d.fallback.safe
