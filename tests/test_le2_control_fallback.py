"""Phase 18 fallback contract tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import ComponentKind, ConfidencePurpose
from custom_components.thermosmart.learning.runtime.control import (
    ControlContext, ControlFeature, ControlPolicy, ControlRejection)
from tests.helpers_control import control_decision, strong_preheat, weak_purpose

F = ControlFeature.PREHEAT_START


class TestFallback:
    def test_fallback_preserves_baseline(self):
        d = control_decision(F, 55.0, 70.0,
                             weak_purpose(ConfidencePurpose.PREHEAT, ComponentKind.HEAT_RATE),
                             unit="minutes")
        assert d.final_value == 55.0 and d.fallback.applied is False and d.fallback.safe is True

    def test_fallback_has_baseline_source(self):
        d = control_decision(F, 55.0, 70.0, strong_preheat(), runtime_is_control=False)
        assert d.fallback.baseline_source == "existing_controller"
        assert d.fallback.baseline_value == 55.0

    def test_fallback_records_prediction(self):
        d = control_decision(F, 55.0, 70.0,
                             weak_purpose(ConfidencePurpose.PREHEAT, ComponentKind.HEAT_RATE),
                             unit="minutes")
        assert d.fallback.le2_prediction == 70.0

    def test_nan_prediction_falls_back(self):
        d = control_decision(F, 55.0, float("inf"), strong_preheat(), unit="minutes")
        assert d.final_value == 55.0

    def test_excessive_deviation_falls_back(self):
        d = control_decision(F, 10.0, 150.0, strong_preheat(), unit="minutes")
        assert d.final_value == 10.0 and d.rejection == ControlRejection.EXCESSIVE_DEVIATION.value

    def test_device_limit_safe(self):
        d = control_decision(F, 60.0, 75.0, strong_preheat(), unit="minutes",
                             context=ControlContext(device_max=70.0))
        # final clamped within device max
        assert d.final_value <= 70.0

    def test_policy_exception_path_safe(self):
        # a broken clamp config -> feature treated as not-enabled -> baseline kept
        from custom_components.thermosmart.learning.runtime.control import ControlPolicyParameters
        pol = ControlPolicy(ControlPolicyParameters(clamps={}), runtime_is_control=True)
        d = control_decision(F, 60.0, 70.0, strong_preheat(), unit="minutes", policy=pol)
        assert d.final_value == 60.0
