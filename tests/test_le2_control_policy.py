"""Phase 18 ControlPolicy core contract tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.confidence import ComponentKind, ConfidencePurpose
from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.runtime.control import (
    CONTROL_POLICY_VERSION,
    ControlContext,
    ControlDecisionStatus,
    ControlFeature,
    ControlPolicy,
    ControlPolicyParameters,
    ControlRejection,
)
from tests.helpers_control import control_decision, strong_preheat, strong_purpose, weak_purpose


class TestPolicy:
    def test_applied_within_clamp(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 68.0, strong_preheat(),
                             unit="minutes")
        assert d.applied and d.final_value == 68.0 and d.status is ControlDecisionStatus.APPLIED

    def test_runtime_not_control_fallback(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 70.0, strong_preheat(),
                             runtime_is_control=False)
        assert not d.applied and d.rejection == ControlRejection.RUNTIME_NOT_CONTROL.value
        assert d.final_value == 60.0

    def test_advisory_only_feature_blocked(self):
        d = control_decision(ControlFeature.MANUAL_PREFERENCE_BIAS, 0.0, 0.3, strong_preheat())
        assert not d.applied and d.rejection == ControlRejection.FEATURE_NOT_ENABLED.value

    def test_forecast_advisory_only(self):
        d = control_decision(ControlFeature.FORECAST_ADJUSTMENT, 0.0, 0.5, strong_preheat())
        assert not d.applied

    def test_prediction_missing(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, None, strong_preheat(),
                             unit="minutes")
        assert d.rejection == ControlRejection.PREDICTION_MISSING.value and d.final_value == 60.0

    def test_baseline_missing(self):
        d = control_decision(ControlFeature.PREHEAT_START, None, 70.0, strong_preheat())
        assert d.rejection == ControlRejection.BASELINE_MISSING.value

    def test_nan_prediction_invalid(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, float("nan"), strong_preheat())
        assert d.rejection == ControlRejection.PREDICTION_INVALID.value

    def test_gate_closed_fallback(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 70.0,
                             weak_purpose(ConfidencePurpose.PREHEAT, ComponentKind.HEAT_RATE),
                             unit="minutes")
        assert not d.applied and d.final_value == 60.0

    def test_excessive_deviation_rejected(self):
        d = control_decision(ControlFeature.PREHEAT_START, 10.0, 200.0, strong_preheat(),
                             unit="minutes")
        assert d.rejection == ControlRejection.EXCESSIVE_DEVIATION.value

    def test_disturbed_rejected(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 70.0, strong_preheat(),
                             context=ControlContext(regime=Regime.DISTURBED), unit="minutes")
        assert d.rejection == ControlRejection.DISTURBED.value

    def test_window_open_rejected(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 70.0, strong_preheat(),
                             context=ControlContext(window_open=True), unit="minutes")
        assert d.rejection == ControlRejection.WINDOW_OPEN.value

    def test_heating_failure_rejected(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 70.0, strong_preheat(),
                             context=ControlContext(heating_failure=True), unit="minutes")
        assert d.rejection == ControlRejection.HEATING_FAILURE.value

    def test_frost_protection_rejected(self):
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 70.0, strong_preheat(),
                             context=ControlContext(frost_protection=True), unit="minutes")
        assert d.rejection == ControlRejection.FROST_PROTECTION.value

    def test_control_value_always_finite(self):
        for d in (control_decision(ControlFeature.PREHEAT_START, 60.0, 70.0, strong_preheat(),
                                   unit="minutes"),
                  control_decision(ControlFeature.PREHEAT_START, 60.0, None, strong_preheat())):
            assert d.control_value is not None and math.isfinite(d.control_value)

    def test_policy_version(self):
        assert CONTROL_POLICY_VERSION == 1
        assert ControlPolicyParameters().policy_version == 1

    def test_disabled_feature_via_params(self):
        params = ControlPolicyParameters(enabled_features=frozenset())
        pol = ControlPolicy(params, runtime_is_control=True)
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 70.0, strong_preheat(),
                             unit="minutes", policy=pol)
        assert d.rejection == ControlRejection.FEATURE_NOT_ENABLED.value
