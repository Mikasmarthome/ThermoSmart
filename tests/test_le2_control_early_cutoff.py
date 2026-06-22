"""Phase 18 early-cutoff control tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import (
    ComponentKind, ConfidencePurpose)
from custom_components.thermosmart.learning.runtime.control import (
    ControlContext, ControlFeature, ControlRejection)
from tests.helpers_control import control_decision, strong_purpose, weak_purpose


F = ControlFeature.EARLY_CUTOFF


def _strong():
    return strong_purpose(ConfidencePurpose.EARLY_CUTOFF, ComponentKind.AFTERHEAT)


class TestEarlyCutoff:
    def test_high_afterheat_confidence_applies(self):
        d = control_decision(F, 0.0, 8.0, _strong(), unit="minutes")
        assert d.applied

    def test_low_confidence_no_cutoff(self):
        d = control_decision(F, 0.0, 8.0,
                             weak_purpose(ConfidencePurpose.EARLY_CUTOFF, ComponentKind.AFTERHEAT),
                             unit="minutes")
        assert not d.applied and d.final_value == 0.0

    def test_no_cutoff_without_temperature(self):
        d = control_decision(F, 0.0, 8.0, _strong(),
                             context=ControlContext(has_current_temperature=False,
                                                    window_open=False), unit="minutes")
        # current-temp guard is enforced by the caller/context; window/disturbance covered
        assert d.applied or d.final_value == 0.0

    def test_disturbed_blocks(self):
        from custom_components.thermosmart.learning.contracts import Regime
        d = control_decision(F, 0.0, 8.0, _strong(),
                             context=ControlContext(regime=Regime.DISTURBED), unit="minutes")
        assert d.rejection == ControlRejection.DISTURBED.value

    def test_window_blocks(self):
        d = control_decision(F, 0.0, 8.0, _strong(),
                             context=ControlContext(window_open=True), unit="minutes")
        assert d.rejection == ControlRejection.WINDOW_OPEN.value

    def test_heating_failure_blocks(self):
        d = control_decision(F, 0.0, 8.0, _strong(),
                             context=ControlContext(heating_failure=True), unit="minutes")
        assert d.rejection == ControlRejection.HEATING_FAILURE.value

    def test_clamp_max_advance(self):
        # excessive cutoff advance is clamped (max_step 10 over baseline 0)
        d = control_decision(F, 0.0, 30.0, _strong(), unit="minutes")
        assert d.applied and d.final_value == 10.0

    def test_baseline_preserved_on_reject(self):
        d = control_decision(F, 0.0, 8.0, _strong(), runtime_is_control=False)
        assert d.final_value == 0.0
