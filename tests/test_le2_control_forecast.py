"""Phase 18 forecast control behaviour (advisory-only; optional)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import ComponentKind, ConfidencePurpose
from custom_components.thermosmart.learning.runtime.control import (
    ControlFeature, ControlRejection)
from tests.helpers_control import control_decision, strong_purpose


class TestForecastControl:
    def test_forecast_adjustment_advisory_only(self):
        d = control_decision(ControlFeature.FORECAST_ADJUSTMENT, 0.0, 0.5,
                             strong_purpose(ConfidencePurpose.FORECAST, ComponentKind.FORECAST))
        assert not d.applied and d.rejection == ControlRejection.FEATURE_NOT_ENABLED.value

    def test_forecast_never_applies_directly(self):
        d = control_decision(ControlFeature.FORECAST_ADJUSTMENT, 0.0, 1.0,
                             strong_purpose(ConfidencePurpose.FORECAST, ComponentKind.FORECAST),
                             runtime_is_control=True)
        assert d.final_value == 0.0 or d.final_value is None

    def test_local_preheat_independent_of_forecast(self):
        # preheat control works without any forecast component present
        from tests.helpers_control import strong_preheat
        d = control_decision(ControlFeature.PREHEAT_START, 60.0, 68.0, strong_preheat(),
                             unit="minutes")
        assert d.applied
