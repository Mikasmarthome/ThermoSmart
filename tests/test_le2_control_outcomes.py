"""Phase 18 outcome binding: an applied control decision records its decision-time context."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import ConfidencePurpose
from custom_components.thermosmart.learning.runtime import (
    LearningRuntime, LearningRuntimeConfig, LearningRuntimeMode)
from custom_components.thermosmart.learning.runtime.control import ControlFeature
from tests.helpers_control import strong_preheat
from tests.helpers_runtime_scenarios import heating_ramp_then_settle


def _rt():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.CONTROL,
                                                 startup_grace_cycles=0),
                              clock=lambda: "2025-01-01T00:00:00+00:00")


class TestControlOutcomes:
    def test_applied_decision_in_ledger_with_confidence(self):
        rt = _rt()
        heating_ramp_then_settle(rt)
        rt._zone("lz").last_confidence["preheat"] = strong_preheat()
        d = rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 66.0,
                                ConfidencePurpose.PREHEAT, unit="minutes")
        entry = rt._zone("lz").control_ledger.recent()[-1]
        assert entry.applied == d.applied
        assert entry.baseline_value == 60.0 and entry.proposed_value == 66.0
        assert entry.confidence_value is not None

    def test_decision_time_confidence_not_recomputed(self):
        rt = _rt()
        heating_ramp_then_settle(rt)
        cr = strong_preheat()
        rt._zone("lz").last_confidence["preheat"] = cr
        d = rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 66.0,
                                ConfidencePurpose.PREHEAT, unit="minutes")
        assert d.confidence.value == round(cr.value, 6)

    def test_final_value_recorded(self):
        rt = _rt()
        heating_ramp_then_settle(rt)
        rt._zone("lz").last_confidence["preheat"] = strong_preheat()
        rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 90.0,
                            ConfidencePurpose.PREHEAT, unit="minutes")
        entry = rt._zone("lz").control_ledger.recent()[-1]
        assert entry.final_value == 75.0  # clamped baseline+max_step

    def test_versions_recorded(self):
        rt = _rt()
        heating_ramp_then_settle(rt)
        rt._zone("lz").last_confidence["preheat"] = strong_preheat()
        rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 66.0,
                            ConfidencePurpose.PREHEAT, unit="minutes",
                            model_version=1, parameter_version=1)
        entry = rt._zone("lz").control_ledger.recent()[-1]
        assert entry.policy_version == 1
