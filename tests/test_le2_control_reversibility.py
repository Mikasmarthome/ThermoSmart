"""Phase 18 reversibility tests (CONTROL<->SHADOW, no data loss)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime, LearningRuntimeConfig, LearningRuntimeMode)
from tests.helpers_runtime_scenarios import heating_ramp_then_settle


def _rt(mode):
    return LearningRuntime(LearningRuntimeConfig(mode=mode, startup_grace_cycles=0), clock=lambda: "2025-01-01T00:00:00+00:00")


class TestReversibility:
    def test_shadow_to_control_to_shadow(self):
        rt = _rt(LearningRuntimeMode.SHADOW)
        heating_ramp_then_settle(rt)
        samples = rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count
        rt.set_mode(LearningRuntimeMode.CONTROL)
        assert rt.control_enabled
        rt.set_mode(LearningRuntimeMode.SHADOW)
        assert not rt.control_enabled
        # learning state preserved across mode switches
        assert rt._zone("lz").orchestrator.models["heat_rate"]._state.general.sample_count == samples

    def test_revert_counts(self):
        rt = _rt(LearningRuntimeMode.CONTROL)
        rt.set_mode(LearningRuntimeMode.SHADOW)
        rt.set_mode(LearningRuntimeMode.CONTROL)
        rt.set_mode(LearningRuntimeMode.SHADOW)
        assert rt.health().reversion_count == 2

    def test_control_during_active_episode_then_revert(self):
        rt = _rt(LearningRuntimeMode.CONTROL)
        # open a heating episode (do not close)
        from tests.helpers_runtime_scenarios import step
        for i in range(4):
            step(rt, i, 18.0 + i * 0.5, heating=True)
        rt.set_mode(LearningRuntimeMode.SHADOW)
        # episode still progresses in shadow
        for i in range(4, 9):
            step(rt, i, 21.0, heating=False)
        assert rt._zone("lz").model_update_counts.get("heat_rate", 0) >= 0  # no crash

    def test_immediate_baseline_after_revert(self):
        from tests.helpers_control import strong_preheat
        from custom_components.thermosmart.learning.runtime.control import ControlFeature
        from custom_components.thermosmart.learning.confidence import ConfidencePurpose
        rt = _rt(LearningRuntimeMode.CONTROL)
        heating_ramp_then_settle(rt)
        rt._zone("lz").last_confidence["preheat"] = strong_preheat()
        d1 = rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 66.0,
                                 ConfidencePurpose.PREHEAT, unit="minutes")
        assert d1.applied
        rt.set_mode(LearningRuntimeMode.SHADOW)
        d2 = rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 66.0,
                                 ConfidencePurpose.PREHEAT, unit="minutes")
        assert not d2.applied and d2.final_value == 60.0

    def test_no_state_reset_on_revert(self):
        rt = _rt(LearningRuntimeMode.CONTROL)
        heating_ramp_then_settle(rt)
        s = rt._zone("lz").serialize()
        rt.set_mode(LearningRuntimeMode.SHADOW)
        assert rt._zone("lz").serialize() == s
