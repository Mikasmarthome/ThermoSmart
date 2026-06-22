"""Phase 18 control diagnostics/export tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.confidence import ConfidencePurpose
from custom_components.thermosmart.learning.runtime import (
    LearningRuntime, LearningRuntimeConfig, LearningRuntimeMode)
from custom_components.thermosmart.learning.runtime.control import ControlFeature
from tests.helpers_control import strong_preheat
from tests.helpers_runtime_scenarios import heating_ramp_then_settle
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


def _rt():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.CONTROL,
                                                 startup_grace_cycles=0))


class TestControlDiagnostics:
    def test_health_control_fields(self):
        rt = _rt()
        h = rt.health()
        for f in ("control_enabled", "control_policy_version", "control_applied",
                  "control_fallback", "control_rejections", "reversion_count"):
            assert hasattr(h, f)
        assert h.control_enabled is True

    def test_applied_count_in_health(self):
        rt = _rt()
        heating_ramp_then_settle(rt)
        rt._zone("lz").last_confidence["preheat"] = strong_preheat()
        rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 66.0,
                            ConfidencePurpose.PREHEAT, unit="minutes")
        assert rt.health().control_applied >= 1

    def test_rejection_count_in_health(self):
        rt = _rt()
        rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 66.0,
                            ConfidencePurpose.PREHEAT, unit="minutes")  # no confidence -> fallback
        assert rt.health().control_fallback >= 1

    async def test_controller_diagnostics_control_summary(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)  # SHADOW
        await sh.async_setup()
        d = sh.diagnostics()
        assert "control_enabled" in d and d["control_enabled"] is False
        assert "control_applied" in d and "control_rejections" in d

    def test_export_no_ids(self):
        rt = _rt()
        heating_ramp_then_settle(rt)
        rt._zone("lz").last_confidence["preheat"] = strong_preheat()
        rt.control_decision("lz", ControlFeature.PREHEAT_START, 60.0, 66.0,
                            ConfidencePurpose.PREHEAT, unit="minutes")
        samples = rt._zone("lz").control_ledger.research_samples()
        assert "decision_id" not in json.dumps(samples)
