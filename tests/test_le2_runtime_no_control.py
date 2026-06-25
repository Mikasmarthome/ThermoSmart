"""Phase 17 NO-CONTROL guarantee: shadow output can never become a control command."""
from __future__ import annotations

import dataclasses

import pytest

from custom_components.thermosmart.learning.runtime import (
    CoordinatorBridge,
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
    RuntimeCycleResult,
)
from tests.helpers_runtime import cycle_input


def shadow_runtime():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW), clock=lambda: "2025-01-01T00:00:00+00:00")


class TestNoControl:
    def test_result_has_no_control_commands(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert r.control_commands == ()
        assert r.has_control_effect is False

    def test_cannot_construct_control_result(self):
        with pytest.raises(ValueError):
            RuntimeCycleResult(zone_id="lr", ts="t", mode="shadow", decision_id=None,
                               captured=True, has_control_effect=True)

    def test_cannot_flip_control_via_replace(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        with pytest.raises(ValueError):
            dataclasses.replace(r, has_control_effect=True)

    def test_no_setpoint_or_target_fields(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        names = {f.name for f in dataclasses.fields(r)}
        for forbidden in ("setpoint", "new_setpoint", "target_command", "service_call",
                          "command", "apply"):
            assert forbidden not in names

    def test_shadow_predictions_no_control_effect(self):
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert all(not sp.control_effect for sp in r.shadow_predictions)

    def test_bridge_returns_only_diagnostics(self):
        bridge = CoordinatorBridge(shadow_runtime())
        r = bridge.process(cycle_input("2026-06-22T05:00:00+00:00"))
        assert isinstance(r, RuntimeCycleResult) and r.control_commands == ()

    def test_attempt_to_use_shadow_as_control_fails(self):
        # a caller trying to extract a setpoint to write gets nothing usable
        r = shadow_runtime().run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        assert getattr(r, "control_commands") == ()
        assert not any(getattr(sp, "control_effect", False) for sp in r.shadow_predictions)
