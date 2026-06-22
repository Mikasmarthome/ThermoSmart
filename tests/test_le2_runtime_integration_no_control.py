"""Phase 17B integration-level no-control guarantee (bridge over full pipeline)."""
from __future__ import annotations

import dataclasses

import pytest

from custom_components.thermosmart.learning.runtime import (
    CoordinatorBridge,
    RuntimeCycleResult,
)
from tests.helpers_runtime_scenarios import heating_ramp_then_settle, runtime, step


class TestIntegrationNoControl:
    def test_bridge_never_returns_control(self):
        rt = runtime()
        bridge = CoordinatorBridge(rt)
        from tests.helpers_runtime import cycle_input
        r = bridge.process(cycle_input("2025-01-01T06:00:00+00:00", zone="lz"))
        assert r.control_commands == () and not r.has_control_effect

    def test_full_learning_run_no_control_effect(self):
        rt = runtime()
        results = heating_ramp_then_settle(rt)
        assert all(r.control_commands == () for r in results)
        assert all(not r.has_control_effect for r in results)

    def test_high_confidence_no_control(self):
        rt = runtime()
        # learn a lot, then a fresh cycle still emits no command
        for _ in range(3):
            heating_ramp_then_settle(rt)
        last = step(rt, 999, 19.0, heating=True)
        assert last.control_commands == ()

    def test_bridge_isolates_exception(self):
        rt = runtime()
        # corrupt a model so the cycle would raise internally
        class _Boom:
            def predict(self, ctx):
                raise RuntimeError("boom")
        # force an error deep in orchestration; bridge must still not raise
        rt._zone("lz").orchestrator._models["heat_rate"] = _Boom()
        bridge = CoordinatorBridge(rt)
        from tests.helpers_runtime import cycle_input
        r = bridge.process(cycle_input("2025-01-01T06:00:00+00:00", zone="lz"))
        assert isinstance(r, RuntimeCycleResult)  # no raise, structured result

    def test_result_type_not_control_compatible(self):
        rt = runtime()
        from tests.helpers_runtime import cycle_input
        r = rt.run_cycle(cycle_input("2025-01-01T06:00:00+00:00", zone="lz"))
        # cannot be flipped into a control-bearing object
        with pytest.raises(ValueError):
            dataclasses.replace(r, has_control_effect=True)

    def test_no_setpoint_extractable(self):
        rt = runtime()
        results = heating_ramp_then_settle(rt)
        for r in results:
            for sp in r.shadow_predictions:
                assert not getattr(sp, "control_effect", False)
