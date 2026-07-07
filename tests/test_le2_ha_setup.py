"""Phase 17C: shadow controller setup lifecycle (attach to real coordinator)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import LearningRuntimeMode
from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_recording_coordinator


class TestHaSetup:
    async def test_setup_initializes_runtime(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        assert await sh.async_setup()
        assert sh.runtime.initialized

    async def test_mode_is_shadow(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        assert sh.runtime.mode is LearningRuntimeMode.SHADOW

    async def test_setup_failure_does_not_raise(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord, store=FakeStore(fail_load=True))
        # load failure is isolated inside the runtime; setup still completes
        assert await sh.async_setup() in (True, False)

    async def test_attach_then_cycle(self):
        coord = make_recording_coordinator(indoor="19.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        result = await coord._async_update_data()
        assert result is not None and sh.diagnostics()["enabled"]

    async def test_diagnostics_shape(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        d = sh.diagnostics()
        for k in ("mode", "initialized", "dirty", "open_decisions", "model_update_total",
                  "prediction_snapshots", "learning_errors", "enabled"):
            assert k in d
