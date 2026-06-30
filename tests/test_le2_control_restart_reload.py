"""Phase 18 restart/reload: CONTROL is never a persisted default."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.runtime import (
    LearningRuntime, LearningRuntimeConfig, LearningRuntimeMode)
from custom_components.thermosmart.learning.runtime.ha_integration import LearningShadowController
from tests.helpers_runtime import MemoryStore
from tests.helpers_runtime_scenarios import heating_ramp_then_settle

pytestmark = pytest.mark.asyncio

_T0 = datetime(2025, 1, 1, 7, 0, 0, tzinfo=timezone.utc)


class TestRestartReload:
    async def test_restart_defaults_to_shadow(self):
        store = MemoryStore()
        rt1 = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.CONTROL,
                                                    startup_grace_cycles=0),
                              store=store, clock=lambda: "2025-01-01T07:00:00+00:00")
        await rt1.async_setup()
        heating_ramp_then_settle(rt1)
        rt1.mark_dirty(important=True)
        await rt1.async_flush()
        # a fresh runtime constructed at default (SHADOW) restores learning but NOT control
        rt2 = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW,
                                                    startup_grace_cycles=0),
                              store=MemoryStore(data=store.data),
                              clock=lambda: "2025-01-01T08:00:00+00:00")
        await rt2.async_setup()
        assert rt2.mode is LearningRuntimeMode.SHADOW and not rt2.control_enabled

    async def test_controller_defaults_to_shadow(self):
        # default mode is SHADOW; production __init__.py explicitly passes CONTROL
        ctrl = LearningShadowController(
            hass=None, zone_id="z", store=MemoryStore(),
            clock=FakeClock(start=_T0),
        )
        await ctrl.async_setup()
        assert not ctrl.control_enabled

    async def test_persisted_state_has_no_control_activation(self):
        store = MemoryStore()
        rt = LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.CONTROL,
                                                   startup_grace_cycles=0),
                             store=store, clock=lambda: "2025-01-01T07:00:00+00:00")
        await rt.async_setup()
        heating_ramp_then_settle(rt)
        rt.mark_dirty(important=True)
        await rt.async_flush()
        import json
        blob = json.dumps(store.data)
        assert "control" not in blob.lower() or "control_" not in blob  # no control activation flag

    async def test_explicit_injection_enables_control(self):
        ctrl = LearningShadowController(
            hass=None, zone_id="z", store=MemoryStore(),
            mode=LearningRuntimeMode.CONTROL,
            clock=FakeClock(start=_T0),
        )
        await ctrl.async_setup()
        assert ctrl.control_enabled
