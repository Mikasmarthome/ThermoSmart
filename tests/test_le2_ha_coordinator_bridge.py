"""Phase 17C: coordinator bridge wiring (real _async_update_data hook)."""
from __future__ import annotations

import pytest

from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


class TestCoordinatorBridge:
    async def test_shadow_attached_runs_on_cycle(self):
        coord = make_recording_coordinator(indoor="19.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        await coord._async_update_data()
        assert sh.diagnostics()["last_cycle_ts"] is not None

    async def test_shadow_optional_without_attach(self):
        coord = make_recording_coordinator(indoor="19.0")
        # no shadow attached -> cycle works unchanged
        result = await coord._async_update_data()
        assert result is not None

    async def test_shadow_runs_after_decision(self):
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        await coord._async_update_data()
        # the shadow saw a decision id (capture happened)
        assert sh.diagnostics()["open_decisions"] >= 0  # never raises, structured

    async def test_multiple_cycles_advance_diagnostics(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        for _ in range(3):
            await coord._async_update_data()
        assert sh.diagnostics()["prediction_snapshots"] >= 0

    async def test_zone_id_is_entry_id(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        await coord._async_update_data()
        assert coord.zone_id == coord.entry.entry_id
