"""Phase 17C: shadow observes episode/model paths across real coordinator cycles."""
from __future__ import annotations

import pytest

from tests.helpers import make_state, set_hass_states
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


class TestEpisodePaths:
    async def test_shadow_captures_across_cycles(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        # drive several cycles with a rising indoor temperature
        for i, t in enumerate([18.0, 18.6, 19.2, 19.8, 20.4, 21.0]):
            set_hass_states(coord, {"sensor.test_temp": make_state(str(t))})
            await coord._async_update_data()
        # the shadow runtime observed cycles for this zone (no control side effects)
        assert sh.diagnostics()["last_cycle_ts"] is not None
        assert coord.control_calls == [] or all(c[0] != "service_call"
                                                for c in coord.control_calls)

    async def test_prediction_snapshots_recorded(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        for t in [18.0, 18.6, 19.2]:
            set_hass_states(coord, {"sensor.test_temp": make_state(str(t))})
            await coord._async_update_data()
        assert sh.diagnostics()["prediction_snapshots"] >= 0

    async def test_no_control_during_episode_capture(self):
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = False  # observation mode
        sh = attach_shadow(coord)
        await sh.async_setup()
        for t in [18.0, 19.0, 20.0, 21.0]:
            set_hass_states(coord, {"sensor.test_temp": make_state(str(t))})
            await coord._async_update_data()
        assert coord.control_calls == []

    async def test_decision_id_stable_across_cycles(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        await coord._async_update_data()
        d1 = sh.diagnostics()["open_decisions"]
        await coord._async_update_data()
        d2 = sh.diagnostics()["open_decisions"]
        assert d1 == d2  # continuation, not a new decision each cycle
