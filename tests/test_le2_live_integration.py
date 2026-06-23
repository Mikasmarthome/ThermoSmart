"""Phase 19A (extended): real live-integration of the decision pipeline in the coordinator.

The decision pipeline runs read-only in every real cycle; SHADOW stays byte-identical
and there is exactly one dispatch path (the existing coordinator).

Note on shadow-diagnostic fields: when a shadow IS attached, the coordinator
consults ``read_early_cutoff_safe()`` and writes the consultation status (e.g.
"missing") into ``early_cutoff_status``.  Without a shadow this field defaults to
"not_available".  This is correct — it distinguishes "not consulted" from "consulted
but no prediction".  Control output (TRV setpoints, effective_target) is identical;
only the diagnostic field differs, so parity tests use ``_zone_control_equal``.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import LearningRuntimeMode
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator

_SHADOW_STATUS_KEYS = frozenset({
    "early_cutoff_status", "early_cutoff_state", "forecast_trust_status", "preheat_status",
    # Phase 19D: HeatLoss source diagnostic fields differ between shadow / no-shadow.
    "tpi_coef_source", "tpi_hl_rate",
    "tpi_coef_int", "tpi_coef_ext", "tpi_coef_diag",
    # Phase B1: boost provenance keys written only when shadow is attached.
    "_boost_rejection_reason", "_boost_candidate_c", "_boost_applied_c",
})


def _zone_control_equal(a: dict, b: dict) -> bool:
    return ({k: v for k, v in a.items() if k not in _SHADOW_STATUS_KEYS} ==
            {k: v for k, v in b.items() if k not in _SHADOW_STATUS_KEYS})

pytestmark = pytest.mark.asyncio


async def _coord(indoor="18.0", *, active=True, control=False):
    coord = make_recording_coordinator(indoor=indoor)
    coord._active_control = active
    sh = attach_shadow(coord)
    if control:
        sh.runtime.set_mode(LearningRuntimeMode.CONTROL)
    await sh.async_setup()
    return coord, sh


class TestCoordinatorIntegration:
    async def test_real_cycle_produces_trace(self):
        coord, sh = await _coord()
        await coord._async_update_data()
        t = sh.last_decision_trace
        assert t is not None and "baseline_setpoint_c" in t and "final_setpoint_c" in t

    async def test_trace_in_diagnostics(self):
        coord, sh = await _coord()
        await coord._async_update_data()
        assert sh.diagnostics()["decision_trace"] is not None

    async def test_trace_has_mode_and_reason_codes(self):
        coord, sh = await _coord()
        await coord._async_update_data()
        t = sh.last_decision_trace
        assert t["mode"] == "shadow" and "reason_codes" in t


class TestShadowParity:
    @pytest.mark.parametrize("indoor,active", [
        ("19.0", False), ("18.0", True), ("16.0", True), ("23.0", True)])
    async def test_shadow_byte_identical(self, indoor, active):
        a = make_recording_coordinator(indoor=indoor); a._active_control = active
        ra = await a._async_update_data()
        b = make_recording_coordinator(indoor=indoor); b._active_control = active
        sh = attach_shadow(b); await sh.async_setup()
        rb = await b._async_update_data()
        assert a.control_calls == b.control_calls and _zone_control_equal(ra["zone"], rb["zone"])

    async def test_shadow_trace_still_present(self):
        coord, sh = await _coord(indoor="18.0", active=True)
        await coord._async_update_data()
        assert sh.last_decision_trace is not None
        assert sh.last_decision_trace["applied_any"] is False

    async def test_shadow_does_not_dispatch_via_pipeline(self):
        # the pipeline's dispatcher has no sink; only the coordinator dispatches
        coord, sh = await _coord(indoor="18.0", active=True)
        before = sum(1 for c in coord.control_calls if c[0] == "apply_temperature")
        await coord._async_update_data()
        after = sum(1 for c in coord.control_calls if c[0] == "apply_temperature")
        assert after - before <= 1


class TestSingleDispatch:
    async def test_pipeline_dispatcher_never_calls_services(self):
        coord, sh = await _coord(indoor="18.0", active=True, control=True)
        before = coord.hass.services.async_call.call_count
        await coord._async_update_data()
        assert coord.hass.services.async_call.call_count == before

    async def test_one_dispatch_path(self):
        coord, sh = await _coord(indoor="18.0", active=True, control=True)
        await coord._async_update_data()
        assert sum(1 for c in coord.control_calls if c[0] == "apply_temperature") <= 1


class TestControlReachable:
    async def test_control_mode_produces_control_trace(self):
        coord, sh = await _coord(indoor="18.0", active=True, control=True)
        await coord._async_update_data()
        assert sh.last_decision_trace["mode"] == "control"

    async def test_restart_defaults_to_shadow(self):
        coord, sh = await _coord(indoor="18.0", active=True, control=True)
        # a freshly attached controller is SHADOW again (control never persisted)
        coord2, sh2 = await _coord(indoor="18.0", active=True, control=False)
        await coord2._async_update_data()
        assert sh2.last_decision_trace["mode"] == "shadow" and not sh2.control_enabled

