"""Phase 18 shadow compatibility: SHADOW behaves exactly as before Phase 18."""
from __future__ import annotations

import pytest

from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator

# Shadow-diagnostic status fields that legitimately differ when a shadow is attached:
# "not_available" (no shadow) vs "missing" (shadow present, prediction not yet produced)
# are both valid no-cutoff states; only the CONTROL output must be identical.
_SHADOW_STATUS_KEYS = frozenset({
    "early_cutoff_status", "early_cutoff_state", "forecast_trust_status", "preheat_status",
    # Phase 19D: HeatLoss source diagnostic fields differ between shadow / no-shadow.
    "tpi_coef_source", "tpi_hl_rate",
    "tpi_coef_int", "tpi_coef_ext", "tpi_coef_diag",
})


def _zone_control_equal(a: dict, b: dict) -> bool:
    """Compare zone dicts for control equality, ignoring shadow-diagnostic status fields."""
    return ({k: v for k, v in a.items() if k not in _SHADOW_STATUS_KEYS} ==
            {k: v for k, v in b.items() if k not in _SHADOW_STATUS_KEYS})


async def _run(indoor, *, active, with_shadow):
    coord = make_recording_coordinator(indoor=indoor)
    coord._active_control = active
    if with_shadow:
        sh = attach_shadow(coord)  # default SHADOW
        await sh.async_setup()
    return coord.control_calls, await coord._async_update_data()


class TestShadowCompatibility:
    @pytest.mark.parametrize("indoor,active", [
        ("19.0", False), ("18.0", True), ("16.0", True), ("23.0", True)])
    async def test_shadow_identical_to_no_shadow(self, indoor, active):
        off_calls, off = await _run(indoor, active=active, with_shadow=False)
        on_calls, on = await _run(indoor, active=active, with_shadow=True)
        assert off_calls == on_calls and _zone_control_equal(off["zone"], on["zone"])

    async def test_no_control_ledger_application_in_shadow(self):
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        for _ in range(3):
            await coord._async_update_data()
        assert sh.diagnostics()["control_applied"] == 0

    async def test_setpoint_not_adjusted_in_shadow(self):
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        result = await coord._async_update_data()
        assert "le2_boost_adjusted" not in result["zone"]

