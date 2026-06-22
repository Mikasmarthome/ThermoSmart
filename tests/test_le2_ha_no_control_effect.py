"""Phase 17C: byte-/argument-equal no-control proof (LE2 shadow on vs off).

Runs the REAL coordinator update loop with and without the LE 2.0 shadow attached
and asserts the existing control actions and result are identical.

Note on shadow-diagnostic fields: when a shadow IS attached, the coordinator
consults ``read_early_cutoff_safe()`` and reflects the consultation result as
``early_cutoff_status`` (e.g. "missing" when no prediction exists yet). Without a
shadow, the field keeps its default ``"not_available"``. This is correct behaviour —
it distinguishes "not consulted" from "consulted, no prediction yet." The CONTROL
output (TRV setpoints) is identical in both cases; only the diagnostic field differs.
"""
from __future__ import annotations

import pytest

from tests.helpers_ha_runtime import (
    FakeStore,
    attach_shadow,
    make_recording_coordinator,
)

# Shadow-diagnostic status fields that legitimately differ when a shadow is attached
# ("not_available" = no shadow; "missing"/"cold_start"/... = shadow consulted).
_SHADOW_STATUS_KEYS = frozenset({"early_cutoff_status", "early_cutoff_state", "forecast_trust_status", "preheat_status"})


def _zone_control_equal(a: dict, b: dict) -> bool:
    """Compare zone dicts for control equality, ignoring shadow-diagnostic status fields."""
    return ({k: v for k, v in a.items() if k not in _SHADOW_STATUS_KEYS} ==
            {k: v for k, v in b.items() if k not in _SHADOW_STATUS_KEYS})


async def _run(indoor, *, active_control, with_shadow, store=None):
    coord = make_recording_coordinator(indoor=indoor)
    coord._active_control = active_control
    if with_shadow:
        sh = attach_shadow(coord, store=store)
        await sh.async_setup()
    result = await coord._async_update_data()
    return coord.control_calls, result


class TestNoControlEffect:
    @pytest.mark.parametrize("indoor,active", [
        ("19.0", False), ("19.0", True), ("16.0", True), ("23.0", True), ("15.0", False),
    ])
    async def test_control_identical_shadow_on_off(self, indoor, active):
        off_calls, off_res = await _run(indoor, active_control=active, with_shadow=False)
        on_calls, on_res = await _run(indoor, active_control=active, with_shadow=True)
        assert off_calls == on_calls
        assert _zone_control_equal(off_res["zone"], on_res["zone"])
        assert off_res["active_control"] == on_res["active_control"]

    async def test_high_confidence_no_control_change(self):
        # even after the shadow has learned, control stays identical
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        for _ in range(5):
            await coord._async_update_data()
        with_shadow = list(coord.control_calls)

        plain = make_recording_coordinator(indoor="18.0")
        plain._active_control = True
        for _ in range(5):
            await plain._async_update_data()
        assert with_shadow == plain.control_calls

    async def test_shadow_exception_no_control_change(self):
        off_calls, off_res = await _run("17.0", active_control=True, with_shadow=False)
        coord = make_recording_coordinator(indoor="17.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        # force the shadow to raise internally on every observe
        sh._runtime.run_cycle_safe = lambda inp: (_ for _ in ()).throw(RuntimeError("boom"))
        on_calls, on_res = (coord.control_calls, await coord._async_update_data())
        assert off_calls == on_calls and _zone_control_equal(off_res["zone"], on_res["zone"])

    async def test_store_failure_no_control_change(self):
        off_calls, off_res = await _run("17.0", active_control=True, with_shadow=False)
        on_calls, on_res = await _run("17.0", active_control=True, with_shadow=True,
                                      store=FakeStore(fail_save=True, fail_load=True))
        assert off_calls == on_calls and _zone_control_equal(off_res["zone"], on_res["zone"])

    async def test_no_update_failed_from_shadow(self):
        coord = make_recording_coordinator(indoor="19.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        sh.observe_safe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        # must not raise UpdateFailed
        result = await coord._async_update_data()
        assert result is not None
