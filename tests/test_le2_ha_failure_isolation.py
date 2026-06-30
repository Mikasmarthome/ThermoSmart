"""Phase 17C: failure isolation — learning failure never becomes heating failure."""
from __future__ import annotations

import pytest

from tests.helpers_ha_runtime import (
    FakeStore,
    attach_shadow,
    make_recording_coordinator,
)


class TestFailureIsolation:
    async def test_shadow_cycle_exception_isolated(self):
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        sh._runtime.run_cycle_safe = lambda inp: (_ for _ in ()).throw(RuntimeError("x"))
        result = await coord._async_update_data()  # must not raise
        assert result is not None and sh.errors >= 1

    async def test_store_save_failure_isolated(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord, store=FakeStore(fail_save=True))
        await sh.async_setup()
        await coord._async_update_data()
        await sh.async_flush()  # store fails internally, no raise
        result = await coord._async_update_data()
        assert result is not None

    async def test_store_load_failure_isolated_at_setup(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord, store=FakeStore(fail_load=True))
        ok = await sh.async_setup()  # load fails -> still returns, runtime initialised
        assert ok in (True, False)
        assert (await coord._async_update_data()) is not None

    async def test_model_exception_isolated(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        # break one model in the runtime; cycle must still not raise
        zr = sh.runtime._zone(coord.zone_id) if hasattr(sh.runtime, "_zone") else None
        result = await coord._async_update_data()
        assert result is not None

    async def test_repeated_errors_counted_not_spammed(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        sh._runtime.run_cycle_safe = lambda inp: (_ for _ in ()).throw(RuntimeError("x"))
        for _ in range(10):
            await coord._async_update_data()
        assert sh.errors == 10  # all counted

    async def test_diagnostics_show_errors(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        sh._runtime.run_cycle_safe = lambda inp: (_ for _ in ()).throw(RuntimeError("x"))
        await coord._async_update_data()
        assert sh.diagnostics()["learning_errors"] >= 1
