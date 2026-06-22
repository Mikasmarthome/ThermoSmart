"""Phase 17C: multi-zone HA isolation."""
from __future__ import annotations

import pytest

from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


class TestHaMultiZone:
    async def test_two_zones_distinct_decisions(self):
        a = make_recording_coordinator(indoor="18.0")
        a.entry.entry_id = "zone_a"
        b = make_recording_coordinator(indoor="20.0")
        b.entry.entry_id = "zone_b"
        sa = attach_shadow(a, zone_id="zone_a")
        sb = attach_shadow(b, zone_id="zone_b")
        await sa.async_setup()
        await sb.async_setup()
        await a._async_update_data()
        await b._async_update_data()
        assert sa.diagnostics()["last_cycle_ts"] is not None
        assert sb.diagnostics()["last_cycle_ts"] is not None

    async def test_zone_error_isolated(self):
        a = make_recording_coordinator(indoor="18.0")
        a.entry.entry_id = "zone_a"
        b = make_recording_coordinator(indoor="20.0")
        b.entry.entry_id = "zone_b"
        sa = attach_shadow(a, zone_id="zone_a")
        sb = attach_shadow(b, zone_id="zone_b")
        await sa.async_setup()
        await sb.async_setup()
        sa._runtime.run_cycle_safe = lambda inp: (_ for _ in ()).throw(RuntimeError("x"))
        await a._async_update_data()   # zone a learning fails
        rb = await b._async_update_data()  # zone b unaffected
        assert rb is not None and sb.errors == 0

    async def test_separate_runtimes(self):
        a = make_recording_coordinator()
        a.entry.entry_id = "zone_a"
        b = make_recording_coordinator()
        b.entry.entry_id = "zone_b"
        sa = attach_shadow(a, zone_id="zone_a")
        sb = attach_shadow(b, zone_id="zone_b")
        assert sa.runtime is not sb.runtime
