"""Phase 17 failure degradation: learning may degrade, heating control must continue."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import MemoryStore, cycle_input


def rt(**kw):
    kw.setdefault("clock", lambda: "2025-01-01T00:00:00+00:00")
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW), **kw)


class TestDegradation:
    def test_missing_indoor_temp(self):
        r = rt().run_cycle(cycle_input("2026-06-22T05:00:00+00:00", indoor=None))
        assert r.captured  # no crash; preheat plan flagged no_indoor_temp
        assert r.preheat_plan is not None and "no_indoor_temp" in r.preheat_plan.reason_codes

    def test_stale_indoor_temp(self):
        r = rt().run_cycle(cycle_input("2026-06-22T05:00:00+00:00",
                                       indoor_quality=DataQuality.STALE))
        assert r.captured

    def test_heating_failure_does_not_crash(self):
        r = rt().run_cycle(cycle_input("2026-06-22T05:00:00+00:00", heating_failure=True))
        assert r.captured

    def test_window_open_handled(self):
        r = rt().run_cycle(cycle_input("2026-06-22T05:00:00+00:00", window_open=True))
        assert r.captured

    def test_no_schedule_no_preheat(self):
        r = rt().run_cycle(cycle_input("2026-06-22T05:00:00+00:00", schedule=False))
        assert r.captured and r.preheat_plan is None

    def test_cycle_safe_never_raises(self):
        # a malformed-ish input still cannot raise out of the safe entrypoint
        from custom_components.thermosmart.learning.runtime import RuntimeCycleInput
        bad = RuntimeCycleInput(zone_id="lr", ts="not-a-timestamp")
        r = rt().run_cycle_safe(bad)
        # capture/parse issues are isolated; existing control unaffected
        assert r is not None

    def test_zone_isolated(self):
        r = rt()
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="a"))
        out = r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="b"))
        assert out.captured and r.health().zones == 2

    def test_store_load_failure_isolated(self):
        import asyncio
        r = rt(store=MemoryStore(fail_load=True), clock=lambda: "2026-06-22T05:00:00+00:00")
        assert asyncio.run(r.async_setup())  # load failure does not block setup

    def test_duplicate_decision_no_crash(self):
        r = rt()
        a = r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        b = r.run_cycle(cycle_input("2026-06-22T05:05:00+00:00"))
        assert a.decision_id == b.decision_id  # continuation, not duplicate

    def test_timestamp_anomaly_isolated(self):
        r = rt()
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00"))
        # an earlier timestamp than the previous cycle must not crash
        out = r.run_cycle_safe(cycle_input("2026-06-22T04:00:00+00:00"))
        assert out is not None
