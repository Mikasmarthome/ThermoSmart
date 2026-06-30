"""Phase 17 multi-zone behaviour."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)
from tests.helpers_runtime import cycle_input


def rt():
    return LearningRuntime(LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW), clock=lambda: "2025-01-01T00:00:00+00:00")


class TestMultiZone:
    def test_two_zones_independent(self):
        r = rt()
        a = r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="a"))
        b = r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="b"))
        assert a.decision_id != b.decision_id
        assert r.health().zones == 2

    def test_same_setpoint_different_zone_distinct_decisions(self):
        r = rt()
        a = r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="a"))
        b = r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="b"))
        assert a.decision_id != b.decision_id  # zone-scoped ids

    def test_zone_error_isolated(self):
        r = rt()
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="a"))
        out = r.run_cycle_safe(cycle_input("2026-06-22T05:00:00+00:00", zone="b"))
        assert out.captured

    def test_no_cross_zone_state(self):
        r = rt()
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="a", decision_target=21.0))
        r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone="b", decision_target=19.0))
        # changing zone b's decision must not affect zone a's open decision
        a2 = r.run_cycle(cycle_input("2026-06-22T05:05:00+00:00", zone="a", decision_target=21.0))
        a1 = r.run_cycle(cycle_input("2026-06-22T05:10:00+00:00", zone="a", decision_target=21.0))
        assert a1.decision_id == a2.decision_id

    def test_health_counts_zones(self):
        r = rt()
        for z in ("a", "b", "c"):
            r.run_cycle(cycle_input("2026-06-22T05:00:00+00:00", zone=z))
        assert r.health().zones == 3
