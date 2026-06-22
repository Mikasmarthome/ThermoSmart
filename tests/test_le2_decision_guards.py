"""Phase 19A: Guard Layer (plausibility, anti-chatter, device range)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import GuardLayer, GuardParameters


@pytest.fixture
def g():
    return GuardLayer()


class TestGuardLayer:
    def test_allows_normal(self, g):
        r = g.check_setpoint(21.0, 22.0)
        assert r.allowed and r.final_value == 22.0

    def test_implausible_step_blocked(self, g):
        r = g.check_setpoint(21.0, 29.0)
        assert not r.allowed and r.final_value == 21.0 and "implausible_step" in r.reasons

    def test_anti_chatter(self, g):
        r = g.check_setpoint(21.0, 21.05)
        assert not r.allowed and "anti_chatter" in r.reasons and r.final_value == 21.0

    def test_device_min_clamp(self, g):
        r = g.check_setpoint(6.0, 4.0, device_min=5.0)
        assert r.allowed and r.final_value == 5.0 and r.clamp_applied == 1.0

    def test_device_max_clamp(self, g):
        r = g.check_setpoint(26.0, 28.0, device_max=27.0)
        assert r.allowed and r.final_value == 27.0

    def test_none_value(self, g):
        r = g.check_setpoint(21.0, None)
        assert not r.allowed and r.final_value == 21.0

    def test_custom_params(self):
        g = GuardLayer(GuardParameters(max_abs_step_c=10.0))
        assert g.check_setpoint(21.0, 29.0).allowed

    def test_always_falls_back_to_baseline(self, g):
        # whenever blocked, final_value is exactly the baseline
        for proposed in (29.0, 21.05, None):
            assert g.check_setpoint(21.0, proposed).final_value == 21.0
