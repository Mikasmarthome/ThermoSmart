"""Phase 19A: characterization parity — the new pipeline reproduces baseline behaviour.

The baseline command produced for a recommendation must equal the recommendation's
own setpoint (after device quantization) whenever no LE2 value is applied.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import preds, rec, run

SCENARIOS = [
    ("normal_heating", rec(target=21.0, setpoint=21.0)),
    ("schedule_change", rec(target=20.0, setpoint=22.0)),
    ("target_reached", rec(target=21.0, setpoint=21.0, indoor=21.0)),
    ("overshoot", rec(target=21.0, setpoint=21.0, indoor=22.5)),
    ("window_open", rec(target=21.0, setpoint=21.0, window_open=True)),
    ("summer", rec(target=21.0, setpoint=5.0, is_summer=True)),
    ("frost", rec(target=21.0, setpoint=5.0, frost_protection=True)),
    ("absence", rec(target=18.0, setpoint=18.0, absence=True)),
    ("boost", rec(target=21.0, setpoint=24.0)),
]


class TestCharacterizationParity:
    @pytest.mark.parametrize("name,recommendation", SCENARIOS)
    def test_shadow_parity_no_change(self, name, recommendation):
        t, d = run(DecisionMode.SHADOW, recommendation=recommendation,
                   predictions=preds(boost=None))
        assert t.final_setpoint_c == recommendation["trv_setpoint"]
        assert not t.applied_any

    @pytest.mark.parametrize("name,recommendation", SCENARIOS)
    def test_control_locked_scenarios_keep_baseline(self, name, recommendation):
        # safety/lock scenarios must keep the baseline even in CONTROL with predictions
        t, _ = run(DecisionMode.CONTROL, recommendation=recommendation,
                   predictions=preds(boost=1.0))
        if name in ("window_open", "summer", "frost", "absence", "target_reached",
                    "normal_heating"):
            assert t.final_setpoint_c == recommendation["trv_setpoint"]

    def test_baseline_command_quantized(self):
        # parity holds at device step resolution
        t, _ = run(DecisionMode.SHADOW, recommendation=rec(target=21.0, setpoint=21.3),
                   predictions=preds(boost=None))
        assert t.final_setpoint_c == 21.5
