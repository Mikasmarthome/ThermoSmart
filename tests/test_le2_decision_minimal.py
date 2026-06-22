"""Phase 19A: TRV-only minimal setup + failure degradation."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import (
    DecisionMode, LearningPredictionSet)
from tests.helpers_decision import preds, rec, run


class TestMinimalSetup:
    def test_trv_only_no_optional_data(self):
        # no indoor sensor, no outdoor, no forecast, no window -> still produces a decision
        t, d = run(DecisionMode.CONTROL,
                   recommendation={"effective_target": 21.0, "trv_setpoint": 21.0},
                   predictions=preds(boost=None))
        assert t.final_setpoint_c == 21.0 and d.command is not None

    def test_missing_predictions_degrade_to_baseline(self):
        t, _ = run(DecisionMode.CONTROL,
                   predictions=LearningPredictionSet("z", "dec1"))  # empty
        assert t.final_setpoint_c == 23.0 and not t.applied_any

    def test_missing_confidence_falls_back(self):
        t, _ = run(DecisionMode.CONTROL, predictions=preds(boost=1.5, with_conf=False))
        assert t.final_setpoint_c == 23.0 and not t.applied_any

    def test_no_target_no_boost_application(self):
        t, _ = run(DecisionMode.CONTROL,
                   recommendation={"trv_setpoint": 23.0, "indoor_temperature": 19.0},
                   predictions=preds(boost=1.5))
        assert not t.applied_any  # no target -> cannot compute offset safely

    def test_invalid_indoor_reduces_not_breaks(self):
        t, d = run(DecisionMode.SHADOW,
                   recommendation={"effective_target": 21.0, "trv_setpoint": 21.0},
                   predictions=preds(boost=None))
        assert d.command is not None  # base function intact without indoor temp
