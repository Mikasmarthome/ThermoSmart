"""Phase 19A: Manual Correction is advisory and fully traced (no auto target change)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import (
    DecisionMode, LearningPrediction, LearningPredictionSet)
from tests.helpers_decision import rec, run
from tests.helpers_control import strong_purpose
from custom_components.thermosmart.learning.confidence import ComponentKind, ConfidencePurpose


def _manual_preds():
    return LearningPredictionSet("z", "dec1", predictions={
        "manual_preference_bias": LearningPrediction(
            "manual_preference_bias", 0.4, "celsius", 0.8, "manual_preference_bias", False)},
        confidence_results={ConfidencePurpose.MANUAL_CORRECTION.value: strong_purpose(
            ConfidencePurpose.MANUAL_CORRECTION, ComponentKind.MANUAL_CORRECTION)})


class TestManualCorrection:
    def test_manual_bias_does_not_change_setpoint(self):
        t, _ = run(DecisionMode.CONTROL, predictions=_manual_preds())
        assert t.final_setpoint_c == 23.0 and not t.applied_any

    def test_manual_bias_not_dispatched(self):
        sent = []
        run(DecisionMode.CONTROL, predictions=_manual_preds(), sink=sent.append)
        assert sent == []

    def test_no_automatic_target_change(self):
        # the resolver never writes a new target/schedule from a manual bias
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=21.0),
                   predictions=_manual_preds())
        assert t.final_setpoint_c == 21.0
