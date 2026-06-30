"""Phase 15 ManualCorrectionModel preference-bias / calibration / confidence tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.raw_schemas import CorrectionSource
from custom_components.thermosmart.learning.models import (
    CorrectionIntent,
    ManualCorrectionModel,
    ManualCorrectionParameters,
    ManualCorrectionPredictionContext,
)
from tests.helpers_manual_correction import (
    correction_context,
    correction_event,
    recurring_direct,
)


def m(params=None):
    return ManualCorrectionModel("lz", params)


def ctx(**kw):
    return ManualCorrectionPredictionContext(**kw)


class TestPreferenceBias:
    def test_cold_start_zero(self):
        rec = m().predict_correction(ctx())
        assert rec.preference_bias_c == 0.0 and rec.fallback_used
        assert rec.confidence <= 0.3

    def test_positive_bias(self):
        model = m()
        recurring_direct(model, 6, new=21.3)
        rec = model.predict_correction(ctx())
        assert rec.preference_bias_c > 0.0 and not rec.fallback_used

    def test_negative_bias(self):
        model = m()
        recurring_direct(model, 6, prior=21.0, new=20.7)
        rec = model.predict_correction(ctx())
        assert rec.preference_bias_c < 0.0

    def test_bias_clamped(self):
        model = m()
        recurring_direct(model, 8, prior=21.0, new=25.0)  # +4 each, far over cap
        rec = model.predict_correction(ctx())
        assert rec.preference_bias_c <= ManualCorrectionParameters().max_preference_pos_c + 1e-9

    def test_insufficient_days_no_bias(self):
        model = m()
        # 6 corrections but all on the same day
        for i in range(6):
            ev = correction_event(f"e{i}", 21.0, 21.3, day=0, hour_offset=i)
            model.update(ev, correction_context(ev))
        rec = model.predict_correction(ctx())
        assert rec.preference_bias_c == 0.0 and rec.fallback_used

    def test_predict_protocol(self):
        model = m()
        recurring_direct(model, 6)
        p = model.predict(ctx())
        assert p.prediction_type is PredictionType.MANUAL_PREFERENCE_BIAS
        assert math.isfinite(p.control_value("preference_bias"))

    def test_predict_no_nan_cold(self):
        p = m().predict(ctx())
        assert math.isfinite(p.values["preference_bias"]) and p.values["preference_bias"] == 0.0

    def test_low_confidence_cap(self):
        p = m().predict(ctx())
        assert p.confidence <= p.confidence_cap

    def test_max_change_per_update(self):
        model = m()
        # build a stable +0.3, then a burst of large corrections in one update step
        recurring_direct(model, 6, new=21.3)
        before = model._state.preference_bias_c
        ev = correction_event("big", 21.0, 23.0, day=10)
        model.update(ev, correction_context(ev))
        after = model._state.preference_bias_c
        assert abs(after - before) <= ManualCorrectionParameters().max_change_per_update_c + 1e-9


class TestCalibration:
    def test_calibration_signal_from_recurring_overshoot(self):
        model = m()
        for d in range(6):
            ev = correction_event(f"o{d}", 21.0, 20.5, day=d)
            model.update(ev, correction_context(ev, outcome_overshoot_c=1.5))
        rec = model.controller_calibration_signal(ctx())
        assert rec is not None and rec.intent is CorrectionIntent.OVERSHOOT_CORRECTION
        assert 0.0 <= rec.strength <= 1.0

    def test_no_calibration_without_error_intents(self):
        model = m()
        recurring_direct(model, 6)  # plain preference, no controller-error intent
        assert model.controller_calibration_signal(ctx()) is None


class TestConfidence:
    def test_cold_start_low(self):
        c = m().confidence()
        assert c.value <= 0.3 and "cold_start" in c.reasons

    def test_component_domains(self):
        model = m(); recurring_direct(model, 6)
        c = model.confidence()
        assert c.component == "manual_correction"
        assert "user_correction" in c.evidence_domains

    def test_mostly_automation_low_confidence(self):
        model = m()
        for d in range(8):
            ev = correction_event(f"a{d}", 21.0, 21.5, source=CorrectionSource.AUTOMATION, day=d)
            model.update(ev, correction_context(ev))
        c = model.confidence()
        assert "mostly_automation_or_unknown" in c.reasons

    def test_consistent_direct_higher_confidence(self):
        consistent = m(); recurring_direct(consistent, 8, new=21.3)
        assert consistent.confidence().value > m().confidence().value

    def test_in_range(self):
        model = m(); recurring_direct(model, 6)
        assert 0.0 <= model.confidence().value <= 1.0
