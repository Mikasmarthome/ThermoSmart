"""Phase 13 ForecastModel decision/evaluation matching + lifecycle tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.models import (
    ForecastModel,
    ForecastParameters,
    ForecastRejection,
    ForecastUpdateContext,
)
from tests.helpers_forecast import decision, evaluation, pair


def m(params=None):
    return ForecastModel("lz", params)


class TestMatching:
    def test_decision_and_matching_evaluation(self):
        model = m()
        r = pair(model, 0, forecast=10.0, observed=11.0)
        assert r.accepted and model._state.general.sample_count == 1

    def test_missing_decision_rejected(self):
        r = m().update(evaluation("e1", "ghost", 11.0))
        assert ForecastRejection.MISSING_DECISION.value in r.confounder_flags

    def test_decision_via_context(self):
        model = m()
        ctx = ForecastUpdateContext(source_decision_id="d1", forecast_temperature_c=10.0,
                                    horizon_s=6 * 3600, actual_temperature_c=11.0)
        r = model.update(evaluation("e1", "d1", 11.0), ctx)
        assert r.accepted

    def test_context_decision_mismatch(self):
        model = m()
        ctx = ForecastUpdateContext(source_decision_id="other", forecast_temperature_c=10.0,
                                    horizon_s=6 * 3600)
        r = model.update(evaluation("e1", "d1", 11.0), ctx)
        assert ForecastRejection.CONTEXT_MISMATCH.value in r.confounder_flags

    def test_context_zone_mismatch(self):
        model = m()
        ctx = ForecastUpdateContext(source_decision_id="d1", learning_zone_id="zzz",
                                    forecast_temperature_c=10.0, horizon_s=6 * 3600)
        r = model.update(evaluation("e1", "d1", 11.0), ctx)
        assert ForecastRejection.CONTEXT_MISMATCH.value in r.confounder_flags

    def test_duplicate_evaluation_not_double_learned(self):
        model = m()
        pair(model, 0)
        model.register_decision(decision("d0b", 10.0, created_offset_h=1))
        from custom_components.thermosmart.learning.raw_schemas import ForecastEvaluationEvent
        from tests.helpers_forecast import T0
        dup = ForecastEvaluationEvent(learning_zone_id="lz", ts=T0.replace(hour=12),
                                      evaluation_id="e0", decision_id="d0b",
                                      observed_temp_at_eval=11.0)
        r = model.update(dup)
        assert not r.accepted and model._state.general.sample_count == 1

    def test_expired_decision_dropped(self):
        model = m(ForecastParameters(max_open_decisions=3))
        for i in range(6):
            model.register_decision(decision(f"d{i}", 10.0, created_offset_h=i))
        assert len(model._state.open_decisions) == 3
        assert model._state.expired_decisions == 3

    def test_open_decision_consumed_on_eval(self):
        model = m()
        model.register_decision(decision("d1", 10.0))
        assert "d1" in model._state.open_decisions
        model.update(evaluation("e1", "d1", 11.0))
        assert "d1" not in model._state.open_decisions

    def test_foreign_zone_decision_ignored(self):
        model = m()
        model.register_decision(decision("d1", 10.0, zone="other"))
        assert "d1" not in model._state.open_decisions

    def test_deterministic_matching(self):
        a = m(); b = m()
        for model in (a, b):
            for i in range(4):
                pair(model, i, forecast=10.0, observed=10.0 + i * 0.2)
        assert a.serialize_state() == b.serialize_state()
