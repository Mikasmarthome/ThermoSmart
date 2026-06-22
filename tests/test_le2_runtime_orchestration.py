"""Phase 17 model orchestration: predictions, single confidence aggregation, isolation."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import ConfidencePurpose
from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.runtime import (
    ModelOrchestrator,
    build_zone_models,
)


class _BadModel:
    model_name = "bad"

    def predict(self, ctx):
        raise RuntimeError("boom")

    def confidence(self):
        raise RuntimeError("boom")

    def update(self, episode, context=None):
        raise RuntimeError("boom")


class TestOrchestration:
    def test_build_zone_models(self):
        models = build_zone_models("lr")
        assert set(models) == {"heat_rate", "heat_loss", "afterheat", "outcome", "forecast",
                               "boost", "manual_correction"}

    def test_collect_predictions(self):
        o = ModelOrchestrator("lr")
        preds = o.collect_predictions()
        assert PredictionType.HEAT_RATE in preds and PredictionType.FORECAST_TRUST in preds

    def test_run_produces_confidence(self):
        o = ModelOrchestrator("lr")
        res = o.run()
        assert "heat_rate" in res.confidence_results
        assert res.confidence_results["heat_rate"].purpose is ConfidencePurpose.HEAT_RATE

    def test_single_aggregation_per_purpose(self):
        o = ModelOrchestrator("lr")
        res = o.run()
        # one ConfidenceResult per requested purpose
        assert len(res.confidence_results) >= 5

    def test_model_error_isolated(self):
        models = build_zone_models("lr")
        models["bad"] = _BadModel()
        o = ModelOrchestrator("lr", models=models)
        res = o.run({ConfidencePurpose.HEAT_RATE: ("heat_rate",)})
        assert any("bad" in e for e in res.model_errors)
        assert "heat_rate" in res.confidence_results  # others survive

    def test_apply_update_isolated(self):
        models = build_zone_models("lr")
        models["bad"] = _BadModel()
        o = ModelOrchestrator("lr", models=models)
        assert o.apply_update("bad", object()) is False  # no raise

    def test_serialize_restore_models(self):
        o = ModelOrchestrator("lr")
        data = o.serialize_models()
        o2 = ModelOrchestrator("lr")
        errors = o2.restore_models(data)
        assert errors == ()

    def test_restore_corrupt_isolated(self):
        o = ModelOrchestrator("lr")
        errors = o.restore_models({"heat_rate": {"garbage": 1}})
        assert any("heat_rate" in e for e in errors)

    def test_zone_binding_in_models(self):
        o = ModelOrchestrator("zone_x")
        assert all(getattr(m, "_zone", "zone_x") == "zone_x" for m in o.models.values())
