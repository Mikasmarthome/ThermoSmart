"""Phase 19A (extended): real runtime-prediction -> LearningPredictionSet adapter."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.confidence import ConfidencePurpose
from custom_components.thermosmart.learning.decision.runtime_adapter import (
    build_learning_prediction_set)
from tests.helpers_control import strong_boost, strong_preheat


class _Pred:
    def __init__(self, values, *, fallback_used=False, learned=0.7):
        self.values = values
        self.fallback_used = fallback_used
        self.learned_contribution = learned
        self.model_version = 1
        self.parameter_version = 1


class TestPredictionAdapter:
    def test_boost_factor_is_additive_offset(self):
        # BOOST_FACTOR carries an additive °C offset, mapped to boost_offset/celsius_offset
        ps = build_learning_prediction_set(
            "z", {PredictionType.BOOST_FACTOR: _Pred({"boost_factor": 1.5})},
            {ConfidencePurpose.BOOST.value: strong_boost()})
        bo = ps.get("boost_offset")
        assert bo.value == 1.5 and bo.unit == "celsius_offset"

    def test_preheat_minutes_not_time(self):
        ps = build_learning_prediction_set(
            "z", {PredictionType.PREHEAT_MINUTES: _Pred({"preheat_minutes": 42.0})}, {})
        assert ps.get("preheat_start").value == 42.0 and ps.get("preheat_start").unit == "minutes"

    def test_early_cutoff_is_residual_rise(self):
        ps = build_learning_prediction_set(
            "z", {PredictionType.RECOMMENDED_EARLY_CUTOFF:
                  _Pred({"expected_residual_rise": 0.4})}, {})
        ec = ps.get("early_cutoff")
        assert ec.value == 0.4 and ec.unit == "celsius"

    def test_forecast_bias_celsius(self):
        ps = build_learning_prediction_set(
            "z", {PredictionType.FORECAST_BIAS: _Pred({"forecast_bias": -0.3})}, {})
        assert ps.get("forecast_bias").value == -0.3 and ps.get("forecast_bias").unit == "celsius"

    def test_confidence_from_results(self):
        ps = build_learning_prediction_set(
            "z", {PredictionType.PREHEAT_MINUTES: _Pred({"preheat_minutes": 42.0})},
            {ConfidencePurpose.PREHEAT.value: strong_preheat()})
        assert ps.get("preheat_start").confidence > 0.5

    def test_confidence_falls_back_to_learned(self):
        ps = build_learning_prediction_set(
            "z", {PredictionType.HEAT_RATE: _Pred({"heat_rate": 2.0}, learned=0.6)}, {})
        assert ps.get("heat_rate").confidence == 0.6

    def test_missing_prediction_absent(self):
        ps = build_learning_prediction_set("z", {}, {})
        assert ps.get("boost_offset") is None and ps.predictions == {}

    def test_missing_value_key_skipped(self):
        ps = build_learning_prediction_set(
            "z", {PredictionType.BOOST_FACTOR: _Pred({"wrong_key": 1.0})}, {})
        assert ps.get("boost_offset") is None

    def test_fallback_flag_propagated(self):
        ps = build_learning_prediction_set(
            "z", {PredictionType.BOOST_FACTOR: _Pred({"boost_factor": 1.5}, fallback_used=True)},
            {})
        assert ps.get("boost_offset").fallback_used is True

    def test_confidence_results_carried(self):
        ps = build_learning_prediction_set(
            "z", {PredictionType.BOOST_FACTOR: _Pred({"boost_factor": 1.5})},
            {ConfidencePurpose.BOOST.value: strong_boost()})
        assert ConfidencePurpose.BOOST.value in ps.confidence_results
