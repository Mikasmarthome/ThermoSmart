"""How the learning models really come into existence and what they publish.

The orchestrator builds the fixed set of zone models itself; there is no
declarative model registry deciding anything. This pins that set, the model
contract each of them satisfies and the predictions the orchestrator collects —
including the afterheat predictions that are deliberately not published.
"""
from __future__ import annotations

from custom_components.thermosmart.learning import registry
from custom_components.thermosmart.learning.contracts import Model, PredictionType
from custom_components.thermosmart.learning.runtime.orchestration import (
    ModelOrchestrator,
    build_zone_models,
)

_MODELS = {"heat_rate", "onset_delay", "heat_loss", "afterheat", "outcome", "forecast", "boost"}


class TestZoneModels:
    def test_the_orchestrator_constructs_exactly_these_models(self):
        assert set(build_zone_models("lz_1")) == _MODELS

    def test_every_model_satisfies_the_model_contract(self):
        for name, model in build_zone_models("lz_1").items():
            assert isinstance(model, Model), name

    def test_every_model_belongs_to_its_zone_and_is_versioned(self):
        for name, model in build_zone_models("lz_1").items():
            assert model.model_name, name
            assert model.model_version > 0 and model.parameter_version > 0, name


class TestPublishedPredictions:
    def test_the_collected_predictions_are_exactly_these(self):
        published = ModelOrchestrator("lz_1").collect_predictions()
        assert set(published) == {
            PredictionType.HEAT_RATE, PredictionType.ONSET_DELAY, PredictionType.HEAT_LOSS_RATE,
            PredictionType.EXPECTED_OVERSHOOT, PredictionType.OUTCOME_QUALITY,
            PredictionType.FORECAST_TRUST, PredictionType.BOOST_FACTOR,
        }

    def test_retired_afterheat_predictions_have_no_producer(self):
        published = ModelOrchestrator("lz_1").collect_predictions()
        assert not hasattr(PredictionType, "AFTERHEAT_TIME_TO_PEAK")
        assert not hasattr(PredictionType, "RECOMMENDED_EARLY_CUTOFF")
        # the afterheat duration is read by the boost observation window but is not
        # published yet (see the architecture notes of the simplification phases)
        assert PredictionType.AFTERHEAT_DURATION not in published


class TestNoDeclarativeModelRegistry:
    def test_the_unused_registry_layer_is_gone(self):
        for name in ("ModelRegistry", "ModelDefinition", "validate_registry_graph"):
            assert not hasattr(registry, name), name
