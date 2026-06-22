"""Phase 13 ForecastModel eligibility + registry tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import Model, PredictionType
from custom_components.thermosmart.learning.raw_schemas import (
    ForecastDecisionEvent,
    ForecastEvaluationEvent,
    RawTrackName,
)
from custom_components.thermosmart.learning.models import (
    ForecastModel,
    ForecastRejection,
    forecast_model_definition,
)
from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.registry import (
    EpisodeRegistry,
    ModelRegistry,
    PrivacyClass,
    RawTrackDefinition,
    RawTrackRegistry,
    RetentionPolicy,
    validate_registry_graph,
)
from tests.helpers_forecast import decision, evaluation, pair


def _raw_track(track_name, schema_type):
    return RawTrackDefinition(
        track_name=track_name, schema_type=schema_type, track_schema_version=1,
        privacy_class=PrivacyClass.LOW, allowed_export_scopes=(ExportScope.SUPPORT,),
        retention=RetentionPolicy(max_records=5000, segmented=True), segmented=True,
        immutable_events=True)


def m():
    return ForecastModel("lz")


class TestEligibility:
    def test_is_model(self):
        assert isinstance(m(), Model)

    def test_valid_pair(self):
        model = m()
        model.register_decision(decision("d1", 10.0))
        r = model.update_eligibility(evaluation("e1", "d1", 11.0))
        assert r.accepted

    def test_wrong_event_type(self):
        assert ForecastRejection.WRONG_EVENT_TYPE.value in \
            m().update_eligibility(decision("d1", 10.0)).confounder_flags

    def test_missing_decision(self):
        r = m().update_eligibility(evaluation("e1", "unknown", 11.0))
        assert ForecastRejection.MISSING_DECISION.value in r.confounder_flags

    def test_wrong_zone(self):
        model = m()
        r = model.update_eligibility(evaluation("e1", "d1", 11.0, zone="other"))
        assert ForecastRejection.WRONG_ZONE.value in r.confounder_flags

    def test_duplicate_evaluation(self):
        model = m()
        pair(model, 0)
        model.register_decision(decision("d0b", 10.0))
        # same evaluation_id reused
        from tests.helpers_forecast import T0
        dup = ForecastEvaluationEvent(learning_zone_id="lz", ts=T0.replace(hour=12),
                                      evaluation_id="e0", decision_id="d0b",
                                      observed_temp_at_eval=11.0)
        assert ForecastRejection.DUPLICATE_EVALUATION.value in \
            model.update_eligibility(dup).confounder_flags

    def test_invalid_horizon_negative(self):
        model = m()
        model.register_decision(decision("d1", 10.0, created_offset_h=10.0))
        # evaluation before the decision -> negative horizon
        r = model.update_eligibility(evaluation("e1", "d1", 11.0, created_offset_h=0.0,
                                                horizon_h=0.0))
        assert ForecastRejection.INVALID_HORIZON.value in r.confounder_flags

    def test_disturbed_reality_extreme(self):
        model = m()
        model.register_decision(decision("d1", 10.0))
        r = model.update_eligibility(evaluation("e1", "d1", 99.0))  # 89 deg error
        assert ForecastRejection.DISTURBED_REALITY.value in r.confounder_flags


class TestRegistry:
    def test_definition_valid(self):
        raw = RawTrackRegistry()
        raw.register(_raw_track(RawTrackName.FORECAST_DECISION, ForecastDecisionEvent))
        raw.register(_raw_track(RawTrackName.FORECAST_EVALUATION, ForecastEvaluationEvent))
        ep = EpisodeRegistry()
        models = ModelRegistry()
        models.register(forecast_model_definition())
        assert validate_registry_graph(raw, ep, models) == []

    def test_definition_properties(self):
        d = forecast_model_definition()
        assert d.advisory and not d.control_relevant and d.rebuildable
        assert not d.min_trv_only
        assert d.required_capability_flags == ("has_forecast",)
        assert PredictionType.FORECAST_TRUST in d.supported_prediction_types
        assert PredictionType.FORECAST_BIAS in d.supported_prediction_types
        assert PredictionType.CORRECTED_FORECAST in d.supported_prediction_types

    def test_no_capability_tier(self):
        assert "tier" not in forecast_model_definition().required_capability_flags

    def test_factory_produces_model(self):
        assert isinstance(forecast_model_definition().create(), Model)
