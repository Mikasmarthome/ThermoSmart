"""Phase 2 tests for the LE 2.0 registry foundations."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import (
    ConfidenceContribution,
    ExportScope,
    Model,
    PredictionType,
)
from custom_components.thermosmart.learning.episode_schemas import (
    AfterheatEpisode,
    EpisodeType,
    HeatingEpisode,
    OutcomeEpisode,
)
from custom_components.thermosmart.learning.raw_schemas import (
    RawTrackName,
    RoomObservation,
    TrvObservation,
)
from custom_components.thermosmart.learning.registry import (
    DuplicateRegistrationError,
    EpisodeDefinition,
    EpisodeRegistry,
    InvalidDefinitionError,
    ModelDefinition,
    ModelRegistry,
    PrivacyClass,
    RawTrackDefinition,
    RawTrackRegistry,
    RetentionPolicy,
    UnknownDefinitionError,
    ValidationCode,
    validate_registry_graph,
)


# -- helpers ------------------------------------------------------------------

def raw_room() -> RawTrackDefinition:
    return RawTrackDefinition(
        track_name=RawTrackName.ROOM,
        schema_type=RoomObservation,
        track_schema_version=1,
        privacy_class=PrivacyClass.BEHAVIORAL,
        allowed_export_scopes=(ExportScope.SUPPORT, ExportScope.RESEARCH),
        retention=RetentionPolicy(max_records=5000, segmented=True),
        segmented=True,
        immutable_events=False,
    )


def raw_trv() -> RawTrackDefinition:
    return RawTrackDefinition(
        track_name=RawTrackName.TRV,
        schema_type=TrvObservation,
        track_schema_version=1,
        privacy_class=PrivacyClass.LOW,
        allowed_export_scopes=(ExportScope.SUPPORT,),
        retention=RetentionPolicy(max_records=5000, segmented=True),
        segmented=True,
        immutable_events=False,
    )


def episode_heating() -> EpisodeDefinition:
    return EpisodeDefinition(
        episode_type=EpisodeType.HEATING,
        schema_type=HeatingEpisode,
        episode_schema_version=1,
        builder_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM, RawTrackName.TRV),
        max_trajectory_points=120,
        max_duration_seconds=5400,
        retention=RetentionPolicy(max_records=3000),
        trajectory_required=False,
        materialized=True,
    )


def episode_outcome() -> EpisodeDefinition:
    return EpisodeDefinition(
        episode_type=EpisodeType.OUTCOME,
        schema_type=OutcomeEpisode,
        episode_schema_version=1,
        builder_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM,),
        max_trajectory_points=120,
        max_duration_seconds=5400,
        retention=RetentionPolicy(max_records=200),
        trajectory_required=True,
        materialized=True,
    )


class StubModel:
    """Minimal object satisfying the Model protocol for tests."""

    model_name = "stub"
    model_version = 1
    parameter_version = 1
    consumed_raw_tracks = ()
    consumed_episode_types = ()
    required_features = ()
    optional_features = ()
    supported_prediction_types = ()
    min_tier = 0

    def cold_start_prior(self): return {}
    def update_eligibility(self, sample): ...
    def update(self, sample): ...
    def predict(self, context): ...
    def confidence(self): return ConfidenceContribution(value=0.0)
    def diagnostics(self): return {}
    def export(self, scope): return {}
    def reset(self): ...
    def rebuild(self, raw, episodes): ...
    def serialize_state(self): return {}
    def deserialize_state(self, state): ...
    def validate_state(self, state): return []
    def migrate_state(self, omv, opv, state): return {}


def model_heat_rate() -> ModelDefinition:
    return ModelDefinition(
        model_name="heat_rate",
        model_factory=StubModel,
        model_version=1,
        parameter_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM, RawTrackName.TRV),
        consumed_episode_types=(EpisodeType.HEATING,),
        supported_prediction_types=(PredictionType.HEAT_RATE, PredictionType.PREHEAT_MINUTES),
        required_features=("indoor_temp", "trv_setpoint", "target"),
        optional_features=("outdoor_temp", "wind_speed"),
        required_capability_flags=(),
        regime_dependent=True,
        advisory=False,
        control_relevant=True,
        rebuildable=True,
        can_be_not_available=False,
        min_trv_only=True,
    )


# -- RetentionPolicy ----------------------------------------------------------

class TestRetentionPolicy:
    def test_valid(self):
        p = RetentionPolicy(max_records=100, max_age_days=30, segmented=True)
        assert p.max_records == 100

    def test_non_positive_records_rejected(self):
        with pytest.raises(InvalidDefinitionError):
            RetentionPolicy(max_records=0)

    def test_non_positive_age_rejected(self):
        with pytest.raises(InvalidDefinitionError):
            RetentionPolicy(max_age_days=-1)


# -- RawTrackRegistry ---------------------------------------------------------

class TestRawTrackRegistry:
    def test_empty_registry(self):
        reg = RawTrackRegistry()
        assert len(reg) == 0
        assert reg.definitions() == ()
        assert not reg.contains(RawTrackName.ROOM)

    def test_register_and_get(self):
        reg = RawTrackRegistry()
        reg.register(raw_room())
        assert reg.contains(RawTrackName.ROOM)
        assert reg.get(RawTrackName.ROOM).schema_type is RoomObservation

    def test_duplicate_rejected(self):
        reg = RawTrackRegistry()
        reg.register(raw_room())
        with pytest.raises(DuplicateRegistrationError):
            reg.register(raw_room())

    def test_unknown_lookup(self):
        reg = RawTrackRegistry()
        with pytest.raises(UnknownDefinitionError):
            reg.get(RawTrackName.TRV)

    def test_deterministic_order(self):
        reg = RawTrackRegistry()
        reg.register(raw_room())
        reg.register(raw_trv())
        assert [d.track_name for d in reg.definitions()] == [RawTrackName.ROOM, RawTrackName.TRV]

    def test_record_validation_ok(self):
        reg = RawTrackRegistry()
        reg.register(raw_room())
        from datetime import datetime, timezone
        rec = RoomObservation(learning_zone_id="lz_1",
                              ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
                              indoor_temp=20.0, target=21.0)
        assert reg.validate_record(RawTrackName.ROOM, rec) == []

    def test_record_validation_wrong_type(self):
        reg = RawTrackRegistry()
        reg.register(raw_room())
        from datetime import datetime, timezone
        wrong = TrvObservation(learning_zone_id="lz_1",
                               ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
                               trv_binding_id="t", trv_setpoint=22.0, target=21.0)
        errors = reg.validate_record(RawTrackName.ROOM, wrong)
        assert errors and "expected RoomObservation" in errors[0]

    def test_wrong_schema_type_rejected_at_definition(self):
        with pytest.raises(InvalidDefinitionError):
            RawTrackDefinition(
                track_name=RawTrackName.ROOM, schema_type=TrvObservation,
                track_schema_version=1, privacy_class=PrivacyClass.LOW,
                allowed_export_scopes=(), retention=RetentionPolicy(),
                segmented=False, immutable_events=False,
            )

    def test_window_cooling_is_not_a_raw_track(self):
        # window_cooling is exclusively an episode type, never a raw track:
        # the enum value must not exist on RawTrackName at all ...
        assert "WINDOW_COOLING" not in RawTrackName.__members__
        assert not any(d.value == "window_cooling" for d in RawTrackName)
        # ... while it remains a valid episode type.
        assert EpisodeType.WINDOW_COOLING.value == "window_cooling"


# -- EpisodeRegistry ----------------------------------------------------------

class TestEpisodeRegistry:
    def test_register_and_get(self):
        reg = EpisodeRegistry()
        reg.register(episode_heating())
        d = reg.get(EpisodeType.HEATING)
        assert d.trajectory_required is False
        assert d.materialized is True

    def test_duplicate_rejected(self):
        reg = EpisodeRegistry()
        reg.register(episode_heating())
        with pytest.raises(DuplicateRegistrationError):
            reg.register(episode_heating())

    def test_trajectory_metadata(self):
        d = episode_outcome()
        assert d.max_trajectory_points == 120 and d.trajectory_required is True

    def test_deterministic_order(self):
        reg = EpisodeRegistry()
        reg.register(episode_heating())
        reg.register(episode_outcome())
        assert [d.episode_type for d in reg.definitions()] == [
            EpisodeType.HEATING, EpisodeType.OUTCOME,
        ]

    def test_unknown_consumed_track_detected(self):
        raw = RawTrackRegistry()
        raw.register(raw_room())  # TRV intentionally missing
        ep = EpisodeRegistry()
        ep.register(episode_heating())  # consumes ROOM + TRV
        missing = ep.unknown_consumed_tracks(raw)
        assert (EpisodeType.HEATING, RawTrackName.TRV) in missing


# -- ModelRegistry ------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        reg.register(model_heat_rate())
        assert reg.get("heat_rate").control_relevant is True

    def test_duplicate_rejected(self):
        reg = ModelRegistry()
        reg.register(model_heat_rate())
        with pytest.raises(DuplicateRegistrationError):
            reg.register(model_heat_rate())

    def test_by_prediction_type(self):
        reg = ModelRegistry()
        reg.register(model_heat_rate())
        assert reg.by_prediction_type(PredictionType.HEAT_RATE)
        assert reg.by_prediction_type(PredictionType.BOOST_FACTOR) == ()

    def test_advisory_and_control_filters(self):
        reg = ModelRegistry()
        reg.register(model_heat_rate())
        assert reg.control_relevant()
        assert reg.advisory() == ()

    def test_factory_protocol_check(self):
        reg = ModelRegistry()
        reg.register(model_heat_rate())
        instance = reg.get("heat_rate").create()
        assert isinstance(instance, Model)

    def test_factory_not_a_model_rejected(self):
        bad = ModelDefinition(
            model_name="bad", model_factory=lambda: object(),
            model_version=1, parameter_version=1,
            consumed_raw_tracks=(), consumed_episode_types=(),
            supported_prediction_types=(), required_features=(), optional_features=(),
            required_capability_flags=(), regime_dependent=False, advisory=True,
            control_relevant=False, rebuildable=False, can_be_not_available=True,
        )
        with pytest.raises(InvalidDefinitionError):
            bad.create()

    def test_concrete_capability_flag_accepted(self):
        d = ModelDefinition(
            model_name="afterheat_valve_aware", model_factory=StubModel,
            model_version=1, parameter_version=1,
            consumed_raw_tracks=(), consumed_episode_types=(),
            supported_prediction_types=(PredictionType.EXPECTED_OVERSHOOT,),
            required_features=(), optional_features=("valve_opening",),
            required_capability_flags=("has_valve_opening",),
            regime_dependent=True, advisory=False, control_relevant=True,
            rebuildable=True, can_be_not_available=False,
        )
        assert "has_valve_opening" in d.required_capability_flags

    def test_tier_as_capability_rejected(self):
        with pytest.raises(InvalidDefinitionError):
            ModelDefinition(
                model_name="bad", model_factory=StubModel, model_version=1,
                parameter_version=1, consumed_raw_tracks=(), consumed_episode_types=(),
                supported_prediction_types=(), required_features=(), optional_features=(),
                required_capability_flags=("tier",), regime_dependent=False,
                advisory=True, control_relevant=False, rebuildable=False,
                can_be_not_available=True,
            )

    def test_unknown_capability_flag_rejected(self):
        with pytest.raises(InvalidDefinitionError):
            ModelDefinition(
                model_name="bad", model_factory=StubModel, model_version=1,
                parameter_version=1, consumed_raw_tracks=(), consumed_episode_types=(),
                supported_prediction_types=(), required_features=(), optional_features=(),
                required_capability_flags=("has_unicorn",), regime_dependent=False,
                advisory=True, control_relevant=False, rebuildable=False,
                can_be_not_available=True,
            )

    def test_trv_only_required_env_feature_rejected(self):
        with pytest.raises(InvalidDefinitionError):
            ModelDefinition(
                model_name="bad_heat_rate", model_factory=StubModel, model_version=1,
                parameter_version=1, consumed_raw_tracks=(), consumed_episode_types=(),
                supported_prediction_types=(PredictionType.HEAT_RATE,),
                required_features=("indoor_temp", "outdoor_temp"), optional_features=(),
                required_capability_flags=(), regime_dependent=True, advisory=False,
                control_relevant=True, rebuildable=True, can_be_not_available=False,
                min_trv_only=True,
            )

    def test_trv_only_required_capability_flag_rejected(self):
        with pytest.raises(InvalidDefinitionError):
            ModelDefinition(
                model_name="bad", model_factory=StubModel, model_version=1,
                parameter_version=1, consumed_raw_tracks=(), consumed_episode_types=(),
                supported_prediction_types=(), required_features=(), optional_features=(),
                required_capability_flags=("has_valve_opening",), regime_dependent=False,
                advisory=False, control_relevant=True, rebuildable=False,
                can_be_not_available=False, min_trv_only=True,
            )

    def test_advisory_control_conflict_rejected(self):
        with pytest.raises(InvalidDefinitionError):
            ModelDefinition(
                model_name="bad", model_factory=StubModel, model_version=1,
                parameter_version=1, consumed_raw_tracks=(), consumed_episode_types=(),
                supported_prediction_types=(), required_features=(), optional_features=(),
                required_capability_flags=(), regime_dependent=False, advisory=True,
                control_relevant=True, rebuildable=False, can_be_not_available=True,
            )


# -- Cross-registry -----------------------------------------------------------

class TestCrossRegistry:
    def _full_valid_graph(self):
        raw = RawTrackRegistry()
        raw.register(raw_room())
        raw.register(raw_trv())
        ep = EpisodeRegistry()
        ep.register(episode_heating())
        ep.register(episode_outcome())
        models = ModelRegistry()
        models.register(model_heat_rate())
        return raw, ep, models

    def test_valid_graph_has_no_issues(self):
        raw, ep, models = self._full_valid_graph()
        assert validate_registry_graph(raw, ep, models) == []

    def test_episode_unknown_track(self):
        raw = RawTrackRegistry()
        raw.register(raw_room())  # TRV missing
        ep = EpisodeRegistry()
        ep.register(episode_heating())
        issues = validate_registry_graph(raw, ep, ModelRegistry())
        assert any(i.code is ValidationCode.EPISODE_UNKNOWN_TRACK for i in issues)

    def test_model_unknown_track(self):
        raw = RawTrackRegistry()
        raw.register(raw_room())  # TRV missing
        ep = EpisodeRegistry()
        ep.register(episode_heating())
        models = ModelRegistry()
        models.register(model_heat_rate())  # consumes TRV
        issues = validate_registry_graph(raw, ep, models)
        assert any(i.code is ValidationCode.MODEL_UNKNOWN_TRACK for i in issues)

    def test_model_unknown_episode(self):
        raw = RawTrackRegistry()
        raw.register(raw_room())
        raw.register(raw_trv())
        ep = EpisodeRegistry()  # HEATING episode missing
        models = ModelRegistry()
        models.register(model_heat_rate())  # consumes HEATING
        issues = validate_registry_graph(raw, ep, models)
        assert any(i.code is ValidationCode.MODEL_UNKNOWN_EPISODE for i in issues)

    def test_two_registries_are_isolated(self):
        a = RawTrackRegistry()
        b = RawTrackRegistry()
        a.register(raw_room())
        assert a.contains(RawTrackName.ROOM)
        assert not b.contains(RawTrackName.ROOM)

    def test_multiple_zones_and_trvs_irrelevant_to_registry(self):
        # The registry describes schemas only; zone/TRV multiplicity is data,
        # not registry state. Registering once is enough for any topology.
        raw, ep, models = self._full_valid_graph()
        assert validate_registry_graph(raw, ep, models) == []
        # no per-zone or per-TRV state exists on the registries
        assert len(raw) == 2 and len(ep) == 2 and len(models) == 1
