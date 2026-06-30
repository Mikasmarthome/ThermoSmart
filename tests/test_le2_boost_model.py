"""Phase 14 BoostModel eligibility + registry tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import Model, PredictionType
from custom_components.thermosmart.learning.episode_schemas import (
    EpisodeReason,
    EpisodeType,
    HeatingEpisode,
    Trajectory,
    TrajectoryPoint,
    OutcomeEpisode,
)
from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.models import (
    BoostModel,
    BoostRejection,
    boost_model_definition,
)
from custom_components.thermosmart.learning.registry import (
    EpisodeDefinition,
    EpisodeRegistry,
    ModelRegistry,
    RawTrackRegistry,
    RetentionPolicy,
    validate_registry_graph,
)
from tests.helpers_boost import T0, boost_context, boost_episode


def m():
    return BoostModel("lz")


class TestEligibility:
    def test_is_model(self):
        assert isinstance(m(), Model)

    def test_valid_boost(self):
        ep = boost_episode("e", [18.0, 19.5, 21.0, 21.0])
        assert m().update(ep, boost_context(ep)).accepted

    def test_missing_boost_decision_no_context(self):
        ep = boost_episode("e", [18.0, 19.5, 21.0])
        assert BoostRejection.MISSING_BOOST_DECISION.value in \
            m().update_eligibility(ep).confounder_flags

    def test_wrong_episode_type(self):
        traj = Trajectory(points=(TrajectoryPoint(0, 18.0), TrajectoryPoint(60000, 18.5)),
                          max_points=10)
        he = HeatingEpisode(episode_id="h", learning_zone_id="lz", episode_schema_version=1,
                            builder_version=1, classifier_version=1, start_ts=T0,
                            end_ts=T0.replace(minute=10), regime=Regime.ACTIVE_HEATING,
                            reliability=0.8, start_temp=18.0, target=21.0, trajectory=traj)
        assert BoostRejection.WRONG_EPISODE_TYPE.value in \
            m().update_eligibility(he).confounder_flags

    def test_wrong_zone(self):
        ep = boost_episode("e", [18.0, 19.5, 21.0], zone="other")
        r = m().update(ep, boost_context(ep))
        assert BoostRejection.WRONG_ZONE.value in r.confounder_flags

    def test_decision_mismatch(self):
        ep = boost_episode("e", [18.0, 19.5, 21.0], decision_id="dec_e")
        ctx = boost_context(ep)
        bad = type(ctx)(**{**ctx.to_dict(), "decision_id": "other"})
        r = m().update(ep, bad)
        assert BoostRejection.DECISION_MISMATCH.value in r.confounder_flags

    def test_disturbed(self):
        ep = boost_episode("e", [18.0, 19.0, 20.0], reason=EpisodeReason.DISTURBED)
        assert BoostRejection.DISTURBED.value in m().update(ep, boost_context(ep)).confounder_flags

    def test_window_confounder(self):
        ep = boost_episode("e", [18.0, 19.0, 20.0], confounder_flags=("window_open",))
        assert BoostRejection.WINDOW_CONFOUNDER.value in \
            m().update(ep, boost_context(ep)).confounder_flags

    def test_heating_failure(self):
        ep = boost_episode("e", [18.0, 19.0, 20.0], confounder_flags=("heating_failure",))
        assert BoostRejection.HEATING_FAILURE.value in \
            m().update(ep, boost_context(ep)).confounder_flags

    def test_solar_confounder(self):
        ep = boost_episode("e", [18.0, 19.0, 20.0], confounder_flags=("solar_gain",))
        assert BoostRejection.SOLAR_CONFOUNDER.value in \
            m().update(ep, boost_context(ep)).confounder_flags

    def test_manual_correction_contamination(self):
        ep = boost_episode("e", [18.0, 19.0, 20.0], reason=EpisodeReason.MANUAL_CORRECTION)
        assert BoostRejection.MANUAL_CORRECTION_CONTAMINATION.value in \
            m().update(ep, boost_context(ep)).confounder_flags

    def test_no_boost_offset_rejected(self):
        ep = boost_episode("e", [18.0, 19.0, 20.0])
        assert BoostRejection.MISSING_BOOST_DECISION.value in \
            m().update(ep, boost_context(ep, requested_offset_c=0.0)).confounder_flags

    def test_duplicate_decision(self):
        model = m()
        ep = boost_episode("e", [18.0, 19.5, 21.0], decision_id="dec_x")
        model.update(ep, boost_context(ep))
        ep2 = boost_episode("e2", [18.0, 19.5, 21.0], decision_id="dec_x")
        assert BoostRejection.DUPLICATE_DECISION.value in \
            model.update(ep2, boost_context(ep2)).confounder_flags

    def test_version_mismatch(self):
        ep = boost_episode("e", [18.0, 19.5, 21.0], schema_version=2)
        assert BoostRejection.VERSION_MISMATCH.value in \
            m().update(ep, boost_context(ep)).confounder_flags


class TestRegistry:
    def test_definition_valid(self):
        raw = RawTrackRegistry()
        ep = EpisodeRegistry()
        ep.register(EpisodeDefinition(
            episode_type=EpisodeType.OUTCOME, schema_type=OutcomeEpisode,
            episode_schema_version=1, builder_version=1, consumed_raw_tracks=(),
            max_trajectory_points=120, max_duration_seconds=10800,
            retention=RetentionPolicy(), trajectory_required=True, materialized=True))
        models = ModelRegistry()
        models.register(boost_model_definition())
        assert validate_registry_graph(raw, ep, models) == []

    def test_definition_properties(self):
        d = boost_model_definition()
        assert d.min_trv_only and d.control_relevant and not d.advisory and d.rebuildable
        assert EpisodeType.OUTCOME in d.consumed_episode_types
        assert PredictionType.BOOST_FACTOR in d.supported_prediction_types
        assert PredictionType.BOOST_DURATION in d.supported_prediction_types
        assert PredictionType.BOOST_OUTCOME in d.supported_prediction_types

    def test_no_capability_tier(self):
        assert "tier" not in boost_model_definition().required_capability_flags

    def test_factory_produces_model(self):
        assert isinstance(boost_model_definition().create(), Model)
