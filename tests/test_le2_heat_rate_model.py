"""Phase 8 HeatRateModel eligibility + registry tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import (
    Model,
    PredictionType,
    Regime,
    RejectionReason,
)
from custom_components.thermosmart.learning.episode_schemas import (
    AfterheatEpisode,
    Trajectory,
    TrajectoryPoint,
)
from custom_components.thermosmart.learning.models import (
    HeatRateModel,
    HeatRateRejection,
    heat_rate_model_definition,
)
from custom_components.thermosmart.learning.registry import (
    EpisodeDefinition,
    EpisodeRegistry,
    ModelRegistry,
    RawTrackRegistry,
    RetentionPolicy,
    validate_registry_graph,
)
from custom_components.thermosmart.learning.episode_schemas import EpisodeType, HeatingEpisode
from tests.helpers_heat_rate import T0, heating_episode


def m():
    return HeatRateModel("lz")


class TestEligibility:
    def test_valid_trv_only_episode_accepted(self):
        r = m().update_eligibility(heating_episode("e1", 2.4))
        assert r.accepted and r.weight > 0
        assert r.regime is Regime.ACTIVE_HEATING

    def test_satisfies_model_protocol(self):
        assert isinstance(m(), Model)

    def test_wrong_episode_type_rejected(self):
        traj = Trajectory(points=(TrajectoryPoint(0, 20.0), TrajectoryPoint(60000, 20.5)),
                          max_points=10)
        ah = AfterheatEpisode(
            episode_id="a", learning_zone_id="lz", episode_schema_version=1,
            builder_version=1, classifier_version=1, start_ts=T0,
            end_ts=T0.replace(minute=30), regime=Regime.AFTERHEAT, reliability=0.8,
            indoor_temp_at_close=20.5, target=21.0, trv_setpoint_before=22.0,
            trv_setpoint_after=16.0, trajectory=traj)
        r = m().update_eligibility(ah)
        assert not r.accepted
        assert HeatRateRejection.WRONG_EPISODE_TYPE.value in r.confounder_flags

    def test_disturbed_rejected(self):
        ep = heating_episode("e", 2.4, regime=Regime.DISTURBED)
        r = m().update_eligibility(ep)
        assert not r.accepted and r.rejection_reason is RejectionReason.DISTURBED_REGIME

    def test_low_reliability_rejected(self):
        r = m().update_eligibility(heating_episode("e", 2.4, reliability=0.1))
        assert HeatRateRejection.LOW_RELIABILITY.value in r.confounder_flags

    def test_too_short_rejected(self):
        r = m().update_eligibility(heating_episode("e", 2.4, duration_min=1.0))
        assert HeatRateRejection.INSUFFICIENT_DURATION.value in r.confounder_flags

    def test_window_confounder_rejected(self):
        r = m().update_eligibility(heating_episode("e", 2.4, confounder_flags=("window_open",)))
        assert HeatRateRejection.WINDOW_CONFOUNDER.value in r.confounder_flags

    def test_heating_failure_rejected(self):
        r = m().update_eligibility(heating_episode("e", 2.4, confounder_flags=("heating_failure",)))
        assert HeatRateRejection.HEATING_FAILURE.value in r.confounder_flags

    def test_too_few_points_rejected(self):
        r = m().update_eligibility(heating_episode("e", 2.4, points=2))
        assert HeatRateRejection.INSUFFICIENT_POINTS.value in r.confounder_flags

    def test_negative_rate_rejected(self):
        # a "heating" episode that actually cools -> not learned
        ep = heating_episode("e", -2.0)
        r = m().update_eligibility(ep)
        assert HeatRateRejection.IMPLAUSIBLE_RATE.value in r.confounder_flags

    def test_implausible_jump_rejected(self):
        ep = heating_episode("e", 30.0, duration_min=5.0, points=3)  # huge slope
        r = m().update_eligibility(ep)
        assert not r.accepted

    def test_version_mismatch_rejected(self):
        r = m().update_eligibility(heating_episode("e", 2.4, episode_schema_version=2))
        assert HeatRateRejection.VERSION_MISMATCH.value in r.confounder_flags

    def test_duplicate_rejected(self):
        model = m()
        model.update(heating_episode("dup", 2.4))
        r = model.update_eligibility(heating_episode("dup", 2.4))
        assert HeatRateRejection.DUPLICATE_EPISODE.value in r.confounder_flags


class TestRegistry:
    def test_definition_valid(self):
        raw = RawTrackRegistry()
        ep = EpisodeRegistry()
        ep.register(EpisodeDefinition(
            episode_type=EpisodeType.HEATING, schema_type=HeatingEpisode,
            episode_schema_version=1, builder_version=1, consumed_raw_tracks=(),
            max_trajectory_points=120, max_duration_seconds=5400,
            retention=RetentionPolicy(), trajectory_required=False, materialized=True))
        models = ModelRegistry()
        models.register(heat_rate_model_definition())
        assert validate_registry_graph(raw, ep, models) == []

    def test_definition_is_trv_only_and_control(self):
        d = heat_rate_model_definition()
        assert d.min_trv_only is True
        assert d.control_relevant is True and d.advisory is False
        assert d.rebuildable is True
        assert EpisodeType.HEATING in d.consumed_episode_types
        assert PredictionType.HEAT_RATE in d.supported_prediction_types
        assert PredictionType.PREHEAT_MINUTES in d.supported_prediction_types

    def test_definition_factory_produces_model(self):
        assert isinstance(heat_rate_model_definition().create(), Model)
