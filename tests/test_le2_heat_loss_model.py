"""Phase 9 HeatLossModel eligibility + registry tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import (
    Model,
    PredictionType,
    Regime,
    RejectionReason,
)
from custom_components.thermosmart.learning.episode_schemas import (
    EpisodeType,
    HeatingEpisode,
    PassiveCoolingEpisode,
    Trajectory,
    TrajectoryPoint,
)
from custom_components.thermosmart.learning.models import (
    HeatLossModel,
    HeatLossRejection,
    heat_loss_model_definition,
)
from custom_components.thermosmart.learning.registry import (
    EpisodeDefinition,
    EpisodeRegistry,
    ModelRegistry,
    RawTrackRegistry,
    RetentionPolicy,
    validate_registry_graph,
)
from tests.helpers_heat_loss import T0, cooling_episode


def m():
    return HeatLossModel("lz")


class TestEligibility:
    def test_valid_trv_only_accepted(self):
        r = m().update_eligibility(cooling_episode("e", 1.0))
        assert r.accepted and r.regime is Regime.PASSIVE_COOLING

    def test_is_model(self):
        assert isinstance(m(), Model)

    def test_wrong_episode_type(self):
        traj = Trajectory(points=(TrajectoryPoint(0, 20.0), TrajectoryPoint(60000, 19.5)),
                          max_points=10)
        he = HeatingEpisode(episode_id="h", learning_zone_id="lz", episode_schema_version=1,
                            builder_version=1, classifier_version=1, start_ts=T0,
                            end_ts=T0.replace(minute=30), regime=Regime.ACTIVE_HEATING,
                            reliability=0.8, start_temp=18.0, target=21.0, trajectory=traj)
        r = m().update_eligibility(he)
        assert HeatLossRejection.WRONG_EPISODE_TYPE.value in r.confounder_flags

    def test_disturbed(self):
        r = m().update_eligibility(cooling_episode("e", 1.0, regime=Regime.DISTURBED))
        assert r.rejection_reason is RejectionReason.DISTURBED_REGIME

    def test_window_confounder(self):
        r = m().update_eligibility(cooling_episode("e", 1.0, confounder_flags=("window_open",)))
        assert HeatLossRejection.WINDOW_CONFOUNDER.value in r.confounder_flags

    def test_solar_confounder(self):
        r = m().update_eligibility(cooling_episode("e", 1.0, confounder_flags=("solar_gain",)))
        assert HeatLossRejection.SOLAR_CONFOUNDER.value in r.confounder_flags

    def test_afterheat_contamination(self):
        r = m().update_eligibility(cooling_episode("e", 1.0, confounder_flags=("reheating",)))
        assert HeatLossRejection.AFTERHEAT_CONTAMINATION.value in r.confounder_flags

    def test_no_cooling(self):
        r = m().update_eligibility(cooling_episode("e", 0.0))  # flat
        assert HeatLossRejection.NO_COOLING.value in r.confounder_flags

    def test_rising_temperature(self):
        r = m().update_eligibility(cooling_episode("e", -1.0))  # warming
        assert not r.accepted

    def test_too_short(self):
        r = m().update_eligibility(cooling_episode("e", 1.0, duration_min=2.0))
        assert HeatLossRejection.INSUFFICIENT_DURATION.value in r.confounder_flags

    def test_too_few_points(self):
        r = m().update_eligibility(cooling_episode("e", 1.0, points=2))
        assert HeatLossRejection.INSUFFICIENT_POINTS.value in r.confounder_flags

    def test_low_reliability(self):
        r = m().update_eligibility(cooling_episode("e", 1.0, reliability=0.1))
        assert HeatLossRejection.LOW_RELIABILITY.value in r.confounder_flags

    def test_version_mismatch(self):
        r = m().update_eligibility(cooling_episode("e", 1.0, episode_schema_version=2))
        assert HeatLossRejection.VERSION_MISMATCH.value in r.confounder_flags

    def test_duplicate(self):
        model = m()
        model.update(cooling_episode("dup", 1.0))
        r = model.update_eligibility(cooling_episode("dup", 1.0))
        assert HeatLossRejection.DUPLICATE_EPISODE.value in r.confounder_flags

    def test_wrong_zone(self):
        r = m().update(cooling_episode("e", 1.0, zone="other"))
        assert HeatLossRejection.WRONG_ZONE.value in r.confounder_flags


class TestRegistry:
    def test_definition_valid(self):
        raw = RawTrackRegistry()
        ep = EpisodeRegistry()
        ep.register(EpisodeDefinition(
            episode_type=EpisodeType.PASSIVE_COOLING, schema_type=PassiveCoolingEpisode,
            episode_schema_version=1, builder_version=1, consumed_raw_tracks=(),
            max_trajectory_points=240, max_duration_seconds=5400,
            retention=RetentionPolicy(), trajectory_required=True, materialized=True))
        models = ModelRegistry()
        models.register(heat_loss_model_definition())
        assert validate_registry_graph(raw, ep, models) == []

    def test_definition_properties(self):
        d = heat_loss_model_definition()
        assert d.min_trv_only and d.control_relevant and not d.advisory and d.rebuildable
        assert EpisodeType.PASSIVE_COOLING in d.consumed_episode_types
        assert PredictionType.HEAT_LOSS_RATE in d.supported_prediction_types
        assert PredictionType.COOLING_MINUTES in d.supported_prediction_types

    def test_factory_produces_model(self):
        assert isinstance(heat_loss_model_definition().create(), Model)
