"""Phase 10 AfterheatModel eligibility + drive-end + registry tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import Model, PredictionType, Regime
from custom_components.thermosmart.learning.episode_schemas import (
    AfterheatEpisode,
    EpisodeType,
    HeatingEpisode,
    Trajectory,
    TrajectoryPoint,
)
from custom_components.thermosmart.learning.models import (
    AfterheatModel,
    AfterheatRejection,
    AfterheatUpdateContext,
    afterheat_model_definition,
)
from custom_components.thermosmart.learning.registry import (
    EpisodeDefinition,
    EpisodeRegistry,
    ModelRegistry,
    RawTrackRegistry,
    RetentionPolicy,
    validate_registry_graph,
)
from tests.helpers_afterheat import T0, afterheat_episode


def m():
    return AfterheatModel("lz")


class TestEligibility:
    def test_valid_trv_only(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5))
        assert r.accepted and r.regime is Regime.AFTERHEAT

    def test_is_model(self):
        assert isinstance(m(), Model)

    def test_wrong_episode_type(self):
        traj = Trajectory(points=(TrajectoryPoint(0, 18.0), TrajectoryPoint(60000, 18.5)),
                          max_points=10)
        he = HeatingEpisode(episode_id="h", learning_zone_id="lz", episode_schema_version=1,
                            builder_version=1, classifier_version=1, start_ts=T0,
                            end_ts=T0.replace(minute=10), regime=Regime.ACTIVE_HEATING,
                            reliability=0.8, start_temp=18.0, target=21.0, trajectory=traj)
        assert AfterheatRejection.WRONG_EPISODE_TYPE.value in \
            m().update_eligibility(he).confounder_flags

    def test_missing_drive_end(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, setpoint_before=16.0,
                                                     setpoint_after=16.0))
        assert AfterheatRejection.MISSING_DRIVE_END.value in r.confounder_flags

    def test_disturbed(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, regime=Regime.DISTURBED))
        assert not r.accepted

    def test_reheating(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, confounder_flags=("reheating",)))
        assert AfterheatRejection.REHEATING_CONTAMINATION.value in r.confounder_flags

    def test_window_confounder(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, confounder_flags=("window_open",)))
        assert AfterheatRejection.WINDOW_CONFOUNDER.value in r.confounder_flags

    def test_solar_confounder(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, confounder_flags=("solar_gain",)))
        assert AfterheatRejection.SOLAR_CONFOUNDER.value in r.confounder_flags

    def test_heating_failure(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, confounder_flags=("heating_failure",)))
        assert AfterheatRejection.HEATING_FAILURE.value in r.confounder_flags

    def test_too_short(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, duration_min=1.0))
        assert AfterheatRejection.INSUFFICIENT_DURATION.value in r.confounder_flags

    def test_too_few_points(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, points=2))
        assert AfterheatRejection.INSUFFICIENT_POINTS.value in r.confounder_flags

    def test_negative_rise(self):
        # indirect start (first point beyond tolerance) below the drive-end temp
        traj = Trajectory(points=tuple(
            TrajectoryPoint(180000 + i * 120000, v)
            for i, v in enumerate([20.0, 19.9, 19.8, 19.7])), max_points=120)
        ep = AfterheatEpisode(
            episode_id="neg", learning_zone_id="lz", episode_schema_version=1,
            builder_version=1, classifier_version=1, start_ts=T0,
            end_ts=T0.replace(minute=10), regime=Regime.AFTERHEAT, reliability=0.8,
            indoor_temp_at_close=20.5, target=21.0, trv_setpoint_before=22.0,
            trv_setpoint_after=16.0, trajectory=traj)
        r = m().update_eligibility(ep)
        assert AfterheatRejection.NEGATIVE_RESIDUAL_RISE.value in r.confounder_flags


class TestDriveEndReference:
    def _episode(self, vals, *, close, first_offset_ms=0, step_ms=120000, eid="dre"):
        traj = Trajectory(points=tuple(
            TrajectoryPoint(first_offset_ms + i * step_ms, float(v))
            for i, v in enumerate(vals)), max_points=120)
        return AfterheatEpisode(
            episode_id=eid, learning_zone_id="lz", episode_schema_version=1,
            builder_version=1, classifier_version=1, start_ts=T0,
            end_ts=T0.replace(minute=12), regime=Regime.AFTERHEAT, reliability=0.8,
            indoor_temp_at_close=close, target=21.0, trv_setpoint_before=22.0,
            trv_setpoint_after=16.0, trajectory=traj)

    def test_peak_minus_exact_drive_end(self):
        # first point at offset 0 = drive end (20.0); peak 21.0 -> rise 1.0
        model = m()
        model.update(self._episode([20.0, 20.6, 21.0, 20.95], close=20.0))
        assert model._state.general.rise.value == pytest.approx(1.0, abs=0.05)

    def test_first_valid_point_within_tolerance(self):
        # first point at 60s (<=120s tolerance) is used directly, no indirect flag
        model = m()
        r = model.update(self._episode([20.0, 20.5, 20.7, 20.6], close=20.0,
                                       first_offset_ms=60000))
        assert r.accepted
        assert model._state.general.rise.value == pytest.approx(0.7, abs=0.05)

    def test_end_temperature_not_used_as_start(self):
        # start (drive end) 18.0, peak 19.0, end 18.5 -> rise must be 1.0 (not 0.5)
        model = m()
        model.update(self._episode([18.0, 19.0, 18.7, 18.5], close=18.0))
        assert model._state.general.rise.value == pytest.approx(1.0, abs=0.05)

    def test_zero_residual_rise_valid(self):
        model = m()
        r = model.update(self._episode([20.0, 20.0, 20.0, 20.0], close=20.0))
        assert r.accepted and model._state.near_zero_count == 1

    def test_missing_start_when_trajectory_all_gap(self):
        from custom_components.thermosmart.learning.contracts import DataQuality
        traj = Trajectory(points=tuple(
            TrajectoryPoint(i * 120000, None, DataQuality.UNAVAILABLE, gap=True)
            for i in range(4)), max_points=120)
        ep = AfterheatEpisode(
            episode_id="g", learning_zone_id="lz", episode_schema_version=1,
            builder_version=1, classifier_version=1, start_ts=T0,
            end_ts=T0.replace(minute=12), regime=Regime.AFTERHEAT, reliability=0.8,
            indoor_temp_at_close=20.0, target=21.0, trv_setpoint_before=22.0,
            trv_setpoint_after=16.0, trajectory=traj)
        assert AfterheatRejection.MISSING_INDOOR_TEMP.value in \
            m().update_eligibility(ep).confounder_flags

    def test_indirect_reference_reduces_weight(self):
        direct = m()
        rd = direct.update(self._episode([20.0, 20.6, 21.0, 20.95], close=20.0, eid="d"))
        indirect = m()
        ri = indirect.update(self._episode([20.0, 20.6, 21.0, 20.95], close=20.0,
                                           first_offset_ms=300000, eid="i"))
        assert ri.weight < rd.weight  # indirect start carries less weight

    def test_early_cutoff_uses_same_residual(self):
        model = m()
        for i in range(3):
            model.update(self._episode([20.0, 20.6, 21.0, 20.95], close=20.0,
                                       eid=f"e{i}"))
        from custom_components.thermosmart.learning.models import AfterheatPredictionContext
        rise = model.predict_afterheat(AfterheatPredictionContext()).residual_rise_c
        cutoff = model.predict_early_cutoff(
            AfterheatPredictionContext(current_temp=18.0, target=21.0))
        assert cutoff.control_value("expected_residual_rise") == pytest.approx(rise, abs=1e-6)

    def test_implausible_rise(self):
        r = m().update_eligibility(afterheat_episode("e", 10.0))
        assert not r.accepted

    def test_low_reliability(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, reliability=0.1))
        assert AfterheatRejection.LOW_RELIABILITY.value in r.confounder_flags

    def test_version_mismatch(self):
        r = m().update_eligibility(afterheat_episode("e", 0.5, episode_schema_version=2))
        assert AfterheatRejection.VERSION_MISMATCH.value in r.confounder_flags

    def test_duplicate(self):
        model = m()
        model.update(afterheat_episode("dup", 0.5))
        assert AfterheatRejection.DUPLICATE_EPISODE.value in \
            model.update_eligibility(afterheat_episode("dup", 0.5)).confounder_flags

    def test_wrong_zone(self):
        r = m().update(afterheat_episode("e", 0.5, zone="other"))
        assert AfterheatRejection.WRONG_ZONE.value in r.confounder_flags


class TestDriveEnd:
    def test_inferred_from_setpoint_reduction(self):
        # setpoint reduced -> inferred drive end accepted
        assert m().update_eligibility(afterheat_episode("e", 0.5)).accepted

    def test_context_direct_source_counts_as_direct(self):
        model = m()
        ep = afterheat_episode("e", 0.5)
        ctx = AfterheatUpdateContext(source_episode_id="e", drive_end_source="valve_closing",
                                     drive_end_source_reliability=0.9)
        model.update(ep, ctx)
        assert model._state.direct_drive_end_count == 1

    def test_inferred_without_context_counts_inferred(self):
        model = m()
        model.update(afterheat_episode("e", 0.5))
        assert model._state.inferred_drive_end_count == 1


class TestRegistry:
    def test_definition_valid(self):
        raw = RawTrackRegistry()
        ep = EpisodeRegistry()
        ep.register(EpisodeDefinition(
            episode_type=EpisodeType.AFTERHEAT, schema_type=AfterheatEpisode,
            episode_schema_version=1, builder_version=1, consumed_raw_tracks=(),
            max_trajectory_points=120, max_duration_seconds=5400,
            retention=RetentionPolicy(), trajectory_required=True, materialized=True))
        models = ModelRegistry()
        models.register(afterheat_model_definition())
        assert validate_registry_graph(raw, ep, models) == []

    def test_definition_properties(self):
        d = afterheat_model_definition()
        assert d.min_trv_only and d.control_relevant and not d.advisory and d.rebuildable
        assert EpisodeType.AFTERHEAT in d.consumed_episode_types
        assert PredictionType.EXPECTED_OVERSHOOT in d.supported_prediction_types
        assert PredictionType.RECOMMENDED_EARLY_CUTOFF in d.supported_prediction_types

    def test_factory_produces_model(self):
        assert isinstance(afterheat_model_definition().create(), Model)
