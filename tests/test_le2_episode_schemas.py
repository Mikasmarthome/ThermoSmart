"""Phase 1 tests for LE 2.0 episode schemas."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Regime
from custom_components.thermosmart.learning.episode_schemas import (
    AfterheatEpisode,
    ControllerKind,
    EpisodeReason,
    EpisodeType,
    HeatingEpisode,
    OutcomeEpisode,
    PassiveCoolingEpisode,
    Trajectory,
    TrajectoryPoint,
    WindowCoolingEpisode,
)

UTC = timezone.utc
START = datetime(2026, 1, 15, 20, 0, 0, tzinfo=UTC)
END = datetime(2026, 1, 15, 20, 30, 0, tzinfo=UTC)


def _traj(points=None, max_points=120) -> Trajectory:
    if points is None:
        points = (
            TrajectoryPoint(0, 20.0),
            TrajectoryPoint(60000, 20.4),
            TrajectoryPoint(120000, 20.6),
        )
    return Trajectory(points=points, max_points=max_points)


# -- Trajectory ---------------------------------------------------------------

class TestTrajectory:
    def test_valid_monotonic(self):
        t = _traj()
        assert len(t.points) == 3
        assert t.has_gaps is False

    def test_non_increasing_offsets_rejected(self):
        with pytest.raises(ValueError):
            Trajectory(points=(TrajectoryPoint(0, 20.0), TrajectoryPoint(0, 20.1)),
                       max_points=10)

    def test_decreasing_offsets_rejected(self):
        with pytest.raises(ValueError):
            Trajectory(points=(TrajectoryPoint(60000, 20.0), TrajectoryPoint(0, 20.1)),
                       max_points=10)

    def test_exceeding_cap_rejected(self):
        pts = tuple(TrajectoryPoint(i * 1000, float(i)) for i in range(5))
        with pytest.raises(ValueError):
            Trajectory(points=pts, max_points=3)

    def test_is_capped(self):
        pts = tuple(TrajectoryPoint(i * 1000, float(i)) for i in range(3))
        assert Trajectory(points=pts, max_points=3).is_capped is True

    def test_gap_point_marked(self):
        t = Trajectory(
            points=(TrajectoryPoint(0, 20.0),
                    TrajectoryPoint(60000, None, DataQuality.UNAVAILABLE, gap=True),
                    TrajectoryPoint(120000, 20.5)),
            max_points=10,
        )
        assert t.has_gaps is True

    def test_gap_point_with_value_rejected(self):
        with pytest.raises(ValueError):
            TrajectoryPoint(0, 20.0, gap=True)

    def test_non_gap_ok_point_requires_value(self):
        with pytest.raises(ValueError):
            TrajectoryPoint(0, None, DataQuality.OK)

    def test_negative_offset_rejected(self):
        with pytest.raises(ValueError):
            TrajectoryPoint(-1, 20.0)


# -- Episodes -----------------------------------------------------------------

def _common(**kw):
    base = dict(
        episode_id="ep_1", learning_zone_id="lz_1",
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=START, end_ts=END, reliability=0.7,
    )
    base.update(kw)
    return base


class TestHeatingEpisode:
    def test_minimal_valid_without_weather_or_valve(self):
        ep = HeatingEpisode(**_common(regime=Regime.ACTIVE_HEATING, start_temp=18.0, target=21.0))
        assert ep.trajectory is None and ep.trv_binding_id is None

    def test_invalid_reliability(self):
        with pytest.raises(ValueError):
            HeatingEpisode(**_common(regime=Regime.ACTIVE_HEATING, start_temp=18.0,
                                     target=21.0, reliability=1.5))

    def test_end_before_start_rejected(self):
        with pytest.raises(ValueError):
            HeatingEpisode(**_common(regime=Regime.ACTIVE_HEATING, start_temp=18.0,
                                     target=21.0, start_ts=END, end_ts=START))

    def test_missing_identity_rejected(self):
        with pytest.raises(ValueError):
            HeatingEpisode(**_common(episode_id="", regime=Regime.ACTIVE_HEATING,
                                     start_temp=18.0, target=21.0))

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError):
            HeatingEpisode(**_common(regime=Regime.ACTIVE_HEATING, start_temp=18.0,
                                     target=21.0, start_ts=datetime(2026, 1, 15, 20, 0, 0)))


class TestAfterheatEpisode:
    def test_works_without_valve_opening(self):
        ep = AfterheatEpisode(**_common(
            regime=Regime.AFTERHEAT, indoor_temp_at_close=20.6, target=21.0,
            trv_setpoint_before=22.0, trv_setpoint_after=16.0, trajectory=_traj(),
        ))
        assert ep.valve_before is None and ep.valve_after is None

    def test_requires_trajectory(self):
        with pytest.raises(TypeError):
            AfterheatEpisode(**_common(
                regime=Regime.AFTERHEAT, indoor_temp_at_close=20.6, target=21.0,
                trv_setpoint_before=22.0, trv_setpoint_after=16.0,
            ))


class TestPassiveCoolingEpisode:
    def test_valid(self):
        ep = PassiveCoolingEpisode(**_common(
            regime=Regime.PASSIVE_COOLING, start_temp=21.0, end_temp=20.0, trajectory=_traj(),
        ))
        assert ep.start_temp == 21.0


class TestWindowCoolingEpisode:
    def test_valid_without_trajectory(self):
        ep = WindowCoolingEpisode(**_common(
            regime=Regime.DISTURBED, temp_at_open=21.0, temp_at_close=19.4,
        ))
        assert ep.trajectory is None


class TestOutcomeEpisode:
    def test_valid_with_trajectory_no_stored_score_or_peak(self):
        ep = OutcomeEpisode(**_common(
            regime=Regime.ACTIVE_HEATING, start_temp=18.0, end_temp=21.0, target=21.0,
            comfort_tolerance_at_start=0.5, reason=EpisodeReason.REACHED,
            controller=ControllerKind.THERMOSMART, trajectory=_traj(),
        ))
        # outcome score and peak temp are derived, not stored
        assert not hasattr(ep, "outcome_score")
        assert not hasattr(ep, "peak_temp")
        assert ep.legacy_peak_temp is None

    def test_requires_trajectory(self):
        with pytest.raises(TypeError):
            OutcomeEpisode(**_common(
                regime=Regime.ACTIVE_HEATING, start_temp=18.0, end_temp=21.0, target=21.0,
                comfort_tolerance_at_start=0.5, reason=EpisodeReason.REACHED,
                controller=ControllerKind.THERMOSMART,
            ))

    def test_stores_tolerance_not_band(self):
        ep = OutcomeEpisode(**_common(
            regime=Regime.ACTIVE_HEATING, start_temp=18.0, end_temp=21.0, target=21.0,
            comfort_tolerance_at_start=0.5, reason=EpisodeReason.REACHED,
            controller=ControllerKind.THERMOSMART, trajectory=_traj(),
        ))
        assert ep.comfort_tolerance_at_start == 0.5
        assert not hasattr(ep, "comfort_band")


# -- Topology (no "one TRV == one zone" assumption) ---------------------------

class TestTopology:
    def test_multiple_zones_with_independent_episodes(self):
        a = HeatingEpisode(**_common(episode_id="ep_a", learning_zone_id="lz_1",
                                     regime=Regime.ACTIVE_HEATING, start_temp=18.0, target=21.0))
        b = HeatingEpisode(**_common(episode_id="ep_b", learning_zone_id="lz_2",
                                     regime=Regime.ACTIVE_HEATING, start_temp=17.0, target=20.0))
        assert a.learning_zone_id != b.learning_zone_id

    def test_episode_is_zone_scoped_not_trv_scoped(self):
        # An episode binds to a zone; a TRV binding is optional metadata.
        ep = AfterheatEpisode(**_common(
            regime=Regime.AFTERHEAT, indoor_temp_at_close=20.6, target=21.0,
            trv_setpoint_before=22.0, trv_setpoint_after=16.0, trajectory=_traj(),
            trv_binding_id="trv_a",
        ))
        assert ep.learning_zone_id == "lz_1"
        assert ep.trv_binding_id == "trv_a"


def test_episode_types_stable():
    assert EpisodeType.AFTERHEAT.value == "afterheat"
    assert EpisodeType.OUTCOME.value == "outcome"
