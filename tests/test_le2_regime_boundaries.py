"""Phase 5 boundary, reliability and numeric-robustness tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement, Regime
from custom_components.thermosmart.learning.episode_schemas import Trajectory, TrajectoryPoint
from custom_components.thermosmart.learning.features import FeatureExtractor
from custom_components.thermosmart.learning.raw_schemas import DriveEndSource, HeatingDriveEndEvent
from custom_components.thermosmart.learning.regime import (
    RegimeInput,
    RegimeParameters,
    RegimeResult,
    ThermalRegimeClassifier,
)

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 20, 0, tzinfo=UTC)
EX = FeatureExtractor()
STEP_MS = 5 * 60 * 1000


def traj(points):
    return Trajectory(points=tuple(points), max_points=120)


def even(values):
    return EX.extract_trajectory_features(
        traj([TrajectoryPoint(i * STEP_MS, float(v)) for i, v in enumerate(values)]))


def clf():
    return ThermalRegimeClassifier()


def drive_end(conf=0.9):
    return HeatingDriveEndEvent(learning_zone_id="lz_1", ts=NOW, event_id="ev",
                               source=DriveEndSource.VALVE_CLOSING, source_confidence=conf)


class TestBoundaries:
    def test_exactly_flat_is_stable(self):
        r = clf().classify(RegimeInput(controller_demand=False, indoor_temp=20.0,
                                       target=20.0, trajectory_features=even([20.0] * 4)))
        assert r.regime is Regime.STABLE

    def test_clear_cooling_threshold(self):
        # uniform -0.04 C/min >> cooling eps 0.01
        r = clf().classify(RegimeInput(controller_demand=False,
                                       trajectory_features=even([20.0, 19.8, 19.6, 19.4])))
        assert r.regime is Regime.PASSIVE_COOLING

    def test_mild_negative_below_cooling_eps_is_not_cooling(self):
        # ~-0.007 C/min: between stable eps and cooling eps -> ambiguous unknown
        r = clf().classify(RegimeInput(
            controller_demand=False,
            trajectory_features=even([20.0, 19.965, 19.93, 19.895])))
        assert r.regime in (Regime.UNKNOWN, Regime.STABLE)

    def test_negative_temperatures(self):
        r = clf().classify(RegimeInput(controller_demand=False, indoor_temp=-5.0,
                                       target=-5.0, trajectory_features=even([-5.0] * 4)))
        assert r.regime is Regime.STABLE

    def test_zero_values(self):
        r = clf().classify(RegimeInput(controller_demand=False, indoor_temp=0.0,
                                       target=0.0, trajectory_features=even([0.0] * 4)))
        assert r.regime is Regime.STABLE


class TestReliability:
    def test_strong_beats_weak_active(self):
        strong = clf().classify(RegimeInput(
            controller_demand=True, valve_opening=Measurement(60.0, DataQuality.OK),
            hvac_action="heating", trajectory_features=even([18.0, 18.2, 18.4, 18.6])))
        weak = clf().classify(RegimeInput(
            indoor_temp=18.0, trv_setpoint=22.0, target=21.0, controller_demand=True,
            trajectory_features=even([18.0, 18.05, 18.1, 18.15])))
        assert strong.reliability > weak.reliability

    def test_direct_drive_end_beats_indirect(self):
        decay = even([20.0, 20.3, 20.45, 20.5])
        direct = clf().classify(RegimeInput(
            controller_demand=False, heating_drive_end=drive_end(),
            seconds_since_drive_end=120.0, trajectory_features=decay))
        indirect = clf().classify(RegimeInput(
            controller_demand=False, trajectory_features=decay))
        assert direct.regime is Regime.AFTERHEAT and indirect.regime is Regime.AFTERHEAT
        assert direct.reliability > indirect.reliability

    def test_stale_points_lower_reliability(self):
        fresh = even([20.0, 19.8, 19.6, 19.4])
        stale_traj = traj([
            TrajectoryPoint(0, 20.0),
            TrajectoryPoint(STEP_MS, 19.8, DataQuality.STALE),
            TrajectoryPoint(2 * STEP_MS, 19.6, DataQuality.STALE),
            TrajectoryPoint(3 * STEP_MS, 19.4),
        ])
        a = clf().classify(RegimeInput(controller_demand=False, trajectory_features=fresh))
        b = clf().classify(RegimeInput(
            controller_demand=False,
            trajectory_features=EX.extract_trajectory_features(stale_traj)))
        assert b.reliability <= a.reliability

    def test_conflicting_evidence_low_reliability(self):
        r = clf().classify(RegimeInput(
            controller_demand=True, trajectory_features=even([20.0, 19.8, 19.6, 19.4])))
        assert r.regime is Regime.UNKNOWN
        assert r.reliability <= 0.4

    def test_reliability_in_range_across_scenarios(self):
        scenarios = [
            RegimeInput(controller_demand=True, valve_opening=Measurement(60.0, DataQuality.OK),
                        trajectory_features=even([18.0, 18.2, 18.4, 18.6])),
            RegimeInput(controller_demand=False, trajectory_features=even([20.0, 19.8, 19.6, 19.4])),
            RegimeInput(heating_failure=True),
            RegimeInput(),
            RegimeInput(controller_demand=False, indoor_temp=20.0, target=20.0,
                        trajectory_features=even([20.0] * 4)),
        ]
        for s in scenarios:
            r = clf().classify(s)
            assert 0.0 <= r.reliability <= 1.0


class TestResultContract:
    def test_disturbed_requires_reason(self):
        with pytest.raises(ValueError):
            RegimeResult(regime=Regime.DISTURBED, reliability=0.5)

    def test_reliability_range_enforced(self):
        with pytest.raises(ValueError):
            RegimeResult(regime=Regime.UNKNOWN, reliability=1.5)

    def test_parameters_versioned(self):
        p = RegimeParameters()
        assert p.parameter_version == 1
        assert p.max_seconds_since_drive_end == 1800.0


class TestDeterminism:
    def test_identical_inputs_identical_output(self):
        inp = RegimeInput(controller_demand=False,
                          trajectory_features=even([20.0, 20.3, 20.45, 20.5]),
                          heating_drive_end=drive_end(), seconds_since_drive_end=120.0)
        a = clf().classify(inp)
        b = clf().classify(inp)
        assert a == b
