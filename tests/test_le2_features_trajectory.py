"""Phase 4 tests for trajectory and comfort feature extraction."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality
from custom_components.thermosmart.learning.episode_schemas import Trajectory, TrajectoryPoint
from custom_components.thermosmart.learning.features import (
    FeatureExtractor,
    FeatureName,
    FeatureStatus,
)

STEP_MIN = 5
STEP_MS = STEP_MIN * 60 * 1000


def ex() -> FeatureExtractor:
    return FeatureExtractor()


def traj(values, *, step_ms=STEP_MS, max_points=120):
    pts = tuple(TrajectoryPoint(i * step_ms, float(v)) for i, v in enumerate(values))
    return Trajectory(points=pts, max_points=max_points)


def traj_points(points, max_points=120):
    return Trajectory(points=tuple(points), max_points=max_points)


# -- trajectory features ------------------------------------------------------

class TestTrajectoryFeatures:
    def test_rising(self):
        fs = ex().extract_trajectory_features(traj([18.0, 18.2, 18.4, 18.6]))
        assert fs.get(FeatureName.TRAJ_DELTA).value == pytest.approx(0.6)
        assert fs.get(FeatureName.TRAJ_LINEAR_SLOPE).value > 0
        assert fs.get(FeatureName.TRAJ_PLAUSIBILITY).is_present  # plausible rate

    def test_falling(self):
        fs = ex().extract_trajectory_features(traj([20.6, 20.4, 20.2, 20.0]))
        assert fs.get(FeatureName.TRAJ_LINEAR_SLOPE).value < 0

    def test_stable(self):
        fs = ex().extract_trajectory_features(traj([20.0, 20.0, 20.0, 20.0]))
        assert fs.get(FeatureName.TRAJ_DELTA).value == 0.0
        assert fs.get(FeatureName.TRAJ_LINEAR_SLOPE).value == 0.0

    def test_afterheat_decay(self):
        # rising but with decreasing slope -> negative second derivative, peak near end
        fs = ex().extract_trajectory_features(traj([20.0, 20.3, 20.45, 20.5]))
        assert fs.get(FeatureName.TRAJ_SECOND_DERIVATIVE).value < 0
        assert fs.get(FeatureName.TRAJ_PEAK).value == pytest.approx(20.5)

    def test_peak_and_time_to_peak(self):
        fs = ex().extract_trajectory_features(traj([18.0, 19.0 * 0 + 18.3, 18.5, 18.4]))
        assert fs.get(FeatureName.TRAJ_PEAK).value == pytest.approx(18.5)
        # peak at index 2 -> 2 * 5min = 600s
        assert fs.get(FeatureName.TRAJ_TIME_TO_PEAK).value == pytest.approx(600.0)

    def test_min_max(self):
        fs = ex().extract_trajectory_features(traj([18.0, 18.5, 18.2, 18.6]))
        assert fs.get(FeatureName.TRAJ_MIN).value == 18.0
        assert fs.get(FeatureName.TRAJ_MAX).value == 18.6

    def test_gaps_lower_reliability(self):
        pts = [
            TrajectoryPoint(0, 18.0),
            TrajectoryPoint(STEP_MS, None, DataQuality.UNAVAILABLE, gap=True),
            TrajectoryPoint(2 * STEP_MS, 18.4),
            TrajectoryPoint(3 * STEP_MS, 18.6),
        ]
        fs = ex().extract_trajectory_features(traj_points(pts))
        assert fs.get(FeatureName.TRAJ_GAP_RATIO).value == pytest.approx(0.25)
        assert fs.get(FeatureName.TRAJ_LINEAR_SLOPE).reliability < 1.0
        assert "gaps" in fs.get(FeatureName.TRAJ_LINEAR_SLOPE).reason_codes

    def test_irregular_sampling_flagged(self):
        pts = [
            TrajectoryPoint(0, 18.0),
            TrajectoryPoint(1000, 18.1),
            TrajectoryPoint(2000, 18.2),
            TrajectoryPoint(600000, 18.3),  # large irregular gap in spacing
        ]
        fs = ex().extract_trajectory_features(traj_points(pts))
        slope = fs.get(FeatureName.TRAJ_ROBUST_SLOPE)
        assert "irregular_sampling" in slope.reason_codes

    def test_implausible_jump(self):
        # 20 -> 24 within one minute -> 4 C/min >> 0.5 limit
        pts = [TrajectoryPoint(0, 20.0), TrajectoryPoint(60000, 24.0),
               TrajectoryPoint(120000, 24.1), TrajectoryPoint(180000, 24.2)]
        fs = ex().extract_trajectory_features(traj_points(pts))
        assert fs.get(FeatureName.TRAJ_PLAUSIBILITY).status is FeatureStatus.INVALID
        assert fs.get(FeatureName.TRAJ_LINEAR_SLOPE).reliability < 0.6

    def test_too_few_points_for_robust(self):
        fs = ex().extract_trajectory_features(traj([18.0, 18.2]))  # 2 points
        assert fs.get(FeatureName.TRAJ_LINEAR_SLOPE).is_present
        assert fs.get(FeatureName.TRAJ_ROBUST_SLOPE).status is FeatureStatus.NOT_COMPUTABLE

    def test_single_point(self):
        fs = ex().extract_trajectory_features(traj([18.0]))
        assert fs.get(FeatureName.TRAJ_START).value == 18.0
        assert fs.get(FeatureName.TRAJ_LINEAR_SLOPE).status is FeatureStatus.NOT_COMPUTABLE

    def test_empty_trajectory(self):
        fs = ex().extract_trajectory_features(Trajectory(points=(), max_points=10))
        assert fs.get(FeatureName.TRAJ_VALID_POINTS).value == 0.0
        assert fs.get(FeatureName.TRAJ_START).status is FeatureStatus.NOT_COMPUTABLE

    def test_full_gap_trajectory(self):
        pts = [TrajectoryPoint(i * STEP_MS, None, DataQuality.UNAVAILABLE, gap=True)
               for i in range(3)]
        fs = ex().extract_trajectory_features(traj_points(pts))
        assert fs.get(FeatureName.TRAJ_VALID_POINTS).value == 0.0
        assert fs.get(FeatureName.TRAJ_GAP_RATIO).value == 1.0
        assert fs.get(FeatureName.TRAJ_DELTA).status is FeatureStatus.NOT_COMPUTABLE

    def test_nan_point_excluded(self):
        pts = [TrajectoryPoint(0, 18.0), TrajectoryPoint(STEP_MS, float("nan")),
               TrajectoryPoint(2 * STEP_MS, 18.4), TrajectoryPoint(3 * STEP_MS, 18.6)]
        fs = ex().extract_trajectory_features(traj_points(pts))
        # NaN excluded -> 3 usable, no NaN leaks into any value
        assert fs.get(FeatureName.TRAJ_VALID_POINTS).value == 3.0
        for name in fs.present():
            assert math.isfinite(fs.get(name).value)

    def test_infinity_point_excluded(self):
        pts = [TrajectoryPoint(0, 18.0), TrajectoryPoint(STEP_MS, float("inf")),
               TrajectoryPoint(2 * STEP_MS, 18.4)]
        fs = ex().extract_trajectory_features(traj_points(pts))
        assert fs.get(FeatureName.TRAJ_VALID_POINTS).value == 2.0


# -- comfort features ---------------------------------------------------------

class TestComfortFeatures:
    def test_target_never_reached(self):
        fs = ex().extract_comfort_features(traj([18.0, 18.2, 18.4, 18.6]), target=21.0)
        assert fs.get(FeatureName.TIME_TO_FIRST_TARGET).status is FeatureStatus.NOT_COMPUTABLE
        assert fs.get(FeatureName.OVERSHOOT).value == 0.0
        assert fs.get(FeatureName.UNDERSHOOT).value > 0

    def test_target_exactly_reached(self):
        fs = ex().extract_comfort_features(traj([19.0, 20.0, 21.0, 21.0]), target=21.0)
        assert fs.get(FeatureName.TIME_TO_FIRST_TARGET).is_present
        assert fs.get(FeatureName.OVERSHOOT).value == 0.0

    def test_overshoot(self):
        fs = ex().extract_comfort_features(traj([20.0, 21.0, 21.8, 21.5]), target=21.0)
        assert fs.get(FeatureName.OVERSHOOT).value == pytest.approx(0.8)

    def test_undershoot(self):
        fs = ex().extract_comfort_features(traj([21.0, 20.4, 21.0, 21.0]), target=21.0)
        assert fs.get(FeatureName.UNDERSHOOT).value == pytest.approx(0.6)

    def test_time_in_band(self):
        # tol 0.5: points within [20.5,21.5]; step 5min between samples
        fs = ex().extract_comfort_features(
            traj([21.0, 21.0, 19.0, 19.0]), target=21.0, comfort_tolerance=0.5)
        # first two intervals in band (each 300s) -> 600s
        assert fs.get(FeatureName.TIME_IN_COMFORT_BAND).value == pytest.approx(600.0)

    def test_oscillation(self):
        fs = ex().extract_comfort_features(traj([20.0, 21.0, 20.0, 21.0]), target=21.0)
        assert fs.get(FeatureName.OSCILLATION).value >= 2.0

    def test_stability_after_target(self):
        fs = ex().extract_comfort_features(traj([20.0, 21.0, 21.1, 21.0]), target=21.0)
        r = fs.get(FeatureName.STABILITY_AFTER_TARGET)
        assert r.is_present and r.value == pytest.approx(0.1, abs=1e-6)

    def test_late_drop(self):
        fs = ex().extract_comfort_features(traj([20.0, 21.5, 21.5, 20.8]), target=21.0)
        assert fs.get(FeatureName.LATE_DROP).value == pytest.approx(0.7)
