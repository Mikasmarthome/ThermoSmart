"""Phase 4 TRV-only, no-indoor-temp, degradation and numeric-guard tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.episode_schemas import Trajectory, TrajectoryPoint
from custom_components.thermosmart.learning.features import (
    FeatureExtractor,
    FeatureName,
    FeatureStatus,
)

STEP_MS = 5 * 60 * 1000


def ex() -> FeatureExtractor:
    return FeatureExtractor()


def traj(values):
    return Trajectory(points=tuple(TrajectoryPoint(i * STEP_MS, float(v))
                                   for i, v in enumerate(values)), max_points=120)


# -- TRV-only -----------------------------------------------------------------

class TestTrvOnly:
    def test_core_features_from_trv_only(self):
        # only TRV temp, setpoint, target -> deltas + setpoint excess compute
        fs = ex().extract_temperature_features(
            indoor_temp=20.0, target=21.0, trv_setpoint=22.0)
        assert fs.get(FeatureName.INDOOR_TARGET_DELTA).value == 1.0
        assert fs.get(FeatureName.SETPOINT_EXCESS).value == 2.0
        # no environmental inputs -> indoor/outdoor delta simply not computable
        assert fs.get(FeatureName.INDOOR_OUTDOOR_DELTA).status is FeatureStatus.NOT_COMPUTABLE

    def test_trajectory_features_from_trv_only(self):
        fs = ex().extract_trajectory_features(traj([19.0, 19.2, 19.4, 19.6]))
        assert fs.get(FeatureName.TRAJ_LINEAR_SLOPE).is_present

    def test_no_environmental_no_crash(self):
        # context with nothing configured -> all not_computable, no exception
        fs = ex().extract_context_features()
        assert all(fs.get(n).status is FeatureStatus.NOT_COMPUTABLE
                   for n in (FeatureName.OUTDOOR_TEMP_BUCKET, FeatureName.WIND_BUCKET,
                             FeatureName.SOLAR_BUCKET, FeatureName.HUMIDITY_BUCKET))


# -- no indoor temperature ----------------------------------------------------

class TestNoIndoorTemp:
    def test_no_invented_temperature_features(self):
        fs = ex().extract_temperature_features(indoor_temp=None, target=21.0, trv_setpoint=22.0)
        for name in (FeatureName.INDOOR_TARGET_DELTA, FeatureName.INDOOR_OUTDOOR_DELTA,
                     FeatureName.SETPOINT_EXCESS):
            r = fs.get(name)
            assert r.status is FeatureStatus.NOT_COMPUTABLE
            assert r.value is None
            assert "indoor_temp" in r.missing_evidence

    def test_time_and_heating_features_still_available(self):
        # controller/time-based features do not depend on indoor temperature
        fs = ex().extract_heating_features(heating_duration_s=1200.0)
        assert fs.get(FeatureName.HEATING_DURATION).is_present


# -- degradation per optional field ------------------------------------------

class TestDegradation:
    @pytest.mark.parametrize("measurement,expected", [
        (None, FeatureStatus.NOT_COMPUTABLE),                          # not configured
        (Measurement(None, DataQuality.UNAVAILABLE), FeatureStatus.UNAVAILABLE),
        (Measurement(5.0, DataQuality.STALE), FeatureStatus.STALE),
        (Measurement(None, DataQuality.INVALID), FeatureStatus.UNAVAILABLE),
    ])
    def test_outdoor_delta_degradation(self, measurement, expected):
        fs = ex().extract_temperature_features(
            indoor_temp=20.0, target=21.0, outdoor_temp=measurement)
        assert fs.get(FeatureName.INDOOR_OUTDOOR_DELTA).status is expected

    def test_recovered_value_becomes_present(self):
        fs = ex().extract_temperature_features(
            indoor_temp=20.0, target=21.0, outdoor_temp=Measurement(5.0, DataQuality.OK))
        assert fs.get(FeatureName.INDOOR_OUTDOOR_DELTA).is_present

    def test_missing_optional_does_not_zero_core_reliability(self):
        # absence of outdoor must not drag the indoor-target delta reliability down
        fs = ex().extract_temperature_features(indoor_temp=20.0, target=21.0)
        assert fs.get(FeatureName.INDOOR_TARGET_DELTA).reliability == 1.0


# -- numeric guards -----------------------------------------------------------

class TestNumericGuards:
    def test_zero_values(self):
        fs = ex().extract_temperature_features(indoor_temp=0.0, target=0.0, trv_setpoint=0.0)
        assert fs.get(FeatureName.SETPOINT_EXCESS).value == 0.0

    def test_nan_input_marked_invalid(self):
        fs = ex().extract_temperature_features(indoor_temp=float("nan"), target=21.0)
        assert fs.get(FeatureName.INDOOR_TARGET_DELTA).status is FeatureStatus.NOT_COMPUTABLE

    def test_infinity_input_marked_invalid(self):
        fs = ex().extract_heating_features(heating_duration_s=float("inf"))
        assert fs.get(FeatureName.HEATING_DURATION).status is FeatureStatus.NOT_COMPUTABLE

    def test_zero_duration_trajectory(self):
        fs = ex().extract_trajectory_features(traj([20.0]))
        assert fs.get(FeatureName.TRAJ_DURATION).value == 0.0

    def test_very_large_duration(self):
        big = Trajectory(points=(TrajectoryPoint(0, 18.0),
                                 TrajectoryPoint(10 ** 12, 18.4)), max_points=10)
        fs = ex().extract_trajectory_features(big)
        assert math.isfinite(fs.get(FeatureName.TRAJ_DURATION).value)

    def test_no_nan_or_inf_escapes_any_present_feature(self):
        pts = [TrajectoryPoint(0, 18.0), TrajectoryPoint(STEP_MS, float("nan")),
               TrajectoryPoint(2 * STEP_MS, float("inf")),
               TrajectoryPoint(3 * STEP_MS, 18.6)]
        fs = ex().extract_trajectory_features(
            Trajectory(points=tuple(pts), max_points=10))
        for name in fs.present():
            v = fs.get(name).value
            assert v is not None and math.isfinite(v)


# -- purity (input immutability + determinism) -------------------------------

class TestPurity:
    def test_inputs_not_mutated(self):
        t = traj([18.0, 18.2, 18.4, 18.6])
        before = tuple((p.offset_ms, p.value) for p in t.points)
        ex().extract_trajectory_features(t)
        after = tuple((p.offset_ms, p.value) for p in t.points)
        assert before == after

    def test_deterministic(self):
        t = traj([18.0, 18.3, 18.45, 18.5])
        a = ex().extract_trajectory_features(t)
        b = ex().extract_trajectory_features(t)
        assert a.get(FeatureName.TRAJ_ROBUST_SLOPE).value == \
            b.get(FeatureName.TRAJ_ROBUST_SLOPE).value
