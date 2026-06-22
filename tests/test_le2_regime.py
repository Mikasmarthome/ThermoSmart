"""Phase 5 core tests for the ThermalRegimeClassifier."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.contracts import (
    DataQuality, Measurement, Regime,
)
from custom_components.thermosmart.learning.episode_schemas import Trajectory, TrajectoryPoint
from custom_components.thermosmart.learning.features import FeatureExtractor
from custom_components.thermosmart.learning.raw_schemas import DriveEndSource, HeatingDriveEndEvent
from custom_components.thermosmart.learning.regime import (
    CLASSIFIER_VERSION,
    DisturbanceCode,
    ReasonCode,
    RegimeInput,
    ThermalRegimeClassifier,
)

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 20, 0, tzinfo=UTC)
EX = FeatureExtractor()
STEP_MS = 5 * 60 * 1000


def traj(values):
    return Trajectory(points=tuple(TrajectoryPoint(i * STEP_MS, float(v))
                                   for i, v in enumerate(values)), max_points=120)


def tf(values):
    return EX.extract_trajectory_features(traj(values))


def clf() -> ThermalRegimeClassifier:
    return ThermalRegimeClassifier()


RISING = [18.0, 18.2, 18.4, 18.6]
FALLING = [20.0, 19.8, 19.6, 19.4]
FLAT = [20.0, 20.0, 20.0, 20.0]
DECAY = [20.0, 20.3, 20.45, 20.5]  # rising, decreasing slope


def drive_end(conf=0.9):
    return HeatingDriveEndEvent(learning_zone_id="lz_1", ts=NOW, event_id="ev",
                               source=DriveEndSource.VALVE_CLOSING, source_confidence=conf)


# -- active heating -----------------------------------------------------------

class TestActiveHeating:
    def test_strong_direct_evidence(self):
        r = clf().classify(RegimeInput(
            controller_demand=True, valve_opening=Measurement(60.0, DataQuality.OK),
            trajectory_features=tf(RISING)))
        assert r.regime is Regime.ACTIVE_HEATING
        assert r.reliability > 0.6

    def test_trv_only_demand_and_setpoint(self):
        r = clf().classify(RegimeInput(
            indoor_temp=18.0, trv_setpoint=22.0, target=21.0,
            controller_demand=True, trajectory_features=tf(RISING)))
        assert r.regime is Regime.ACTIVE_HEATING

    def test_setpoint_over_only_is_not_active(self):
        r = clf().classify(RegimeInput(
            indoor_temp=18.0, trv_setpoint=22.0, target=21.0,
            trajectory_features=tf(FLAT)))
        assert r.regime is not Regime.ACTIVE_HEATING

    def test_demand_but_falling_is_conflicting(self):
        r = clf().classify(RegimeInput(
            controller_demand=True, trajectory_features=tf(FALLING)))
        assert r.regime is Regime.UNKNOWN
        assert ReasonCode.CONFLICTING_EVIDENCE.value in r.reason_codes

    def test_valve_open_without_rise_is_active(self):
        r = clf().classify(RegimeInput(
            valve_opening=Measurement(50.0, DataQuality.OK),
            trajectory_features=tf(FLAT)))
        assert r.regime is Regime.ACTIVE_HEATING
        assert ReasonCode.NO_POSITIVE_TREND.value in r.reason_codes

    def test_hvac_heating(self):
        r = clf().classify(RegimeInput(hvac_action="heating", trajectory_features=tf(RISING)))
        assert r.regime is Regime.ACTIVE_HEATING


# -- afterheat ----------------------------------------------------------------

class TestAfterheat:
    def test_drive_end_with_decaying_rise(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, heating_drive_end=drive_end(),
            seconds_since_drive_end=300.0, trajectory_features=tf(DECAY)))
        assert r.regime is Regime.AFTERHEAT
        assert ReasonCode.RECENT_DRIVE_END.value in r.reason_codes

    def test_indirect_drive_end(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, trajectory_features=tf(DECAY)))
        assert r.regime is Regime.AFTERHEAT
        assert ReasonCode.INDIRECT_DRIVE_END.value in r.reason_codes

    def test_without_valve_still_afterheat_lower_reliability(self):
        with_valve = clf().classify(RegimeInput(
            controller_demand=False, valve_opening=Measurement(0.0, DataQuality.OK),
            heating_drive_end=drive_end(), seconds_since_drive_end=300.0,
            trajectory_features=tf(DECAY)))
        without_valve = clf().classify(RegimeInput(
            controller_demand=False, heating_drive_end=drive_end(),
            seconds_since_drive_end=300.0, trajectory_features=tf(DECAY)))
        assert without_valve.regime is Regime.AFTERHEAT
        assert without_valve.reliability <= with_valve.reliability

    def test_renewed_demand_ends_afterheat(self):
        r = clf().classify(RegimeInput(
            controller_demand=True, heating_drive_end=drive_end(),
            seconds_since_drive_end=300.0, trajectory_features=tf(DECAY)))
        assert r.regime is not Regime.AFTERHEAT

    def test_valve_reopen_ends_afterheat(self):
        r = clf().classify(RegimeInput(
            valve_opening=Measurement(40.0, DataQuality.OK),
            heating_drive_end=drive_end(), seconds_since_drive_end=300.0,
            trajectory_features=tf(DECAY)))
        assert r.regime is Regime.ACTIVE_HEATING

    def test_solar_confounder_blocks_afterheat(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, heating_drive_end=drive_end(),
            seconds_since_drive_end=300.0,
            solar_radiation=Measurement(600.0, DataQuality.OK),
            trajectory_features=tf(DECAY)))
        assert r.regime is Regime.UNKNOWN
        assert ReasonCode.SOLAR_CONFOUNDER.value in r.reason_codes

    def test_too_long_since_drive_end(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, heating_drive_end=drive_end(),
            seconds_since_drive_end=3600.0, trajectory_features=tf(DECAY)))
        assert r.regime is not Regime.AFTERHEAT

    def test_too_few_points(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, heating_drive_end=drive_end(),
            seconds_since_drive_end=300.0, trajectory_features=tf([20.0, 20.3])))
        assert r.regime is Regime.UNKNOWN


# -- passive cooling ----------------------------------------------------------

class TestPassiveCooling:
    def test_falling_without_heating(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, trajectory_features=tf(FALLING)))
        assert r.regime is Regime.PASSIVE_COOLING

    def test_without_outdoor_is_relative(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, trajectory_features=tf(FALLING)))
        assert ReasonCode.RELATIVE_COOLING_ONLY.value in r.reason_codes

    def test_window_open_is_disturbed_not_cooling(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, window_open=True, trajectory_features=tf(FALLING)))
        assert r.regime is Regime.DISTURBED

    def test_outdoor_present_higher_reliability(self):
        rel = clf().classify(RegimeInput(
            controller_demand=False, trajectory_features=tf(FALLING)))
        abs_ = clf().classify(RegimeInput(
            controller_demand=False, outdoor_temp=Measurement(2.0, DataQuality.OK),
            trajectory_features=tf(FALLING)))
        assert abs_.reliability >= rel.reliability


# -- stable -------------------------------------------------------------------

class TestStable:
    def test_constant_temperature(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, indoor_temp=20.0, target=21.0,
            trajectory_features=tf(FLAT)))
        assert r.regime is Regime.STABLE

    def test_stable_in_comfort_band(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, indoor_temp=21.0, target=21.0,
            trajectory_features=tf([21.0, 21.0, 21.0, 21.0])))
        assert ReasonCode.WITHIN_COMFORT_BAND.value in r.reason_codes

    def test_stable_outside_comfort_band(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, indoor_temp=18.0, target=21.0,
            trajectory_features=tf([18.0, 18.0, 18.0, 18.0])))
        assert r.regime is Regime.STABLE
        assert ReasonCode.OUTSIDE_COMFORT_BAND.value in r.reason_codes

    def test_too_short_history_is_unknown(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, trajectory_features=tf([20.0, 20.0])))
        assert r.regime is Regime.UNKNOWN


# -- disturbed / unknown ------------------------------------------------------

class TestDisturbedUnknown:
    def test_heating_failure(self):
        r = clf().classify(RegimeInput(heating_failure=True, trajectory_features=tf(RISING)))
        assert r.regime is Regime.DISTURBED
        assert DisturbanceCode.HEATING_FAILURE.value in r.disturbance_reasons

    def test_window_open(self):
        r = clf().classify(RegimeInput(window_open=True, trajectory_features=tf(FLAT)))
        assert r.regime is Regime.DISTURBED

    def test_invalid_core_measurement(self):
        r = clf().classify(RegimeInput(indoor_temp=float("nan")))
        assert r.regime is Regime.DISTURBED
        assert DisturbanceCode.INVALID_CORE_MEASUREMENT.value in r.disturbance_reasons

    def test_high_gap_ratio(self):
        pts = [TrajectoryPoint(0, 20.0)] + [
            TrajectoryPoint((i + 1) * STEP_MS, None, DataQuality.UNAVAILABLE, gap=True)
            for i in range(3)]
        r = clf().classify(RegimeInput(
            trajectory_features=EX.extract_trajectory_features(
                Trajectory(points=tuple(pts), max_points=10))))
        assert r.regime is Regime.DISTURBED
        assert DisturbanceCode.HIGH_GAP_RATIO.value in r.disturbance_reasons

    def test_no_indoor_no_trajectory_is_unknown(self):
        r = clf().classify(RegimeInput(controller_demand=False))
        assert r.regime is Regime.UNKNOWN
        assert ReasonCode.NO_INDOOR_TEMP.value in r.reason_codes

    def test_classifier_version_in_result(self):
        r = clf().classify(RegimeInput(trajectory_features=tf(FLAT),
                                       indoor_temp=20.0, target=20.0,
                                       controller_demand=False))
        assert r.classifier_version == CLASSIFIER_VERSION
