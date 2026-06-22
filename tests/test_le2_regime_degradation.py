"""Phase 5 TRV-only, no-indoor-temp, degradation and missing-evidence tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement, Regime
from custom_components.thermosmart.learning.episode_schemas import Trajectory, TrajectoryPoint
from custom_components.thermosmart.learning.features import FeatureExtractor
from custom_components.thermosmart.learning.regime import (
    MissingEvidence,
    ReasonCode,
    RegimeInput,
    ThermalRegimeClassifier,
)

EX = FeatureExtractor()
STEP_MS = 5 * 60 * 1000


def traj(values):
    return Trajectory(points=tuple(TrajectoryPoint(i * STEP_MS, float(v))
                                   for i, v in enumerate(values)), max_points=120)


def tf(values):
    return EX.extract_trajectory_features(traj(values))


def clf():
    return ThermalRegimeClassifier()


FLAT = [20.0, 20.0, 20.0, 20.0]
RISING = [18.0, 18.2, 18.4, 18.6]
FALLING = [20.0, 19.8, 19.6, 19.4]


class TestTrvOnly:
    def test_trv_only_stable_classification(self):
        # only TRV temp/setpoint/target/demand + trend, no telemetry/weather
        r = clf().classify(RegimeInput(
            indoor_temp=20.0, trv_setpoint=20.0, target=21.0,
            controller_demand=False, trajectory_features=tf(FLAT)))
        assert r.regime is Regime.STABLE
        assert r.reliability > 0.0

    def test_trv_only_active_classification(self):
        r = clf().classify(RegimeInput(
            indoor_temp=18.0, trv_setpoint=22.0, target=21.0,
            controller_demand=True, trajectory_features=tf(RISING)))
        assert r.regime is Regime.ACTIVE_HEATING

    def test_missing_telemetry_documented_not_disturbed(self):
        r = clf().classify(RegimeInput(
            indoor_temp=20.0, target=21.0, controller_demand=False,
            trajectory_features=tf(FLAT)))
        assert r.regime is not Regime.DISTURBED
        for code in (MissingEvidence.VALVE_OPENING.value, MissingEvidence.HVAC_ACTION.value,
                     MissingEvidence.WINDOW_STATE.value, MissingEvidence.OUTDOOR_TEMP.value,
                     MissingEvidence.SOLAR.value):
            assert code in r.missing_evidence

    def test_no_crash_with_minimal_input(self):
        r = clf().classify(RegimeInput(trajectory_features=tf(FALLING),
                                       controller_demand=False))
        assert r.regime in tuple(Regime)


class TestNoIndoorTemp:
    def test_no_temp_no_trajectory_is_unknown(self):
        r = clf().classify(RegimeInput(controller_demand=True, trv_setpoint=22.0))
        assert r.regime is Regime.UNKNOWN
        assert ReasonCode.NO_TEMP_TREND.value in r.reason_codes

    def test_no_indoor_documents_missing(self):
        r = clf().classify(RegimeInput(controller_demand=False))
        assert MissingEvidence.INDOOR_TEMP.value in r.missing_evidence
        assert MissingEvidence.TEMPERATURE_TREND.value in r.missing_evidence

    def test_no_indoor_does_not_invent_regime(self):
        r = clf().classify(RegimeInput(trv_setpoint=22.0, target=21.0))
        # no demand, no trend -> honest unknown, never a thermal regime
        assert r.regime is Regime.UNKNOWN


class TestDegradation:
    def test_valve_unavailable_does_not_disturb(self):
        r = clf().classify(RegimeInput(
            indoor_temp=20.0, target=21.0, controller_demand=False,
            valve_opening=Measurement(None, DataQuality.UNAVAILABLE),
            trajectory_features=tf(FLAT)))
        assert r.regime is Regime.STABLE

    def test_missing_optional_does_not_zero_reliability(self):
        r = clf().classify(RegimeInput(
            controller_demand=False, trajectory_features=tf(FALLING)))
        assert r.regime is Regime.PASSIVE_COOLING
        assert r.reliability > 0.0

    def test_recovered_window_state_changes_nothing_thermally(self):
        # window known-closed vs unknown both allow the thermal regime
        closed = clf().classify(RegimeInput(
            controller_demand=False, window_open=False, trajectory_features=tf(FALLING)))
        unknown = clf().classify(RegimeInput(
            controller_demand=False, trajectory_features=tf(FALLING)))
        assert closed.regime is unknown.regime is Regime.PASSIVE_COOLING
