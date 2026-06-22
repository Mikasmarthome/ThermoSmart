"""Phase 4 tests: time, temperature, heating and context features + reliability."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.features import (
    FEATURE_EXTRACTOR_VERSION,
    FeatureExtractor,
    FeatureName,
    FeatureParameters,
    FeatureResult,
    FeatureStatus,
)

UTC = timezone.utc
BERLIN = ZoneInfo("Europe/Berlin")


def ex() -> FeatureExtractor:
    return FeatureExtractor()


# -- FeatureResult contract ---------------------------------------------------

class TestFeatureResult:
    def test_present_requires_finite_value(self):
        with pytest.raises(ValueError):
            FeatureResult(FeatureName.OVERSHOOT, "C", FeatureStatus.OK, 1.0, value=float("nan"))

    def test_present_requires_value(self):
        with pytest.raises(ValueError):
            FeatureResult(FeatureName.OVERSHOOT, "C", FeatureStatus.OK, 1.0, value=None)

    def test_non_present_must_not_carry_value(self):
        with pytest.raises(ValueError):
            FeatureResult(FeatureName.OVERSHOOT, "C", FeatureStatus.NOT_COMPUTABLE, 0.0, value=1.0)

    def test_reliability_range(self):
        with pytest.raises(ValueError):
            FeatureResult(FeatureName.OVERSHOOT, "C", FeatureStatus.OK, 1.5, value=1.0)

    def test_version_default(self):
        r = FeatureResult(FeatureName.OVERSHOOT, "C", FeatureStatus.OK, 1.0, value=1.0)
        assert r.extractor_version == FEATURE_EXTRACTOR_VERSION
        assert r.is_present


# -- time ---------------------------------------------------------------------

class TestTimeFeatures:
    def test_utc_to_civil_winter(self):
        fs = ex().extract_time_features(datetime(2026, 1, 1, 12, 0, tzinfo=UTC), BERLIN)
        assert fs.get(FeatureName.HOUR_OF_DAY).value == 13.0  # CET +1

    def test_utc_to_civil_summer(self):
        fs = ex().extract_time_features(datetime(2026, 7, 1, 12, 0, tzinfo=UTC), BERLIN)
        assert fs.get(FeatureName.HOUR_OF_DAY).value == 14.0  # CEST +2

    def test_dst_spring_forward(self):
        # 2026-03-29 01:30Z -> CEST (+2) -> 03:30 local
        fs = ex().extract_time_features(datetime(2026, 3, 29, 1, 30, tzinfo=UTC), BERLIN)
        assert fs.get(FeatureName.HOUR_OF_DAY).value == 3.0

    def test_dst_fall_back(self):
        # 2026-10-25 01:30Z -> after fall-back CET (+1) -> 02:30 local
        fs = ex().extract_time_features(datetime(2026, 10, 25, 1, 30, tzinfo=UTC), BERLIN)
        assert fs.get(FeatureName.HOUR_OF_DAY).value == 2.0

    def test_weekday(self):
        fs = ex().extract_time_features(datetime(2026, 1, 5, 12, 0, tzinfo=UTC), BERLIN)
        assert fs.get(FeatureName.WEEKDAY).value == 0.0  # Monday

    def test_naive_rejected(self):
        with pytest.raises(ValueError):
            ex().extract_time_features(datetime(2026, 1, 1, 12, 0), BERLIN)

    def test_duration_from_offsets(self):
        r = ex().duration_from_offsets(0, 90000)
        assert r.value == 90.0

    def test_duration_negative_not_computable(self):
        r = ex().duration_seconds(datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
                                  datetime(2026, 1, 1, 0, 0, tzinfo=UTC))
        assert r.status is FeatureStatus.NOT_COMPUTABLE


# -- temperature --------------------------------------------------------------

class TestTemperatureFeatures:
    def test_target_delta(self):
        fs = ex().extract_temperature_features(indoor_temp=19.0, target=21.0)
        assert fs.get(FeatureName.INDOOR_TARGET_DELTA).value == 2.0

    def test_zero_indoor_valid(self):
        fs = ex().extract_temperature_features(indoor_temp=0.0, target=0.0)
        r = fs.get(FeatureName.INDOOR_TARGET_DELTA)
        assert r.is_present and r.value == 0.0

    def test_negative_outdoor(self):
        fs = ex().extract_temperature_features(
            indoor_temp=20.0, target=21.0,
            outdoor_temp=Measurement(-8.0, DataQuality.OK))
        assert fs.get(FeatureName.INDOOR_OUTDOOR_DELTA).value == 28.0

    def test_missing_outdoor_not_computable(self):
        fs = ex().extract_temperature_features(indoor_temp=20.0, target=21.0)
        r = fs.get(FeatureName.INDOOR_OUTDOOR_DELTA)
        assert r.status is FeatureStatus.NOT_COMPUTABLE
        assert "outdoor_temp" in r.missing_evidence

    def test_missing_indoor_blocks_temp_features(self):
        fs = ex().extract_temperature_features(indoor_temp=None, target=21.0,
                                               trv_setpoint=22.0)
        assert fs.get(FeatureName.INDOOR_TARGET_DELTA).status is FeatureStatus.NOT_COMPUTABLE
        assert fs.get(FeatureName.SETPOINT_EXCESS).status is FeatureStatus.NOT_COMPUTABLE
        assert "indoor_temp" in fs.get(FeatureName.INDOOR_TARGET_DELTA).missing_evidence

    def test_setpoint_excess(self):
        fs = ex().extract_temperature_features(indoor_temp=20.0, target=21.0, trv_setpoint=22.5)
        assert fs.get(FeatureName.SETPOINT_EXCESS).value == 2.5

    def test_outdoor_stale_marks_stale(self):
        fs = ex().extract_temperature_features(
            indoor_temp=20.0, target=21.0,
            outdoor_temp=Measurement(5.0, DataQuality.STALE))
        assert fs.get(FeatureName.INDOOR_OUTDOOR_DELTA).status is FeatureStatus.STALE

    def test_outdoor_unavailable(self):
        fs = ex().extract_temperature_features(
            indoor_temp=20.0, target=21.0,
            outdoor_temp=Measurement(None, DataQuality.UNAVAILABLE))
        assert fs.get(FeatureName.INDOOR_OUTDOOR_DELTA).status is FeatureStatus.UNAVAILABLE


# -- heating / charge ---------------------------------------------------------

class TestHeatingFeatures:
    def test_heating_duration(self):
        fs = ex().extract_heating_features(heating_duration_s=1800.0)
        assert fs.get(FeatureName.HEATING_DURATION).value == 1800.0

    def test_setpoint_excess_integral(self):
        # constant excess 2.0 over 10 minutes -> 20 C*min
        samples = [(0, 2.0), (600000, 2.0)]
        fs = ex().extract_heating_features(setpoint_excess_samples=samples)
        assert fs.get(FeatureName.SETPOINT_EXCESS_INTEGRAL).value == pytest.approx(20.0)

    def test_thermal_charge_uses_integral(self):
        samples = [(0, 2.0), (600000, 2.0)]
        fs = ex().extract_heating_features(setpoint_excess_samples=samples)
        r = fs.get(FeatureName.THERMAL_CHARGE_PROXY)
        assert r.status is FeatureStatus.OK and r.value == pytest.approx(20.0)

    def test_thermal_charge_falls_back_to_duration(self):
        fs = ex().extract_heating_features(heating_duration_s=1800.0)
        r = fs.get(FeatureName.THERMAL_CHARGE_PROXY)
        assert r.status is FeatureStatus.FALLBACK and r.reliability < 1.0

    def test_valve_open_time_unavailable_when_absent(self):
        fs = ex().extract_heating_features(heating_duration_s=60.0)
        assert fs.get(FeatureName.VALVE_OPEN_TIME).status is FeatureStatus.UNAVAILABLE
        assert fs.get(FeatureName.TPI_ON_TIME).status is FeatureStatus.UNAVAILABLE

    def test_valve_open_time_present_when_given(self):
        fs = ex().extract_heating_features(valve_open_time_s=120.0, tpi_on_time_s=90.0)
        assert fs.get(FeatureName.VALVE_OPEN_TIME).value == 120.0
        assert fs.get(FeatureName.TPI_ON_TIME).value == 90.0


# -- context ------------------------------------------------------------------

class TestContextFeatures:
    def test_no_weather_not_computable(self):
        fs = ex().extract_context_features()
        assert fs.get(FeatureName.OUTDOOR_TEMP_BUCKET).status is FeatureStatus.NOT_COMPUTABLE
        assert fs.get(FeatureName.WEATHER_QUALITY).status is FeatureStatus.NOT_COMPUTABLE

    def test_outdoor_bucket(self):
        fs = ex().extract_context_features(outdoor_temp=Measurement(7.0, DataQuality.OK))
        # edges (-10,-5,0,5,10,15): 7.0 -> index 4
        assert fs.get(FeatureName.OUTDOOR_TEMP_BUCKET).value == 4.0

    def test_valve_present_without_weather_is_nonlinear(self):
        # context only cares about what's present; weather absent is fine
        fs = ex().extract_context_features(wind_speed=Measurement(3.0, DataQuality.OK))
        assert fs.get(FeatureName.WIND_BUCKET).is_present
        assert fs.get(FeatureName.OUTDOOR_TEMP_BUCKET).status is FeatureStatus.NOT_COMPUTABLE

    def test_weather_quality_ratio(self):
        fs = ex().extract_context_features(
            outdoor_temp=Measurement(5.0, DataQuality.OK),
            wind_speed=Measurement(None, DataQuality.UNAVAILABLE))
        assert fs.get(FeatureName.WEATHER_QUALITY).value == pytest.approx(0.5)


def test_parameters_are_versioned():
    p = FeatureParameters()
    assert p.parameter_version == 1
    assert p.plausibility_max_rate_per_min == 0.5
