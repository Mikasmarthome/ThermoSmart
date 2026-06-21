"""Phase 1 tests for LE 2.0 raw record schemas (TRV-only invariant)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.raw_schemas import (
    CorrectionActionType,
    CorrectionSource,
    DriveEndSource,
    ForecastDecisionEvent,
    ForecastEvaluationEvent,
    HeatingDriveEndEvent,
    ManualCorrectionEvent,
    RawTrackName,
    RoomObservation,
    TrvObservation,
)

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 20, 30, 0, tzinfo=UTC)


# -- RoomObservation ----------------------------------------------------------

class TestRoomObservation:
    def test_trv_only_minimal_record_valid(self):
        obs = RoomObservation(learning_zone_id="lz_1", ts=NOW, indoor_temp=20.5, target=21.0)
        assert obs.outdoor_temp is None
        assert obs.window_open is None

    def test_zero_indoor_temp_is_valid(self):
        obs = RoomObservation(learning_zone_id="lz_1", ts=NOW, indoor_temp=0.0, target=21.0)
        assert obs.indoor_temp == 0.0

    def test_optional_environment_fully_absent(self):
        # No outdoor/weather/window/humidity at all -> still valid.
        obs = RoomObservation(learning_zone_id="lz_1", ts=NOW, indoor_temp=19.0, target=20.0)
        assert obs.outdoor_temp is None and obs.solar_radiation is None

    def test_optional_unavailable_quality(self):
        obs = RoomObservation(
            learning_zone_id="lz_1", ts=NOW, indoor_temp=19.0, target=20.0,
            outdoor_temp=Measurement(None, DataQuality.UNAVAILABLE),
        )
        assert obs.outdoor_temp.is_usable is False

    def test_optional_stale_quality(self):
        obs = RoomObservation(
            learning_zone_id="lz_1", ts=NOW, indoor_temp=19.0, target=20.0,
            outdoor_temp=Measurement(2.0, DataQuality.STALE),
        )
        assert obs.outdoor_temp.quality is DataQuality.STALE

    def test_missing_identity_rejected(self):
        with pytest.raises(ValueError):
            RoomObservation(learning_zone_id="", ts=NOW, indoor_temp=20.0, target=21.0)

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError):
            RoomObservation(
                learning_zone_id="lz_1",
                ts=datetime(2026, 1, 15, 20, 30, 0),
                indoor_temp=20.0, target=21.0,
            )

    def test_non_utc_timestamp_normalised(self):
        tz = timezone(timedelta(hours=2))
        obs = RoomObservation(
            learning_zone_id="lz_1",
            ts=datetime(2026, 1, 15, 22, 30, 0, tzinfo=tz),
            indoor_temp=20.0, target=21.0,
        )
        assert obs.ts.utcoffset() == timedelta(0)
        assert obs.ts.hour == 20


# -- TrvObservation -----------------------------------------------------------

class TestTrvObservation:
    def test_minimal_without_temp_or_valve(self):
        # TRV that reports neither temperature nor valve opening -> still valid.
        obs = TrvObservation(
            learning_zone_id="lz_1", ts=NOW, trv_binding_id="trv_a",
            trv_setpoint=22.0, target=21.0,
        )
        assert obs.indoor_temp is None and obs.valve_opening is None

    def test_valve_opening_zero_valid(self):
        obs = TrvObservation(
            learning_zone_id="lz_1", ts=NOW, trv_binding_id="trv_a",
            trv_setpoint=22.0, target=21.0,
            valve_opening=Measurement(0.0, DataQuality.OK),
        )
        assert obs.valve_opening.value == 0.0 and obs.valve_opening.is_usable

    def test_missing_binding_rejected(self):
        with pytest.raises(ValueError):
            TrvObservation(learning_zone_id="lz_1", ts=NOW, trv_binding_id="",
                           trv_setpoint=22.0, target=21.0)

    def test_multiple_trvs_in_one_zone(self):
        a = TrvObservation(learning_zone_id="lz_1", ts=NOW, trv_binding_id="trv_a",
                           trv_setpoint=22.0, target=21.0)
        b = TrvObservation(learning_zone_id="lz_1", ts=NOW, trv_binding_id="trv_b",
                           trv_setpoint=23.0, target=21.0)
        assert a.learning_zone_id == b.learning_zone_id
        assert a.trv_binding_id != b.trv_binding_id


# -- HeatingDriveEndEvent -----------------------------------------------------

class TestHeatingDriveEndEvent:
    def test_inferred_source_without_valve_valid(self):
        ev = HeatingDriveEndEvent(
            learning_zone_id="lz_1", ts=NOW, event_id="ev_1",
            source=DriveEndSource.INFERRED_HEAT_INPUT_END, source_confidence=0.4,
        )
        assert ev.valve_before is None and ev.valve_after is None

    def test_source_confidence_range(self):
        with pytest.raises(ValueError):
            HeatingDriveEndEvent(
                learning_zone_id="lz_1", ts=NOW, event_id="ev_1",
                source=DriveEndSource.VALVE_CLOSING, source_confidence=1.5,
            )


# -- Forecast events ----------------------------------------------------------

class TestForecastEvents:
    def test_decision_valid(self):
        d = ForecastDecisionEvent(
            learning_zone_id="lz_1", ts=NOW, decision_id="dc_1",
            decision_target=20.0, raw_target=21.0, forecast_high=18.0, suppression=0.5,
        )
        assert d.decision_id == "dc_1"

    def test_suppression_range(self):
        with pytest.raises(ValueError):
            ForecastDecisionEvent(
                learning_zone_id="lz_1", ts=NOW, decision_id="dc_1",
                decision_target=20.0, raw_target=21.0, forecast_high=18.0, suppression=1.5,
            )

    def test_evaluation_links_via_decision_id(self):
        ev = ForecastEvaluationEvent(
            learning_zone_id="lz_1", ts=NOW, evaluation_id="ev_1",
            decision_id="dc_1", observed_temp_at_eval=19.8,
        )
        assert ev.decision_id == "dc_1"


# -- ManualCorrectionEvent ----------------------------------------------------

class TestManualCorrectionEvent:
    def test_valid(self):
        ev = ManualCorrectionEvent(
            learning_zone_id="lz_1", ts=NOW, event_id="mc_1",
            action_type=CorrectionActionType.SETPOINT_CHANGE,
            prior_target=21.0, new_target=22.5, source=CorrectionSource.UI,
        )
        # magnitude/direction are derived later, not stored
        assert not hasattr(ev, "magnitude")

    def test_requires_targets(self):
        with pytest.raises(ValueError):
            ManualCorrectionEvent(
                learning_zone_id="lz_1", ts=NOW, event_id="mc_1",
                action_type=CorrectionActionType.SETPOINT_CHANGE,
                prior_target=None, new_target=22.5, source=CorrectionSource.UI,
            )


def test_track_names_stable():
    assert RawTrackName.ROOM.value == "room"
    assert RawTrackName.TRV.value == "trv"
    assert RawTrackName.HEATING_DRIVE_END.value == "heating_drive_end"
    assert RawTrackName.FORECAST_DECISION.value == "forecast_decision"
