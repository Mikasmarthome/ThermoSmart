"""Phase 3 tests for the authoritative LE 2.0 raw serializer."""
from __future__ import annotations

from datetime import datetime, timezone

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
from custom_components.thermosmart.learning.storage.serialization import (
    SerializationError,
    decode_raw,
    decode_utc,
    encode_raw,
    encode_utc,
    unknown_raw_fields,
)

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 20, 30, 0, tzinfo=UTC)


class TestTimestamps:
    def test_canonical_format(self):
        assert encode_utc(NOW) == "2026-01-15T20:30:00.000Z"

    def test_roundtrip(self):
        assert decode_utc(encode_utc(NOW)) == NOW

    def test_millisecond_precision(self):
        ts = datetime(2026, 1, 15, 20, 30, 0, 123000, tzinfo=UTC)
        assert encode_utc(ts).endswith(".123Z")

    def test_invalid_rejected(self):
        with pytest.raises(SerializationError):
            decode_utc("2026-01-15 20:30:00")


class TestRoomRoundtrip:
    def test_trv_only_roundtrip(self):
        r = RoomObservation(learning_zone_id="lz_1", ts=NOW, indoor_temp=20.5, target=21.0)
        back = decode_raw(RawTrackName.ROOM, encode_raw(RawTrackName.ROOM, r))
        assert back == r

    def test_zero_and_false_preserved(self):
        r = RoomObservation(learning_zone_id="lz_1", ts=NOW, indoor_temp=0.0, target=0.0,
                            active_control=False, window_open=False)
        d = encode_raw(RawTrackName.ROOM, r)
        assert d["indoor_temp"] == 0.0 and d["active_control"] is False
        assert d["window_open"] is False
        assert decode_raw(RawTrackName.ROOM, d) == r

    def test_unconfigured_optional_is_null(self):
        r = RoomObservation(learning_zone_id="lz_1", ts=NOW, indoor_temp=20.0, target=21.0)
        d = encode_raw(RawTrackName.ROOM, r)
        assert d["outdoor_temp"] is None

    def test_unavailable_measurement_preserved(self):
        r = RoomObservation(learning_zone_id="lz_1", ts=NOW, indoor_temp=20.0, target=21.0,
                            outdoor_temp=Measurement(None, DataQuality.UNAVAILABLE))
        d = encode_raw(RawTrackName.ROOM, r)
        assert d["outdoor_temp"] == {"value": None, "quality": "unavailable"}
        back = decode_raw(RawTrackName.ROOM, d)
        assert back.outdoor_temp.quality is DataQuality.UNAVAILABLE

    def test_zero_measurement_preserved(self):
        r = RoomObservation(learning_zone_id="lz_1", ts=NOW, indoor_temp=20.0, target=21.0,
                            wind_speed=Measurement(0.0, DataQuality.OK))
        d = encode_raw(RawTrackName.ROOM, r)
        assert d["wind_speed"]["value"] == 0.0
        assert decode_raw(RawTrackName.ROOM, d).wind_speed.value == 0.0


class TestAllSchemasRoundtrip:
    def test_trv(self):
        r = TrvObservation(learning_zone_id="lz_1", ts=NOW, trv_binding_id="trv_a",
                           trv_setpoint=22.0, target=21.0,
                           valve_opening=Measurement(0.0, DataQuality.OK))
        assert decode_raw(RawTrackName.TRV, encode_raw(RawTrackName.TRV, r)) == r

    def test_drive_end(self):
        r = HeatingDriveEndEvent(learning_zone_id="lz_1", ts=NOW, event_id="ev",
                                 source=DriveEndSource.VALVE_CLOSING, source_confidence=0.9)
        assert decode_raw(RawTrackName.HEATING_DRIVE_END,
                          encode_raw(RawTrackName.HEATING_DRIVE_END, r)) == r

    def test_forecast_decision(self):
        r = ForecastDecisionEvent(learning_zone_id="lz_1", ts=NOW, decision_id="dc",
                                  decision_target=20.0, raw_target=21.0,
                                  forecast_high=18.0, suppression=0.5)
        assert decode_raw(RawTrackName.FORECAST_DECISION,
                          encode_raw(RawTrackName.FORECAST_DECISION, r)) == r

    def test_forecast_evaluation(self):
        r = ForecastEvaluationEvent(learning_zone_id="lz_1", ts=NOW, evaluation_id="ev",
                                    decision_id="dc", observed_temp_at_eval=19.8)
        assert decode_raw(RawTrackName.FORECAST_EVALUATION,
                          encode_raw(RawTrackName.FORECAST_EVALUATION, r)) == r

    def test_manual_correction(self):
        r = ManualCorrectionEvent(learning_zone_id="lz_1", ts=NOW, event_id="mc",
                                  action_type=CorrectionActionType.SETPOINT_CHANGE,
                                  prior_target=21.0, new_target=22.5,
                                  source=CorrectionSource.UI)
        assert decode_raw(RawTrackName.MANUAL_CORRECTION,
                          encode_raw(RawTrackName.MANUAL_CORRECTION, r)) == r


class TestDecodeErrors:
    def test_wrong_type_on_encode(self):
        r = TrvObservation(learning_zone_id="lz_1", ts=NOW, trv_binding_id="t",
                           trv_setpoint=22.0, target=21.0)
        with pytest.raises(SerializationError):
            encode_raw(RawTrackName.ROOM, r)

    def test_invalid_payload(self):
        with pytest.raises(SerializationError):
            decode_raw(RawTrackName.ROOM, {"learning_zone_id": "lz_1"})  # missing ts/temp

    def test_unknown_quality(self):
        with pytest.raises(SerializationError):
            decode_raw(RawTrackName.ROOM, {
                "learning_zone_id": "lz_1", "ts": "2026-01-15T20:30:00.000Z",
                "indoor_temp": 20.0, "target": 21.0,
                "outdoor_temp": {"value": 1.0, "quality": "bogus"},
            })

    def test_future_unknown_field_is_tolerated(self):
        d = encode_raw(RawTrackName.ROOM, RoomObservation(
            learning_zone_id="lz_1", ts=NOW, indoor_temp=20.0, target=21.0))
        d["future_field"] = 123
        back = decode_raw(RawTrackName.ROOM, d)  # ignored, not fatal
        assert back.indoor_temp == 20.0
        assert unknown_raw_fields(RawTrackName.ROOM, d) == ("future_field",)
