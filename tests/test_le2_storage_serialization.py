"""Tests for the shared Learning storage codecs (timestamps and measurements)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.storage.serialization import (
    SerializationError,
    decode_measurement,
    decode_utc,
    encode_measurement,
    encode_utc,
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


class TestMeasurements:
    def test_unconfigured_is_null(self):
        assert encode_measurement(None) is None
        assert decode_measurement(None) is None

    def test_unavailable_measurement_preserved(self):
        encoded = encode_measurement(Measurement(None, DataQuality.UNAVAILABLE))
        assert encoded == {"value": None, "quality": "unavailable"}
        assert decode_measurement(encoded).quality is DataQuality.UNAVAILABLE

    def test_zero_measurement_preserved(self):
        encoded = encode_measurement(Measurement(0.0, DataQuality.OK))
        assert encoded["value"] == 0.0
        assert decode_measurement(encoded).value == 0.0

    def test_unknown_quality_rejected(self):
        with pytest.raises(SerializationError):
            decode_measurement({"value": 1.0, "quality": "bogus"})

    def test_invalid_payload_rejected(self):
        with pytest.raises(SerializationError):
            decode_measurement({"value": 1.0})
