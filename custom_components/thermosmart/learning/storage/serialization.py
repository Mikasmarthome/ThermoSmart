"""Shared JSON-compatible codecs for Learning storage payloads.

One implementation, no competing ones. Rules:

* datetimes -> canonical UTC string ``YYYY-MM-DDTHH:MM:SS.mmmZ``;
* ``Measurement`` -> ``{"value": <float|null>, "quality": <str>}`` (so a
  configured-but-unavailable value survives), or ``null`` when the field is not
  configured at all;
* no pickle / no Python-specific persistence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from ..contracts import DataQuality, Measurement, require_utc


class SerializationError(Exception):
    """Raised on an invalid payload during decode/encode."""


def encode_utc(value: datetime) -> str:
    dt = require_utc(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def decode_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SerializationError(f"invalid UTC timestamp: {value!r}")
    try:
        naive = datetime.fromisoformat(value[:-1])
    except ValueError as err:
        raise SerializationError(f"invalid UTC timestamp: {value!r}") from err
    return naive.replace(tzinfo=timezone.utc)


def encode_measurement(value: Optional[Measurement]) -> Optional[dict]:
    if value is None:
        return None  # field not configured
    return {"value": value.value, "quality": value.quality.value}


def decode_measurement(value: Any) -> Optional[Measurement]:
    if value is None:
        return None
    if not isinstance(value, Mapping) or "quality" not in value:
        raise SerializationError(f"invalid measurement payload: {value!r}")
    try:
        quality = DataQuality(value["quality"])
    except ValueError as err:
        raise SerializationError(f"unknown quality: {value['quality']!r}") from err
    return Measurement(value.get("value"), quality)
