"""Deterministic, validated store-key naming for LE 2.0 (pure Python).

Store keys are derived solely from the opaque ``learning_zone_id`` and the
track/segment context — never from visible zone names or entity ids. Keys are
filesystem-safe (Home Assistant persists each store as a ``.storage`` file),
versioned, and collision-free across zones, tracks and segments.
"""
from __future__ import annotations

import re

from ..raw_schemas import RawTrackName

# Identifies the Learning Engine generation these stores belong to (LE 2.0).
# Distinct from the v1 store key ``thermosmart_learning_data``.
LEARNING_ENGINE_GENERATION = 2

_DOMAIN = "thermosmart"
_PREFIX = f"{_DOMAIN}_le{LEARNING_ENGINE_GENERATION}"

# Components must be filesystem-safe and must not contain the "__" separator.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class StorageNamingError(Exception):
    """Raised when an id or naming component is unsafe or malformed."""


def validate_learning_zone_id(learning_zone_id: str) -> str:
    """Validate an opaque learning zone id for safe use in a store key."""
    if not isinstance(learning_zone_id, str) or not _ID_PATTERN.match(learning_zone_id):
        raise StorageNamingError(
            f"invalid learning_zone_id: {learning_zone_id!r}"
        )
    if "__" in learning_zone_id:
        raise StorageNamingError("learning_zone_id must not contain '__'")
    return learning_zone_id


def _track_component(track_name: RawTrackName) -> str:
    if not isinstance(track_name, RawTrackName):
        raise StorageNamingError("track_name must be a RawTrackName")
    value = track_name.value
    if "__" in value:  # defensive: canonical track values never contain "__"
        raise StorageNamingError(f"unsafe track component: {value!r}")
    return value


def _key(learning_zone_id: str, suffix: str) -> str:
    zid = validate_learning_zone_id(learning_zone_id)
    return f"{_PREFIX}__{zid}__{suffix}"


def container_key(learning_zone_id: str) -> str:
    return _key(learning_zone_id, "container")


def raw_index_key(learning_zone_id: str, track_name: RawTrackName) -> str:
    return _key(learning_zone_id, f"raw_{_track_component(track_name)}_index")


def raw_segment_key(
    learning_zone_id: str, track_name: RawTrackName, sequence_number: int
) -> str:
    if not isinstance(sequence_number, int) or sequence_number < 0:
        raise StorageNamingError("sequence_number must be a non-negative int")
    return _key(
        learning_zone_id,
        f"raw_{_track_component(track_name)}_seg_{sequence_number}",
    )


def episodes_key(learning_zone_id: str) -> str:
    return _key(learning_zone_id, "episodes")


def model_state_key(learning_zone_id: str) -> str:
    return _key(learning_zone_id, "models")


def global_index_key() -> str:
    """The single global index store key (a reconstructable cache)."""
    return f"{_PREFIX}__global_index"


def adaptation_history_key(learning_zone_id: str) -> str:
    """Store key for the passive adaptation candidate history of one zone."""
    return _key(learning_zone_id, "adaptation_history")
