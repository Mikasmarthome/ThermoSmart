"""Deterministic, validated store-key naming for Learning (pure Python).

Store keys are derived solely from the opaque ``learning_zone_id`` and the
track/segment context — never from visible zone names or entity ids. Keys are
filesystem-safe (Home Assistant persists each store as a ``.storage`` file),
versioned, and collision-free across zones, tracks and segments.

Neutral naming migration (Learning Naming / Storage-Key audit):
  The historical ``thermosmart_le2__`` prefix exposed an internal generation
  codename directly in ``.storage`` filenames — a technically curious
  user browsing that folder would see it. ``LEARNING_STORAGE_PREFIX`` is the
  neutral prefix every store now saves under; ``LEGACY_LEARNING_STORAGE_PREFIX``
  is the old prefix, still read as a one-time lazy-migration fallback by the
  store wrapper classes in ``storage/stores.py`` and by
  ``runtime/ha_store.py``'s ``HomeAssistantStoreAdapter`` (both read the
  neutral key first, then the legacy key if the neutral one is empty,
  persisting the legacy data onto the neutral key on that same read — the
  legacy key itself is never deleted by this, only on explicit zone removal).
  The plain ``*_key()`` functions below therefore now only describe the
  legacy half of each pair; production code builds keys through the
  ``*_key_pair()`` functions instead. Suffixes (``container``, ``models``,
  ``episodes``, ...) are identical between the two prefixes — only the
  domain/generation prefix differs.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

from ..raw_schemas import RawTrackName

# Identifies the learning-engine generation these stores belong to.
# Distinct from the v1 store key ``thermosmart_learning_data``.
LEARNING_ENGINE_GENERATION = 2

_DOMAIN = "thermosmart"

# Target neutral prefix (no internal generation codename) and the current/
# actual prefix every existing *_key() function below still produces.
# _PREFIX stays an alias of the legacy constant so none of the functions
# already in production use change their output as a side effect of this
# module gaining neutral-naming awareness.
LEARNING_STORAGE_PREFIX = f"{_DOMAIN}_learning"
LEGACY_LEARNING_STORAGE_PREFIX = f"{_DOMAIN}_le{LEARNING_ENGINE_GENERATION}"
_PREFIX = LEGACY_LEARNING_STORAGE_PREFIX

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


def application_lifecycle_key(learning_zone_id: str) -> str:
    """Store key for the adaptation application lifecycle state of one zone.

    Uses suffix ``application_lifecycle`` — fachlich klar (Lifecycle-State von
    angewendeten Adaptations) and parallel to ``adaptation_history``.
    """
    return _key(learning_zone_id, "application_lifecycle")


def support_critical_events_key(learning_zone_id: str) -> str:
    """Store key for the bounded Support Critical Event history of one zone.

    Parallel to ``episodes_key`` — one store per zone, keyed by the same
    opaque ``learning_zone_id``, never a real zone name.
    """
    return _key(learning_zone_id, "support_critical_events")


def research_daily_key(learning_zone_id: str) -> str:
    """Store key for the bounded Research Daily Bucket history of one zone.

    Parallel to ``episodes_key``/``support_critical_events_key`` — one store
    per zone, keyed by the same opaque ``learning_zone_id``, never a real
    zone name.
    """
    return _key(learning_zone_id, "research_daily")


def storage_metadata_key(learning_zone_id: str) -> str:
    """Store key for the per-zone storage-metadata index (Storage-Metadata
    concept, Commit A): tracks per-store created/updated timestamps and
    last-write-reason for Support/Research export — never raw learning data.

    Parallel to ``episodes_key``/``research_daily_key`` — one store per zone,
    keyed by the same opaque ``learning_zone_id``, never a real zone name.
    """
    return _key(learning_zone_id, "storage_metadata")


# ── Neutral-naming migration prep (no production key has switched yet) ──────

@dataclass(frozen=True)
class LearningStorageKeyPair:
    """One store's target (neutral) key alongside its legacy read-fallback key.

    ``current`` is the neutral ``thermosmart_learning__...`` name a fresh or
    migrated store should be saved under. ``legacy`` is the
    ``thermosmart_le2__...`` name every store in production actually uses
    today — the read-fallback a future migration commit would try if
    ``current`` does not exist yet. This pairing is pure data; nothing in this
    module reads or writes either key. ``legacy`` is ``None`` only in the
    (currently unused) general case where a store type has no predecessor.
    """
    current: str
    legacy: Optional[str] = None


def _neutral_key(learning_zone_id: str, suffix: str) -> str:
    """Build the neutral-prefix counterpart of ``_key()`` — same validation,
    same suffix, only ``LEARNING_STORAGE_PREFIX`` instead of the legacy one."""
    zid = validate_learning_zone_id(learning_zone_id)
    return f"{LEARNING_STORAGE_PREFIX}__{zid}__{suffix}"


def container_key_pair(learning_zone_id: str) -> LearningStorageKeyPair:
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, "container"),
        legacy=container_key(learning_zone_id),
    )


def raw_index_key_pair(learning_zone_id: str, track_name: RawTrackName) -> LearningStorageKeyPair:
    suffix = f"raw_{_track_component(track_name)}_index"
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, suffix),
        legacy=raw_index_key(learning_zone_id, track_name),
    )


def raw_segment_key_pair(
    learning_zone_id: str, track_name: RawTrackName, sequence_number: int
) -> LearningStorageKeyPair:
    if not isinstance(sequence_number, int) or sequence_number < 0:
        raise StorageNamingError("sequence_number must be a non-negative int")
    suffix = f"raw_{_track_component(track_name)}_seg_{sequence_number}"
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, suffix),
        legacy=raw_segment_key(learning_zone_id, track_name, sequence_number),
    )


def episodes_key_pair(learning_zone_id: str) -> LearningStorageKeyPair:
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, "episodes"),
        legacy=episodes_key(learning_zone_id),
    )


def model_state_key_pair(learning_zone_id: str) -> LearningStorageKeyPair:
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, "models"),
        legacy=model_state_key(learning_zone_id),
    )


def global_index_key_pair() -> LearningStorageKeyPair:
    return LearningStorageKeyPair(
        current=f"{LEARNING_STORAGE_PREFIX}__global_index",
        legacy=global_index_key(),
    )


def adaptation_history_key_pair(learning_zone_id: str) -> LearningStorageKeyPair:
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, "adaptation_history"),
        legacy=adaptation_history_key(learning_zone_id),
    )


def application_lifecycle_key_pair(learning_zone_id: str) -> LearningStorageKeyPair:
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, "application_lifecycle"),
        legacy=application_lifecycle_key(learning_zone_id),
    )


def support_critical_events_key_pair(learning_zone_id: str) -> LearningStorageKeyPair:
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, "support_critical_events"),
        legacy=support_critical_events_key(learning_zone_id),
    )


def research_daily_key_pair(learning_zone_id: str) -> LearningStorageKeyPair:
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, "research_daily"),
        legacy=research_daily_key(learning_zone_id),
    )


def storage_metadata_key_pair(learning_zone_id: str) -> LearningStorageKeyPair:
    """Key pair for the storage-metadata index.

    This store type never existed under the legacy ``thermosmart_le2__``
    prefix (it's new as of the Storage-Metadata concept, introduced after the
    neutral-naming migration) — no real legacy data can ever be found under
    ``legacy``. The pair is still built the same way as every other
    zone-scoped store for two reasons: (1) API uniformity with
    ``_VersionedStore``/every other ``*_key_pair()`` function here, and (2) it
    costs nothing — constructing a ``Store`` object doesn't touch disk, and
    ``_VersionedStore`` never writes to ``legacy``, only reads it as a
    fallback that will simply stay empty forever for this store type.
    """
    return LearningStorageKeyPair(
        current=_neutral_key(learning_zone_id, "storage_metadata"),
        legacy=storage_metadata_key(learning_zone_id),
    )


def zone_segment(learning_zone_id: str) -> str:
    """Non-identifying, stable token for the Learning runtime-snapshot store key
    (ha_store.py's ``store_key()``) — never an entity name.

    Consolidated here (single source of truth) from two previously
    independent, byte-identical implementations:
    ``learning/runtime/ha_integration.py``'s private ``_zone_segment()`` and
    an inline duplicate in ``__init__.py``'s ``_async_purge_learning_zone_storage()``
    (which deliberately avoided importing the heavier runtime module — this
    module, already a pure/no-HA-dependency leaf, satisfies that same
    constraint). Same algorithm, same output, for any given input — this is a
    pure refactor, not a behavior change.
    """
    return hashlib.sha256(learning_zone_id.encode("utf-8")).hexdigest()[:16]
