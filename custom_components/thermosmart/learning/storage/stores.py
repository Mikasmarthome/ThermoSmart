"""Home Assistant store wrappers for LE 2.0 (the only HA-dependent storage layer).

Thin, versioned, injectable wrappers. They make no business decisions, import
no models and no coordinator, and create no store at import time. Tests inject a
fake store factory so no real Home Assistant instance is required.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from homeassistant.helpers.storage import Store

from ..raw_schemas import RawTrackName
from . import naming

# Per-store schema versions (independent of the key-layout version).
ZONE_METADATA_STORE_VERSION = 1
RAW_SEGMENT_STORE_VERSION = 1
RAW_INDEX_STORE_VERSION = 1
EPISODES_STORE_VERSION = 1
MODEL_STATE_STORE_VERSION = 1
GLOBAL_INDEX_STORE_VERSION = 1
ADAPTATION_HISTORY_STORE_VERSION = 1
APPLICATION_LIFECYCLE_STORE_VERSION = 1
SUPPORT_CRITICAL_EVENT_STORE_VERSION = 1
RESEARCH_DAILY_STORE_VERSION = 1


class StoreError(Exception):
    """Base class for store wrapper errors."""


class StoreVersionError(StoreError):
    """Raised when a loaded store reports an unexpected schema version."""


@runtime_checkable
class StoreLike(Protocol):
    """The subset of the Home Assistant Store API the wrappers rely on."""

    async def async_load(self) -> Any: ...
    async def async_save(self, data: Any) -> None: ...
    async def async_remove(self) -> None: ...


@runtime_checkable
class StoreFactory(Protocol):
    def create(self, key: str, version: int) -> StoreLike: ...


class HomeAssistantStoreFactory:
    """Default factory backed by the real Home Assistant ``Store``."""

    def __init__(self, hass: Any) -> None:
        self._hass = hass

    def create(self, key: str, version: int) -> StoreLike:
        return Store(self._hass, version, key)


class _VersionedStore:
    """Wraps one keyed store with a schema-version envelope.

    Accepts either a plain ``str`` key (no migration — used by callers/tests
    that only ever care about one fixed key) or a
    ``naming.LearningStorageKeyPair`` (neutral ``current`` key plus the
    legacy ``thermosmart_le2__`` fallback). With a pair:

    - ``load()`` tries ``current`` first. If it holds valid data, that's
      returned — legacy is never even read.
    - If ``current`` is absent, ``legacy`` is tried. Valid legacy data is
      returned AND persisted under ``current`` (lazy migration on read) —
      ``legacy`` itself is left untouched, exactly as before.
    - A version mismatch on ``current`` raises ``StoreVersionError`` rather
      than silently falling back to legacy — an existing-but-unreadable new
      store must never be quietly superseded by old data.
    - ``delete()`` removes both keys — only ever reached via explicit zone
      removal (``learning/reset.py``), never unload/reload.
    """

    def __init__(self, factory: StoreFactory, key, version: int) -> None:
        if isinstance(key, str):
            current, legacy = key, None
        else:
            current, legacy = key.current, key.legacy
        self._key = current
        self._legacy_key = legacy
        self._version = version
        self._store = factory.create(current, version)
        self._legacy_store = factory.create(legacy, version) if legacy else None

    @property
    def key(self) -> str:
        return self._key

    def _parse(self, raw: Any, *, key_for_error: str) -> Any:
        if not isinstance(raw, dict) or "store_schema_version" not in raw:
            raise StoreVersionError(f"store '{key_for_error}' has no schema version")
        found = raw["store_schema_version"]
        if found != self._version:
            raise StoreVersionError(
                f"store '{key_for_error}' schema version {found} != expected {self._version}"
            )
        return raw.get("data")

    async def load(self) -> Optional[Any]:
        raw = await self._store.async_load()
        if raw is not None:
            return self._parse(raw, key_for_error=self._key)
        if self._legacy_store is None:
            return None
        legacy_raw = await self._legacy_store.async_load()
        if legacy_raw is None:
            return None
        legacy_data = self._parse(legacy_raw, key_for_error=self._legacy_key)
        # Lazy migration: persist under the neutral key now that it's been
        # read once. Legacy key is a deliberate safety fallback — never
        # deleted here (only on explicit zone removal, see delete() below).
        await self.save(legacy_data)
        return legacy_data

    async def save(self, data: Any) -> None:
        await self._store.async_save(
            {"store_schema_version": self._version, "data": data}
        )

    async def delete(self) -> None:
        try:
            await self._store.async_remove()
        finally:
            if self._legacy_store is not None:
                await self._legacy_store.async_remove()


class ZoneMetadataStore(_VersionedStore):
    """Authority for ``learning_zone_id`` and store versions of a zone."""

    def __init__(self, factory: StoreFactory, learning_zone_id: str) -> None:
        super().__init__(factory, naming.container_key_pair(learning_zone_id),
                         ZONE_METADATA_STORE_VERSION)


class RawSegmentIndexStore(_VersionedStore):
    """Reconstructable cache of a track's segment layout."""

    def __init__(self, factory: StoreFactory, learning_zone_id: str,
                 track_name: RawTrackName) -> None:
        super().__init__(factory, naming.raw_index_key_pair(learning_zone_id, track_name),
                         RAW_INDEX_STORE_VERSION)


class RawSegmentStore:
    """One store per raw segment, addressed by sequence number."""

    def __init__(self, factory: StoreFactory, learning_zone_id: str,
                 track_name: RawTrackName) -> None:
        self._factory = factory
        self._zone = learning_zone_id
        self._track = track_name

    def _segment(self, sequence_number: int) -> _VersionedStore:
        key = naming.raw_segment_key_pair(self._zone, self._track, sequence_number)
        return _VersionedStore(self._factory, key, RAW_SEGMENT_STORE_VERSION)

    async def load_segment(self, sequence_number: int) -> Optional[Any]:
        return await self._segment(sequence_number).load()

    async def save_segment(self, sequence_number: int, payload: Any) -> None:
        await self._segment(sequence_number).save(payload)

    async def delete_segment(self, sequence_number: int) -> None:
        await self._segment(sequence_number).delete()


class EpisodesStore(_VersionedStore):
    """Generic typed container for materialised episodes (form only in Phase 3)."""

    def __init__(self, factory: StoreFactory, learning_zone_id: str) -> None:
        super().__init__(factory, naming.episodes_key_pair(learning_zone_id),
                         EPISODES_STORE_VERSION)


class ModelStateStore(_VersionedStore):
    """Generic versioned container for model state (form only in Phase 3)."""

    def __init__(self, factory: StoreFactory, learning_zone_id: str) -> None:
        super().__init__(factory, naming.model_state_key_pair(learning_zone_id),
                         MODEL_STATE_STORE_VERSION)


class GlobalIndexStore(_VersionedStore):
    """Fully reconstructable global cache; never an identity authority."""

    def __init__(self, factory: StoreFactory) -> None:
        super().__init__(factory, naming.global_index_key_pair(),
                         GLOBAL_INDEX_STORE_VERSION)


class AdaptationHistoryStore(_VersionedStore):
    """Versioned per-zone store for passive adaptation candidate history.

    Keyed by ``adaptation_history_key(learning_zone_id)``. Not part of the
    core initialization components — reset separately by ``reset_v2_learning_state``.
    """

    def __init__(self, factory: StoreFactory, learning_zone_id: str) -> None:
        super().__init__(
            factory,
            naming.adaptation_history_key_pair(learning_zone_id),
            ADAPTATION_HISTORY_STORE_VERSION,
        )


class ApplicationLifecycleStore(_VersionedStore):
    """Versioned per-zone store for adaptation application lifecycle state.

    Keyed by ``application_lifecycle_key(learning_zone_id)``. Not part of the
    core initialization components — reset separately by ``reset_v2_learning_state``.
    Empty on first load; non-fatal on missing or corrupt data.
    """

    def __init__(self, factory: StoreFactory, learning_zone_id: str) -> None:
        super().__init__(
            factory,
            naming.application_lifecycle_key_pair(learning_zone_id),
            APPLICATION_LIFECYCLE_STORE_VERSION,
        )


class SupportCriticalEventStore(_VersionedStore):
    """Versioned per-zone store for the bounded Support Critical Event history.

    Keyed by ``support_critical_events_key(learning_zone_id)``. Constructed,
    loaded, and saved through the runtime shadow controller's own
    ``_async_load_support_critical_events_safe()``/
    ``_async_save_support_critical_events_safe()`` methods, reached via the
    capture-stores accessor facade (``learning/storage/capture_stores.py``)
    — see ``support_event_persistence.py``'s module docstring for the pure
    append/prune pipeline that runs before saving through this class. Empty
    on first load; non-fatal on missing or corrupt data, exactly like the
    other stores here.
    """

    def __init__(self, factory: StoreFactory, learning_zone_id: str) -> None:
        super().__init__(
            factory,
            naming.support_critical_events_key_pair(learning_zone_id),
            SUPPORT_CRITICAL_EVENT_STORE_VERSION,
        )


class ResearchDailyStore(_VersionedStore):
    """Versioned per-zone store for the bounded Research Daily Bucket history.

    Keyed by ``research_daily_key(learning_zone_id)``. Constructed, loaded,
    and saved through the runtime shadow controller's own
    ``_async_load_research_daily_safe()``/``_async_save_research_daily_safe()``
    methods, reached via the capture-stores accessor facade
    (``learning/storage/capture_stores.py``) — see
    ``research_daily_persistence.py``'s module docstring for the pure
    merge/persist pipeline that runs before saving through this class.
    Empty on first load; non-fatal on missing or corrupt data, exactly like
    the other stores here.
    """

    def __init__(self, factory: StoreFactory, learning_zone_id: str) -> None:
        super().__init__(
            factory,
            naming.research_daily_key_pair(learning_zone_id),
            RESEARCH_DAILY_STORE_VERSION,
        )
