"""Home Assistant store wrappers for Learning (the only HA-dependent storage layer).

Thin, versioned, injectable wrappers. They make no business decisions, import
no models and no coordinator, and create no store at import time. Tests inject a
fake store factory so no real Home Assistant instance is required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
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
STORAGE_METADATA_STORE_VERSION = 1

# Inner schema version of the storage-metadata index's own "data" payload —
# independent of STORAGE_METADATA_STORE_VERSION (the _VersionedStore envelope
# version). Bump this only if the {"schema_version": ..., "stores": {...}}
# shape itself changes.
STORAGE_METADATA_SCHEMA_VERSION = 1


def _iso_utc_z(value: datetime) -> str:
    """Format a UTC-aware datetime as ISO-8601 with a literal 'Z' suffix
    (not '+00:00') — the timestamp style requested for storage-metadata."""
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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

    async def save(self, data: Any, *, now_utc: Optional[datetime] = None) -> None:
        """Save ``data`` under the schema-version envelope.

        ``now_utc``, if given, additionally stamps ``created_at_utc``/
        ``updated_at_utc`` at the envelope's top level (next to
        ``store_schema_version``, alongside — never inside — ``data``).
        ``created_at_utc`` is preserved across saves once set (one extra
        read of the current envelope to check); ``updated_at_utc`` always
        reflects this save. Omitting ``now_utc`` (the default — every
        existing call site today) leaves the envelope exactly as before this
        was added: no timestamp fields at all. This is intentionally inert
        until a caller opts in by passing a clock-derived ``now_utc``.
        """
        envelope: dict[str, Any] = {"store_schema_version": self._version, "data": data}
        if now_utc is not None:
            created = _iso_utc_z(now_utc)
            try:
                existing_raw = await self._store.async_load()
                if isinstance(existing_raw, dict) and existing_raw.get("created_at_utc"):
                    created = existing_raw["created_at_utc"]
            except Exception:
                pass  # best-effort only — a read failure here must never block the save
            envelope["created_at_utc"] = created
            envelope["updated_at_utc"] = _iso_utc_z(now_utc)
        await self._store.async_save(envelope)

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


# ── Storage-Metadata concept (Commit A: foundation; Commit B: wired into the ──
# ── production save paths in ha_integration.py/reset.py) ────────────────────

class StorageWriteReason(Enum):
    """Stable, small catalog of why a store was last written.

    One value per store type's own save path (so it's always unambiguous
    which caller set it), plus ``migration`` (set when a lazy legacy-key
    migration triggered the save) and ``reset``/``unspecified`` fallbacks.
    ``zone_remove`` is deliberately not a member: on zone removal the whole
    storage-metadata index is deleted, so no reason value for that event
    would ever be read back.
    """
    RUNTIME_SNAPSHOT_UPDATE = "runtime_snapshot_update"
    MODEL_UPDATE = "model_update"
    EPISODE_CLOSED = "episode_closed"
    RAW_SAMPLE_APPEND = "raw_sample_append"
    RESEARCH_DAILY_UPDATE = "research_daily_update"
    SUPPORT_EVENT_APPEND = "support_event_append"
    ADAPTATION_HISTORY_UPDATE = "adaptation_history_update"
    LIFECYCLE_UPDATE = "lifecycle_update"
    MIGRATION = "migration"
    RESET = "reset"
    UNSPECIFIED = "unspecified"


class StorageKeyState(Enum):
    """Which key variant a store's data currently lives under (or would, if
    the store existed) — surfaced in Support/Research export so a stale
    installation's still-legacy-only zones are visible at a glance."""
    CURRENT = "current"
    MIGRATED_FROM_LEGACY = "migrated_from_legacy"
    MISSING = "missing"
    UNKNOWN = "unknown"


class StorageMetadataStore(_VersionedStore):
    """Per-zone index of when each learning store was last created/updated,
    and why — used by Support/Research export to answer "when was store X
    last written" without touching the actual data stores.

    Never stores raw learning data, entity ids, device ids, or person ids —
    only store names (short internal labels like "episodes"/"models"),
    ISO-8601 UTC timestamps, and the small enums above. Store names are
    accepted as plain strings, not validated against a fixed list: an
    unknown/future store name is simply recorded as-is, never rejected —
    this index must never be the reason a real save path breaks.

    Tolerant by design: a missing index (``get_store_summary()`` returns an
    empty v1 shell) or a corrupt one (``load()`` raising ``StoreVersionError``
    is the caller's responsibility to catch, exactly like every other store
    here) must never affect the actual data stores this index describes —
    it is purely descriptive, never authoritative.
    """

    def __init__(self, factory: StoreFactory, learning_zone_id: str) -> None:
        super().__init__(
            factory,
            naming.storage_metadata_key_pair(learning_zone_id),
            STORAGE_METADATA_STORE_VERSION,
        )

    async def get_store_summary(self) -> dict:
        """Return the current ``{"schema_version": ..., "stores": {...}}``
        dict, or an empty v1 shell if nothing has been recorded yet. Never
        raises for "not found" — only a genuine schema-version mismatch on
        the envelope itself propagates (see ``_VersionedStore.load()``)."""
        data = await self.load()
        if not isinstance(data, dict) or "stores" not in data:
            return {"schema_version": STORAGE_METADATA_SCHEMA_VERSION, "stores": {}}
        return data

    async def mark_store_written(
        self,
        store_name: str,
        reason: StorageWriteReason,
        now_utc: datetime,
        *,
        storage_key_state: StorageKeyState = StorageKeyState.CURRENT,
    ) -> None:
        """Record that ``store_name`` was just written. ``created_at_utc``
        is set once (first call for this store name) and preserved on every
        later call; ``updated_at_utc`` always reflects this call."""
        summary = await self.get_store_summary()
        stores = summary.setdefault("stores", {})
        existing = stores.get(store_name) or {}
        created = existing.get("created_at_utc") or _iso_utc_z(now_utc)
        stores[store_name] = {
            "exists": True,
            "created_at_utc": created,
            "updated_at_utc": _iso_utc_z(now_utc),
            "last_write_reason": reason.value,
            "storage_key_state": storage_key_state.value,
        }
        await self.save(summary)

    async def mark_store_missing(
        self, store_name: str, now_utc: Optional[datetime] = None,
    ) -> None:
        """Record that ``store_name`` does not (or no longer) exist —
        e.g. after an explicit reset deleted it. ``now_utc`` is currently
        unused (kept as an explicit parameter for a future last-cleared
        timestamp) — only ``exists: false`` is recorded for v1."""
        summary = await self.get_store_summary()
        stores = summary.setdefault("stores", {})
        stores[store_name] = {"exists": False}
        await self.save(summary)
