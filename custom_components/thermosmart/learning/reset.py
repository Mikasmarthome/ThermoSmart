"""Purge of a zone's Learning storage on real removal (HA-dependent storage shell).

Uses an injectable store factory so it is testable without a running Home
Assistant. Performs no automatic work on import and never imports or modifies the
v1 store ``thermosmart_learning_data``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .storage import naming
from .storage.stores import (
    AdaptationHistoryStore,
    ApplicationLifecycleStore,
    EpisodesStore,
    ResearchDailyStore,
    StorageMetadataStore,
    StoreFactory,
    SupportCriticalEventStore,
)


@dataclass(frozen=True)
class ZonePurgeResult:
    """Outcome of purging one zone's Learning storage on real removal. Never raises."""
    deleted_keys: tuple[str, ...]
    errors: tuple[str, ...]


async def async_purge_zone_storage(factory: StoreFactory, entry_id: str) -> ZonePurgeResult:
    """Delete every Learning storage key the integration writes for one zone.

    Only ever called on REAL zone/entry removal (``async_remove_entry`` in
    ``__init__.py``) — never on unload/reload, which must never lose data.
    Every key deleted here is derived the same way it was created (via
    ``learning/storage/naming.py`` through the store wrapper classes) — no
    directory scan, no wildcard, no cross-zone reach. Best-effort per store:
    one failing delete never blocks the rest. Does not touch the v1 store
    ``thermosmart_learning_data`` (shared across zones — that is the caller's
    responsibility once it has confirmed no zone remains) nor the Learning
    shadow-state store (a different key scheme; see
    ``learning/runtime/ha_store.py``'s ``HomeAssistantStoreAdapter.async_delete()``).
    """
    lz = naming.validate_learning_zone_id(entry_id)
    deleted: list[str] = []
    errors: list[str] = []

    async def _delete(store, key: str) -> None:
        try:
            await store.delete()
            deleted.append(key)
        except Exception as err:
            errors.append(f"{key}: {type(err).__name__}")

    await _delete(AdaptationHistoryStore(factory, lz), naming.adaptation_history_key(lz))
    await _delete(ApplicationLifecycleStore(factory, lz), naming.application_lifecycle_key(lz))
    await _delete(EpisodesStore(factory, lz), naming.episodes_key(lz))
    await _delete(SupportCriticalEventStore(factory, lz), naming.support_critical_events_key(lz))
    await _delete(ResearchDailyStore(factory, lz), naming.research_daily_key(lz))
    # Storage-Metadata index: the whole per-zone index is gone once the zone
    # itself is gone — no per-store mark_store_missing() needed here.
    await _delete(StorageMetadataStore(factory, lz), naming.storage_metadata_key(lz))

    return ZonePurgeResult(deleted_keys=tuple(deleted), errors=tuple(errors))
