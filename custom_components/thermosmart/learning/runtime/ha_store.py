"""Home Assistant Store adapter for the LE 2.0 persistence protocol.

This is the only LE-2.0 runtime module that touches Home Assistant, and it does
so lazily (HA is imported inside the methods) so the pure runtime stays importable
on plain Python. It adapts HA's ``Store`` to the injected ``AsyncStore`` protocol
used by :class:`PersistenceOrchestrator`. Store keys stay under the
``thermosmart_le2__`` namespace and never embed entity names.

Neutral naming migration (Learning Naming / Storage-Key audit): NEW_STORE_KEY_PREFIX
is the target neutral prefix. STORE_KEY_PREFIX (unchanged) is what store_key() still
actually produces — production behavior here is unaffected by this addition. See
learning/storage/naming.py's module docstring for the full migration rationale;
store_key_pair() mirrors that module's LearningStorageKeyPair pattern for this
runtime-snapshot key specifically.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

STORE_KEY_PREFIX = "thermosmart_le2__"
NEW_STORE_KEY_PREFIX = "thermosmart_learning__"
STORE_VERSION = 2


def store_key(zone_segment: str) -> str:
    """Build a namespaced, name-free store key for a zone segment.

    ``zone_segment`` must already be a non-identifying token (e.g. a stable hash
    of the learning zone id), never an entity name.
    """
    if not zone_segment:
        raise ValueError("zone_segment must be non-empty")
    return f"{STORE_KEY_PREFIX}{zone_segment}"


def store_key_pair(zone_segment: str):
    """Target (neutral) and legacy runtime-snapshot key for one zone segment.

    Returns a ``learning.storage.naming.LearningStorageKeyPair`` — imported
    lazily here to avoid a module-level dependency from this HA-only adapter
    onto the pure storage-naming module (mirrors this file's existing lazy-HA-
    import style, just in the other direction). Does not change what
    ``store_key()``/``HomeAssistantStoreAdapter`` actually use.
    """
    from ..storage.naming import LearningStorageKeyPair
    if not zone_segment:
        raise ValueError("zone_segment must be non-empty")
    return LearningStorageKeyPair(
        current=f"{NEW_STORE_KEY_PREFIX}{zone_segment}",
        legacy=store_key(zone_segment),
    )


class HomeAssistantStoreAdapter:
    """Wraps an injected HA ``Store`` (or any object with async_load/async_save)."""

    def __init__(self, hass: Any, zone_segment: str, *, store: Any = None) -> None:
        self._key = store_key(zone_segment)
        if store is not None:
            self._store = store
        else:
            from homeassistant.helpers.storage import Store  # lazy: HA-only path
            self._store = Store(hass, STORE_VERSION, self._key)

    @property
    def key(self) -> str:
        return self._key

    async def async_load(self) -> Optional[Mapping[str, Any]]:
        return await self._store.async_load()

    async def async_save(self, data: Mapping[str, Any]) -> None:
        await self._store.async_save(dict(data))

    async def async_delete(self) -> None:
        """Delete this zone's shadow-state store via HA's safe Store.async_remove().

        Exact known key only — no directory scan, no wildcard. Only ever
        called on real zone/entry removal (async_remove_entry), never on
        unload/reload (which must never lose data).
        """
        await self._store.async_remove()
