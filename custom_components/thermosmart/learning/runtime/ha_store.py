"""Home Assistant Store adapter for the LE 2.0 persistence protocol.

This is the only LE-2.0 runtime module that touches Home Assistant, and it does
so lazily (HA is imported inside the methods) so the pure runtime stays importable
on plain Python. It adapts HA's ``Store`` to the injected ``AsyncStore`` protocol
used by :class:`PersistenceOrchestrator`. Store keys never embed entity names.

Neutral naming migration (Learning Naming / Storage-Key audit): new saves go
under ``NEW_STORE_KEY_PREFIX`` (``thermosmart_learning__``).
``HomeAssistantStoreAdapter`` reads that key first and falls back to the
legacy ``STORE_KEY_PREFIX`` (``thermosmart_le2__``) key on a lazy, one-time
migration — see the class docstring. ``store_key()`` (legacy-only) is kept
for the legacy key derivation ``store_key_pair()`` uses internally; no
production code calls ``store_key()`` directly anymore.
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
    """Wraps an injected HA ``Store`` (or any object with async_load/async_save).

    Reads/writes the neutral ``thermosmart_learning__`` key, with a
    read-fallback to the legacy ``thermosmart_le2__`` key: if the neutral key
    has no data yet, the legacy key is tried and (if valid) its data is
    migrated onto the neutral key on this same read — the legacy key itself
    is left in place as a safety fallback, only removed via ``async_delete()``
    (explicit zone removal).
    """

    def __init__(self, hass: Any, zone_segment: str, *, store: Any = None) -> None:
        pair = store_key_pair(zone_segment)
        self._key = pair.current
        self._legacy_key = pair.legacy
        if store is not None:
            self._store = store
            self._legacy_store = None
        else:
            from homeassistant.helpers.storage import Store  # lazy: HA-only path
            self._store = Store(hass, STORE_VERSION, self._key)
            self._legacy_store = Store(hass, STORE_VERSION, self._legacy_key)

    @property
    def key(self) -> str:
        return self._key

    async def async_load(self) -> Optional[Mapping[str, Any]]:
        data = await self._store.async_load()
        if data is not None:
            return data
        if self._legacy_store is None:
            return None
        legacy_data = await self._legacy_store.async_load()
        if legacy_data is None:
            return None
        # Lazy migration: persist under the neutral key now that it's been
        # read once. Legacy key is a deliberate safety fallback — never
        # deleted here (only on explicit zone removal, see async_delete()).
        await self.async_save(legacy_data)
        return legacy_data

    async def async_save(self, data: Mapping[str, Any]) -> None:
        await self._store.async_save(dict(data))

    async def async_delete(self) -> None:
        """Delete this zone's shadow-state store (both key variants) via HA's
        safe Store.async_remove().

        Exact known keys only — no directory scan, no wildcard. Only ever
        called on real zone/entry removal (async_remove_entry), never on
        unload/reload (which must never lose data).
        """
        try:
            await self._store.async_remove()
        finally:
            if self._legacy_store is not None:
                await self._legacy_store.async_remove()
