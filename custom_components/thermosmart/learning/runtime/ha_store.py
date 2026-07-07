"""Home Assistant Store adapter for the LE 2.0 persistence protocol.

This is the only LE-2.0 runtime module that touches Home Assistant, and it does
so lazily (HA is imported inside the methods) so the pure runtime stays importable
on plain Python. It adapts HA's ``Store`` to the injected ``AsyncStore`` protocol
used by :class:`PersistenceOrchestrator`. Store keys stay under the
``thermosmart_le2__`` namespace and never embed entity names.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

STORE_KEY_PREFIX = "thermosmart_le2__"
STORE_VERSION = 2


def store_key(zone_segment: str) -> str:
    """Build a namespaced, name-free store key for a zone segment.

    ``zone_segment`` must already be a non-identifying token (e.g. a stable hash
    of the learning zone id), never an entity name.
    """
    if not zone_segment:
        raise ValueError("zone_segment must be non-empty")
    return f"{STORE_KEY_PREFIX}{zone_segment}"


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
