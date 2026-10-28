"""The legacy v1 store is still purged on the last zone removal (S7.0).

``thermosmart_learning_data`` is the shared Learning-Engine v1 store. It is
deleted only when the last ThermoSmart zone is removed, and that deletion must
not depend on a live LearningEngine instance (it is a plain Home Assistant
Store removal by key). This pins that contract across the LE v1 removal: the
per-zone purge and the runtime-snapshot adapter are patched out here because
they are covered by their own tests; the focus is the v1 key.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components import thermosmart as ts_init
from custom_components.thermosmart.const import DOMAIN as _DOMAIN
from custom_components.thermosmart.const import STORAGE_KEY


class _FakeStore:
    created: list[str] = []
    removed: list[str] = []

    def __init__(self, hass, version, key):
        self._key = key
        _FakeStore.created.append(key)

    async def async_remove(self):
        _FakeStore.removed.append(self._key)


def _patches():
    """Patch the two separately-tested storage paths and the raw HA Store."""
    return (
        patch("homeassistant.helpers.storage.Store", _FakeStore),
        patch(
            "custom_components.thermosmart.learning.reset.async_purge_zone_storage",
            AsyncMock(return_value=SimpleNamespace(errors=())),
        ),
        patch(
            "custom_components.thermosmart.learning.runtime.ha_store.HomeAssistantStoreAdapter",
            MagicMock(return_value=SimpleNamespace(async_delete=AsyncMock())),
        ),
    )


async def _run(*, last_zone: bool):
    _FakeStore.created = []
    _FakeStore.removed = []
    hass = MagicMock()
    # Deliberately no "learning_engine" in hass.data — the v1 purge must not need it.
    hass.data = {_DOMAIN: {}}
    p1, p2, p3 = _patches()
    with p1, p2, p3:
        await ts_init._async_purge_learning_zone_storage(
            hass, "zone_1", last_zone=last_zone
        )
    return hass


class TestV1LegacyPurge:
    async def test_last_zone_removes_v1_store(self):
        await _run(last_zone=True)
        assert STORAGE_KEY == "thermosmart_learning_data"
        assert STORAGE_KEY in _FakeStore.removed

    async def test_not_last_zone_keeps_v1_store(self):
        await _run(last_zone=False)
        assert STORAGE_KEY not in _FakeStore.removed

    async def test_no_engine_instance_required(self):
        hass = await _run(last_zone=True)
        assert "learning_engine" not in hass.data[_DOMAIN]
