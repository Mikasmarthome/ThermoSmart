"""Phase 3 tests for the HA store wrappers, using an injected fake factory."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.raw_schemas import RawTrackName
from custom_components.thermosmart.learning.storage.stores import (
    EpisodesStore,
    GlobalIndexStore,
    ModelStateStore,
    RawSegmentIndexStore,
    RawSegmentStore,
    StoreVersionError,
    ZoneMetadataStore,
)


class FakeStore:
    def __init__(self) -> None:
        self._data = None
        self.save_count = 0
        self.removed = False

    async def async_load(self):
        return self._data

    async def async_save(self, data):
        self._data = data
        self.save_count += 1

    async def async_remove(self):
        self._data = None
        self.removed = True


class FakeFactory:
    def __init__(self) -> None:
        self.stores: dict[str, FakeStore] = {}

    def create(self, key: str, version: int) -> FakeStore:
        return self.stores.setdefault(key, FakeStore())


class TestVersionedStore:
    async def test_load_none_when_absent(self):
        store = ZoneMetadataStore(FakeFactory(), "lz_1")
        assert await store.load() is None

    async def test_save_then_load_roundtrip(self):
        store = ZoneMetadataStore(FakeFactory(), "lz_1")
        await store.save({"learning_zone_id": "lz_1"})
        assert await store.load() == {"learning_zone_id": "lz_1"}

    async def test_version_envelope_applied(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        await store.save({"x": 1})
        raw = factory.stores[store.key]._data
        assert raw["store_schema_version"] == 1 and raw["data"] == {"x": 1}

    async def test_version_mismatch_not_swallowed(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        factory.stores[store.key]._data = {"store_schema_version": 999, "data": {}}
        with pytest.raises(StoreVersionError):
            await store.load()

    async def test_missing_version_rejected(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        factory.stores[store.key]._data = {"data": {}}
        with pytest.raises(StoreVersionError):
            await store.load()

    async def test_delete(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        await store.save({"x": 1})
        await store.delete()
        assert factory.stores[store.key].removed is True


class TestRawSegmentStore:
    async def test_segment_roundtrip(self):
        store = RawSegmentStore(FakeFactory(), "lz_1", RawTrackName.ROOM)
        await store.save_segment(0, {"records": []})
        assert await store.load_segment(0) == {"records": []}

    async def test_segments_have_distinct_keys(self):
        factory = FakeFactory()
        store = RawSegmentStore(factory, "lz_1", RawTrackName.ROOM)
        await store.save_segment(0, {"a": 1})
        await store.save_segment(1, {"b": 2})
        assert len(factory.stores) == 2


class TestIsolationAndAuthority:
    async def test_two_zones_isolated(self):
        factory = FakeFactory()
        a = ZoneMetadataStore(factory, "lz_1")
        b = ZoneMetadataStore(factory, "lz_2")
        await a.save({"zone": "a"})
        assert await b.load() is None  # different key, isolated

    async def test_multiple_tracks_per_zone_distinct(self):
        factory = FakeFactory()
        room_idx = RawSegmentIndexStore(factory, "lz_1", RawTrackName.ROOM)
        trv_idx = RawSegmentIndexStore(factory, "lz_1", RawTrackName.TRV)
        assert room_idx.key != trv_idx.key

    async def test_all_wrapper_kinds_constructible(self):
        factory = FakeFactory()
        EpisodesStore(factory, "lz_1")
        ModelStateStore(factory, "lz_1")
        GlobalIndexStore(factory)
        # no store created any data merely by construction
        assert all(s._data is None for s in factory.stores.values())
