"""Storage-Metadata concept — Commit A: foundation tests.

Covers the new StorageMetadataStore type, its naming.py key-pair helper, and
the additive (opt-in, backward-compatible) envelope timestamp stamp on
_VersionedStore.save(). No save-path wiring, no Support/Research export
changes — see module docstrings in stores.py for the full Commit A/B/C/D
breakdown this belongs to.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.storage.stores import (
    EpisodesStore,
    StorageKeyState,
    StorageMetadataStore,
    StorageWriteReason,
    StoreVersionError,
    ZoneMetadataStore,
)

_T0 = datetime(2026, 7, 7, 5, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 7, 7, 5, 10, 0, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self):
        self.data = None
        self.saves = 0
        self.removed = False

    async def async_load(self):
        return self.data

    async def async_save(self, data):
        self.data = data
        self.saves += 1

    async def async_remove(self):
        self.data = None
        self.removed = True


class FakeFactory:
    def __init__(self):
        self.stores: dict[str, FakeStore] = {}

    def create(self, key: str, version: int) -> FakeStore:
        return self.stores.setdefault(key, FakeStore())


# ── naming.py: storage_metadata_key / storage_metadata_key_pair ─────────────

class TestStorageMetadataKeyPair:
    def test_current_key_is_neutral(self):
        pair = naming.storage_metadata_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__storage_metadata"

    def test_legacy_key_uses_legacy_prefix(self):
        pair = naming.storage_metadata_key_pair("lz_1")
        assert pair.legacy == "thermosmart_le2__lz_1__storage_metadata"

    def test_plain_key_function_matches_pair_legacy(self):
        assert naming.storage_metadata_key("lz_1") == naming.storage_metadata_key_pair("lz_1").legacy

    def test_keys_differ_per_zone(self):
        a = naming.storage_metadata_key_pair("lz_a")
        b = naming.storage_metadata_key_pair("lz_b")
        assert a.current != b.current
        assert a.legacy != b.legacy

    def test_forbidden_separator_rejected(self):
        with pytest.raises(naming.StorageNamingError):
            naming.storage_metadata_key_pair("a__b")


# ── StorageMetadataStore: basic load/save behavior ──────────────────────────

class TestStorageMetadataStoreBasics:
    async def test_missing_store_returns_empty_v1_shell(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        summary = await store.get_store_summary()
        assert summary == {"schema_version": 1, "stores": {}}

    async def test_save_and_load_roundtrip(self):
        factory = FakeFactory()
        store = StorageMetadataStore(factory, "lz_1")
        await store.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T0)

        store2 = StorageMetadataStore(factory, "lz_1")
        summary = await store2.get_store_summary()
        assert summary["schema_version"] == 1
        assert summary["stores"]["episodes"]["exists"] is True

    async def test_corrupt_index_raises_but_does_not_touch_other_stores(self):
        factory = FakeFactory()
        meta_store = StorageMetadataStore(factory, "lz_1")
        pair = naming.storage_metadata_key_pair("lz_1")
        # Mutate the SAME FakeStore instance the store already holds a
        # reference to — replacing the dict entry would leave meta_store
        # pointing at the old (stale) object instead.
        factory.stores[pair.current].data = {"store_schema_version": 999, "data": {}}

        with pytest.raises(StoreVersionError):
            await meta_store.get_store_summary()

        # A real data store under a completely different key is unaffected.
        episodes = EpisodesStore(factory, "lz_1")
        assert await episodes.load() is None


# ── mark_store_written: exists/created/updated/reason/state ─────────────────

class TestMarkStoreWritten:
    async def test_sets_exists_true(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written("models", StorageWriteReason.MODEL_UPDATE, _T0)
        summary = await store.get_store_summary()
        assert summary["stores"]["models"]["exists"] is True

    async def test_created_at_utc_stable_across_updates(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T0)
        await store.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T1)
        summary = await store.get_store_summary()
        assert summary["stores"]["episodes"]["created_at_utc"] == "2026-07-07T05:00:00Z"

    async def test_updated_at_utc_changes_on_second_write(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T0)
        await store.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T1)
        summary = await store.get_store_summary()
        assert summary["stores"]["episodes"]["updated_at_utc"] == "2026-07-07T05:10:00Z"

    async def test_last_write_reason_updates(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written("research_daily", StorageWriteReason.RESEARCH_DAILY_UPDATE, _T0)
        await store.mark_store_written("research_daily", StorageWriteReason.MIGRATION, _T1)
        summary = await store.get_store_summary()
        assert summary["stores"]["research_daily"]["last_write_reason"] == "migration"

    async def test_storage_key_state_recorded(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written(
            "container", StorageWriteReason.UNSPECIFIED, _T0,
            storage_key_state=StorageKeyState.MIGRATED_FROM_LEGACY,
        )
        summary = await store.get_store_summary()
        assert summary["stores"]["container"]["storage_key_state"] == "migrated_from_legacy"

    async def test_storage_key_state_defaults_to_current(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written("models", StorageWriteReason.MODEL_UPDATE, _T0)
        summary = await store.get_store_summary()
        assert summary["stores"]["models"]["storage_key_state"] == "current"

    async def test_unknown_store_name_accepted_without_error(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written("some_future_store", StorageWriteReason.UNSPECIFIED, _T0)
        summary = await store.get_store_summary()
        assert summary["stores"]["some_future_store"]["exists"] is True

    async def test_multiple_stores_tracked_independently(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T0)
        await store.mark_store_written("models", StorageWriteReason.MODEL_UPDATE, _T1)
        summary = await store.get_store_summary()
        assert set(summary["stores"]) == {"episodes", "models"}
        assert summary["stores"]["episodes"]["updated_at_utc"] == "2026-07-07T05:00:00Z"
        assert summary["stores"]["models"]["updated_at_utc"] == "2026-07-07T05:10:00Z"


class TestMarkStoreMissing:
    async def test_sets_exists_false(self):
        store = StorageMetadataStore(FakeFactory(), "lz_1")
        await store.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T0)
        await store.mark_store_missing("episodes")
        summary = await store.get_store_summary()
        assert summary["stores"]["episodes"] == {"exists": False}


# ── delete() removes both key variants ───────────────────────────────────────

class TestStorageMetadataDelete:
    async def test_delete_removes_current_and_legacy(self):
        factory = FakeFactory()
        store = StorageMetadataStore(factory, "lz_1")
        await store.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T0)
        pair = naming.storage_metadata_key_pair("lz_1")

        await store.delete()

        assert factory.stores[pair.current].removed is True
        assert factory.stores[pair.legacy].removed is True


# ── Multi-zone isolation ──────────────────────────────────────────────────

class TestMultiZoneIsolation:
    async def test_zone_a_and_zone_b_have_separate_indexes(self):
        factory = FakeFactory()
        store_a = StorageMetadataStore(factory, "lz_a")
        store_b = StorageMetadataStore(factory, "lz_b")

        await store_a.mark_store_written("episodes", StorageWriteReason.EPISODE_CLOSED, _T0)

        summary_a = await store_a.get_store_summary()
        summary_b = await store_b.get_store_summary()
        assert summary_a["stores"] == {"episodes": summary_a["stores"]["episodes"]}
        assert summary_b["stores"] == {}


# ── _VersionedStore.save() additive envelope stamp ───────────────────────────

class TestVersionedStoreEnvelopeStamp:
    async def test_no_now_utc_leaves_envelope_unchanged(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        await store.save({"x": 1})
        raw = factory.stores[store.key].data
        assert "created_at_utc" not in raw
        assert "updated_at_utc" not in raw
        assert raw == {"store_schema_version": 1, "data": {"x": 1}}

    async def test_now_utc_stamps_created_and_updated(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        await store.save({"x": 1}, now_utc=_T0)
        raw = factory.stores[store.key].data
        assert raw["created_at_utc"] == "2026-07-07T05:00:00Z"
        assert raw["updated_at_utc"] == "2026-07-07T05:00:00Z"
        assert raw["data"] == {"x": 1}

    async def test_created_at_utc_stable_across_stamped_saves(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        await store.save({"x": 1}, now_utc=_T0)
        await store.save({"x": 2}, now_utc=_T1)
        raw = factory.stores[store.key].data
        assert raw["created_at_utc"] == "2026-07-07T05:00:00Z"
        assert raw["updated_at_utc"] == "2026-07-07T05:10:00Z"

    async def test_old_payload_without_timestamps_still_loads(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        factory.stores[store.key].data = {"store_schema_version": 1, "data": {"legacy": True}}
        assert await store.load() == {"legacy": True}

    async def test_unstamped_save_after_stamped_save_does_not_crash(self):
        """Mixed usage (some callers pass now_utc, some don't) must never
        corrupt the envelope — an unstamped save simply omits the fields
        going forward; it must not raise trying to read a prior stamp."""
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        await store.save({"x": 1}, now_utc=_T0)
        await store.save({"x": 2})  # no now_utc this time
        raw = factory.stores[store.key].data
        assert raw["data"] == {"x": 2}
        assert "created_at_utc" not in raw  # this save didn't opt in
