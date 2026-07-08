"""Learning Naming / Storage-Key-Migration audit — Commit 1 + Commit 2 tests.

Commit 1 (key-derivation groundwork, non-production-affecting):
  - the new neutral prefix (LEARNING_STORAGE_PREFIX) vs the legacy prefix
    every existing *_key() function still actually produces
  - LearningStorageKeyPair is computable for every store type
  - thermosmart_learning_data (the v1 legacy store) is untouched — it was
    already neutral and must never be renamed
  - the global-index and runtime-snapshot key pairs specifically

Every existing naming.*_key() function and ha_store.store_key() still
produce exactly their pre-existing thermosmart_le2__ values (those are the
legacy half of each pair now).

Commit 2 (the classes below `TestVersionedStoreMigration` onward): the actual
lazy read-migration wired into `learning/storage/stores.py`'s
`_VersionedStore` and `learning/runtime/ha_store.py`'s
`HomeAssistantStoreAdapter` — new key first, legacy key as read-fallback,
migrate-on-read, legacy never deleted except via explicit zone removal.
"""
from __future__ import annotations

from unittest.mock import patch

from custom_components.thermosmart.const import STORAGE_KEY
from custom_components.thermosmart.learning.raw_schemas import RawTrackName
from custom_components.thermosmart.learning.runtime import ha_store
from custom_components.thermosmart.learning.runtime.ha_store import HomeAssistantStoreAdapter
from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.storage.stores import (
    EpisodesStore,
    GlobalIndexStore,
    ModelStateStore,
    RawSegmentIndexStore,
    RawSegmentStore,
    StoreVersionError,
    ZoneMetadataStore,
)


class TestNeutralPrefixConstants:
    def test_neutral_prefix_has_no_generation_codename(self):
        assert naming.LEARNING_STORAGE_PREFIX == "thermosmart_learning"
        assert "le2" not in naming.LEARNING_STORAGE_PREFIX
        assert "le1" not in naming.LEARNING_STORAGE_PREFIX

    def test_legacy_prefix_constant_matches_current_production_prefix(self):
        assert naming.LEGACY_LEARNING_STORAGE_PREFIX == "thermosmart_le2"
        assert naming.container_key("lz_1").startswith(
            f"{naming.LEGACY_LEARNING_STORAGE_PREFIX}__"
        )


class TestExistingKeyFunctionsUnchanged:
    """Production behavior must be identical to before this commit."""

    def test_container_key_unchanged(self):
        assert naming.container_key("lz_1") == "thermosmart_le2__lz_1__container"

    def test_episodes_key_unchanged(self):
        assert naming.episodes_key("lz_1") == "thermosmart_le2__lz_1__episodes"

    def test_model_state_key_unchanged(self):
        assert naming.model_state_key("lz_1") == "thermosmart_le2__lz_1__models"

    def test_global_index_key_unchanged(self):
        assert naming.global_index_key() == "thermosmart_le2__global_index"

    def test_raw_index_key_unchanged(self):
        assert naming.raw_index_key("lz_1", RawTrackName.ROOM) == \
            "thermosmart_le2__lz_1__raw_room_index"

    def test_raw_segment_key_unchanged(self):
        assert naming.raw_segment_key("lz_1", RawTrackName.ROOM, 3) == \
            "thermosmart_le2__lz_1__raw_room_seg_3"

    def test_adaptation_history_key_unchanged(self):
        assert naming.adaptation_history_key("lz_1") == \
            "thermosmart_le2__lz_1__adaptation_history"

    def test_application_lifecycle_key_unchanged(self):
        assert naming.application_lifecycle_key("lz_1") == \
            "thermosmart_le2__lz_1__application_lifecycle"

    def test_support_critical_events_key_unchanged(self):
        assert naming.support_critical_events_key("lz_1") == \
            "thermosmart_le2__lz_1__support_critical_events"

    def test_research_daily_key_unchanged(self):
        assert naming.research_daily_key("lz_1") == \
            "thermosmart_le2__lz_1__research_daily"


class TestKeyPairsComputable:
    """Every store type must have a computable (current, legacy) pair —
    current always neutral, legacy always the existing production key."""

    def test_container_key_pair(self):
        pair = naming.container_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__container"
        assert pair.legacy == "thermosmart_le2__lz_1__container"

    def test_episodes_key_pair(self):
        pair = naming.episodes_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__episodes"
        assert pair.legacy == naming.episodes_key("lz_1")

    def test_model_state_key_pair(self):
        pair = naming.model_state_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__models"
        assert pair.legacy == naming.model_state_key("lz_1")

    def test_raw_index_key_pair(self):
        pair = naming.raw_index_key_pair("lz_1", RawTrackName.TRV)
        assert pair.current == "thermosmart_learning__lz_1__raw_trv_index"
        assert pair.legacy == naming.raw_index_key("lz_1", RawTrackName.TRV)

    def test_raw_segment_key_pair(self):
        pair = naming.raw_segment_key_pair("lz_1", RawTrackName.TRV, 7)
        assert pair.current == "thermosmart_learning__lz_1__raw_trv_seg_7"
        assert pair.legacy == naming.raw_segment_key("lz_1", RawTrackName.TRV, 7)

    def test_adaptation_history_key_pair(self):
        pair = naming.adaptation_history_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__adaptation_history"
        assert pair.legacy == naming.adaptation_history_key("lz_1")

    def test_application_lifecycle_key_pair(self):
        pair = naming.application_lifecycle_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__application_lifecycle"
        assert pair.legacy == naming.application_lifecycle_key("lz_1")

    def test_support_critical_events_key_pair(self):
        pair = naming.support_critical_events_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__support_critical_events"
        assert pair.legacy == naming.support_critical_events_key("lz_1")

    def test_research_daily_key_pair(self):
        pair = naming.research_daily_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__research_daily"
        assert pair.legacy == naming.research_daily_key("lz_1")

    def test_global_index_key_pair(self):
        pair = naming.global_index_key_pair()
        assert pair.current == "thermosmart_learning__global_index"
        assert pair.legacy == "thermosmart_le2__global_index"

    def test_no_current_key_contains_legacy_codename(self):
        pairs = [
            naming.container_key_pair("lz_1"),
            naming.episodes_key_pair("lz_1"),
            naming.model_state_key_pair("lz_1"),
            naming.adaptation_history_key_pair("lz_1"),
            naming.application_lifecycle_key_pair("lz_1"),
            naming.support_critical_events_key_pair("lz_1"),
            naming.research_daily_key_pair("lz_1"),
            naming.global_index_key_pair(),
        ]
        for pair in pairs:
            assert "le2" not in pair.current
            assert pair.legacy is not None and "le2" in pair.legacy


class TestLegacyStoreDataKeyUntouched:
    """thermosmart_learning_data (v1) is already neutral and must never be
    renamed by this migration — no le1/le2/v1/v2 codename in it."""

    def test_storage_key_constant_unchanged(self):
        assert STORAGE_KEY == "thermosmart_learning_data"

    def test_storage_key_has_no_generation_codename(self):
        for token in ("le1", "le2", "v1", "v2", "legacy"):
            assert token not in STORAGE_KEY

    def test_global_index_key_pair_is_distinct_from_legacy_v1_store(self):
        pair = naming.global_index_key_pair()
        assert pair.current != STORAGE_KEY
        assert pair.legacy != STORAGE_KEY


class TestRuntimeSnapshotKeyPair:
    """ha_store.py's runtime-snapshot key (hashed zone segment), mirrored via
    its own store_key_pair() rather than naming.py's suffix-based pairs."""

    def test_store_key_unchanged(self):
        assert ha_store.store_key("abc123") == "thermosmart_le2__abc123"

    def test_new_prefix_constant_has_no_generation_codename(self):
        assert ha_store.NEW_STORE_KEY_PREFIX == "thermosmart_learning__"

    def test_store_key_pair(self):
        pair = ha_store.store_key_pair("abc123")
        assert pair.current == "thermosmart_learning__abc123"
        assert pair.legacy == "thermosmart_le2__abc123"

    def test_store_key_pair_empty_segment_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            ha_store.store_key_pair("")


class TestZoneSegmentConsolidation:
    """naming.zone_segment() is the single source of truth for the hash used
    by both ha_integration.py's _zone_segment() and __init__.py's cleanup
    path — same algorithm, same output for any given input."""

    def test_zone_segment_deterministic(self):
        assert naming.zone_segment("entry_abc") == naming.zone_segment("entry_abc")

    def test_zone_segment_differs_per_zone(self):
        assert naming.zone_segment("entry_abc") != naming.zone_segment("entry_xyz")

    def test_zone_segment_is_16_hex_chars(self):
        seg = naming.zone_segment("entry_abc")
        assert len(seg) == 16
        int(seg, 16)  # raises ValueError if not valid hex

    def test_zone_segment_matches_known_sha256_prefix(self):
        import hashlib
        expected = hashlib.sha256("entry_abc".encode("utf-8")).hexdigest()[:16]
        assert naming.zone_segment("entry_abc") == expected

    def test_store_key_pair_uses_same_hash_as_zone_segment(self):
        seg = naming.zone_segment("entry_abc")
        pair = ha_store.store_key_pair(seg)
        assert pair.legacy == ha_store.store_key(naming.zone_segment("entry_abc"))


# ── Commit 2: actual lazy read-migration behavior ────────────────────────────

class FakeStore:
    def __init__(self):
        self.data = None
        self.saves = 0
        self.loads = 0
        self.removed = False

    async def async_load(self):
        self.loads += 1
        return self.data

    async def async_save(self, data):
        self.data = data
        self.saves += 1

    async def async_remove(self):
        self.data = None
        self.removed = True


class FakeFactory:
    """Matches learning/storage/stores.py's StoreFactory protocol."""

    def __init__(self):
        self.stores: dict[str, FakeStore] = {}

    def create(self, key: str, version: int) -> FakeStore:
        return self.stores.setdefault(key, FakeStore())


class TestVersionedStoreMigration:
    """learning/storage/stores.py's _VersionedStore, exercised through
    ZoneMetadataStore (any *_key_pair()-based subclass behaves identically)."""

    async def test_new_key_present_loads_new_without_touching_legacy(self):
        factory = FakeFactory()
        pair = naming.container_key_pair("lz_1")
        factory.stores[pair.current] = FakeStore()
        factory.stores[pair.current].data = {"store_schema_version": 1, "data": {"v": "new"}}
        store = ZoneMetadataStore(factory, "lz_1")

        result = await store.load()

        assert result == {"v": "new"}
        assert factory.stores[pair.legacy].loads == 0  # legacy never even read

    async def test_new_key_missing_legacy_present_loads_legacy_and_writes_new(self):
        factory = FakeFactory()
        pair = naming.container_key_pair("lz_1")
        factory.stores[pair.legacy] = FakeStore()
        factory.stores[pair.legacy].data = {"store_schema_version": 1, "data": {"v": "old"}}
        store = ZoneMetadataStore(factory, "lz_1")

        result = await store.load()

        assert result == {"v": "old"}
        assert factory.stores[pair.current].data == {"store_schema_version": 1, "data": {"v": "old"}}
        assert factory.stores[pair.legacy].removed is False  # legacy NOT deleted

    async def test_migration_is_idempotent(self):
        factory = FakeFactory()
        pair = naming.container_key_pair("lz_1")
        factory.stores[pair.legacy] = FakeStore()
        factory.stores[pair.legacy].data = {"store_schema_version": 1, "data": {"v": "old"}}
        store = ZoneMetadataStore(factory, "lz_1")

        first = await store.load()
        saves_after_first = factory.stores[pair.current].saves
        second = await store.load()  # new key now present -> legacy never read again

        assert first == second == {"v": "old"}
        assert factory.stores[pair.current].saves == saves_after_first  # no duplicate write

    async def test_both_keys_absent_returns_none(self):
        factory = FakeFactory()
        store = ZoneMetadataStore(factory, "lz_1")
        assert await store.load() is None

    async def test_new_writes_use_new_key_only(self):
        factory = FakeFactory()
        pair = naming.container_key_pair("lz_1")
        store = ZoneMetadataStore(factory, "lz_1")

        await store.save({"v": "fresh"})

        assert factory.stores[pair.current].data == {"store_schema_version": 1, "data": {"v": "fresh"}}
        assert factory.stores[pair.legacy].data is None
        assert factory.stores[pair.legacy].saves == 0

    async def test_new_key_invalid_does_not_silently_fall_back_to_legacy(self):
        """A version mismatch on the new key must raise, not be swallowed by
        quietly returning legacy data instead — conservative-by-design."""
        factory = FakeFactory()
        pair = naming.container_key_pair("lz_1")
        factory.stores[pair.current] = FakeStore()
        factory.stores[pair.current].data = {"store_schema_version": 999, "data": {}}
        factory.stores[pair.legacy] = FakeStore()
        factory.stores[pair.legacy].data = {"store_schema_version": 1, "data": {"v": "old"}}
        store = ZoneMetadataStore(factory, "lz_1")

        import pytest
        with pytest.raises(StoreVersionError):
            await store.load()

    async def test_delete_removes_both_new_and_legacy(self):
        factory = FakeFactory()
        pair = naming.container_key_pair("lz_1")
        store = ZoneMetadataStore(factory, "lz_1")
        await store.save({"v": 1})

        await store.delete()

        assert factory.stores[pair.current].removed is True
        assert factory.stores[pair.legacy].removed is True

    async def test_multi_zone_migration_does_not_cross_contaminate(self):
        """Migrating zone A's legacy data must never populate zone B's keys."""
        factory = FakeFactory()
        pair_a = naming.container_key_pair("lz_a")
        pair_b = naming.container_key_pair("lz_b")
        factory.stores[pair_a.legacy] = FakeStore()
        factory.stores[pair_a.legacy].data = {"store_schema_version": 1, "data": {"zone": "a"}}

        store_a = ZoneMetadataStore(factory, "lz_a")
        store_b = ZoneMetadataStore(factory, "lz_b")

        result_a = await store_a.load()
        result_b = await store_b.load()

        assert result_a == {"zone": "a"}
        assert result_b is None
        assert factory.stores[pair_b.current].data is None
        assert factory.stores[pair_b.legacy].data is None


class TestEachStoreTypeMigrates:
    """One migration assertion per remaining store type sharing _VersionedStore."""

    async def test_episodes_old_to_new(self):
        factory = FakeFactory()
        pair = naming.episodes_key_pair("lz_1")
        factory.stores[pair.legacy] = FakeStore()
        factory.stores[pair.legacy].data = {"store_schema_version": 1, "data": {"e": 1}}
        store = EpisodesStore(factory, "lz_1")
        assert await store.load() == {"e": 1}
        assert factory.stores[pair.current].data["data"] == {"e": 1}

    async def test_models_old_to_new(self):
        factory = FakeFactory()
        pair = naming.model_state_key_pair("lz_1")
        factory.stores[pair.legacy] = FakeStore()
        factory.stores[pair.legacy].data = {"store_schema_version": 1, "data": {"m": 1}}
        store = ModelStateStore(factory, "lz_1")
        assert await store.load() == {"m": 1}
        assert factory.stores[pair.current].data["data"] == {"m": 1}

    async def test_global_index_old_to_new(self):
        factory = FakeFactory()
        pair = naming.global_index_key_pair()
        factory.stores[pair.legacy] = FakeStore()
        factory.stores[pair.legacy].data = {"store_schema_version": 1, "data": {"g": 1}}
        store = GlobalIndexStore(factory)
        assert await store.load() == {"g": 1}
        assert factory.stores[pair.current].data["data"] == {"g": 1}

    async def test_raw_index_old_to_new(self):
        factory = FakeFactory()
        pair = naming.raw_index_key_pair("lz_1", RawTrackName.ROOM)
        factory.stores[pair.legacy] = FakeStore()
        factory.stores[pair.legacy].data = {"store_schema_version": 1, "data": {"i": 1}}
        store = RawSegmentIndexStore(factory, "lz_1", RawTrackName.ROOM)
        assert await store.load() == {"i": 1}
        assert factory.stores[pair.current].data["data"] == {"i": 1}

    async def test_raw_segment_old_to_new(self):
        factory = FakeFactory()
        pair = naming.raw_segment_key_pair("lz_1", RawTrackName.ROOM, 0)
        factory.stores[pair.legacy] = FakeStore()
        factory.stores[pair.legacy].data = {"store_schema_version": 1, "data": {"s": 1}}
        store = RawSegmentStore(factory, "lz_1", RawTrackName.ROOM)
        assert await store.load_segment(0) == {"s": 1}
        assert factory.stores[pair.current].data["data"] == {"s": 1}


class TestRuntimeSnapshotAdapterMigration:
    """HomeAssistantStoreAdapter — the runtime-snapshot store (hashed zone
    segment, no naming.py suffix). Patches homeassistant.helpers.storage.Store
    so both the new and legacy Store objects are fakes under test control."""

    def _patched_store_map(self):
        stores: dict[str, FakeStore] = {}

        def _factory(hass, version, key):
            return stores.setdefault(key, FakeStore())

        return stores, _factory

    async def test_new_key_present_loads_new(self):
        stores, factory_fn = self._patched_store_map()
        pair = ha_store.store_key_pair("seg1")
        with patch("homeassistant.helpers.storage.Store", side_effect=factory_fn):
            adapter = HomeAssistantStoreAdapter(hass=None, zone_segment="seg1")
        stores[pair.current].data = {"v": "new"}

        assert await adapter.async_load() == {"v": "new"}
        assert pair.legacy not in stores or stores[pair.legacy].data is None

    async def test_new_key_missing_legacy_present_migrates(self):
        stores, factory_fn = self._patched_store_map()
        pair = ha_store.store_key_pair("seg1")
        with patch("homeassistant.helpers.storage.Store", side_effect=factory_fn):
            adapter = HomeAssistantStoreAdapter(hass=None, zone_segment="seg1")
        stores[pair.legacy].data = {"v": "old"}

        result = await adapter.async_load()

        assert result == {"v": "old"}
        assert stores[pair.current].data == {"v": "old"}
        assert stores[pair.legacy].removed is False

    async def test_both_absent_returns_none(self):
        stores, factory_fn = self._patched_store_map()
        with patch("homeassistant.helpers.storage.Store", side_effect=factory_fn):
            adapter = HomeAssistantStoreAdapter(hass=None, zone_segment="seg1")
        assert await adapter.async_load() is None

    async def test_delete_removes_both(self):
        stores, factory_fn = self._patched_store_map()
        pair = ha_store.store_key_pair("seg1")
        with patch("homeassistant.helpers.storage.Store", side_effect=factory_fn):
            adapter = HomeAssistantStoreAdapter(hass=None, zone_segment="seg1")
        await adapter.async_save({"v": 1})

        await adapter.async_delete()

        assert stores[pair.current].removed is True
        assert stores[pair.legacy].removed is True

    async def test_explicit_store_override_skips_legacy_entirely(self):
        """The store= override (used by shadow-controller tests) must behave
        exactly as before — no legacy store created, no migration attempted."""
        fake = FakeStore()
        adapter = HomeAssistantStoreAdapter(hass=None, zone_segment="seg1", store=fake)
        await adapter.async_save({"x": 1})
        assert fake.saves == 1
        assert await adapter.async_load() == {"x": 1}
