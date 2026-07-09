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

from custom_components.thermosmart.learning import reset
from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.init_state import ResetReason
from custom_components.thermosmart.learning.raw_schemas import RawTrackName, RoomObservation
from custom_components.thermosmart.learning.registry import (
    EpisodeRegistry,
    ModelRegistry,
    PrivacyClass,
    RawTrackDefinition,
    RawTrackRegistry,
    RetentionPolicy,
)
from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.storage.stores import (
    AdaptationHistoryStore,
    ApplicationLifecycleStore,
    EpisodesStore,
    ResearchDailyStore,
    StorageKeyState,
    StorageMetadataStore,
    StorageWriteReason,
    StoreVersionError,
    SupportCriticalEventStore,
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


# ── Commit B: reset.py wiring (init/reset/purge mark the metadata index) ────

def _raw_registry() -> RawTrackRegistry:
    reg = RawTrackRegistry()
    reg.register(RawTrackDefinition(
        track_name=RawTrackName.ROOM, schema_type=RoomObservation, track_schema_version=1,
        privacy_class=PrivacyClass.LOW, allowed_export_scopes=(ExportScope.SUPPORT,),
        retention=RetentionPolicy(), segmented=True, immutable_events=False))
    return reg


def _registries() -> dict:
    return {
        "raw_registry": _raw_registry(),
        "episode_registry": EpisodeRegistry(),
        "model_registry": ModelRegistry(),
    }


class TestResetWiringMarksWritten:
    async def test_ensure_v2_initialized_marks_container_models_episodes_raw_index(self):
        factory = FakeFactory()
        clock = FakeClock(_T0)
        result = await reset.ensure_v2_initialized(
            factory, "lz_1", clock=clock, **_registries())
        assert result.status.value == "initialized"

        meta = StorageMetadataStore(factory, "lz_1")
        summary = await meta.get_store_summary()
        stores = summary["stores"]
        for name in ("container", "models", "episodes", "raw_index:room"):
            assert stores[name]["exists"] is True
            assert stores[name]["last_write_reason"] == "reset"
            assert stores[name]["created_at_utc"] == "2026-07-07T05:00:00Z"


class TestResetAsymmetryWiring:
    async def test_reset_marks_deleted_stores_missing_but_leaves_support_research_untouched(self):
        factory = FakeFactory()
        clock = FakeClock(_T0)
        await reset.ensure_v2_initialized(factory, "lz_1", clock=clock, **_registries())

        # Simulate Support/Research having been written by production runtime —
        # reset_v2_learning_state must not touch these (pre-existing asymmetry).
        meta = StorageMetadataStore(factory, "lz_1")
        await meta.mark_store_written(
            "support_critical_events", StorageWriteReason.SUPPORT_EVENT_APPEND, _T0)
        await meta.mark_store_written(
            "research_daily", StorageWriteReason.RESEARCH_DAILY_UPDATE, _T0)

        clock.set_utc(_T1)
        await reset.reset_v2_learning_state(
            factory, "lz_1", ResetReason.TEST, clock=clock, reinitialize=False,
            **_registries())

        summary = await meta.get_store_summary()
        stores = summary["stores"]
        for name in ("container", "models", "episodes", "raw_index:room"):
            assert stores[name]["exists"] is False
        # Untouched by this reset path — still reflects the earlier writes.
        assert stores["support_critical_events"]["exists"] is True
        assert stores["research_daily"]["exists"] is True


class TestResetMetadataResilience:
    async def test_corrupt_metadata_index_does_not_break_init(self):
        """A pre-corrupted Storage-Metadata envelope must never block real
        component creation during ensure_v2_initialized (mirrors
        TestHaIntegrationSaveWiring.test_metadata_failure_does_not_break_real_save,
        for the reset.py/init wiring path instead of ha_integration.py)."""
        factory = FakeFactory()
        pair = naming.storage_metadata_key_pair("lz_1")
        factory.stores[pair.current] = FakeStore()
        factory.stores[pair.current].data = {"store_schema_version": 999, "data": {}}

        clock = FakeClock(_T0)
        result = await reset.ensure_v2_initialized(
            factory, "lz_1", clock=clock, **_registries())

        assert result.status.value == "initialized"
        episodes = EpisodesStore(factory, "lz_1")
        assert await episodes.load() == {"episodes": {}}


class TestPurgeWiringDeletesMetadataIndex:
    async def test_async_purge_zone_storage_deletes_storage_metadata_store(self):
        factory = FakeFactory()
        clock = FakeClock(_T0)
        await reset.ensure_v2_initialized(factory, "lz_1", clock=clock, **_registries())

        pair = naming.storage_metadata_key_pair("lz_1")
        assert pair.current in factory.stores  # created by the init above

        await reset.async_purge_zone_storage(factory, "lz_1", raw_registry=_raw_registry())

        assert factory.stores[pair.current].removed is True
        assert factory.stores[pair.legacy].removed is True


# ── Commit B: ha_integration.py save-path wiring ─────────────────────────────

class TestHaIntegrationSaveWiring:
    """Exercises LearningShadowController's private save-safe methods with
    FakeFactory-backed stores substituted in after construction — hermetic,
    no real Home Assistant Store I/O."""

    def _make_shadow(self):
        from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord)
        factory = FakeFactory()
        shadow._adaptation_store = AdaptationHistoryStore(factory, shadow._zone)
        shadow._application_lifecycle_store = ApplicationLifecycleStore(factory, shadow._zone)
        shadow._storage_metadata_store = StorageMetadataStore(factory, shadow._zone)

        class _FakeCaptureStores:
            def episodes_store(self_inner):
                return EpisodesStore(factory, shadow._zone)

            def support_critical_events_store(self_inner):
                return SupportCriticalEventStore(factory, shadow._zone)

            def research_daily_store(self_inner):
                return ResearchDailyStore(factory, shadow._zone)

        shadow._capture_stores = _FakeCaptureStores()
        return shadow, factory

    async def _summary(self, shadow):
        return await shadow._storage_metadata_store.get_store_summary()

    async def test_adaptation_history_save_marks_written(self):
        shadow, _ = self._make_shadow()
        shadow._adaptation_save_needed = True
        await shadow._async_save_adaptation_history_safe()
        stores = (await self._summary(shadow))["stores"]
        assert stores["adaptation_history"]["exists"] is True
        assert stores["adaptation_history"]["last_write_reason"] == "adaptation_history_update"

    async def test_application_lifecycle_save_marks_written(self):
        from custom_components.thermosmart.learning.adaptation.application_state import (
            ApplicationLifecycleState,
        )
        shadow, _ = self._make_shadow()
        shadow._application_lifecycle_state = ApplicationLifecycleState(
            learning_zone_id=shadow._zone, updated_at="", entries={})
        shadow._application_lifecycle_save_needed = True
        await shadow._async_save_application_lifecycle_safe()
        stores = (await self._summary(shadow))["stores"]
        assert stores["application_lifecycle"]["exists"] is True
        assert stores["application_lifecycle"]["last_write_reason"] == "lifecycle_update"

    async def test_episode_history_save_marks_written(self):
        shadow, _ = self._make_shadow()
        shadow._episode_save_needed = True
        await shadow._async_save_episode_history_safe()
        stores = (await self._summary(shadow))["stores"]
        assert stores["episodes"]["exists"] is True
        assert stores["episodes"]["last_write_reason"] == "episode_closed"

    async def test_support_critical_events_save_marks_written(self):
        shadow, _ = self._make_shadow()
        shadow._support_critical_events_save_needed = True
        await shadow._async_save_support_critical_events_safe(force=True)
        stores = (await self._summary(shadow))["stores"]
        assert stores["support_critical_events"]["exists"] is True
        assert stores["support_critical_events"]["last_write_reason"] == "support_event_append"

    async def test_research_daily_save_marks_written(self):
        shadow, _ = self._make_shadow()
        shadow._research_daily_save_needed = True
        await shadow._async_save_research_daily_safe(force=True)
        stores = (await self._summary(shadow))["stores"]
        assert stores["research_daily"]["exists"] is True
        assert stores["research_daily"]["last_write_reason"] == "research_daily_update"

    async def test_updated_at_utc_changes_on_second_save(self):
        shadow, _ = self._make_shadow()
        shadow._episode_save_needed = True
        await shadow._async_save_episode_history_safe()
        first = (await self._summary(shadow))["stores"]["episodes"]
        shadow._episode_save_needed = True
        await shadow._async_save_episode_history_safe()
        second = (await self._summary(shadow))["stores"]["episodes"]
        assert second["created_at_utc"] == first["created_at_utc"]
        assert second["updated_at_utc"] >= first["updated_at_utc"]

    async def test_metadata_failure_does_not_break_real_save(self):
        """A corrupted StorageMetadataStore envelope must never block the
        real episode-history save it decorates."""
        shadow, factory = self._make_shadow()
        meta_pair = naming.storage_metadata_key_pair(shadow._zone)
        factory.stores[meta_pair.current] = FakeStore()
        factory.stores[meta_pair.current].data = {"store_schema_version": 999, "data": {}}

        shadow._episode_save_needed = True
        shadow._episode_history = {"ep1": {"dummy": True}}
        await shadow._async_save_episode_history_safe()

        assert shadow._episode_save_needed is False
        assert shadow._episode_last_error is None
        episodes_store = shadow._capture_stores.episodes_store()
        saved = await episodes_store.load()
        assert saved == {"episodes": {"ep1": {"dummy": True}}}
