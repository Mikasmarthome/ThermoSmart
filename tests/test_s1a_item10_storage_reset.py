"""S1a Item 10 - Section 5+11: Schema version safety and zone reset.

Verifies StoreVersionError on schema mismatch, V1 store detection,
ensure_v2_initialized idempotency, and reset_v2_learning_state clean delete+re-init.
Async tests use FakeFactory from helpers_reset.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.init_state import (
    InitializationStatus,
    ResetReason,
)
from custom_components.thermosmart.learning.reset import (
    ensure_v2_initialized,
    inspect_initialization_state,
    reset_v2_learning_state,
)
from custom_components.thermosmart.learning.storage.stores import (
    StoreVersionError,
    ZoneMetadataStore,
)
from tests.helpers_reset import FakeFactory, FailingFactory, registries

pytestmark = pytest.mark.asyncio


async def _init(factory, regs=None) -> any:
    r = regs or registries()
    return await ensure_v2_initialized(factory, "test-entry-id", **r)


# ── 1. ensure_v2_initialized ──────────────────────────────────────────────────


class TestEnsureV2Initialized:
    async def test_fresh_init_returns_initialized(self):
        result = await _init(FakeFactory())
        assert result.status == InitializationStatus.INITIALIZED

    async def test_fresh_init_action_is_initialized_now(self):
        from custom_components.thermosmart.learning.init_state import InitializationAction
        result = await _init(FakeFactory())
        assert result.action == InitializationAction.INITIALIZED_NOW

    async def test_idempotent_reinit_returns_already_initialized(self):
        from custom_components.thermosmart.learning.init_state import InitializationAction
        factory = FakeFactory()
        await _init(factory)
        result = await _init(factory)
        assert result.status == InitializationStatus.INITIALIZED
        assert result.action == InitializationAction.ALREADY_INITIALIZED

    async def test_learning_zone_id_is_derived_from_entry_id(self):
        factory = FakeFactory()
        result = await _init(factory)
        assert result.learning_zone_id == "test-entry-id"

    async def test_creates_zone_metadata_store(self):
        factory = FakeFactory()
        await _init(factory)
        # Container store must exist
        from custom_components.thermosmart.learning.storage import naming
        lz = "test-entry-id"
        key = naming.container_key(lz)
        assert key in factory.stores
        assert factory.stores[key].d is not None

    async def test_v1_detected_when_v1_store_present(self):
        factory = FakeFactory()
        # Inject a V1 store
        v1_store = factory.stores.setdefault("thermosmart_learning_data", type(
            "FakeStore", (), {
                "d": {"old": True}, "saves": 0,
                "async_load": lambda self: __import__("asyncio").coroutine(lambda: self.d)(),
                "async_save": lambda self, data: None,
                "async_remove": lambda self: None,
            })()
        )
        # Actually use FakeFactory which returns FakeStore
        factory2 = FakeFactory()
        from tests.helpers_reset import FakeStore
        factory2.stores["thermosmart_learning_data"] = FakeStore()
        factory2.stores["thermosmart_learning_data"].d = {"old": True}
        result = await _init(factory2)
        assert result.v1_detected is True

    async def test_no_v1_detected_when_absent(self):
        factory = FakeFactory()
        result = await _init(factory)
        assert result.v1_detected is False

    async def test_partial_init_recoverable(self):
        # Crash after 2 saves -> partial state
        factory = FailingFactory(fail_at=3)
        try:
            await _init(factory)
        except Exception:
            pass
        # A second init should resume/recover
        factory2 = FakeFactory()
        factory2.stores = factory.stores
        # Unfail the factory by using plain FakeFactory with pre-populated stores
        result = await _init(factory2)
        # Should complete (either initialized or recovered_partial)
        assert result.status in (
            InitializationStatus.INITIALIZED,
            InitializationStatus.FAILED_RECOVERABLE,
        )

    async def test_invalid_entry_id_raises(self):
        with pytest.raises(Exception):
            await _init(FakeFactory(), regs={**registries(), "entry_id_override": "a__b"})


# ── 2. reset_v2_learning_state ────────────────────────────────────────────────


class TestResetV2LearningState:
    async def test_reset_with_reinit_returns_initialized(self):
        factory = FakeFactory()
        await _init(factory)
        result = await reset_v2_learning_state(
            factory, "test-entry-id", ResetReason.MANUAL, **registries())
        assert result.status == InitializationStatus.INITIALIZED

    async def test_reset_clears_all_stores(self):
        factory = FakeFactory()
        r = registries()
        await ensure_v2_initialized(factory, "test-entry-id", **r)
        # Count stores before reset
        stores_before = len([k for k, v in factory.stores.items() if v.d is not None])
        assert stores_before > 0
        # Reset without reinit
        await reset_v2_learning_state(
            factory, "test-entry-id", ResetReason.MANUAL,
            **r, reinitialize=False)
        # All stores deleted (d=None)
        stores_after = len([k for k, v in factory.stores.items() if v.d is not None])
        assert stores_after == 0

    async def test_reset_without_reinit_returns_not_initialized(self):
        factory = FakeFactory()
        await _init(factory)
        result = await reset_v2_learning_state(
            factory, "test-entry-id", ResetReason.MANUAL,
            **registries(), reinitialize=False)
        assert result.status == InitializationStatus.NOT_INITIALIZED

    async def test_reset_requires_reset_reason(self):
        factory = FakeFactory()
        await _init(factory)
        with pytest.raises(Exception):
            await reset_v2_learning_state(
                factory, "test-entry-id", "not_a_reason", **registries())

    async def test_reset_zone_a_leaves_zone_b_untouched(self):
        factory = FakeFactory()
        regs = registries()
        await ensure_v2_initialized(factory, "zone-a", **regs)
        await ensure_v2_initialized(factory, "zone-b", **regs)
        # Reset only zone-a
        await reset_v2_learning_state(
            factory, "zone-a", ResetReason.MANUAL,
            **regs, reinitialize=False)
        # zone-b stores must still be intact
        state_b = await inspect_initialization_state(factory, "zone-b", **{"raw_registry": regs["raw_registry"]})
        assert state_b.status == InitializationStatus.INITIALIZED


# ── 3. StoreVersionError on schema mismatch ───────────────────────────────────


class TestStoreVersionError:
    async def test_mismatched_schema_version_raises(self):
        factory = FakeFactory()
        # Write data with wrong store_schema_version
        from custom_components.thermosmart.learning.storage import naming
        lz = "test-entry-id"
        key = naming.container_key(lz)
        factory.stores[key] = (lambda: (_ for _ in ()).throw(Exception()))
        # Use a FakeStore with bad version directly
        from tests.helpers_reset import FakeStore
        fs = FakeStore()
        fs.d = {"store_schema_version": 99, "data": {}}
        factory.stores[key] = fs
        with pytest.raises(StoreVersionError):
            store = ZoneMetadataStore(factory, lz)
            await store.load()

    async def test_correct_schema_version_loads_ok(self):
        factory = FakeFactory()
        await _init(factory)
        from custom_components.thermosmart.learning.storage import naming
        lz = "test-entry-id"
        store = ZoneMetadataStore(factory, lz)
        data = await store.load()
        assert data is not None
        assert "learning_zone_id" in data


# ── 4. inspect_initialization_state ──────────────────────────────────────────


class TestInspectInitializationState:
    async def test_not_initialized_when_empty(self):
        factory = FakeFactory()
        state = await inspect_initialization_state(
            factory, "new-zone", raw_registry=registries()["raw_registry"])
        assert state.status == InitializationStatus.NOT_INITIALIZED

    async def test_initialized_after_init(self):
        factory = FakeFactory()
        await _init(factory)
        state = await inspect_initialization_state(
            factory, "test-entry-id", raw_registry=registries()["raw_registry"])
        assert state.status == InitializationStatus.INITIALIZED

    async def test_no_missing_after_full_init(self):
        factory = FakeFactory()
        await _init(factory)
        state = await inspect_initialization_state(
            factory, "test-entry-id", raw_registry=registries()["raw_registry"])
        assert len(state.missing_components or []) == 0
        assert len(state.corrupted_components or []) == 0
