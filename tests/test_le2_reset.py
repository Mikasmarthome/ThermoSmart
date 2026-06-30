"""Phase 7 tests: first initialization, idempotency, identity, v1 detection, registry."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.init_state import (
    InitializationAction,
    InitializationStatus,
)
from custom_components.thermosmart.learning.registry import ModelRegistry
from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.storage.naming import StorageNamingError
from custom_components.thermosmart.learning.reset import (
    ensure_v2_initialized,
    inspect_initialization_state,
)
from tests.helpers_reset import FakeFactory, raw_registry, registries


async def init(factory, entry_id="entry_1", **over):
    kw = registries()
    kw.update(over)
    return await ensure_v2_initialized(factory, entry_id, **kw)


class TestFirstInitialization:
    async def test_empty_zone_initialized(self):
        f = FakeFactory()
        r = await init(f)
        assert r.status is InitializationStatus.INITIALIZED
        assert r.action is InitializationAction.INITIALIZED_NOW

    async def test_creates_all_required_components(self):
        f = FakeFactory()
        await init(f)
        keys = set(f.stores.keys())
        assert naming.container_key("entry_1") in keys
        assert naming.episodes_key("entry_1") in keys
        assert naming.model_state_key("entry_1") in keys
        assert naming.raw_index_key("entry_1", _room()) in keys
        assert naming.raw_index_key("entry_1", _trv()) in keys

    async def test_no_active_segments_created(self):
        f = FakeFactory()
        await init(f)
        # only index/container/episodes/models stores exist; no raw segment stores
        assert not any("_seg_" in k for k in f.stores)

    async def test_model_state_empty_and_versioned(self):
        f = FakeFactory()
        await init(f)
        models = f.stores[naming.model_state_key("entry_1")].d["data"]
        assert models["models"] == {}
        assert "model_state_container_version" in models

    async def test_trv_only_zone_initializes(self):
        # no optional sensors anywhere -> still fully initializes
        f = FakeFactory()
        r = await init(f)
        assert r.status is InitializationStatus.INITIALIZED

    async def test_empty_model_registry_allowed(self):
        f = FakeFactory()
        r = await init(f, model_registry=ModelRegistry())
        assert r.status is InitializationStatus.INITIALIZED

    async def test_final_marker_set_last(self):
        f = FakeFactory()
        await init(f)
        container = f.stores[naming.container_key("entry_1")].d["data"]
        assert container["init_status"] == "initialized"
        assert container["initialized_ts"] is not None


class TestIdempotency:
    async def test_second_call_already_initialized(self):
        f = FakeFactory()
        await init(f)
        r = await init(f)
        assert r.action is InitializationAction.ALREADY_INITIALIZED

    async def test_ten_repeated_calls_stable(self):
        f = FakeFactory()
        await init(f)
        container_saves = f.stores[naming.container_key("entry_1")].saves
        for _ in range(10):
            await init(f)
        # no further writes to the container after it was initialized
        assert f.stores[naming.container_key("entry_1")].saves == container_saves

    async def test_no_new_learning_zone_id(self):
        f = FakeFactory()
        r1 = await init(f)
        r2 = await init(f)
        assert r1.learning_zone_id == r2.learning_zone_id == "entry_1"


class TestIdentity:
    async def test_learning_zone_id_equals_entry_id(self):
        f = FakeFactory()
        r = await init(f, "entry_abc")
        assert r.learning_zone_id == "entry_abc"

    async def test_same_entry_keeps_id_across_calls(self):
        f = FakeFactory()
        await init(f, "entry_x")
        st = await inspect_initialization_state(f, "entry_x", raw_registry=raw_registry())
        assert st.learning_zone_id == "entry_x"

    async def test_invalid_id_rejected(self):
        f = FakeFactory()
        with pytest.raises(StorageNamingError):
            await init(f, "../evil")

    async def test_double_underscore_id_rejected(self):
        f = FakeFactory()
        with pytest.raises(StorageNamingError):
            await init(f, "a__b")


class TestV1Detection:
    async def test_v1_absent(self):
        f = FakeFactory()
        r = await init(f)
        assert r.v1_detected is False
        assert r.v1_import_skipped is True

    async def test_v1_present_detected_not_imported(self):
        f = FakeFactory()
        # seed a v1 store with arbitrary content (no envelope, like the real v1)
        v1 = f.create("thermosmart_learning_data", 1)
        v1.d = {"observations": {"old_zone": [{"ts": "x"}]}}
        before = dict(v1.d)
        r = await init(f)
        assert r.v1_detected is True
        assert r.v1_import_skipped is True
        # v1 content untouched
        assert v1.d == before
        # no v1 data leaked into v2 model state
        assert f.stores[naming.model_state_key("entry_1")].d["data"]["models"] == {}


class TestInspect:
    async def test_not_initialized(self):
        f = FakeFactory()
        st = await inspect_initialization_state(f, "entry_1", raw_registry=raw_registry())
        assert st.status is InitializationStatus.NOT_INITIALIZED

    async def test_initialized(self):
        f = FakeFactory()
        await init(f)
        st = await inspect_initialization_state(f, "entry_1", raw_registry=raw_registry())
        assert st.status is InitializationStatus.INITIALIZED
        assert st.missing_components == ()


def _room():
    from custom_components.thermosmart.learning.raw_schemas import RawTrackName
    return RawTrackName.ROOM


def _trv():
    from custom_components.thermosmart.learning.raw_schemas import RawTrackName
    return RawTrackName.TRV
