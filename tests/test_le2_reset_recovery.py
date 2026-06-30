"""Phase 7 partial-recovery, abort, corruption and version-conflict tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.init_state import (
    InitializationAction,
    InitializationStatus,
    ResetReason,
)
from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.reset import (
    ensure_v2_initialized,
    inspect_initialization_state,
    reset_v2_learning_state,
)
from tests.helpers_reset import FailingFactory, FakeFactory, raw_registry, registries


async def init(factory, entry_id="entry_1"):
    return await ensure_v2_initialized(factory, entry_id, **registries())


class TestAbortAndResume:
    @pytest.mark.parametrize("fail_at", [1, 2, 3, 4, 5])
    async def test_abort_after_each_step_then_resume(self, fail_at):
        f = FailingFactory(fail_at=fail_at)
        r1 = await init(f)
        assert r1.status is InitializationStatus.FAILED_RECOVERABLE
        # final marker must NOT be 'initialized'
        container = f.stores.get(naming.container_key("entry_1"))
        if container is not None and container.d is not None:
            assert container.d["data"]["init_status"] != "initialized"

        # disable failure and resume
        f._fail_at = 9999
        r2 = await init(f)
        assert r2.status is InitializationStatus.INITIALIZED
        assert r2.action in (InitializationAction.INITIALIZED_NOW,
                             InitializationAction.RECOVERED_PARTIAL)
        st = await inspect_initialization_state(f, "entry_1", raw_registry=raw_registry())
        assert st.status is InitializationStatus.INITIALIZED
        assert st.missing_components == ()

    async def test_resume_does_not_duplicate(self):
        f = FailingFactory(fail_at=4)  # fail at episodes
        await init(f)
        f._fail_at = 9999
        await init(f)
        r3 = await init(f)
        assert r3.action is InitializationAction.ALREADY_INITIALIZED
        # exactly one of each component key
        assert sum(1 for k in f.stores if k.endswith("__episodes")) == 1
        assert sum(1 for k in f.stores if k.endswith("__models")) == 1


class TestCorruption:
    async def test_corrupted_component_requires_recovery(self):
        f = FakeFactory()
        await init(f)
        # corrupt the episodes component
        ep_key = naming.episodes_key("entry_1")
        f.stores[ep_key].d = {"store_schema_version": 1, "data": {"bogus": 1}}
        r = await init(f)
        assert r.status is InitializationStatus.RECOVERY_REQUIRED
        assert r.failed_component is not None
        # not silently overwritten
        assert f.stores[ep_key].d["data"] == {"bogus": 1}

    async def test_version_conflict_requires_recovery(self):
        f = FakeFactory()
        await init(f)
        c_key = naming.container_key("entry_1")
        f.stores[c_key].d["data"]["container_version"] = 999
        r = await init(f)
        assert r.status is InitializationStatus.RECOVERY_REQUIRED

    async def test_conflicting_learning_zone_id(self):
        f = FakeFactory()
        await init(f)
        c_key = naming.container_key("entry_1")
        f.stores[c_key].d["data"]["learning_zone_id"] = "someone_else"
        r = await init(f)
        assert r.status is InitializationStatus.RECOVERY_REQUIRED


class TestExplicitReset:
    async def test_reset_requires_reason(self):
        f = FakeFactory()
        await init(f)
        with pytest.raises(Exception):
            await reset_v2_learning_state(f, "entry_1", "not_a_reason", **registries())

    async def test_reset_reinitializes_clean(self):
        f = FakeFactory()
        await init(f)
        # write some episode data, then reset
        ep_key = naming.episodes_key("entry_1")
        f.stores[ep_key].d = {"store_schema_version": 1, "data": {"episodes": {"x": [1]}}}
        r = await reset_v2_learning_state(f, "entry_1", ResetReason.MANUAL, **registries())
        assert r.status is InitializationStatus.INITIALIZED
        # episodes container is empty again
        assert f.stores[ep_key].d["data"]["episodes"] == {}

    async def test_reset_does_not_touch_v1(self):
        f = FakeFactory()
        v1 = f.create("thermosmart_learning_data", 1)
        v1.d = {"observations": {"z": []}}
        await init(f)
        await reset_v2_learning_state(f, "entry_1", ResetReason.MANUAL, **registries())
        assert v1.d == {"observations": {"z": []}}
