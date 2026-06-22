"""Phase 7 multi-zone isolation and global-index rebuild tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.init_state import InitializationStatus
from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.reset import (
    ensure_v2_initialized,
    rebuild_global_index,
)
from tests.helpers_reset import FailingFactory, FakeFactory, registries


async def init(factory, entry_id):
    return await ensure_v2_initialized(factory, entry_id, **registries())


class TestMultiZone:
    async def test_two_zones_independent(self):
        f = FakeFactory()
        ra = await init(f, "zone_a")
        rb = await init(f, "zone_b")
        assert ra.status is rb.status is InitializationStatus.INITIALIZED
        assert ra.learning_zone_id != rb.learning_zone_id

    async def test_zone_keys_are_disjoint(self):
        f = FakeFactory()
        await init(f, "zone_a")
        await init(f, "zone_b")
        assert naming.container_key("zone_a") != naming.container_key("zone_b")
        a_keys = {k for k in f.stores if "zone_a" in k}
        b_keys = {k for k in f.stores if "zone_b" in k}
        assert a_keys and b_keys and a_keys.isdisjoint(b_keys)

    async def test_one_zone_failure_does_not_block_other(self):
        # zone_a fails mid-init, zone_b still initializes on a healthy factory
        fa = FailingFactory(fail_at=2)
        ra = await init(fa, "zone_a")
        assert ra.status is InitializationStatus.FAILED_RECOVERABLE
        fb = FakeFactory()
        rb = await init(fb, "zone_b")
        assert rb.status is InitializationStatus.INITIALIZED

    async def test_initializing_one_zone_does_not_change_other(self):
        f = FakeFactory()
        await init(f, "zone_a")
        a_container = dict(f.stores[naming.container_key("zone_a")].d)
        await init(f, "zone_b")
        # zone_a container untouched by zone_b initialization
        assert f.stores[naming.container_key("zone_a")].d == a_container


class TestGlobalIndex:
    async def test_rebuild_from_containers(self):
        f = FakeFactory()
        await init(f, "zone_a")
        await init(f, "zone_b")
        index = await rebuild_global_index(f, ["zone_a", "zone_b"])
        assert set(index["learning_zones"].keys()) == {"zone_a", "zone_b"}
        assert index["learning_zones"]["zone_a"]["entry_id"] == "zone_a"

    async def test_global_index_is_reconstructable_cache(self):
        f = FakeFactory()
        await init(f, "zone_a")
        idx1 = await rebuild_global_index(f, ["zone_a"])
        idx2 = await rebuild_global_index(f, ["zone_a"])
        assert idx1["learning_zones"] == idx2["learning_zones"]  # deterministic from containers

    async def test_rebuild_skips_uninitialized(self):
        f = FakeFactory()
        await init(f, "zone_a")
        index = await rebuild_global_index(f, ["zone_a", "zone_missing"])
        assert "zone_missing" not in index["learning_zones"]
