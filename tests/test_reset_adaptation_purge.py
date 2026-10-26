"""Zone removal still purges the removed adaptation stores' legacy data.

The adaptation candidate-history and application-lifecycle layers were removed,
along with their ``AdaptationHistoryStore`` / ``ApplicationLifecycleStore``
wrapper classes. Existing installations may still hold inert data under those
keys, so ``async_purge_zone_storage`` must keep deleting all four historical
keys on real zone/entry removal:

    thermosmart_learning__{zid}__adaptation_history      (neutral)
    thermosmart_le2__{zid}__adaptation_history           (legacy)
    thermosmart_learning__{zid}__application_lifecycle   (neutral)
    thermosmart_le2__{zid}__application_lifecycle         (legacy)

The purge is keyed straight off naming.py through the generic versioned-store
delete — no dependency on the (now removed) adaptation package.
"""
from __future__ import annotations

from custom_components.thermosmart.learning import reset
from custom_components.thermosmart.learning.storage import naming


class FakeStore:
    def __init__(self):
        self.data = None
        self.removed = False

    async def async_load(self):
        return self.data

    async def async_save(self, data):
        self.data = data

    async def async_remove(self):
        self.data = None
        self.removed = True


class FakeFactory:
    def __init__(self):
        self.stores: dict[str, FakeStore] = {}

    def create(self, key: str, version: int) -> FakeStore:
        return self.stores.setdefault(key, FakeStore())


def _seed(factory: FakeFactory, key: str) -> FakeStore:
    store = factory.create(key, 1)
    store.data = {"store_schema_version": 1, "data": {"x": 1}}
    return store


# The four historical keys, for zone "lz_1".
def _all_four_keys(zid: str) -> list[str]:
    ah = naming.adaptation_history_key_pair(zid)
    al = naming.application_lifecycle_key_pair(zid)
    return [ah.current, ah.legacy, al.current, al.legacy]


class TestAdaptationPurge:
    async def test_all_four_historical_keys_removed(self):
        factory = FakeFactory()
        keys = _all_four_keys("lz_1")
        for key in keys:
            _seed(factory, key)

        result = await reset.async_purge_zone_storage(factory, "lz_1")

        assert result.errors == ()
        for key in keys:
            assert factory.stores[key].removed is True, key
            assert factory.stores[key].data is None, key

    async def test_neutral_only_present_is_removed(self):
        factory = FakeFactory()
        ah = naming.adaptation_history_key_pair("lz_1")
        al = naming.application_lifecycle_key_pair("lz_1")
        _seed(factory, ah.current)
        _seed(factory, al.current)

        await reset.async_purge_zone_storage(factory, "lz_1")

        assert factory.stores[ah.current].removed is True
        assert factory.stores[al.current].removed is True

    async def test_legacy_only_present_is_removed(self):
        factory = FakeFactory()
        ah = naming.adaptation_history_key_pair("lz_1")
        al = naming.application_lifecycle_key_pair("lz_1")
        _seed(factory, ah.legacy)
        _seed(factory, al.legacy)

        await reset.async_purge_zone_storage(factory, "lz_1")

        assert factory.stores[ah.legacy].removed is True
        assert factory.stores[al.legacy].removed is True

    async def test_mixed_neutral_and_legacy_all_removed(self):
        factory = FakeFactory()
        ah = naming.adaptation_history_key_pair("lz_1")
        al = naming.application_lifecycle_key_pair("lz_1")
        # adaptation_history only neutral, application_lifecycle only legacy
        _seed(factory, ah.current)
        _seed(factory, al.legacy)

        await reset.async_purge_zone_storage(factory, "lz_1")

        assert factory.stores[ah.current].removed is True
        assert factory.stores[al.legacy].removed is True

    async def test_other_zone_is_left_untouched(self):
        factory = FakeFactory()
        keep = _all_four_keys("lz_2")
        for key in keep:
            _seed(factory, key)
        for key in _all_four_keys("lz_1"):
            _seed(factory, key)

        await reset.async_purge_zone_storage(factory, "lz_1")

        for key in keep:
            assert factory.stores[key].removed is False, key
            assert factory.stores[key].data is not None, key
