"""Shared helpers for Phase 7 reset/initialization tests."""
from __future__ import annotations

from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.raw_schemas import (
    RawTrackName,
    RoomObservation,
    TrvObservation,
)
from custom_components.thermosmart.learning.registry import (
    EpisodeRegistry,
    ModelRegistry,
    PrivacyClass,
    RawTrackDefinition,
    RawTrackRegistry,
    RetentionPolicy,
)


class FakeStore:
    def __init__(self) -> None:
        self.d = None
        self.saves = 0

    async def async_load(self):
        return self.d

    async def async_save(self, data):
        self.saves += 1
        self.d = data

    async def async_remove(self):
        self.d = None


class FakeFactory:
    def __init__(self) -> None:
        self.stores: dict[str, FakeStore] = {}

    def create(self, key: str, version: int) -> FakeStore:
        return self.stores.setdefault(key, FakeStore())


class FailingFactory(FakeFactory):
    """Raises on the N-th save across all stores (to simulate a crash mid-init)."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self._fail_at = fail_at
        self._save_count = 0

    def create(self, key: str, version: int) -> FakeStore:
        store = self.stores.get(key)
        if store is None:
            store = _FailStore(self)
            self.stores[key] = store
        return store


class _FailStore(FakeStore):
    def __init__(self, factory: FailingFactory) -> None:
        super().__init__()
        self._factory = factory

    async def async_save(self, data):
        self._factory._save_count += 1
        if self._factory._save_count == self._factory._fail_at:
            raise RuntimeError("simulated disk failure")
        await super().async_save(data)


def raw_registry() -> RawTrackRegistry:
    reg = RawTrackRegistry()
    for tn, schema in ((RawTrackName.ROOM, RoomObservation), (RawTrackName.TRV, TrvObservation)):
        reg.register(RawTrackDefinition(
            track_name=tn, schema_type=schema, track_schema_version=1,
            privacy_class=PrivacyClass.LOW, allowed_export_scopes=(ExportScope.SUPPORT,),
            retention=RetentionPolicy(), segmented=True, immutable_events=False))
    return reg


def registries():
    return dict(raw_registry=raw_registry(), episode_registry=EpisodeRegistry(),
                model_registry=ModelRegistry())
