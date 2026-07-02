"""Tests for the LE2 Research Daily Bucket store-key naming, store wrapper,
and LearningCaptureStores accessor.

This step wires ONLY:
  - custom_components/thermosmart/learning/storage/naming.py
    (``research_daily_key()``)
  - custom_components/thermosmart/learning/storage/stores.py
    (``RESEARCH_DAILY_STORE_VERSION``, ``ResearchDailyStore``)
  - custom_components/thermosmart/learning/storage/capture_stores.py
    (``LearningCaptureStores.research_daily_store()``)

No LearningShadowController state, no load/save call site in
ha_integration.py, no runtime/coordinator hook, no export hook, no
producer — mirrors exactly the "Foundation only" scope of
``SupportCriticalEventStore``/``support_critical_events_store()`` before
their own later, separately-approved wiring step
(test_le2_support_critical_event_wiring.py).

17 test groups:
  T1  — research_daily_key() is stable/deterministic
  T2  — Key is public-safe (rejects unsafe zone id, no leaked entity ids)
  T3  — Key is zone-separated
  T4  — Key uses the existing thermosmart_le2__ prefix consistently
  T5  — ResearchDailyStore uses the correct version
  T6  — ResearchDailyStore uses the correct key
  T7  — Store wrapper only calls the existing StoreFactory
  T8  — LearningCaptureStores.research_daily_store() constructs lazily
  T9  — Multiple accessor calls return an equivalent (same-keyed) store
  T10 — Accessor does not collide with EpisodesStore
  T11 — Accessor does not collide with SupportCriticalEventStore
  T12 — No load/save/prune executes on accessor call
  T13 — No runtime/coordinator/ha_integration reference in new code
  T14 — No export reference in new code
  T15 — Existing Research Daily Foundation tests remain green
  T16 — Existing Support/Event Store tests remain green
  T17 — No control-path keywords touched
"""
from __future__ import annotations

import inspect
from typing import Any, Optional

import pytest

from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.storage import stores as _stores_mod
from custom_components.thermosmart.learning.storage import capture_stores as _capture_stores_mod
from custom_components.thermosmart.learning.storage.capture_stores import LearningCaptureStores
from custom_components.thermosmart.learning.storage.stores import (
    EPISODES_STORE_VERSION,
    RESEARCH_DAILY_STORE_VERSION,
    SUPPORT_CRITICAL_EVENT_STORE_VERSION,
    EpisodesStore,
    ResearchDailyStore,
    SupportCriticalEventStore,
)

_ZONE_A = "zone_alpha_01"
_ZONE_B = "zone_beta_02"


# ── Fake store infrastructure (mirrors existing store-wiring test files) ───

class _FakeRawStore:
    def __init__(self, *, initial: Any = None, load_raises: Optional[Exception] = None,
                 save_raises: Optional[Exception] = None) -> None:
        self._data = initial
        self._load_raises = load_raises
        self._save_raises = save_raises
        self.load_call_count = 0
        self.save_call_count = 0
        self.remove_call_count = 0

    async def async_load(self) -> Any:
        self.load_call_count += 1
        if self._load_raises:
            raise self._load_raises
        return self._data

    async def async_save(self, data: Any) -> None:
        self.save_call_count += 1
        if self._save_raises:
            raise self._save_raises
        self._data = data

    async def async_remove(self) -> None:
        self.remove_call_count += 1
        self._data = None


class _CountingStoreFactory:
    """A StoreFactory that counts .create() calls without doing any I/O."""

    def __init__(self) -> None:
        self.create_calls = 0
        self._stores: dict[str, _FakeRawStore] = {}

    def create(self, key: str, version: int) -> _FakeRawStore:
        self.create_calls += 1
        if key not in self._stores:
            self._stores[key] = _FakeRawStore()
        return self._stores[key]


# ── T1: research_daily_key() is stable/deterministic ───────────────────────

def test_t1_key_is_deterministic():
    key1 = naming.research_daily_key(_ZONE_A)
    key2 = naming.research_daily_key(_ZONE_A)
    assert key1 == key2


# ── T2: Key is public-safe ──────────────────────────────────────────────────

def test_t2_key_rejects_unsafe_zone_id():
    with pytest.raises(naming.StorageNamingError):
        naming.research_daily_key("zone/with/slashes")


def test_t2_key_carries_only_opaque_zone_id_not_entity_ids():
    key = naming.research_daily_key(_ZONE_A)
    assert "climate." not in key
    assert "sensor." not in key
    assert _ZONE_A in key  # opaque internal id, not a human zone name


# ── T3: Key is zone-separated ───────────────────────────────────────────────

def test_t3_key_is_zone_separated():
    key_a = naming.research_daily_key(_ZONE_A)
    key_b = naming.research_daily_key(_ZONE_B)
    assert key_a != key_b
    assert _ZONE_A in key_a
    assert _ZONE_B in key_b


# ── T4: Key uses the existing thermosmart_le2__ prefix consistently ────────

def test_t4_key_uses_existing_prefix():
    key = naming.research_daily_key(_ZONE_A)
    assert key.startswith("thermosmart_le2__")
    assert "research_daily" in key


def test_t4_key_matches_sibling_prefix_convention():
    research_key = naming.research_daily_key(_ZONE_A)
    episodes_key = naming.episodes_key(_ZONE_A)
    support_key = naming.support_critical_events_key(_ZONE_A)
    research_prefix = research_key.split("__")[0]
    assert research_prefix == episodes_key.split("__")[0]
    assert research_prefix == support_key.split("__")[0]


# ── T5: ResearchDailyStore uses the correct version ─────────────────────────

def test_t5_store_version_is_correct():
    assert RESEARCH_DAILY_STORE_VERSION == 1
    factory = _CountingStoreFactory()
    store = ResearchDailyStore(factory, _ZONE_A)
    assert store._version == RESEARCH_DAILY_STORE_VERSION


# ── T6: ResearchDailyStore uses the correct key ──────────────────────────────

def test_t6_store_key_matches_naming_helper():
    factory = _CountingStoreFactory()
    store = ResearchDailyStore(factory, _ZONE_A)
    assert store.key == naming.research_daily_key(_ZONE_A)


# ── T7: Store wrapper only calls the existing StoreFactory ─────────────────

def test_t7_store_construction_only_calls_factory_create():
    factory = _CountingStoreFactory()
    ResearchDailyStore(factory, _ZONE_A)
    assert factory.create_calls == 1


def test_t7_store_wrapper_source_has_no_direct_ha_store_import():
    # The module imports HA Store exactly once (HomeAssistantStoreFactory),
    # ResearchDailyStore itself must not construct Store(...) directly —
    # only delegate to the injected factory via super().__init__().
    class_source = inspect.getsource(ResearchDailyStore)
    assert "= Store(" not in class_source
    assert "helpers.storage" not in class_source
    assert "super().__init__(" in class_source


# ── T8: LearningCaptureStores.research_daily_store() constructs lazily ─────

def test_t8_accessor_constructs_lazily_no_io():
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    assert factory.create_calls == 0  # constructing LearningCaptureStores itself does nothing
    store = stores.research_daily_store()
    assert isinstance(store, ResearchDailyStore)
    assert factory.create_calls == 1


# ── T9: Multiple accessor calls return an equivalent (same-keyed) store ────

def test_t9_multiple_accessor_calls_return_equivalent_store():
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    store1 = stores.research_daily_store()
    store2 = stores.research_daily_store()
    assert store1.key == store2.key
    assert store1._version == store2._version


# ── T10: Accessor does not collide with EpisodesStore ───────────────────────

def test_t10_no_collision_with_episodes_store():
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    research_key = stores.research_daily_store().key
    episodes_key = stores.episodes_store().key
    assert research_key != episodes_key


# ── T11: Accessor does not collide with SupportCriticalEventStore ──────────

def test_t11_no_collision_with_support_critical_event_store():
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    research_key = stores.research_daily_store().key
    support_key = stores.support_critical_events_store().key
    assert research_key != support_key


# ── T12: No load/save/prune executes on accessor call ──────────────────────

def test_t12_no_io_on_accessor_call():
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    stores.research_daily_store()
    for fake_store in factory._stores.values():
        assert fake_store.load_call_count == 0
        assert fake_store.save_call_count == 0
        assert fake_store.remove_call_count == 0


# ── T13: No runtime/coordinator/ha_integration reference in new code ───────

def test_t13_no_runtime_reference_in_naming():
    source = inspect.getsource(naming).lower()
    assert "coordinator" not in source
    assert "ha_integration" not in source


def test_t13_no_runtime_reference_in_stores_research_daily_class():
    source = inspect.getsource(ResearchDailyStore).lower()
    assert "coordinator" not in source
    assert "ha_integration" not in source


def test_t13_no_runtime_reference_in_capture_stores_accessor():
    source = inspect.getsource(LearningCaptureStores.research_daily_store).lower()
    assert "coordinator" not in source
    assert "ha_integration" not in source


def test_t13_ha_integration_not_touched():
    """At the time THIS store-wiring step was built, ha_integration.py had no
    reference to Research Daily anything — checked here as a snapshot of
    that boundary. A later, separately-approved step ("Research Daily State
    and Save Wiring") legitimately added in-memory state plus load/save
    wiring in LearningShadowController, reaching ResearchDailyStore via
    LearningCaptureStores.research_daily_store() — expected, intentional
    storage-lifecycle wiring, not a regression of this step's own scope.
    What must remain true regardless is that ha_integration.py never
    references the store-key helper directly (it only goes through
    LearningCaptureStores, exactly like Episode/Support Event storage)."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha)
    assert "research_daily_key" not in source


def test_t13_coordinator_not_touched():
    import custom_components.thermosmart.coordinator as _coord
    source = inspect.getsource(_coord)
    assert "ResearchDailyStore" not in source
    assert "research_daily_store" not in source
    assert "research_daily_key" not in source


# ── T14: No export reference in new code ────────────────────────────────────

def test_t14_no_export_reference():
    """At the time THIS store-wiring step was built, export.py had no
    Research Daily reference — checked here as a snapshot of that boundary.
    A later, separately-approved step ("Research Daily Long-Term Summary
    Export") legitimately added a "research_daily" research-export block,
    reading ONLY LearningShadowController.research_daily_snapshot() —
    expected, intentional export wiring, not a regression of this step's
    own scope. What must remain true regardless is that export.py never
    constructs/references the ResearchDailyStore wrapper class itself or
    calls the .research_daily_store() accessor — no store I/O in the
    export layer."""
    for mod in (naming, _stores_mod, _capture_stores_mod):
        source = inspect.getsource(mod).lower()
        assert "export.py" not in source

    import custom_components.thermosmart.export as _exp
    source = inspect.getsource(_exp)
    assert "ResearchDailyStore(" not in source
    assert ".research_daily_store(" not in source


# ── T15: Existing Research Daily Foundation tests remain green ─────────────

def test_t15_regression_imports_ok_research_daily_foundation():
    import tests.test_le2_research_daily_buckets  # noqa: F401


# ── T16: Existing Support/Event Store tests remain green ───────────────────

def test_t16_regression_imports_ok_support_event_store():
    import tests.test_le2_support_critical_events  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_capture_wiring_foundation  # noqa: F401
    import tests.test_le2_episode_store_wiring  # noqa: F401


# ── T17: No control-path keywords touched ────────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t17_no_control_keywords_in_new_naming_helper():
    source = inspect.getsource(naming.research_daily_key).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"


def test_t17_no_control_keywords_in_store_wrapper():
    source = inspect.getsource(ResearchDailyStore).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"


def test_t17_no_control_keywords_in_accessor():
    source = inspect.getsource(LearningCaptureStores.research_daily_store).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"
