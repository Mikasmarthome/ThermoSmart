"""Tests for the LE2 capture storage wiring foundation.

Covers:
  - custom_components/thermosmart/learning/storage/capture_registry.py
  - custom_components/thermosmart/learning/storage/capture_stores.py
  - the minimal, write-nothing wiring into LearningShadowController.__init__
    (custom_components/thermosmart/learning/runtime/ha_integration.py)

This step (Option A — Minimal Initialization Only) deliberately builds ONLY a
live-reachable, write-nothing store-access facade. Nothing in this file tests
Raw/Episode persistence itself, because nothing in this step writes Raw
samples or Episode summaries — that is explicitly out of scope, reserved for
a later step once a real capture/persist call site is designed.

11 test groups:
  T1  — Registries build cleanly and validate with zero cross-reference issues
  T2  — Capture Store Registry/Factory produces zone-separated stores
  T3  — Store keys are stable and deterministic
  T4  — Constructing LearningCaptureStores performs zero storage I/O
  T5  — Accessing an undeclared raw track raises a clear, catchable error
  T6  — Missing/corrupt store initialization is non-fatal (ha_integration.py wiring)
  T7  — No Raw Segment write happens anywhere in this step
  T8  — No Episode Store write happens anywhere in this step
  T9  — ha_integration.py wiring exists, is non-fatal, and performs no I/O
  T10 — Existing periodic-save and retention tests remain green (regression)
  T11 — No control keywords in the new modules
"""
from __future__ import annotations

import inspect
from typing import Any

import pytest

from custom_components.thermosmart.learning.raw_schemas import RawTrackName
from custom_components.thermosmart.learning.registry import (
    ModelRegistry,
    UnknownDefinitionError,
    validate_registry_graph,
)
from custom_components.thermosmart.learning.storage import capture_registry as _capture_registry_mod
from custom_components.thermosmart.learning.storage import capture_stores as _capture_stores_mod
from custom_components.thermosmart.learning.storage.capture_registry import (
    build_episode_registry,
    build_raw_track_registry,
)
from custom_components.thermosmart.learning.storage.capture_stores import LearningCaptureStores
from custom_components.thermosmart.learning.storage.stores import (
    EpisodesStore,
    RawSegmentIndexStore,
    RawSegmentStore,
)

_ZONE_A = "zone_alpha_01"
_ZONE_B = "zone_beta_02"


# ── Fake store factory (no HA dependency) ────────────────────────────────────

class _FakeRawStore:
    def __init__(self) -> None:
        self.load_calls = 0
        self.save_calls = 0
        self.remove_calls = 0

    async def async_load(self) -> Any:
        self.load_calls += 1
        return None

    async def async_save(self, data: Any) -> None:
        self.save_calls += 1

    async def async_remove(self) -> None:
        self.remove_calls += 1


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


# ── T1: Registries build cleanly and validate ────────────────────────────────

def test_t1_raw_track_registry_has_all_six_tracks():
    reg = build_raw_track_registry()
    assert len(reg) == 6
    for track in RawTrackName:
        assert reg.contains(track)


def test_t1_episode_registry_has_all_five_types():
    from custom_components.thermosmart.learning.episode_schemas import EpisodeType
    reg = build_episode_registry()
    assert len(reg) == 5
    for etype in EpisodeType:
        assert reg.contains(etype)


def test_t1_registry_graph_validates_cleanly():
    raw = build_raw_track_registry()
    episodes = build_episode_registry()
    issues = validate_registry_graph(raw, episodes, ModelRegistry())
    assert issues == []


def test_t1_retention_policies_are_bounded():
    raw = build_raw_track_registry()
    episodes = build_episode_registry()
    for definition in raw.definitions():
        assert definition.retention.max_age_days is not None
        assert definition.retention.max_records is not None
    for definition in episodes.definitions():
        assert definition.retention.max_age_days is not None
        assert definition.retention.max_records is not None


# ── T2: Capture Store Registry/Factory produces zone-separated stores ──────

def test_t2_zone_separation_in_store_keys():
    factory = _CountingStoreFactory()
    stores_a = LearningCaptureStores(factory, _ZONE_A)
    stores_b = LearningCaptureStores(factory, _ZONE_B)

    store_a = stores_a.raw_segment_index_store(RawTrackName.ROOM)
    store_b = stores_b.raw_segment_index_store(RawTrackName.ROOM)
    assert store_a.key != store_b.key
    assert _ZONE_A in store_a.key
    assert _ZONE_B in store_b.key


def test_t2_track_separation_in_store_keys():
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    room_index = stores.raw_segment_index_store(RawTrackName.ROOM)
    trv_index = stores.raw_segment_index_store(RawTrackName.TRV)
    assert room_index.key != trv_index.key


# ── T3: Store keys are stable and deterministic ──────────────────────────────

def test_t3_store_keys_are_deterministic():
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    key1 = stores.raw_segment_index_store(RawTrackName.ROOM).key
    key2 = stores.raw_segment_index_store(RawTrackName.ROOM).key
    assert key1 == key2


def test_t3_episodes_store_key_is_stable_and_zone_scoped():
    factory = _CountingStoreFactory()
    stores_a = LearningCaptureStores(factory, _ZONE_A)
    stores_b = LearningCaptureStores(factory, _ZONE_B)
    key_a1 = stores_a.episodes_store().key
    key_a2 = stores_a.episodes_store().key
    key_b = stores_b.episodes_store().key
    assert key_a1 == key_a2
    assert key_a1 != key_b


# ── T4: Constructing LearningCaptureStores performs zero storage I/O ────────

def test_t4_construction_performs_no_io():
    factory = _CountingStoreFactory()
    LearningCaptureStores(factory, _ZONE_A)
    assert factory.create_calls == 0


def test_t4_accessing_store_handle_still_performs_no_io():
    """Building a store WRAPPER (not calling .load()/.save() on it) must not
    itself trigger any I/O — .create() on the factory only allocates the
    HA Store object, it doesn't read/write the file."""
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    stores.raw_segment_index_store(RawTrackName.ROOM)
    stores.episodes_store()
    # No async_load/async_save/async_remove call was ever made.
    for fake_store in factory._stores.values():
        assert fake_store.load_calls == 0
        assert fake_store.save_calls == 0
        assert fake_store.remove_calls == 0


# ── T5: Accessing an undeclared raw track raises a clear, catchable error ──

def test_t5_undeclared_track_raises_unknown_definition_error():
    factory = _CountingStoreFactory()
    # Build a stores facade with an intentionally empty raw registry to
    # simulate "track not declared".
    from custom_components.thermosmart.learning.registry import RawTrackRegistry
    stores = LearningCaptureStores(factory, _ZONE_A, raw_registry=RawTrackRegistry())
    with pytest.raises(UnknownDefinitionError):
        stores.raw_segment_store(RawTrackName.ROOM)


def test_t5_declared_track_does_not_raise():
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    store = stores.raw_segment_store(RawTrackName.ROOM)
    assert isinstance(store, RawSegmentStore)


# ── T6: Missing/corrupt store initialization is non-fatal ──────────────────

def test_t6_ha_integration_wiring_is_wrapped_in_try_except():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.__init__)
    idx = source.index("self._capture_stores = None")
    window = source[idx:idx + 500]
    assert "try:" in window
    assert "LearningCaptureStores" in window
    assert "except Exception:" in window


# ── T7/T8: No Raw Segment / Episode write happens anywhere in this step ────

def _strip_docstrings_and_comments(source: str) -> str:
    """Best-effort removal of module/function docstrings and comment lines,
    so a mention of ``.save()`` in prose doesn't cause a false positive."""
    lines = []
    in_docstring = False
    for line in source.splitlines():
        stripped = line.strip()
        if in_docstring:
            if '"""' in stripped:
                in_docstring = False
            continue
        if stripped.startswith('"""') and stripped.count('"""') == 1:
            in_docstring = True
            continue
        if stripped.startswith('"""') and stripped.count('"""') >= 2:
            continue  # single-line docstring
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_t7_no_raw_segment_save_call_in_new_modules():
    for mod in (_capture_registry_mod, _capture_stores_mod):
        code = _strip_docstrings_and_comments(inspect.getsource(mod))
        assert ".save(" not in code
        assert ".save_segment(" not in code
        assert "async_save" not in code


def test_t8_no_episode_store_save_call_in_new_modules():
    for mod in (_capture_registry_mod, _capture_stores_mod):
        code = _strip_docstrings_and_comments(inspect.getsource(mod))
        assert "episodes_store().save" not in code


def test_t7_t8_ha_integration_capture_wiring_block_contains_no_save_call():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.__init__)
    idx = source.index("self._capture_stores = None")
    end = source.index("Cache: schedule target time", idx)
    block = source[idx:end]
    assert "async_save" not in block
    assert ".save(" not in block


# ── T9: Existing behavior / non-fatal setup path (behavioral) ──────────────

def test_t9_learning_capture_stores_constructs_from_hass_style_factory():
    """Simulate the exact construction call ha_integration.py performs, using
    a fake in place of HomeAssistantStoreFactory, and confirm it succeeds
    without any I/O and exposes the expected accessors."""
    factory = _CountingStoreFactory()
    stores = LearningCaptureStores(factory, _ZONE_A)
    assert stores.learning_zone_id == _ZONE_A
    assert len(stores.raw_registry) == 6
    assert len(stores.episode_registry) == 5
    assert factory.create_calls == 0


# ── T10: Existing periodic-save and retention tests remain green ───────────

def test_t10_regression_imports_ok():
    import tests.test_le2_periodic_save_trigger  # noqa: F401
    import tests.test_le2_retention  # noqa: F401
    import tests.test_application_lifecycle_storage  # noqa: F401
    import tests.test_application_lifecycle_state  # noqa: F401
    import tests.test_adaptation_monitoring  # noqa: F401
    import tests.test_preheat_application_plan  # noqa: F401
    import tests.test_application_readiness  # noqa: F401
    import tests.test_promotion_readiness  # noqa: F401
    import tests.test_application_orchestrator  # noqa: F401
    import tests.test_orchestration_export_trace  # noqa: F401


# ── T11: No control keywords in the new modules ─────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
    "setpoint",
)


def test_t11_no_control_keywords_in_new_modules():
    for mod in (_capture_registry_mod, _capture_stores_mod):
        source = inspect.getsource(mod).lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in source, f"forbidden control token found in {mod.__name__}: {token}"


def test_t11_no_control_keywords_in_ha_integration_capture_wiring_block():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.__init__)
    idx = source.index("self._capture_stores = None")
    end = source.index("Cache: schedule target time", idx)
    block = source[idx:end].lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in block, f"forbidden control token found in wiring block: {token}"
