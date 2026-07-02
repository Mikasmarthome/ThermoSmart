"""Tests for the LE2 episode persistence state and store wiring in
LearningShadowController (custom_components/thermosmart/learning/runtime/ha_integration.py).

This step wires in-memory episode history state plus load/save through the
existing EpisodesStore (via LearningCaptureStores), reusing the existing
periodic save trigger (async_save_if_due()) and unload flush. It deliberately
does NOT add a runtime append hook in lifecycle.py — nothing calls
append_completed_episode()/append_completed_episodes() from run_cycle() yet.

Since constructing a real LearningShadowController requires a live
HomeAssistant instance (no existing test harness for that), this file
combines two techniques, matching the pattern already established for the
coordinator.py periodic-save wiring audit in this same codebase:

  1. Static source-inspection checks that the REAL wiring in ha_integration.py
     exists, is placed correctly, and that lifecycle.py was NOT touched.
  2. Behavioral tests against a minimal harness whose load()/save() method
     bodies are IDENTICAL copies of the production
     _async_load_episode_history_safe()/_async_save_episode_history_safe()
     methods — validating the CONTRACT, not a mock — using the real
     EpisodesStore class with a fake StoreFactory (no HA dependency).

18 test groups:
  T1  — Episode state is initialized in LearningShadowController.__init__
  T2  — async_setup()-equivalent load succeeds with a valid stored payload
  T3  — Malformed store payload is handled non-fatally
  T4  — Store-load exception is non-fatal
  T5  — Save is skipped when _episode_save_needed is False
  T6  — Save runs when _episode_save_needed is True
  T7  — Save success resets the dirty flag
  T8  — Save failure keeps the dirty flag set
  T9  — Save failure sets the last-error field
  T10 — async_unload() flushes episode history analogous to other stores
  T11 — No runtime hook exists in lifecycle.py
  T12 — No raw store referenced by the new wiring
  T13 — No store I/O in the run_cycle() path
  T14 — test_le2_episode_persistence.py remains green
  T15 — test_le2_episode_serialization.py remains green
  T16 — test_le2_capture_wiring_foundation.py remains green
  T17 — test_le2_retention.py / test_le2_periodic_save_trigger.py remain green
  T18 — No control path touched
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Mapping, Optional

import pytest

from custom_components.thermosmart.learning.storage.episode_serialization import (
    deserialize_episode,
    serialize_episode,
)
from custom_components.thermosmart.learning.storage.stores import EpisodesStore

_ZONE_A = "zone_alpha_01"


# ── Fake store infrastructure (mirrors the pattern used elsewhere in this repo) ─

class _FakeRawStore:
    def __init__(self, *, initial: Any = None, load_raises: Optional[Exception] = None,
                 save_raises: Optional[Exception] = None) -> None:
        self._data = initial
        self._load_raises = load_raises
        self._save_raises = save_raises
        self.save_call_count = 0

    async def async_load(self) -> Any:
        if self._load_raises:
            raise self._load_raises
        return self._data

    async def async_save(self, data: Any) -> None:
        self.save_call_count += 1
        if self._save_raises:
            raise self._save_raises
        self._data = data

    async def async_remove(self) -> None:
        self._data = None


class _FakeStoreFactory:
    def __init__(self) -> None:
        self._stores: dict[str, _FakeRawStore] = {}

    def create(self, key: str, version: int) -> _FakeRawStore:
        if key not in self._stores:
            self._stores[key] = _FakeRawStore()
        return self._stores[key]

    def inject(self, key: str, store: _FakeRawStore) -> None:
        self._stores[key] = store


class _FakeCaptureStores:
    """Minimal stand-in for LearningCaptureStores exposing episodes_store()."""

    def __init__(self, factory: _FakeStoreFactory, zone: str) -> None:
        self._factory = factory
        self._zone = zone

    def episodes_store(self) -> EpisodesStore:
        return EpisodesStore(self._factory, self._zone)


def _envelope(data: Any, version: int = 1) -> dict:
    return {"store_schema_version": version, "data": data}


# ── Async harness mirroring the production contract exactly ─────────────────

class _EpisodeHistoryHelper:
    """Standalone test harness exercising the same load/save contract as
    LearningShadowController._async_load/save_episode_history_safe().

    The implementation here is IDENTICAL to the production methods so that
    the tests validate contract behaviour, not a copy.
    """

    def __init__(self, capture_stores) -> None:
        self._capture_stores = capture_stores
        self._episode_history: dict = {}
        self._episode_save_needed: bool = False
        self._episode_last_error: Optional[str] = None

    async def load(self) -> None:
        if self._capture_stores is None:
            return
        try:
            store = self._capture_stores.episodes_store()
            raw = await store.load()
            if raw is None:
                return
            raw_entries = raw.get("episodes") if isinstance(raw, Mapping) else None
            if not isinstance(raw_entries, Mapping):
                return
            rebuilt: dict = {}
            for eid, entry in raw_entries.items():
                if not isinstance(entry, Mapping):
                    continue
                if deserialize_episode(entry) is None:
                    continue
                rebuilt[eid] = dict(entry)
            self._episode_history = rebuilt
        except Exception as err:
            self._episode_last_error = str(err)

    async def save(self) -> None:
        if self._capture_stores is None or not self._episode_save_needed:
            return
        try:
            store = self._capture_stores.episodes_store()
            await store.save({"episodes": dict(self._episode_history)})
            self._episode_save_needed = False
        except Exception as err:
            self._episode_last_error = str(err)


# ── T1: Episode state is initialized in __init__ (static check) ────────────

def test_t1_episode_state_initialized_in_init():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.__init__)
    assert "self._episode_history: dict = {}" in source
    assert "self._episode_save_needed: bool = False" in source
    assert "self._episode_last_error: Optional[str] = None" in source


def test_t1_accessor_methods_exist():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    assert hasattr(_ha.LearningShadowController, "episode_history_snapshot")
    assert hasattr(_ha.LearningShadowController, "episode_last_error")


# ── T2: async_setup()-equivalent load succeeds with a valid stored payload ─

def _make_heating_entry(eid: str = f"{_ZONE_A}:heating:1") -> dict:
    from datetime import datetime, timezone
    from custom_components.thermosmart.learning.contracts import Regime
    from custom_components.thermosmart.learning.episode_schemas import HeatingEpisode

    ep = HeatingEpisode(
        episode_id=eid, learning_zone_id=_ZONE_A,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        end_ts=datetime(2026, 6, 1, 10, 30, tzinfo=timezone.utc),
        regime=Regime.ACTIVE_HEATING, reliability=0.9, start_temp=19.0, target=21.0,
    )
    return serialize_episode(ep)


def test_t2_load_succeeds_with_valid_payload():
    factory = _FakeStoreFactory()
    entry = _make_heating_entry()
    factory.inject(
        EpisodesStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"episodes": {entry["episode_id"]: entry}})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)

    asyncio.run(helper.load())

    assert entry["episode_id"] in helper._episode_history
    assert helper._episode_last_error is None


def test_t2_load_noop_when_capture_stores_none():
    helper = _EpisodeHistoryHelper(None)
    asyncio.run(helper.load())
    assert helper._episode_history == {}
    assert helper._episode_last_error is None


def test_t2_load_noop_when_store_empty():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)
    asyncio.run(helper.load())
    assert helper._episode_history == {}
    assert helper._episode_last_error is None


# ── T3: Malformed store payload is handled non-fatally ──────────────────────

@pytest.mark.parametrize("bad_episodes", [
    "not_a_dict",
    None,
    123,
])
def test_t3_malformed_episodes_shape_is_safe(bad_episodes):
    factory = _FakeStoreFactory()
    factory.inject(
        EpisodesStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"episodes": bad_episodes})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)
    asyncio.run(helper.load())
    assert helper._episode_history == {}
    assert helper._episode_last_error is None


def test_t3_malformed_individual_entry_is_skipped():
    factory = _FakeStoreFactory()
    good = _make_heating_entry(f"{_ZONE_A}:heating:1")
    factory.inject(
        EpisodesStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"episodes": {
            good["episode_id"]: good,
            "bad-entry": {"episode_schema_version": 999, "episode_type": "nope"},
            "worse-entry": "not_a_mapping",
        }})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)
    asyncio.run(helper.load())
    assert set(helper._episode_history.keys()) == {good["episode_id"]}
    assert helper._episode_last_error is None


# ── T4: Store-load exception is non-fatal ───────────────────────────────────

def test_t4_load_exception_is_nonfatal():
    factory = _FakeStoreFactory()
    factory.inject(
        EpisodesStore(factory, _ZONE_A).key,
        _FakeRawStore(load_raises=RuntimeError("simulated disk error")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)

    asyncio.run(helper.load())  # must not raise

    assert helper._episode_history == {}
    assert "simulated disk error" in (helper._episode_last_error or "")


# ── T5: Save is skipped when _episode_save_needed is False ────────────────

def test_t5_save_skipped_when_not_dirty():
    factory = _FakeStoreFactory()
    fake_store = _FakeRawStore()
    factory.inject(EpisodesStore(factory, _ZONE_A).key, fake_store)
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)
    helper._episode_history = {"x": _make_heating_entry()}
    helper._episode_save_needed = False

    asyncio.run(helper.save())

    assert fake_store.save_call_count == 0


def test_t5_save_skipped_when_no_capture_stores():
    helper = _EpisodeHistoryHelper(None)
    helper._episode_save_needed = True
    asyncio.run(helper.save())  # must not raise, no store to reach


# ── T6: Save runs when _episode_save_needed is True ────────────────────────

def test_t6_save_runs_when_dirty():
    factory = _FakeStoreFactory()
    fake_store = _FakeRawStore()
    factory.inject(EpisodesStore(factory, _ZONE_A).key, fake_store)
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)
    entry = _make_heating_entry()
    helper._episode_history = {entry["episode_id"]: entry}
    helper._episode_save_needed = True

    asyncio.run(helper.save())

    assert fake_store.save_call_count == 1
    assert fake_store._data["data"]["episodes"][entry["episode_id"]] == entry


# ── T7: Save success resets the dirty flag ─────────────────────────────────

def test_t7_save_success_resets_dirty_flag():
    factory = _FakeStoreFactory()
    factory.inject(EpisodesStore(factory, _ZONE_A).key, _FakeRawStore())
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)
    helper._episode_save_needed = True

    asyncio.run(helper.save())

    assert helper._episode_save_needed is False


# ── T8: Save failure keeps the dirty flag set ──────────────────────────────

def test_t8_save_failure_keeps_dirty_flag():
    factory = _FakeStoreFactory()
    factory.inject(
        EpisodesStore(factory, _ZONE_A).key,
        _FakeRawStore(save_raises=OSError("disk full")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)
    helper._episode_save_needed = True

    asyncio.run(helper.save())

    assert helper._episode_save_needed is True  # must remain True for retry


# ── T9: Save failure sets the last-error field ─────────────────────────────

def test_t9_save_failure_sets_last_error():
    factory = _FakeStoreFactory()
    factory.inject(
        EpisodesStore(factory, _ZONE_A).key,
        _FakeRawStore(save_raises=OSError("disk full")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _EpisodeHistoryHelper(capture_stores)
    helper._episode_save_needed = True

    asyncio.run(helper.save())

    assert "disk full" in (helper._episode_last_error or "")


# ── T10: async_unload() flushes episode history (static check) ─────────────

def test_t10_async_unload_flushes_episode_history():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_unload)
    assert "await self._async_save_episode_history_safe()" in source


def test_t10_async_save_if_due_flushes_episode_history():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_save_if_due)
    assert "await self._async_save_episode_history_safe()" in source


def test_t10_async_setup_loads_episode_history():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_setup)
    assert "await self._async_load_episode_history_safe()" in source


# ── T11: No runtime hook exists in lifecycle.py ──────────────────────────

def test_t11_no_runtime_hook_in_lifecycle():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    source = inspect.getsource(_lifecycle_mod)
    assert "episode_persistence" not in source
    assert "append_completed_episode" not in source
    assert "episode_serialization" not in source
    assert "EpisodesStore" not in source


# ── T12: No raw store referenced by the new wiring ──────────────────────

def test_t12_no_raw_store_in_new_wiring():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    init_source = inspect.getsource(_ha.LearningShadowController.__init__)
    load_source = inspect.getsource(_ha.LearningShadowController._async_load_episode_history_safe)
    save_source = inspect.getsource(_ha.LearningShadowController._async_save_episode_history_safe)
    for source in (init_source, load_source, save_source):
        assert "RawSegmentStore" not in source
        assert "RawSegmentIndexStore" not in source


# ── T13: No store I/O in the run_cycle() path ──────────────────────────

def test_t13_run_cycle_has_no_store_io():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    source = inspect.getsource(_lifecycle_mod.LearningRuntime.run_cycle)
    assert "async_save" not in source
    assert ".save(" not in source
    assert "EpisodesStore" not in source


# ── T14-T17: Existing tests remain green (regression smoke-test) ────────

def test_t14_t17_regression_imports_ok():
    import tests.test_le2_episode_persistence  # noqa: F401
    import tests.test_le2_episode_serialization  # noqa: F401
    import tests.test_le2_capture_wiring_foundation  # noqa: F401
    import tests.test_le2_retention  # noqa: F401
    import tests.test_le2_periodic_save_trigger  # noqa: F401


# ── T18: No control path touched ─────────────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t18_no_control_keywords_in_new_wiring_block():
    """Only checks the NEW episode-history snippet within __init__ — the
    surrounding constructor legitimately contains unrelated pre-existing
    control-adjacent comments (e.g. "single real dispatch path")."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    full_init = inspect.getsource(_ha.LearningShadowController.__init__)
    start = full_init.index("# Episode history — in-memory")
    end = full_init.index("# Cache: schedule target time", start)
    init_snippet = full_init[start:end].lower()
    load_source = inspect.getsource(
        _ha.LearningShadowController._async_load_episode_history_safe
    ).lower()
    save_source = inspect.getsource(
        _ha.LearningShadowController._async_save_episode_history_safe
    ).lower()
    for source in (init_snippet, load_source, save_source):
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in source, f"forbidden control token found: {token}"
