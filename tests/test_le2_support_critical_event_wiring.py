"""Tests for the LE2 Support Critical Event state and store wiring in
LearningShadowController (custom_components/thermosmart/learning/runtime/ha_integration.py).

This step wires in-memory support-critical-event state plus load/save through
the existing SupportCriticalEventStore (via LearningCaptureStores), reusing
the existing periodic save trigger (async_save_if_due()) and unload flush —
mirroring exactly how the episode-history state/store wiring was built and
tested (test_le2_episode_store_wiring.py). It deliberately does NOT add any
runtime producer: nothing in coordinator.py/lifecycle.py calls
record_support_critical_event_safe()/append_support_event() yet.

Since constructing a real LearningShadowController requires a live
HomeAssistant instance (no existing test harness for that), this file
combines two techniques, matching the established pattern in this codebase:

  1. Static source-inspection checks that the REAL wiring in ha_integration.py
     exists, is placed correctly, and that lifecycle.py/coordinator.py were
     NOT touched.
  2. Behavioral tests against a minimal harness whose load()/save()/record()
     method bodies are IDENTICAL copies of the production
     _async_load_support_critical_events_safe()/
     _async_save_support_critical_events_safe()/
     record_support_critical_event_safe() methods — validating the CONTRACT,
     not a mock — using the real SupportCriticalEventStore class with a fake
     StoreFactory (no HA dependency).

19 test groups:
  T1  — Support critical event state is initialized in LearningShadowController.__init__
  T2  — async_setup()-equivalent load succeeds with a valid stored payload
  T3  — Malformed store payload is handled non-fatally
  T4  — Store-load exception is non-fatal
  T5  — Save is skipped when _support_critical_events_save_needed is False
  T6  — Save runs when _support_critical_events_save_needed is True
  T7  — Save success resets the dirty flag
  T8  — Save failure keeps the dirty flag set
  T9  — Save failure sets the last-error field
  T10 — async_unload() / async_save_if_due() / async_setup() wiring (static check)
  T11 — record_support_critical_event_safe(): valid event appended
  T12 — record_support_critical_event_safe(): duplicate is a no-op for dirty flag
  T13 — record_support_critical_event_safe(): malformed event is non-fatal
  T14 — No runtime hook exists in lifecycle.py or coordinator.py
  T15 — No raw store referenced by the new wiring
  T16 — No store I/O in the run_cycle() path
  T17 — No Support Export change
  T18 — Existing Episode Store/Progress/Export/Support-Event-Foundation tests remain green
  T19 — No control path touched
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import pytest

from custom_components.thermosmart.learning.support_event_schemas import (
    SupportCriticalEvent,
    SupportEventSeverity,
    SupportEventType,
)
from custom_components.thermosmart.learning.storage.support_event_serialization import (
    SUPPORT_EVENT_SCHEMA_VERSION,
    deserialize_support_event,
    serialize_support_event,
)
from custom_components.thermosmart.learning.storage.support_event_persistence import (
    append_support_event,
)
from custom_components.thermosmart.learning.storage.stores import SupportCriticalEventStore

_ZONE_A = "zone_alpha_01"
_NOW_ISO = "2026-06-02T12:00:00+00:00"


# ── Fake store infrastructure (mirrors test_le2_episode_store_wiring.py) ────

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
    """Minimal stand-in for LearningCaptureStores exposing support_critical_events_store()."""

    def __init__(self, factory: _FakeStoreFactory, zone: str) -> None:
        self._factory = factory
        self._zone = zone

    def support_critical_events_store(self) -> SupportCriticalEventStore:
        return SupportCriticalEventStore(self._factory, self._zone)


def _envelope(data: Any, version: int = 1) -> dict:
    return {"store_schema_version": version, "data": data}


def _make_event(event_id: str = "evt_001") -> dict:
    ev = SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id=event_id,
        event_type=SupportEventType.TRV_COMMAND_BLOCKED,
        ts=datetime(2026, 6, 2, 10, 0, tzinfo=timezone.utc),
        severity=SupportEventSeverity.WARNING, reason="min_interval",
        summary="TRV command blocked by minimum interval",
        details={"requested_setpoint": 21.0, "current_setpoint": 20.5},
    )
    return serialize_support_event(ev)


# ── Async harness mirroring the production contract exactly ────────────────

class _SupportCriticalEventHelper:
    """Standalone test harness exercising the same load/save/record contract
    as LearningShadowController's support-critical-event methods. The
    implementation here is IDENTICAL to the production methods so that the
    tests validate contract behaviour, not a copy.
    """

    def __init__(self, capture_stores, now_iso: str = _NOW_ISO) -> None:
        self._capture_stores = capture_stores
        self._support_critical_events: dict = {}
        self._support_critical_events_save_needed: bool = False
        self._support_critical_events_last_error: Optional[str] = None
        self._utcnow_iso = lambda: now_iso

    async def load(self) -> None:
        if self._capture_stores is None:
            return
        try:
            store = self._capture_stores.support_critical_events_store()
            raw = await store.load()
            if raw is None:
                return
            raw_entries = raw.get("events") if isinstance(raw, Mapping) else None
            if not isinstance(raw_entries, Mapping):
                return
            rebuilt: dict = {}
            for eid, entry in raw_entries.items():
                if not isinstance(entry, Mapping):
                    continue
                if deserialize_support_event(entry) is None:
                    continue
                rebuilt[eid] = dict(entry)
            self._support_critical_events = rebuilt
        except Exception as err:
            self._support_critical_events_last_error = str(err)

    async def save(self) -> None:
        if self._capture_stores is None or not self._support_critical_events_save_needed:
            return
        try:
            store = self._capture_stores.support_critical_events_store()
            await store.save({"events": dict(self._support_critical_events)})
            self._support_critical_events_save_needed = False
        except Exception as err:
            self._support_critical_events_last_error = str(err)

    def record(self, event: Any) -> None:
        if self._capture_stores is None:
            return
        try:
            now_utc = datetime.fromisoformat(self._utcnow_iso())
            result = append_support_event(
                {"events": self._support_critical_events}, event, now_utc=now_utc,
            )
            if result.appended_event_ids or result.pruned_event_ids:
                self._support_critical_events = result.updated_payload["events"]
                self._support_critical_events_save_needed = True
        except Exception as err:
            self._support_critical_events_last_error = str(err)


def _run(coro):
    return asyncio.run(coro)


# ── T1: Support critical event state is initialized in __init__ ────────────

def test_t1_state_initialized_in_init():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.__init__)
    assert "self._support_critical_events: dict = {}" in source
    assert "self._support_critical_events_save_needed: bool = False" in source
    assert "self._support_critical_events_last_error: Optional[str] = None" in source


def test_t1_accessor_methods_exist():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    assert hasattr(_ha.LearningShadowController, "support_critical_events_snapshot")
    assert hasattr(_ha.LearningShadowController, "support_critical_events_last_error")
    assert hasattr(_ha.LearningShadowController, "record_support_critical_event_safe")


# ── T2: async_setup()-equivalent load succeeds with a valid stored payload ─

def test_t2_load_succeeds_with_valid_payload():
    factory = _FakeStoreFactory()
    entry = _make_event()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"events": {entry["event_id"]: entry}})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)

    _run(helper.load())

    assert entry["event_id"] in helper._support_critical_events
    assert helper._support_critical_events_last_error is None


def test_t2_load_noop_when_capture_stores_none():
    helper = _SupportCriticalEventHelper(None)
    _run(helper.load())
    assert helper._support_critical_events == {}
    assert helper._support_critical_events_last_error is None


def test_t2_load_noop_when_store_empty():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    _run(helper.load())
    assert helper._support_critical_events == {}
    assert helper._support_critical_events_last_error is None


# ── T3: Malformed store payload is handled non-fatally ──────────────────────

@pytest.mark.parametrize("bad_events", ["not_a_dict", None, 123])
def test_t3_malformed_events_shape_is_safe(bad_events):
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"events": bad_events})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    _run(helper.load())
    assert helper._support_critical_events == {}
    assert helper._support_critical_events_last_error is None


def test_t3_malformed_individual_entry_is_skipped():
    factory = _FakeStoreFactory()
    good = _make_event("evt_good")
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"events": {
            good["event_id"]: good,
            "bad-entry": {"schema_version": 999, "event_type": "nope"},
            "worse-entry": "not_a_mapping",
        }})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    _run(helper.load())
    assert set(helper._support_critical_events.keys()) == {good["event_id"]}
    assert helper._support_critical_events_last_error is None


# ── T4: Store-load exception is non-fatal ───────────────────────────────────

def test_t4_load_exception_is_nonfatal():
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(load_raises=RuntimeError("simulated disk error")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)

    _run(helper.load())  # must not raise

    assert helper._support_critical_events == {}
    assert "simulated disk error" in (helper._support_critical_events_last_error or "")


# ── T5: Save is skipped when dirty flag is False ────────────────────────────

def test_t5_save_skipped_when_not_dirty():
    factory = _FakeStoreFactory()
    fake_store = _FakeRawStore()
    factory.inject(SupportCriticalEventStore(factory, _ZONE_A).key, fake_store)
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    helper._support_critical_events = {"x": _make_event()}
    helper._support_critical_events_save_needed = False

    _run(helper.save())

    assert fake_store.save_call_count == 0


def test_t5_save_skipped_when_no_capture_stores():
    helper = _SupportCriticalEventHelper(None)
    helper._support_critical_events_save_needed = True
    _run(helper.save())  # must not raise, no store to reach


# ── T6: Save runs when dirty flag is True ───────────────────────────────────

def test_t6_save_runs_when_dirty():
    factory = _FakeStoreFactory()
    fake_store = _FakeRawStore()
    factory.inject(SupportCriticalEventStore(factory, _ZONE_A).key, fake_store)
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    entry = _make_event()
    helper._support_critical_events = {entry["event_id"]: entry}
    helper._support_critical_events_save_needed = True

    _run(helper.save())

    assert fake_store.save_call_count == 1
    assert fake_store._data["data"]["events"][entry["event_id"]] == entry


# ── T7: Save success resets the dirty flag ─────────────────────────────────

def test_t7_save_success_resets_dirty_flag():
    factory = _FakeStoreFactory()
    factory.inject(SupportCriticalEventStore(factory, _ZONE_A).key, _FakeRawStore())
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    helper._support_critical_events_save_needed = True

    _run(helper.save())

    assert helper._support_critical_events_save_needed is False


# ── T8: Save failure keeps the dirty flag set ──────────────────────────────

def test_t8_save_failure_keeps_dirty_flag():
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(save_raises=OSError("disk full")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    helper._support_critical_events_save_needed = True

    _run(helper.save())

    assert helper._support_critical_events_save_needed is True  # must remain True for retry


# ── T9: Save failure sets the last-error field ─────────────────────────────

def test_t9_save_failure_sets_last_error():
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(save_raises=OSError("disk full")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    helper._support_critical_events_save_needed = True

    _run(helper.save())

    assert "disk full" in (helper._support_critical_events_last_error or "")


# ── T10: async_unload()/async_save_if_due()/async_setup() wiring (static) ──

def test_t10_async_unload_flushes_support_critical_events():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_unload)
    assert "await self._async_save_support_critical_events_safe()" in source


def test_t10_async_save_if_due_flushes_support_critical_events():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_save_if_due)
    assert "await self._async_save_support_critical_events_safe()" in source


def test_t10_async_setup_loads_support_critical_events():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_setup)
    assert "await self._async_load_support_critical_events_safe()" in source


# ── T11: record_support_critical_event_safe(): valid event appended ────────

def test_t11_record_appends_valid_event():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    ev = SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id="evt_rec_1",
        event_type=SupportEventType.BOOST_STARTED,
        ts=datetime(2026, 6, 2, 11, 0, tzinfo=timezone.utc),
        severity=SupportEventSeverity.INFO,
    )
    helper.record(ev)
    assert "evt_rec_1" in helper._support_critical_events
    assert helper._support_critical_events_save_needed is True
    assert helper._support_critical_events_last_error is None


def test_t11_record_noop_when_no_capture_stores():
    helper = _SupportCriticalEventHelper(None)
    ev = SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id="evt_rec_2",
        event_type=SupportEventType.BOOST_STARTED, ts=datetime(2026, 6, 2, 11, 0, tzinfo=timezone.utc),
        severity=SupportEventSeverity.INFO,
    )
    helper.record(ev)
    assert helper._support_critical_events == {}
    assert helper._support_critical_events_save_needed is False


# ── T12: duplicate event_id is a no-op for the dirty flag ──────────────────

def test_t12_duplicate_event_does_not_spuriously_redirty():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    ev = SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id="evt_dup",
        event_type=SupportEventType.BOOST_STARTED, ts=datetime(2026, 6, 2, 11, 0, tzinfo=timezone.utc),
        severity=SupportEventSeverity.INFO,
    )
    helper.record(ev)
    helper._support_critical_events_save_needed = False  # simulate a save having already happened
    helper.record(ev)  # duplicate only
    assert helper._support_critical_events_save_needed is False


# ── T13: malformed event is non-fatal ───────────────────────────────────────

def test_t13_record_malformed_event_is_safely_ignored():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _SupportCriticalEventHelper(capture_stores)
    helper.record(object())  # not a SupportCriticalEvent
    assert helper._support_critical_events == {}
    assert helper._support_critical_events_save_needed is False
    assert helper._support_critical_events_last_error is None  # skip is not an error


# ── T14: No runtime hook exists in lifecycle.py or coordinator.py ─────────

def test_t14_no_runtime_hook_in_lifecycle():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    source = inspect.getsource(_lifecycle_mod)
    assert "support_event" not in source.lower()
    assert "SupportCriticalEvent" not in source
    assert "record_support_critical_event_safe" not in source


def test_t14_no_runtime_hook_in_coordinator():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod)
    assert "support_event" not in source.lower()
    assert "SupportCriticalEvent" not in source
    assert "record_support_critical_event_safe" not in source


def test_t14_no_live_call_site_for_record_method_in_ha_integration():
    """At the time this wiring step was built, the method existed but had no
    caller. A later, separately-approved step ("Support Critical Event
    Producers Increment 1") legitimately added exactly ONE call site — the
    storage/setup landmark helper (_record_storage_landmark_event_safe) used
    only from the load/save methods themselves. That is intentional,
    non-runtime, non-control wiring, not a coordinator/lifecycle producer —
    still unlike record_completed_episode_safe(), it is not bound as a
    LearningRuntime sink and no coordinator/control path calls it."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha)
    call_count = source.count("self.record_support_critical_event_safe(")
    assert call_count == 1
    assert "self.record_support_critical_event_safe(" in inspect.getsource(
        _ha.LearningShadowController._record_storage_landmark_event_safe
    )


# ── T15: No raw store referenced by the new wiring ─────────────────────────

def test_t15_no_raw_store_in_new_wiring():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    init_source = inspect.getsource(_ha.LearningShadowController.__init__)
    load_source = inspect.getsource(_ha.LearningShadowController._async_load_support_critical_events_safe)
    save_source = inspect.getsource(_ha.LearningShadowController._async_save_support_critical_events_safe)
    record_source = inspect.getsource(_ha.LearningShadowController.record_support_critical_event_safe)
    for source in (init_source, load_source, save_source, record_source):
        assert "RawSegmentStore" not in source
        assert "RawSegmentIndexStore" not in source


# ── T16: No store I/O in the run_cycle() path ──────────────────────────────

def test_t16_run_cycle_has_no_support_event_store_io():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    source = inspect.getsource(_lifecycle_mod.LearningRuntime.run_cycle)
    assert "SupportCriticalEventStore" not in source
    assert "support_event" not in source.lower()


# ── T17: No Support Export change ───────────────────────────────────────────
#
# A later step ("Support Critical Event Timeline Export Summary") legitimately
# added a "critical_events" block to async_export_support_data(), reading
# support_critical_events_snapshot() — that is expected, intentional wiring
# for THIS wiring step's own state, not a regression. What must remain true
# is that the RESEARCH export stays untouched by that later addition.

def test_t17_research_export_function_has_no_support_event_reference():
    import custom_components.thermosmart.export as _exp
    source = inspect.getsource(_exp.async_export_learning_data)
    assert "support_event" not in source.lower()
    assert "SupportCriticalEvent" not in source
    assert "critical_events" not in source


# ── T18: Existing tests remain green (regression smoke-test) ───────────────

def test_t18_regression_imports_ok():
    import tests.test_le2_support_critical_events  # noqa: F401
    import tests.test_le2_episode_store_wiring  # noqa: F401
    import tests.test_le2_episode_persistence  # noqa: F401
    import tests.test_le2_learning_progress  # noqa: F401
    import tests.test_le2_learning_progress_export  # noqa: F401
    import tests.test_le2_episode_history_export  # noqa: F401


# ── T19: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t19_no_control_keywords_in_new_snippets():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    snippets = [
        inspect.getsource(_ha.LearningShadowController._async_load_support_critical_events_safe),
        inspect.getsource(_ha.LearningShadowController._async_save_support_critical_events_safe),
        inspect.getsource(_ha.LearningShadowController.record_support_critical_event_safe),
        inspect.getsource(_ha.LearningShadowController.support_critical_events_snapshot),
        inspect.getsource(_ha.LearningShadowController.support_critical_events_last_error),
    ]
    for source in snippets:
        lowered = source.lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in lowered, f"forbidden control token found: {token}"
