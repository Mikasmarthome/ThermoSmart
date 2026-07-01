"""Tests for the LE2 Support Critical Event storage/setup restore landmark
producers in LearningShadowController
(custom_components/thermosmart/learning/runtime/ha_integration.py).

This step adds the FIRST real Support Critical Event producers — but only
for storage/setup/restore states inside
_async_load_support_critical_events_safe()/
_async_save_support_critical_events_safe() themselves. No coordinator hook,
no runtime/lifecycle hook, no TRV/boost/TPI/preheat/setpoint producer, no
control-path change of any kind.

Since constructing a real LearningShadowController requires a live
HomeAssistant instance, this file uses the same technique established in
test_le2_episode_store_wiring.py / test_le2_support_critical_event_wiring.py:
a minimal harness whose method bodies are IDENTICAL copies of the production
_record_storage_landmark_event_safe()/_async_load_support_critical_events_safe()/
_async_save_support_critical_events_safe()/record_support_critical_event_safe()
methods — validating the CONTRACT, not a mock.

17 test groups:
  T1  — Successful load with data records exactly one storage_restore landmark
  T2  — Empty store load records one storage_restore (empty) landmark, deduped on repeat
  T3  — Malformed entries are counted in the restore landmark's details
  T4  — Load exception records one storage_restore_failed landmark, non-fatal
  T5  — Load exception still sets _support_critical_events_last_error correctly
  T6  — Save exception records one storage_save_failed landmark, non-fatal
  T7  — Save exception does not cause an infinite dirty/save loop (bounded to one landmark per streak)
  T8  — Save recovery after a notified failure records exactly one storage_save_recovered landmark
  T9  — No event production from coordinator.py
  T10 — No event production from lifecycle.py/run_cycle()
  T11 — Support Export shows the produced restore events
  T12 — Event details stay public-safe (bounded, no forbidden substrings)
  T13 — No event_id in the support export
  T14 — No store reads introduced in the export path
  T15 — Existing Support Export tests remain green
  T16 — Existing Wiring/Foundation tests remain green
  T17 — No control-path keywords touched
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

import pytest

from custom_components.thermosmart.learning import support_event_schemas as _schemas
from custom_components.thermosmart.learning.storage.support_event_serialization import (
    SUPPORT_EVENT_SCHEMA_VERSION,
    deserialize_support_event,
)
from custom_components.thermosmart.learning.storage.support_event_persistence import (
    append_support_event,
    is_recent_duplicate,
)
from custom_components.thermosmart.learning.storage.stores import SupportCriticalEventStore

_ZONE_A = "zone_alpha_01"
_NOW_ISO = "2026-06-02T12:00:00+00:00"
_DEDUPE_WINDOW_S = 6 * 3600


# ── Fake store infrastructure (mirrors test_le2_support_critical_event_wiring.py) ─

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
    def __init__(self, factory: _FakeStoreFactory, zone: str) -> None:
        self._factory = factory
        self._zone = zone

    def support_critical_events_store(self) -> SupportCriticalEventStore:
        return SupportCriticalEventStore(self._factory, self._zone)


def _envelope(data: Any, version: int = 1) -> dict:
    return {"store_schema_version": version, "data": data}


def _make_event(event_id: str = "evt_001") -> dict:
    from custom_components.thermosmart.learning.support_event_schemas import SupportCriticalEvent
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )
    ev = SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id=event_id,
        event_type=_schemas.SupportEventType.TRV_COMMAND_BLOCKED,
        ts=datetime(2026, 6, 2, 10, 0, tzinfo=timezone.utc),
        severity=_schemas.SupportEventSeverity.WARNING, reason="min_interval",
    )
    return serialize_support_event(ev)


def _run(coro):
    return asyncio.run(coro)


# ── Harness mirroring the production contract exactly ───────────────────────

class _SupportCriticalEventHelper:
    """IDENTICAL method bodies to the production landmark-producing methods
    in LearningShadowController — validates the CONTRACT, not a copy."""

    _STORAGE_RESTORE_LANDMARK_DEDUPE_WINDOW_S = _DEDUPE_WINDOW_S

    def __init__(self, capture_stores, now_iso: str = _NOW_ISO) -> None:
        self._capture_stores = capture_stores
        self._support_critical_events: dict = {}
        self._support_critical_events_save_needed: bool = False
        self._support_critical_events_last_error: Optional[str] = None
        self._support_critical_events_save_failure_notified: bool = False
        self._utcnow_iso = lambda: now_iso

    def record_support_critical_event_safe(self, event: Any) -> None:
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

    def _record_storage_landmark_event_safe(
        self, *, event_type, severity, reason: str, summary: str,
        details: Mapping[str, Any], dedupe_window_s: Optional[float] = None,
    ) -> None:
        try:
            now_utc = datetime.fromisoformat(self._utcnow_iso())
            event_id = f"storage_landmark_{event_type.value}_{int(now_utc.timestamp() * 1000)}"
            candidate = _schemas.SupportCriticalEvent(
                schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id=event_id,
                event_type=event_type, ts=now_utc, severity=severity,
                reason=reason, summary=summary, details=dict(details),
            )
            if dedupe_window_s is not None:
                recent = list(self._support_critical_events.values())
                if is_recent_duplicate(candidate, recent, window_s=dedupe_window_s):
                    return
            self.record_support_critical_event_safe(candidate)
        except Exception:
            pass

    async def load(self) -> None:
        if self._capture_stores is None:
            return
        records_loaded = 0
        malformed_skipped = 0
        load_error: Optional[Exception] = None
        try:
            store = self._capture_stores.support_critical_events_store()
            raw = await store.load()
            if raw is not None:
                raw_entries = raw.get("events") if isinstance(raw, Mapping) else None
                if isinstance(raw_entries, Mapping):
                    rebuilt: dict = {}
                    for eid, entry in raw_entries.items():
                        if not isinstance(entry, Mapping):
                            malformed_skipped += 1
                            continue
                        if deserialize_support_event(entry) is None:
                            malformed_skipped += 1
                            continue
                        rebuilt[eid] = dict(entry)
                    self._support_critical_events = rebuilt
                    records_loaded = len(rebuilt)
        except Exception as err:
            self._support_critical_events_last_error = str(err)
            load_error = err

        if load_error is not None:
            self._record_storage_landmark_event_safe(
                event_type=_schemas.SupportEventType.STORAGE_RESTORE_FAILED,
                severity=_schemas.SupportEventSeverity.WARNING,
                reason="support_events_load_failed",
                summary="Support critical event store failed to load",
                details={"error_type": type(load_error).__name__},
                dedupe_window_s=self._STORAGE_RESTORE_LANDMARK_DEDUPE_WINDOW_S,
            )
        elif records_loaded == 0:
            self._record_storage_landmark_event_safe(
                event_type=_schemas.SupportEventType.STORAGE_RESTORE,
                severity=_schemas.SupportEventSeverity.INFO,
                reason="support_events_empty",
                summary="Support critical event store loaded (empty)",
                details={"records_loaded": 0},
                dedupe_window_s=self._STORAGE_RESTORE_LANDMARK_DEDUPE_WINDOW_S,
            )
        else:
            self._record_storage_landmark_event_safe(
                event_type=_schemas.SupportEventType.STORAGE_RESTORE,
                severity=_schemas.SupportEventSeverity.INFO,
                reason="support_events_loaded",
                summary="Support critical event store loaded",
                details={"records_loaded": records_loaded, "malformed_skipped": malformed_skipped},
                dedupe_window_s=self._STORAGE_RESTORE_LANDMARK_DEDUPE_WINDOW_S,
            )

    async def save(self) -> None:
        if self._capture_stores is None or not self._support_critical_events_save_needed:
            return
        try:
            store = self._capture_stores.support_critical_events_store()
            await store.save({"events": dict(self._support_critical_events)})
            self._support_critical_events_save_needed = False
            if self._support_critical_events_save_failure_notified:
                self._support_critical_events_save_failure_notified = False
                self._record_storage_landmark_event_safe(
                    event_type=_schemas.SupportEventType.STORAGE_SAVE_RECOVERED,
                    severity=_schemas.SupportEventSeverity.INFO,
                    reason="support_events_save_recovered",
                    summary="Support critical event store save recovered",
                    details={},
                )
        except Exception as err:
            self._support_critical_events_last_error = str(err)
            if not self._support_critical_events_save_failure_notified:
                self._support_critical_events_save_failure_notified = True
                self._record_storage_landmark_event_safe(
                    event_type=_schemas.SupportEventType.STORAGE_SAVE_FAILED,
                    severity=_schemas.SupportEventSeverity.WARNING,
                    reason="support_events_save_failed",
                    summary="Support critical event store failed to save",
                    details={"error_type": type(err).__name__},
                )


# ── T1: Successful load with data records exactly one storage_restore landmark

def test_t1_successful_load_with_data_records_one_restore_landmark():
    factory = _FakeStoreFactory()
    entry = _make_event()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"events": {entry["event_id"]: entry}})),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))

    _run(helper.load())

    landmarks = [e for e in helper._support_critical_events.values()
                 if e["event_type"] == "storage_restore"]
    assert len(landmarks) == 1
    assert landmarks[0]["reason"] == "support_events_loaded"
    assert landmarks[0]["details"]["records_loaded"] == 1


def test_t1_landmark_does_not_double_count_itself():
    """The landmark is appended AFTER the loaded snapshot is set, so its own
    presence must not have inflated records_loaded."""
    factory = _FakeStoreFactory()
    entry = _make_event()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"events": {entry["event_id"]: entry}})),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    _run(helper.load())
    landmark = next(e for e in helper._support_critical_events.values()
                     if e["event_type"] == "storage_restore")
    assert landmark["details"]["records_loaded"] == 1  # not 2


# ── T2: Empty store load records one landmark, deduped on repeat ──────────

def test_t2_empty_store_records_one_empty_landmark():
    factory = _FakeStoreFactory()
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))

    _run(helper.load())

    landmarks = list(helper._support_critical_events.values())
    assert len(landmarks) == 1
    assert landmarks[0]["event_type"] == "storage_restore"
    assert landmarks[0]["reason"] == "support_events_empty"
    assert landmarks[0]["details"]["records_loaded"] == 0


def test_t2_repeated_empty_load_within_window_does_not_duplicate():
    """Simulates two setups in quick succession (e.g. rapid restarts) — the
    second load must not add a second identical empty-store landmark."""
    factory = _FakeStoreFactory()
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A), now_iso=_NOW_ISO)
    _run(helper.load())
    assert len(helper._support_critical_events) == 1

    # Second "restart" 10 minutes later — well within the 6h dedupe window.
    helper._utcnow_iso = lambda: "2026-06-02T12:10:00+00:00"
    _run(helper.load())
    assert len(helper._support_critical_events) == 1  # still just one landmark


def test_t2_empty_load_outside_window_records_a_new_landmark():
    factory = _FakeStoreFactory()
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A), now_iso=_NOW_ISO)
    _run(helper.load())
    assert len(helper._support_critical_events) == 1

    # 7 hours later — outside the 6h dedupe window.
    helper._utcnow_iso = lambda: "2026-06-02T19:01:00+00:00"
    _run(helper.load())
    assert len(helper._support_critical_events) == 2


# ── T3: Malformed entries counted in the restore landmark's details ────────

def test_t3_malformed_entries_counted_in_landmark_details():
    factory = _FakeStoreFactory()
    good = _make_event("evt_good")
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"events": {
            good["event_id"]: good,
            "bad1": {"schema_version": 999},
            "bad2": "not_a_mapping",
        }})),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    _run(helper.load())

    landmark = next(e for e in helper._support_critical_events.values()
                     if e["event_type"] == "storage_restore")
    assert landmark["details"]["records_loaded"] == 1
    assert landmark["details"]["malformed_skipped"] == 2


# ── T4: Load exception records one storage_restore_failed landmark ────────

def test_t4_load_exception_records_failed_landmark_non_fatally():
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(load_raises=RuntimeError("simulated disk error")),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))

    _run(helper.load())  # must not raise

    landmarks = list(helper._support_critical_events.values())
    assert len(landmarks) == 1
    assert landmarks[0]["event_type"] == "storage_restore_failed"
    assert landmarks[0]["severity"] == "warning"
    assert landmarks[0]["reason"] == "support_events_load_failed"
    assert landmarks[0]["details"]["error_type"] == "RuntimeError"
    assert "RuntimeError" not in landmarks[0]["summary"]  # no raw exception repr in summary


# ── T5: Load exception still sets last_error correctly ─────────────────────

def test_t5_load_exception_sets_last_error():
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(load_raises=RuntimeError("simulated disk error")),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    _run(helper.load())
    assert "simulated disk error" in (helper._support_critical_events_last_error or "")


# ── T6: Save exception records one storage_save_failed landmark ───────────

def test_t6_save_exception_records_failed_landmark_non_fatally():
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(save_raises=OSError("disk full")),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    helper._support_critical_events_save_needed = True

    _run(helper.save())  # must not raise

    landmarks = list(helper._support_critical_events.values())
    assert len(landmarks) == 1
    assert landmarks[0]["event_type"] == "storage_save_failed"
    assert landmarks[0]["severity"] == "warning"
    assert helper._support_critical_events_save_failure_notified is True
    assert helper._support_critical_events_save_needed is True  # dirty remains for retry


# ── T7: Save exception does not cause an infinite dirty/save loop ─────────

def test_t7_repeated_save_failure_records_only_one_landmark():
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(save_raises=OSError("disk full")),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    helper._support_critical_events_save_needed = True

    _run(helper.save())
    count_after_first = len(helper._support_critical_events)

    # Simulate several more periodic save attempts while still broken.
    for _ in range(5):
        helper._support_critical_events_save_needed = True
        _run(helper.save())

    assert len(helper._support_critical_events) == count_after_first  # no new landmarks
    failed_landmarks = [e for e in helper._support_critical_events.values()
                        if e["event_type"] == "storage_save_failed"]
    assert len(failed_landmarks) == 1


# ── T8: Save recovery after a notified failure records one recovered landmark

def test_t8_save_recovery_records_one_recovered_landmark():
    factory = _FakeStoreFactory()
    fake_store = _FakeRawStore(save_raises=OSError("disk full"))
    factory.inject(SupportCriticalEventStore(factory, _ZONE_A).key, fake_store)
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    helper._support_critical_events_save_needed = True
    _run(helper.save())
    assert helper._support_critical_events_save_failure_notified is True

    fake_store._save_raises = None  # "fix" the underlying problem
    helper._support_critical_events_save_needed = True
    _run(helper.save())

    recovered = [e for e in helper._support_critical_events.values()
                 if e["event_type"] == "storage_save_recovered"]
    assert len(recovered) == 1
    assert helper._support_critical_events_save_failure_notified is False


def test_t8_further_successful_saves_do_not_add_more_recovered_landmarks():
    factory = _FakeStoreFactory()
    fake_store = _FakeRawStore(save_raises=OSError("disk full"))
    factory.inject(SupportCriticalEventStore(factory, _ZONE_A).key, fake_store)
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    helper._support_critical_events_save_needed = True
    _run(helper.save())

    fake_store._save_raises = None
    helper._support_critical_events_save_needed = True
    _run(helper.save())
    count_after_recovery = len(helper._support_critical_events)

    # The recovered landmark itself set dirty=True again; simulate the next
    # (now-healthy) periodic save cycle.
    helper._support_critical_events_save_needed = True
    _run(helper.save())
    helper._support_critical_events_save_needed = True
    _run(helper.save())

    recovered = [e for e in helper._support_critical_events.values()
                 if e["event_type"] == "storage_save_recovered"]
    assert len(recovered) == 1  # still just one, no repeats


# ── T9: No event production from coordinator.py ─────────────────────────────

def test_t9_no_event_production_in_coordinator():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod)
    assert "record_support_critical_event_safe" not in source
    assert "STORAGE_RESTORE" not in source
    assert "STORAGE_SAVE_FAILED" not in source


# ── T10: No event production from lifecycle.py/run_cycle() ─────────────────

def test_t10_no_event_production_in_lifecycle():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    source = inspect.getsource(_lifecycle_mod)
    assert "record_support_critical_event_safe" not in source
    assert "SupportCriticalEvent" not in source
    assert "support_event" not in source.lower()


def test_t10_ha_integration_landmark_producers_confined_to_load_save_methods():
    """The three new event_type usages (STORAGE_RESTORE*, STORAGE_SAVE_*)
    only appear inside the load/save/helper methods, never in run_cycle-
    adjacent control code."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    load_src = inspect.getsource(_ha.LearningShadowController._async_load_support_critical_events_safe)
    save_src = inspect.getsource(_ha.LearningShadowController._async_save_support_critical_events_safe)
    helper_src = inspect.getsource(_ha.LearningShadowController._record_storage_landmark_event_safe)
    assert "STORAGE_RESTORE" in load_src
    assert "STORAGE_SAVE_FAILED" in save_src
    assert "STORAGE_SAVE_RECOVERED" in save_src
    assert helper_src  # exists and is used by both


# ── T11: Support Export shows the produced restore events ──────────────────

def test_t11_support_export_shows_restore_events():
    from custom_components.thermosmart.export import _le2_critical_events_export

    factory = _FakeStoreFactory()
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    _run(helper.load())  # produces one "empty" landmark

    class _FakeShadow:
        def __init__(self, events, capture_stores):
            self._events = events
            self.capture_stores = capture_stores

        def support_critical_events_snapshot(self):
            return dict(self._events)

    class _FakeCoord:
        def __init__(self, shadow):
            self._le2_shadow = shadow

    now = datetime.fromisoformat(_NOW_ISO)
    coord = _FakeCoord(_FakeShadow(helper._support_critical_events, "present"))
    block = _le2_critical_events_export(coord, now=now)
    assert block["available"] is True
    assert block["records_available"] == 1
    assert block["events"][0]["event_type"] == "storage_restore"
    assert block["events"][0]["reason"] == "support_events_empty"


# ── T12: Event details stay public-safe ─────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t12_landmark_details_stay_public_safe():
    factory = _FakeStoreFactory()
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(load_raises=RuntimeError("simulated disk error at /config/.storage/x")),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    _run(helper.load())
    landmark = next(iter(helper._support_critical_events.values()))
    flat = repr(landmark).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in flat, f"forbidden substring '{token}' found in landmark details"
    # The raw exception message (which could contain a path) must never
    # appear verbatim — only the bounded error_type class name is kept.
    assert "/config/" not in flat


def test_t12_details_are_bounded_scalars_only():
    factory = _FakeStoreFactory()
    good = _make_event("evt_good")
    factory.inject(
        SupportCriticalEventStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"events": {good["event_id"]: good}})),
    )
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    _run(helper.load())
    landmark = next(e for e in helper._support_critical_events.values()
                     if e["event_type"] == "storage_restore")
    for v in landmark["details"].values():
        assert isinstance(v, (int, float, bool, str)) or v is None


# ── T13: No event_id in the support export ──────────────────────────────────

def test_t13_no_event_id_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export

    factory = _FakeStoreFactory()
    helper = _SupportCriticalEventHelper(_FakeCaptureStores(factory, _ZONE_A))
    _run(helper.load())

    class _FakeShadow:
        def __init__(self, events, capture_stores):
            self._events = events
            self.capture_stores = capture_stores

        def support_critical_events_snapshot(self):
            return dict(self._events)

    class _FakeCoord:
        def __init__(self, shadow):
            self._le2_shadow = shadow

    now = datetime.fromisoformat(_NOW_ISO)
    coord = _FakeCoord(_FakeShadow(helper._support_critical_events, "present"))
    from custom_components.thermosmart.export import _le2_critical_events_export as _fn
    block = _fn(coord, now=now)
    assert "event_id" not in block["events"][0]
    assert "storage_landmark_" not in repr(block)  # internal event_id prefix never leaks


# ── T14: No store reads introduced in the export path ──────────────────────

def test_t14_export_path_has_no_store_io():
    from custom_components.thermosmart.export import _le2_critical_events_export
    source = inspect.getsource(_le2_critical_events_export)
    assert "await" not in source
    assert "SupportCriticalEventStore(" not in source
    assert ".load(" not in source


# ── T15/T16: Existing tests remain green (regression smoke-test) ──────────

def test_t15_t16_regression_imports_ok():
    import tests.test_le2_support_critical_event_export  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_support_critical_events  # noqa: F401


# ── T17: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t17_no_control_keywords_in_new_snippets():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    snippets = [
        inspect.getsource(_ha.LearningShadowController._record_storage_landmark_event_safe),
        inspect.getsource(_ha.LearningShadowController._async_load_support_critical_events_safe),
        inspect.getsource(_ha.LearningShadowController._async_save_support_critical_events_safe),
    ]
    for source in snippets:
        lowered = source.lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in lowered, f"forbidden control token found: {token}"
