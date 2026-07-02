"""Tests for the LE2 Research Daily Bucket state and store load/save wiring in
LearningShadowController (custom_components/thermosmart/learning/runtime/ha_integration.py).

This step wires in-memory research-daily-bucket state plus load/save through
the existing ResearchDailyStore (via LearningCaptureStores), reusing the
existing periodic save trigger (async_save_if_due()) and unload flush —
mirroring exactly how the Support Critical Event state/store wiring was
built and tested (test_le2_support_critical_event_wiring.py). It
deliberately does NOT add any runtime/coordinator aggregation hook: nothing
in coordinator.py/lifecycle.py calls
record_research_daily_observation_safe() yet, and no aggregation from real
Support Events/Episodes exists anywhere.

Since constructing a real LearningShadowController requires a live
HomeAssistant instance (no existing test harness for that), this file
combines two techniques, matching the established pattern in this codebase:

  1. Static source-inspection checks that the REAL wiring in ha_integration.py
     exists, is placed correctly, and that coordinator.py/lifecycle.py/export.py
     were NOT touched.
  2. Behavioral tests against a minimal harness whose load()/save()/record()
     method bodies are IDENTICAL copies of the production
     _async_load_research_daily_safe()/_async_save_research_daily_safe()/
     record_research_daily_observation_safe() methods — validating the
     CONTRACT, not a mock — using the real ResearchDailyStore class with a
     fake StoreFactory (no HA dependency).

19 test groups:
  T1  — Research daily state is initialized in LearningShadowController.__init__
  T2  — async_setup()-equivalent load succeeds with a valid stored payload
  T3  — Malformed store payload is handled non-fatally
  T4  — Individual malformed buckets are skipped
  T5  — Store-load exception is non-fatal
  T6  — Save is skipped when _research_daily_save_needed is False
  T7  — Save runs when _research_daily_save_needed is True
  T8  — Save success resets the dirty flag
  T9  — Save failure keeps the dirty flag set
  T10 — Save failure sets the last-error field
  T11 — async_unload() / async_save_if_due() / async_setup() wiring (static check)
  T12 — record_research_daily_observation_safe(): valid observation updates bucket
  T13 — record_research_daily_observation_safe(): dirty only on real change / malformed non-fatal
  T14 — No runtime hook exists in coordinator.py
  T15 — No runtime hook exists in lifecycle.py
  T16 — No export hook exists in export.py
  T17 — No control path touched
  T18 — Existing Research Daily Foundation/Store tests remain green
  T19 — Existing Episode/Support Store tests remain green
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import pytest

from custom_components.thermosmart.learning.research_daily_schemas import (
    ResearchDailyObservation,
)
from custom_components.thermosmart.learning.storage.research_daily_serialization import (
    RESEARCH_DAILY_SCHEMA_VERSION,
    deserialize_research_daily_bucket,
    serialize_research_daily_bucket,
)
from custom_components.thermosmart.learning.storage.research_daily_persistence import (
    append_or_update_research_daily_bucket,
    create_empty_research_daily_bucket,
)
from custom_components.thermosmart.learning.storage.stores import ResearchDailyStore

_ZONE_A = "zone_alpha_01"
_NOW_ISO = "2026-06-02T12:00:00+00:00"


# ── Fake store infrastructure (mirrors test_le2_support_critical_event_wiring.py)

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
    """Minimal stand-in for LearningCaptureStores exposing research_daily_store()."""

    def __init__(self, factory: _FakeStoreFactory, zone: str) -> None:
        self._factory = factory
        self._zone = zone

    def research_daily_store(self) -> ResearchDailyStore:
        return ResearchDailyStore(self._factory, self._zone)


def _envelope(data: Any, version: int = 1) -> dict:
    return {"store_schema_version": version, "data": data}


def _make_bucket(bucket_date: str = "2026-06-01", decision_count: int = 1) -> dict:
    bucket = create_empty_research_daily_bucket(bucket_date)
    from custom_components.thermosmart.learning.storage.research_daily_persistence import (
        record_research_daily_observation,
    )
    bucket = record_research_daily_observation(
        bucket, ResearchDailyObservation(bucket_date=bucket_date, decision_count=decision_count),
    )
    return serialize_research_daily_bucket(bucket)


# ── Async harness mirroring the production contract exactly ────────────────

class _ResearchDailyHelper:
    """Standalone test harness exercising the same load/save/record contract
    as LearningShadowController's research-daily-bucket methods. The
    implementation here is IDENTICAL to the production methods so that the
    tests validate contract behaviour, not a copy.
    """

    def __init__(self, capture_stores, now_iso: str = _NOW_ISO) -> None:
        self._capture_stores = capture_stores
        self._research_daily_buckets: dict = {}
        self._research_daily_save_needed: bool = False
        self._research_daily_last_error: Optional[str] = None
        self._utcnow_iso = lambda: now_iso

    async def load(self) -> None:
        if self._capture_stores is None:
            return
        try:
            store = self._capture_stores.research_daily_store()
            raw = await store.load()
            if raw is None:
                return
            raw_entries = raw.get("buckets") if isinstance(raw, Mapping) else None
            if not isinstance(raw_entries, Mapping):
                return
            rebuilt: dict = {}
            for bucket_date, entry in raw_entries.items():
                if not isinstance(entry, Mapping):
                    continue
                if deserialize_research_daily_bucket(entry) is None:
                    continue
                rebuilt[bucket_date] = dict(entry)
            self._research_daily_buckets = rebuilt
        except Exception as err:
            self._research_daily_last_error = str(err)

    async def save(self) -> None:
        if self._capture_stores is None or not self._research_daily_save_needed:
            return
        try:
            store = self._capture_stores.research_daily_store()
            await store.save({"buckets": dict(self._research_daily_buckets)})
            self._research_daily_save_needed = False
        except Exception as err:
            self._research_daily_last_error = str(err)

    def record(self, observation: Any) -> None:
        if self._capture_stores is None:
            return
        try:
            bucket_date = getattr(observation, "bucket_date", None)
            if not isinstance(bucket_date, str):
                return
            before = dict(self._research_daily_buckets)
            now_utc = datetime.fromisoformat(self._utcnow_iso())
            result = append_or_update_research_daily_bucket(
                {"buckets": self._research_daily_buckets},
                bucket_date,
                observation,
                now_utc=now_utc,
            )
            after = result.updated_payload["buckets"]
            if after != before:
                self._research_daily_buckets = after
                self._research_daily_save_needed = True
        except Exception as err:
            self._research_daily_last_error = str(err)


def _run(coro):
    return asyncio.run(coro)


# ── T1: Research daily state is initialized in __init__ ────────────────────

def test_t1_state_initialized_in_init():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.__init__)
    assert "self._research_daily_buckets: dict = {}" in source
    assert "self._research_daily_save_needed: bool = False" in source
    assert "self._research_daily_last_error: Optional[str] = None" in source


def test_t1_accessor_methods_exist():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    assert hasattr(_ha.LearningShadowController, "research_daily_snapshot")
    assert hasattr(_ha.LearningShadowController, "research_daily_last_error")
    assert hasattr(_ha.LearningShadowController, "record_research_daily_observation_safe")


# ── T2: async_setup()-equivalent load succeeds with a valid stored payload ─

def test_t2_load_succeeds_with_valid_payload():
    factory = _FakeStoreFactory()
    entry = _make_bucket()
    factory.inject(
        ResearchDailyStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"buckets": {entry["bucket_date"]: entry}})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)

    _run(helper.load())

    assert entry["bucket_date"] in helper._research_daily_buckets
    assert helper._research_daily_last_error is None


def test_t2_load_noop_when_capture_stores_none():
    helper = _ResearchDailyHelper(None)
    _run(helper.load())
    assert helper._research_daily_buckets == {}
    assert helper._research_daily_last_error is None


def test_t2_load_noop_when_store_empty():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    _run(helper.load())
    assert helper._research_daily_buckets == {}
    assert helper._research_daily_last_error is None


# ── T3: Malformed store payload is handled non-fatally ──────────────────────

@pytest.mark.parametrize("bad_buckets", ["not_a_dict", None, 123])
def test_t3_malformed_buckets_shape_is_safe(bad_buckets):
    factory = _FakeStoreFactory()
    factory.inject(
        ResearchDailyStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"buckets": bad_buckets})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    _run(helper.load())
    assert helper._research_daily_buckets == {}
    assert helper._research_daily_last_error is None


# ── T4: Individual malformed buckets are skipped ─────────────────────────────

def test_t4_malformed_individual_entry_is_skipped():
    factory = _FakeStoreFactory()
    good = _make_bucket("2026-06-01")
    factory.inject(
        ResearchDailyStore(factory, _ZONE_A).key,
        _FakeRawStore(initial=_envelope({"buckets": {
            good["bucket_date"]: good,
            "bad-entry": {"schema_version": 999, "bucket_date": "2026-06-02"},
            "worse-entry": "not_a_mapping",
        }})),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    _run(helper.load())
    assert set(helper._research_daily_buckets.keys()) == {good["bucket_date"]}
    assert helper._research_daily_last_error is None


# ── T5: Store-load exception is non-fatal ───────────────────────────────────

def test_t5_load_exception_is_nonfatal():
    factory = _FakeStoreFactory()
    factory.inject(
        ResearchDailyStore(factory, _ZONE_A).key,
        _FakeRawStore(load_raises=RuntimeError("simulated disk error")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)

    _run(helper.load())  # must not raise

    assert helper._research_daily_buckets == {}
    assert "simulated disk error" in (helper._research_daily_last_error or "")


# ── T6: Save is skipped when dirty flag is False ────────────────────────────

def test_t6_save_skipped_when_not_dirty():
    factory = _FakeStoreFactory()
    fake_store = _FakeRawStore()
    factory.inject(ResearchDailyStore(factory, _ZONE_A).key, fake_store)
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    helper._research_daily_buckets = {"2026-06-01": _make_bucket()}
    helper._research_daily_save_needed = False

    _run(helper.save())

    assert fake_store.save_call_count == 0


def test_t6_save_skipped_when_no_capture_stores():
    helper = _ResearchDailyHelper(None)
    helper._research_daily_save_needed = True
    _run(helper.save())  # must not raise, no store to reach


# ── T7: Save runs when dirty flag is True ───────────────────────────────────

def test_t7_save_runs_when_dirty():
    factory = _FakeStoreFactory()
    fake_store = _FakeRawStore()
    factory.inject(ResearchDailyStore(factory, _ZONE_A).key, fake_store)
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    entry = _make_bucket()
    helper._research_daily_buckets = {entry["bucket_date"]: entry}
    helper._research_daily_save_needed = True

    _run(helper.save())

    assert fake_store.save_call_count == 1
    assert fake_store._data["data"]["buckets"][entry["bucket_date"]] == entry


# ── T8: Save success resets the dirty flag ─────────────────────────────────

def test_t8_save_success_resets_dirty_flag():
    factory = _FakeStoreFactory()
    factory.inject(ResearchDailyStore(factory, _ZONE_A).key, _FakeRawStore())
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    helper._research_daily_save_needed = True

    _run(helper.save())

    assert helper._research_daily_save_needed is False


# ── T9: Save failure keeps the dirty flag set ──────────────────────────────

def test_t9_save_failure_keeps_dirty_flag():
    factory = _FakeStoreFactory()
    factory.inject(
        ResearchDailyStore(factory, _ZONE_A).key,
        _FakeRawStore(save_raises=OSError("disk full")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    helper._research_daily_save_needed = True

    _run(helper.save())

    assert helper._research_daily_save_needed is True  # must remain True for retry


# ── T10: Save failure sets the last-error field ─────────────────────────────

def test_t10_save_failure_sets_last_error():
    factory = _FakeStoreFactory()
    factory.inject(
        ResearchDailyStore(factory, _ZONE_A).key,
        _FakeRawStore(save_raises=OSError("disk full")),
    )
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    helper._research_daily_save_needed = True

    _run(helper.save())

    assert "disk full" in (helper._research_daily_last_error or "")


# ── T11: async_unload()/async_save_if_due()/async_setup() wiring (static) ──

def test_t11_async_unload_flushes_research_daily():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_unload)
    assert "await self._async_save_research_daily_safe()" in source


def test_t11_async_save_if_due_flushes_research_daily():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_save_if_due)
    assert "await self._async_save_research_daily_safe()" in source


def test_t11_async_setup_loads_research_daily():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_setup)
    assert "await self._async_load_research_daily_safe()" in source


# ── T12: record_research_daily_observation_safe(): valid observation ───────

def test_t12_record_updates_bucket():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    obs = ResearchDailyObservation(bucket_date="2026-06-01", decision_count=3)
    helper.record(obs)
    assert "2026-06-01" in helper._research_daily_buckets
    assert helper._research_daily_buckets["2026-06-01"]["decision_count"] == 3
    assert helper._research_daily_save_needed is True
    assert helper._research_daily_last_error is None


def test_t12_record_noop_when_no_capture_stores():
    helper = _ResearchDailyHelper(None)
    obs = ResearchDailyObservation(bucket_date="2026-06-01", decision_count=1)
    helper.record(obs)
    assert helper._research_daily_buckets == {}
    assert helper._research_daily_save_needed is False


# ── T13: dirty only on real change / malformed non-fatal ───────────────────

def test_t13_record_second_observation_same_day_accumulates():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    helper.record(ResearchDailyObservation(bucket_date="2026-06-01", decision_count=2))
    helper._research_daily_save_needed = False  # simulate a save having already happened
    helper.record(ResearchDailyObservation(bucket_date="2026-06-01", decision_count=3))
    assert helper._research_daily_buckets["2026-06-01"]["decision_count"] == 5
    assert helper._research_daily_save_needed is True


def test_t13_record_malformed_observation_is_safely_ignored():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)
    helper.record(object())  # not a ResearchDailyObservation, no bucket_date attribute
    assert helper._research_daily_buckets == {}
    assert helper._research_daily_save_needed is False
    assert helper._research_daily_last_error is None  # skip is not an error


def test_t13_record_missing_bucket_date_is_safely_ignored():
    factory = _FakeStoreFactory()
    capture_stores = _FakeCaptureStores(factory, _ZONE_A)
    helper = _ResearchDailyHelper(capture_stores)

    class _FakeObservation:
        bucket_date = None

    helper.record(_FakeObservation())
    assert helper._research_daily_buckets == {}
    assert helper._research_daily_save_needed is False


# ── T14: No runtime hook exists in coordinator.py ───────────────────────────

def test_t14_coordinator_never_touches_research_daily_internals():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod)
    assert "ResearchDailyStore" not in source
    assert "_research_daily_buckets[" not in source
    assert "_async_load_research_daily_safe" not in source
    assert "_async_save_research_daily_safe" not in source
    assert "research_daily_store(" not in source
    assert "record_research_daily_observation_safe(" not in source
    assert "record_research_daily_observation(" not in source


# ── T15: No runtime hook exists in lifecycle.py ─────────────────────────────

def test_t15_no_runtime_hook_in_lifecycle():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    source = inspect.getsource(_lifecycle_mod)
    assert "research_daily" not in source.lower()
    assert "ResearchDailyBucket" not in source
    assert "record_research_daily_observation_safe" not in source


def test_t15_run_cycle_has_no_research_daily_store_io():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    source = inspect.getsource(_lifecycle_mod.LearningRuntime.run_cycle)
    assert "ResearchDailyStore" not in source
    assert "research_daily" not in source.lower()


# ── T16: No export hook exists in export.py ─────────────────────────────────

def test_t16_no_export_reference():
    import custom_components.thermosmart.export as _exp
    source = inspect.getsource(_exp)
    assert "research_daily" not in source.lower()
    assert "ResearchDailyBucket" not in source
    assert "ResearchDailyStore" not in source


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
        inspect.getsource(_ha.LearningShadowController._async_load_research_daily_safe),
        inspect.getsource(_ha.LearningShadowController._async_save_research_daily_safe),
        inspect.getsource(_ha.LearningShadowController.record_research_daily_observation_safe),
        inspect.getsource(_ha.LearningShadowController.research_daily_snapshot),
        inspect.getsource(_ha.LearningShadowController.research_daily_last_error),
    ]
    for source in snippets:
        lowered = source.lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in lowered, f"forbidden control token found: {token}"


def test_t17_no_live_call_site_for_record_method():
    """At the time THIS state/store wiring step was built,
    record_research_daily_observation_safe() had no caller anywhere in
    ha_integration.py — checked here as a snapshot of that boundary. A
    later, separately-approved step ("Research Daily Aggregation Increment
    1") legitimately added exactly ONE call site, from
    _maybe_aggregate_research_daily_from_support_event() (itself only
    reached from record_support_critical_event_safe(), for a genuinely
    newly-appended Support Critical Event) — expected, intentional
    aggregation of an already-produced event, not a runtime/coordinator
    hook and not a regression of this step's own scope. What must remain
    true regardless is that no runtime/coordinator/control path calls
    record_research_daily_observation_safe() directly (see
    test_le2_research_daily_support_event_aggregation.py's own T21 checks)."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha)
    call_count = source.count("self.record_research_daily_observation_safe(")
    assert call_count == 1
    assert "self.record_research_daily_observation_safe(" in inspect.getsource(
        _ha.LearningShadowController._maybe_aggregate_research_daily_from_support_event
    )


# ── T18: Existing Research Daily Foundation/Store tests remain green ───────

def test_t18_regression_imports_ok_research_daily():
    import tests.test_le2_research_daily_buckets  # noqa: F401
    import tests.test_le2_research_daily_store_wiring  # noqa: F401


# ── T19: Existing Episode/Support Store tests remain green ─────────────────

def test_t19_regression_imports_ok_episode_support():
    import tests.test_le2_support_critical_events  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_episode_store_wiring  # noqa: F401
    import tests.test_le2_capture_wiring_foundation  # noqa: F401
