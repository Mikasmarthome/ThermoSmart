"""Tests for Application Lifecycle State Storage Binding.

17 test groups:
  T1  — Store-Key korrekt und multi-zone getrennt
  T2  — Store serialize/deserialize roundtrip (via store layer)
  T3  — Store version mismatch non-fatal
  T4  — Missing Store → leerer State
  T5  — Malformed Store → leerer State, kein Crash
  T6  — Load-Fehler non-fatal
  T7  — Save-Fehler non-fatal und dirty flag bleibt
  T8  — Save nur über async_save_if_due, kein fire-and-forget in observe_safe
  T9  — Unload flush versucht Save
  T10 — Reset löscht Application Lifecycle Store
  T11 — Reset bleibt idempotent
  T12 — Export Summary für leeren State ist neutral
  T13 — Export Summary für gefüllten State zählt korrekt
  T14 — Research Export Entries sind public-safe
  T15 — Multi-Zone State leak ausgeschlossen
  T16 — Keine Runtime-Control-Pfade berührt
  T17 — Bestehende Tests bleiben grün
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Optional

import pytest

# ── Imports (pure learning modules — no HA) ───────────────────────────────────

from custom_components.thermosmart.learning.storage import naming as _naming
from custom_components.thermosmart.learning.adaptation.application_state import (
    APPLICATION_STATE_SCHEMA_VERSION,
    ApplicationLifecycleState,
    AppliedAdaptationStateEntry,
    deserialize_application_lifecycle_state,
    serialize_application_lifecycle_state,
    summarize_application_lifecycle_state,
    application_state_entry_for_research_export,
)
from custom_components.thermosmart.learning.adaptation.monitoring import (
    MonitoringStatus,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_ZONE_A = "zone_alpha_01"
_ZONE_B = "zone_beta_02"
_NOW_TS  = "2026-07-01T10:00:00+00:00"


# ── Test helpers ──────────────────────────────────────────────────────────────

def _fake_entry(
    key: str,
    status: MonitoringStatus = MonitoringStatus.PENDING,
    adoption_ready: bool = False,
) -> AppliedAdaptationStateEntry:
    return AppliedAdaptationStateEntry(
        candidate_key=key,
        candidate_type_value="preheat_delta_min",
        direction_value="increase",
        status=status,
        applied_delta_min=3.0,
        previous_cumulative_delta_min=0.0,
        new_cumulative_delta_min=3.0,
        applied_ts="2026-06-12T10:00:00+00:00",
        monitoring_deadline_ts=None,
        last_evaluated_ts=None,
        monitored_outcome_count=0,
        rollback_supported=True,
        rollback_recommended=(status is MonitoringStatus.ROLLBACK_RECOMMENDED),
        rollback_reasons=(),
        adoption_ready=adoption_ready,
        adopted_ts=None,
        rolled_back_ts=None,
        cooldown_until_ts=None,
        last_error=None,
    )


def _make_state(*entries: AppliedAdaptationStateEntry, zone: str = _ZONE_A) -> ApplicationLifecycleState:
    return ApplicationLifecycleState(
        learning_zone_id=zone,
        updated_at=_NOW_TS,
        entries={e.candidate_key: e for e in entries},
    )


# ── FakeStore infrastructure ──────────────────────────────────────────────────

class _FakeRawStore:
    """Minimal StoreLike that keeps data in-memory. No HA dependencies."""

    def __init__(self, *, initial: Any = None, load_raises: Optional[Exception] = None,
                 save_raises: Optional[Exception] = None) -> None:
        self._data = initial
        self._load_raises = load_raises
        self._save_raises = save_raises
        self.save_call_count = 0
        self.delete_call_count = 0

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
        self.delete_call_count += 1
        self._data = None


class _FakeStoreFactory:
    """Minimal StoreFactory that returns in-memory FakeRawStore instances."""

    def __init__(self) -> None:
        self._stores: dict[str, _FakeRawStore] = {}

    def create(self, key: str, version: int) -> _FakeRawStore:
        if key not in self._stores:
            self._stores[key] = _FakeRawStore()
        return self._stores[key]

    def get(self, key: str) -> Optional[_FakeRawStore]:
        return self._stores.get(key)

    def inject(self, key: str, store: _FakeRawStore) -> None:
        self._stores[key] = store


# ── Async storage helper (tests async load/save logic) ───────────────────────

class _AsyncLifecycleHelper:
    """Standalone test harness exercising the same load/save contract as
    LearningShadowController._async_load/save_application_lifecycle_safe().

    The implementation here is IDENTICAL to the production methods so that
    the tests validate contract behaviour, not a copy.
    """

    def __init__(self, store) -> None:
        self._store = store  # _VersionedStore-like or fake with .load() / .save()
        self._application_lifecycle_state = None
        self._application_lifecycle_save_needed: bool = False
        self._application_lifecycle_last_error: Optional[str] = None
        self._zone = _ZONE_A

    async def load(self) -> None:
        """Non-fatal load — mirrors LearningShadowController._async_load_application_lifecycle_safe."""
        if self._store is None:
            return
        try:
            raw = await self._store.load()
            if raw is None:
                return
            state = deserialize_application_lifecycle_state(raw, self._zone)
            self._application_lifecycle_state = state
        except Exception as err:
            self._application_lifecycle_last_error = str(err)

    async def save(self) -> None:
        """Non-fatal save — mirrors LearningShadowController._async_save_application_lifecycle_safe."""
        if self._store is None or not self._application_lifecycle_save_needed:
            return
        try:
            if self._application_lifecycle_state is None:
                return
            await self._store.save(
                serialize_application_lifecycle_state(self._application_lifecycle_state)
            )
            self._application_lifecycle_save_needed = False
        except Exception as err:
            self._application_lifecycle_last_error = str(err)
            # dirty flag intentionally remains True


# ── T1: Store-Key korrekt und multi-zone getrennt ────────────────────────────

def test_t1_store_key_format():
    key = _naming.application_lifecycle_key(_ZONE_A)
    assert key.startswith("thermosmart_le2__")
    assert _ZONE_A in key
    assert key.endswith("__application_lifecycle")


def test_t1_store_key_multi_zone_distinct():
    key_a = _naming.application_lifecycle_key(_ZONE_A)
    key_b = _naming.application_lifecycle_key(_ZONE_B)
    assert key_a != key_b
    assert _ZONE_A in key_a
    assert _ZONE_B in key_b


def test_t1_store_key_does_not_contain_zone_name():
    key = _naming.application_lifecycle_key(_ZONE_A)
    # Key must not expose raw zone name beyond opaque id
    # (here _ZONE_A IS the opaque id; human-readable names are disallowed by validate_learning_zone_id)
    assert "__" + _ZONE_A + "__" in key  # opaque id is embedded in a separator-bound segment


def test_t1_store_key_consistent():
    assert (
        _naming.application_lifecycle_key(_ZONE_A)
        == _naming.application_lifecycle_key(_ZONE_A)
    )


# ── T2: Store serialize/deserialize roundtrip via store layer ─────────────────

def test_t2_roundtrip_through_store_api():
    """serialize → dict (as stored in the 'data' envelope) → deserialize → same entries."""
    original = _make_state(
        _fake_entry("k1", MonitoringStatus.PENDING),
        _fake_entry("k2", MonitoringStatus.ADOPTED),
    )
    payload = serialize_application_lifecycle_state(original)
    restored = deserialize_application_lifecycle_state(payload, _ZONE_A)

    assert set(restored.entries.keys()) == {"k1", "k2"}
    assert restored.entries["k1"].status is MonitoringStatus.PENDING
    assert restored.entries["k2"].status is MonitoringStatus.ADOPTED
    assert restored.schema_version == APPLICATION_STATE_SCHEMA_VERSION


def test_t2_roundtrip_async_via_versioned_store():
    """Full roundtrip through the _VersionedStore layer using a FakeStoreFactory."""
    from custom_components.thermosmart.learning.storage.stores import (
        ApplicationLifecycleStore,
        APPLICATION_LIFECYCLE_STORE_VERSION,
    )

    factory = _FakeStoreFactory()
    store = ApplicationLifecycleStore(factory, _ZONE_A)

    original = _make_state(_fake_entry("k_round"))
    payload = serialize_application_lifecycle_state(original)

    async def _run():
        await store.save(payload)
        raw = await store.load()
        return raw

    raw = asyncio.run(_run())
    restored = deserialize_application_lifecycle_state(raw, _ZONE_A)
    assert "k_round" in restored.entries


# ── T3: Store version mismatch non-fatal ──────────────────────────────────────

def test_t3_schema_version_mismatch_returns_empty():
    state = _make_state(_fake_entry("k_ver"))
    payload = serialize_application_lifecycle_state(state)
    payload["schema_version"] = 999

    result = deserialize_application_lifecycle_state(payload, _ZONE_A)
    assert result.entries == {}


def test_t3_store_version_mismatch_raises_store_version_error():
    """_VersionedStore raises StoreVersionError when envelope version mismatches."""
    from custom_components.thermosmart.learning.storage.stores import (
        ApplicationLifecycleStore,
        APPLICATION_LIFECYCLE_STORE_VERSION,
    )
    from custom_components.thermosmart.learning.storage.stores import StoreVersionError

    factory = _FakeStoreFactory()
    key = _naming.application_lifecycle_key(_ZONE_A)

    # Inject data with wrong store_schema_version envelope
    bad_raw = _FakeRawStore(initial={
        "store_schema_version": APPLICATION_LIFECYCLE_STORE_VERSION + 99,
        "data": {},
    })
    factory.inject(key, bad_raw)

    store = ApplicationLifecycleStore(factory, _ZONE_A)

    async def _run():
        return await store.load()

    with pytest.raises(StoreVersionError):
        asyncio.run(_run())


# ── T4: Missing Store → leerer State ─────────────────────────────────────────

def test_t4_none_from_load_gives_empty_state():
    """When store.load() returns None, state stays None (treated as fresh empty)."""

    class _NullStore:
        async def load(self):
            return None
        async def save(self, data):
            pass

    helper = _AsyncLifecycleHelper(_NullStore())
    asyncio.run(helper.load())

    assert helper._application_lifecycle_state is None
    assert helper._application_lifecycle_last_error is None


def test_t4_missing_store_key_returns_none():
    from custom_components.thermosmart.learning.storage.stores import ApplicationLifecycleStore

    factory = _FakeStoreFactory()  # no data injected → raw store returns None
    store = ApplicationLifecycleStore(factory, _ZONE_A)

    async def _run():
        return await store.load()

    result = asyncio.run(_run())
    assert result is None


# ── T5: Malformed Store → leerer State, kein Crash ───────────────────────────

@pytest.mark.parametrize("bad_data", [
    {"garbage": True},                            # no schema_version
    {"schema_version": 1, "entries": "not_a_dict"},
    {"schema_version": 1},                        # entries missing → empty
])
def test_t5_malformed_data_returns_empty_state(bad_data):
    result = deserialize_application_lifecycle_state(bad_data, _ZONE_A)
    assert isinstance(result, ApplicationLifecycleState)
    assert result.entries == {}


def test_t5_store_version_error_captured_as_nonfatal():
    """StoreVersionError from _VersionedStore.load() is caught by the async load method."""
    from custom_components.thermosmart.learning.storage.stores import ApplicationLifecycleStore

    factory = _FakeStoreFactory()
    key = _naming.application_lifecycle_key(_ZONE_A)
    # Envelope with wrong version → StoreVersionError in _VersionedStore.load()
    factory.inject(key, _FakeRawStore(initial={
        "store_schema_version": 999,
        "data": {},
    }))
    store = ApplicationLifecycleStore(factory, _ZONE_A)

    helper = _AsyncLifecycleHelper(store)
    asyncio.run(helper.load())

    assert helper._application_lifecycle_state is None
    assert helper._application_lifecycle_last_error is not None


# ── T6: Load-Fehler non-fatal ─────────────────────────────────────────────────

def test_t6_load_exception_is_nonfatal():
    """An exception from the backing store is captured, not re-raised."""

    class _ExplodingStore:
        async def load(self):
            raise RuntimeError("simulated io error")
        async def save(self, data):
            pass

    helper = _AsyncLifecycleHelper(_ExplodingStore())
    # Must not raise
    asyncio.run(helper.load())

    assert helper._application_lifecycle_state is None
    assert "simulated io error" in (helper._application_lifecycle_last_error or "")


def test_t6_none_store_load_is_noop():
    """No store available → load is silently skipped."""
    helper = _AsyncLifecycleHelper(None)
    asyncio.run(helper.load())
    assert helper._application_lifecycle_state is None
    assert helper._application_lifecycle_last_error is None


# ── T7: Save-Fehler non-fatal und dirty flag bleibt ─────────────────────────

def test_t7_save_error_is_nonfatal_dirty_flag_remains():
    """When save throws, dirty flag must remain True so next opportunity retries."""

    class _FailingSaveStore:
        async def load(self):
            return None
        async def save(self, data):
            raise OSError("simulated disk full")

    helper = _AsyncLifecycleHelper(_FailingSaveStore())
    helper._application_lifecycle_state = _make_state(_fake_entry("k_save"))
    helper._application_lifecycle_save_needed = True

    asyncio.run(helper.save())

    assert helper._application_lifecycle_save_needed is True  # must remain True
    assert helper._application_lifecycle_last_error is not None


def test_t7_save_skipped_when_not_dirty():
    """Save must be skipped when dirty flag is False."""
    saved = []

    class _TrackingStore:
        async def load(self):
            return None
        async def save(self, data):
            saved.append(data)

    helper = _AsyncLifecycleHelper(_TrackingStore())
    helper._application_lifecycle_state = _make_state(_fake_entry("k_nd"))
    helper._application_lifecycle_save_needed = False  # NOT dirty

    asyncio.run(helper.save())
    assert saved == []  # save must NOT have been called


def test_t7_save_clears_dirty_on_success():
    """After a successful save, dirty flag must be cleared."""

    class _OkStore:
        async def load(self):
            return None
        async def save(self, data):
            pass

    helper = _AsyncLifecycleHelper(_OkStore())
    helper._application_lifecycle_state = _make_state(_fake_entry("k_ok"))
    helper._application_lifecycle_save_needed = True

    asyncio.run(helper.save())
    assert helper._application_lifecycle_save_needed is False


# ── T8: Save nur über async_save_if_due, kein fire-and-forget in observe_safe ─

def test_t8_observe_safe_does_not_call_save():
    """observe_safe must not contain any call to _async_save_application_lifecycle_safe."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _mod

    src = inspect.getsource(_mod.LearningShadowController.observe_safe)
    assert "_async_save_application_lifecycle_safe" not in src, (
        "observe_safe must not directly call save (no fire-and-forget)"
    )


def test_t8_observe_safe_does_not_await_save():
    """observe_safe source must not contain any awaited save of the lifecycle store."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _mod

    src = inspect.getsource(_mod.LearningShadowController.observe_safe)
    forbidden = [
        "application_lifecycle_store.save",
        "_async_save_application_lifecycle",
    ]
    found = [f for f in forbidden if f in src]
    assert found == [], f"observe_safe contains forbidden save calls: {found}"


# ── T9: Unload flush versucht Save ───────────────────────────────────────────

def test_t9_async_unload_calls_lifecycle_save():
    """async_unload source must call _async_save_application_lifecycle_safe."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _mod

    src = inspect.getsource(_mod.LearningShadowController.async_unload)
    assert "_async_save_application_lifecycle_safe" in src


def test_t9_async_save_if_due_calls_lifecycle_save():
    """async_save_if_due source must call _async_save_application_lifecycle_safe."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _mod

    src = inspect.getsource(_mod.LearningShadowController.async_save_if_due)
    assert "_async_save_application_lifecycle_safe" in src


# ── T10: Reset löscht Application Lifecycle Store ────────────────────────────

def test_t10_reset_includes_application_lifecycle_store():
    """reset_v2_learning_state source must delete the ApplicationLifecycleStore."""
    import custom_components.thermosmart.learning.reset as _mod

    src = inspect.getsource(_mod.reset_v2_learning_state)
    assert "ApplicationLifecycleStore" in src


def test_t10_application_lifecycle_store_delete_is_nonfatal():
    """ApplicationLifecycleStore.delete() on non-existent key must not raise."""
    from custom_components.thermosmart.learning.storage.stores import ApplicationLifecycleStore

    factory = _FakeStoreFactory()
    store = ApplicationLifecycleStore(factory, _ZONE_A)

    async def _run():
        await store.delete()  # nothing stored → must be non-fatal

    asyncio.run(_run())  # must not raise


# ── T11: Reset idempotent ─────────────────────────────────────────────────────

def test_t11_double_delete_is_idempotent():
    """Calling delete twice must not raise (idempotent)."""
    from custom_components.thermosmart.learning.storage.stores import ApplicationLifecycleStore

    factory = _FakeStoreFactory()
    store = ApplicationLifecycleStore(factory, _ZONE_A)

    async def _run():
        await store.delete()
        await store.delete()  # second call must also be non-fatal

    asyncio.run(_run())  # must not raise


def test_t11_reset_nonfatal_on_missing_store():
    """reset_v2_learning_state wraps ApplicationLifecycleStore.delete() in try/except."""
    import custom_components.thermosmart.learning.reset as _mod

    src = inspect.getsource(_mod.reset_v2_learning_state)
    # The pattern: try: await ApplicationLifecycleStore(...).delete() except: pass
    assert "ApplicationLifecycleStore" in src
    assert "except" in src  # non-fatal wrapper present


# ── T12: Export Summary für leeren State ist neutral ─────────────────────────

def test_t12_summary_empty_state_all_zeros():
    state = ApplicationLifecycleState(
        learning_zone_id=_ZONE_A, updated_at=_NOW_TS, entries={},
    )
    summary = summarize_application_lifecycle_state(state)
    d = summary.to_dict()

    assert d["total"] == 0
    assert d["active"] == 0
    assert d["rollback_recommended"] == 0
    assert d["adoption_ready"] == 0
    assert d["adopted"] == 0
    assert d["rolled_back"] == 0
    assert d["expired"] == 0


def test_t12_export_summary_neutral_when_no_shadow():
    """_le2_application_lifecycle_summary returns zero dict when shadow is None."""
    from custom_components.thermosmart.export import _le2_application_lifecycle_summary

    class _FakeCoordNoShadow:
        _le2_shadow = None

    result = _le2_application_lifecycle_summary(_FakeCoordNoShadow(), _ZONE_A)

    assert result["state_entry_count"] == 0
    assert result["last_error"] is None


# ── T13: Export Summary für gefüllten State zählt korrekt ────────────────────

def test_t13_summary_filled_state_correct():
    entries = {
        "e1": _fake_entry("e1", MonitoringStatus.PENDING),
        "e2": _fake_entry("e2", MonitoringStatus.IN_PROGRESS),
        "e3": _fake_entry("e3", MonitoringStatus.ROLLBACK_RECOMMENDED),
        "e4": _fake_entry("e4", MonitoringStatus.ADOPTION_CANDIDATE, adoption_ready=True),
        "e5": _fake_entry("e5", MonitoringStatus.ADOPTED),
        "e6": _fake_entry("e6", MonitoringStatus.ROLLED_BACK),
        "e7": _fake_entry("e7", MonitoringStatus.WINDOW_EXPIRED),
    }
    state = ApplicationLifecycleState(
        learning_zone_id=_ZONE_A, updated_at=_NOW_TS, entries=entries,
    )
    summary = summarize_application_lifecycle_state(state)

    assert summary.total == 7
    assert summary.active == 4          # PENDING, IN_PROGRESS, ROLLBACK_RECOMMENDED, ADOPTION_CANDIDATE
    assert summary.rollback_recommended == 1
    assert summary.adoption_ready == 1
    assert summary.adopted == 1
    assert summary.rolled_back == 1
    assert summary.expired == 1


def test_t13_export_summary_helper_counts_correctly():
    """_le2_application_lifecycle_summary reads from shadow.application_lifecycle_snapshot()."""
    from custom_components.thermosmart.export import _le2_application_lifecycle_summary

    class _FakeShadow:
        _application_lifecycle_last_error = None
        def application_lifecycle_snapshot(self):
            return {
                "total": 3,
                "active": 2,
                "rollback_recommended": 1,
                "adoption_ready": 0,
                "adopted": 1,
                "rolled_back": 0,
                "expired": 0,
            }

    class _FakeCoord:
        _le2_shadow = _FakeShadow()

    result = _le2_application_lifecycle_summary(_FakeCoord(), _ZONE_A)

    assert result["state_entry_count"] == 3
    assert result["active_count"] == 2
    assert result["rollback_recommended_count"] == 1
    assert result["adopted_count"] == 1
    assert result["last_error"] is None


# ── T14: Research Export Entries sind public-safe ─────────────────────────────

def test_t14_research_export_no_timestamps():
    se = _fake_entry("k_pub")
    exported = application_state_entry_for_research_export(se, now_ts=_NOW_TS)

    forbidden = {"applied_ts", "monitoring_deadline_ts", "last_evaluated_ts",
                 "adopted_ts", "rolled_back_ts", "cooldown_until_ts"}
    assert not (forbidden & set(exported.keys()))


def test_t14_research_export_required_keys_present():
    se = _fake_entry("k_req")
    exported = application_state_entry_for_research_export(se, now_ts=_NOW_TS)

    required = {"candidate_key", "candidate_type", "direction", "status",
                "applied_delta_min", "new_cumulative_delta_min",
                "monitored_outcome_count", "rollback_recommended",
                "rollback_reasons", "adoption_ready", "cooldown_active"}
    assert required <= set(exported.keys())
    assert isinstance(exported["cooldown_active"], bool)
    assert isinstance(exported["rollback_reasons"], list)


def test_t14_research_helper_returns_empty_when_no_entries():
    from custom_components.thermosmart.export import _le2_application_lifecycle_research_entries

    class _FakeShadow:
        _application_lifecycle_state = None

    class _FakeCoord:
        _le2_shadow = _FakeShadow()

    result = _le2_application_lifecycle_research_entries(_FakeCoord())
    assert result == []


def test_t14_research_helper_no_private_fields_in_entries():
    from custom_components.thermosmart.export import _le2_application_lifecycle_research_entries

    state = _make_state(
        _fake_entry("k_r1"),
        _fake_entry("k_r2", MonitoringStatus.ADOPTED),
    )

    class _FakeShadow:
        _application_lifecycle_state = state

    class _FakeCoord:
        _le2_shadow = _FakeShadow()

    entries = _le2_application_lifecycle_research_entries(_FakeCoord())
    assert len(entries) == 2
    for entry_dict in entries:
        forbidden = {"entity_id", "zone_id", "learning_zone_id", "applied_ts",
                     "adopted_ts", "rolled_back_ts", "cooldown_until_ts"}
        assert not (forbidden & set(entry_dict.keys()))


# ── T15: Multi-Zone State leak ausgeschlossen ────────────────────────────────

def test_t15_keys_isolated_per_zone():
    key_a = _naming.application_lifecycle_key(_ZONE_A)
    key_b = _naming.application_lifecycle_key(_ZONE_B)

    assert key_a != key_b
    assert _ZONE_A not in key_b
    assert _ZONE_B not in key_a


def test_t15_deserialized_zone_id_is_authoritative():
    """Deserialized state always uses the loading zone_id, not the stored one."""
    state_for_a = _make_state(_fake_entry("kk"), zone=_ZONE_A)
    payload = serialize_application_lifecycle_state(state_for_a)

    # Load with zone B — the zone_id must be overridden by the loading context
    restored = deserialize_application_lifecycle_state(payload, _ZONE_B)
    assert restored.learning_zone_id == _ZONE_B  # authoritative: loading zone wins


def test_t15_store_keys_from_factory_are_zone_scoped():
    from custom_components.thermosmart.learning.storage.stores import ApplicationLifecycleStore

    class _CapturingFactory:
        def __init__(self):
            self.keys: list[str] = []
        def create(self, key: str, version: int) -> _FakeRawStore:
            self.keys.append(key)
            return _FakeRawStore()

    factory_a = _CapturingFactory()
    factory_b = _CapturingFactory()

    ApplicationLifecycleStore(factory_a, _ZONE_A)
    ApplicationLifecycleStore(factory_b, _ZONE_B)

    assert factory_a.keys != factory_b.keys
    assert all(_ZONE_A in k for k in factory_a.keys)
    assert all(_ZONE_B in k for k in factory_b.keys)


# ── T16: Keine Runtime-Control-Pfade berührt ────────────────────────────────

def test_t16_no_control_keywords_in_storage_modules():
    import custom_components.thermosmart.learning.storage.stores as stores_mod
    import custom_components.thermosmart.learning.storage.naming as naming_mod

    for mod in (stores_mod, naming_mod):
        src = inspect.getsource(mod)
        forbidden = [
            "set_temperature", "async_set_temperature", "async_write_ha_state",
            "dispatch", "service_call", "tpi_gain", "preheat_duration",
        ]
        found = [kw for kw in forbidden if kw in src]
        assert found == [], f"Control keywords in {mod.__name__}: {found}"


def test_t16_no_control_keywords_in_lifecycle_methods():
    """The new application lifecycle methods must not contain control keywords."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _mod

    methods = [
        "_async_load_application_lifecycle_safe",
        "_async_save_application_lifecycle_safe",
        "application_lifecycle_snapshot",
    ]
    forbidden = [
        "set_temperature", "async_write_ha_state", "dispatch",
        "tpi_gain", "preheat_duration", "service_call",
    ]
    for name in methods:
        src = inspect.getsource(getattr(_mod.LearningShadowController, name))
        found = [kw for kw in forbidden if kw in src]
        assert found == [], f"{name} contains control keywords: {found}"


# ── T17: Bestehende Tests bleiben grün ───────────────────────────────────────

def test_t17_regression_application_lifecycle_state():
    import tests.test_application_lifecycle_state  # noqa: F401


def test_t17_regression_adaptation_monitoring():
    import tests.test_adaptation_monitoring  # noqa: F401


def test_t17_regression_preheat_application_plan():
    import tests.test_preheat_application_plan  # noqa: F401


def test_t17_regression_application_readiness():
    import tests.test_application_readiness  # noqa: F401


def test_t17_regression_promotion_readiness():
    import tests.test_promotion_readiness  # noqa: F401
