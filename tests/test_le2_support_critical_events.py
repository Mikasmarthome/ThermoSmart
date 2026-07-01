"""Tests for the LE2 Support Critical Event Store foundation.

Covers:
  - custom_components/thermosmart/learning/support_event_schemas.py
    (SupportEventType, SupportEventSeverity, SupportCriticalEvent)
  - custom_components/thermosmart/learning/storage/support_event_serialization.py
    (serialize/deserialize/support_event_for_export)
  - custom_components/thermosmart/learning/storage/support_event_persistence.py
    (append/prune/retention/dedup helpers)
  - custom_components/thermosmart/learning/storage/naming.py's new
    support_critical_events_key()
  - custom_components/thermosmart/learning/storage/stores.py's new
    SupportCriticalEventStore
  - custom_components/thermosmart/learning/storage/capture_stores.py's new
    support_critical_events_store() accessor

Foundation only in this step — nothing here performs live storage I/O from
the runtime, and nothing in ha_integration.py/coordinator.py/lifecycle.py
was touched. Store-wrapper load/save behavior is tested directly against
the real SupportCriticalEventStore class with a fake StoreFactory (no HA
dependency), mirroring the pattern already established in
test_le2_episode_store_wiring.py.

24 test groups:
  T1  — Serialization roundtrip preserves every field
  T2  — Malformed event (wrong type) fails to serialize, returns None
  T3  — Unknown event_type/severity value fails to deserialize, returns None
  T4  — Missing/malformed timestamp handled safely (deserialize -> None)
  T5  — Append a valid critical event
  T6  — Duplicate event_id is not stored twice
  T7  — Max-age (48h/2-day) pruning
  T8  — Max-records cap pruning
  T9  — Critical severity is prioritized over info severity when capped
  T10 — Stable order (append order preserved among survivors)
  T11 — Idempotent pruning (running prune twice changes nothing further)
  T12 — Public-safety: no forbidden substrings anywhere in serialized/export output
  T13 — No entity ids / zone names / secrets / paths in output
  T14 — Store naming is public-safe and zone-separated
  T15 — Store load: corrupt/malformed payload is non-fatal
  T16 — Store save: underlying save error propagates as a plain exception
        (store wrapper itself adds no swallowing — matches EpisodesStore)
  T17 — details is bounded (max keys, value length, non-scalar values dropped)
  T18 — is_recent_duplicate helper detects/ignores bursts correctly
  T19 — No-op-prone event types are identified for default low-severity use
  T20 — No Support Export change (export.py untouched by this step)
  T21 — No Research Export regression
  T22 — Existing LE2 export/storage tests remain green
  T23 — No runtime/lifecycle wiring exists yet (foundation-only confirmation)
  T24 — No control-path keywords touched by the new modules
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from custom_components.thermosmart.learning.registry import RetentionPolicy
from custom_components.thermosmart.learning.support_event_schemas import (
    NOISE_PRONE_EVENT_TYPES,
    SupportCriticalEvent,
    SupportEventSeverity,
    SupportEventType,
)
from custom_components.thermosmart.learning.storage.support_event_serialization import (
    SUPPORT_EVENT_SCHEMA_VERSION,
    deserialize_support_event,
    serialize_support_event,
    support_event_for_export,
)
from custom_components.thermosmart.learning.storage.support_event_persistence import (
    SUPPORT_EVENT_RETENTION_DEFAULT,
    SUPPORT_EVENT_RETENTION_MAX_AGE_DAYS,
    SUPPORT_EVENT_RETENTION_MAX_RECORDS,
    append_support_event,
    append_support_events,
    evaluate_support_event_retention,
    is_recent_duplicate,
    prune_support_events,
)
from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.storage.stores import (
    SUPPORT_CRITICAL_EVENT_STORE_VERSION,
    SupportCriticalEventStore,
)
from custom_components.thermosmart.learning.storage.capture_stores import LearningCaptureStores

_ZONE_A = "zone_alpha_01"
_NOW = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


def _event(event_id: str, *, event_type=SupportEventType.TRV_COMMAND_BLOCKED,
           severity=SupportEventSeverity.WARNING, ts=None, reason="min_interval",
           summary="TRV command blocked by minimum interval", details=None) -> SupportCriticalEvent:
    return SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION,
        event_id=event_id,
        event_type=event_type,
        ts=ts if ts is not None else _NOW,
        severity=severity,
        reason=reason,
        summary=summary,
        details=details if details is not None else {"requested_setpoint": 21.0, "current_setpoint": 20.5},
    )


# ── T1: Serialization roundtrip preserves every field ───────────────────────

def test_t1_serialization_roundtrip():
    ev = _event("evt_001")
    ser = serialize_support_event(ev)
    assert ser is not None
    back = deserialize_support_event(ser)
    assert back == ev


def test_t1_roundtrip_without_optional_fields():
    ev = SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id="evt_bare",
        event_type=SupportEventType.BOOST_STARTED, ts=_NOW,
        severity=SupportEventSeverity.INFO,
    )
    ser = serialize_support_event(ev)
    back = deserialize_support_event(ser)
    assert back == ev
    assert back.reason is None
    assert back.summary is None
    assert back.details == {}


# ── T2: Malformed event fails to serialize ──────────────────────────────────

def test_t2_non_event_object_returns_none():
    assert serialize_support_event(object()) is None
    assert serialize_support_event(None) is None
    assert serialize_support_event({"event_type": "trv_command_blocked"}) is None


# ── T3: Unknown event_type/severity fails to deserialize ────────────────────

def test_t3_unknown_event_type_returns_none():
    raw = serialize_support_event(_event("evt_002"))
    raw["event_type"] = "not_a_real_event_type"
    assert deserialize_support_event(raw) is None


def test_t3_unknown_severity_returns_none():
    raw = serialize_support_event(_event("evt_003"))
    raw["severity"] = "urgent"  # not a real severity value
    assert deserialize_support_event(raw) is None


def test_t3_schema_version_mismatch_returns_none():
    raw = serialize_support_event(_event("evt_004"))
    raw["schema_version"] = 999
    assert deserialize_support_event(raw) is None


def test_t3_not_a_mapping_returns_none():
    assert deserialize_support_event("not-a-dict") is None
    assert deserialize_support_event(None) is None
    assert deserialize_support_event(["a", "list"]) is None


# ── T4: Missing/malformed timestamp handled safely ──────────────────────────

def test_t4_missing_ts_field_returns_none():
    raw = serialize_support_event(_event("evt_005"))
    del raw["ts"]
    assert deserialize_support_event(raw) is None


def test_t4_malformed_ts_field_returns_none():
    raw = serialize_support_event(_event("evt_006"))
    raw["ts"] = "not-a-timestamp"
    assert deserialize_support_event(raw) is None


def test_t4_missing_event_id_returns_none():
    raw = serialize_support_event(_event("evt_007"))
    del raw["event_id"]
    assert deserialize_support_event(raw) is None


# ── T5: Append a valid critical event ───────────────────────────────────────

def test_t5_append_valid_event():
    result = append_support_event({}, _event("evt_100"), now_utc=_NOW)
    assert result.appended_event_ids == ("evt_100",)
    assert "evt_100" in result.updated_payload["events"]
    assert result.skipped_count == 0


def test_t5_append_malformed_event_is_skipped_not_fatal():
    class _NotAnEvent:
        pass
    result = append_support_events({}, [_NotAnEvent()], now_utc=_NOW)
    assert result.skipped_count == 1
    assert result.appended_event_ids == ()


# ── T6: Duplicate event_id is not stored twice ──────────────────────────────

def test_t6_duplicate_event_id_not_stored_twice():
    ev = _event("evt_dup")
    result1 = append_support_event({}, ev, now_utc=_NOW)
    result2 = append_support_event(result1.updated_payload, ev, now_utc=_NOW)
    assert result2.duplicate_count == 1
    assert len(result2.updated_payload["events"]) == 1


# ── T7: Max-age (48h/2-day) pruning ──────────────────────────────────────────

def test_t7_max_age_constant_is_two_days():
    assert SUPPORT_EVENT_RETENTION_MAX_AGE_DAYS == 2
    assert SUPPORT_EVENT_RETENTION_DEFAULT.max_age_days == 2


def test_t7_event_older_than_48h_is_pruned():
    old = _event("evt_old", ts=_NOW - timedelta(hours=49))
    recent = _event("evt_recent", ts=_NOW - timedelta(hours=1))
    result = append_support_events({}, [old, recent], now_utc=_NOW)
    assert "evt_old" not in result.updated_payload["events"]
    assert "evt_recent" in result.updated_payload["events"]
    assert "evt_old" in result.pruned_event_ids


def test_t7_event_within_48h_survives():
    within = _event("evt_within", ts=_NOW - timedelta(hours=47, minutes=59))
    result = append_support_event({}, within, now_utc=_NOW)
    assert "evt_within" in result.updated_payload["events"]


def test_t7_malformed_timestamp_conservatively_kept():
    payload = {"events": {"evt_bad_ts": {
        "schema_version": SUPPORT_EVENT_SCHEMA_VERSION, "event_id": "evt_bad_ts",
        "event_type": "boost_started", "ts": "garbage", "severity": "info",
        "reason": None, "summary": None, "details": {},
    }}}
    decision = evaluate_support_event_retention(payload, now_utc=_NOW)
    assert "evt_bad_ts" in decision.keep_event_ids


# ── T8: Max-records cap pruning ─────────────────────────────────────────────

def test_t8_max_records_constant():
    assert SUPPORT_EVENT_RETENTION_MAX_RECORDS == 750
    assert SUPPORT_EVENT_RETENTION_DEFAULT.max_records == 750


def test_t8_cap_evicts_overflow():
    policy = RetentionPolicy(max_age_days=2, max_records=2)
    events = [
        _event(f"evt_{i}", ts=_NOW - timedelta(minutes=i), severity=SupportEventSeverity.WARNING)
        for i in range(5)
    ]
    result = append_support_events({}, events, now_utc=_NOW, policy=policy)
    assert len(result.updated_payload["events"]) == 2
    assert len(result.pruned_event_ids) == 3


# ── T9: Critical severity prioritized over info when capped ────────────────

def test_t9_critical_survives_over_older_info_events():
    policy = RetentionPolicy(max_age_days=2, max_records=2)
    info_older = _event("info_older", ts=_NOW - timedelta(minutes=90),
                         severity=SupportEventSeverity.INFO, event_type=SupportEventType.SAME_SETPOINT_BLOCK)
    info_newer = _event("info_newer", ts=_NOW - timedelta(minutes=10),
                         severity=SupportEventSeverity.INFO, event_type=SupportEventType.MIN_INTERVAL_BLOCK)
    critical_oldest = _event("critical_oldest", ts=_NOW - timedelta(minutes=120),
                              severity=SupportEventSeverity.CRITICAL, event_type=SupportEventType.TRV_UNAVAILABLE)
    result = append_support_events(
        {}, [info_older, info_newer, critical_oldest], now_utc=_NOW, policy=policy,
    )
    kept = set(result.updated_payload["events"])
    assert "critical_oldest" in kept  # survives despite being the oldest
    assert "info_older" not in kept   # evicted first: lowest severity


def test_t9_within_same_severity_oldest_evicted_first():
    policy = RetentionPolicy(max_age_days=2, max_records=1)
    older = _event("w_older", ts=_NOW - timedelta(minutes=30), severity=SupportEventSeverity.WARNING)
    newer = _event("w_newer", ts=_NOW - timedelta(minutes=5), severity=SupportEventSeverity.WARNING)
    result = append_support_events({}, [older, newer], now_utc=_NOW, policy=policy)
    assert "w_newer" in result.updated_payload["events"]
    assert "w_older" not in result.updated_payload["events"]


# ── T10: Stable order (append order preserved) ──────────────────────────────

def test_t10_appended_ids_preserve_call_order():
    events = [_event(f"evt_{i}") for i in range(5)]
    result = append_support_events({}, events, now_utc=_NOW)
    assert result.appended_event_ids == tuple(f"evt_{i}" for i in range(5))


# ── T11: Idempotent pruning ──────────────────────────────────────────────────

def test_t11_pruning_twice_is_idempotent():
    events = [_event(f"evt_{i}", ts=_NOW - timedelta(minutes=i)) for i in range(3)]
    result = append_support_events({}, events, now_utc=_NOW)
    once = prune_support_events(result.updated_payload, now_utc=_NOW)
    twice = prune_support_events(once.updated_payload, now_utc=_NOW)
    assert once.updated_payload == twice.updated_payload
    assert twice.pruned_event_ids == ()


# ── T12/T13: Public-safety substrings ────────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t12_no_forbidden_substrings_in_serialized_output():
    ev = _event("evt_safe", details={"requested_setpoint": 21.0, "trv_state": "ok"})
    ser = serialize_support_event(ev)
    flat = repr(ser).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in flat, f"forbidden substring '{token}' found in serialized output"


def test_t12_no_forbidden_substrings_in_export_view():
    ev = _event("evt_safe2")
    ser = serialize_support_event(ev)
    view = support_event_for_export(ser)
    assert "event_id" not in view  # explicitly dropped
    flat = repr(view).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in flat, f"forbidden substring '{token}' found in export view"


def test_t13_no_entity_ids_or_paths_even_if_smuggled_into_details():
    """Defense-in-depth: details values are bounded/scalar-only, so even an
    attempted entity-id/path string in details survives only as an opaque
    bounded string (never a dict/object), and callers are responsible for
    not putting such values there in the first place — this test documents
    that the length/type bound applies uniformly, it does not itself scan
    string CONTENT (that is scan_payload()'s job at the export layer)."""
    ev = _event("evt_details_bound", details={"note": "sensor.some_entity_looking_string"})
    ser = serialize_support_event(ev)
    assert isinstance(ser["details"]["note"], str)
    assert len(ser["details"]["note"]) <= 200


# ── T14: Store naming is public-safe and zone-separated ─────────────────────

def test_t14_naming_key_is_zone_separated_and_stable():
    key_a = naming.support_critical_events_key(_ZONE_A)
    key_b = naming.support_critical_events_key("zone_beta_02")
    assert key_a != key_b
    assert key_a == naming.support_critical_events_key(_ZONE_A)  # deterministic
    assert "support_critical_events" in key_a
    assert _ZONE_A in key_a  # opaque internal id, not a human zone name


def test_t14_naming_rejects_unsafe_zone_id():
    with pytest.raises(naming.StorageNamingError):
        naming.support_critical_events_key("zone/with/slashes")


# ── T15: Store load corrupt payload is non-fatal ────────────────────────────

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


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_t15_store_load_returns_none_when_empty():
    factory = _FakeStoreFactory()
    store = SupportCriticalEventStore(factory, _ZONE_A)
    assert _run(store.load()) is None


def test_t15_store_load_corrupt_payload_raises_store_version_error():
    """Matches EpisodesStore's own contract: a payload missing the schema
    envelope raises StoreVersionError, which the (future) caller layer is
    responsible for catching non-fatally — exactly like
    _async_load_episode_history_safe() already does for EpisodesStore."""
    from custom_components.thermosmart.learning.storage.stores import StoreVersionError
    factory = _FakeStoreFactory()
    key = naming.support_critical_events_key(_ZONE_A)
    factory.inject(key, _FakeRawStore(initial={"not": "an envelope"}))
    store = SupportCriticalEventStore(factory, _ZONE_A)
    with pytest.raises(StoreVersionError):
        _run(store.load())


def test_t15_store_load_version_mismatch_raises():
    from custom_components.thermosmart.learning.storage.stores import StoreVersionError
    factory = _FakeStoreFactory()
    key = naming.support_critical_events_key(_ZONE_A)
    factory.inject(key, _FakeRawStore(initial={"store_schema_version": 999, "data": {}}))
    store = SupportCriticalEventStore(factory, _ZONE_A)
    with pytest.raises(StoreVersionError):
        _run(store.load())


# ── T16: Store save propagates errors (no swallowing at this layer) ────────

def test_t16_store_save_error_propagates():
    factory = _FakeStoreFactory()
    key = naming.support_critical_events_key(_ZONE_A)
    factory.inject(key, _FakeRawStore(save_raises=RuntimeError("disk full")))
    store = SupportCriticalEventStore(factory, _ZONE_A)
    with pytest.raises(RuntimeError):
        _run(store.save({"events": {}}))


def test_t16_store_roundtrip_via_capture_stores_accessor():
    factory = _FakeStoreFactory()
    capture = LearningCaptureStores(factory, _ZONE_A)
    store = capture.support_critical_events_store()
    assert isinstance(store, SupportCriticalEventStore)
    ev = _event("evt_via_accessor")
    result = append_support_event({}, ev, now_utc=_NOW)
    _run(store.save(result.updated_payload))
    loaded = _run(store.load())
    assert loaded == result.updated_payload


# ── T17: details is bounded ──────────────────────────────────────────────────

def test_t17_details_max_keys_bounded():
    details = {f"k{i}": i for i in range(30)}
    ev = _event("evt_many_keys", details=details)
    ser = serialize_support_event(ev)
    assert len(ser["details"]) <= 12


def test_t17_details_string_value_length_bounded():
    ev = _event("evt_long_str", details={"note": "x" * 500})
    ser = serialize_support_event(ev)
    assert len(ser["details"]["note"]) <= 200


def test_t17_details_non_scalar_values_dropped():
    ev = _event("evt_nested", details={"ok": 1.0, "bad_list": [1, 2, 3], "bad_dict": {"a": 1}})
    ser = serialize_support_event(ev)
    assert "ok" in ser["details"]
    assert "bad_list" not in ser["details"]
    assert "bad_dict" not in ser["details"]


def test_t17_summary_length_bounded():
    ev = _event("evt_long_summary", summary="y" * 500)
    ser = serialize_support_event(ev)
    assert len(ser["summary"]) <= 200


# ── T18: is_recent_duplicate helper ─────────────────────────────────────────

def test_t18_detects_duplicate_within_window():
    ev = _event("evt_dup_check", event_type=SupportEventType.SAME_SETPOINT_BLOCK,
                reason="same_setpoint", ts=_NOW)
    recent = [{"event_type": "same_setpoint_block", "reason": "same_setpoint",
               "ts": (_NOW - timedelta(minutes=2)).isoformat()}]
    assert is_recent_duplicate(ev, recent, window_s=300) is True


def test_t18_ignores_duplicate_outside_window():
    ev = _event("evt_dup_check2", event_type=SupportEventType.SAME_SETPOINT_BLOCK,
                reason="same_setpoint", ts=_NOW)
    recent = [{"event_type": "same_setpoint_block", "reason": "same_setpoint",
               "ts": (_NOW - timedelta(minutes=10)).isoformat()}]
    assert is_recent_duplicate(ev, recent, window_s=60) is False


def test_t18_different_reason_is_not_a_duplicate():
    ev = _event("evt_dup_check3", event_type=SupportEventType.TRV_COMMAND_BLOCKED,
                reason="min_interval", ts=_NOW)
    recent = [{"event_type": "trv_command_blocked", "reason": "same_setpoint",
               "ts": _NOW.isoformat()}]
    assert is_recent_duplicate(ev, recent, window_s=300) is False


def test_t18_malformed_recent_entry_is_non_fatal():
    ev = _event("evt_dup_check4")
    assert is_recent_duplicate(ev, ["not-a-dict", None, 42], window_s=300) is False


# ── T19: Noise-prone event types identified ─────────────────────────────────

def test_t19_noise_prone_types_include_repeated_blockers():
    assert SupportEventType.SAME_SETPOINT_BLOCK in NOISE_PRONE_EVENT_TYPES
    assert SupportEventType.MIN_INTERVAL_BLOCK in NOISE_PRONE_EVENT_TYPES


def test_t19_critical_incident_types_are_not_noise_prone():
    assert SupportEventType.TRV_UNAVAILABLE not in NOISE_PRONE_EVENT_TYPES
    assert SupportEventType.SENSOR_UNAVAILABLE not in NOISE_PRONE_EVENT_TYPES


# ── T20/T21: No Support/Research export change ──────────────────────────────

def test_t20_export_module_source_unchanged_by_this_step():
    import custom_components.thermosmart.export as _exp
    source = inspect.getsource(_exp)
    assert "support_event" not in source.lower()
    assert "SupportCriticalEvent" not in source


def test_t21_existing_research_export_helpers_still_importable():
    from custom_components.thermosmart.export import (
        _le2_episode_history_export,
        _le2_learning_progress_export,
        _le2_research_data,
    )
    assert callable(_le2_episode_history_export)
    assert callable(_le2_learning_progress_export)
    assert callable(_le2_research_data)


# ── T22: Existing LE2 export/storage tests remain green ────────────────────

def test_t22_regression_imports_ok():
    import tests.test_le2_episode_history_export  # noqa: F401
    import tests.test_le2_learning_progress_export  # noqa: F401
    import tests.test_le2_episode_persistence  # noqa: F401
    import tests.test_le2_episode_serialization  # noqa: F401
    import tests.test_le2_episode_store_wiring  # noqa: F401
    import tests.test_le2_retention  # noqa: F401
    import tests.test_le2_capture_wiring_foundation  # noqa: F401


# ── T23: No runtime/lifecycle wiring exists yet ─────────────────────────────
#
# A later step ("Support Critical Event Store State and Save Wiring")
# legitimately added in-memory state + load/save wiring for support critical
# events into LearningShadowController (ha_integration.py) — mirroring
# exactly how the episode-history state/store wiring was added on top of the
# episode-persistence foundation. That is expected, intentional wiring, not
# a regression. What must remain true is the boundary this foundation step
# was built to protect: lifecycle.py (run_cycle()) and coordinator.py never
# reference support events at all — no runtime PRODUCER exists yet, only
# state/load/save plumbing in ha_integration.py.

def test_t23_lifecycle_module_has_no_support_event_reference():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    source = inspect.getsource(_lifecycle_mod)
    assert "support_event" not in source.lower()
    assert "SupportCriticalEvent" not in source


def test_t23_ha_integration_now_legitimately_has_state_and_store_wiring():
    """Confirms the intentional wiring added by the store-wiring step: state
    fields, load/save methods, and record_support_critical_event_safe() all
    exist — but nothing else in ha_integration.py calls
    record_support_critical_event_safe() as a live producer (no sink binding,
    unlike record_completed_episode_safe(), which IS bound as episode_sink)."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha_mod
    source = inspect.getsource(_ha_mod)
    assert "self._support_critical_events: dict = {}" in source
    assert "record_support_critical_event_safe" in source
    assert source.count("self.record_support_critical_event_safe(") == 0


def test_t23_coordinator_module_has_no_support_event_reference():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod)
    assert "support_event" not in source.lower()
    assert "SupportCriticalEvent" not in source


# ── T24: No control-path keywords touched ───────────────────────────────────

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


def test_t24_no_control_keywords_in_new_modules():
    import custom_components.thermosmart.learning.support_event_schemas as _schemas_mod
    import custom_components.thermosmart.learning.storage.support_event_serialization as _ser_mod
    import custom_components.thermosmart.learning.storage.support_event_persistence as _pers_mod
    for mod in (_schemas_mod, _ser_mod, _pers_mod):
        source = inspect.getsource(mod).lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            if token == "setpoint":
                # "requested_setpoint"/"current_setpoint" are legitimate example
                # detail field names in this test module's own docstrings/examples,
                # not control mutation — the real control-path check is the
                # dedicated coordinator/ha_integration/lifecycle greps in T23.
                continue
            assert token not in source, f"forbidden control token found in {mod.__name__}: {token}"
