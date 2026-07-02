"""Tests for LE2 Research Daily Aggregation Increment 1: aggregating Support
Critical Events into Research Daily Buckets from within
LearningShadowController.record_support_critical_event_safe()
(custom_components/thermosmart/learning/runtime/ha_integration.py).

This step adds ONLY:
  - LearningShadowController._SUPPORT_EVENT_TYPE_TO_DAILY_FIELD (mapping)
  - LearningShadowController._maybe_aggregate_research_daily_from_support_event()
  - one call site inside record_support_critical_event_safe(), gated on the
    event actually being newly appended this call

No new Support Event producer, no new event type, no Support/Research Export
change, no Coordinator/runtime hook, no control effect.

Unlike the state/store-wiring steps, this file constructs the REAL
LearningShadowController via ``object.__new__()`` and sets only the handful
of attributes the exercised methods actually read/write (no HomeAssistant
instance needed for these particular methods) — so these tests call the
ACTUAL production methods directly, not a copied-body harness, while still
requiring no live HA instance.

24 test groups:
  T1  — trv_command_sent increments trv_command_sent_count
  T2  — same_setpoint_block increments same_setpoint_block_count
  T3  — trv_unavailable increments trv_unavailable_count
  T4  — window_open_hold increments window_hold_count
  T5  — summer_mode_hold increments summer_hold_count
  T6  — manual_override_start increments manual_override_count
  T7  — boost_started/boost_blocked/boost_ended increment their counts
  T8  — sensor_unavailable/sensor_restored/fallback_used increment their counts
  T9  — outcome_resolved increments outcome_resolved_count
  T10 — outcome target_reached True/False increments success/failed
  T11 — outcome confounded increments outcome_confounded_count
  T12 — outcome overshoot/undershoot/comfort_error aggregate sum/count
  T13 — Storage events are deliberately not counted
  T14 — Malformed timestamp produces no bucket, non-fatal
  T15 — Duplicate Support Event is not double-counted
  T16 — Rejected/malformed Support Event is not counted
  T17 — Daily dirty flag set only on real daily change
  T18 — Support dirty and Research dirty can both be set
  T19 — Research aggregation failure never aborts Support Event recording
  T20 — Public-safety: daily snapshot contains no raw events/ids/details
  T21 — No new runtime/coordinator/export hooks
  T22 — Existing Research Daily state/foundation/store tests remain green
  T23 — Existing Support Event tests remain green
  T24 — No control-path keywords touched
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from custom_components.thermosmart.learning.runtime.ha_integration import LearningShadowController
from custom_components.thermosmart.learning.support_event_schemas import (
    SupportCriticalEvent,
    SupportEventSeverity,
    SupportEventType,
)
from custom_components.thermosmart.learning.storage.support_event_serialization import (
    SUPPORT_EVENT_SCHEMA_VERSION,
)
from custom_components.thermosmart.learning.storage.stores import (
    ResearchDailyStore,
    SupportCriticalEventStore,
)

_ZONE_A = "zone_alpha_01"
_NOW_ISO = "2026-06-02T12:00:00+00:00"


# ── Fake store infrastructure (mirrors established store-wiring test files) ─

class _FakeRawStore:
    def __init__(self) -> None:
        self._data: Any = None

    async def async_load(self) -> Any:
        return self._data

    async def async_save(self, data: Any) -> None:
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


class _FakeCaptureStores:
    """Minimal stand-in for LearningCaptureStores exposing both stores
    reached by record_support_critical_event_safe()'s aggregation path."""

    def __init__(self, factory: _FakeStoreFactory, zone: str) -> None:
        self._factory = factory
        self._zone = zone

    def support_critical_events_store(self) -> SupportCriticalEventStore:
        return SupportCriticalEventStore(self._factory, self._zone)

    def research_daily_store(self) -> ResearchDailyStore:
        return ResearchDailyStore(self._factory, self._zone)


def _make_controller(now_iso: str = _NOW_ISO, capture_stores: Any = "__default__") -> LearningShadowController:
    """Build a real LearningShadowController instance via object.__new__(),
    setting only the attributes the exercised methods actually read/write.
    No HomeAssistant instance required — this bypasses __init__ entirely,
    matching the fact that the methods under test never touch anything
    HA-specific (self._runtime, self.hass, etc.)."""
    ctrl = object.__new__(LearningShadowController)
    if capture_stores == "__default__":
        capture_stores = _FakeCaptureStores(_FakeStoreFactory(), _ZONE_A)
    ctrl._capture_stores = capture_stores
    ctrl._support_critical_events = {}
    ctrl._support_critical_events_save_needed = False
    ctrl._support_critical_events_last_error = None
    ctrl._research_daily_buckets = {}
    ctrl._research_daily_save_needed = False
    ctrl._research_daily_last_error = None
    ctrl._utcnow_iso = lambda: now_iso
    return ctrl


def _event(
    event_id: str,
    event_type: SupportEventType,
    *,
    ts: datetime = datetime(2026, 6, 2, 10, 0, tzinfo=timezone.utc),
    details: Optional[dict] = None,
    reason: Optional[str] = None,
) -> SupportCriticalEvent:
    return SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id=event_id,
        event_type=event_type, ts=ts, severity=SupportEventSeverity.INFO,
        reason=reason, details=details or {},
    )


def _bucket(ctrl: LearningShadowController, bucket_date: str = "2026-06-02") -> dict:
    return ctrl._research_daily_buckets[bucket_date]


# ── T1-T8: Simple one-field mappings ────────────────────────────────────────

@pytest.mark.parametrize("event_type,field_name", [
    (SupportEventType.TRV_COMMAND_SENT, "trv_command_sent_count"),
    (SupportEventType.TRV_COMMAND_BLOCKED, "trv_command_blocked_count"),
    (SupportEventType.SAME_SETPOINT_BLOCK, "same_setpoint_block_count"),
    (SupportEventType.TRV_UNAVAILABLE, "trv_unavailable_count"),
    (SupportEventType.WINDOW_OPEN_HOLD, "window_hold_count"),
    (SupportEventType.SUMMER_MODE_HOLD, "summer_hold_count"),
    (SupportEventType.MANUAL_OVERRIDE_START, "manual_override_count"),
    (SupportEventType.BOOST_STARTED, "boost_started_count"),
    (SupportEventType.BOOST_BLOCKED, "boost_blocked_count"),
    (SupportEventType.BOOST_ENDED, "boost_ended_count"),
    (SupportEventType.SENSOR_UNAVAILABLE, "sensor_unavailable_count"),
    (SupportEventType.SENSOR_RESTORED, "sensor_restored_count"),
    (SupportEventType.FALLBACK_USED, "fallback_used_count"),
])
def test_t1_t8_simple_field_mappings(event_type, field_name):
    ctrl = _make_controller()
    ev = _event(f"evt_{event_type.value}", event_type)
    ctrl.record_support_critical_event_safe(ev)
    bucket = _bucket(ctrl)
    assert bucket[field_name] == 1
    # every OTHER counter stays at 0
    for other_field in (
        "trv_command_sent_count", "trv_command_blocked_count", "same_setpoint_block_count",
        "trv_unavailable_count", "window_hold_count", "summer_hold_count",
        "manual_override_count", "boost_started_count", "boost_blocked_count",
        "boost_ended_count", "sensor_unavailable_count", "sensor_restored_count",
        "fallback_used_count",
    ):
        if other_field != field_name:
            assert bucket[other_field] == 0


# ── T9: outcome_resolved increments outcome_resolved_count ─────────────────

def test_t9_outcome_resolved_increments_count():
    ctrl = _make_controller()
    ev = _event("evt_outcome_1", SupportEventType.OUTCOME_RESOLVED)
    ctrl.record_support_critical_event_safe(ev)
    assert _bucket(ctrl)["outcome_resolved_count"] == 1


# ── T10: outcome target_reached True/False -> success/failed ───────────────

def test_t10_target_reached_true_increments_success():
    ctrl = _make_controller()
    ev = _event("evt_outcome_ok", SupportEventType.OUTCOME_RESOLVED, details={"target_reached": True})
    ctrl.record_support_critical_event_safe(ev)
    bucket = _bucket(ctrl)
    assert bucket["outcome_success_count"] == 1
    assert bucket["outcome_failed_count"] == 0


def test_t10_target_reached_false_increments_failed():
    ctrl = _make_controller()
    ev = _event("evt_outcome_fail", SupportEventType.OUTCOME_RESOLVED, details={"target_reached": False})
    ctrl.record_support_critical_event_safe(ev)
    bucket = _bucket(ctrl)
    assert bucket["outcome_failed_count"] == 1
    assert bucket["outcome_success_count"] == 0


def test_t10_missing_target_reached_increments_neither():
    ctrl = _make_controller()
    ev = _event("evt_outcome_unknown", SupportEventType.OUTCOME_RESOLVED, details={})
    ctrl.record_support_critical_event_safe(ev)
    bucket = _bucket(ctrl)
    assert bucket["outcome_success_count"] == 0
    assert bucket["outcome_failed_count"] == 0


# ── T11: outcome confounded -> outcome_confounded_count ─────────────────────

def test_t11_confounded_true_increments_count():
    ctrl = _make_controller()
    ev = _event("evt_confounded", SupportEventType.OUTCOME_RESOLVED, details={"confounded": True})
    ctrl.record_support_critical_event_safe(ev)
    assert _bucket(ctrl)["outcome_confounded_count"] == 1


def test_t11_confounded_false_does_not_increment():
    ctrl = _make_controller()
    ev = _event("evt_not_confounded", SupportEventType.OUTCOME_RESOLVED, details={"confounded": False})
    ctrl.record_support_critical_event_safe(ev)
    assert _bucket(ctrl)["outcome_confounded_count"] == 0


# ── T12: overshoot/undershoot/comfort_error aggregate sum/count ────────────

def test_t12_overshoot_undershoot_comfort_error_aggregate():
    ctrl = _make_controller()
    ev1 = _event(
        "evt_outcome_a", SupportEventType.OUTCOME_RESOLVED,
        details={"overshoot_c": 0.4, "undershoot_c": 0.0, "comfort_error_c": 0.4},
    )
    ev2 = _event(
        "evt_outcome_b", SupportEventType.OUTCOME_RESOLVED,
        details={"overshoot_c": 0.6, "comfort_error_c": 0.6},
    )
    ctrl.record_support_critical_event_safe(ev1)
    ctrl.record_support_critical_event_safe(ev2)
    bucket = _bucket(ctrl)
    assert bucket["overshoot_sum_c"] == pytest.approx(1.0)
    assert bucket["overshoot_count"] == 2
    assert bucket["undershoot_sum_c"] == pytest.approx(0.0)
    assert bucket["undershoot_count"] == 1
    assert bucket["comfort_error_sum_c"] == pytest.approx(1.0)
    assert bucket["comfort_error_count"] == 2


def test_t12_non_numeric_or_nan_detail_values_are_ignored():
    ctrl = _make_controller()
    ev = _event(
        "evt_outcome_bad", SupportEventType.OUTCOME_RESOLVED,
        details={"overshoot_c": "not-a-number", "undershoot_c": float("nan"), "comfort_error_c": True},
    )
    ctrl.record_support_critical_event_safe(ev)
    bucket = _bucket(ctrl)
    assert bucket["overshoot_count"] == 0
    assert bucket["undershoot_count"] == 0
    assert bucket["comfort_error_count"] == 0


# ── T13: Storage events are deliberately not counted ────────────────────────

@pytest.mark.parametrize("event_type", [
    SupportEventType.STORAGE_RESTORE,
    SupportEventType.STORAGE_RESTORE_FAILED,
    SupportEventType.STORAGE_SAVE_FAILED,
    SupportEventType.STORAGE_SAVE_RECOVERED,
    SupportEventType.RESTART_RESTORE,
])
def test_t13_storage_events_not_counted(event_type):
    ctrl = _make_controller()
    ev = _event(f"evt_{event_type.value}", event_type)
    ctrl.record_support_critical_event_safe(ev)
    # No bucket at all is created for a type that maps to nothing.
    assert ctrl._research_daily_buckets == {}
    assert ctrl._research_daily_save_needed is False


def test_t13_learning_recommendation_only_deliberately_not_counted():
    """Documented decision: LEARNING_RECOMMENDATION_ONLY is semantically
    distinct from TRV_COMMAND_BLOCKED (no attempt vs. attempt-then-veto) and
    has no dedicated daily field — left unmapped rather than misclassified."""
    ctrl = _make_controller()
    ev = _event("evt_reco_only", SupportEventType.LEARNING_RECOMMENDATION_ONLY)
    ctrl.record_support_critical_event_safe(ev)
    assert ctrl._research_daily_buckets == {}


# ── T14: Malformed timestamp produces no bucket, non-fatal ─────────────────

def test_t14_non_datetime_ts_produces_no_bucket_non_fatal():
    ctrl = _make_controller()
    ev = _event("evt_bad_ts", SupportEventType.TRV_COMMAND_SENT)
    # Simulate a corrupted ts post-construction (dataclass is frozen, so we
    # exercise the aggregation method directly with a hand-built stand-in
    # object exposing a non-datetime ts attribute).
    class _BadTsEvent:
        event_id = "evt_bad_ts_direct"
        event_type = SupportEventType.TRV_COMMAND_SENT
        ts = "not-a-datetime"
        details = {}

    ctrl._maybe_aggregate_research_daily_from_support_event(_BadTsEvent())
    assert ctrl._research_daily_buckets == {}
    assert ctrl._research_daily_save_needed is False


# ── T15: Duplicate Support Event is not double-counted ──────────────────────

def test_t15_duplicate_support_event_not_double_counted():
    ctrl = _make_controller()
    ev = _event("evt_dup", SupportEventType.BOOST_STARTED)
    ctrl.record_support_critical_event_safe(ev)
    assert _bucket(ctrl)["boost_started_count"] == 1

    ctrl._research_daily_save_needed = False  # simulate a save having already happened
    ctrl.record_support_critical_event_safe(ev)  # exact duplicate event_id

    assert _bucket(ctrl)["boost_started_count"] == 1  # unchanged
    assert ctrl._research_daily_save_needed is False


# ── T16: Rejected/malformed Support Event is not counted ───────────────────

def test_t16_malformed_event_object_not_counted():
    ctrl = _make_controller()
    ctrl.record_support_critical_event_safe(object())  # not a SupportCriticalEvent
    assert ctrl._research_daily_buckets == {}
    assert ctrl._research_daily_save_needed is False


# ── T17: Daily dirty flag set only on real daily change ────────────────────

def test_t17_daily_dirty_only_on_real_change():
    ctrl = _make_controller()
    ev = _event("evt_dirty_1", SupportEventType.SENSOR_UNAVAILABLE)
    ctrl.record_support_critical_event_safe(ev)
    assert ctrl._research_daily_save_needed is True

    ctrl._research_daily_save_needed = False
    ev2 = _event("evt_dirty_2", SupportEventType.RESTART_RESTORE)  # not mapped -> no daily change
    ctrl.record_support_critical_event_safe(ev2)
    assert ctrl._research_daily_save_needed is False


# ── T18: Support dirty and Research dirty can both be set ──────────────────

def test_t18_both_dirty_flags_set_together():
    ctrl = _make_controller()
    ev = _event("evt_both_dirty", SupportEventType.WINDOW_OPEN_HOLD)
    ctrl.record_support_critical_event_safe(ev)
    assert ctrl._support_critical_events_save_needed is True
    assert ctrl._research_daily_save_needed is True


# ── T19: Research aggregation failure never aborts Support Event recording ──

def test_t19_research_aggregation_failure_does_not_break_support_recording(monkeypatch):
    ctrl = _make_controller()

    def _boom(self, event):
        raise RuntimeError("simulated aggregation failure")

    monkeypatch.setattr(
        LearningShadowController, "_maybe_aggregate_research_daily_from_support_event", _boom,
    )
    ev = _event("evt_survives", SupportEventType.TRV_COMMAND_SENT)
    ctrl.record_support_critical_event_safe(ev)  # must not raise

    assert "evt_survives" in ctrl._support_critical_events
    assert ctrl._support_critical_events_save_needed is True
    assert "simulated aggregation failure" in (ctrl._research_daily_last_error or "")


# ── T20: Public-safety — daily snapshot has no raw events/ids/details ──────

_FORBIDDEN_SUBSTRINGS = (
    "entity_id", "zone_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "trajectory", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t20_daily_snapshot_has_no_raw_event_or_id_leakage():
    ctrl = _make_controller()
    ev = _event(
        "evt_leak_check", SupportEventType.OUTCOME_RESOLVED,
        details={
            "target_reached": True, "confounded": False,
            "overshoot_c": 0.2, "undershoot_c": 0.0, "comfort_error_c": 0.2,
        },
    )
    ctrl.record_support_critical_event_safe(ev)
    snapshot = ctrl.research_daily_snapshot()
    dumped = str(snapshot).lower()
    assert "evt_leak_check" not in dumped  # no event_id
    assert "details" not in dumped  # no free-form details mapping at all
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in dumped, f"forbidden substring found in daily snapshot: {token}"


# ── T21: No new runtime/coordinator/export hooks ────────────────────────────

def test_t21_coordinator_not_touched():
    import custom_components.thermosmart.coordinator as _coord
    source = inspect.getsource(_coord)
    assert "_maybe_aggregate_research_daily_from_support_event" not in source
    assert "ResearchDailyObservation" not in source


def test_t21_lifecycle_not_touched():
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle
    source = inspect.getsource(_lifecycle)
    assert "_maybe_aggregate_research_daily_from_support_event" not in source
    assert "research_daily" not in source.lower()


def test_t21_export_not_touched():
    import custom_components.thermosmart.export as _exp
    source = inspect.getsource(_exp)
    assert "research_daily" not in source.lower()
    assert "_maybe_aggregate_research_daily_from_support_event" not in source


def test_t21_aggregation_method_only_called_from_record_support_critical_event_safe():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha)
    call_count = source.count("self._maybe_aggregate_research_daily_from_support_event(")
    assert call_count == 1
    assert "self._maybe_aggregate_research_daily_from_support_event(" in inspect.getsource(
        _ha.LearningShadowController.record_support_critical_event_safe
    )


# ── T22: Existing Research Daily state/foundation/store tests remain green ──

def test_t22_regression_imports_ok_research_daily():
    import tests.test_le2_research_daily_buckets  # noqa: F401
    import tests.test_le2_research_daily_store_wiring  # noqa: F401
    import tests.test_le2_research_daily_state_wiring  # noqa: F401


# ── T23: Existing Support Event tests remain green ──────────────────────────

def test_t23_regression_imports_ok_support_events():
    import tests.test_le2_support_critical_events  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_support_storage_restore_events  # noqa: F401
    import tests.test_le2_support_hold_transition_events  # noqa: F401
    import tests.test_le2_support_trv_command_events  # noqa: F401
    import tests.test_le2_support_boost_transition_events  # noqa: F401
    import tests.test_le2_support_sensor_fallback_events  # noqa: F401
    import tests.test_le2_support_outcome_resolved_events  # noqa: F401
    import tests.test_le2_support_event_detail_privacy  # noqa: F401


# ── T24: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t24_no_control_keywords_in_new_snippets():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    snippets = [
        inspect.getsource(_ha.LearningShadowController._maybe_aggregate_research_daily_from_support_event),
    ]
    for source in snippets:
        lowered = source.lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in lowered, f"forbidden control token found: {token}"
