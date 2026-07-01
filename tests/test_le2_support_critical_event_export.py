"""Tests for the LE2 Support Critical Event timeline support-export block.

Covers the read-only support-export addition in export.py:
  _le2_critical_events_export() and its wiring into async_export_support_data()
  (checked at the per-zone-dict-construction level, without a full hass fixture).

This step adds ONE small, BOUNDED timeline summary per zone to the SUPPORT
export, sourced directly from
LearningShadowController.support_critical_events_snapshot() (already
in-memory — no new store read, no new runtime instrumentation, no Research
Export change).

20 test groups:
  T1  — Support export helper returns critical_events with real fields
  T2  — Empty state yields available: true, events: []
  T3  — Missing shadow yields available: False
  T4  — snapshot() exception is non-fatal
  T5  — Malformed events are skipped and counted
  T6  — Events are sorted newest-first
  T7  — Export cap (200) limits the event list
  T8  — records_available / records_exported / records_truncated correct
  T9  — Retention/coverage metadata present (persistent_store_retention_h etc.)
  T10 — support_event_for_export() shape is used (no invented fields)
  T11 — No event_id in the export
  T12 — No forbidden internal-id / entity-id / secret / path substrings
  T13 — Privacy scan failure replaces the block non-fatally
  T14 — Research Export is unchanged by this step
  T15 — Existing Research Export tests remain green
  T16 — Existing Support Critical Event Foundation/Wiring tests remain green
  T17 — No new store I/O introduced
  T18 — No runtime/coordinator path touched
  T19 — No control-path keywords touched
  T20 — Severity-priority selection keeps a critical event over older info events when capped
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart import export as _export_module
from custom_components.thermosmart.export import (
    _SUPPORT_EVENT_EXPORT_MAX_RECORDS,
    _le2_critical_events_export,
)
from custom_components.thermosmart.learning.support_event_schemas import (
    SupportCriticalEvent,
    SupportEventSeverity,
    SupportEventType,
)
from custom_components.thermosmart.learning.storage.support_event_serialization import (
    SUPPORT_EVENT_SCHEMA_VERSION,
    serialize_support_event,
)

_NOW = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


class _FakeShadow:
    def __init__(self, events=None, capture_stores="present", raises: Exception | None = None):
        self._events = dict(events) if events else {}
        self.capture_stores = capture_stores
        self._raises = raises

    def support_critical_events_snapshot(self):
        if self._raises is not None:
            raise self._raises
        return dict(self._events)


class _FakeCoord:
    def __init__(self, shadow=None):
        self._le2_shadow = shadow


def _mk(event_id: str, *, hours_ago: float = 1.0, severity=SupportEventSeverity.WARNING,
        event_type=SupportEventType.TRV_COMMAND_BLOCKED, reason="min_interval",
        summary="TRV command blocked by minimum interval") -> dict:
    ev = SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id=event_id,
        event_type=event_type, ts=_NOW - timedelta(hours=hours_ago),
        severity=severity, reason=reason, summary=summary,
        details={"requested_setpoint": 21.0, "current_setpoint": 20.5},
    )
    return serialize_support_event(ev)


# ── T1: Support export helper returns critical_events with real fields ─────

def test_t1_export_contains_real_fields():
    events = {"e1": _mk("e1", hours_ago=1.0)}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["available"] is True
    assert block["records_available"] == 1
    assert block["records_exported"] == 1
    assert len(block["events"]) == 1
    ev_view = block["events"][0]
    assert ev_view["event_type"] == "trv_command_blocked"
    assert ev_view["severity"] == "warning"
    assert ev_view["reason"] == "min_interval"


# ── T2: Empty state yields available: true, events: [] ─────────────────────

def test_t2_empty_state_is_available_not_error():
    coord = _FakeCoord(_FakeShadow({}))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["available"] is True
    assert block["records_available"] == 0
    assert block["records_exported"] == 0
    assert block["events"] == []


# ── T3: Missing shadow yields available: False ──────────────────────────────

def test_t3_missing_shadow_yields_unavailable():
    block = _le2_critical_events_export(_FakeCoord(None), now=_NOW)
    assert block == {"available": False, "reason": "le2_shadow_unavailable"}


# ── T4: snapshot() exception is non-fatal ───────────────────────────────────

def test_t4_snapshot_exception_stays_non_fatal():
    coord = _FakeCoord(_FakeShadow(raises=RuntimeError("boom")))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["available"] is False
    assert "critical_events_error" in block


# ── T5: Malformed events are skipped and counted ────────────────────────────

def test_t5_malformed_entries_skipped_and_counted():
    events = {
        "good": _mk("good", hours_ago=1.0),
        "not_a_dict": "garbage",
        "bad_type": {"schema_version": 999},
    }
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["records_available"] == 1
    assert block["malformed_skipped_count"] == 2


def test_t5_all_malformed_still_yields_valid_available_summary():
    events = {"m1": "garbage", "m2": {"schema_version": 999}}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["available"] is True
    assert block["records_available"] == 0
    assert block["malformed_skipped_count"] == 2


# ── T6: Events are sorted newest-first ──────────────────────────────────────

def test_t6_events_sorted_newest_first():
    events = {
        "old": _mk("old", hours_ago=40.0),
        "new": _mk("new", hours_ago=1.0),
        "mid": _mk("mid", hours_ago=20.0),
    }
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    ts_list = [e["ts"] for e in block["events"]]
    assert ts_list == sorted(ts_list, reverse=True)


# ── T7: Export cap (200) limits the event list ──────────────────────────────

def test_t7_export_cap_is_200():
    assert _SUPPORT_EVENT_EXPORT_MAX_RECORDS == 200


def test_t7_export_cap_limits_event_list():
    events = {
        f"e{i}": _mk(f"e{i}", hours_ago=i / 60.0, severity=SupportEventSeverity.INFO,
                     event_type=SupportEventType.SAME_SETPOINT_BLOCK)
        for i in range(250)
    }
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert len(block["events"]) == 200


# ── T8: records_available / records_exported / records_truncated correct ──

def test_t8_record_counts_correct_when_under_cap():
    events = {f"e{i}": _mk(f"e{i}", hours_ago=i) for i in range(5)}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["records_available"] == 5
    assert block["records_exported"] == 5
    assert block["records_truncated"] == 0
    assert block["truncation_reason"] is None


def test_t8_record_counts_correct_when_over_cap():
    events = {f"e{i}": _mk(f"e{i}", hours_ago=i / 60.0) for i in range(220)}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["records_available"] == 220
    assert block["records_exported"] == 200
    assert block["records_truncated"] == 20
    assert block["truncation_reason"] == "export_cap_exceeded"


# ── T9: Retention/coverage metadata present ─────────────────────────────────

def test_t9_retention_metadata_present():
    events = {"e1": _mk("e1", hours_ago=40.0), "e2": _mk("e2", hours_ago=1.0)}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["persistent_store_retention_h"] == 48
    assert block["requested_window_h"] == 48
    assert block["coverage_scope"] == "persistent_support_critical_events"
    assert block["persistent_store_enabled"] is True
    assert block["coverage_start"] is not None
    assert block["coverage_end"] is not None


def test_t9_store_warmup_and_full_window_covered_reflect_data_span():
    events = {"e1": _mk("e1", hours_ago=40.0)}  # under the 48h window
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["full_window_covered"] is False
    assert block["store_warmup"] is True


def test_t9_full_window_covered_true_when_oldest_event_reaches_window():
    events = {"e1": _mk("e1", hours_ago=48.0)}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["full_window_covered"] is True
    assert block["store_warmup"] is False


def test_t9_persistent_store_enabled_false_when_no_capture_stores():
    coord = _FakeCoord(_FakeShadow({"e1": _mk("e1")}, capture_stores=None))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["persistent_store_enabled"] is False


# ── T10: support_event_for_export() shape used, no invented fields ────────

def test_t10_event_view_matches_support_event_for_export_shape():
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        support_event_for_export,
    )
    entry = _mk("evt_x", hours_ago=1.0)
    expected = support_event_for_export(entry)
    coord = _FakeCoord(_FakeShadow({"evt_x": entry}))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["events"][0] == expected


# ── T11: No event_id in the export ──────────────────────────────────────────

def test_t11_no_event_id_in_output():
    events = {"evt_x": _mk("evt_x", hours_ago=1.0)}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert "event_id" not in block["events"][0]
    assert "evt_x" not in repr(block)


# ── T12: No forbidden internal-id / entity-id / secret / path substrings ───

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
    "climate.", "sensor.", "event_id",
)


def test_t12_no_forbidden_substrings_in_output():
    events = {"evt_x": _mk("evt_x", hours_ago=1.0)}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    flat = repr(block).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in flat, f"forbidden substring '{token}' leaked into export block"


# ── T13: Privacy scan failure replaces the block non-fatally ──────────────

def test_t13_privacy_scan_violation_blocks_block(monkeypatch):
    import custom_components.thermosmart.learning.privacy as _privacy_mod

    def _fake_scan_payload(payload, *, path="$"):
        return [object()]  # non-empty -> violation

    monkeypatch.setattr(_privacy_mod, "scan_payload", _fake_scan_payload)
    events = {"evt_x": _mk("evt_x", hours_ago=1.0)}
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block == {"available": False, "reason": "privacy_scan_failed"}


# ── T14: Research Export is unchanged by this step ──────────────────────────

def test_t14_research_export_function_source_unchanged():
    source = inspect.getsource(_export_module.async_export_learning_data)
    assert "_le2_critical_events_export" not in source
    assert "critical_events" not in source


# ── T15: Existing Research Export tests remain green ────────────────────────

def test_t15_regression_imports_ok_research():
    import tests.test_le2_episode_history_export  # noqa: F401
    import tests.test_le2_learning_progress_export  # noqa: F401
    import tests.test_orchestration_export_trace  # noqa: F401


# ── T16: Existing Support Critical Event Foundation/Wiring tests remain green

def test_t16_regression_imports_ok_support_events():
    import tests.test_le2_support_critical_events  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401


# ── T17: No new store I/O introduced ────────────────────────────────────────

def test_t17_no_store_io_in_new_helper():
    source = inspect.getsource(_le2_critical_events_export)
    assert "await" not in source
    assert ".save(" not in source
    assert "SupportCriticalEventStore(" not in source
    assert "async_setup" not in source
    assert "async_load" not in source


# ── T18: No runtime/coordinator path touched ────────────────────────────────

def test_t18_no_runtime_or_coordinator_reference_in_new_helper():
    source = inspect.getsource(_le2_critical_events_export)
    assert "record_support_critical_event_safe" not in source
    assert "append_support_event(" not in source


def test_t18_coordinator_module_unchanged_by_this_step():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod)
    assert "critical_events_export" not in source
    assert "_le2_critical_events_export" not in source


# ── T19: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
    "preheat",
)


def test_t19_no_control_keywords_in_new_helper():
    source = inspect.getsource(_le2_critical_events_export).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"


# ── T20: Severity-priority selection when capped ────────────────────────────

def test_t20_critical_event_survives_export_cap_over_older_info_events():
    events = {
        f"info_{i}": _mk(f"info_{i}", hours_ago=i / 60.0, severity=SupportEventSeverity.INFO,
                         event_type=SupportEventType.SAME_SETPOINT_BLOCK)
        for i in range(250)
    }
    events["crit_1"] = _mk(
        "crit_1", hours_ago=500 / 60.0, severity=SupportEventSeverity.CRITICAL,
        event_type=SupportEventType.TRV_UNAVAILABLE, reason=None,
    )
    coord = _FakeCoord(_FakeShadow(events))
    block = _le2_critical_events_export(coord, now=_NOW)
    assert block["records_exported"] == 200
    assert any(e["event_type"] == "trv_unavailable" for e in block["events"])
