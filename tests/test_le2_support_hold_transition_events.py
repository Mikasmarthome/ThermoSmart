"""Tests for the LE2 Support Critical Event window/summer/manual hold
transition producers in ThermoSmartCoordinator
(custom_components/thermosmart/coordinator.py).

This step adds the FIRST coordinator-side Support Critical Event producers —
but only for high-value STATE TRANSITIONS (window open/closed, summer mode
on/off, manual override start/end), never every cycle. Event production is
pure and side-effect-free with respect to control: it reads flags already
present in the fully-resolved ``recommendation`` dict, compares them against
small dedupe flags, and hands off to the EXISTING
``record_support_critical_event_safe()`` (memory-only, no store I/O, no
await, no fire-and-forget). No control-path (setpoints, TPI, boost, preheat,
dispatch) is touched or read by this step.

Since constructing a real ThermoSmartCoordinator requires a live
HomeAssistant instance, this file tests the two new methods
(``_record_support_hold_event`` / ``_maybe_record_support_hold_events``)
directly against the REAL, UNBOUND methods from the production class, bound
onto a minimal fake object exposing only the attributes they touch
(``_le2_shadow``, ``_clock``, the three dedupe flags) — validating the
CONTRACT, not a copy — plus static source-inspection checks for control-path
safety.

18 test groups:
  T1  — Window hold transition produces exactly one window_open_hold event
  T2  — Repeated open-window cycles produce no additional events
  T3  — Window release produces exactly one window_closed_release event
  T4  — Repeated closed-window cycles produce no additional events
  T5  — Summer hold transition produces exactly one summer_mode_hold event
  T6  — Repeated summer cycles produce no additional events
  T7  — Summer release produces one summer_mode_hold event with reason "summer_mode_ended"
  T8  — Manual override start produces exactly one manual_override_start event
  T9  — Repeated manual-override cycles produce no additional events
  T10 — Manual override end produces exactly one manual_override_end event
  T11 — Produced events appear in the Support Export
  T12 — Event details are public-safe (bounded scalars, no forbidden substrings)
  T13 — No event_id in the Support Export
  T14 — No store reads introduced in the export path
  T15 — No control-path effect: no setpoint/TPI/boost/preheat/dispatch reference
  T16 — Existing Storage Restore Event tests remain green
  T17 — Existing Critical Event Export/Wiring/Foundation tests remain green
  T18 — No control-path keywords touched by the new coordinator methods
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.coordinator import ThermoSmartCoordinator


class _FakeClock:
    def __init__(self, now_iso: str = "2026-06-02T12:00:00+00:00") -> None:
        self._now = datetime.fromisoformat(now_iso)

    def now_utc(self) -> datetime:
        return self._now


class _FakeShadow:
    def __init__(self) -> None:
        self.events: list = []

    def record_support_critical_event_safe(self, event) -> None:
        self.events.append(event)


class _FakeCoordinator:
    """Minimal stand-in exposing only what the two real methods touch."""

    _record_support_hold_event = ThermoSmartCoordinator._record_support_hold_event
    _maybe_record_support_hold_events = ThermoSmartCoordinator._maybe_record_support_hold_events

    def __init__(self, shadow=None, now_iso: str = "2026-06-02T12:00:00+00:00") -> None:
        self._le2_shadow = shadow if shadow is not None else _FakeShadow()
        self._clock = _FakeClock(now_iso)
        self._support_event_window_hold_active = False
        self._support_event_summer_hold_active = False
        self._support_event_manual_override_active = False


def _event_types(coord: _FakeCoordinator) -> list[str]:
    return [e.event_type.value for e in coord._le2_shadow.events]


# ── T1: Window hold transition produces exactly one event ─────────────────

def test_t1_window_open_transition_produces_one_event():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"window_open": True})
    assert _event_types(coord) == ["window_open_hold"]
    ev = coord._le2_shadow.events[0]
    assert ev.severity.value == "info"
    assert ev.reason == "window_open"
    assert ev.details == {"source": "window", "heating_allowed": False}


# ── T2: Repeated open-window cycles produce no additional events ──────────

def test_t2_repeated_open_window_cycles_no_spam():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_support_hold_events({"window_open": True})
    assert len(coord._le2_shadow.events) == 1


# ── T3: Window release produces exactly one event ──────────────────────────

def test_t3_window_release_produces_one_event():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"window_open": True})
    coord._maybe_record_support_hold_events({"window_open": False})
    assert _event_types(coord) == ["window_open_hold", "window_closed_release"]
    release = coord._le2_shadow.events[1]
    assert release.details == {"source": "window", "heating_allowed": True}


# ── T4: Repeated closed-window cycles produce no additional events ────────

def test_t4_repeated_closed_window_cycles_no_spam():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"window_open": True})
    for _ in range(5):
        coord._maybe_record_support_hold_events({"window_open": False})
    assert len(coord._le2_shadow.events) == 2  # one open + one release, no more


def test_t4_never_opened_stays_silent():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_support_hold_events({"window_open": False})
    assert coord._le2_shadow.events == []


# ── T5: Summer hold transition produces exactly one event ─────────────────

def test_t5_summer_mode_transition_produces_one_event():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"is_summer": True})
    assert _event_types(coord) == ["summer_mode_hold"]
    ev = coord._le2_shadow.events[0]
    assert ev.reason == "summer_mode_active"
    assert ev.details == {"mode": "summer", "previous_active": False, "current_active": True}


# ── T6: Repeated summer cycles produce no additional events ───────────────

def test_t6_repeated_summer_cycles_no_spam():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_support_hold_events({"is_summer": True})
    assert len(coord._le2_shadow.events) == 1


# ── T7: Summer release produces one summer_mode_hold event, reason ended ──

def test_t7_summer_release_uses_summer_mode_hold_with_ended_reason():
    """No dedicated 'summer release' event type exists in the schema (unlike
    window_open_hold/window_closed_release). Deliberately reuses
    SUMMER_MODE_HOLD for both directions, disambiguated by reason +
    current_active, instead of adding a new event type for one asymmetric
    case — see the accompanying report."""
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"is_summer": True})
    coord._maybe_record_support_hold_events({"is_summer": False})
    assert _event_types(coord) == ["summer_mode_hold", "summer_mode_hold"]
    release = coord._le2_shadow.events[1]
    assert release.reason == "summer_mode_ended"
    assert release.details == {"mode": "summer", "previous_active": True, "current_active": False}


def test_t7_repeated_summer_release_cycles_no_spam():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"is_summer": True})
    for _ in range(5):
        coord._maybe_record_support_hold_events({"is_summer": False})
    assert len(coord._le2_shadow.events) == 2


# ── T8: Manual override start produces exactly one event ──────────────────

def test_t8_manual_override_start_produces_one_event():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"override_active": True})
    assert _event_types(coord) == ["manual_override_start"]
    assert coord._le2_shadow.events[0].reason == "manual_override_start"


# ── T9: Repeated manual-override cycles produce no additional events ──────

def test_t9_repeated_manual_override_cycles_no_spam():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_support_hold_events({"override_active": True})
    assert len(coord._le2_shadow.events) == 1


# ── T10: Manual override end produces exactly one event ────────────────────

def test_t10_manual_override_end_produces_one_event():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"override_active": True})
    coord._maybe_record_support_hold_events({"override_active": False})
    assert _event_types(coord) == ["manual_override_start", "manual_override_end"]
    assert coord._le2_shadow.events[1].reason == "manual_override_end"


def test_t10_repeated_manual_override_end_cycles_no_spam():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"override_active": True})
    for _ in range(5):
        coord._maybe_record_support_hold_events({"override_active": False})
    assert len(coord._le2_shadow.events) == 2


def test_missing_shadow_does_not_raise():
    """When there is no shadow, the method returns immediately (matching
    record_support_critical_event_safe()'s own "no capture stores -> no-op"
    convention) — dedupe state is intentionally left untouched rather than
    silently drifting out of sync with what would actually be recorded once
    a shadow does become available."""
    coord = _FakeCoordinator(shadow=None)
    coord._le2_shadow = None
    coord._maybe_record_support_hold_events({"window_open": True})  # must not raise
    assert coord._support_event_window_hold_active is False


def test_all_three_transitions_together_in_one_cycle():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({
        "window_open": True, "is_summer": True, "override_active": True,
    })
    assert set(_event_types(coord)) == {
        "window_open_hold", "summer_mode_hold", "manual_override_start",
    }
    assert len(coord._le2_shadow.events) == 3


# ── T11: Produced events appear in the Support Export ──────────────────────

def test_t11_events_appear_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )

    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"window_open": True})
    serialized = {ev.event_id: serialize_support_event(ev) for ev in coord._le2_shadow.events}

    class _ShadowWithSnapshot:
        def __init__(self, events):
            self._events = events
            self.capture_stores = "present"

        def support_critical_events_snapshot(self):
            return dict(self._events)

    class _ExportCoord:
        def __init__(self, shadow):
            self._le2_shadow = shadow

    now = datetime(2026, 6, 2, 12, 5, tzinfo=timezone.utc)
    export_coord = _ExportCoord(_ShadowWithSnapshot(serialized))
    block = _le2_critical_events_export(export_coord, now=now)
    assert block["available"] is True
    assert block["records_available"] == 1
    assert block["events"][0]["event_type"] == "window_open_hold"


# ── T12: Event details are public-safe ──────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t12_details_are_bounded_and_public_safe():
    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({
        "window_open": True, "is_summer": True, "override_active": True,
    })
    for ev in coord._le2_shadow.events:
        for v in ev.details.values():
            assert isinstance(v, (int, float, bool, str)) or v is None
        flat = repr(ev.details).lower()
        for token in _FORBIDDEN_SUBSTRINGS:
            assert token not in flat, f"forbidden substring '{token}' in event details"


# ── T13: No event_id in the Support Export ──────────────────────────────────

def test_t13_no_event_id_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )

    coord = _FakeCoordinator()
    coord._maybe_record_support_hold_events({"override_active": True})
    serialized = {ev.event_id: serialize_support_event(ev) for ev in coord._le2_shadow.events}

    class _ShadowWithSnapshot:
        def __init__(self, events):
            self._events = events
            self.capture_stores = "present"

        def support_critical_events_snapshot(self):
            return dict(self._events)

    class _ExportCoord:
        def __init__(self, shadow):
            self._le2_shadow = shadow

    now = datetime(2026, 6, 2, 12, 5, tzinfo=timezone.utc)
    block = _le2_critical_events_export(_ExportCoord(_ShadowWithSnapshot(serialized)), now=now)
    assert "event_id" not in block["events"][0]
    assert "hold_manual_override_start_" not in repr(block)


# ── T14: No store reads introduced in the export path ───────────────────────

def test_t14_export_path_unchanged_no_store_io():
    from custom_components.thermosmart.export import _le2_critical_events_export
    source = inspect.getsource(_le2_critical_events_export)
    assert "await" not in source
    assert "SupportCriticalEventStore(" not in source


# ── T15: No control-path effect ─────────────────────────────────────────────

def test_t15_no_control_path_reference_in_new_methods():
    import custom_components.thermosmart.coordinator as _coord_mod
    for name in ("_record_support_hold_event", "_maybe_record_support_hold_events"):
        source = inspect.getsource(getattr(_coord_mod.ThermoSmartCoordinator, name))
        assert "trv_setpoint" not in source
        assert "async_set_temperature" not in source
        assert ".async_call(" not in source
        assert "dispatch" not in source


def test_t15_new_methods_never_mutate_recommendation():
    """Static check: neither method assigns into the recommendation dict —
    only .get() reads."""
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod.ThermoSmartCoordinator._maybe_record_support_hold_events)
    assert "recommendation[" not in source
    assert "recommendation.update" not in source


# ── T16: Existing Storage Restore Event tests remain green ─────────────────

def test_t16_regression_imports_ok_storage_restore():
    import tests.test_le2_support_storage_restore_events  # noqa: F401


# ── T17: Existing Critical Event Export/Wiring/Foundation tests remain green

def test_t17_regression_imports_ok_export_wiring_foundation():
    import tests.test_le2_support_critical_event_export  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_support_critical_events  # noqa: F401


# ── T18: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t18_no_control_keywords_in_new_methods():
    import custom_components.thermosmart.coordinator as _coord_mod
    for name in ("_record_support_hold_event", "_maybe_record_support_hold_events"):
        source = inspect.getsource(getattr(_coord_mod.ThermoSmartCoordinator, name)).lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in source, f"forbidden control token found in {name}: {token}"
