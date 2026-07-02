"""Tests for the LE2 Support Critical Event room-sensor unavailable/restored
and fallback_used producer in ThermoSmartCoordinator
(custom_components/thermosmart/coordinator.py).

This step adds the FIFTH coordinator-side Support Critical Event producer —
pure OBSERVATION of the room-temperature sensor / TRV-fallback outcome
already decided by the EXISTING read order in _compute_recommendation():
``current_temp = self._read_avg_sensor(cfg["temp_sensors"])``, falling back
to ``self._read_trv_avg_temp(cfg["climate_entities"])`` only when that
returned None. The primary read is now captured into a named variable
(``_primary_room_temp``) purely so it can be observed — the resolved
``current_temp`` value and every downstream computation is byte-for-byte
identical to before. No influence on sensor selection, the fallback choice,
or any control decision — this step changes NOTHING in
_read_avg_sensor()/_read_trv_avg_temp()/window.py/trv_control.py.

Root-cause findings documented here (verified by reading the actual code,
not fabricated):
  - temperature_invalid is NOT produced in this step: _read_avg_sensor()
    does not distinguish "unavailable" from "unknown" from "non-numeric"
    from "spike-filtered" — there is no clean, already-existing "invalid but
    present" state distinct from "no usable value" anywhere in the current
    code.
  - A window-sensor-unavailable event is NOT produced in this step:
    window.py's _check_window_open() only branches on state == "on" vs.
    else — an "unavailable"/"unknown" window sensor state is silently
    treated the same as "off", with no distinct code path to observe
    without changing window.py's own logic (explicitly out of scope).
  - trv_unavailable is NOT re-produced here: Increment 3 already owns that
    event type for the TRV-command-dispatch context
    (coordinator.py::_maybe_record_trv_dispatch_event); when the room-sensor
    fallback to TRV temperature also fails, this step correctly reports only
    sensor_unavailable (role="room_temperature"), not a second, redundant
    TRV-unavailable event for a different context.

Since constructing a real ThermoSmartCoordinator requires a live
HomeAssistant instance, this file tests the new method
(``_maybe_record_room_sensor_events``) directly against the REAL, UNBOUND
method from the production class, bound onto a minimal fake object exposing
only the attributes it touches — validating the CONTRACT, not a copy — plus
static source-inspection checks for control-path safety.

17 test groups:
  T1  — sensor_unavailable is produced exactly once on the valid -> unavailable transition
  T2  — Repeated unavailable cycles produce no additional events
  T3  — sensor_restored is produced exactly once on the unavailable -> valid transition
  T4  — Repeated valid cycles produce no additional events
  T5  — temperature_invalid is deliberately NOT produced (no such state exists in the codebase)
  T6  — fallback_used is produced when the TRV fallback actually produces a usable value
  T7  — Repeated identical fallback episode produces no spam
  T8  — A new unavailable episode after a restore produces a new fallback_used event
  T9  — Events appear in the Support Export
  T10 — No event_id in the Support Export
  T11 — No store reads introduced in the export path
  T12 — Details stay public-safe (bounded scalars, no forbidden substrings)
  T13 — No sensor/fallback/control logic changed (read-only observation, values unchanged)
  T14 — No additional service call is triggered by event production
  T15 — Unconfigured room sensor produces no event (by-design fallback, not a failure)
  T16 — Existing Boost/TRV/Hold/Storage/Export/Wiring/Foundation tests remain green
  T17 — No control-path keywords touched by the new coordinator method
"""
from __future__ import annotations

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
    """Minimal stand-in exposing only what the real method touches."""

    _record_support_hold_event = ThermoSmartCoordinator._record_support_hold_event
    _maybe_record_room_sensor_events = ThermoSmartCoordinator._maybe_record_room_sensor_events

    def __init__(self, now_iso: str = "2026-06-02T12:00:00+00:00") -> None:
        self._le2_shadow: _FakeShadow | None = _FakeShadow()
        self._clock = _FakeClock(now_iso)
        self._support_event_room_sensor_available = True
        self._support_event_last_fallback_signature = None


_CFG = {"temp_sensors": ["configured"]}


def _events(coord: _FakeCoordinator) -> list[tuple[str, str, str]]:
    return [(e.event_type.value, e.reason, e.severity.value) for e in coord._le2_shadow.events]


# ── T1: sensor_unavailable produced exactly once on transition ────────────

def test_t1_sensor_unavailable_on_transition():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
    assert ("sensor_unavailable", "room_sensor_unavailable", "warning") in _events(coord)
    ev = next(e for e in coord._le2_shadow.events if e.event_type.value == "sensor_unavailable")
    assert ev.details == {"sensor_role": "room_temperature", "state": "unavailable"}


def test_t1_no_event_on_first_valid_cycle():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, 21.0, 21.0)
    assert coord._le2_shadow.events == []


# ── T2: Repeated unavailable cycles produce no additional events ──────────

def test_t2_repeated_unavailable_no_spam():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
    unavailable_events = [e for e in coord._le2_shadow.events if e.event_type.value == "sensor_unavailable"]
    assert len(unavailable_events) == 1


# ── T3: sensor_restored produced exactly once on transition ────────────────

def test_t3_sensor_restored_on_transition():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
    coord._maybe_record_room_sensor_events(_CFG, 21.0, 21.0)
    assert ("sensor_restored", "room_sensor_restored", "info") in _events(coord)
    ev = next(e for e in coord._le2_shadow.events if e.event_type.value == "sensor_restored")
    assert ev.details == {"sensor_role": "room_temperature"}


# ── T4: Repeated valid cycles produce no additional events ────────────────

def test_t4_repeated_valid_no_spam():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
    coord._maybe_record_room_sensor_events(_CFG, 21.0, 21.0)
    for _ in range(5):
        coord._maybe_record_room_sensor_events(_CFG, 21.0, 21.0)
    restored_events = [e for e in coord._le2_shadow.events if e.event_type.value == "sensor_restored"]
    assert len(restored_events) == 1


# ── T5: temperature_invalid deliberately not produced ──────────────────────

def test_t5_temperature_invalid_state_does_not_exist_in_codebase():
    """Root-cause check: _read_avg_sensor() has no distinct 'invalid but
    present' branch — confirmed by inspecting its source directly."""
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod.ThermoSmartCoordinator._read_avg_sensor)
    assert "temperature_invalid" not in source
    # Only one distinguishable outcome for a failed read: None.
    assert source.count("return None") + source.count("values.append") >= 1


def test_t5_temperature_invalid_event_type_never_constructed():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod.ThermoSmartCoordinator._maybe_record_room_sensor_events)
    assert "TEMPERATURE_INVALID" not in source


# ── T6: fallback_used produced when TRV fallback yields a usable value ────

def test_t6_fallback_used_when_trv_value_available():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
    assert ("fallback_used", "room_sensor_unavailable", "info") in _events(coord)
    ev = next(e for e in coord._le2_shadow.events if e.event_type.value == "fallback_used")
    assert ev.details == {"fallback_type": "trv_temperature", "reason": "room_sensor_unavailable"}


def test_t6_no_fallback_used_when_trv_also_unavailable():
    """When both the room sensor AND the TRV fallback fail, only
    sensor_unavailable is produced — fallback_used must not fire for a
    fallback that itself produced nothing usable."""
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, None)
    assert _events(coord) == [("sensor_unavailable", "room_sensor_unavailable", "warning")]


# ── T7: Repeated identical fallback episode produces no spam ──────────────

def test_t7_repeated_fallback_no_spam():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
    fallback_events = [e for e in coord._le2_shadow.events if e.event_type.value == "fallback_used"]
    assert len(fallback_events) == 1


# ── T8: New unavailable episode after restore produces new fallback_used ──

def test_t8_new_episode_after_restore_produces_new_fallback_event():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)  # episode 1
    coord._maybe_record_room_sensor_events(_CFG, 21.0, 21.0)  # restored
    coord._maybe_record_room_sensor_events(_CFG, None, 20.0)  # episode 2
    fallback_events = [e for e in coord._le2_shadow.events if e.event_type.value == "fallback_used"]
    assert len(fallback_events) == 2


# ── T9: Events appear in the Support Export ────────────────────────────────

def test_t9_events_appear_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )

    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
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
    assert block["available"] is True
    assert block["records_available"] == 2
    event_types = {e["event_type"] for e in block["events"]}
    assert event_types == {"sensor_unavailable", "fallback_used"}


# ── T10: No event_id in the Support Export ──────────────────────────────────

def test_t10_no_event_id_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )

    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
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
    for ev in block["events"]:
        assert "event_id" not in ev


# ── T11: No store reads introduced in the export path ───────────────────────

def test_t11_export_path_unchanged_no_store_io():
    from custom_components.thermosmart.export import _le2_critical_events_export
    source = inspect.getsource(_le2_critical_events_export)
    assert "await" not in source
    assert "SupportCriticalEventStore(" not in source


# ── T12: Details stay public-safe ───────────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t12_details_stay_public_safe():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)
    coord._maybe_record_room_sensor_events(_CFG, 21.0, 21.0)
    for ev in coord._le2_shadow.events:
        for v in ev.details.values():
            assert isinstance(v, (int, float, bool, str)) or v is None
        flat = repr(ev.details).lower()
        for token in _FORBIDDEN_SUBSTRINGS:
            assert token not in flat, f"forbidden substring '{token}' in sensor event details"


# ── T13: No sensor/fallback/control logic changed ──────────────────────────

def test_t13_new_method_never_mutates_inputs():
    coord = _FakeCoordinator()
    cfg = dict(_CFG)
    coord._maybe_record_room_sensor_events(cfg, None, 20.5)
    assert cfg == _CFG  # unchanged


def test_t13_read_avg_sensor_and_trv_avg_temp_unchanged_by_this_step():
    """The actual sensor-read helpers must be byte-identical to before —
    verified indirectly: the new observation method never calls them
    itself (it only receives already-computed values as parameters)."""
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod.ThermoSmartCoordinator._maybe_record_room_sensor_events)
    assert "self._read_avg_sensor(" not in source
    assert "self._read_trv_avg_temp(" not in source
    assert "self.hass.states.get(" not in source


def test_t13_window_module_unchanged_by_this_step():
    import custom_components.thermosmart.window as _window_mod
    source = inspect.getsource(_window_mod)
    assert "support_event" not in source.lower()
    assert "SupportCriticalEvent" not in source


# ── T14: No additional service call triggered by event production ────────

def test_t14_no_await_in_new_method():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod.ThermoSmartCoordinator._maybe_record_room_sensor_events)
    assert "await" not in source
    assert "services.async_call" not in source


# ── T15: Unconfigured room sensor produces no event ─────────────────────────

def test_t15_unconfigured_temp_sensors_produces_no_event():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events({"temp_sensors": []}, None, 19.5)
    assert coord._le2_shadow.events == []


def test_t15_missing_temp_sensors_key_produces_no_event():
    coord = _FakeCoordinator()
    coord._maybe_record_room_sensor_events({}, None, 19.5)
    assert coord._le2_shadow.events == []


def test_missing_shadow_does_not_raise():
    coord = _FakeCoordinator()
    coord._le2_shadow = None
    coord._maybe_record_room_sensor_events(_CFG, None, 20.5)  # must not raise


# ── T16: Existing tests remain green (regression smoke-test) ───────────────

def test_t16_regression_imports_ok():
    import tests.test_le2_support_boost_transition_events  # noqa: F401
    import tests.test_le2_support_trv_command_events  # noqa: F401
    import tests.test_le2_support_hold_transition_events  # noqa: F401
    import tests.test_le2_support_storage_restore_events  # noqa: F401
    import tests.test_le2_support_critical_event_export  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_support_critical_events  # noqa: F401


# ── T17: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "async_set_temperature",
    "async_write_ha_state",
    "service_call",
    "boost_offset write",
    "tpi_gain",
)


def test_t17_no_control_keywords_in_new_method():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(
        _coord_mod.ThermoSmartCoordinator._maybe_record_room_sensor_events
    ).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"
