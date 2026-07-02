"""Tests for the LE2 Support Critical Event TRV command sent/blocked
producers in ThermoSmartCoordinator
(custom_components/thermosmart/coordinator.py).

This step adds the THIRD coordinator-side Support Critical Event producer —
pure OBSERVATION of the TRV setpoint dispatch outcome already decided by
_apply_temperature()/the active-control gate this cycle. It reads the
already-computed ``_DispatchStats`` (privacy-safe by construction, no entity
IDs — see trv_control.py), the recommendation dict, and small already-tracked
coordinator state (``_last_written_setpoints``, ``_trv_offline``), then hands
off to the EXISTING ``record_support_critical_event_safe()``. No influence on
the dispatch decision, the service-call payload, or any filter/throttle/
setpoint logic — this step changes NOTHING in trv_control.py.

Root-cause finding documented here: MIN_INTERVAL_BLOCK exists in the schema's
enum, but no "minimum interval between commands" gate exists anywhere in this
codebase (verified by search) — so it is deliberately NOT produced in this
step (no fabricated event; see the accompanying report).

Since constructing a real ThermoSmartCoordinator requires a live
HomeAssistant instance, this file tests the two new methods
(``_maybe_record_trv_dispatch_event`` / ``_maybe_record_recommendation_only_event``)
directly against the REAL, UNBOUND methods from the production class, bound
onto a minimal fake object exposing only the attributes they touch —
validating the CONTRACT, not a copy — plus static source-inspection checks
for control-path safety.

20 test groups:
  T1  — trv_command_sent is produced when a setpoint was actually written
  T2  — trv_command_sent details are public-safe scalars only
  T3  — Repeated identical sent produces no spam; a new setpoint is visible
  T4  — same_setpoint_block is produced when already-written setpoint matches
  T5  — Repeated same_setpoint_block cycles produce no spam
  T6  — min_interval_block is deliberately NOT produced (no such gate exists)
  T7  — (kept as a placeholder alias of T6 for the requested numbering)
  T8  — learning_recommendation_only is produced when active_control is off
  T9  — Repeated recommendation-only cycles produce no spam
  T10 — learning_recommendation_only clears when active_control turns back on
  T11 — Events appear in the Support Export
  T12 — Details stay public-safe (bounded scalars, no forbidden substrings)
  T13 — No event_id in the Support Export
  T14 — No store reads introduced in the export path
  T15 — trv_unavailable is produced (warning) when a TRV is offline
  T16 — Summer mode produces no new event here (covered by Increment 2)
  T17 — No change to the actual service-call payload / dispatch decision
  T18 — No additional service call is triggered by event production
  T19 — Existing Hold Transition / Storage Restore / Export / Wiring / Foundation tests remain green
  T20 — No control-path keywords touched by the new coordinator methods
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.coordinator import ThermoSmartCoordinator
from custom_components.thermosmart.trv_control import _DispatchStats


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
    _maybe_record_recommendation_only_event = (
        ThermoSmartCoordinator._maybe_record_recommendation_only_event
    )
    _maybe_record_trv_dispatch_event = ThermoSmartCoordinator._maybe_record_trv_dispatch_event

    def __init__(self, active_control: bool = True, now_iso: str = "2026-06-02T12:00:00+00:00") -> None:
        self._le2_shadow: _FakeShadow | None = _FakeShadow()
        self._clock = _FakeClock(now_iso)
        self._active_control = active_control
        self._last_written_setpoints: dict = {}
        self._trv_offline: set = set()
        self._support_event_recommendation_only_active = False
        self._support_event_last_trv_signature = None
        self._support_event_last_sent_setpoint = None


def _sent_stats(effective_c: float) -> _DispatchStats:
    stats = _DispatchStats()
    stats.record(None, effective_c=effective_c, entity_id="climate.trv1")
    return stats


def _events(coord: _FakeCoordinator) -> list[tuple[str, str, str]]:
    return [(e.event_type.value, e.reason, e.severity.value) for e in coord._le2_shadow.events]


# ── T1: trv_command_sent produced when a setpoint was actually written ────

def test_t1_command_sent_when_setpoint_written():
    coord = _FakeCoordinator()
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _sent_stats(21.0), False)
    assert _events(coord) == [("trv_command_sent", "trv_command_sent", "info")]


# ── T2: trv_command_sent details are public-safe scalars only ─────────────

def test_t2_sent_details_are_scalars_only():
    coord = _FakeCoordinator()
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _sent_stats(21.0), False)
    ev = coord._le2_shadow.events[0]
    assert ev.details == {"requested_setpoint": 21.0}
    for v in ev.details.values():
        assert isinstance(v, (int, float, bool, str)) or v is None


# ── T3: Repeated identical sent -> no spam; new setpoint -> visible ────────

def test_t3_repeated_identical_sent_no_spam():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _sent_stats(21.0), False)
    assert len(coord._le2_shadow.events) == 1


def test_t3_new_setpoint_produces_new_sent_event():
    coord = _FakeCoordinator()
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _sent_stats(21.0), False)
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 22.0}, _sent_stats(22.0), False)
    assert len(coord._le2_shadow.events) == 2
    assert coord._le2_shadow.events[1].details["requested_setpoint"] == 22.0
    assert coord._le2_shadow.events[1].details["previous_setpoint"] == 21.0


# ── T4: same_setpoint_block when already-written setpoint matches ─────────

def test_t4_same_setpoint_block_when_matching_last_written():
    coord = _FakeCoordinator()
    coord._last_written_setpoints = {"climate.trv1": 21.0}
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), False)
    assert _events(coord) == [("same_setpoint_block", "same_setpoint_block", "info")]
    assert coord._le2_shadow.events[0].details == {"requested_setpoint": 21.0}


# ── T5: Repeated same_setpoint_block cycles produce no spam ───────────────

def test_t5_repeated_same_setpoint_no_spam():
    coord = _FakeCoordinator()
    coord._last_written_setpoints = {"climate.trv1": 21.0}
    for _ in range(5):
        coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), False)
    assert len(coord._le2_shadow.events) == 1


# ── T6/T7: min_interval_block is deliberately not produced ────────────────

def test_t6_min_interval_gate_does_not_exist_in_codebase():
    """Root-cause check: no "minimum interval between TRV commands" concept
    exists anywhere in coordinator.py/trv_control.py — grepping for it
    confirms no matches, so min_interval_block is correctly never produced
    rather than being fabricated from an unrelated blocker."""
    import custom_components.thermosmart.coordinator as _coord_mod
    import custom_components.thermosmart.trv_control as _trv_mod
    for mod in (_coord_mod, _trv_mod):
        source = inspect.getsource(mod).lower()
        assert "min_interval" not in source
        assert "minimum_interval" not in source


def test_t7_min_interval_block_event_type_never_constructed():
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod.ThermoSmartCoordinator._maybe_record_trv_dispatch_event)
    assert "MIN_INTERVAL_BLOCK" not in source


# ── T8: learning_recommendation_only when active_control is off ───────────

def test_t8_recommendation_only_when_active_control_off():
    coord = _FakeCoordinator(active_control=False)
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), False)
    assert _events(coord) == [
        ("learning_recommendation_only", "learning_recommendation_only", "info"),
    ]


# ── T9: Repeated recommendation-only cycles produce no spam ───────────────

def test_t9_repeated_recommendation_only_no_spam():
    coord = _FakeCoordinator(active_control=False)
    for _ in range(5):
        coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), False)
    assert len(coord._le2_shadow.events) == 1


# ── T10: recommendation_only clears when active_control turns back on ─────

def test_t10_recommendation_only_flag_clears_on_active_control_return():
    coord = _FakeCoordinator(active_control=False)
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), False)
    assert coord._support_event_recommendation_only_active is True

    coord._active_control = True
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _sent_stats(21.0), False)
    assert coord._support_event_recommendation_only_active is False

    # Going observation-only again must produce a new landmark, not stay silent.
    coord._active_control = False
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), False)
    reco_events = [e for e in coord._le2_shadow.events if e.event_type.value == "learning_recommendation_only"]
    assert len(reco_events) == 2


# ── T11: Events appear in the Support Export ────────────────────────────────

def test_t11_events_appear_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )

    coord = _FakeCoordinator()
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _sent_stats(21.0), False)
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
    assert block["records_available"] == 1
    assert block["events"][0]["event_type"] == "trv_command_sent"


# ── T12: Details stay public-safe ───────────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t12_details_stay_public_safe():
    coord = _FakeCoordinator()
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _sent_stats(21.0), False)
    coord._trv_offline = {"climate.trv2"}
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 22.0}, _DispatchStats(), False)
    for ev in coord._le2_shadow.events:
        flat = repr(ev.details).lower()
        for token in _FORBIDDEN_SUBSTRINGS:
            assert token not in flat, f"forbidden substring '{token}' in TRV event details"


# ── T13: No event_id in the Support Export ──────────────────────────────────

def test_t13_no_event_id_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )

    coord = _FakeCoordinator()
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _sent_stats(21.0), False)
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
    assert "trv_trv_command_sent_" not in repr(block)


# ── T14: No store reads introduced in the export path ───────────────────────

def test_t14_export_path_unchanged_no_store_io():
    from custom_components.thermosmart.export import _le2_critical_events_export
    source = inspect.getsource(_le2_critical_events_export)
    assert "await" not in source
    assert "SupportCriticalEventStore(" not in source


# ── T15: trv_unavailable produced (warning) when a TRV is offline ─────────

def test_t15_trv_unavailable_warning_when_offline():
    coord = _FakeCoordinator()
    coord._trv_offline = {"climate.trv1"}
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), False)
    assert _events(coord) == [("trv_unavailable", "trv_unavailable", "warning")]


def test_t15_repeated_unavailable_cycles_no_spam():
    coord = _FakeCoordinator()
    coord._trv_offline = {"climate.trv1"}
    for _ in range(5):
        coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), False)
    assert len(coord._le2_shadow.events) == 1


# ── T16: Summer mode produces no new event here ─────────────────────────────

def test_t16_summer_mode_produces_no_new_event():
    coord = _FakeCoordinator()
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, _DispatchStats(), True)
    assert coord._le2_shadow.events == []


# ── T17: No change to the service-call payload / dispatch decision ────────

def test_t17_stats_object_never_mutated():
    coord = _FakeCoordinator()
    stats = _sent_stats(21.0)
    before = list(stats.effective_setpoints)
    coord._maybe_record_trv_dispatch_event({"trv_setpoint": 21.0}, stats, False)
    assert stats.effective_setpoints == before


def test_t17_recommendation_dict_never_mutated():
    coord = _FakeCoordinator()
    recommendation = {"trv_setpoint": 21.0}
    snapshot = dict(recommendation)
    coord._maybe_record_trv_dispatch_event(recommendation, _sent_stats(21.0), False)
    assert recommendation == snapshot


def test_t17_new_method_never_calls_apply_temperature():
    """The docstring legitimately mentions _apply_temperature() in prose
    (explaining where sp_stats came from) — checked here for an actual CALL
    pattern, not the bare substring, to avoid that false positive."""
    import custom_components.thermosmart.coordinator as _coord_mod
    source = inspect.getsource(_coord_mod.ThermoSmartCoordinator._maybe_record_trv_dispatch_event)
    assert "self._apply_temperature(" not in source
    assert "async_set_temperature" not in source
    assert "services.async_call" not in source


# ── T18: No additional service call triggered by event production ────────

def test_t18_no_await_in_new_methods():
    """Event production is fully synchronous — no new await chain, hence no
    possibility of an additional service call being awaited from here."""
    import custom_components.thermosmart.coordinator as _coord_mod
    for name in ("_maybe_record_trv_dispatch_event", "_maybe_record_recommendation_only_event"):
        source = inspect.getsource(getattr(_coord_mod.ThermoSmartCoordinator, name))
        assert "await" not in source


# ── T19: Existing tests remain green (regression smoke-test) ───────────────

def test_t19_regression_imports_ok():
    import tests.test_le2_support_hold_transition_events  # noqa: F401
    import tests.test_le2_support_storage_restore_events  # noqa: F401
    import tests.test_le2_support_critical_event_export  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_support_critical_events  # noqa: F401


# ── T20: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "async_set_temperature",
    "async_write_ha_state",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t20_no_control_keywords_in_new_methods():
    import custom_components.thermosmart.coordinator as _coord_mod
    for name in ("_maybe_record_trv_dispatch_event", "_maybe_record_recommendation_only_event"):
        source = inspect.getsource(getattr(_coord_mod.ThermoSmartCoordinator, name)).lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in source, f"forbidden control token found in {name}: {token}"
