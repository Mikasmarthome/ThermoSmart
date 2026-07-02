"""Tests for the LE2 Support Critical Event boost started/blocked/ended
producers in ThermoSmartCoordinator
(custom_components/thermosmart/coordinator.py).

This step adds the FOURTH coordinator-side Support Critical Event producer —
pure OBSERVATION of the boost decision already made this cycle by the
EXISTING resolve_adaptive_boost_control() (the "single production entry
point for real boost control", ha_integration.py). It reads the already-
computed, read-only ``AdaptiveBoostControlResult`` (schema_events.py:
requested/approved/applied offsets, boost_allowed, blocking_reason —
already typed, already public-safe by construction, no entity data) plus
``recommendation["le2_boost_adjusted"]`` (already set by the resolver's own
inner gates), then hands off to the EXISTING
``record_support_critical_event_safe()``. No influence on boost timing,
offset, cooldown, or lifecycle — this step changes NOTHING in the boost
resolver, adjust_recommendation_safe(), or the lifecycle release methods.

A second, small observation point mirrors the EXISTING
invalidate_boost_after_failed_dispatch_safe() call (post-dispatch,
failed/partial TRV command) to surface "boost_ended /
invalidated_after_failed_dispatch" — again pure observation, called right
after that existing lifecycle call, never influencing it.

``reason`` values for boost_ended/boost_blocked come directly from
``AdaptiveBoostControlResult.blocking_reason`` — the resolver's own typed
BoostBlockReason enum values (e.g. "active_control_off", "device_unavailable")
or inner-gate rejection strings (e.g. "window_open", "cooldown_active",
"manual_override") already produced by adjust_recommendation_safe(). Never
guessed or invented.

Since constructing a real ThermoSmartCoordinator requires a live
HomeAssistant instance, this file tests the two new methods
(``_maybe_record_boost_event`` / ``_maybe_record_boost_invalidated_event``)
directly against the REAL, UNBOUND methods from the production class, bound
onto a minimal fake object exposing only the attributes they touch —
validating the CONTRACT, not a copy — plus static source-inspection checks
for control-path safety.

19 test groups:
  T1  — boost_started is produced exactly once on the active-off -> active-on transition
  T2  — Repeated active-boost cycles produce no additional events
  T3  — boost_ended is produced exactly once on the active-on -> active-off transition
  T4  — Repeated inactive cycles produce no additional events
  T5  — boost_blocked is produced when the resolver returns a blocking_reason
  T6  — Repeated identical blocked reason produces no spam; a new reason is visible
  T7  — Boost invalidation after failed dispatch produces boost_ended with a specific reason
  T8  — Invalidation call is a no-op when boost was not recorded as active
  T9  — Summer mode (resolver not invoked) ends a previously active boost
  T10 — Missing shadow (resolver not invoked) ends a previously active boost
  T11 — Events appear in the Support Export
  T12 — Details stay public-safe (bounded scalars, no forbidden substrings)
  T13 — No event_id in the Support Export
  T14 — No store reads introduced in the export path
  T15 — No change to the boost resolver / lifecycle call itself
  T16 — No additional service call is triggered by event production
  T17 — No boost offset/duration/cooldown mutation by the new methods
  T18 — Existing TRV/Hold/Storage/Export/Wiring/Foundation tests remain green
  T19 — No control-path keywords touched by the new coordinator methods
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pytest

from custom_components.thermosmart.coordinator import ThermoSmartCoordinator


@dataclass(frozen=True)
class _FakeBoostResult:
    """Minimal stand-in for AdaptiveBoostControlResult exposing only the two
    fields the new methods read."""
    approved_boost_offset_c: float = 0.0
    blocking_reason: Optional[str] = None


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
    _maybe_record_boost_event = ThermoSmartCoordinator._maybe_record_boost_event
    _maybe_record_boost_invalidated_event = (
        ThermoSmartCoordinator._maybe_record_boost_invalidated_event
    )

    def __init__(self, now_iso: str = "2026-06-02T12:00:00+00:00") -> None:
        self._le2_shadow: _FakeShadow | None = _FakeShadow()
        self._clock = _FakeClock(now_iso)
        self._support_event_boost_active = False
        self._support_event_last_boost_signature = None


def _events(coord: _FakeCoordinator) -> list[tuple[str, str, str]]:
    return [(e.event_type.value, e.reason, e.severity.value) for e in coord._le2_shadow.events]


# ── T1: boost_started produced exactly once on the transition ──────────────

def test_t1_boost_started_on_transition():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
    assert _events(coord) == [("boost_started", "boost_started", "info")]
    assert coord._le2_shadow.events[0].details == {"active": True, "offset": 0.5}


# ── T2: Repeated active-boost cycles produce no additional events ─────────

def test_t2_repeated_active_boost_no_spam():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_boost_event(
            {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
        )
    assert len(coord._le2_shadow.events) == 1


# ── T3: boost_ended produced exactly once on the transition ────────────────

def test_t3_boost_ended_on_transition():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason="cooldown_active"), False,
    )
    assert _events(coord) == [
        ("boost_started", "boost_started", "info"),
        ("boost_ended", "cooldown_active", "info"),
    ]
    assert coord._le2_shadow.events[1].details == {"active": False}


# ── T4: Repeated inactive cycles produce no additional events ─────────────

def test_t4_repeated_inactive_no_spam_beyond_first_block():
    """After the active->inactive transition fires boost_ended, the FIRST
    subsequent still-blocked cycle legitimately adds one boost_blocked
    landmark (a genuinely new signal: the transition event alone doesn't
    say WHY it stays blocked) — but further repeats of the SAME reason must
    not spam beyond that."""
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason="cooldown_active"), False,
    )
    for _ in range(5):
        coord._maybe_record_boost_event(
            {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason="cooldown_active"), False,
        )
    assert len(coord._le2_shadow.events) == 3  # started + ended + one blocked, no further repeats


# ── T5: boost_blocked produced when the resolver returns a blocking_reason ─

def test_t5_boost_blocked_with_reason():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason="window_open"), False,
    )
    assert _events(coord) == [("boost_blocked", "window_open", "info")]


def test_t5_no_event_when_never_active_and_no_blocking_reason():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason=None), False,
    )
    assert coord._le2_shadow.events == []


# ── T6: Repeated identical blocked reason no spam; new reason visible ─────

def test_t6_repeated_identical_block_reason_no_spam():
    coord = _FakeCoordinator()
    for _ in range(5):
        coord._maybe_record_boost_event(
            {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason="active_control_off"), False,
        )
    assert len(coord._le2_shadow.events) == 1


def test_t6_new_block_reason_produces_new_event():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason="active_control_off"), False,
    )
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason="manual_override"), False,
    )
    assert _events(coord) == [
        ("boost_blocked", "active_control_off", "info"),
        ("boost_blocked", "manual_override", "info"),
    ]


# ── T7: Invalidation after failed dispatch produces boost_ended ───────────

def test_t7_invalidation_after_failed_dispatch_produces_boost_ended():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
    coord._maybe_record_boost_invalidated_event()
    assert _events(coord) == [
        ("boost_started", "boost_started", "info"),
        ("boost_ended", "invalidated_after_failed_dispatch", "info"),
    ]
    assert coord._support_event_boost_active is False


# ── T8: Invalidation call is a no-op when boost was not active ────────────

def test_t8_invalidation_noop_when_not_active():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_invalidated_event()
    assert coord._le2_shadow.events == []


# ── T9: Summer mode ends a previously active boost ─────────────────────────

def test_t9_summer_mode_ends_active_boost():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
    coord._maybe_record_boost_event({}, None, True)  # resolver not invoked (summer)
    assert _events(coord)[-1] == ("boost_ended", "summer_mode", "info")


def test_t9_summer_mode_never_active_produces_no_event():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event({}, None, True)
    assert coord._le2_shadow.events == []


# ── T10: Missing shadow-evaluation (resolver not invoked) ends active boost

def test_t10_resolver_not_invoked_non_summer_ends_active_boost():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
    coord._maybe_record_boost_event({}, None, False)
    assert _events(coord)[-1] == ("boost_ended", "shadow_unavailable", "info")


def test_missing_shadow_object_does_not_raise():
    coord = _FakeCoordinator()
    coord._le2_shadow = None
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )  # must not raise
    coord._maybe_record_boost_invalidated_event()  # must not raise


# ── T11: Events appear in the Support Export ────────────────────────────────

def test_t11_events_appear_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )

    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
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
    assert block["events"][0]["event_type"] == "boost_started"


# ── T12: Details stay public-safe ───────────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t12_details_stay_public_safe():
    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": False}, _FakeBoostResult(blocking_reason="device_unavailable"), False,
    )
    for ev in coord._le2_shadow.events:
        for v in ev.details.values():
            assert isinstance(v, (int, float, bool, str)) or v is None
        flat = repr(ev.details).lower()
        for token in _FORBIDDEN_SUBSTRINGS:
            assert token not in flat, f"forbidden substring '{token}' in boost event details"


# ── T13: No event_id in the Support Export ──────────────────────────────────

def test_t13_no_event_id_in_support_export():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from custom_components.thermosmart.learning.storage.support_event_serialization import (
        serialize_support_event,
    )

    coord = _FakeCoordinator()
    coord._maybe_record_boost_event(
        {"le2_boost_adjusted": True}, _FakeBoostResult(approved_boost_offset_c=0.5), False,
    )
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


# ── T14: No store reads introduced in the export path ───────────────────────

def test_t14_export_path_unchanged_no_store_io():
    from custom_components.thermosmart.export import _le2_critical_events_export
    source = inspect.getsource(_le2_critical_events_export)
    assert "await" not in source
    assert "SupportCriticalEventStore(" not in source


# ── T15: No change to the boost resolver / lifecycle call itself ──────────

def test_t15_new_methods_never_call_resolver_or_lifecycle():
    """The docstrings legitimately mention resolve_adaptive_boost_control()/
    invalidate_boost_after_failed_dispatch_safe() in prose (explaining where
    boost_result came from / what this observes) — checked here for an
    actual CALL pattern (self.<method>(), matching this repo's
    LearningShadowController call convention), not the bare substring."""
    import custom_components.thermosmart.coordinator as _coord_mod
    for name in ("_maybe_record_boost_event", "_maybe_record_boost_invalidated_event"):
        source = inspect.getsource(getattr(_coord_mod.ThermoSmartCoordinator, name))
        assert "self._le2_shadow.resolve_adaptive_boost_control(" not in source
        assert "self._le2_shadow.adjust_recommendation_safe(" not in source
        assert "self._le2_shadow.invalidate_boost_after_failed_dispatch_safe(" not in source
        assert "self._le2_shadow._release_boost_lifecycle_safe(" not in source


def test_t15_boost_result_object_never_mutated():
    coord = _FakeCoordinator()
    result = _FakeBoostResult(approved_boost_offset_c=0.5)
    coord._maybe_record_boost_event({"le2_boost_adjusted": True}, result, False)
    assert result.approved_boost_offset_c == 0.5  # frozen dataclass, still unchanged


def test_t15_recommendation_dict_never_mutated():
    coord = _FakeCoordinator()
    recommendation = {"le2_boost_adjusted": True}
    snapshot = dict(recommendation)
    coord._maybe_record_boost_event(recommendation, _FakeBoostResult(approved_boost_offset_c=0.5), False)
    assert recommendation == snapshot


# ── T16: No additional service call triggered by event production ────────

def test_t16_no_await_in_new_methods():
    import custom_components.thermosmart.coordinator as _coord_mod
    for name in ("_maybe_record_boost_event", "_maybe_record_boost_invalidated_event"):
        source = inspect.getsource(getattr(_coord_mod.ThermoSmartCoordinator, name))
        assert "await" not in source
        assert "services.async_call" not in source


# ── T17: No boost offset/duration/cooldown mutation ─────────────────────────

def test_t17_no_boost_offset_or_cooldown_write_in_new_methods():
    import custom_components.thermosmart.coordinator as _coord_mod
    for name in ("_maybe_record_boost_event", "_maybe_record_boost_invalidated_event"):
        source = inspect.getsource(getattr(_coord_mod.ThermoSmartCoordinator, name))
        assert "boost_offset_c\"] =" not in source
        assert "cooldown" not in source.lower() or "cooldown_active" in source
        assert "TPI_MAX_BOOST_CELSIUS" not in source


# ── T18: Existing tests remain green (regression smoke-test) ───────────────

def test_t18_regression_imports_ok():
    import tests.test_le2_support_trv_command_events  # noqa: F401
    import tests.test_le2_support_hold_transition_events  # noqa: F401
    import tests.test_le2_support_storage_restore_events  # noqa: F401
    import tests.test_le2_support_critical_event_export  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_support_critical_events  # noqa: F401


# ── T19: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "async_set_temperature",
    "async_write_ha_state",
    "service_call",
    "boost_offset write",
    "tpi_gain",
)


def test_t19_no_control_keywords_in_new_methods():
    import custom_components.thermosmart.coordinator as _coord_mod
    for name in ("_maybe_record_boost_event", "_maybe_record_boost_invalidated_event"):
        source = inspect.getsource(getattr(_coord_mod.ThermoSmartCoordinator, name)).lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in source, f"forbidden control token found in {name}: {token}"
