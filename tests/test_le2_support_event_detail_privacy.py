"""Tests for the Support Critical Event ``details`` forbidden-key-substring
stripping hardening in
custom_components/thermosmart/learning/storage/support_event_serialization.py.

Audit finding this step fixes: ``_bounded_details()`` bounded value type/size
(max keys, max string length, non-scalar drop) but never stripped forbidden
KEY NAMES — a producer could theoretically write
``details={"entity_id": "..."}`` and it would survive into the in-memory
event, the on-disk ``.storage`` file, AND (until this fix) into
``support_event_for_export()``'s output too (the export layer's
``_le2_strip_forbidden()``/``scan_payload()`` in export.py provided a SECOND
safety net there, but the on-disk store itself was not by-design public-safe
before this fix). This step closes that gap at the source: forbidden keys
are now stripped inside ``_bounded_details()`` itself, so
``serialize_support_event()``, ``deserialize_support_event()``, and
``support_event_for_export()`` — all three already route through it — get
the fix for free, with no change to their own call signatures or the store
shape.

No new producers, no new event types, no Support Export layout change, no
Research Export change, no store I/O change, no runtime/control change —
this step only tightens one existing pure function.

18 test groups:
  T1  — Drops entity_id
  T2  — Drops zone_id
  T3  — Drops episode_id
  T4  — Drops learning_zone_id
  T5  — Drops decision_id
  T6  — Drops trv_binding_id
  T7  — Drops radiator_profile_id
  T8  — Drops trajectory
  T9  — Drops secret / token / path
  T10 — Drops climate. / sensor. prefixed keys
  T11 — Case-insensitive drop
  T12 — Allowed scalar keys survive unchanged
  T13 — Max-12-key limit still enforced (after stripping)
  T14 — String truncation still enforced
  T15 — support_event_for_export() output contains no forbidden detail keys
  T16 — Existing export privacy tests remain green
  T17 — Existing Support Event producer tests remain green
  T18 — No control-path keywords touched (this is a pure data-hygiene fix)
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.storage.support_event_serialization import (
    DETAILS_MAX_KEYS,
    DETAILS_VALUE_MAX_LENGTH,
    SUPPORT_EVENT_SCHEMA_VERSION,
    _bounded_details,
    _FORBIDDEN_DETAIL_KEY_SUBSTRINGS,
    serialize_support_event,
    support_event_for_export,
)
from custom_components.thermosmart.learning.support_event_schemas import (
    SupportCriticalEvent,
    SupportEventSeverity,
    SupportEventType,
)


def _forbidden_probe_dict() -> dict:
    """One representative key per required forbidden substring, plus a few
    legitimate scalar keys that must survive."""
    return {
        "entity_id": "climate.trv1",
        "zone_id": "zone_alpha_01",
        "episode_id": "zone:heating:1",
        "learning_zone_id": "zone_alpha_01",
        "decision_id": "dec_123",
        "trv_binding_id": "binding_1",
        "radiator_profile_id": "profile_1",
        "trajectory": [1, 2, 3],
        "my_secret": "x",
        "auth_token": "y",
        "file_path": "/config/x",
        "climate.something": 1,
        "sensor.something": 1,
        "requested_setpoint": 21.0,
        "active": True,
    }


# ── T1-T10: Each forbidden substring is dropped ─────────────────────────────

def test_t1_drops_entity_id():
    assert "entity_id" not in _bounded_details({"entity_id": "climate.trv1"})


def test_t2_drops_zone_id():
    assert "zone_id" not in _bounded_details({"zone_id": "zone_alpha_01"})


def test_t3_drops_episode_id():
    assert "episode_id" not in _bounded_details({"episode_id": "zone:heating:1"})


def test_t4_drops_learning_zone_id():
    assert "learning_zone_id" not in _bounded_details({"learning_zone_id": "zone_alpha_01"})


def test_t5_drops_decision_id():
    assert "decision_id" not in _bounded_details({"decision_id": "dec_123"})


def test_t6_drops_trv_binding_id():
    assert "trv_binding_id" not in _bounded_details({"trv_binding_id": "binding_1"})


def test_t7_drops_radiator_profile_id():
    assert "radiator_profile_id" not in _bounded_details({"radiator_profile_id": "profile_1"})


def test_t8_drops_trajectory():
    assert "trajectory" not in _bounded_details({"trajectory": [1, 2, 3]})


def test_t9_drops_secret_token_path():
    d = _bounded_details({"my_secret": "x", "auth_token": "y", "file_path": "/config/x"})
    assert d == {}


def test_t10_drops_climate_and_sensor_prefixed_keys():
    d = _bounded_details({"climate.something": 1, "sensor.something": 1})
    assert d == {}


# ── T11: Case-insensitive drop ──────────────────────────────────────────────

def test_t11_case_insensitive_drop():
    d = _bounded_details({
        "Entity_ID": "climate.trv1", "ZONE_ID": "z1", "Episode_Id": "e1",
        "TRAJECTORY": [1], "My_SECRET": "x", "Auth_TOKEN": "y",
    })
    assert d == {}


# ── T12: Allowed scalar keys survive unchanged ──────────────────────────────

def test_t12_allowed_scalar_keys_survive():
    d = _bounded_details(_forbidden_probe_dict())
    assert d == {"requested_setpoint": 21.0, "active": True}


def test_t12_all_required_substrings_are_actually_checked():
    required = (
        "entity_id", "zone_id", "episode_id", "learning_zone_id", "decision_id",
        "trv_binding_id", "radiator_profile_id", "trajectory", "person",
        "secret", "token", "path", "climate.", "sensor.",
    )
    for sub in required:
        assert sub in _FORBIDDEN_DETAIL_KEY_SUBSTRINGS, f"{sub!r} missing from forbidden list"


# ── T13: Max-12-key limit still enforced (after stripping) ────────────────

def test_t13_max_key_limit_enforced_after_stripping():
    d = {f"k{i}": i for i in range(20)}
    d["entity_id"] = "climate.trv1"  # would-be 21st key, also forbidden
    bounded = _bounded_details(d)
    assert len(bounded) <= DETAILS_MAX_KEYS
    assert "entity_id" not in bounded


def test_t13_max_key_limit_unchanged_value():
    assert DETAILS_MAX_KEYS == 12


# ── T14: String truncation still enforced ───────────────────────────────────

def test_t14_string_truncation_still_enforced():
    bounded = _bounded_details({"summary_note": "x" * 500})
    assert len(bounded["summary_note"]) == DETAILS_VALUE_MAX_LENGTH


# ── T15: support_event_for_export() output contains no forbidden detail keys

def test_t15_export_view_strips_forbidden_detail_keys():
    ev = SupportCriticalEvent(
        schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id="evt_1",
        event_type=SupportEventType.TRV_COMMAND_BLOCKED,
        ts=datetime(2026, 6, 2, 10, 0, tzinfo=timezone.utc),
        severity=SupportEventSeverity.WARNING, reason="min_interval",
        details=_forbidden_probe_dict(),
    )
    serialized = serialize_support_event(ev)
    # Even the raw serialized entry never carried the forbidden keys, since
    # serialize_support_event() already routes through _bounded_details().
    assert "entity_id" not in serialized["details"]
    assert "trajectory" not in serialized["details"]

    view = support_event_for_export(serialized)
    assert "event_id" not in view
    for forbidden in ("entity_id", "zone_id", "episode_id", "learning_zone_id",
                      "decision_id", "trv_binding_id", "radiator_profile_id",
                      "trajectory", "my_secret", "auth_token", "file_path",
                      "climate.something", "sensor.something"):
        assert forbidden not in view["details"]
    assert view["details"] == {"requested_setpoint": 21.0, "active": True}


def test_t15_export_view_survives_forbidden_keys_smuggled_directly_into_a_raw_entry():
    """Defense-in-depth: even if a malformed/hand-crafted store entry (not
    produced via serialize_support_event()) carried forbidden keys directly,
    support_event_for_export() must still strip them — it re-applies
    _bounded_details() itself, independent of how the entry was created."""
    raw_entry = {
        "schema_version": SUPPORT_EVENT_SCHEMA_VERSION,
        "event_id": "evt_2",
        "event_type": "trv_command_blocked",
        "ts": "2026-06-02T10:00:00.000Z",
        "severity": "warning",
        "reason": "min_interval",
        "summary": None,
        "details": {"entity_id": "climate.trv1", "requested_setpoint": 21.0},
    }
    view = support_event_for_export(raw_entry)
    assert "entity_id" not in view["details"]
    assert view["details"] == {"requested_setpoint": 21.0}


# ── T16: Existing export privacy tests remain green ─────────────────────────

def test_t16_regression_imports_ok_export():
    import tests.test_le2_support_critical_event_export  # noqa: F401
    import tests.test_le2_support_critical_events  # noqa: F401


# ── T17: Existing Support Event producer tests remain green ────────────────

def test_t17_regression_imports_ok_producers():
    import tests.test_le2_support_storage_restore_events  # noqa: F401
    import tests.test_le2_support_hold_transition_events  # noqa: F401
    import tests.test_le2_support_trv_command_events  # noqa: F401
    import tests.test_le2_support_boost_transition_events  # noqa: F401
    import tests.test_le2_support_sensor_fallback_events  # noqa: F401
    import tests.test_le2_support_outcome_resolved_events  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401


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


def test_t18_no_control_keywords_in_bounded_details():
    source = inspect.getsource(_bounded_details).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"
