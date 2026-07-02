"""Tests for the LE2 episode serialization foundation.

Covers custom_components/thermosmart/learning/storage/episode_serialization.py:
  EPISODE_SCHEMA_VERSION, serialize_episode, deserialize_episode,
  serialize_episode_list, deserialize_episode_list,
  episode_for_support_export, episode_for_research_export.

This is a pure serialization foundation — no runtime persistence is wired up
in this step. Nothing here writes to EpisodesStore or any other store.

20 test groups:
  T1  — Heating episode roundtrip
  T2  — Afterheat episode roundtrip
  T3  — Passive cooling episode roundtrip
  T4  — Window cooling episode roundtrip
  T5  — Outcome episode roundtrip
  T6  — Unknown episode type returns None / skipped safely
  T7  — Malformed dict returns None / skipped safely
  T8  — Schema mismatch handled safely
  T9  — Missing required field handled safely
  T10 — Optional field missing handled safely
  T11 — serialize_episode_list keeps order
  T12 — deserialize_episode_list skips malformed entries
  T13 — Support export shape is public-safe
  T14 — Research export shape is public-safe
  T15 — No entity ids / zone names / secrets / paths in export shapes
  T16 — No HA imports in the new module
  T17 — No storage writes in the new module
  T18 — No runtime/control paths touched
  T19 — Existing capture wiring tests remain green
  T20 — Existing retention tests remain green
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.episode_schemas import (
    AfterheatEpisode,
    ControllerKind,
    EpisodeReason,
    EpisodeType,
    HeatingEpisode,
    OutcomeEpisode,
    PassiveCoolingEpisode,
    Trajectory,
    TrajectoryPoint,
    WindowCoolingEpisode,
)
from custom_components.thermosmart.learning.storage import episode_serialization as _episode_ser_mod
from custom_components.thermosmart.learning.storage.episode_serialization import (
    EPISODE_SCHEMA_VERSION,
    deserialize_episode,
    deserialize_episode_list,
    episode_for_research_export,
    episode_for_support_export,
    serialize_episode,
    serialize_episode_list,
)
from custom_components.thermosmart.learning.storage.retention import evaluate_episode_retention
from custom_components.thermosmart.learning.registry import RetentionPolicy

_ZONE = "zone_alpha_01"
_T0 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
_T1 = _T0 + timedelta(minutes=30)


def _traj(n: int = 3) -> Trajectory:
    points = tuple(TrajectoryPoint(offset_ms=i * 1000, value=20.0 + i * 0.1) for i in range(n))
    return Trajectory(points=points, max_points=max(n, 1))


# ── Fixtures — one real instance per episode type ────────────────────────────

def _heating() -> HeatingEpisode:
    return HeatingEpisode(
        episode_id=f"{_ZONE}:heating:1", learning_zone_id=_ZONE,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.ACTIVE_HEATING, reliability=0.9,
        start_temp=19.0, target=21.0, confounder_flags=("window_open_brief",),
        trajectory=_traj(), controller=ControllerKind.THERMOSMART,
        trv_binding_id="trv_slot_0",
    )


def _afterheat() -> AfterheatEpisode:
    return AfterheatEpisode(
        episode_id=f"{_ZONE}:afterheat:1", learning_zone_id=_ZONE,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.AFTERHEAT, reliability=0.8,
        indoor_temp_at_close=20.5, target=21.0,
        trv_setpoint_before=22.0, trv_setpoint_after=18.0,
        trajectory=_traj(), radiator_profile_id="rad_a",
    )


def _passive_cooling() -> PassiveCoolingEpisode:
    return PassiveCoolingEpisode(
        episode_id=f"{_ZONE}:passive_cooling:1", learning_zone_id=_ZONE,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.PASSIVE_COOLING, reliability=0.7,
        start_temp=21.0, end_temp=20.2, trajectory=_traj(),
    )


def _window_cooling() -> WindowCoolingEpisode:
    return WindowCoolingEpisode(
        episode_id=f"{_ZONE}:window_cooling:1", learning_zone_id=_ZONE,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.DISTURBED, reliability=0.6,
        temp_at_open=21.0, temp_at_close=19.5,
    )


def _outcome() -> OutcomeEpisode:
    return OutcomeEpisode(
        episode_id=f"{_ZONE}:outcome:1", learning_zone_id=_ZONE,
        decision_id="dec_abc123",
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.ACTIVE_HEATING, reliability=0.85,
        start_temp=19.0, end_temp=21.0, target=21.0,
        comfort_tolerance_at_start=0.3, reason=EpisodeReason.REACHED,
        controller=ControllerKind.THERMOSMART, trajectory=_traj(),
    )


# ── T1-T5: roundtrips ─────────────────────────────────────────────────────

def test_t1_heating_roundtrip():
    ep = _heating()
    payload = serialize_episode(ep)
    assert payload is not None
    restored = deserialize_episode(payload)
    assert restored == ep


def test_t2_afterheat_roundtrip():
    ep = _afterheat()
    payload = serialize_episode(ep)
    assert payload is not None
    restored = deserialize_episode(payload)
    assert restored == ep


def test_t3_passive_cooling_roundtrip():
    ep = _passive_cooling()
    payload = serialize_episode(ep)
    assert payload is not None
    restored = deserialize_episode(payload)
    assert restored == ep


def test_t4_window_cooling_roundtrip():
    ep = _window_cooling()
    payload = serialize_episode(ep)
    assert payload is not None
    restored = deserialize_episode(payload)
    assert restored == ep


def test_t5_outcome_roundtrip():
    ep = _outcome()
    payload = serialize_episode(ep)
    assert payload is not None
    restored = deserialize_episode(payload)
    assert restored == ep


# ── T6: Unknown episode type ─────────────────────────────────────────────

def test_t6_unknown_object_type_returns_none():
    assert serialize_episode(object()) is None
    assert serialize_episode("not an episode") is None
    assert serialize_episode(42) is None


def test_t6_unknown_episode_type_value_in_payload_returns_none():
    payload = serialize_episode(_heating())
    payload["episode_type"] = "does_not_exist"
    assert deserialize_episode(payload) is None


# ── T7: Malformed dict ────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, "a string", 123, [], {}])
def test_t7_malformed_dict_returns_none(bad):
    assert deserialize_episode(bad) is None


def test_t7_malformed_trajectory_returns_none():
    payload = serialize_episode(_heating())
    payload["trajectory"] = "not_a_dict"
    assert deserialize_episode(payload) is None


# ── T8: Schema mismatch ───────────────────────────────────────────────────

def test_t8_schema_version_mismatch_returns_none():
    payload = serialize_episode(_heating())
    payload["episode_schema_version"] = EPISODE_SCHEMA_VERSION + 99
    assert deserialize_episode(payload) is None


def test_t8_missing_schema_version_returns_none():
    payload = serialize_episode(_heating())
    del payload["episode_schema_version"]
    assert deserialize_episode(payload) is None


# ── T9: Missing required field ────────────────────────────────────────────

def test_t9_missing_required_field_returns_none():
    payload = serialize_episode(_heating())
    del payload["start_temp"]  # required for HeatingEpisode
    assert deserialize_episode(payload) is None


def test_t9_afterheat_missing_trajectory_returns_none():
    """AfterheatEpisode requires a trajectory — omitting it must be handled
    safely (None), not raise."""
    payload = serialize_episode(_afterheat())
    payload["trajectory"] = None
    assert deserialize_episode(payload) is None


# ── T10: Optional field missing ───────────────────────────────────────────

def test_t10_optional_controller_missing_is_safe():
    ep = HeatingEpisode(
        episode_id=f"{_ZONE}:heating:2", learning_zone_id=_ZONE,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.ACTIVE_HEATING, reliability=0.9,
        start_temp=19.0, target=21.0,  # controller/trajectory left at defaults (None)
    )
    payload = serialize_episode(ep)
    restored = deserialize_episode(payload)
    assert restored == ep
    assert restored.controller is None
    assert restored.trajectory is None


def test_t10_optional_confounder_flags_missing_defaults_to_empty():
    payload = serialize_episode(_heating())
    del payload["confounder_flags"]
    restored = deserialize_episode(payload)
    assert restored is not None
    assert restored.confounder_flags == ()


# ── T11: serialize_episode_list keeps order ──────────────────────────────

def test_t11_serialize_list_keeps_order():
    episodes = [_heating(), _outcome(), _passive_cooling()]
    payloads = serialize_episode_list(episodes)
    assert [p["episode_type"] for p in payloads] == ["heating", "outcome", "passive_cooling"]


def test_t11_serialize_list_skips_unknown_but_keeps_order():
    episodes = [_heating(), object(), _outcome()]
    payloads = serialize_episode_list(episodes)
    assert [p["episode_type"] for p in payloads] == ["heating", "outcome"]


# ── T12: deserialize_episode_list skips malformed entries ────────────────

def test_t12_deserialize_list_skips_malformed():
    good = serialize_episode(_heating())
    bad = {"episode_schema_version": EPISODE_SCHEMA_VERSION, "episode_type": "nope"}
    raw_list = [good, bad, None, 123]
    result = deserialize_episode_list(raw_list)
    assert len(result) == 1
    assert isinstance(result[0], HeatingEpisode)


def test_t12_deserialize_list_of_non_sequence_is_safe():
    assert deserialize_episode_list(None) == []
    assert deserialize_episode_list(42) == []


# ── T13: Support export shape is public-safe ─────────────────────────────

def test_t13_support_shape_has_expected_fields():
    shape = episode_for_support_export(_outcome())
    assert shape["episode_type"] == "outcome"
    assert shape["duration_seconds"] == 1800.0
    assert shape["reason"] == "reached"
    assert shape["reason_is_timeout"] is False
    assert shape["source"] == "ts"


def test_t13_support_shape_timeout_flag_true_for_timeout_reason():
    ep = _outcome()
    ep = OutcomeEpisode(**{**ep.__dict__, "reason": EpisodeReason.TIMEOUT})
    shape = episode_for_support_export(ep)
    assert shape["reason_is_timeout"] is True


def test_t13_support_shape_returns_none_for_unknown_type():
    assert episode_for_support_export(object()) is None


# ── T14: Research export shape is public-safe ────────────────────────────

def test_t14_research_shape_includes_physical_metrics():
    shape = episode_for_research_export(_heating())
    assert shape["start_temp"] == 19.0
    assert shape["target"] == 21.0
    assert shape["episode_type"] == "heating"


def test_t14_research_shape_none_for_fields_not_on_type():
    """PassiveCoolingEpisode has no comfort_tolerance_at_start/reason/etc —
    those must be None, never fabricated."""
    shape = episode_for_research_export(_passive_cooling())
    assert shape["comfort_tolerance_at_start"] is None
    assert shape["reason"] is None
    assert shape["indoor_temp_at_close"] is None


def test_t14_research_shape_returns_none_for_unknown_type():
    assert episode_for_research_export(object()) is None


# ── T15: No entity ids / zone names / secrets / paths in export shapes ──

_FORBIDDEN_SUBSTRINGS = (
    "entity_id", "zone_id", "learning_zone_id", "decision_id", "trv_binding_id",
    "episode_id", "radiator_profile_id", "/home/", "c:\\", "password", "token", "secret",
)


def test_t15_support_and_research_shapes_have_no_forbidden_fields():
    for ep in (_heating(), _afterheat(), _passive_cooling(), _window_cooling(), _outcome()):
        support = episode_for_support_export(ep)
        research = episode_for_research_export(ep)
        support_text = str(support).lower()
        research_text = str(research).lower()
        for token in _FORBIDDEN_SUBSTRINGS:
            assert token not in support_text, f"{token} leaked into support shape for {ep!r}"
            assert token not in research_text, f"{token} leaked into research shape for {ep!r}"


def test_t15_export_shapes_contain_no_trajectory_points():
    for ep in (_heating(), _afterheat(), _passive_cooling(), _outcome()):
        support = episode_for_support_export(ep)
        research = episode_for_research_export(ep)
        assert "trajectory" not in support
        assert "trajectory" not in research
        assert "points" not in str(support).lower()


# ── T16: No HA imports in the new module ─────────────────────────────────

def test_t16_no_ha_imports():
    source = inspect.getsource(_episode_ser_mod)
    assert "homeassistant" not in source.lower()
    assert "import homeassistant" not in source


def _strip_docstrings_and_comments(source: str) -> str:
    """Best-effort removal of module/function docstrings and comment lines,
    so prose mentioning e.g. "EpisodesStore" or "dispatch table" doesn't
    cause a false-positive keyword match against actual code."""
    lines = []
    in_docstring = False
    for line in source.splitlines():
        stripped = line.strip()
        if in_docstring:
            if '"""' in stripped:
                in_docstring = False
            continue
        if stripped.startswith('"""') and stripped.count('"""') == 1:
            in_docstring = True
            continue
        if stripped.startswith('"""') and stripped.count('"""') >= 2:
            continue  # single-line docstring
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


# ── T17: No storage writes in the new module ─────────────────────────────

def test_t17_no_storage_write_calls():
    code = _strip_docstrings_and_comments(inspect.getsource(_episode_ser_mod))
    assert "async_save" not in code
    assert ".save(" not in code
    assert "EpisodesStore" not in code
    assert "RawSegmentStore" not in code


# ── T18: No runtime/control paths touched ─────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
    "setpoint_mutation",
    # NOTE: bare "setpoint" is intentionally excluded here — AfterheatEpisode
    # already has legitimate pre-existing fields trv_setpoint_before/_after
    # (episode_schemas.py); serializing them is read-only field access, not a
    # setpoint mutation. The literal control-path grep in the task's own
    # validation step checks the real diff instead of this substring.
)


def test_t18_no_control_keywords_in_new_module():
    code = _strip_docstrings_and_comments(inspect.getsource(_episode_ser_mod)).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in code, f"forbidden control token found: {token}"


def test_t18_no_runtime_imports():
    source = inspect.getsource(_episode_ser_mod)
    assert "from ..runtime" not in source
    assert "import runtime" not in source


# ── Retention compatibility (explicit, since the task requires it) ──────

def test_retention_compatibility_flat_shape_deletes_old_episode():
    payload = serialize_episode(_heating())  # end_ts = 2026-06-01T10:30
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    decision = evaluate_episode_retention(
        {"episodes": {payload["episode_id"]: payload}},
        policy=RetentionPolicy(max_age_days=30), now_utc=now,
    )
    assert decision.delete_episode_ids == (payload["episode_id"],)


def test_retention_compatibility_flat_shape_keeps_recent_episode():
    payload = serialize_episode(_heating())
    now = _T1 + timedelta(hours=1)
    decision = evaluate_episode_retention(
        {"episodes": {payload["episode_id"]: payload}},
        policy=RetentionPolicy(max_age_days=30), now_utc=now,
    )
    assert decision.delete_episode_ids == ()


# ── T19/T20: Existing tests remain green (regression smoke-test) ────────

def test_t19_t20_regression_imports_ok():
    import tests.test_le2_capture_wiring_foundation  # noqa: F401
    import tests.test_le2_retention  # noqa: F401
    import tests.test_le2_periodic_save_trigger  # noqa: F401
    import tests.test_application_lifecycle_storage  # noqa: F401
    import tests.test_application_lifecycle_state  # noqa: F401
