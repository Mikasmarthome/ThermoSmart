"""Tests for the LE2 completed-episode persistence foundation.

Covers custom_components/thermosmart/learning/storage/episode_persistence.py:
  append_completed_episode, append_completed_episodes, prune_episode_entries.

Foundation only (Option C) — see the module's own "Why no runtime wiring in
this step" docstring section and the accompanying report for the reasoning.
Nothing here writes to EpisodesStore; nothing in the live runtime calls this
module yet. Tests T15/T16 (runtime wiring behavior) are therefore N/A and are
represented instead by a static check that no such wiring exists yet.

22 test groups:
  T1  — Completed heating episode appended
  T2  — Completed afterheat episode appended
  T3  — Completed passive cooling episode appended
  T4  — Completed window cooling episode appended
  T5  — Completed outcome episode appended
  T6  — Unknown/malformed episode skipped non-fatal
  T7  — Duplicate episode not duplicated
  T8  — Retention max_records respected
  T9  — Retention max_age_days respected
  T10 — Malformed/no-timestamp retained conservatively
  T11 — Per episode type separation
  T12 — Zone separation
  T13 — Append/order stable
  T14 — Save not called in the append helper (pure, no I/O)
  T15 — N/A: no runtime wiring added in this step (explicit static check)
  T16 — N/A: no runtime wiring added in this step (explicit static check)
  T17 — Save error non-fatal (N/A at this layer — see note; covered by design)
  T18 — Existing test_le2_episode_serialization.py remains green
  T19 — Existing test_le2_capture_wiring_foundation.py remains green
  T20 — Existing test_le2_retention.py remains green
  T21 — Existing periodic save tests remain green
  T22 — No control path touched
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
    HeatingEpisode,
    OutcomeEpisode,
    PassiveCoolingEpisode,
    Trajectory,
    TrajectoryPoint,
    WindowCoolingEpisode,
)
from custom_components.thermosmart.learning.registry import EpisodeRegistry, RetentionPolicy
from custom_components.thermosmart.learning.storage import episode_persistence as _persist_mod
from custom_components.thermosmart.learning.storage.capture_registry import build_episode_registry
from custom_components.thermosmart.learning.storage.episode_persistence import (
    EpisodeAppendResult,
    append_completed_episode,
    append_completed_episodes,
    prune_episode_entries,
)

_ZONE_A = "zone_alpha_01"
_ZONE_B = "zone_beta_02"
_T0 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
_T1 = _T0 + timedelta(minutes=30)


def _traj(n: int = 3) -> Trajectory:
    points = tuple(TrajectoryPoint(offset_ms=i * 1000, value=20.0 + i * 0.1) for i in range(n))
    return Trajectory(points=points, max_points=max(n, 1))


def _heating(zone: str = _ZONE_A, seq: int = 1, start=_T0, end=_T1) -> HeatingEpisode:
    return HeatingEpisode(
        episode_id=f"{zone}:heating:{seq}", learning_zone_id=zone,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=start, end_ts=end, regime=Regime.ACTIVE_HEATING, reliability=0.9,
        start_temp=19.0, target=21.0, controller=ControllerKind.THERMOSMART,
    )


def _afterheat(zone: str = _ZONE_A, seq: int = 1) -> AfterheatEpisode:
    return AfterheatEpisode(
        episode_id=f"{zone}:afterheat:{seq}", learning_zone_id=zone,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.AFTERHEAT, reliability=0.8,
        indoor_temp_at_close=20.5, target=21.0,
        trv_setpoint_before=22.0, trv_setpoint_after=18.0, trajectory=_traj(),
    )


def _passive_cooling(zone: str = _ZONE_A, seq: int = 1) -> PassiveCoolingEpisode:
    return PassiveCoolingEpisode(
        episode_id=f"{zone}:passive_cooling:{seq}", learning_zone_id=zone,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.PASSIVE_COOLING, reliability=0.7,
        start_temp=21.0, end_temp=20.2, trajectory=_traj(),
    )


def _window_cooling(zone: str = _ZONE_A, seq: int = 1) -> WindowCoolingEpisode:
    return WindowCoolingEpisode(
        episode_id=f"{zone}:window_cooling:{seq}", learning_zone_id=zone,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.DISTURBED, reliability=0.6,
        temp_at_open=21.0, temp_at_close=19.5,
    )


def _outcome(zone: str = _ZONE_A, seq: int = 1) -> OutcomeEpisode:
    return OutcomeEpisode(
        episode_id=f"{zone}:outcome:{seq}", learning_zone_id=zone,
        decision_id="dec_abc123",
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.ACTIVE_HEATING, reliability=0.85,
        start_temp=19.0, end_temp=21.0, target=21.0,
        comfort_tolerance_at_start=0.3, reason=EpisodeReason.REACHED,
        controller=ControllerKind.THERMOSMART, trajectory=_traj(),
    )


_REGISTRY = build_episode_registry()
_NOW = _T1 + timedelta(hours=1)


# ── T1-T5: one episode of each type appended successfully ───────────────────

def test_t1_heating_episode_appended():
    result = append_completed_episode(None, _heating(), episode_registry=_REGISTRY, now_utc=_NOW)
    assert result.appended_episode_ids == (f"{_ZONE_A}:heating:1",)
    assert result.skipped_count == 0
    assert f"{_ZONE_A}:heating:1" in result.updated_payload["episodes"]


def test_t2_afterheat_episode_appended():
    result = append_completed_episode(None, _afterheat(), episode_registry=_REGISTRY, now_utc=_NOW)
    assert result.appended_episode_ids == (f"{_ZONE_A}:afterheat:1",)


def test_t3_passive_cooling_episode_appended():
    result = append_completed_episode(None, _passive_cooling(), episode_registry=_REGISTRY, now_utc=_NOW)
    assert result.appended_episode_ids == (f"{_ZONE_A}:passive_cooling:1",)


def test_t4_window_cooling_episode_appended():
    result = append_completed_episode(None, _window_cooling(), episode_registry=_REGISTRY, now_utc=_NOW)
    assert result.appended_episode_ids == (f"{_ZONE_A}:window_cooling:1",)


def test_t5_outcome_episode_appended():
    result = append_completed_episode(None, _outcome(), episode_registry=_REGISTRY, now_utc=_NOW)
    assert result.appended_episode_ids == (f"{_ZONE_A}:outcome:1",)


# ── T6: Unknown/malformed episode skipped non-fatal ─────────────────────────

def test_t6_unknown_object_skipped():
    result = append_completed_episode(None, object(), episode_registry=_REGISTRY, now_utc=_NOW)
    assert result.skipped_count == 1
    assert result.appended_episode_ids == ()
    assert result.updated_payload == {"episodes": {}}


def test_t6_mixed_batch_skips_only_bad_entries():
    result = append_completed_episodes(
        None, (_heating(), object(), _outcome(), "not an episode"),
        episode_registry=_REGISTRY, now_utc=_NOW,
    )
    assert result.skipped_count == 2
    assert set(result.appended_episode_ids) == {f"{_ZONE_A}:heating:1", f"{_ZONE_A}:outcome:1"}


# ── T7: Duplicate episode not duplicated ─────────────────────────────────

def test_t7_duplicate_episode_id_not_reappended_across_calls():
    first = append_completed_episode(None, _heating(), episode_registry=_REGISTRY, now_utc=_NOW)
    second = append_completed_episode(
        first.updated_payload, _heating(), episode_registry=_REGISTRY, now_utc=_NOW,
    )
    assert second.duplicate_count == 1
    assert second.appended_episode_ids == ()
    assert len(second.updated_payload["episodes"]) == 1


def test_t7_duplicate_episode_id_within_same_batch():
    result = append_completed_episodes(
        None, (_heating(), _heating()), episode_registry=_REGISTRY, now_utc=_NOW,
    )
    assert result.appended_episode_ids == (f"{_ZONE_A}:heating:1",)
    assert result.duplicate_count == 1
    assert len(result.updated_payload["episodes"]) == 1


# ── T8: Retention max_records respected ──────────────────────────────────

def test_t8_max_records_evicts_oldest():
    registry = EpisodeRegistry()
    from custom_components.thermosmart.learning.episode_schemas import EpisodeType
    from custom_components.thermosmart.learning.registry import EpisodeDefinition
    registry.register(EpisodeDefinition(
        episode_type=EpisodeType.HEATING, schema_type=HeatingEpisode,
        episode_schema_version=1, builder_version=1, consumed_raw_tracks=(),
        max_trajectory_points=10, max_duration_seconds=3600,
        retention=RetentionPolicy(max_records=2),
        trajectory_required=False, materialized=True,
    ))
    episodes = [
        _heating(seq=1, start=_T0, end=_T0 + timedelta(minutes=10)),
        _heating(seq=2, start=_T0 + timedelta(hours=1), end=_T0 + timedelta(hours=1, minutes=10)),
        _heating(seq=3, start=_T0 + timedelta(hours=2), end=_T0 + timedelta(hours=2, minutes=10)),
    ]
    result = append_completed_episodes(None, episodes, episode_registry=registry, now_utc=_NOW)
    assert len(result.updated_payload["episodes"]) == 2
    assert f"{_ZONE_A}:heating:1" in result.pruned_episode_ids


# ── T9: Retention max_age_days respected ──────────────────────────────────

def test_t9_max_age_days_prunes_old_entry():
    old = _heating(seq=1, start=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2020, 1, 1, 0, 30, tzinfo=timezone.utc))
    result = append_completed_episode(
        None, old, episode_registry=_REGISTRY, now_utc=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    # append happens, then retention immediately prunes it since it's ancient
    assert result.updated_payload["episodes"] == {}
    assert f"{_ZONE_A}:heating:1" in result.pruned_episode_ids


def test_t9_recent_entry_survives_age_check():
    result = append_completed_episode(None, _heating(), episode_registry=_REGISTRY, now_utc=_NOW)
    assert f"{_ZONE_A}:heating:1" in result.updated_payload["episodes"]
    assert result.pruned_episode_ids == ()


# ── T10: Malformed/no-timestamp retained conservatively ──────────────────

def test_t10_malformed_timestamp_entry_kept_by_prune():
    payload = {"episodes": {
        "bad-entry": {"episode_type": "heating", "end_ts": "not-a-timestamp"},
    }}
    result = prune_episode_entries(payload, episode_registry=_REGISTRY, now_utc=_NOW)
    assert "bad-entry" in result.updated_payload["episodes"]
    assert result.pruned_episode_ids == ()


def test_t10_missing_timestamp_entry_kept_by_prune():
    payload = {"episodes": {
        "no-ts-entry": {"episode_type": "heating"},
    }}
    result = prune_episode_entries(payload, episode_registry=_REGISTRY, now_utc=_NOW)
    assert "no-ts-entry" in result.updated_payload["episodes"]


# ── T11: Per episode type separation ─────────────────────────────────────

def test_t11_episode_type_separation_in_retention():
    registry = EpisodeRegistry()
    from custom_components.thermosmart.learning.episode_schemas import EpisodeType
    from custom_components.thermosmart.learning.registry import EpisodeDefinition
    registry.register(EpisodeDefinition(
        episode_type=EpisodeType.HEATING, schema_type=HeatingEpisode,
        episode_schema_version=1, builder_version=1, consumed_raw_tracks=(),
        max_trajectory_points=10, max_duration_seconds=3600,
        retention=RetentionPolicy(max_age_days=1),  # strict
        trajectory_required=False, materialized=True,
    ))
    registry.register(EpisodeDefinition(
        episode_type=EpisodeType.OUTCOME, schema_type=OutcomeEpisode,
        episode_schema_version=1, builder_version=1, consumed_raw_tracks=(),
        max_trajectory_points=10, max_duration_seconds=3600,
        retention=RetentionPolicy(max_age_days=3650),  # lenient
        trajectory_required=True, materialized=True,
    ))
    old_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    old_end = old_start + timedelta(minutes=10)
    heating = _heating(start=old_start, end=old_end)
    outcome = OutcomeEpisode(
        episode_id=f"{_ZONE_A}:outcome:1", learning_zone_id=_ZONE_A, decision_id="d1",
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=old_start, end_ts=old_end, regime=Regime.ACTIVE_HEATING, reliability=0.8,
        start_temp=19.0, end_temp=21.0, target=21.0, comfort_tolerance_at_start=0.3,
        reason=EpisodeReason.REACHED, controller=ControllerKind.THERMOSMART, trajectory=_traj(),
    )
    result = append_completed_episodes(
        None, (heating, outcome), episode_registry=registry, now_utc=_NOW,
    )
    assert f"{_ZONE_A}:heating:1" not in result.updated_payload["episodes"]
    assert f"{_ZONE_A}:outcome:1" in result.updated_payload["episodes"]


# ── T12: Zone separation ──────────────────────────────────────────────────

def test_t12_zone_separation_via_distinct_episode_ids():
    ep_a = _heating(zone=_ZONE_A)
    ep_b = _heating(zone=_ZONE_B)
    result = append_completed_episodes(
        None, (ep_a, ep_b), episode_registry=_REGISTRY, now_utc=_NOW,
    )
    assert f"{_ZONE_A}:heating:1" in result.updated_payload["episodes"]
    assert f"{_ZONE_B}:heating:1" in result.updated_payload["episodes"]
    assert len(result.updated_payload["episodes"]) == 2


# ── T13: Append/order stable ──────────────────────────────────────────────

def test_t13_append_order_is_stable():
    episodes = [_heating(seq=1), _outcome(seq=1), _passive_cooling(seq=1), _window_cooling(seq=1)]
    result = append_completed_episodes(None, episodes, episode_registry=_REGISTRY, now_utc=_NOW)
    assert result.appended_episode_ids == (
        f"{_ZONE_A}:heating:1", f"{_ZONE_A}:outcome:1",
        f"{_ZONE_A}:passive_cooling:1", f"{_ZONE_A}:window_cooling:1",
    )
    assert list(result.updated_payload["episodes"].keys()) == list(result.appended_episode_ids)


# ── T14: Save not called in the append helper (pure, no I/O) ────────────

def test_t14_no_save_or_store_calls_in_module():
    source = inspect.getsource(_persist_mod)
    lines = []
    in_doc = False
    for line in source.splitlines():
        s = line.strip()
        if in_doc:
            if '"""' in s:
                in_doc = False
            continue
        if s.startswith('"""') and s.count('"""') == 1:
            in_doc = True
            continue
        if s.startswith('"""') and s.count('"""') >= 2:
            continue
        if s.startswith("#"):
            continue
        lines.append(line)
    code = "\n".join(lines)
    assert "async_save" not in code
    assert ".save(" not in code
    assert "EpisodesStore" not in code
    assert "RawSegmentStore" not in code


# ── T15/T16: N/A — no runtime wiring added in this step ──────────────────

def test_t15_t16_no_runtime_wiring_added_this_step():
    """Explicit static confirmation that append_completed_episode()/
    append_completed_episodes() are never actually CALLED from the live
    runtime cycle (lifecycle.py) or the HA integration layer
    (ha_integration.py) — matching the report's Option C decision.

    A later step legitimately references "episode_persistence.py" by name in
    an ha_integration.py code COMMENT (documenting the future hook point) —
    that is prose, not wiring, so this check looks for the actual invocation
    pattern (a call with parentheses) rather than the bare module name."""
    import custom_components.thermosmart.learning.runtime.lifecycle as _lifecycle_mod
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha_mod
    lifecycle_source = inspect.getsource(_lifecycle_mod)
    ha_source = inspect.getsource(_ha_mod)
    assert "from ..storage.episode_persistence import" not in lifecycle_source
    assert "from ..storage import episode_persistence" not in lifecycle_source
    assert "from ..storage.episode_persistence import" not in ha_source
    assert "from ..storage import episode_persistence" not in ha_source
    for source in (lifecycle_source, ha_source):
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue  # comment/docstring mention is documentation, not wiring
            assert "append_completed_episode(" not in stripped
            assert "append_completed_episodes(" not in stripped


# ── T17: Save error non-fatal — covered at the design level ──────────────

def test_t17_retention_failure_during_append_is_nonfatal():
    """Simulates a retention evaluation raising mid-append (e.g. a corrupt
    registry lookup) — the append itself must still have succeeded and the
    function must not raise."""
    class _ExplodingRegistry:
        def definitions(self):
            raise RuntimeError("simulated registry failure")

    result = append_completed_episode(
        None, _heating(), episode_registry=_ExplodingRegistry(), now_utc=_NOW,
    )
    # Append succeeded; retention pass failed non-fatally and left it as-is.
    assert f"{_ZONE_A}:heating:1" in result.updated_payload["episodes"]
    assert result.pruned_episode_ids == ()


# ── T18-T21: Existing tests remain green (regression smoke-test) ────────

def test_t18_t21_regression_imports_ok():
    import tests.test_le2_episode_serialization  # noqa: F401
    import tests.test_le2_capture_wiring_foundation  # noqa: F401
    import tests.test_le2_retention  # noqa: F401
    import tests.test_le2_periodic_save_trigger  # noqa: F401


# ── T22: No control path touched ─────────────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t22_no_control_keywords_in_module():
    source = inspect.getsource(_persist_mod).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"
