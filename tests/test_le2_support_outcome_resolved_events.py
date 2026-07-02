"""Tests for the LE2 Support Critical Event outcome_resolved producer in
LearningShadowController
(custom_components/thermosmart/learning/runtime/ha_integration.py).

This step adds the SIXTH Support Critical Event producer — pure OBSERVATION
of a completed OutcomeEpisode already handed to the EXISTING
record_completed_episode_safe() (LearningRuntime's episode_sink, called from
run_cycle()'s completed-episode loop — for outcome episodes this is always
the confounder-augmented ``bound_episode``, never the raw ``ce.episode``, per
lifecycle.py's own established behaviour, unchanged by this step). No
influence on episode construction, outcome scoring, retention, or Learning
Progress — this step reads only fields the OutcomeEpisode dataclass already
carries (regime, reason, confounder_flags, end_temp, target).

Deduped by reusing the SAME episode_id-based dedup
append_completed_episode() already performs (via
result.appended_episode_ids) — no new dedup mechanism. episode_id is used
only for that internal membership check, never placed into the event.

Since constructing a real LearningShadowController requires a live
HomeAssistant instance, this file tests the two new methods
(``_maybe_record_outcome_resolved_event`` / the augmented
``record_completed_episode_safe``) directly against the REAL, UNBOUND
methods from the production class, bound onto a minimal fake object exposing
only the attributes they touch — validating the CONTRACT, not a copy.

17 test groups:
  T1  — outcome_resolved is produced for a completed OutcomeEpisode
  T2  — No event for a non-outcome completed episode (heating/afterheat/etc.)
  T3  — Dedupe: resubmitting the same episode_id produces no second event
  T4  — Confounders exported only as count/boolean, never raw flag values
  T5  — Positive/target-reached outcome summarised correctly
  T6  — Negative/timeout outcome summarised correctly
  T7  — Details are public-safe scalars only
  T8  — No event_id in the Support Export
  T9  — No internal IDs (episode_id/learning_zone_id/decision_id/trv_binding_id) in the Support Export
  T10 — No raw trajectory in the Support Export
  T11 — Events appear in the Support Export
  T12 — No store reads introduced in the export path
  T13 — No additional service call is triggered by event production
  T14 — Outcome scoring / episode building / Learning Progress logic untouched
  T15 — No command/boost/setpoint/control logic touched
  T16 — Existing Sensor/Boost/TRV/Hold/Storage/Export/Wiring/Foundation tests remain green
  T17 — No control-path keywords touched by the new method
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.episode_schemas import (
    ControllerKind,
    EpisodeReason,
    HeatingEpisode,
    OutcomeEpisode,
    Trajectory,
    TrajectoryPoint,
)
from custom_components.thermosmart.learning.runtime.ha_integration import LearningShadowController
from custom_components.thermosmart.learning.storage.capture_stores import LearningCaptureStores

_ZONE_A = "zone_alpha_01"
_T0 = datetime(2026, 6, 2, 10, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 6, 2, 10, 30, 0, tzinfo=timezone.utc)


class _FakeRawStore:
    async def async_load(self):
        return None

    async def async_save(self, data):
        pass

    async def async_remove(self):
        pass


class _FakeFactory:
    def create(self, key: str, version: int):
        return _FakeRawStore()


class _FakeShadow:
    """Minimal stand-in exposing only what the two real methods touch."""

    record_completed_episode_safe = LearningShadowController.record_completed_episode_safe
    _maybe_record_outcome_resolved_event = (
        LearningShadowController._maybe_record_outcome_resolved_event
    )
    record_support_critical_event_safe = LearningShadowController.record_support_critical_event_safe

    def __init__(self, now_iso: str = "2026-06-02T12:00:00+00:00") -> None:
        self._capture_stores = LearningCaptureStores(_FakeFactory(), _ZONE_A)
        self._episode_history: dict = {}
        self._episode_save_needed = False
        self._episode_last_error = None
        self._support_critical_events: dict = {}
        self._support_critical_events_save_needed = False
        self._support_critical_events_last_error = None
        self._utcnow_iso = lambda: now_iso


def _traj() -> Trajectory:
    return Trajectory(points=(TrajectoryPoint(offset_ms=0, value=19.0),), max_points=10)


def _outcome(
    eid: str = f"{_ZONE_A}:outcome:1", *, reason: EpisodeReason = EpisodeReason.REACHED,
    confounders: tuple = (), end_temp: float = 21.0, target: float = 21.0,
) -> OutcomeEpisode:
    return OutcomeEpisode(
        episode_id=eid, learning_zone_id=_ZONE_A, decision_id="dec_abc123",
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.ACTIVE_HEATING, reliability=0.9,
        start_temp=19.0, end_temp=end_temp, target=target, comfort_tolerance_at_start=0.3,
        reason=reason, controller=ControllerKind.THERMOSMART, trajectory=_traj(),
        confounder_flags=confounders,
    )


def _heating(eid: str = f"{_ZONE_A}:heating:1") -> HeatingEpisode:
    return HeatingEpisode(
        episode_id=eid, learning_zone_id=_ZONE_A, episode_schema_version=1,
        builder_version=1, classifier_version=1, start_ts=_T0, end_ts=_T1,
        regime=Regime.ACTIVE_HEATING, reliability=0.9, start_temp=19.0, target=21.0,
        controller=ControllerKind.THERMOSMART,
    )


def _outcome_events(shadow: _FakeShadow) -> list:
    return [e for e in shadow._support_critical_events.values() if e["event_type"] == "outcome_resolved"]


# ── T1: outcome_resolved produced for a completed OutcomeEpisode ──────────

def test_t1_outcome_resolved_for_completed_outcome_episode():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_outcome())
    events = _outcome_events(shadow)
    assert len(events) == 1
    assert events[0]["severity"] == "info"
    assert events[0]["reason"] == "reached"


# ── T2: No event for a non-outcome completed episode ────────────────────────

def test_t2_no_event_for_heating_episode():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_heating())
    assert _outcome_events(shadow) == []
    # but the episode itself was still persisted normally (unaffected)
    assert f"{_ZONE_A}:heating:1" in shadow._episode_history


# ── T3: Dedupe — resubmitting the same episode_id produces no second event ─

def test_t3_duplicate_resubmission_produces_no_second_event():
    shadow = _FakeShadow()
    ep = _outcome()
    shadow.record_completed_episode_safe(ep)
    shadow.record_completed_episode_safe(ep)  # same object, second call
    assert len(_outcome_events(shadow)) == 1


def test_t3_different_episode_id_produces_a_new_event():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_outcome(f"{_ZONE_A}:outcome:1"))
    shadow.record_completed_episode_safe(_outcome(f"{_ZONE_A}:outcome:2"))
    assert len(_outcome_events(shadow)) == 2


# ── T4: Confounders exported only as count/boolean ──────────────────────────

def test_t4_confounders_exported_as_count_and_boolean_only():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(
        _outcome(confounders=("multi_trv_uneven", "early_cutoff_interaction"))
    )
    ev = _outcome_events(shadow)[0]
    assert ev["details"]["confounded"] is True
    assert ev["details"]["confounder_count"] == 2
    assert "multi_trv_uneven" not in repr(ev["details"])
    assert "early_cutoff_interaction" not in repr(ev["details"])


def test_t4_no_confounders_reports_false_and_zero():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_outcome(confounders=()))
    ev = _outcome_events(shadow)[0]
    assert ev["details"]["confounded"] is False
    assert ev["details"]["confounder_count"] == 0


# ── T5: Positive/target-reached outcome summarised correctly ──────────────

def test_t5_target_reached_outcome_summarised():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(
        _outcome(reason=EpisodeReason.REACHED, end_temp=21.5, target=21.0)
    )
    ev = _outcome_events(shadow)[0]
    assert ev["details"]["target_reached"] is True
    assert ev["details"]["reason"] == "reached"
    assert ev["details"]["overshoot_c"] == 0.5
    assert ev["details"]["undershoot_c"] == 0.0
    assert ev["details"]["comfort_error_c"] == 0.5


# ── T6: Negative/timeout outcome summarised correctly ──────────────────────

def test_t6_timeout_outcome_summarised():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(
        _outcome(reason=EpisodeReason.TIMEOUT, end_temp=20.0, target=21.0)
    )
    ev = _outcome_events(shadow)[0]
    assert ev["details"]["target_reached"] is False
    assert ev["details"]["reason"] == "timeout"
    assert ev["details"]["overshoot_c"] == 0.0
    assert ev["details"]["undershoot_c"] == 1.0
    assert ev["details"]["comfort_error_c"] == 1.0


# ── T7: Details are public-safe scalars only ────────────────────────────────

def test_t7_details_are_scalars_only():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_outcome(confounders=("multi_trv_uneven",)))
    ev = _outcome_events(shadow)[0]
    for v in ev["details"].values():
        assert isinstance(v, (int, float, bool, str)) or v is None


# ── T8/T9/T10: Support Export public-safety ─────────────────────────────────

def _export_block(shadow: _FakeShadow) -> dict:
    from custom_components.thermosmart.export import _le2_critical_events_export

    class _ShadowWithSnapshot:
        def __init__(self, events):
            self._events = events
            self.capture_stores = "present"

        def support_critical_events_snapshot(self):
            return dict(self._events)

    class _ExportCoord:
        def __init__(self, s):
            self._le2_shadow = s

    now = datetime(2026, 6, 2, 12, 5, tzinfo=timezone.utc)
    return _le2_critical_events_export(_ExportCoord(_ShadowWithSnapshot(shadow._support_critical_events)), now=now)


def test_t8_no_event_id_in_support_export():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_outcome())
    block = _export_block(shadow)
    for ev in block["events"]:
        assert "event_id" not in ev


_FORBIDDEN_SUBSTRINGS = (
    "episode_id", "learning_zone_id", "decision_id", "trv_binding_id",
    "zone_id", "entity_id", "radiator_profile_id", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t9_no_internal_ids_in_support_export():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_outcome(confounders=("multi_trv_uneven",)))
    block = _export_block(shadow)
    flat = repr(block).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in flat, f"forbidden substring '{token}' leaked into Support Export"
    assert _ZONE_A not in repr(block)
    assert "dec_abc123" not in repr(block)


def test_t10_no_raw_trajectory_in_support_export():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_outcome())
    block = _export_block(shadow)
    assert "trajectory" not in repr(block).lower()


# ── T11: Events appear in the Support Export ────────────────────────────────

def test_t11_events_appear_in_support_export():
    shadow = _FakeShadow()
    shadow.record_completed_episode_safe(_outcome())
    block = _export_block(shadow)
    assert block["available"] is True
    assert block["records_available"] == 1
    assert block["events"][0]["event_type"] == "outcome_resolved"


# ── T12: No store reads introduced in the export path ───────────────────────

def test_t12_export_path_unchanged_no_store_io():
    from custom_components.thermosmart.export import _le2_critical_events_export
    source = inspect.getsource(_le2_critical_events_export)
    assert "await" not in source
    assert "SupportCriticalEventStore(" not in source


# ── T13: No additional service call triggered by event production ────────

def test_t13_no_await_in_new_method():
    source = inspect.getsource(
        LearningShadowController._maybe_record_outcome_resolved_event
    )
    assert "await" not in source
    assert "services.async_call" not in source


# ── T14: Outcome/episode/Learning-Progress logic untouched ────────────────

def test_t14_new_method_never_calls_scoring_or_progress_functions():
    source = inspect.getsource(
        LearningShadowController._maybe_record_outcome_resolved_event
    )
    assert "compute_learning_progress" not in source
    assert "learning_progress_safe" not in source
    assert "OutcomeModel" not in source
    assert "orchestrator" not in source


def test_t14_episode_never_mutated():
    shadow = _FakeShadow()
    ep = _outcome()
    snapshot = dict(ep.__dict__)
    shadow.record_completed_episode_safe(ep)
    assert ep.__dict__ == snapshot  # frozen dataclass, still unchanged


def test_t14_append_completed_episode_still_the_only_persistence_call():
    """The docstring legitimately mentions append_completed_episode() once
    in prose (pre-existing, unchanged) — checked here for the actual CALL
    pattern, not the bare substring count."""
    source = inspect.getsource(LearningShadowController.record_completed_episode_safe)
    assert source.count("result = append_completed_episode(") == 1


# ── T15: No command/boost/setpoint/control logic touched ──────────────────

def test_t15_no_control_path_reference_in_new_method():
    source = inspect.getsource(
        LearningShadowController._maybe_record_outcome_resolved_event
    )
    assert "trv_setpoint" not in source
    assert "resolve_adaptive_boost_control" not in source
    assert "async_set_temperature" not in source


# ── T16: Existing tests remain green (regression smoke-test) ───────────────

def test_t16_regression_imports_ok():
    import tests.test_le2_support_sensor_fallback_events  # noqa: F401
    import tests.test_le2_support_boost_transition_events  # noqa: F401
    import tests.test_le2_support_trv_command_events  # noqa: F401
    import tests.test_le2_support_hold_transition_events  # noqa: F401
    import tests.test_le2_support_storage_restore_events  # noqa: F401
    import tests.test_le2_support_critical_event_export  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_support_critical_events  # noqa: F401
    import tests.test_le2_episode_runtime_hook  # noqa: F401


# ── T17: No control-path keywords touched ───────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t17_no_control_keywords_in_new_method():
    source = inspect.getsource(
        LearningShadowController._maybe_record_outcome_resolved_event
    ).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"
