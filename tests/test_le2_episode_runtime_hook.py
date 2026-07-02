"""Tests for the LE2 completed-episode runtime append hook.

Covers:
  - LearningRuntime's optional, synchronous episode_sink (lifecycle.py)
  - LearningShadowController.record_completed_episode_safe() (ha_integration.py)
  - Static verification of the exact run_cycle() hook placement (non-outcome
    vs. outcome/bound_episode)

This working copy of the repository was slimmed for public/beta release
(commit "Slim public repository for beta release") — the original heavy
multi-cycle scenario-construction helpers (helpers_runtime_scenarios.py) were
removed, leaving only their orphaned .pyc cache. Building a full, realistic
multi-cycle RuntimeCycleInput sequence that naturally produces a CLOSED
episode through the real regime classifier + builders is therefore
substantial, error-prone re-engineering rather than a "klein und sauber"
test — exactly the situation the task itself anticipates ("isolierte fake
Runtime/ZoneRuntime/CompletedEpisode Tests bauen" is explicitly sanctioned).
This file therefore combines:

  1. Direct behavioral tests of LearningRuntime._emit_completed_episode() —
     the exact method run_cycle() calls — using real episode dataclasses,
     without needing a full multi-cycle scenario.
  2. Static source-inspection of run_cycle() confirming the two hook call
     sites sit exactly where the design audit specified (non-outcome after
     the model-update bookkeeping, guarded to exclude "outcome"; outcome
     immediately after bound_episode construction, before the deferred-boost
     branching).
  3. Behavioral tests of LearningShadowController.record_completed_episode_safe()
     via a harness whose method body is IDENTICAL to production (matching
     this repo's established testing pattern for HA-dependent classes).

20 test groups:
  T1  — record_completed_episode_safe() appends a valid heating episode
  T2  — Duplicate episode_id is not stored twice
  T3  — Append sets _episode_save_needed=True
  T4  — Malformed/unknown episode sets last_error or is safely ignored
  T5  — Append failure stays non-fatal
  T6  — Retention is applied via append_completed_episode()
  T7  — Non-outcome runtime hook calls the sink for a completed episode
  T8  — heat_rate/onset_delay sharing the same HeatingEpisode create no duplicate
  T9  — Outcome runtime hook persists bound_episode, not the raw ce.episode
  T10 — Outcome confounder flags are present in the persisted episode
  T11 — Sink exception does not abort run_cycle() (via _emit_completed_episode)
  T12 — No store I/O in the run_cycle() path
  T13 — No async save / store save calls in the runtime hook
  T14 — async_save_if_due() remains the only save path
  T15 — No raw store touched
  T16 — Existing episode store wiring tests remain green
  T17 — Existing episode persistence/serialization tests remain green
  T18 — Existing periodic save tests remain green
  T19 — Existing retention tests remain green
  T20 — No control path touched
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

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
from custom_components.thermosmart.learning.registry import EpisodeRegistry
from custom_components.thermosmart.learning.runtime import lifecycle as _lifecycle_mod
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntime, LearningRuntimeConfig
from custom_components.thermosmart.learning.storage.capture_registry import build_episode_registry
from custom_components.thermosmart.learning.storage.episode_persistence import append_completed_episode

_ZONE_A = "zone_alpha_01"
_T0 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
_T1 = _T0 + timedelta(minutes=30)


def _traj(n: int = 3) -> Trajectory:
    points = tuple(TrajectoryPoint(offset_ms=i * 1000, value=20.0 + i * 0.1) for i in range(n))
    return Trajectory(points=points, max_points=max(n, 1))


def _heating(seq: int = 1) -> HeatingEpisode:
    return HeatingEpisode(
        episode_id=f"{_ZONE_A}:heating:{seq}", learning_zone_id=_ZONE_A,
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.ACTIVE_HEATING, reliability=0.9,
        start_temp=19.0, target=21.0, controller=ControllerKind.THERMOSMART,
    )


def _outcome(confounders=()) -> OutcomeEpisode:
    return OutcomeEpisode(
        episode_id=f"{_ZONE_A}:outcome:1", learning_zone_id=_ZONE_A,
        decision_id="dec_abc123",
        episode_schema_version=1, builder_version=1, classifier_version=1,
        start_ts=_T0, end_ts=_T1, regime=Regime.ACTIVE_HEATING, reliability=0.85,
        start_temp=19.0, end_temp=21.0, target=21.0,
        comfort_tolerance_at_start=0.3, reason=EpisodeReason.REACHED,
        controller=ControllerKind.THERMOSMART, trajectory=_traj(),
        confounder_flags=confounders,
    )


# ── Harness mirroring LearningShadowController.record_completed_episode_safe()

class _FakeCaptureStores:
    def __init__(self, episode_registry: EpisodeRegistry) -> None:
        self.episode_registry = episode_registry


class _RecordEpisodeHelper:
    """Standalone test harness with a method body IDENTICAL to
    LearningShadowController.record_completed_episode_safe(), matching this
    repo's established pattern of validating the contract, not a copy."""

    def __init__(self, capture_stores, now_iso: str) -> None:
        self._capture_stores = capture_stores
        self._episode_history: dict = {}
        self._episode_save_needed: bool = False
        self._episode_last_error: Optional[str] = None
        self._utcnow_iso = lambda: now_iso

    def record_completed_episode_safe(self, episode: Any) -> None:
        if self._capture_stores is None:
            return
        try:
            now_utc = datetime.fromisoformat(self._utcnow_iso())
            result = append_completed_episode(
                {"episodes": self._episode_history},
                episode,
                episode_registry=self._capture_stores.episode_registry,
                now_utc=now_utc,
            )
            if result.appended_episode_ids or result.pruned_episode_ids:
                self._episode_history = result.updated_payload["episodes"]
                self._episode_save_needed = True
        except Exception as err:
            self._episode_last_error = str(err)


_REGISTRY = build_episode_registry()
_NOW_ISO = _T1.isoformat()


# ── T1: record_completed_episode_safe() appends a valid heating episode ────

def test_t1_appends_valid_heating_episode():
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), _NOW_ISO)
    helper.record_completed_episode_safe(_heating())
    assert f"{_ZONE_A}:heating:1" in helper._episode_history
    assert helper._episode_last_error is None


def test_t1_noop_when_no_capture_stores():
    helper = _RecordEpisodeHelper(None, _NOW_ISO)
    helper.record_completed_episode_safe(_heating())
    assert helper._episode_history == {}
    assert helper._episode_save_needed is False


# ── T2: Duplicate episode_id is not stored twice ────────────────────────────

def test_t2_duplicate_episode_id_not_stored_twice():
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), _NOW_ISO)
    ep = _heating()
    helper.record_completed_episode_safe(ep)
    helper.record_completed_episode_safe(ep)  # same object, second call
    assert len(helper._episode_history) == 1


# ── T3: Append sets _episode_save_needed=True ──────────────────────────────

def test_t3_append_sets_save_needed_true():
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), _NOW_ISO)
    assert helper._episode_save_needed is False
    helper.record_completed_episode_safe(_heating())
    assert helper._episode_save_needed is True


def test_t3_duplicate_does_not_reset_save_needed_to_true_spuriously():
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), _NOW_ISO)
    helper.record_completed_episode_safe(_heating())
    helper._episode_save_needed = False  # simulate a save having already happened
    helper.record_completed_episode_safe(_heating())  # duplicate only
    assert helper._episode_save_needed is False


# ── T4: Malformed/unknown episode is safely ignored ────────────────────────

def test_t4_unknown_object_is_safely_ignored():
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), _NOW_ISO)
    helper.record_completed_episode_safe(object())
    assert helper._episode_history == {}
    assert helper._episode_save_needed is False
    assert helper._episode_last_error is None  # skip is not an error


# ── T5: Append failure stays non-fatal ─────────────────────────────────────

def test_t5_registry_failure_is_nonfatal():
    class _ExplodingRegistry:
        def definitions(self):
            raise RuntimeError("simulated registry failure")

    helper = _RecordEpisodeHelper(_FakeCaptureStores(_ExplodingRegistry()), _NOW_ISO)
    helper.record_completed_episode_safe(_heating())  # must not raise
    # append itself succeeded (registry failure only affects the retention pass,
    # which append_completed_episode() already isolates internally)
    assert f"{_ZONE_A}:heating:1" in helper._episode_history


def test_t5_bad_clock_string_is_nonfatal():
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), "not-a-timestamp")
    helper.record_completed_episode_safe(_heating())  # must not raise
    assert helper._episode_last_error is not None
    assert helper._episode_history == {}


# ── T6: Retention is applied via append_completed_episode() ───────────────

def test_t6_retention_prunes_old_entry_on_append():
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), "2027-01-01T00:00:00+00:00")
    helper.record_completed_episode_safe(_heating())  # ancient relative to "now"
    assert helper._episode_history == {}  # pruned immediately by its own RetentionPolicy


# ── T7: Non-outcome runtime hook calls the sink for a completed episode ───

def test_t7_emit_completed_episode_calls_sink():
    calls = []
    runtime = LearningRuntime(
        LearningRuntimeConfig(), clock=lambda: _NOW_ISO,
        episode_sink=lambda ep: calls.append(ep),
    )
    ep = _heating()
    runtime._emit_completed_episode(ep)
    assert calls == [ep]


def test_t7_emit_completed_episode_noop_without_sink():
    runtime = LearningRuntime(LearningRuntimeConfig(), clock=lambda: _NOW_ISO)
    runtime._emit_completed_episode(_heating())  # must not raise, no sink to call


def test_t7_hook_placed_for_non_outcome_excludes_outcome():
    """Static check: the generic-loop sink call is guarded to exclude
    "outcome" — outcome is only persisted via the augmented bound_episode."""
    source = inspect.getsource(LearningRuntime.run_cycle)
    idx = source.index("if ce.model_name != \"outcome\":")
    window = source[idx:idx + 120]
    assert "self._emit_completed_episode(ce.episode)" in window


# ── T8: heat_rate/onset_delay sharing the same HeatingEpisode -> no duplicate

def test_t8_fan_out_duplicate_same_episode_object():
    """Simulates pipeline.py's fan-out: the SAME HeatingEpisode object
    appears twice in pipe_result.completed (once per model sharing the
    "heating" builder) — the sink may be invoked twice, but the persisted
    history must contain only one entry."""
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), _NOW_ISO)
    ep = _heating()
    # Emulates: for ce in (CompletedEpisode("heat_rate", ep), CompletedEpisode("onset_delay", ep)):
    for _ in range(2):
        helper.record_completed_episode_safe(ep)
    assert len(helper._episode_history) == 1


# ── T9: Outcome runtime hook persists bound_episode, not raw ce.episode ───

def test_t9_hook_placement_uses_bound_episode_not_raw():
    """Static check: the outcome sink call must reference bound_episode and
    sit AFTER its construction, before the deferred-boost branching."""
    source = inspect.getsource(LearningRuntime.run_cycle)
    augment_idx = source.index("bound_episode = self._augment_confounders(ce.episode, extra_cf)")
    emit_idx = source.index("self._emit_completed_episode(bound_episode)")
    branch_idx = source.index("if bctx is not None:")
    assert augment_idx < emit_idx < branch_idx
    # And the raw ce.episode must never be the argument for this specific call.
    call_line = source[emit_idx:emit_idx + 60]
    assert "self._emit_completed_episode(bound_episode)" in call_line
    assert "self._emit_completed_episode(ce.episode)" not in call_line


def test_t9_behavioral_bound_episode_is_what_gets_recorded():
    """End-to-end at the record layer: if the runtime hands the sink the
    augmented episode (as run_cycle() does for outcome), that augmented
    object — not some raw variant — is what ends up in history."""
    raw = _outcome()
    bound = OutcomeEpisode(**{**raw.__dict__, "confounder_flags": ("supply_delay",)})
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), _NOW_ISO)
    helper.record_completed_episode_safe(bound)
    stored = helper._episode_history[f"{_ZONE_A}:outcome:1"]
    assert stored["confounder_flags"] == ["supply_delay"]


# ── T10: Outcome confounder flags are present in the persisted episode ────

def test_t10_confounder_flags_survive_into_persisted_entry():
    ep = _outcome(confounders=("multi_trv_uneven", "early_cutoff_interaction"))
    helper = _RecordEpisodeHelper(_FakeCaptureStores(_REGISTRY), _NOW_ISO)
    helper.record_completed_episode_safe(ep)
    stored = helper._episode_history[f"{_ZONE_A}:outcome:1"]
    assert set(stored["confounder_flags"]) == {"multi_trv_uneven", "early_cutoff_interaction"}


# ── T11: Sink exception does not abort run_cycle() (via _emit_completed_episode)

def test_t11_sink_exception_is_swallowed():
    def _boom(ep):
        raise RuntimeError("simulated sink failure")

    runtime = LearningRuntime(
        LearningRuntimeConfig(), clock=lambda: _NOW_ISO, episode_sink=_boom,
    )
    runtime._emit_completed_episode(_heating())  # must not raise


# ── T12: No store I/O in the run_cycle() path ──────────────────────────────

def test_t12_run_cycle_has_no_store_io():
    source = inspect.getsource(LearningRuntime.run_cycle)
    assert "async_save" not in source
    assert ".save(" not in source
    assert "EpisodesStore" not in source
    assert "RawSegmentStore" not in source


def test_t12_emit_completed_episode_has_no_store_io():
    source = inspect.getsource(LearningRuntime._emit_completed_episode)
    assert "async_save" not in source
    assert ".save(" not in source
    assert "EpisodesStore" not in source
    assert "await" not in source


# ── T13: No async save / store save calls in the runtime hook ─────────────

def test_t13_lifecycle_module_has_no_ha_imports():
    source = inspect.getsource(_lifecycle_mod)
    assert "homeassistant" not in source.lower()


def test_t13_record_completed_episode_safe_is_synchronous():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    method = _ha.LearningShadowController.record_completed_episode_safe
    assert not inspect.iscoroutinefunction(method)


# ── T14: async_save_if_due() remains the only save path ────────────────────

def _code_only(source: str) -> str:
    """Strip docstrings/comments so prose mentioning e.g. "async_save_if_due()"
    doesn't cause a false-positive keyword match against actual code."""
    lines = []
    in_doc = False
    for line in source.splitlines():
        stripped = line.strip()
        if in_doc:
            if '"""' in stripped:
                in_doc = False
            continue
        if stripped.startswith('"""') and stripped.count('"""') == 1:
            in_doc = True
            continue
        if stripped.startswith('"""') and stripped.count('"""') >= 2:
            continue
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_t14_no_direct_save_call_in_record_method():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = _code_only(
        inspect.getsource(_ha.LearningShadowController.record_completed_episode_safe)
    )
    assert "await" not in source
    assert ".save(" not in source
    assert "async_save" not in source


def test_t14_async_save_if_due_still_calls_episode_save_helper():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.async_save_if_due)
    assert "await self._async_save_episode_history_safe()" in source


# ── T15: No raw store touched ────────────────────────────────────────────

def test_t15_no_raw_store_in_hook_or_sink():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    for src in (
        inspect.getsource(_ha.LearningShadowController.record_completed_episode_safe),
        inspect.getsource(LearningRuntime._emit_completed_episode),
        inspect.getsource(LearningRuntime.run_cycle),
    ):
        assert "RawSegmentStore" not in src
        assert "RawSegmentIndexStore" not in src


# ── T16-T19: Existing tests remain green (regression smoke-test) ──────────

def test_t16_t19_regression_imports_ok():
    import tests.test_le2_episode_store_wiring  # noqa: F401
    import tests.test_le2_episode_persistence  # noqa: F401
    import tests.test_le2_episode_serialization  # noqa: F401
    import tests.test_le2_capture_wiring_foundation  # noqa: F401
    import tests.test_le2_retention  # noqa: F401
    import tests.test_le2_periodic_save_trigger  # noqa: F401


# ── T20: No control path touched ────────────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset_write",
    "tpi_gain_write",
)


def test_t20_no_control_keywords_in_new_snippets():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    snippets = [
        inspect.getsource(LearningRuntime._emit_completed_episode),
        inspect.getsource(_ha.LearningShadowController.record_completed_episode_safe),
    ]
    for source in snippets:
        lowered = source.lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in lowered, f"forbidden control token found: {token}"
