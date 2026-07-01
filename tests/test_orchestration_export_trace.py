"""Tests for the runtime-disabled Application Orchestrator export trace.

Covers the read-only research/support export integration added in export.py:
  _le2_orchestration_preview_dict, _le2_adaptation_history_for_research (with
  lifecycle_state), _le2_orchestration_preview_summary.

17 test groups:
  T1  — Research export contains orchestrator preview for a history entry
  T2  — application_enabled in export preview is always False
  T3  — would_apply in real export preview stays False
  T4  — would_apply_if_enabled can be diagnostically True with a full test context
       (only via evaluate_application_orchestration() directly — never via the
       export helper, which always passes runtime_context=None)
  T5  — No raw timestamps in research export preview
  T6  — No entity ids / zone names / paths / secrets in research export preview
  T7  — Support export counts would_apply_if_enabled_count correctly
  T8  — Support export would_apply_count stays 0
  T9  — Missing runtime context blocks with unknown_context
  T10 — lifecycle_state None (error path) blocks with lifecycle_state_unavailable
  T11 — Empty lifecycle state allows evaluation
  T12 — Existing active lifecycle entry blocks
  T13 — Cooldown blocks
  T14 — Export helpers mutate nothing (history / lifecycle state untouched)
  T15 — No application_enabled=True in production code
  T16 — No control keywords touched by the new export helpers
  T17 — Existing tests remain green (regression smoke-tests)
"""
from __future__ import annotations

import inspect

import pytest

from custom_components.thermosmart import export as _export_module
from custom_components.thermosmart.export import (
    _le2_adaptation_history_for_research,
    _le2_orchestration_preview_dict,
    _le2_orchestration_preview_summary,
)
from custom_components.thermosmart.learning.adaptation.application import (
    ApplicationRuntimeContext,
)
from custom_components.thermosmart.learning.adaptation.application_state import (
    ApplicationLifecycleState,
    AppliedAdaptationStateEntry,
)
from custom_components.thermosmart.learning.adaptation.contracts import (
    AdaptationDirection,
    AdaptationLifecycle,
    CandidateType,
)
from custom_components.thermosmart.learning.adaptation.history import CandidateHistoryEntry
from custom_components.thermosmart.learning.adaptation.monitoring import MonitoringStatus
from custom_components.thermosmart.learning.adaptation.orchestrator import (
    evaluate_application_orchestration,
)

_ZONE_A = "zone_alpha_01"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _entry(**overrides) -> CandidateHistoryEntry:
    defaults = dict(
        candidate_key="deadbeef01234567",
        candidate_type=CandidateType.PREHEAT_DELTA_MIN,
        direction=AdaptationDirection.INCREASE,
        first_seen_ts="2026-06-01T10:00:00+00:00",
        last_seen_ts="2026-06-12T10:00:00+00:00",
        seen_count=5,
        supporting_outcome_count=15,
        avg_outcome_quality=0.65,
        avg_timeout_rate=0.32,
        avg_overshoot_rate=0.10,
        last_lifecycle=AdaptationLifecycle.SHADOW,
        dominant_reason="high_timeout_rate:0.32",
        dominant_reason_ratio=0.90,
    )
    defaults.update(overrides)
    return CandidateHistoryEntry(**defaults)


def _lifecycle_entry(status: MonitoringStatus, cooldown_until_ts=None) -> AppliedAdaptationStateEntry:
    return AppliedAdaptationStateEntry(
        candidate_key="deadbeef01234567",
        candidate_type_value="preheat_delta_min",
        direction_value="increase",
        status=status,
        applied_delta_min=3.0,
        previous_cumulative_delta_min=0.0,
        new_cumulative_delta_min=3.0,
        applied_ts="2026-06-01T10:00:00+00:00",
        monitoring_deadline_ts=None,
        last_evaluated_ts=None,
        monitored_outcome_count=0,
        rollback_supported=True,
        rollback_recommended=(status is MonitoringStatus.ROLLBACK_RECOMMENDED),
        rollback_reasons=(),
        adoption_ready=False,
        adopted_ts=None,
        rolled_back_ts=None,
        cooldown_until_ts=cooldown_until_ts,
        last_error=None,
    )


class _FakeShadow:
    def __init__(self, history=None, lifecycle_state=None, last_error=None):
        self._history = dict(history) if history else {}
        self._application_lifecycle_state = lifecycle_state
        self._last_error = last_error

    def adaptation_history_snapshot(self):
        return dict(self._history)

    def adaptation_last_error(self):
        return self._last_error


class _FakeCoord:
    def __init__(self, shadow=None):
        self._le2_shadow = shadow
        self.data = {}


# ── T1: Research export contains orchestrator preview for a history entry ────

def test_t1_research_export_contains_preview():
    tuples = [(_entry(), 11.0, 0.0)]
    result = _le2_adaptation_history_for_research(tuples, lifecycle_state=None)
    assert len(result) == 1
    assert "application_orchestration_preview" in result[0]
    preview = result[0]["application_orchestration_preview"]
    assert preview["candidate_type"] == "preheat_delta_min"
    assert preview["direction"] == "increase"


def test_t1_research_export_empty_when_no_entries():
    assert _le2_adaptation_history_for_research([], lifecycle_state=None) == []


# ── T2: application_enabled in export preview is always False ───────────────

def test_t2_application_enabled_always_false():
    preview = _le2_orchestration_preview_dict(
        _entry(), lifecycle_state=None, span_days=11.0, confounder_ratio=0.0,
    )
    assert preview["application_enabled"] is False


# ── T3: would_apply in real export preview stays False ───────────────────────

def test_t3_would_apply_stays_false():
    state = ApplicationLifecycleState(learning_zone_id=_ZONE_A, updated_at="", entries={})
    preview = _le2_orchestration_preview_dict(
        _entry(), lifecycle_state=state, span_days=11.0, confounder_ratio=0.0,
    )
    assert preview["would_apply"] is False


# ── T4: would_apply_if_enabled can be diagnostically True with a full test
# context — but only via the pure orchestrator function directly, never via
# the export helper (which always passes runtime_context=None) ──────────────

def test_t4_would_apply_if_enabled_true_with_full_test_context():
    ctx = ApplicationRuntimeContext(
        proposed_delta=4.0, current_cumulative_delta=0.0,
        last_applied_ts=None, now_ts="2026-06-12T10:00:00+00:00",
        window_open=False, manual_override=False, vacation_mode=False,
        summer_mode=False, heating_unavailable=False, sensor_unreliable=False,
        control_disabled=False, learning_disabled=False,
        active_control_disabled=False, storage_corrupt=False, confounder_ratio=0.0,
    )
    state = ApplicationLifecycleState(learning_zone_id=_ZONE_A, updated_at="", entries={})
    result = evaluate_application_orchestration(
        history_entry=_entry(), lifecycle_state=state, runtime_context=ctx,
        span_days=11.0, confounder_ratio=0.0,
    )
    assert result.would_apply_if_enabled is True
    assert result.application_enabled is False  # kill-switch untouched


def test_t4_export_helper_never_diagnostically_true():
    """The export helper always passes runtime_context=None — it must never
    report would_apply_if_enabled=True, regardless of how favorable the
    candidate/lifecycle state look."""
    state = ApplicationLifecycleState(learning_zone_id=_ZONE_A, updated_at="", entries={})
    preview = _le2_orchestration_preview_dict(
        _entry(), lifecycle_state=state, span_days=11.0, confounder_ratio=0.0,
    )
    assert preview["would_apply_if_enabled"] is False
    assert "unknown_context" in preview["blocked_reasons"]


# ── T5: No raw timestamps in research export preview ─────────────────────────

_FORBIDDEN_TS_FIELDS = {
    "applied_ts", "monitoring_deadline_ts", "cooldown_until_ts",
    "last_evaluated_ts", "adopted_ts", "rolled_back_ts",
}


def test_t5_no_raw_timestamps_in_preview():
    tuples = [(_entry(), 11.0, 0.0)]
    result = _le2_adaptation_history_for_research(tuples, lifecycle_state=None)
    preview = result[0]["application_orchestration_preview"]
    assert not (_FORBIDDEN_TS_FIELDS & set(preview.keys()))
    assert "applied_record_preview" not in preview  # would carry raw timestamps


# ── T6: No entity ids / zone names / paths / secrets in research export ─────

_FORBIDDEN_SUBSTRINGS = (
    "entity_id", "zone_name", "person", "/home/", "c:\\", "password", "token", "secret",
)


def test_t6_no_privacy_leaks_in_preview():
    tuples = [(_entry(), 11.0, 0.0)]
    result = _le2_adaptation_history_for_research(tuples, lifecycle_state=None)
    text = str(result).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in text


# ── T7: Support export counts would_apply_if_enabled_count correctly ────────

def test_t7_support_export_would_apply_if_enabled_count():
    # With runtime_context always None in the export path, unknown_context
    # blocks would_apply_if_enabled for every entry — count must be 0.
    history = {"deadbeef01234567": _entry()}
    shadow = _FakeShadow(history=history)
    coord = _FakeCoord(shadow=shadow)

    result = _le2_orchestration_preview_summary(coord, _ZONE_A)
    assert result["entry_count"] == 1
    assert result["would_apply_if_enabled_count"] == 0


# ── T8: Support export would_apply_count stays 0 ─────────────────────────────

def test_t8_support_export_would_apply_count_zero():
    history = {"deadbeef01234567": _entry()}
    shadow = _FakeShadow(history=history)
    coord = _FakeCoord(shadow=shadow)

    result = _le2_orchestration_preview_summary(coord, _ZONE_A)
    assert result["would_apply_count"] == 0
    assert result["application_enabled"] is False


def test_t8_support_export_zero_when_no_shadow():
    class _NoShadowCoord:
        _le2_shadow = None

    result = _le2_orchestration_preview_summary(_NoShadowCoord(), _ZONE_A)
    assert result["entry_count"] == 0
    assert result["would_apply_count"] == 0
    assert result["last_error"] is None


# ── T9: Missing runtime context blocks with unknown_context ──────────────────

def test_t9_missing_runtime_context_blocks():
    preview = _le2_orchestration_preview_dict(
        _entry(), lifecycle_state=None, span_days=11.0, confounder_ratio=0.0,
    )
    assert "unknown_context" in preview["blocked_reasons"]


# ── T10: lifecycle_state None (error path) blocks with lifecycle_state_unavailable

def test_t10_lifecycle_state_none_blocks():
    preview = _le2_orchestration_preview_dict(
        _entry(), lifecycle_state=None, span_days=11.0, confounder_ratio=0.0,
    )
    assert "lifecycle_state_unavailable" in preview["blocked_reasons"]
    assert preview["lifecycle_existing_status"] is None


# ── T11: Empty lifecycle state allows evaluation ─────────────────────────────

def test_t11_empty_lifecycle_state_allows_evaluation():
    state = ApplicationLifecycleState(learning_zone_id=_ZONE_A, updated_at="", entries={})
    preview = _le2_orchestration_preview_dict(
        _entry(), lifecycle_state=state, span_days=11.0, confounder_ratio=0.0,
    )
    assert "lifecycle_state_unavailable" not in preview["blocked_reasons"]
    assert preview["lifecycle_existing_status"] is None


# ── T12: Existing active lifecycle entry blocks ──────────────────────────────

def test_t12_active_lifecycle_entry_blocks():
    state = ApplicationLifecycleState(
        learning_zone_id=_ZONE_A, updated_at="",
        entries={"deadbeef01234567": _lifecycle_entry(MonitoringStatus.PENDING)},
    )
    preview = _le2_orchestration_preview_dict(
        _entry(), lifecycle_state=state, span_days=11.0, confounder_ratio=0.0,
    )
    assert "lifecycle_entry_active" in preview["blocked_reasons"]
    assert preview["lifecycle_existing_status"] == MonitoringStatus.PENDING.value


# ── T13: Cooldown blocks ───────────────────────────────────────────────────────

def test_t13_cooldown_blocks():
    state = ApplicationLifecycleState(
        learning_zone_id=_ZONE_A, updated_at="",
        entries={"deadbeef01234567": _lifecycle_entry(
            MonitoringStatus.ADOPTED, cooldown_until_ts="2099-01-01T00:00:00+00:00",
        )},
    )
    result = evaluate_application_orchestration(
        history_entry=_entry(), lifecycle_state=state, runtime_context=None,
        span_days=11.0, confounder_ratio=0.0, now_ts="2026-06-12T10:00:00+00:00",
    )
    assert result.cooldown_active is True
    assert "lifecycle_cooldown_active" in result.blocked_reasons


# ── T14: Export helpers mutate nothing ────────────────────────────────────────

def test_t14_export_does_not_mutate_history_or_lifecycle_state():
    entry = _entry()
    state = ApplicationLifecycleState(learning_zone_id=_ZONE_A, updated_at="", entries={})
    history = {entry.candidate_key: entry}
    shadow = _FakeShadow(history=history, lifecycle_state=state)
    coord = _FakeCoord(shadow=shadow)

    tuples = [(entry, 11.0, 0.0)]
    _le2_adaptation_history_for_research(tuples, lifecycle_state=state)
    _le2_orchestration_preview_summary(coord, _ZONE_A)

    # Inputs are unchanged — same entry, same empty lifecycle entries dict.
    assert shadow._history == history
    assert shadow._application_lifecycle_state.entries == {}
    assert entry.candidate_key == "deadbeef01234567"


# ── T15: No application_enabled=True in production code ─────────────────────

def test_t15_no_application_enabled_true_in_export_module():
    source = inspect.getsource(_export_module)
    assert "application_enabled=True" not in source


def test_t15_no_application_enabled_true_in_orchestrator_module():
    from custom_components.thermosmart.learning.adaptation import orchestrator as _orch
    source = inspect.getsource(_orch)
    assert "application_enabled=True" not in source


# ── T16: No control keywords touched by the new export helpers ──────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
    "setpoint",
)


def test_t16_no_control_keywords_in_new_helpers():
    for fn in (
        _le2_orchestration_preview_dict,
        _le2_adaptation_history_for_research,
        _le2_orchestration_preview_summary,
    ):
        source = inspect.getsource(fn).lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in source, f"forbidden control token found in {fn.__name__}: {token}"


# ── T17: Existing tests remain green (import smoke-test) ────────────────────

def test_t17_regression_imports_ok():
    import tests.test_application_orchestrator  # noqa: F401
    import tests.test_application_lifecycle_storage  # noqa: F401
    import tests.test_application_lifecycle_state  # noqa: F401
    import tests.test_adaptation_monitoring  # noqa: F401
    import tests.test_preheat_application_plan  # noqa: F401
    import tests.test_application_readiness  # noqa: F401
    import tests.test_promotion_readiness  # noqa: F401
