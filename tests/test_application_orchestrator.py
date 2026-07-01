"""Tests for evaluate_application_orchestration() — the runtime-disabled
adaptation application orchestrator.

19 test groups:
  T1  — Default policy keeps application disabled
  T2  — Eligible PREHEAT candidate produces blocked preview with application_disabled
  T3  — Test policy enabled + all gates green produces would_apply=True
  T4  — would_apply=True does not mutate lifecycle_state
  T5  — Non-eligible candidate blocked
  T6  — Non-SHADOW lifecycle blocked
  T7  — Unsupported candidate type blocked
  T8  — EARLY_CUTOFF_DELTA_MIN still not applied in this orchestrator step
  T9  — Existing active lifecycle entry blocks duplicate application
  T10 — Existing cooldown blocks
  T11 — lifecycle_state None blocks conservatively
  T12 — Empty lifecycle_state allows evaluation
  T13 — Runtime context missing blocks safely
  T14 — Safety flags block
  T15 — Bounded delta is carried into result
  T16 — Monitoring/Rollback flags are carried into result
  T17 — Result to_dict() is public-safe
  T18 — No control keywords in orchestrator module
  T19 — Existing tests remain green (regression smoke-tests)
"""
from __future__ import annotations

import inspect

import pytest

from custom_components.thermosmart.learning.adaptation import orchestrator as _orch_module
from custom_components.thermosmart.learning.adaptation.orchestrator import (
    ApplicationOrchestrationMode,
    ApplicationOrchestrationResult,
    ApplicationOrchestratorPolicy,
    evaluate_application_orchestration,
)
from custom_components.thermosmart.learning.adaptation.application import (
    ApplicationPolicy,
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


def _ctx(**overrides) -> ApplicationRuntimeContext:
    defaults = dict(
        proposed_delta=4.0,
        current_cumulative_delta=0.0,
        last_applied_ts=None,
        now_ts="2026-06-12T10:00:00+00:00",
        window_open=False,
        manual_override=False,
        vacation_mode=False,
        summer_mode=False,
        heating_unavailable=False,
        sensor_unreliable=False,
        control_disabled=False,
        learning_disabled=False,
        active_control_disabled=False,
        storage_corrupt=False,
        confounder_ratio=0.0,
    )
    defaults.update(overrides)
    return ApplicationRuntimeContext(**defaults)


def _run(**overrides):
    defaults = dict(
        history_entry=_entry(),
        lifecycle_state=ApplicationLifecycleState(
            learning_zone_id="zone_a", updated_at="", entries={},
        ),
        runtime_context=_ctx(),
        span_days=11.0,
        confounder_ratio=0.0,
        now_ts="2026-06-12T10:00:00+00:00",
        policy=None,
    )
    defaults.update(overrides)
    return evaluate_application_orchestration(**defaults)


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


# ── T1: Default policy keeps application disabled ────────────────────────────

def test_t1_default_policy_application_disabled():
    result = _run()
    assert result.application_enabled is False


def test_t1_default_policy_object():
    policy = ApplicationOrchestratorPolicy()
    assert policy.application_policy.application_enabled is False


# ── T2: Eligible PREHEAT candidate produces blocked preview with application_disabled ─

def test_t2_eligible_preheat_blocked_by_kill_switch():
    result = _run()
    assert result.would_apply is False
    assert "application_disabled" in result.blocked_reasons
    assert result.would_apply_if_enabled is True
    assert result.plan_status == "blocked_by_kill_switch"
    assert result.application_readiness == "eligible"


# ── T3: Test policy enabled + all gates green → would_apply=True ─────────────

def test_t3_test_policy_enabled_would_apply_true():
    policy = ApplicationOrchestratorPolicy(
        application_policy=ApplicationPolicy(application_enabled=True)
    )
    result = _run(policy=policy)
    assert result.would_apply is True
    assert result.plan_status == "would_apply"
    assert result.blocked_reasons == ()


# ── T4: would_apply=True does not mutate lifecycle_state ─────────────────────

def test_t4_would_apply_does_not_mutate_lifecycle_state():
    policy = ApplicationOrchestratorPolicy(
        application_policy=ApplicationPolicy(application_enabled=True)
    )
    state = ApplicationLifecycleState(learning_zone_id="zone_a", updated_at="", entries={})
    result = _run(policy=policy, lifecycle_state=state)
    assert result.would_apply is True
    # Input object is untouched — same empty entries, no new keys written.
    assert state.entries == {}
    assert result.applied_record_preview is not None
    assert isinstance(result.applied_record_preview, dict)


# ── T5: Non-eligible candidate blocked ────────────────────────────────────────

def test_t5_non_eligible_promotion_blocked():
    entry = _entry(seen_count=1)  # fails min_seen_count gate
    result = _run(history_entry=entry)
    assert result.promotion_readiness == "blocked"
    assert result.would_apply is False
    assert any(r.startswith("not_eligible:") for r in result.blocked_reasons)


# ── T6: Non-SHADOW lifecycle blocked ──────────────────────────────────────────

def test_t6_non_shadow_lifecycle_blocked():
    entry = _entry(last_lifecycle=AdaptationLifecycle.CANDIDATE)
    result = _run(history_entry=entry)
    assert result.would_apply is False
    assert any(r.startswith("lifecycle_not_shadow:") for r in result.blocked_reasons)


# ── T7: Unsupported candidate type blocked ────────────────────────────────────

@pytest.mark.parametrize("ctype", [
    CandidateType.BOOST_BIAS,
    CandidateType.TPI_GAIN_BIAS,
    CandidateType.COMFORT_BIAS_C,
])
def test_t7_unsupported_candidate_type_blocked(ctype):
    entry = _entry(candidate_type=ctype)
    result = _run(history_entry=entry)
    assert result.would_apply is False
    assert result.would_apply_if_enabled is False
    assert any(r.startswith("type_not_allowed:") for r in result.blocked_reasons)


# ── T8: EARLY_CUTOFF_DELTA_MIN still not applied in this orchestrator step ───

def test_t8_early_cutoff_not_applied():
    entry = _entry(candidate_type=CandidateType.EARLY_CUTOFF_DELTA_MIN)
    policy = ApplicationOrchestratorPolicy(
        application_policy=ApplicationPolicy(application_enabled=True)
    )
    result = _run(history_entry=entry, policy=policy)
    assert result.would_apply is False
    assert any(r.startswith("plan_type_not_supported:") for r in result.blocked_reasons)


# ── T9: Existing active lifecycle entry blocks duplicate application ────────

@pytest.mark.parametrize("status", [
    MonitoringStatus.PENDING,
    MonitoringStatus.IN_PROGRESS,
    MonitoringStatus.ADOPTION_CANDIDATE,
    MonitoringStatus.ROLLBACK_RECOMMENDED,
])
def test_t9_existing_active_entry_blocks(status):
    state = ApplicationLifecycleState(
        learning_zone_id="zone_a", updated_at="",
        entries={"deadbeef01234567": _lifecycle_entry(status)},
    )
    policy = ApplicationOrchestratorPolicy(
        application_policy=ApplicationPolicy(application_enabled=True)
    )
    result = _run(lifecycle_state=state, policy=policy)
    assert result.would_apply is False
    assert result.would_apply_if_enabled is False
    assert "lifecycle_entry_active" in result.blocked_reasons
    assert result.lifecycle_existing_status == status.value


# ── T10: Existing cooldown blocks ─────────────────────────────────────────────

def test_t10_existing_cooldown_blocks():
    state = ApplicationLifecycleState(
        learning_zone_id="zone_a", updated_at="",
        entries={"deadbeef01234567": _lifecycle_entry(
            MonitoringStatus.ADOPTED, cooldown_until_ts="2099-01-01T00:00:00+00:00",
        )},
    )
    policy = ApplicationOrchestratorPolicy(
        application_policy=ApplicationPolicy(application_enabled=True)
    )
    result = _run(lifecycle_state=state, policy=policy, now_ts="2026-06-12T10:00:00+00:00")
    assert result.cooldown_active is True
    assert "lifecycle_cooldown_active" in result.blocked_reasons
    assert result.would_apply is False


def test_t10_expired_cooldown_does_not_block_via_cooldown_reason():
    state = ApplicationLifecycleState(
        learning_zone_id="zone_a", updated_at="",
        entries={"deadbeef01234567": _lifecycle_entry(
            MonitoringStatus.ADOPTED, cooldown_until_ts="2020-01-01T00:00:00+00:00",
        )},
    )
    result = _run(lifecycle_state=state, now_ts="2026-06-12T10:00:00+00:00")
    assert result.cooldown_active is False
    assert "lifecycle_cooldown_active" not in result.blocked_reasons
    # still blocked, but via the terminal-status reason, not cooldown
    assert "lifecycle_already_adopted" in result.blocked_reasons


# ── T11: lifecycle_state None blocks conservatively ──────────────────────────

def test_t11_lifecycle_state_none_blocks():
    policy = ApplicationOrchestratorPolicy(
        application_policy=ApplicationPolicy(application_enabled=True)
    )
    result = _run(lifecycle_state=None, policy=policy)
    assert result.would_apply is False
    assert result.would_apply_if_enabled is False
    assert "lifecycle_state_unavailable" in result.blocked_reasons
    assert result.lifecycle_existing_status is None


# ── T12: Empty lifecycle_state allows evaluation ─────────────────────────────

def test_t12_empty_lifecycle_state_allows_evaluation():
    state = ApplicationLifecycleState(learning_zone_id="zone_a", updated_at="", entries={})
    policy = ApplicationOrchestratorPolicy(
        application_policy=ApplicationPolicy(application_enabled=True)
    )
    result = _run(lifecycle_state=state, policy=policy)
    assert result.would_apply is True
    assert result.lifecycle_existing_status is None


# ── T13: Runtime context missing blocks safely ────────────────────────────────

def test_t13_missing_runtime_context_blocks():
    result = _run(runtime_context=None)
    assert result.would_apply is False
    assert result.would_apply_if_enabled is False
    assert "unknown_context" in result.blocked_reasons


# ── T14: Safety flags block ───────────────────────────────────────────────────

def test_t14_safety_flag_window_open_blocks():
    ctx = _ctx(window_open=True)
    result = _run(runtime_context=ctx)
    assert "safety_window_open" in result.blocked_reasons
    assert "safety_window_open" in result.safety_reasons
    assert result.would_apply_if_enabled is False


# ── T15: Bounded delta is carried into result ────────────────────────────────

def test_t15_bounded_delta_carried():
    ctx = _ctx(proposed_delta=9.0, current_cumulative_delta=0.0)
    result = _run(runtime_context=ctx)
    assert result.bounded_delta_min == 5.0  # capped to max step
    assert result.next_cumulative_delta_min == 5.0


# ── T16: Monitoring/Rollback flags are carried into result ──────────────────

def test_t16_monitoring_rollback_flags_carried():
    result = _run()
    assert result.monitoring_required is True
    assert result.rollback_supported is True


# ── T17: Result to_dict() is public-safe ──────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "entity_id", "zone_name", "person", "/home/", "c:\\", "password", "token", "secret",
)


def test_t17_to_dict_public_safe():
    result = _run()
    d = result.to_dict()
    text = str(d).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in text
    # candidate_key is an opaque hash, not a raw zone id
    assert d["candidate_key"] == "deadbeef01234567"


def test_t17_to_dict_all_expected_fields_present():
    result = _run()
    d = result.to_dict()
    expected_keys = {
        "candidate_key", "candidate_type", "direction", "promotion_readiness",
        "application_enabled", "application_readiness", "plan_status",
        "would_apply", "would_apply_if_enabled", "blocked_reasons",
        "safety_reasons", "bounded_delta_min", "next_cumulative_delta_min",
        "monitoring_required", "rollback_supported", "lifecycle_existing_status",
        "cooldown_active",
    }
    assert expected_keys.issubset(d.keys())


# ── T18: No control keywords in orchestrator module ──────────────────────────

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


def test_t18_no_control_keywords_in_orchestrator_module():
    source = inspect.getsource(_orch_module)
    lowered = source.lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in lowered, f"forbidden control token found: {token}"


def test_t18_orchestrator_result_is_frozen_dataclass():
    result = _run()
    with pytest.raises((AttributeError, TypeError)):
        result.would_apply = True  # frozen — must raise


# ── T19: Existing tests remain green (import smoke-test) ────────────────────

def test_t19_regression_imports_ok():
    import tests.test_application_lifecycle_storage  # noqa: F401
    import tests.test_application_lifecycle_state  # noqa: F401
    import tests.test_adaptation_monitoring  # noqa: F401
    import tests.test_preheat_application_plan  # noqa: F401
    import tests.test_application_readiness  # noqa: F401
    import tests.test_promotion_readiness  # noqa: F401
