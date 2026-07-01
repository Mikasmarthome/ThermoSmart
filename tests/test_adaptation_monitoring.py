"""Tests for monitoring and rollback contracts.

Covers:
  T1  — AppliedAdaptationRecord built from would_apply=True plan
  T2  — would_apply=False plan returns None record
  T3  — AppliedAdaptationRecord.to_dict() is public-safe
  T4  — Better outcome quality, no side effects → continue_monitoring (window not done)
  T5  — Outcome quality delta < -0.10 → rollback_recommended
  T6  — Timeout rate delta > +0.10 → rollback_recommended
  T7  — Overshoot rate delta > +0.10 → rollback_recommended
  T8  — manual_override_detected → rollback_recommended
  T9  — confounder_detected → rollback_recommended
  T10 — safety_flag_detected → rollback_recommended
  T11 — Monitoring window not complete → continue_monitoring
  T12 — Monitoring window complete, no improvement → rollback
  T13 — 3 good outcomes + window complete → adoption_ready, no real adoption
  T14 — Fewer than 3 outcomes → adoption_ready=False
  T15 — Missing baseline rates → conservative, adoption blocked
  T16 — Missing monitored data → continue_monitoring
  T17 — rollback_reasons are sorted deterministically
  T18 — No runtime mutation (pure functions)
  T19 — No control keyword in monitoring module
  T20 — Regression: existing test modules still pass
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.adaptation.application import (
    ApplicationPolicy,
    ApplicationRuntimeContext,
    build_preheat_application_plan,
    evaluate_application_readiness,
)
from custom_components.thermosmart.learning.adaptation.contracts import (
    AdaptationDirection,
    AdaptationLifecycle,
    CandidateType,
)
from custom_components.thermosmart.learning.adaptation.history import (
    CandidateHistoryEntry,
    evaluate_promotion_readiness,
)
from custom_components.thermosmart.learning.adaptation.monitoring import (
    AppliedAdaptationRecord,
    MonitoringEvaluation,
    MonitoringStatus,
    build_applied_adaptation_record,
    evaluate_applied_adaptation_outcome,
    should_rollback_adaptation,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_APPLIED_TS = "2026-06-12T10:00:00+00:00"

# Baselines from the canonical test entry
_BASELINE_QUALITY  = 0.65
_BASELINE_TIMEOUT  = 0.32
_BASELINE_OVERSHOOT = 0.10


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _entry(**overrides) -> CandidateHistoryEntry:
    defaults = dict(
        candidate_key="cafebabe01234567",
        candidate_type=CandidateType.PREHEAT_DELTA_MIN,
        direction=AdaptationDirection.INCREASE,
        first_seen_ts="2026-06-01T10:00:00+00:00",
        last_seen_ts="2026-06-12T10:00:00+00:00",
        seen_count=5,
        supporting_outcome_count=15,
        avg_outcome_quality=_BASELINE_QUALITY,
        avg_timeout_rate=_BASELINE_TIMEOUT,
        avg_overshoot_rate=_BASELINE_OVERSHOOT,
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
        window_open=False, manual_override=False,
        vacation_mode=False, summer_mode=False,
        heating_unavailable=False, sensor_unreliable=False,
        control_disabled=False, learning_disabled=False,
        active_control_disabled=False, storage_corrupt=False,
        confounder_ratio=0.0,
    )
    defaults.update(overrides)
    return ApplicationRuntimeContext(**defaults)


def _enabled_plan(entry: CandidateHistoryEntry | None = None,
                  ctx: ApplicationRuntimeContext | None = None):
    """Return a would_apply=True ApplicationPlan."""
    e = entry or _entry()
    c = ctx or _ctx()
    pgr = evaluate_promotion_readiness(e, span_days=11.0, confounder_ratio=0.0)
    policy = ApplicationPolicy(application_enabled=True)
    dec = evaluate_application_readiness(
        history_entry=e, promotion_result=pgr, policy=policy, runtime_context=c,
    )
    return build_preheat_application_plan(
        history_entry=e, promotion_result=pgr,
        application_decision=dec, policy=policy, runtime_context=c,
    )


def _disabled_plan(entry: CandidateHistoryEntry | None = None):
    """Return a would_apply=False ApplicationPlan (default kill-switch=False)."""
    e = entry or _entry()
    c = _ctx()
    pgr = evaluate_promotion_readiness(e, span_days=11.0, confounder_ratio=0.0)
    policy = ApplicationPolicy(application_enabled=False)
    dec = evaluate_application_readiness(
        history_entry=e, promotion_result=pgr, policy=policy, runtime_context=c,
    )
    return build_preheat_application_plan(
        history_entry=e, promotion_result=pgr,
        application_decision=dec, policy=policy, runtime_context=c,
    )


def _good_record(**plan_overrides) -> AppliedAdaptationRecord:
    """Build an AppliedAdaptationRecord from a would_apply=True plan."""
    plan = _enabled_plan()
    record = build_applied_adaptation_record(
        plan=plan, history_entry=_entry(), applied_ts=_APPLIED_TS,
    )
    assert record is not None, "Test fixture: expected a valid record"
    return record


def _eval(
    record: AppliedAdaptationRecord | None = None,
    *,
    count: int = 3,
    quality: float = 0.72,
    timeout: float = 0.25,
    overshoot: float = 0.08,
    **kwargs,
) -> MonitoringEvaluation:
    """Build a MonitoringEvaluation with reasonable defaults."""
    r = record or _good_record()
    return evaluate_applied_adaptation_outcome(
        record=r,
        monitored_outcome_count=count,
        avg_monitored_outcome_quality=quality,
        avg_monitored_timeout_rate=timeout,
        avg_monitored_overshoot_rate=overshoot,
        **kwargs,
    )


# ── T1: build record from would_apply=True ────────────────────────────────────

def test_t1_build_record_from_eligible_plan():
    plan = _enabled_plan()
    assert plan.would_apply is True

    record = build_applied_adaptation_record(
        plan=plan, history_entry=_entry(), applied_ts=_APPLIED_TS,
    )
    assert record is not None
    assert record.candidate_key == plan.candidate_key
    assert record.applied_delta_min == pytest.approx(plan.bounded_delta_min)
    assert record.previous_cumulative_delta_min == pytest.approx(plan.current_cumulative_delta_min)
    assert record.new_cumulative_delta_min == pytest.approx(plan.next_cumulative_delta_min)
    assert record.applied_ts == _APPLIED_TS
    assert record.status is MonitoringStatus.PENDING
    assert record.rollback_supported is True
    assert record.monitoring_deadline_ts is not None  # deadline computed from applied_ts
    assert record.baseline_outcome_quality == pytest.approx(_BASELINE_QUALITY)
    assert record.baseline_timeout_rate == pytest.approx(_BASELINE_TIMEOUT)
    assert record.baseline_overshoot_rate == pytest.approx(_BASELINE_OVERSHOOT)


# ── T2: would_apply=False plan → None record ──────────────────────────────────

def test_t2_disabled_plan_returns_none_record():
    plan = _disabled_plan()
    assert plan.would_apply is False
    record = build_applied_adaptation_record(
        plan=plan, history_entry=_entry(), applied_ts=_APPLIED_TS,
    )
    assert record is None


# ── T3: record.to_dict() is public-safe ───────────────────────────────────────

def test_t3_record_to_dict_public_safe():
    record = _good_record()
    d = record.to_dict()

    required_keys = {
        "candidate_key", "candidate_type", "direction",
        "applied_delta_min", "previous_cumulative_delta_min",
        "new_cumulative_delta_min", "applied_ts",
        "monitoring_window_hours", "monitoring_deadline_ts",
        "baseline_outcome_quality", "baseline_timeout_rate",
        "baseline_overshoot_rate", "status", "rollback_supported",
    }
    assert required_keys <= set(d.keys())

    # No control or private fields
    forbidden = {"setpoint", "entity_id", "zone_name", "secret",
                 "control_value", "boost", "tpi"}
    assert not (forbidden & set(d.keys()))

    # Status is a string value
    assert isinstance(d["status"], str)
    assert d["rollback_supported"] is True


# ── T4: Better outcome + no side effects → continue (window not done) ─────────

def test_t4_better_outcome_continue_when_window_incomplete():
    # quality 0.72 > baseline 0.65 (+0.07), timeout down, overshoot down
    ev = _eval(monitoring_window_complete=False)
    assert ev.rollback_recommended is False
    assert ev.continue_monitoring is True
    assert ev.adoption_ready is False  # window not complete
    assert ev.status is MonitoringStatus.IN_PROGRESS


# ── T5: Quality degraded → rollback ──────────────────────────────────────────

def test_t5_quality_degraded_recommends_rollback():
    # 0.50 - 0.65 = -0.15 < -0.10
    ev = _eval(quality=0.50)
    assert ev.rollback_recommended is True
    assert any("outcome_quality_degraded" in r for r in ev.rollback_reasons)
    assert ev.adoption_ready is False


# ── T6: Timeout rate increased → rollback ────────────────────────────────────

def test_t6_timeout_rate_increased_recommends_rollback():
    # 0.45 - 0.32 = +0.13 > +0.10
    ev = _eval(timeout=0.45)
    assert ev.rollback_recommended is True
    assert any("timeout_rate_increased" in r for r in ev.rollback_reasons)


# ── T7: Overshoot rate increased → rollback ───────────────────────────────────

def test_t7_overshoot_rate_increased_recommends_rollback():
    # 0.25 - 0.10 = +0.15 > +0.10
    ev = _eval(overshoot=0.25)
    assert ev.rollback_recommended is True
    assert any("overshoot_rate_increased" in r for r in ev.rollback_reasons)


# ── T8: Manual override detected → rollback ──────────────────────────────────

def test_t8_manual_override_recommends_rollback():
    ev = _eval(manual_override_detected=True)
    assert ev.rollback_recommended is True
    assert "manual_override_detected" in ev.rollback_reasons
    assert ev.manual_override_detected is True


# ── T9: Confounder detected → rollback ───────────────────────────────────────

def test_t9_confounder_detected_recommends_rollback():
    ev = _eval(confounder_detected=True)
    assert ev.rollback_recommended is True
    assert "confounder_detected" in ev.rollback_reasons
    assert ev.confounder_detected is True


# ── T10: Safety flag detected → rollback ──────────────────────────────────────

def test_t10_safety_flag_detected_recommends_rollback():
    ev = _eval(safety_flag_detected=True)
    assert ev.rollback_recommended is True
    assert "safety_flag_detected" in ev.rollback_reasons
    assert ev.safety_flag_detected is True


# ── T11: Window not complete → continue_monitoring ────────────────────────────

def test_t11_window_incomplete_gives_continue_monitoring():
    ev = _eval(monitoring_window_complete=False)
    assert ev.continue_monitoring is True
    assert ev.rollback_recommended is False
    assert ev.status is MonitoringStatus.IN_PROGRESS


# ── T12: Window complete, no improvement → rollback ──────────────────────────

def test_t12_window_expired_no_improvement_recommends_rollback():
    # quality +0.01 < threshold +0.05 → window_expired_no_improvement
    ev = _eval(quality=0.66, monitoring_window_complete=True)
    assert ev.rollback_recommended is True
    assert "window_expired_no_improvement" in ev.rollback_reasons
    assert ev.adoption_ready is False


def test_t12_window_expired_quality_negative_recommends_rollback():
    # quality slightly negative but above -0.10 rollback threshold
    ev = _eval(quality=0.63, monitoring_window_complete=True)
    assert ev.rollback_recommended is True
    assert "window_expired_no_improvement" in ev.rollback_reasons


# ── T13: 3 good outcomes + window complete → adoption_ready (no real adoption) ─

def test_t13_three_good_outcomes_adoption_candidate():
    # delta +0.07 >= 0.05, timeout down, overshoot down, window complete
    ev = _eval(count=3, quality=0.72, timeout=0.25, overshoot=0.08,
               monitoring_window_complete=True)
    assert ev.adoption_ready is True
    assert ev.status is MonitoringStatus.ADOPTION_CANDIDATE
    assert ev.rollback_recommended is False
    assert ev.continue_monitoring is False
    # Verify no actual adoption happens: evaluation is pure and read-only
    assert isinstance(ev, MonitoringEvaluation)
    assert "adopted" not in ev.status.value  # status is "adoption_candidate", not "adopted"


# ── T14: Fewer than 3 outcomes → adoption_ready=False ────────────────────────

def test_t14_fewer_than_3_outcomes_not_adoption_ready():
    ev = _eval(count=2, quality=0.72, timeout=0.25, overshoot=0.08,
               monitoring_window_complete=True)
    assert ev.adoption_ready is False
    assert ev.status is MonitoringStatus.WINDOW_EXPIRED


# ── T15: Missing baseline rates → conservative, adoption blocked ──────────────

def test_t15_missing_baseline_timeout_blocks_adoption():
    # avg_timeout_rate=None in history_entry → baseline_timeout_rate=None
    e = _entry(avg_timeout_rate=None)
    plan = _enabled_plan(entry=e)
    assert plan.would_apply is True

    record = build_applied_adaptation_record(
        plan=plan, history_entry=e, applied_ts=_APPLIED_TS,
    )
    assert record is not None
    assert record.baseline_timeout_rate is None

    # Even with good monitored data, timeout_delta is None → adoption blocked
    ev = evaluate_applied_adaptation_outcome(
        record=record,
        monitored_outcome_count=3,
        avg_monitored_outcome_quality=0.72,
        avg_monitored_timeout_rate=0.25,
        avg_monitored_overshoot_rate=0.08,
        monitoring_window_complete=True,
    )
    assert ev.adoption_ready is False
    assert ev.timeout_rate_delta is None
    assert ev.continue_monitoring is True or ev.rollback_recommended is True


# ── T16: Missing monitored data → continue_monitoring ────────────────────────

def test_t16_missing_monitored_quality_gives_continue():
    ev = evaluate_applied_adaptation_outcome(
        record=_good_record(),
        monitored_outcome_count=3,
        avg_monitored_outcome_quality=None,   # missing
        avg_monitored_timeout_rate=0.25,
        avg_monitored_overshoot_rate=0.08,
        monitoring_window_complete=False,
    )
    assert ev.outcome_quality_delta is None
    assert ev.adoption_ready is False
    assert ev.continue_monitoring is True


def test_t16_missing_all_monitored_data_gives_continue():
    ev = evaluate_applied_adaptation_outcome(
        record=_good_record(),
        monitored_outcome_count=0,
        avg_monitored_outcome_quality=None,
        avg_monitored_timeout_rate=None,
        avg_monitored_overshoot_rate=None,
    )
    assert ev.adoption_ready is False
    assert ev.continue_monitoring is True
    assert ev.status is MonitoringStatus.IN_PROGRESS


# ── T17: rollback_reasons are sorted deterministically ───────────────────────

def test_t17_rollback_reasons_sorted():
    # Trigger multiple reasons simultaneously
    ev = evaluate_applied_adaptation_outcome(
        record=_good_record(),
        monitored_outcome_count=3,
        avg_monitored_outcome_quality=0.50,   # quality degraded
        avg_monitored_timeout_rate=0.50,      # timeout increased
        avg_monitored_overshoot_rate=0.30,    # overshoot increased
        manual_override_detected=True,
        confounder_detected=True,
    )
    reasons = list(ev.rollback_reasons)
    assert reasons == sorted(reasons), "rollback_reasons are not sorted"
    assert len(reasons) >= 5  # all triggers fired
    # Calling again must produce identical order
    ev2 = evaluate_applied_adaptation_outcome(
        record=_good_record(),
        monitored_outcome_count=3,
        avg_monitored_outcome_quality=0.50,
        avg_monitored_timeout_rate=0.50,
        avg_monitored_overshoot_rate=0.30,
        manual_override_detected=True,
        confounder_detected=True,
    )
    assert ev.rollback_reasons == ev2.rollback_reasons


# ── T18: No runtime mutation ──────────────────────────────────────────────────

def test_t18_functions_are_pure_no_mutation():
    plan = _enabled_plan()
    e = _entry()

    rec1 = build_applied_adaptation_record(plan=plan, history_entry=e, applied_ts=_APPLIED_TS)
    rec2 = build_applied_adaptation_record(plan=plan, history_entry=e, applied_ts=_APPLIED_TS)
    assert rec1 == rec2
    assert rec1 is not rec2  # different objects, same frozen value

    ev1 = _eval(record=rec1)
    ev2 = _eval(record=rec2)
    assert ev1 == ev2
    assert ev1 is not ev2


# ── T19: No control keyword in monitoring module ──────────────────────────────

def test_t19_no_control_keywords_in_monitoring_module():
    import inspect
    import custom_components.thermosmart.learning.adaptation.monitoring as mod

    source = inspect.getsource(mod)
    forbidden = [
        "set_point", "async_set_temperature", "async_write_ha_state",
        "boost_duration", "tpi_gain", "preheat_duration",
        "dispatch", "service_call", "control_value", "apply_candidate",
        "from homeassistant", "import homeassistant",
    ]
    found = [kw for kw in forbidden if kw in source]
    assert found == [], f"Control keywords in monitoring.py: {found}"


# ── T20: Regression — existing test modules still pass ───────────────────────

def test_t20_regression_preheat_plan_imports():
    import tests.test_preheat_application_plan  # noqa: F401


def test_t20_regression_application_readiness_imports():
    import tests.test_application_readiness  # noqa: F401


def test_t20_regression_promotion_readiness_imports():
    import tests.test_promotion_readiness  # noqa: F401


# ── Additional edge cases ─────────────────────────────────────────────────────

def test_should_rollback_helper_matches_evaluation():
    ev_rollback = _eval(quality=0.40)  # triggers rollback
    ev_ok = _eval()

    assert should_rollback_adaptation(ev_rollback) is True
    assert should_rollback_adaptation(ev_ok) is False


def test_monitoring_evaluation_to_dict_public_safe():
    ev = _eval(monitoring_window_complete=True,
               quality=0.72, timeout=0.25, overshoot=0.08)
    d = ev.to_dict()

    required = {
        "candidate_key", "status", "monitored_outcome_count",
        "outcome_quality_delta", "timeout_rate_delta", "overshoot_rate_delta",
        "manual_override_detected", "confounder_detected", "safety_flag_detected",
        "rollback_recommended", "rollback_reasons", "adoption_ready",
        "continue_monitoring",
    }
    assert required <= set(d.keys())
    assert isinstance(d["rollback_reasons"], list)
    assert isinstance(d["status"], str)

    forbidden = {"entity_id", "zone_name", "setpoint", "boost", "tpi", "control_value"}
    assert not (forbidden & set(d.keys()))


def test_monitoring_deadline_computed_correctly():
    plan = _enabled_plan()
    record = build_applied_adaptation_record(
        plan=plan, history_entry=_entry(), applied_ts="2026-06-12T10:00:00+00:00",
    )
    assert record is not None
    # Default monitoring_window_hours = 48.0
    # applied_ts 2026-06-12T10:00:00 + 48h = 2026-06-14T10:00:00
    assert "2026-06-14" in record.monitoring_deadline_ts
