"""Tests for build_preheat_application_plan() and ApplicationPlan.

26 tests covering:
  T1  — PREHEAT eligible + application_enabled=False → blocked (application_disabled)
  T2  — PREHEAT eligible + application_enabled=True + all gates green → would_apply=True
  T3  — step delta is bounded to ±5 min (bounded_delta_min capped)
  T4  — cumulative delta stays ≤ ±15 min when plan is valid
  T5  — cumulative limit exceeded → plan blocks
  T6  — cooldown active → blocked
  T7  — not-eligible promotion → blocked
  T8  — non-SHADOW lifecycle → blocked
  T9  — unsupported candidate type passed to plan → plan_type_not_supported
  T10 — EARLY_CUTOFF_DELTA_MIN remains blocked in this step
  T11 — BOOST_BIAS remains blocked
  T12 — TPI_GAIN_BIAS remains blocked
  T13 — COMFORT_BIAS_C remains blocked
  T14 — missing runtime context → blocked
  T15 — missing requested delta (proposed_delta=None) → blocked
  T16 — NEUTRAL direction → unsupported_direction
  T17 — safety flag window_open → blocked
  T18 — safety flag manual_override → blocked
  T19 — safety flags (vacation/summer/control/learning/active_control) → blocked
  T20 — plan contains monitoring_required=True
  T21 — plan contains rollback_supported=True
  T22 — plan is public-safe (no entity ids, secrets, control fields)
  T23 — plan produces no runtime mutation (pure function)
  T24 — no control keyword in application module
  T25 — regression: test_application_readiness.py still passes (import smoke-test)
  T26 — direction_delta_mismatch: INCREASE with negative delta → blocked
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.adaptation.application import (
    ApplicationDecision,
    ApplicationPlan,
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
    PromotionGateResult,
    evaluate_promotion_readiness,
)

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


def _pgr(entry: CandidateHistoryEntry | None = None) -> PromotionGateResult:
    e = entry or _entry()
    return evaluate_promotion_readiness(e, span_days=11.0, confounder_ratio=0.0)


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


def _decision(
    entry: CandidateHistoryEntry | None = None,
    policy: ApplicationPolicy | None = None,
    runtime_context: ApplicationRuntimeContext | None = None,
) -> ApplicationDecision:
    e = entry or _entry()
    pgr = _pgr(e)
    return evaluate_application_readiness(
        history_entry=e,
        promotion_result=pgr,
        policy=policy,
        runtime_context=runtime_context,
    )


def _plan(
    entry: CandidateHistoryEntry | None = None,
    policy: ApplicationPolicy | None = None,
    runtime_context: ApplicationRuntimeContext | None = _ctx(),
) -> ApplicationPlan:
    """Build a plan using consistent decision + context."""
    e = entry or _entry()
    pgr = _pgr(e)
    ctx = runtime_context
    dec = evaluate_application_readiness(
        history_entry=e,
        promotion_result=pgr,
        policy=policy,
        runtime_context=ctx,
    )
    return build_preheat_application_plan(
        history_entry=e,
        promotion_result=pgr,
        application_decision=dec,
        policy=policy,
        runtime_context=ctx,
    )


# ── T1: kill-switch blocks (default runtime) ─────────────────────────────────

def test_t1_kill_switch_blocks_in_real_runtime():
    plan = _plan()
    assert plan.application_enabled is False
    assert plan.would_apply is False
    assert "application_disabled" in plan.blocking_reasons


# ── T2: would_apply=True with explicit application_enabled=True ───────────────

def test_t2_would_apply_true_when_enabled_and_all_gates_green():
    enabled_policy = ApplicationPolicy(application_enabled=True)
    ctx = _ctx()
    plan = _plan(policy=enabled_policy, runtime_context=ctx)
    assert plan.would_apply is True
    assert plan.application_enabled is True
    # Only application_disabled is absent → no other blocks
    non_app = [r for r in plan.blocking_reasons if r != "application_disabled"]
    assert non_app == [], f"Unexpected blocking reasons: {non_app}"


# ── T3: step delta bounded at ±5 min ─────────────────────────────────────────

def test_t3_step_delta_capped_to_max():
    ctx = _ctx(proposed_delta=9.0)   # exceeds 5.0 max
    plan = _plan(runtime_context=ctx)
    # bounded_delta must be capped at 5.0 regardless of blocking
    assert plan.bounded_delta_min == pytest.approx(5.0)
    assert plan.requested_delta_min == pytest.approx(9.0)


def test_t3_step_delta_at_limit_not_capped():
    ctx = _ctx(proposed_delta=5.0)
    plan = _plan(runtime_context=ctx)
    assert plan.bounded_delta_min == pytest.approx(5.0)


def test_t3_negative_step_delta_bounded_correctly():
    e = _entry(direction=AdaptationDirection.DECREASE)
    ctx = _ctx(proposed_delta=-8.0)
    plan = _plan(entry=e, runtime_context=ctx)
    assert plan.bounded_delta_min == pytest.approx(-5.0)


# ── T4: cumulative stays ≤ ±15 min in valid plan ─────────────────────────────

def test_t4_cumulative_within_limit_on_valid_plan():
    enabled_policy = ApplicationPolicy(application_enabled=True)
    ctx = _ctx(proposed_delta=4.0, current_cumulative_delta=9.0)
    plan = _plan(policy=enabled_policy, runtime_context=ctx)
    # bounded=4.0, current=9.0 → next=13.0 ≤ 15.0 → no cumulative block
    assert plan.next_cumulative_delta_min == pytest.approx(13.0)
    assert plan.would_apply is True
    assert not any(r.startswith("plan_cumulative_exceeds_limit") for r in plan.blocking_reasons)


# ── T5: cumulative limit exceeded → plan blocks ───────────────────────────────

def test_t5_cumulative_exceeded_blocks_plan():
    ctx = _ctx(proposed_delta=5.0, current_cumulative_delta=12.0)
    plan = _plan(runtime_context=ctx)
    # bounded=5.0, current=12.0 → next=17.0 > 15.0
    assert any(r.startswith("plan_cumulative_exceeds_limit:") for r in plan.blocking_reasons)
    assert plan.would_apply is False


# ── T6: cooldown active → blocked ────────────────────────────────────────────

def test_t6_active_cooldown_blocks():
    ctx = _ctx(
        last_applied_ts="2026-06-09T10:00:00+00:00",
        now_ts="2026-06-12T10:00:00+00:00",   # 3 days < 7 day cooldown
    )
    plan = _plan(runtime_context=ctx)
    assert any(r.startswith("cooldown_active:") for r in plan.blocking_reasons)
    assert plan.would_apply is False


# ── T7: not-eligible promotion → blocked ─────────────────────────────────────

def test_t7_not_eligible_promotion_blocks_plan():
    thin = _entry(supporting_outcome_count=2)  # too few samples → BLOCKED readiness
    pgr = _pgr(thin)
    dec = _decision(entry=thin, runtime_context=_ctx())
    plan = build_preheat_application_plan(
        history_entry=thin,
        promotion_result=pgr,
        application_decision=dec,
        runtime_context=_ctx(),
    )
    assert any(r.startswith("not_eligible:") for r in plan.blocking_reasons)
    assert plan.would_apply is False


# ── T8: non-SHADOW lifecycle → blocked ───────────────────────────────────────

def test_t8_non_shadow_lifecycle_blocks():
    e = _entry(last_lifecycle=AdaptationLifecycle.CANDIDATE)
    pgr = _pgr(e)
    dec = _decision(entry=e, runtime_context=_ctx())
    plan = build_preheat_application_plan(
        history_entry=e,
        promotion_result=pgr,
        application_decision=dec,
        runtime_context=_ctx(),
    )
    assert any(r.startswith("lifecycle_not_shadow:") for r in plan.blocking_reasons)
    assert plan.would_apply is False


# ── T9: unsupported type passed to plan function ──────────────────────────────

def test_t9_unsupported_type_in_plan_blocks():
    e = _entry(candidate_type=CandidateType.EARLY_CUTOFF_DELTA_MIN)
    pgr = _pgr(e)
    dec = _decision(entry=e, runtime_context=_ctx())
    plan = build_preheat_application_plan(
        history_entry=e,
        promotion_result=pgr,
        application_decision=dec,
        runtime_context=_ctx(),
    )
    assert any(r.startswith("plan_type_not_supported:") for r in plan.blocking_reasons)
    assert plan.would_apply is False


# ── T10–T13: specific types remain blocked ───────────────────────────────────

@pytest.mark.parametrize("ctype", [
    CandidateType.EARLY_CUTOFF_DELTA_MIN,
    CandidateType.BOOST_BIAS,
    CandidateType.TPI_GAIN_BIAS,
    CandidateType.COMFORT_BIAS_C,
])
def test_t10_t13_non_preheat_types_blocked(ctype):
    e = _entry(candidate_type=ctype)
    pgr = _pgr(e)
    dec = _decision(entry=e, runtime_context=_ctx())
    plan = build_preheat_application_plan(
        history_entry=e,
        promotion_result=pgr,
        application_decision=dec,
        runtime_context=_ctx(),
    )
    assert plan.would_apply is False
    # Either plan_type_not_supported or type_not_allowed should appear
    has_type_block = any(
        r.startswith("plan_type_not_supported:") or r.startswith("type_not_allowed:")
        for r in plan.blocking_reasons
    )
    assert has_type_block, f"Expected type block for {ctype.value}: {plan.blocking_reasons}"


# ── T14: missing runtime context → blocked ───────────────────────────────────

def test_t14_missing_runtime_context_blocks():
    e = _entry()
    pgr = _pgr(e)
    dec = _decision(entry=e, runtime_context=None)
    plan = build_preheat_application_plan(
        history_entry=e,
        promotion_result=pgr,
        application_decision=dec,
        runtime_context=None,
    )
    assert plan.would_apply is False
    assert "missing_requested_delta" in plan.blocking_reasons


# ── T15: missing proposed_delta → blocked ────────────────────────────────────

def test_t15_missing_proposed_delta_blocks():
    ctx = _ctx(proposed_delta=None)
    plan = _plan(runtime_context=ctx)
    assert "missing_requested_delta" in plan.blocking_reasons
    assert plan.bounded_delta_min is None
    assert plan.would_apply is False


# ── T16: NEUTRAL direction → unsupported_direction ───────────────────────────

def test_t16_neutral_direction_blocked():
    e = _entry(direction=AdaptationDirection.NEUTRAL)
    ctx = _ctx()
    plan = _plan(entry=e, runtime_context=ctx)
    assert any(r.startswith("unsupported_direction:") for r in plan.blocking_reasons)
    assert plan.would_apply is False


# ── T17: safety flag window_open → blocked ───────────────────────────────────

def test_t17_window_open_blocks():
    ctx = _ctx(window_open=True)
    plan = _plan(runtime_context=ctx)
    assert "safety_window_open" in plan.blocking_reasons
    assert plan.would_apply is False


# ── T18: safety flag manual_override → blocked ───────────────────────────────

def test_t18_manual_override_blocks():
    ctx = _ctx(manual_override=True)
    plan = _plan(runtime_context=ctx)
    assert "safety_manual_override" in plan.blocking_reasons
    assert plan.would_apply is False


# ── T19: remaining safety flags each block ───────────────────────────────────

@pytest.mark.parametrize("flag", [
    "vacation_mode",
    "summer_mode",
    "control_disabled",
    "learning_disabled",
    "active_control_disabled",
])
def test_t19_safety_flags_block(flag):
    ctx = _ctx(**{flag: True})
    plan = _plan(runtime_context=ctx)
    assert f"safety_{flag}" in plan.blocking_reasons
    assert plan.would_apply is False


# ── T20: monitoring_required=True ────────────────────────────────────────────

def test_t20_monitoring_required_always_true():
    plan = _plan()
    assert plan.monitoring_required is True


# ── T21: rollback_supported=True ─────────────────────────────────────────────

def test_t21_rollback_supported_always_true():
    plan = _plan()
    assert plan.rollback_supported is True


# ── T22: plan is public-safe ─────────────────────────────────────────────────

def test_t22_plan_to_dict_is_public_safe():
    enabled_policy = ApplicationPolicy(application_enabled=True)
    plan = _plan(policy=enabled_policy, runtime_context=_ctx())
    d = plan.to_dict()

    # Required keys present
    required = {
        "candidate_key", "candidate_type", "direction",
        "requested_delta_min", "bounded_delta_min",
        "current_cumulative_delta_min", "next_cumulative_delta_min",
        "would_apply", "application_enabled", "blocking_reasons",
        "monitoring_required", "rollback_supported",
        "monitoring_window_hours", "rollback_reason",
    }
    assert required <= set(d.keys()), f"Missing keys: {required - set(d.keys())}"

    # No control-side or private fields
    forbidden = {"setpoint", "boost", "tpi", "dispatch", "entity_id",
                 "zone_name", "secret", "path", "control_value"}
    assert not (forbidden & set(d.keys())), "Forbidden fields in to_dict()"

    # application_enabled always False in real runtime context
    # (here we used enabled_policy so it may be True in this test only)
    # Verify it reflects the policy value
    assert isinstance(d["application_enabled"], bool)

    # blocking_reasons is a list
    assert isinstance(d["blocking_reasons"], list)

    # rollback_reason is None (plan phase)
    assert d["rollback_reason"] is None


# ── T23: plan produces no runtime mutation ───────────────────────────────────

def test_t23_plan_is_pure_no_mutation():
    ctx = _ctx()
    e = _entry()
    pgr = _pgr(e)
    dec = _decision(entry=e, runtime_context=ctx)

    # Call multiple times — must return identical frozen results
    plan_a = build_preheat_application_plan(
        history_entry=e, promotion_result=pgr,
        application_decision=dec, runtime_context=ctx,
    )
    plan_b = build_preheat_application_plan(
        history_entry=e, promotion_result=pgr,
        application_decision=dec, runtime_context=ctx,
    )
    assert plan_a == plan_b   # frozen dataclass equality
    assert plan_a is not plan_b  # different objects, same values


# ── T24: no control keyword in application module ────────────────────────────

def test_t24_no_control_keywords_in_module():
    import inspect
    import custom_components.thermosmart.learning.adaptation.application as mod

    source = inspect.getsource(mod)
    forbidden = [
        "set_point", "async_set_temperature", "async_write_ha_state",
        "boost_duration", "tpi_gain", "preheat_duration",
        "dispatch", "service_call", "control_value", "apply_candidate",
        "from homeassistant", "import homeassistant",
    ]
    found = [kw for kw in forbidden if kw in source]
    assert found == [], f"Control keywords found in application.py: {found}"


# ── T25: regression — existing test modules import cleanly ───────────────────

def test_t25_regression_application_readiness_imports():
    import tests.test_application_readiness  # noqa: F401


def test_t25_regression_promotion_readiness_imports():
    import tests.test_promotion_readiness  # noqa: F401


# ── T26: direction_delta_mismatch blocks ─────────────────────────────────────

def test_t26_direction_delta_mismatch_increase_but_negative():
    # INCREASE direction but negative proposed_delta
    e = _entry(direction=AdaptationDirection.INCREASE)
    ctx = _ctx(proposed_delta=-3.0)
    plan = _plan(entry=e, runtime_context=ctx)
    assert "direction_delta_mismatch:increase_but_negative" in plan.blocking_reasons
    assert plan.bounded_delta_min is None
    assert plan.would_apply is False


def test_t26_direction_delta_mismatch_decrease_but_positive():
    # DECREASE direction but positive proposed_delta
    e = _entry(direction=AdaptationDirection.DECREASE)
    ctx = _ctx(proposed_delta=3.0)
    plan = _plan(entry=e, runtime_context=ctx)
    assert "direction_delta_mismatch:decrease_but_positive" in plan.blocking_reasons
    assert plan.bounded_delta_min is None
    assert plan.would_apply is False
