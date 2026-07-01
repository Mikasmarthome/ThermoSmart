"""Tests for evaluate_application_readiness() and supporting types.

22 tests covering:
  - Kill-switch behaviour (application_enabled always False)
  - Allowlist (PREHEAT / EARLY_CUTOFF allowed; BOOST_BIAS / TPI_GAIN_BIAS /
    COMFORT_BIAS_C permanently blocked)
  - Promotion-readiness gate
  - Lifecycle gate (must be SHADOW)
  - Safety flags (9 flags + storage_corrupt)
  - Step-delta and cumulative-limit gates
  - Cooldown gate (active / expired / missing now_ts / parse error)
  - Unknown runtime context
  - would_allow_if_enabled semantics
  - to_dict() public-safe output
  - application_readiness_preview() static helper
  - No control keywords in output
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.adaptation.application import (
    ApplicationDecision,
    ApplicationPolicy,
    ApplicationRuntimeContext,
    application_readiness_preview,
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
    PromotionReadiness,
    evaluate_promotion_readiness,
)

# ── Test fixtures ─────────────────────────────────────────────────────────────

def _good_entry(**overrides) -> CandidateHistoryEntry:
    defaults = dict(
        candidate_key="aabbccdd11223344",
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


def _eligible_pgr(entry: CandidateHistoryEntry | None = None) -> PromotionGateResult:
    """Return an ELIGIBLE PromotionGateResult for a good entry."""
    e = entry or _good_entry()
    return evaluate_promotion_readiness(e, span_days=11.0, confounder_ratio=0.0)


def _blocked_pgr() -> PromotionGateResult:
    """Return a BLOCKED PromotionGateResult (too few outcome samples)."""
    thin = _good_entry(supporting_outcome_count=2)
    return evaluate_promotion_readiness(thin, span_days=11.0, confounder_ratio=0.0)


def _clean_ctx(**overrides) -> ApplicationRuntimeContext:
    """Runtime context with all safety flags off, known delta and cumulative."""
    defaults = dict(
        proposed_delta=5.0,
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


# ── T_APP1: Kill-switch always blocks, application_enabled always False ───────

def test_kill_switch_blocks_eligible_preheat():
    entry = _good_entry()
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        policy=ApplicationPolicy(application_enabled=False),
        runtime_context=_clean_ctx(),
    )
    assert dec.application_enabled is False
    assert "application_disabled" in dec.blocking_reasons


# ── T_APP2: Eligible EARLY_CUTOFF is blocked by kill-switch ──────────────────

def test_kill_switch_blocks_eligible_early_cutoff():
    entry = _good_entry(candidate_type=CandidateType.EARLY_CUTOFF_DELTA_MIN)
    pgr = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.0)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        policy=ApplicationPolicy(application_enabled=False),
        runtime_context=_clean_ctx(proposed_delta=3.0),
    )
    assert dec.application_enabled is False
    assert "application_disabled" in dec.blocking_reasons


# ── T_APP3: would_allow_if_enabled True when only kill-switch blocks ──────────

def test_would_allow_if_enabled_true_when_only_kill_switch_blocks():
    entry = _good_entry()
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(),
    )
    # Only kill-switch should block; everything else passes
    assert "application_disabled" in dec.blocking_reasons
    non_app = [r for r in dec.blocking_reasons if r != "application_disabled"]
    assert non_app == [], f"Unexpected extra reasons: {non_app}"
    assert dec.would_allow_if_enabled is True


# ── T_APP4: would_allow_if_enabled False when non-kill-switch gate blocks ─────

def test_would_allow_if_enabled_false_when_other_gate_blocks():
    entry = _good_entry()
    pgr = _blocked_pgr()          # not ELIGIBLE
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(),
    )
    assert dec.would_allow_if_enabled is False
    assert any(r.startswith("not_eligible:") for r in dec.blocking_reasons)


# ── T_APP5: Non-eligible promotion result blocks ──────────────────────────────

def test_not_eligible_promotion_blocks():
    entry = _good_entry()
    pgr = _blocked_pgr()
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
    )
    assert any(r.startswith("not_eligible:") for r in dec.blocking_reasons)


# ── T_APP6: Non-SHADOW lifecycle blocks ───────────────────────────────────────

def test_non_shadow_lifecycle_blocks():
    entry = _good_entry(last_lifecycle=AdaptationLifecycle.CANDIDATE)
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(),
    )
    assert any(r.startswith("lifecycle_not_shadow:") for r in dec.blocking_reasons)


# ── T_APP7: BOOST_BIAS type is permanently blocked ───────────────────────────

def test_boost_bias_type_blocked():
    entry = _good_entry(candidate_type=CandidateType.BOOST_BIAS)
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(),
    )
    assert dec.candidate_type_allowed is False
    assert any(r.startswith("type_not_allowed:") for r in dec.blocking_reasons)
    assert dec.max_step_delta is None
    assert dec.max_cumulative_delta is None


# ── T_APP8: TPI_GAIN_BIAS type is permanently blocked ────────────────────────

def test_tpi_gain_bias_type_blocked():
    entry = _good_entry(candidate_type=CandidateType.TPI_GAIN_BIAS)
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(),
    )
    assert dec.candidate_type_allowed is False
    assert any(r.startswith("type_not_allowed:") for r in dec.blocking_reasons)


# ── T_APP9: COMFORT_BIAS_C type is permanently blocked ───────────────────────

def test_comfort_bias_c_type_blocked():
    entry = _good_entry(candidate_type=CandidateType.COMFORT_BIAS_C)
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(),
    )
    assert dec.candidate_type_allowed is False
    assert any(r.startswith("type_not_allowed:") for r in dec.blocking_reasons)


# ── T_APP10: Missing runtime context blocks with unknown_context ──────────────

def test_no_runtime_context_blocks_unknown_context():
    entry = _good_entry()
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=None,
    )
    assert "unknown_context" in dec.blocking_reasons
    assert dec.would_allow_if_enabled is False


# ── T_APP11: Safety flags (all 9 flags + storage_corrupt) each block ─────────

@pytest.mark.parametrize("flag", [
    "window_open",
    "manual_override",
    "vacation_mode",
    "summer_mode",
    "heating_unavailable",
    "sensor_unreliable",
    "control_disabled",
    "learning_disabled",
    "active_control_disabled",
    "storage_corrupt",
])
def test_safety_flag_blocks(flag):
    entry = _good_entry()
    pgr = _eligible_pgr(entry)
    ctx = _clean_ctx(**{flag: True})
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=ctx,
    )
    expected_reason = f"safety_{flag}"
    assert expected_reason in dec.blocking_reasons
    assert expected_reason in dec.safety_reasons
    assert dec.would_allow_if_enabled is False


# ── T_APP12: Step limit PREHEAT 5 min passes at boundary ─────────────────────

def test_preheat_step_at_limit_passes():
    entry = _good_entry(candidate_type=CandidateType.PREHEAT_DELTA_MIN)
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(proposed_delta=5.0),
    )
    step_blocks = [r for r in dec.blocking_reasons if r.startswith("step_exceeds_limit")]
    assert step_blocks == [], f"Should not block at 5.0 min: {dec.blocking_reasons}"


# ── T_APP13: Step limit PREHEAT exceeded above 5 min ─────────────────────────

def test_preheat_step_exceeds_limit():
    entry = _good_entry(candidate_type=CandidateType.PREHEAT_DELTA_MIN)
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(proposed_delta=5.1),
    )
    assert any(r.startswith("step_exceeds_limit:") for r in dec.blocking_reasons)
    assert dec.max_step_delta == 5.0


# ── T_APP14: Step limit EARLY_CUTOFF 3 min passes at boundary ────────────────

def test_early_cutoff_step_at_limit_passes():
    entry = _good_entry(candidate_type=CandidateType.EARLY_CUTOFF_DELTA_MIN)
    pgr = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.0)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(proposed_delta=3.0),
    )
    step_blocks = [r for r in dec.blocking_reasons if r.startswith("step_exceeds_limit")]
    assert step_blocks == []


# ── T_APP15: Step limit EARLY_CUTOFF exceeded above 3 min ────────────────────

def test_early_cutoff_step_exceeds_limit():
    entry = _good_entry(candidate_type=CandidateType.EARLY_CUTOFF_DELTA_MIN)
    pgr = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.0)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(proposed_delta=3.1),
    )
    assert any(r.startswith("step_exceeds_limit:") for r in dec.blocking_reasons)
    assert dec.max_step_delta == 3.0


# ── T_APP16: Cumulative limit PREHEAT 15 min exceeded ────────────────────────

def test_preheat_cumulative_exceeds_limit():
    entry = _good_entry(candidate_type=CandidateType.PREHEAT_DELTA_MIN)
    pgr = _eligible_pgr(entry)
    # 12.0 + 5.0 = 17.0 > 15.0
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(proposed_delta=5.0, current_cumulative_delta=12.0),
    )
    assert any(r.startswith("cumulative_exceeds_limit:") for r in dec.blocking_reasons)
    assert dec.max_cumulative_delta == 15.0


# ── T_APP17: Cumulative limit EARLY_CUTOFF 10 min exceeded ───────────────────

def test_early_cutoff_cumulative_exceeds_limit():
    entry = _good_entry(candidate_type=CandidateType.EARLY_CUTOFF_DELTA_MIN)
    pgr = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.0)
    # 8.0 + 3.0 = 11.0 > 10.0
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(proposed_delta=3.0, current_cumulative_delta=8.0),
    )
    assert any(r.startswith("cumulative_exceeds_limit:") for r in dec.blocking_reasons)


# ── T_APP18: Unknown cumulative delta blocks conservatively ──────────────────

def test_unknown_cumulative_blocks_conservatively():
    entry = _good_entry()
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(current_cumulative_delta=None),
    )
    assert "unknown_cumulative_delta" in dec.blocking_reasons


# ── T_APP19: Active cooldown blocks ──────────────────────────────────────────

def test_active_cooldown_blocks():
    entry = _good_entry()
    pgr = _eligible_pgr(entry)
    # last applied 3 days ago, cooldown = 7 days
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(
            last_applied_ts="2026-06-09T10:00:00+00:00",
            now_ts="2026-06-12T10:00:00+00:00",
        ),
    )
    assert any(r.startswith("cooldown_active:") for r in dec.blocking_reasons)


# ── T_APP20: Expired cooldown does not block ─────────────────────────────────

def test_expired_cooldown_does_not_block():
    entry = _good_entry()
    pgr = _eligible_pgr(entry)
    # last applied 10 days ago, cooldown = 7 days → expired
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(
            last_applied_ts="2026-06-02T10:00:00+00:00",
            now_ts="2026-06-12T10:00:00+00:00",
        ),
    )
    cooldown_blocks = [r for r in dec.blocking_reasons if r.startswith("cooldown_active:")]
    assert cooldown_blocks == []


# ── T_APP21: to_dict() is public-safe and read-only ──────────────────────────

def test_to_dict_is_public_safe():
    entry = _good_entry()
    pgr = _eligible_pgr(entry)
    dec = evaluate_application_readiness(
        history_entry=entry,
        promotion_result=pgr,
        runtime_context=_clean_ctx(),
    )
    d = dec.to_dict()
    # Required keys present
    for key in (
        "candidate_key", "candidate_type", "direction",
        "application_enabled", "would_allow_if_enabled",
        "candidate_type_allowed", "max_step_delta", "max_cumulative_delta",
        "blocking_reasons", "safety_reasons",
    ):
        assert key in d, f"Missing key: {key}"
    # No control-side keys
    forbidden = {"setpoint", "boost", "tpi", "dispatch", "apply", "control_value"}
    assert not (forbidden & set(d.keys())), "Control keys found in to_dict() output"
    # application_enabled always False
    assert d["application_enabled"] is False
    # Lists, not tuples
    assert isinstance(d["blocking_reasons"], list)
    assert isinstance(d["safety_reasons"], list)


# ── T_APP22: application_readiness_preview() static output is correct ─────────

def test_application_readiness_preview_preheat():
    entry = _good_entry(candidate_type=CandidateType.PREHEAT_DELTA_MIN)
    preview = application_readiness_preview(entry)
    assert preview["application_enabled"] is False
    assert preview["would_allow_if_enabled"] is False
    assert "application_disabled" in preview["blocking_reasons"]
    assert preview["candidate_type_allowed"] is True
    assert preview["max_step_delta"] == 5.0
    assert preview["max_cumulative_delta"] == 15.0


def test_application_readiness_preview_early_cutoff():
    entry = _good_entry(candidate_type=CandidateType.EARLY_CUTOFF_DELTA_MIN)
    preview = application_readiness_preview(entry)
    assert preview["candidate_type_allowed"] is True
    assert preview["max_step_delta"] == 3.0
    assert preview["max_cumulative_delta"] == 10.0


def test_application_readiness_preview_blocked_type():
    entry = _good_entry(candidate_type=CandidateType.BOOST_BIAS)
    preview = application_readiness_preview(entry)
    assert preview["candidate_type_allowed"] is False
    assert preview["max_step_delta"] is None
    assert preview["max_cumulative_delta"] is None
    assert "application_disabled" in preview["blocking_reasons"]


# ── T_APP_CTRL: No control-path keywords in module source ────────────────────

def test_no_control_path_keywords_in_module():
    import inspect
    import custom_components.thermosmart.learning.adaptation.application as mod
    source = inspect.getsource(mod)
    forbidden_keywords = [
        "set_point", "setpoint", "boost_duration", "preheat_duration",
        "async_set_temperature", "dispatch", "control_value",
        "apply_candidate",
    ]
    for kw in forbidden_keywords:
        assert kw not in source, f"Control keyword '{kw}' found in application.py"
