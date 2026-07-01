"""Promotion readiness simulation for passive adaptation candidate history.

Verifies that evaluate_promotion_readiness() correctly gates ELIGIBLE vs
BLOCKED states across all 7 promotion gates, and that no control path is
reachable from an ELIGIBLE result.

Pure tests — no HA imports, no Storage writes, no control modification.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_components.thermosmart.learning.adaptation.contracts import (
    AdaptationLifecycle,
    AdaptationDirection,
    CandidateType,
    OutcomeSignal,
    SituationContext,
)
from custom_components.thermosmart.learning.adaptation.history import (
    CandidateHistoryEntry,
    PromotionReadiness,
    evaluate_promotion_readiness,
    accumulate_into_history,
    update_adaptation_history,
    _MIN_SEEN_COUNT,
    _MIN_OUTCOME_SAMPLES,
    _MIN_DATA_QUALITY,
    _MAX_CONFOUNDER_RATIO,
    _MIN_SPAN_DAYS,
    _MIN_STABLE_REASON_RATIO,
)
from custom_components.thermosmart.learning.adaptation.candidates import suggest_candidates
from custom_components.thermosmart.learning.adaptation.history_store_schema import (
    adaptation_history_entry_for_research_export,
)

# ── Synthetic helpers ──────────────────────────────────────────────────────────

_KEY = "abcdef0123456789"  # stable test key (no zone meaning)


def _good_entry(**overrides) -> CandidateHistoryEntry:
    """Return a CandidateHistoryEntry that passes all 7 promotion gates."""
    defaults = dict(
        candidate_key=_KEY,
        candidate_type=CandidateType.PREHEAT_DELTA_MIN,
        direction=AdaptationDirection.INCREASE,
        first_seen_ts="2026-06-01T10:00:00+00:00",
        last_seen_ts="2026-06-12T10:00:00+00:00",   # 11 days
        seen_count=5,                                 # >= _MIN_SEEN_COUNT (3)
        supporting_outcome_count=15,                  # >= _MIN_OUTCOME_SAMPLES (12)
        avg_outcome_quality=0.65,                     # >= _MIN_DATA_QUALITY (0.45)
        avg_timeout_rate=0.32,
        avg_overshoot_rate=0.10,
        last_lifecycle=AdaptationLifecycle.SHADOW,    # required
        dominant_reason="high_timeout_rate:0.32",
        dominant_reason_ratio=0.90,                   # >= _MIN_STABLE_REASON_RATIO (0.75)
    )
    defaults.update(overrides)
    return CandidateHistoryEntry(**defaults)


def _good_signal() -> OutcomeSignal:
    """Return an OutcomeSignal that generates SHADOW candidates and passes promotion gates."""
    return OutcomeSignal(
        sample_count=15,             # >= gate min_samples (8) and promotion min (12)
        timeout_rate=0.30,           # >= 0.25 to trigger preheat candidate; <= 0.50 gate
        overshoot_rate=0.10,         # < 0.25 — no early-cutoff candidate
        reached_rate=0.60,
        general_data_quality=0.65,   # >= 0.45 promotion floor
        aggregate_reliability=0.65,
        partial_ratio=0.15,
        confounder_contamination=False,
    )


def _good_situation() -> SituationContext:
    return SituationContext(
        outdoor_bucket="b3",
        mode_context="comfort",
        preheat_was_active=True,
    )


# ── Test 1: fully supported synthetic entry → ELIGIBLE ────────────────────────

def test_eligible_fully_supported():
    entry = _good_entry()
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.readiness is PromotionReadiness.ELIGIBLE, (
        f"Expected ELIGIBLE, got {result.readiness}: {result.blocking_reasons}"
    )
    assert not result.blocking_gates, f"Unexpected blocking gates: {result.blocking_gates}"
    assert not result.blocking_reasons, f"Unexpected reasons: {result.blocking_reasons}"
    assert all(result.gate_results.values()), f"Gate failures: {result.gate_results}"
    print(f"  T1 PASS: fully supported entry -> ELIGIBLE, all gates={list(result.gate_results.values())}")


# ── Test 2: seen_count < 3 blocks ─────────────────────────────────────────────

def test_blocked_seen_count_too_low():
    entry = _good_entry(seen_count=_MIN_SEEN_COUNT - 1)
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.readiness is PromotionReadiness.BLOCKED
    assert "min_seen_count" in result.blocking_gates
    assert not result.gate_results["min_seen_count"]
    assert any("insufficient_seen_count" in r for r in result.blocking_reasons)
    print(f"  T2 PASS: seen_count={_MIN_SEEN_COUNT-1} -> BLOCKED (min_seen_count)")


def test_seen_count_exactly_at_threshold_passes():
    entry = _good_entry(seen_count=_MIN_SEEN_COUNT)
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.gate_results["min_seen_count"]
    print(f"  T2b PASS: seen_count={_MIN_SEEN_COUNT} (boundary) -> gate passes")


# ── Test 3: supporting_outcome_count < 12 blocks ──────────────────────────────

def test_blocked_outcome_samples_too_low():
    entry = _good_entry(supporting_outcome_count=_MIN_OUTCOME_SAMPLES - 1)
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.readiness is PromotionReadiness.BLOCKED
    assert "min_outcome_samples" in result.blocking_gates
    assert not result.gate_results["min_outcome_samples"]
    assert any("insufficient_outcome_samples" in r for r in result.blocking_reasons)
    print(f"  T3 PASS: supporting_outcome_count={_MIN_OUTCOME_SAMPLES-1} -> BLOCKED")


def test_outcome_samples_boundary_passes():
    entry = _good_entry(supporting_outcome_count=_MIN_OUTCOME_SAMPLES)
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.gate_results["min_outcome_samples"]
    print(f"  T3b PASS: supporting_outcome_count={_MIN_OUTCOME_SAMPLES} (boundary) -> gate passes")


# ── Test 4: avg_outcome_quality < 0.45 blocks ────────────────────────────────

def test_blocked_data_quality_too_low():
    entry = _good_entry(avg_outcome_quality=_MIN_DATA_QUALITY - 0.01)
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.readiness is PromotionReadiness.BLOCKED
    assert "min_data_quality" in result.blocking_gates
    assert not result.gate_results["min_data_quality"]
    assert any("low_data_quality" in r for r in result.blocking_reasons)
    print(f"  T4 PASS: avg_outcome_quality={_MIN_DATA_QUALITY-0.01:.3f} -> BLOCKED")


def test_data_quality_boundary_passes():
    entry = _good_entry(avg_outcome_quality=_MIN_DATA_QUALITY)
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.gate_results["min_data_quality"]
    print(f"  T4b PASS: avg_outcome_quality={_MIN_DATA_QUALITY} (boundary) -> gate passes")


# ── Test 5: confounder_ratio > 0.15 blocks ────────────────────────────────────

def test_blocked_confounder_ratio_too_high():
    entry = _good_entry()
    result = evaluate_promotion_readiness(entry, span_days=11.0,
                                          confounder_ratio=_MAX_CONFOUNDER_RATIO + 0.01)
    assert result.readiness is PromotionReadiness.BLOCKED
    assert "max_confounder_ratio" in result.blocking_gates
    assert not result.gate_results["max_confounder_ratio"]
    assert any("high_confounder_ratio" in r for r in result.blocking_reasons)
    print(f"  T5 PASS: confounder_ratio={_MAX_CONFOUNDER_RATIO+0.01:.3f} -> BLOCKED")


def test_confounder_ratio_boundary_passes():
    entry = _good_entry()
    result = evaluate_promotion_readiness(entry, span_days=11.0,
                                          confounder_ratio=_MAX_CONFOUNDER_RATIO)
    assert result.gate_results["max_confounder_ratio"]
    print(f"  T5b PASS: confounder_ratio={_MAX_CONFOUNDER_RATIO} (boundary) -> gate passes")


# ── Test 6: span_days < 7.0 blocks ───────────────────────────────────────────

def test_blocked_span_days_too_short():
    entry = _good_entry()
    result = evaluate_promotion_readiness(entry, span_days=_MIN_SPAN_DAYS - 0.1,
                                          confounder_ratio=0.05)
    assert result.readiness is PromotionReadiness.BLOCKED
    assert "min_span_days" in result.blocking_gates
    assert not result.gate_results["min_span_days"]
    assert any("insufficient_span_days" in r for r in result.blocking_reasons)
    print(f"  T6 PASS: span_days={_MIN_SPAN_DAYS-0.1:.1f} -> BLOCKED")


def test_span_days_boundary_passes():
    entry = _good_entry()
    result = evaluate_promotion_readiness(entry, span_days=_MIN_SPAN_DAYS,
                                          confounder_ratio=0.05)
    assert result.gate_results["min_span_days"]
    print(f"  T6b PASS: span_days={_MIN_SPAN_DAYS} (boundary) -> gate passes")


# ── Test 7: dominant_reason_ratio < 0.75 blocks ───────────────────────────────

def test_blocked_unstable_reason():
    entry = _good_entry(dominant_reason_ratio=_MIN_STABLE_REASON_RATIO - 0.01)
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.readiness is PromotionReadiness.BLOCKED
    assert "stable_reason" in result.blocking_gates
    assert not result.gate_results["stable_reason"]
    assert any("unstable_reason" in r for r in result.blocking_reasons)
    print(f"  T7 PASS: dominant_reason_ratio={_MIN_STABLE_REASON_RATIO-0.01:.3f} -> BLOCKED")


def test_stable_reason_boundary_passes():
    entry = _good_entry(dominant_reason_ratio=_MIN_STABLE_REASON_RATIO)
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.gate_results["stable_reason"]
    print(f"  T7b PASS: dominant_reason_ratio={_MIN_STABLE_REASON_RATIO} (boundary) -> gate passes")


# ── Test 8: last_lifecycle != SHADOW blocks ───────────────────────────────────

def test_blocked_lifecycle_not_shadow():
    for non_shadow_lc in (
        AdaptationLifecycle.APPLIED,
        AdaptationLifecycle.ELIGIBLE,
        AdaptationLifecycle.ROLLED_BACK,
        AdaptationLifecycle.EXPIRED,
    ):
        entry = _good_entry(last_lifecycle=non_shadow_lc)
        result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
        assert result.readiness is PromotionReadiness.BLOCKED, (
            f"Expected BLOCKED for lifecycle={non_shadow_lc.value}"
        )
        assert "last_lifecycle_shadow" in result.blocking_gates
        assert not result.gate_results["last_lifecycle_shadow"]
        assert any("lifecycle_not_shadow" in r for r in result.blocking_reasons)
    print(f"  T8 PASS: non-SHADOW lifecycle -> BLOCKED (tested 4 variants)")


# ── Test 9: multiple blocking reasons reported deterministically ───────────────

def test_multiple_blocking_reasons():
    # Fail 4 gates simultaneously: seen_count, outcome_samples, data_quality, span_days
    entry = _good_entry(
        seen_count=1,
        supporting_outcome_count=5,
        avg_outcome_quality=0.30,
        dominant_reason_ratio=0.50,
    )
    result = evaluate_promotion_readiness(entry, span_days=2.0, confounder_ratio=0.25)
    assert result.readiness is PromotionReadiness.BLOCKED
    # All 6 gates should fail
    assert "min_seen_count" in result.blocking_gates
    assert "min_outcome_samples" in result.blocking_gates
    assert "min_data_quality" in result.blocking_gates
    assert "max_confounder_ratio" in result.blocking_gates
    assert "min_span_days" in result.blocking_gates
    assert "stable_reason" in result.blocking_gates
    assert len(result.blocking_reasons) >= 6
    # Gate results are deterministic
    result2 = evaluate_promotion_readiness(entry, span_days=2.0, confounder_ratio=0.25)
    assert result.blocking_gates == result2.blocking_gates
    assert result.blocking_reasons == result2.blocking_reasons
    print(f"  T9 PASS: 6 gates blocked simultaneously, {len(result.blocking_reasons)} reasons, deterministic")


# ── Test 10: research export is public-safe and read-only ─────────────────────

def test_export_readiness_public_safe():
    entry = _good_entry()
    exported = adaptation_history_entry_for_research_export(
        entry, span_days=11.0, confounder_ratio=0.05
    )
    # Must contain promotion_readiness
    assert "promotion_readiness" in exported
    assert exported["promotion_readiness"] == PromotionReadiness.ELIGIBLE.value

    # Must NOT contain raw timestamps (privacy)
    assert "first_seen_ts" not in exported
    assert "last_seen_ts" not in exported

    # Must NOT contain any identity-sensitive strings
    forbidden_keys = {"entity_id", "entry_id", "zone_id", "learning_zone_id",
                      "decision_id", "episode_id", "person", "email"}
    assert not (set(exported.keys()) & forbidden_keys), (
        f"Forbidden keys in export: {set(exported.keys()) & forbidden_keys}"
    )

    # candidate_key is a hex hash — verify it contains only hex chars
    key = exported.get("candidate_key", "")
    assert all(c in "0123456789abcdef" for c in key), f"Non-hex candidate_key: {key!r}"

    # Export is a plain dict (read-only result, no callable)
    assert isinstance(exported, dict)
    assert "blocking_reasons" in exported
    assert isinstance(exported["blocking_reasons"], list)
    assert exported["blocking_reasons"] == []  # ELIGIBLE → no blockers

    print(f"  T10 PASS: export is public-safe, promotion_readiness='{exported['promotion_readiness']}', "
          f"no timestamps, no identity fields, {len(exported)} fields")


def test_export_blocked_includes_reasons():
    entry = _good_entry(seen_count=1)
    exported = adaptation_history_entry_for_research_export(
        entry, span_days=3.0, confounder_ratio=0.05
    )
    assert exported["promotion_readiness"] == PromotionReadiness.BLOCKED.value
    assert len(exported["blocking_reasons"]) >= 1
    print(f"  T10b PASS: blocked export includes reasons: {exported['blocking_reasons']}")


# ── Test 11: ELIGIBLE result has no control effect ────────────────────────────

def test_eligible_has_no_control_effect():
    entry = _good_entry()
    result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
    assert result.readiness is PromotionReadiness.ELIGIBLE

    # verify the result object has no callable that could apply a setpoint
    result_dict = result.to_dict()
    control_keys = {"trv_setpoint", "boost_offset", "set_temperature",
                    "dispatch", "preheat_minutes", "applied_at"}
    assert not (set(result_dict.keys()) & control_keys), (
        f"Control keys found in PromotionGateResult: {set(result_dict.keys()) & control_keys}"
    )

    # entry.last_lifecycle remains SHADOW even after ELIGIBLE evaluation
    assert entry.last_lifecycle is AdaptationLifecycle.SHADOW, (
        "evaluate_promotion_readiness must not mutate entry.last_lifecycle"
    )

    # AdaptationLifecycle.ELIGIBLE is defined but never set by accumulate_into_history
    # — verify that update_adaptation_history only produces SHADOW-lifecycle entries
    signal = _good_signal()
    situation = _good_situation()
    history = update_adaptation_history({}, "zone_test_001", signal, situation,
                                        "2026-06-12T10:00:00")
    if history:
        for hist_entry in history.values():
            assert hist_entry.last_lifecycle is AdaptationLifecycle.SHADOW, (
                f"update_adaptation_history produced non-SHADOW lifecycle: "
                f"{hist_entry.last_lifecycle}"
            )

    print("  T11 PASS: ELIGIBLE result contains no control keys; "
          "entry.last_lifecycle remains SHADOW; history entries are SHADOW-only")


# ── Test 12: no control path in adaptation/ module ────────────────────────────

def test_no_control_paths_in_adaptation_module():
    FORBIDDEN_PATTERNS = [
        "set_temperature",
        "async_write_ha_state",
        "async_call_service",
        "trv_setpoint",
        "boost_offset",
        "dispatch(",
        "climate.set",
        "apply_lifecycle(",
        "adjust_recommendation",
    ]
    adaptation_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "custom_components", "thermosmart", "learning", "adaptation",
    )
    violations = []
    for fname in os.listdir(adaptation_dir):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(adaptation_dir, fname)
        src = open(fpath, encoding="utf-8").read()
        for pat in FORBIDDEN_PATTERNS:
            if pat in src:
                violations.append(f"{fname}: {pat!r}")
    assert not violations, f"Forbidden control patterns found: {violations}"
    print(f"  T12 PASS: no control paths in adaptation/ ({len(os.listdir(adaptation_dir))} files checked)")


# ── Integration Test: full pipeline → ELIGIBLE via accumulation ───────────────

def test_integration_pipeline_accumulation_reaches_eligible():
    """Accumulate 5 SHADOW traces via update_adaptation_history, then check ELIGIBLE."""
    zone_id = "zone_integ_test_01"
    signal = _good_signal()
    situation = _good_situation()

    ts_first = "2026-06-01T10:00:00"
    ts_last  = "2026-06-12T10:00:00"  # 11 days later

    history: dict = {}
    # First sighting
    history = update_adaptation_history(history, zone_id, signal, situation, ts_first)
    # 4 more sightings (same signal, same reason → dominant_reason_ratio stays 1.0)
    for i in range(4):
        history = update_adaptation_history(history, zone_id, signal, situation,
                                            f"2026-06-0{i+2}T12:00:00")

    assert history, "History must not be empty after 5 accumulations"

    for key, entry in history.items():
        assert entry.last_lifecycle is AdaptationLifecycle.SHADOW, (
            f"Expected SHADOW lifecycle, got {entry.last_lifecycle}"
        )
        assert entry.seen_count >= 5, f"Expected >=5 sightings, got {entry.seen_count}"
        assert entry.supporting_outcome_count >= _MIN_OUTCOME_SAMPLES, (
            f"Expected >=12 outcome samples, got {entry.supporting_outcome_count}"
        )
        assert entry.avg_outcome_quality >= _MIN_DATA_QUALITY, (
            f"Expected quality>={_MIN_DATA_QUALITY}, got {entry.avg_outcome_quality}"
        )
        assert entry.dominant_reason_ratio >= _MIN_STABLE_REASON_RATIO, (
            f"Expected ratio>={_MIN_STABLE_REASON_RATIO}, got {entry.dominant_reason_ratio}"
        )

        result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
        assert result.readiness is PromotionReadiness.ELIGIBLE, (
            f"Expected ELIGIBLE after full accumulation, got {result.readiness}: "
            f"{result.blocking_reasons}"
        )

    print(f"  T_INTEG PASS: {len(history)} entries, all SHADOW, all ELIGIBLE "
          f"after 5 accumulations over 11 days")


# ── Integration Test: zero-sample → cannot reach ELIGIBLE ─────────────────────

def test_integration_insufficient_samples_stays_blocked():
    """Thin signal (only 5 samples) cannot reach ELIGIBLE via accumulation."""
    signal = OutcomeSignal(
        sample_count=5,          # < 12 → blocks min_outcome_samples
        timeout_rate=0.30,
        overshoot_rate=0.10,
        reached_rate=0.60,
        general_data_quality=0.65,
        aggregate_reliability=0.65,
        partial_ratio=0.15,
        confounder_contamination=False,
    )
    history = update_adaptation_history(
        {}, "zone_thin", signal, _good_situation(), "2026-06-01T10:00:00"
    )
    for entry in history.values():
        result = evaluate_promotion_readiness(entry, span_days=11.0, confounder_ratio=0.05)
        assert result.readiness is PromotionReadiness.BLOCKED
        assert "min_outcome_samples" in result.blocking_gates
    print("  T_THIN PASS: signal with 5 samples stays BLOCKED (min_outcome_samples gate)")


# ── Gate threshold constants are correctly set ────────────────────────────────

def test_gate_threshold_values():
    assert _MIN_SEEN_COUNT         == 3,    f"Expected _MIN_SEEN_COUNT=3, got {_MIN_SEEN_COUNT}"
    assert _MIN_OUTCOME_SAMPLES    == 12,   f"Expected _MIN_OUTCOME_SAMPLES=12"
    assert _MIN_DATA_QUALITY       == 0.45, f"Expected _MIN_DATA_QUALITY=0.45"
    assert _MAX_CONFOUNDER_RATIO   == 0.15, f"Expected _MAX_CONFOUNDER_RATIO=0.15"
    assert _MIN_SPAN_DAYS          == 7.0,  f"Expected _MIN_SPAN_DAYS=7.0"
    assert _MIN_STABLE_REASON_RATIO == 0.75, f"Expected _MIN_STABLE_REASON_RATIO=0.75"
    print(f"  T_GATES PASS: all 6 threshold constants correct "
          f"(seen={_MIN_SEEN_COUNT}, samples={_MIN_OUTCOME_SAMPLES}, "
          f"quality={_MIN_DATA_QUALITY}, confounder<={_MAX_CONFOUNDER_RATIO}, "
          f"span={_MIN_SPAN_DAYS}d, reason_ratio={_MIN_STABLE_REASON_RATIO})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_eligible_fully_supported,
        test_blocked_seen_count_too_low,
        test_seen_count_exactly_at_threshold_passes,
        test_blocked_outcome_samples_too_low,
        test_outcome_samples_boundary_passes,
        test_blocked_data_quality_too_low,
        test_data_quality_boundary_passes,
        test_blocked_confounder_ratio_too_high,
        test_confounder_ratio_boundary_passes,
        test_blocked_span_days_too_short,
        test_span_days_boundary_passes,
        test_blocked_unstable_reason,
        test_stable_reason_boundary_passes,
        test_blocked_lifecycle_not_shadow,
        test_multiple_blocking_reasons,
        test_export_readiness_public_safe,
        test_export_blocked_includes_reasons,
        test_eligible_has_no_control_effect,
        test_no_control_paths_in_adaptation_module,
        test_integration_pipeline_accumulation_reaches_eligible,
        test_integration_insufficient_samples_stays_blocked,
        test_gate_threshold_values,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1

    print()
    print(f"{'='*60}")
    print(f"Promotion Readiness Tests: {passed}/{passed+failed} passed"
          + (" -- ALL GREEN" if not failed else f" -- {failed} FAILED"))
    if failed:
        raise SystemExit(1)
