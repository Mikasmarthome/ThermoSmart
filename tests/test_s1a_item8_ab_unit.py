"""S1a Item 8 — A/B unit tests: ABRunResult structure and scenario config proofs.

Tests in this file are pure Python, require NO simulation run, and run in
milliseconds.  They validate:

  - ABRunResult dataclass fields and delta computation
  - A/B acceptance-check logic (hard failures, warnings)
  - Scenario config equality guarantees (same seed, same physics)
  - Metric completeness (all Item 8 fields present on SimulationResult)
  - Baseline mode: model_update_total stays zero in SHADOW-only runs

Test count: 12
"""
from __future__ import annotations

import math
import pytest

from tests.simulation.result import (
    SimulationResult,
    ABRunResult,
    AB_COLD_REGRESSION_MAX,
    AB_SERVICE_CALL_RATIO_MAX,
    AB_OVERSHOOT_AREA_RATIO_MAX,
    ACCEPTANCE_P99_MS_MAX,
    ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY,
)
from tests.simulation.scenarios import (
    scenario_s1h_30d_profile_a,
    scenario_s1h_30d_profile_a_baseline,
    scenario_s1h_30d_profile_b,
    scenario_s1h_30d_profile_b_baseline,
    scenario_s1h_30d_stress,
    scenario_s1h_30d_stress_baseline,
    scenario_s1h_30d_trvonly,
    scenario_s1h_30d_trvonly_baseline,
    scenario_s1h_30d_multitrvmixed,
    scenario_s1h_30d_multitrvmixed_baseline,
    scenario_s1g_180d_profile_a,
    scenario_s1g_180d_profile_a_baseline,
    scenario_s1g_180d_profile_b,
    scenario_s1g_180d_profile_b_baseline,
    scenario_s1g_180d_profile_c,
    scenario_s1g_180d_profile_c_baseline,
)

_SEED = 42


def _make_result(**overrides) -> SimulationResult:
    """Build a minimal SimulationResult for unit testing."""
    defaults = dict(
        scenario_id="test",
        profile_name="mock",
        seed=42,
        duration_days=30.0,
        step_s=300.0,
        total_steps=8640,
        n_restarts=2,
        comfort_fraction=0.75,
        cold_fraction=0.03,
        overshoot_fraction=0.05,
        overshoot_area_k_h=1.2,
        undershoot_area_k_h=0.5,
        service_call_count=150,
        window_open_steps=100,
        heating_on_steps=3000,
        total_heating_wh=45000.0,
        model_update_total_final=0,
        final_control_applied=0,
        final_control_fallback=0,
        final_fallback_rate=0.0,
        final_blob_bytes=0,
        final_save_count=0,
        heat_rate_update_count_final=0,
        heat_loss_update_count_final=0,
        onset_delay_update_count_final=0,
        afterheat_update_count_final=0,
        outcome_update_count_final=0,
        final_heat_rate_error=None,
        final_heat_rate_relative_error=None,
        final_learned_heat_rate=None,
        gt_heat_rate=None,
        final_learned_supply_delay_s=None,
        gt_supply_delay_s=None,
        supply_delay_error_s=None,
        supply_delay_relative_error=None,
        mean_step_ms=5.0,
        p95_step_ms=15.0,
        p99_step_ms=25.0,
        max_step_ms=40.0,
        storage_growth_kb_per_day=0.5,
        initial_blob_bytes=0,
        acceptance_passed=True,
        failures=[],
        warnings=[],
    )
    defaults.update(overrides)
    return SimulationResult(**defaults)


# ── 1. ABRunResult field completeness ─────────────────────────────────────────

def test_abrunresult_has_all_required_fields():
    """ABRunResult must expose all Item 8 specification fields."""
    base = _make_result(scenario_id="base")
    adap = _make_result(scenario_id="adap")
    ab = ABRunResult.from_pair(base, adap, scenario_id="test-pair")

    assert hasattr(ab, "scenario_id")
    assert hasattr(ab, "duration_days")
    assert hasattr(ab, "seed")
    assert hasattr(ab, "baseline")
    assert hasattr(ab, "adaptive")
    assert hasattr(ab, "delta_cold_fraction")
    assert hasattr(ab, "delta_overshoot_fraction")
    assert hasattr(ab, "delta_overshoot_area_k_h")
    assert hasattr(ab, "delta_undershoot_area_k_h")
    assert hasattr(ab, "delta_comfort_fraction")
    assert hasattr(ab, "delta_service_calls")
    assert hasattr(ab, "delta_heating_on_steps")
    assert hasattr(ab, "delta_heating_wh")
    assert hasattr(ab, "supply_delay_error_s")
    assert hasattr(ab, "supply_delay_relative_error")
    assert hasattr(ab, "heat_rate_relative_error")
    assert hasattr(ab, "onset_delay_update_count")
    assert hasattr(ab, "afterheat_update_count")
    assert hasattr(ab, "final_control_applied")
    assert hasattr(ab, "final_control_fallback")
    assert hasattr(ab, "final_fallback_rate")
    assert hasattr(ab, "model_update_total")
    assert hasattr(ab, "ab_acceptance_passed")
    assert hasattr(ab, "ab_failures")
    assert hasattr(ab, "ab_warnings")


# ── 2. Delta computation correctness ─────────────────────────────────────────

def test_abrunresult_deltas_computed_correctly():
    """Deltas are (adaptive − baseline) for all comfort and effort fields."""
    base = _make_result(
        cold_fraction=0.04, overshoot_fraction=0.06,
        overshoot_area_k_h=2.0, undershoot_area_k_h=1.0,
        comfort_fraction=0.80,
        service_call_count=100, heating_on_steps=3000, total_heating_wh=40000.0,
    )
    adap = _make_result(
        cold_fraction=0.03, overshoot_fraction=0.05,
        overshoot_area_k_h=1.5, undershoot_area_k_h=0.8,
        comfort_fraction=0.85,
        service_call_count=120, heating_on_steps=2800, total_heating_wh=41000.0,
    )
    ab = ABRunResult.from_pair(base, adap, scenario_id="delta-test")

    assert abs(ab.delta_cold_fraction - (0.03 - 0.04)) < 1e-9
    assert abs(ab.delta_overshoot_fraction - (0.05 - 0.06)) < 1e-9
    assert abs(ab.delta_overshoot_area_k_h - (1.5 - 2.0)) < 1e-6
    assert abs(ab.delta_undershoot_area_k_h - (0.8 - 1.0)) < 1e-6
    assert abs(ab.delta_comfort_fraction - (0.85 - 0.80)) < 1e-9
    assert ab.delta_service_calls == 20
    assert ab.delta_heating_on_steps == -200
    assert abs(ab.delta_heating_wh - 1000.0) < 1e-3


# ── 3. Hard failure: cold_fraction regression ─────────────────────────────────

def test_abrunresult_hard_fail_cold_regression():
    """Adaptive that is >5 pp colder than baseline fails A/B acceptance."""
    base = _make_result(cold_fraction=0.02)
    adap = _make_result(cold_fraction=0.02 + AB_COLD_REGRESSION_MAX + 0.01)
    ab = ABRunResult.from_pair(base, adap, scenario_id="cold-reg-fail")

    assert not ab.ab_acceptance_passed
    assert any("cold_fraction" in f for f in ab.ab_failures)


# ── 4. Hard failure: adaptive p99 step time exceeded ─────────────────────────

def test_abrunresult_hard_fail_p99():
    """Adaptive p99 step time > 100 ms triggers hard failure."""
    base = _make_result()
    adap = _make_result(p99_step_ms=ACCEPTANCE_P99_MS_MAX + 10.0)
    ab = ABRunResult.from_pair(base, adap, scenario_id="p99-fail")

    assert not ab.ab_acceptance_passed
    assert any("p99" in f for f in ab.ab_failures)


# ── 5. Hard failure: storage growth ──────────────────────────────────────────

def test_abrunresult_hard_fail_storage_growth():
    """Adaptive storage growth > 50 KB/day triggers hard failure."""
    base = _make_result()
    adap = _make_result(
        storage_growth_kb_per_day=ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY + 5.0
    )
    ab = ABRunResult.from_pair(base, adap, scenario_id="storage-fail")

    assert not ab.ab_acceptance_passed
    assert any("storage" in f for f in ab.ab_failures)


# ── 6. Warning: service call ratio ────────────────────────────────────────────

def test_abrunresult_warning_service_call_ratio():
    """Adaptive service calls > 2× baseline triggers a warning (not failure)."""
    base = _make_result(service_call_count=100)
    adap = _make_result(service_call_count=250)   # 2.5× > AB_SERVICE_CALL_RATIO_MAX
    ab = ABRunResult.from_pair(base, adap, scenario_id="svc-warn")

    assert ab.ab_acceptance_passed   # warning, not failure
    assert any("service call" in w for w in ab.ab_warnings)


# ── 7. No warnings when within limits ────────────────────────────────────────

def test_abrunresult_no_failures_within_limits():
    """Perfect A/B pair produces no failures and no warnings."""
    base = _make_result(
        cold_fraction=0.04, service_call_count=100,
        overshoot_area_k_h=2.0, total_heating_wh=40000.0,
    )
    adap = _make_result(
        cold_fraction=0.03, service_call_count=105,
        overshoot_area_k_h=2.1, total_heating_wh=40500.0,
    )
    ab = ABRunResult.from_pair(base, adap, scenario_id="ok-pair")

    assert ab.ab_acceptance_passed
    assert ab.ab_failures == []
    assert ab.ab_warnings == []


# ── 8. check_ab_acceptance raises on failure ──────────────────────────────────

def test_check_ab_acceptance_raises_on_failure():
    """check_ab_acceptance() raises AssertionError when acceptance failed."""
    base = _make_result(cold_fraction=0.02)
    adap = _make_result(cold_fraction=0.10)  # +8pp regression
    ab = ABRunResult.from_pair(base, adap, scenario_id="raise-test")

    with pytest.raises(AssertionError, match="cold_fraction"):
        ab.check_ab_acceptance()


# ── 9. Scenario config A/B input equality proofs ─────────────────────────────

@pytest.mark.parametrize("adap_fn,base_fn,pair_name", [
    (scenario_s1h_30d_profile_a, scenario_s1h_30d_profile_a_baseline, "ProfileA-30d"),
    (scenario_s1h_30d_profile_b, scenario_s1h_30d_profile_b_baseline, "ProfileB-30d"),
    (scenario_s1h_30d_stress, scenario_s1h_30d_stress_baseline, "Stress-30d"),
    (scenario_s1h_30d_trvonly, scenario_s1h_30d_trvonly_baseline, "TRVonly-30d"),
    (scenario_s1h_30d_multitrvmixed, scenario_s1h_30d_multitrvmixed_baseline, "Multi-30d"),
    (scenario_s1g_180d_profile_a, scenario_s1g_180d_profile_a_baseline, "ProfileA-180d"),
    (scenario_s1g_180d_profile_b, scenario_s1g_180d_profile_b_baseline, "ProfileB-180d"),
    (scenario_s1g_180d_profile_c, scenario_s1g_180d_profile_c_baseline, "ProfileC-180d"),
])
def test_scenario_pair_physics_identical(adap_fn, base_fn, pair_name):
    """Both sides of an A/B pair must share the same room profile, seed, and duration."""
    adap = adap_fn(seed=_SEED)
    base = base_fn(seed=_SEED)

    assert adap.seed == base.seed, f"{pair_name}: seed mismatch"
    assert adap.step_s == base.step_s, f"{pair_name}: step_s mismatch"
    assert adap.duration_s == base.duration_s, f"{pair_name}: duration mismatch"
    assert adap.room_profile.heat_loss_w_per_k == base.room_profile.heat_loss_w_per_k, (
        f"{pair_name}: room profile mismatch (heat_loss_w_per_k)"
    )
    assert adap.room_profile.heating_power_w == base.room_profile.heating_power_w, (
        f"{pair_name}: room profile mismatch (heating_power_w)"
    )
    assert base.baseline_mode, f"{pair_name}: baseline must have baseline_mode=True"
    assert not adap.baseline_mode, f"{pair_name}: adaptive must have baseline_mode=False"


# ── 10. Baseline config: learning disabled ────────────────────────────────────

@pytest.mark.parametrize("base_fn", [
    scenario_s1h_30d_profile_a_baseline,
    scenario_s1h_30d_profile_b_baseline,
    scenario_s1h_30d_stress_baseline,
    scenario_s1h_30d_trvonly_baseline,
    scenario_s1h_30d_multitrvmixed_baseline,
    scenario_s1g_180d_profile_a_baseline,
])
def test_baseline_config_has_learning_disabled(base_fn):
    """Baseline scenario configs must have learning_mode=False."""
    cfg = base_fn(seed=_SEED)
    assert not cfg.learning_mode, (
        f"{cfg.name}: learning_mode must be False for baseline scenarios"
    )


# ── 11. SimulationResult has all Item 8 comfort fields ───────────────────────

def test_simulation_result_has_item8_fields():
    """SimulationResult must expose all Item 8 A/B metric fields."""
    r = _make_result()
    assert hasattr(r, "overshoot_area_k_h")
    assert hasattr(r, "undershoot_area_k_h")
    assert hasattr(r, "heating_on_steps")
    assert hasattr(r, "total_heating_wh")
    assert hasattr(r, "supply_delay_error_s")
    assert hasattr(r, "supply_delay_relative_error")
    assert hasattr(r, "final_heat_rate_relative_error")
    assert hasattr(r, "onset_delay_update_count_final")
    assert hasattr(r, "afterheat_update_count_final")
    assert hasattr(r, "final_control_applied")
    assert hasattr(r, "final_control_fallback")
    assert hasattr(r, "final_fallback_rate")
    assert hasattr(r, "model_update_total_final")


# ── 12. ABRunResult.report() produces non-empty string ───────────────────────

def test_abrunresult_report_non_empty():
    """ABRunResult.report() returns a non-empty human-readable string."""
    base = _make_result(scenario_id="base-report")
    adap = _make_result(scenario_id="adap-report")
    ab = ABRunResult.from_pair(base, adap, scenario_id="report-test")

    report = ab.report()
    assert isinstance(report, str)
    assert len(report) > 50
    assert "ABRunResult" in report
    assert "cold_fraction" in report
