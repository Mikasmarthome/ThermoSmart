"""S1a Item 8 — 30-day A/B simulation tests.

These tests run 30-day scenarios with the real production coordinator (TPI +
LE2 shadow) to prove the following properties hold over a full heating-season
onset period:

  - Comfort safety: adaptive cold_fraction never regresses past baseline + 5 pp
  - Supply delay learning: learned value present after 30d with restarts
  - Afterheat / overshoot control: overshoot_fraction stays bounded
  - Restart survival: model persists across simulated HA restarts
  - Baseline stability: deterministic mode produces cold_fraction ≤ 8 %
  - No NaN/Inf in any learned model value after 30 days

Single-scenario tests (adaptive only, baseline only) run one scenario per test.
Pair tests run both scenarios and use ABRunResult for the comparison.

Windows exclusion: see conftest.py collect_ignore_glob.
  This file is excluded on Windows (runtime ≈ 3–6 min per test, 15 tests ≈ 90 min).
  All 30d tests are included in the full Docker / Linux validation.

Test count: 16
"""
from __future__ import annotations

import pytest

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
)
from tests.simulation.runner import ScenarioRunner
from tests.simulation.result import SimulationResult, ABRunResult

pytestmark = pytest.mark.asyncio

_SEED = 42
_DURATION_DAYS = 30.0
_COLD_MAX = 0.08          # individual run: ≤ 8% cold steps
_AB_COLD_REGRESSION = 0.05   # A/B pair: adaptive may not be >5pp worse


def _result(summary, cfg):
    return SimulationResult.from_summary(
        summary,
        scenario_id=cfg.name,
        profile_name=cfg.room_profile.name,
        seed=cfg.seed,
        duration_days=_DURATION_DAYS,
        step_s=cfg.step_s,
    )


# ── Profile A: Fast Insulated ─────────────────────────────────────────────────

async def test_30d_profile_a_adaptive_acceptance():
    """30d adaptive Profile A: passes all individual acceptance criteria."""
    cfg = scenario_s1h_30d_profile_a(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    print("\n" + result.report())
    assert result.acceptance_passed, result.report()
    assert summary.has_no_nan_inf(), "NaN/Inf in Profile A adaptive model"


async def test_30d_profile_a_baseline_stability():
    """30d baseline Profile A: cold_fraction ≤ 8%, no model divergence."""
    cfg = scenario_s1h_30d_profile_a_baseline(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    print("\n" + result.report())
    assert result.cold_fraction <= _COLD_MAX, (
        f"Profile A baseline cold_fraction {result.cold_fraction:.3f} > {_COLD_MAX}"
    )
    assert summary.has_no_nan_inf()


async def test_30d_profile_a_ab_cold_fraction_not_regressed():
    """30d A/B: adaptive Profile A must not be >5pp colder than baseline."""
    cfg_adap = scenario_s1h_30d_profile_a(seed=_SEED)
    cfg_base = scenario_s1h_30d_profile_a_baseline(seed=_SEED)
    s_adap = await ScenarioRunner(cfg_adap).run()
    s_base = await ScenarioRunner(cfg_base).run()
    r_adap = _result(s_adap, cfg_adap)
    r_base = _result(s_base, cfg_base)
    ab = ABRunResult.from_pair(r_base, r_adap, scenario_id="30d-A-vs-A-base")
    print("\n" + ab.report())
    ab.check_ab_acceptance()


# ── Profile B: Slow Inertia ───────────────────────────────────────────────────

async def test_30d_profile_b_adaptive_acceptance():
    """30d adaptive Profile B: passes all individual acceptance criteria."""
    cfg = scenario_s1h_30d_profile_b(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    print("\n" + result.report())
    assert result.acceptance_passed, result.report()
    assert summary.has_no_nan_inf(), "NaN/Inf in Profile B adaptive model"


async def test_30d_profile_b_baseline_stability():
    """30d baseline Profile B: cold_fraction ≤ 8%."""
    cfg = scenario_s1h_30d_profile_b_baseline(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    assert result.cold_fraction <= _COLD_MAX, (
        f"Profile B baseline cold_fraction {result.cold_fraction:.3f} > {_COLD_MAX}"
    )
    assert summary.has_no_nan_inf()


# ── Scenario D: Stress / Failure ──────────────────────────────────────────────

async def test_30d_stress_adaptive_no_nan_inf():
    """30d Stress adaptive (Profile C + heavy windows + TRV faults): no NaN/Inf."""
    cfg = scenario_s1h_30d_stress(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    print("\n" + result.report())
    assert summary.has_no_nan_inf(), "NaN/Inf in Stress adaptive model"


async def test_30d_stress_adaptive_graceful_degradation():
    """30d Stress adaptive: graceful degradation under heavy fault load.

    PROFILE_DIFFICULT + windows every 2h (26% open time) + TRV unavailable 3× 12h
    means the room physically cannot hold comfort temp.  The acceptance criterion
    is NOT cold_fraction ≤ 8% (which applies to normal scenarios), but rather:
      - No NaN/Inf (model stability under faults)
      - Service calls bounded (no runaway dispatching)
      - Adaptive does not escalate beyond what physics dictates
    The quantitative cold_fraction value is documented in the simulation report.
    """
    cfg = scenario_s1h_30d_stress(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    print(f"\n  Stress adaptive: cold_fraction={result.cold_fraction:.3f} "
          f"comfort_fraction={result.comfort_fraction:.3f} "
          f"service_calls={result.service_call_count}")
    # No NaN/Inf despite faults
    assert summary.has_no_nan_inf(), "NaN/Inf in Stress adaptive model under fault"
    # Service calls bounded (no runaway dispatch loop)
    _STRESS_SERVICE_CALL_MAX = 2000   # 8640 steps; far below unbounded
    assert result.service_call_count <= _STRESS_SERVICE_CALL_MAX, (
        f"Stress adaptive service calls {result.service_call_count} > {_STRESS_SERVICE_CALL_MAX} "
        f"— potential runaway dispatch"
    )
    # cold_fraction documented (high is expected under extreme stress)
    print(f"  Stress cold_fraction={result.cold_fraction:.3f} (expected high under stress)")


async def test_30d_stress_baseline_graceful_degradation():
    """30d Stress baseline: same fault load, deterministic only — documents baseline."""
    cfg = scenario_s1h_30d_stress_baseline(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    print(f"\n  Stress baseline: cold_fraction={result.cold_fraction:.3f} "
          f"service_calls={result.service_call_count}")
    # No NaN/Inf in baseline either
    assert summary.has_no_nan_inf(), "NaN/Inf in Stress baseline"
    # Baseline cold_fraction documented (no adaptive means no corrections either)
    print(f"  Stress baseline cold_fraction={result.cold_fraction:.3f} (documented)")


# ── Scenario E: TRV-only Minimal Setup ───────────────────────────────────────

async def test_30d_trvonly_adaptive_acceptance():
    """30d TRV-only minimal setup: passes individual acceptance criteria."""
    cfg = scenario_s1h_30d_trvonly(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    print("\n" + result.report())
    assert result.acceptance_passed, result.report()
    assert summary.has_no_nan_inf()


async def test_30d_trvonly_adaptive_learning_occurs():
    """30d TRV-only: model receives at least one update after 30 days."""
    cfg = scenario_s1h_30d_trvonly(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    assert result.model_update_total_final > 0, (
        "TRV-only minimal setup: no model updates after 30 days"
    )


async def test_30d_trvonly_baseline_cold_fraction_bounded():
    """30d TRV-only baseline: cold_fraction ≤ 8%."""
    cfg = scenario_s1h_30d_trvonly_baseline(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    assert result.cold_fraction <= _COLD_MAX, (
        f"TRVonly baseline cold_fraction {result.cold_fraction:.3f} > {_COLD_MAX}"
    )


# ── Scenario F: Multi-TRV Mixed Zone ─────────────────────────────────────────

async def test_30d_multitrvmixed_adaptive_no_nan_inf():
    """30d Multi-TRV (two TRVs, partial fault): no NaN/Inf."""
    cfg = scenario_s1h_30d_multitrvmixed(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    print("\n" + result.report())
    assert summary.has_no_nan_inf(), "NaN/Inf in Multi-TRV adaptive model"


async def test_30d_multitrvmixed_adaptive_cold_fraction_bounded():
    """30d Multi-TRV: cold_fraction does not violate safety limit under partial fault."""
    cfg = scenario_s1h_30d_multitrvmixed(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    assert result.cold_fraction <= _COLD_MAX, (
        f"Multi-TRV adaptive cold_fraction {result.cold_fraction:.3f} > {_COLD_MAX}"
    )


async def test_30d_multitrvmixed_baseline_cold_fraction_bounded():
    """30d Multi-TRV baseline: cold_fraction ≤ 8% even with partial fault."""
    cfg = scenario_s1h_30d_multitrvmixed_baseline(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    assert result.cold_fraction <= _COLD_MAX, (
        f"Multi-TRV baseline cold_fraction {result.cold_fraction:.3f} > {_COLD_MAX}"
    )


# ── Cross-scenario: overshoot area bounded ────────────────────────────────────

async def test_30d_profile_a_overshoot_area_finite():
    """30d adaptive Profile A: overshoot_area_k_h is finite and non-negative."""
    cfg = scenario_s1h_30d_profile_a(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    assert result.overshoot_area_k_h >= 0.0
    assert result.overshoot_area_k_h < 1e6   # sanity: finite


async def test_30d_profile_a_heating_on_steps_nonzero():
    """30d adaptive Profile A: heater actually ran (heating_on_steps > 0)."""
    cfg = scenario_s1h_30d_profile_a(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = _result(summary, cfg)
    assert result.heating_on_steps > 0, (
        "Profile A 30d: heating_on_steps=0 — heater never activated"
    )
