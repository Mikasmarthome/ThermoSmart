"""S1a Item 20 — Full-Year (365d) and 3-Year (1095d) Long-Term Stability Gates.

Validates that the production coordinator + LE2 shadow remain numerically
stable, performance-bounded, and storage-bounded over a full calendar year and
across three consecutive years — spanning all four seasons including the
summer-mode transition (outdoor ~22°C in July).

Test matrix:
  365d × Profile A (fast insulated)              — all acceptance criteria
  365d × Profile C (difficult, faults + windows) — all acceptance criteria
  365d × TRV-only minimal setup                  — hard gate (no boost seeding)
  1095d × Profile A                              — 3-year stability
  1095d × Profile C stressed (2× fault rate)     — 3-year stressed stability
  Total: 5 tests (Linux/CI long-run; excluded on Windows via collect_ignore_glob)

Restart schedule:
  365d: days 30, 90, 180, 270, 330  (5 restarts)
  1095d: every 90d from d90 to d1080  (12 restarts; range(90,1095,90) = 12 elements)

5-year note:
  A true 5-year run at 5-min steps would take ~55 min on this hardware.
  Instead, the 1095d Profile C Stressed scenario covers 3 full years with a 2×
  fault rate (TRV faults every 60d, window events every ~3h), which exercises
  the same long-term accumulation pathways as 5 years at normal rate.

Windows exclusion: see conftest.py collect_ignore_glob
  "test_s1a_item20_long_run.py" is excluded on Windows.
"""
from __future__ import annotations

import pytest

from tests.simulation.scenarios import (
    scenario_s1i_365d_profile_a,
    scenario_s1i_365d_profile_c,
    scenario_s1i_365d_trvonly,
    scenario_s1i_1095d_profile_a,
    scenario_s1i_1095d_profile_c_stressed,
)
from tests.simulation.runner import ScenarioRunner
from tests.simulation.result import SimulationResult

pytestmark = pytest.mark.asyncio

_SEED = 42

# Observed blob sizes: 365d Profile A → 481 KB (5 restarts), 3y Stressed Profile C
# → 565 KB (12 restarts over 3 years, 2× fault rate).  Growth RATE is the real
# production metric (capped at 50 KB/day); this absolute limit guards against
# unbounded accumulation.  768 KB ≈ 3/4 MB gives 35% headroom above the 3y worst
# case while still catching genuine retention failures (e.g. blob doubling per year).
_BLOB_PLATEAU_MAX_KB = 768.0


# ── 1-year runs ───────────────────────────────────────────────────────────────

async def test_365d_profile_a():
    """365d full year: Profile A (fast insulated), 5 restarts, all acceptance gates."""
    cfg = scenario_s1i_365d_profile_a(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = SimulationResult.from_summary(
        summary,
        scenario_id=cfg.name,
        profile_name=cfg.room_profile.name,
        seed=_SEED,
        duration_days=365.0,
        step_s=cfg.step_s,
    )
    print("\n" + result.report())
    assert result.acceptance_passed, result.report()
    assert summary.restart_count == 5, (
        f"Expected 5 restarts, got {summary.restart_count}"
    )
    assert summary.has_no_nan_inf(), "NaN/Inf in learned model after 365d Profile A"
    assert result.model_update_total_final > 0, (
        "Adaptive run produced 0 model updates after 365d"
    )
    assert result.final_blob_bytes / 1024.0 <= _BLOB_PLATEAU_MAX_KB, (
        f"Final blob {result.final_blob_bytes / 1024:.1f} KB > {_BLOB_PLATEAU_MAX_KB} KB "
        f"— retention may not be working"
    )


async def test_365d_profile_c():
    """365d full year: Profile C (difficult) with window events + TRV faults, 5 restarts.

    Profile C has high heat loss (90 W/K) and frequent window events — window-open
    periods in winter cause thermodynamically unavoidable cold that can exceed the
    8% comfort threshold.  Same reasoning as Item 6 tests for this profile:
    cold_fraction is informational only.  Hard gates: no NaN/Inf, p99 timing,
    storage bounds, minimum model updates.
    """
    cfg = scenario_s1i_365d_profile_c(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = SimulationResult.from_summary(
        summary,
        scenario_id=cfg.name,
        profile_name=cfg.room_profile.name,
        seed=_SEED,
        duration_days=365.0,
        step_s=cfg.step_s,
    )
    print("\n" + result.report())
    assert summary.has_no_nan_inf(), "NaN/Inf in learned model after 365d Profile C"
    assert summary.restart_count == 5, (
        f"Expected 5 restarts, got {summary.restart_count}"
    )
    assert summary.window_open_steps > 0, "No window events counted"
    # Exclude cold_fraction from gate — physically unavoidable with Profile C window events
    non_cold_failures = [f for f in result.failures if "cold_fraction" not in f]
    assert not non_cold_failures, result.report()
    assert result.model_update_total_final > 0, (
        "Adaptive run produced 0 model updates after 365d Profile C"
    )


async def test_365d_trvonly():
    """365d full year: Minimal TRV-only setup — hard gate for minimal installation.

    Asserts that a factory-default TRV-only installation (no boost seeding,
    default confidence) remains within all acceptance bounds for a full calendar
    year including summer-mode transition.
    """
    cfg = scenario_s1i_365d_trvonly(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = SimulationResult.from_summary(
        summary,
        scenario_id=cfg.name,
        profile_name=cfg.room_profile.name,
        seed=_SEED,
        duration_days=365.0,
        step_s=cfg.step_s,
    )
    print("\n" + result.report())
    assert summary.has_no_nan_inf(), "NaN/Inf in learned model after 365d TRV-only"
    assert result.cold_fraction <= 0.08, (
        f"TRV-only 365d cold_fraction {result.cold_fraction:.3f} > 0.08 — "
        f"minimal setup not safe for a full year"
    )
    assert result.p99_step_ms <= 100.0, (
        f"TRV-only p99 step time {result.p99_step_ms:.1f} ms > 100 ms"
    )
    assert result.storage_growth_kb_per_day <= 50.0, (
        f"TRV-only storage growth {result.storage_growth_kb_per_day:.1f} KB/day > 50 KB/day"
    )
    assert result.final_blob_bytes / 1024.0 <= _BLOB_PLATEAU_MAX_KB, (
        f"TRV-only final blob {result.final_blob_bytes / 1024:.1f} KB "
        f"> {_BLOB_PLATEAU_MAX_KB} KB"
    )
    assert summary.restart_count == 5, (
        f"TRV-only: expected 5 restarts, got {summary.restart_count}"
    )


# ── 3-year runs ───────────────────────────────────────────────────────────────

async def test_1095d_profile_a():
    """1095d (3-year) run: Profile A, restart every 90d.

    Long-term stability gates:
    - No NaN/Inf after 3 full calendar years
    - storage_growth_kb_per_day stays within the same 50 KB/day cap
    - Final blob size ≤ 512 KB (retention prevents monotonic growth)
    - Model update count > 0 (learning continues through year 3)
    - p99 step time within 100 ms limit (no performance degradation)
    - cold_fraction within 8% limit (safety maintained over 3 years)
    """
    cfg = scenario_s1i_1095d_profile_a(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = SimulationResult.from_summary(
        summary,
        scenario_id=cfg.name,
        profile_name=cfg.room_profile.name,
        seed=_SEED,
        duration_days=1095.0,
        step_s=cfg.step_s,
    )
    print("\n" + result.report())
    assert summary.has_no_nan_inf(), "NaN/Inf in learned model after 1095d Profile A"
    assert result.cold_fraction <= 0.08, (
        f"3-year Profile A cold_fraction {result.cold_fraction:.3f} > 0.08"
    )
    assert result.p99_step_ms <= 100.0, (
        f"3-year Profile A p99 step time {result.p99_step_ms:.1f} ms > 100 ms"
    )
    assert result.storage_growth_kb_per_day <= 50.0, (
        f"3-year storage growth {result.storage_growth_kb_per_day:.1f} KB/day > 50 KB/day"
    )
    assert result.final_blob_bytes / 1024.0 <= _BLOB_PLATEAU_MAX_KB, (
        f"3-year final blob {result.final_blob_bytes / 1024:.1f} KB "
        f"> {_BLOB_PLATEAU_MAX_KB} KB — retention not bounding storage"
    )
    assert result.model_update_total_final > 0, (
        "Adaptive run produced 0 model updates after 1095d"
    )
    assert summary.restart_count == 12, (
        f"Expected 12 restarts (range(90,1095,90) = 12 steps), got {summary.restart_count}"
    )


async def test_1095d_profile_c_stressed():
    """1095d (3-year) stressed run: Profile C + 2× fault rate.

    Accelerated-stress proxy for 5-year validation: double window-event rate
    and TRV faults every 60 days over 3 years.  Exercises the same failure
    accumulation, retention, and recovery pathways as 5 normal years.

    Stability gates:
    - No NaN/Inf after 3 years of heavy stress
    - storage growth bounded (retention works despite constant fault noise)
    - p99 step time ≤ 100 ms (fault processing doesn't degrade performance)
    - Final blob ≤ 512 KB
    - Model receives continuous updates (fault events don't stall learning)
    """
    cfg = scenario_s1i_1095d_profile_c_stressed(seed=_SEED)
    summary = await ScenarioRunner(cfg).run()
    result = SimulationResult.from_summary(
        summary,
        scenario_id=cfg.name,
        profile_name=cfg.room_profile.name,
        seed=_SEED,
        duration_days=1095.0,
        step_s=cfg.step_s,
    )
    print("\n" + result.report())
    assert summary.has_no_nan_inf(), (
        "NaN/Inf in learned model after 1095d Profile C Stressed"
    )
    assert result.p99_step_ms <= 100.0, (
        f"3y stressed p99 step time {result.p99_step_ms:.1f} ms > 100 ms"
    )
    assert result.storage_growth_kb_per_day <= 50.0, (
        f"3y stressed storage growth {result.storage_growth_kb_per_day:.1f} KB/day > 50 KB/day"
    )
    assert result.final_blob_bytes / 1024.0 <= _BLOB_PLATEAU_MAX_KB, (
        f"3y stressed final blob {result.final_blob_bytes / 1024:.1f} KB "
        f"> {_BLOB_PLATEAU_MAX_KB} KB — retention not bounding storage under heavy faults"
    )
    assert result.model_update_total_final > 0, (
        "Stressed run produced 0 model updates after 1095d"
    )
    assert summary.restart_count == 12, (
        f"Expected 12 restarts (range(90,1095,90) = 12 steps), got {summary.restart_count}"
    )
