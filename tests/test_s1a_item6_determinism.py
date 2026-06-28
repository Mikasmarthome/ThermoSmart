"""S1a Item 6 — Determinism proof (Windows + Docker).

Two invariants:
 1. Same seed → identical MetricsSummary metrics across two back-to-back runs.
 2. Different seeds → differences are bounded (no unbounded divergence).

These tests use 7-day runs so they fit on Windows.  They are the canonical
proof that random.seed() is correctly applied before every run() call and that
the simulation produces reproducible physics and learning outcomes.
"""
from __future__ import annotations

import pytest

from tests.simulation.physics import PROFILE_FAST_INSULATED
from tests.simulation.scenarios import _HEATING_SEASON_START_UTC, _outdoor_seasonal, _solar_gain
from tests.simulation.runner import ScenarioConfig, ScenarioRunner

pytestmark = pytest.mark.asyncio

_STEP_S = 300.0
_DUR_7D = 7 * 24 * 3600
_START = _HEATING_SEASON_START_UTC


def _base_cfg(name: str, seed: int) -> ScenarioConfig:
    return ScenarioConfig(
        name=name,
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=_START,
        step_s=_STEP_S,
        duration_s=_DUR_7D,
        climate_entities=[f"climate.trv_det_{name}"],
        sensor_entity=f"sensor.room_det_{name}",
        outdoor_temp_fn=_outdoor_seasonal(_START, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={"boost_bootstrap_prior_c": 1.5},
        seed=seed,
    )


async def test_same_seed_identical_comfort_fraction():
    """Two runs with same seed produce identical comfort_fraction and sensor_readings_sum."""
    cfg1 = _base_cfg("det_s42_r1", seed=42)
    cfg2 = _base_cfg("det_s42_r2", seed=42)
    s1 = await ScenarioRunner(cfg1).run()
    s2 = await ScenarioRunner(cfg2).run()
    assert s1.comfort_fraction == s2.comfort_fraction, (
        f"Same seed 42 → different comfort_fraction: {s1.comfort_fraction} vs {s2.comfort_fraction}"
    )
    assert s1.sensor_readings_sum == s2.sensor_readings_sum, (
        f"Same seed 42 → different sensor_readings_sum: "
        f"{s1.sensor_readings_sum:.4f} vs {s2.sensor_readings_sum:.4f}"
    )


async def test_same_seed_identical_cold_fraction():
    """Two runs with same seed produce identical cold_fraction."""
    cfg1 = _base_cfg("det_cold_r1", seed=42)
    cfg2 = _base_cfg("det_cold_r2", seed=42)
    s1 = await ScenarioRunner(cfg1).run()
    s2 = await ScenarioRunner(cfg2).run()
    assert s1.cold_fraction == s2.cold_fraction, (
        f"Same seed 42 → different cold_fraction: {s1.cold_fraction} vs {s2.cold_fraction}"
    )


async def test_same_seed_identical_total_steps():
    """Two runs with same seed produce identical total_steps."""
    cfg1 = _base_cfg("det_steps_r1", seed=42)
    cfg2 = _base_cfg("det_steps_r2", seed=42)
    s1 = await ScenarioRunner(cfg1).run()
    s2 = await ScenarioRunner(cfg2).run()
    assert s1.total_steps == s2.total_steps


async def test_same_seed_identical_service_call_count():
    """Two runs with same seed dispatch the same number of service calls."""
    cfg1 = _base_cfg("det_svc_r1", seed=42)
    cfg2 = _base_cfg("det_svc_r2", seed=42)
    s1 = await ScenarioRunner(cfg1).run()
    s2 = await ScenarioRunner(cfg2).run()
    assert s1.service_call_count == s2.service_call_count, (
        f"Same seed → different service_call_count: {s1.service_call_count} vs {s2.service_call_count}"
    )


async def test_same_seed_identical_overshoot_fraction():
    """Two runs with same seed produce identical overshoot_fraction."""
    cfg1 = _base_cfg("det_over_r1", seed=42)
    cfg2 = _base_cfg("det_over_r2", seed=42)
    s1 = await ScenarioRunner(cfg1).run()
    s2 = await ScenarioRunner(cfg2).run()
    assert s1.overshoot_fraction == s2.overshoot_fraction


async def test_same_seed_identical_adaptation_mode_counts():
    """Two runs with same seed produce identical adaptation_mode_counts."""
    cfg1 = _base_cfg("det_mode_r1", seed=42)
    cfg2 = _base_cfg("det_mode_r2", seed=42)
    s1 = await ScenarioRunner(cfg1).run()
    s2 = await ScenarioRunner(cfg2).run()
    assert s1.adaptation_mode_counts == s2.adaptation_mode_counts


async def test_different_seeds_bounded_comfort_difference():
    """Different seeds produce bounded (not identical) comfort_fraction difference."""
    cfg42 = _base_cfg("det_seed42", seed=42)
    cfg99 = _base_cfg("det_seed99", seed=99)
    s42 = await ScenarioRunner(cfg42).run()
    s99 = await ScenarioRunner(cfg99).run()
    # Seeds differ → outcomes MAY differ (due to sensor noise)
    # But the difference must stay bounded: noise is small so comfort_fraction
    # should not diverge by more than 0.15 even with different seeds.
    diff = abs(s42.comfort_fraction - s99.comfort_fraction)
    assert diff < 0.15, (
        f"Different seeds diverge more than expected: seed42={s42.comfort_fraction:.3f} "
        f"seed99={s99.comfort_fraction:.3f} diff={diff:.3f}"
    )


async def test_different_seeds_not_all_equal():
    """Different seeds must produce different sensor_readings_sum.

    sensor_readings_sum accumulates every noisy reading: true_temp + N(0, noise_k).
    Because different seeds yield different Gaussian draws, the sum differs even when
    aggregate metrics (heating_wh, comfort) are identical (heating duty-cycle near 100%).

    If this test fails, random.seed() is not applied — the seeding mechanism is broken.
    """
    from tests.simulation.physics import PROFILE_DIFFICULT

    step_s = 300.0
    outdoor_fn = _outdoor_seasonal(_START, base_winter_c=-5.0,
                                   base_summer_c=14.0, day_amplitude=5.0)
    solar_fn = _solar_gain(50.0)

    cfg42 = ScenarioConfig(
        name="det_ne_pc_s42",
        room_profile=PROFILE_DIFFICULT,
        initial_room_temp=17.0,
        start_utc=_START,
        step_s=step_s,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_ne42"],
        sensor_entity="sensor.room_ne42",
        outdoor_temp_fn=outdoor_fn,
        solar_gain_fn=solar_fn,
        learning_mode=True,
        active_control=True,
        seed=42,
    )
    cfg999 = ScenarioConfig(
        name="det_ne_pc_s999",
        room_profile=PROFILE_DIFFICULT,
        initial_room_temp=17.0,
        start_utc=_START,
        step_s=step_s,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_ne999"],
        sensor_entity="sensor.room_ne999",
        outdoor_temp_fn=outdoor_fn,
        solar_gain_fn=solar_fn,
        learning_mode=True,
        active_control=True,
        seed=999,
    )
    s42 = await ScenarioRunner(cfg42).run()
    s999 = await ScenarioRunner(cfg999).run()
    # sensor_readings_sum = Σ(true_temp + noise) over all steps.
    # Different seeds produce different noise draws → different sums.
    assert s42.sensor_readings_sum != s999.sensor_readings_sum, (
        f"Seeds 42 and 999 produced identical sensor_readings_sum "
        f"({s42.sensor_readings_sum:.6f}) — random.seed() may not be applied"
    )


async def test_determinism_with_restart_same_seed():
    """Determinism holds across runs that include restarts."""
    step_s = 300.0
    cfg1 = ScenarioConfig(
        name="det_restart_r1",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=_START,
        step_s=step_s,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_dr1"],
        sensor_entity="sensor.room_dr1",
        outdoor_temp_fn=_outdoor_seasonal(_START, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        seed=42,
        restart_steps={int(3 * 24 * 3600 / step_s)},
    )
    cfg2 = ScenarioConfig(
        name="det_restart_r2",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=_START,
        step_s=step_s,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_dr2"],
        sensor_entity="sensor.room_dr2",
        outdoor_temp_fn=_outdoor_seasonal(_START, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        seed=42,
        restart_steps={int(3 * 24 * 3600 / step_s)},
    )
    s1 = await ScenarioRunner(cfg1).run()
    s2 = await ScenarioRunner(cfg2).run()
    assert s1.restart_count == s2.restart_count == 1
    assert s1.comfort_fraction == s2.comfort_fraction
    assert s1.cold_fraction == s2.cold_fraction
