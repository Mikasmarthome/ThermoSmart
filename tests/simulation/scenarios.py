"""Pre-built scenario factory functions for S1 soak tests.

All scenarios use 5-minute steps over 7 days (2016 ticks).
Start time: 2024-01-08 00:00 UTC = Monday 01:00 CET (UTC+1).
Comfort period (weekdays): 06:00–22:00 local.

Four-mode matrix scenarios (S1a Part 2):
  mode_a  — Learning OFF + Active OFF (INACTIVE): passive room, no ThermoSmart control
  mode_b  — Learning ON  + Active OFF (SHADOW_ONLY): shadow learning only, no dispatch
  mode_c  — Learning OFF + Active ON  (DETERMINISTIC): TPI with static defaults
  mode_d  — Learning ON  + Active ON  (ADAPTIVE): full adaptive (equivalent to S1-A default)

Item 6 scenarios (long-run, 3 room profiles):
  s1g_*_fast_insulated  — Profile A: quick-response, well-insulated
  s1g_*_slow_inertia    — Profile B: heavy mass, significant afterheat
  s1g_*_difficult       — Profile C: high heat loss, window events, noisy sensor
  s1g_*_baseline        — Deterministic baseline for A/B comparison
  Durations: 90-day (summer→autumn) and 180-day (full heating season).
"""
from __future__ import annotations

import math
import random as _random
from datetime import datetime, timezone
from typing import Optional

from tests.simulation.physics import (
    RoomProfile,
    PROFILE_FAST_INSULATED,
    PROFILE_SLOW_INERTIA,
    PROFILE_DIFFICULT,
    PROFILE_CLEAN_AFTERHEAT,
    PROFILE_CLEAN_HEAT_LOSS,
    PROFILE_CLEAN_SUPPLY_DELAY,
)
from tests.simulation.runner import ScenarioConfig

# ── Shared helpers ───────────────────────────────────────────────────────────

_MON_0000_UTC = datetime(2024, 1, 8, 0, 0, 0, tzinfo=timezone.utc)


def _outdoor_sinusoidal(base_c: float = -5.0, amplitude: float = 3.0) -> callable:
    """Realistic day/night outdoor temp: coldest ~02:00, warmest ~14:00."""
    def fn(sim_time: datetime) -> float:
        hour = sim_time.hour + sim_time.minute / 60
        return base_c + amplitude * math.cos(2 * math.pi * (hour - 14.0) / 24.0)
    return fn


def _outdoor_seasonal(
    start_utc: datetime,
    *,
    base_winter_c: float = -5.0,
    base_summer_c: float = 18.0,
    day_amplitude: float = 4.0,
) -> callable:
    """Seasonal + diurnal outdoor temperature for 90/180-day runs.

    Seasonal component: cosine over 365 days.
      - Peak (warmest) at summer solstice: day 172 from Jan 1 ≈ end of June
      - start_utc determines the day-of-year offset at simulation start

    Diurnal component: coldest ~02:00, warmest ~14:00.

    Returns: fn(sim_time) → float (°C)
    """
    start_day_of_year = start_utc.timetuple().tm_yday
    base_mean = (base_winter_c + base_summer_c) / 2.0
    base_range = (base_summer_c - base_winter_c) / 2.0
    # Day 172 = summer solstice (closest to warmest outdoor T)
    summer_day = 172

    def fn(sim_time: datetime) -> float:
        elapsed_s = (sim_time - start_utc).total_seconds()
        elapsed_days = elapsed_s / 86400.0
        day_of_year = (start_day_of_year + elapsed_days) % 365.0
        # Seasonal: cosine — max at summer solstice, min at winter solstice (day ~355)
        seasonal = base_mean + base_range * math.cos(
            2 * math.pi * (day_of_year - summer_day) / 365.0)
        # Diurnal: coldest ~02:00, warmest ~14:00
        hour = sim_time.hour + sim_time.minute / 60.0
        diurnal = day_amplitude * math.cos(2 * math.pi * (hour - 14.0) / 24.0)
        return seasonal + diurnal

    return fn


def _window_schedule_random(
    n_steps: int,
    *,
    seed: int = 99,
    mean_open_duration_steps: int = 6,   # ~30 min for 5-min steps
    mean_closed_duration_steps: int = 72,  # ~6 hours between events
) -> list:
    """Generate a random window open/close schedule as [(step_idx, is_open), ...].

    Uses independent seed so window events are reproducible regardless of
    other random.seed() calls during the run.  Does NOT consume from the global
    RNG during run(); the schedule is pre-computed before the simulation starts.
    """
    rng = _random.Random(seed)
    schedule = []
    step = 0
    while step < n_steps:
        # Next open event
        wait = rng.randint(
            max(1, mean_closed_duration_steps // 2),
            mean_closed_duration_steps * 2
        )
        step += wait
        if step >= n_steps:
            break
        schedule.append((step, True))  # open
        # Duration open
        duration = rng.randint(
            max(1, mean_open_duration_steps // 2),
            mean_open_duration_steps * 3
        )
        step += duration
        if step < n_steps:
            schedule.append((step, False))  # close
    return schedule


def _solar_gain(max_w: float = 150.0) -> callable:
    """Solar gain: bell curve 08:00–17:00 local (UTC passthrough for simplicity)."""
    def fn(sim_time: datetime) -> float:
        hour = sim_time.hour + sim_time.minute / 60
        if 7.0 <= hour <= 16.0:
            return max_w * math.sin(math.pi * (hour - 7.0) / 9.0)
        return 0.0
    return fn


def _standard_profile(**overrides) -> RoomProfile:
    base = dict(
        thermal_mass_kj_per_k=4000.0,
        heat_loss_w_per_k=40.0,   # ~20 m² room, modern insulation; keeps ≤1200W viable
        heating_power_w=1200.0,
        supply_delay_s=300.0,
        afterheat_tau_s=600.0,
        internal_gain_w=30.0,
    )
    base.update(overrides)
    return RoomProfile(**base)


# ── S1-A: Stable room, single TRV ───────────────────────────────────────────

def scenario_s1a(*, seed_boost: bool = False) -> ScenarioConfig:
    """S1-A: standard room, single setpoint TRV, 7 days, no faults."""
    return ScenarioConfig(
        name="S1-A-stable" if not seed_boost else "S1-A-adapted",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_s1a"],
        sensor_entity="sensor.room_s1a",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        seed_boost_model=seed_boost,
        boost_seed_factor_c=1.5,

    )


# ── S1-B: External sensor + longer supply delay ──────────────────────────────

def scenario_s1b() -> ScenarioConfig:
    """S1-B: external temperature sensor, 15-min supply delay."""
    return ScenarioConfig(
        name="S1-B-external-sensor",
        room_profile=_standard_profile(supply_delay_s=900.0),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_s1b"],
        sensor_entity="sensor.external_s1b",
        zone_cfg_overrides={"temp_sensors": ["sensor.external_s1b"]},
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),

    )


# ── S1-C: Afterheat-prone room ───────────────────────────────────────────────

def scenario_s1c() -> ScenarioConfig:
    """S1-C: heavy radiator with 30-min afterheat — tests early cutoff learning."""
    return ScenarioConfig(
        name="S1-C-afterheat",
        room_profile=_standard_profile(heating_power_w=2000.0, afterheat_tau_s=1800.0),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_s1c"],
        sensor_entity="sensor.room_s1c",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(100.0),
        seed_boost_model=True,
        boost_seed_factor_c=2.0,

    )


# ── S1-D: Multi-TRV, partial failure ────────────────────────────────────────

def scenario_s1d() -> ScenarioConfig:
    """S1-D: two TRVs — second TRV becomes unavailable on day 3."""
    fail_step = int(3 * 24 * 3600 / 300)
    return ScenarioConfig(
        name="S1-D-multi-trv",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        zone_cfg_overrides={
            "climate_entities": ["climate.trv_s1d_a", "climate.trv_s1d_b"],
        },
        climate_entities=["climate.trv_s1d_a", "climate.trv_s1d_b"],
        sensor_entity="sensor.room_s1d",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        fault_schedule=[(fail_step, "climate.trv_s1d_b", False)],

    )


# ── S1-E: Restart week ───────────────────────────────────────────────────────

def scenario_s1e() -> ScenarioConfig:
    """S1-E: simulated HA restarts on days 2, 4, 6 — tests persistence/restore."""
    return ScenarioConfig(
        name="S1-E-restart",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_s1e"],
        sensor_entity="sensor.room_s1e",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        seed_boost_model=True,
        boost_seed_factor_c=1.5,
        restart_steps={
            int(2 * 24 * 3600 / 300),
            int(4 * 24 * 3600 / 300),
            int(6 * 24 * 3600 / 300),
        },

    )


# ── Four-mode matrix (S1a Part 2) ────────────────────────────────────────────

def scenario_mode_a() -> ScenarioConfig:
    """Mode A: Learning OFF + Active OFF (INACTIVE) — no ThermoSmart control."""
    return ScenarioConfig(
        name="mode-A-inactive",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_mode_a"],
        sensor_entity="sensor.room_mode_a",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        learning_mode=False,
        active_control=False,
    )


def scenario_mode_b() -> ScenarioConfig:
    """Mode B: Learning ON + Active OFF (SHADOW_ONLY) — shadow learning, no dispatch."""
    return ScenarioConfig(
        name="mode-B-shadow-only",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_mode_b"],
        sensor_entity="sensor.room_mode_b",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        learning_mode=True,
        active_control=False,
    )


def scenario_mode_c() -> ScenarioConfig:
    """Mode C: Learning OFF + Active ON (DETERMINISTIC) — TPI with static defaults only."""
    return ScenarioConfig(
        name="mode-C-deterministic",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_mode_c"],
        sensor_entity="sensor.room_mode_c",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        learning_mode=False,
        active_control=True,
    )


def scenario_mode_d() -> ScenarioConfig:
    """Mode D: Learning ON + Active ON (ADAPTIVE) — full adaptive control."""
    return ScenarioConfig(
        name="mode-D-adaptive",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_mode_d"],
        sensor_entity="sensor.room_mode_d",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        learning_mode=True,
        active_control=True,
    )


# ── S1-F: Self-learning boost path (30 days, no seeded model) ────────────────

def scenario_s1f() -> ScenarioConfig:
    """S1-F: 30 simulated days from scratch — tests natural readiness development.

    Uses boost_bootstrap_prior_c=1.5 to enable the B2b-bootstrap path: the model
    allows up to 3 initial controlled trials (bootstrap_credits=3) using 1.5°C as the
    initial prior offset before any adaptive evidence exists.  After those trials the
    regular evidence-based eligibility path takes over.
    """
    return ScenarioConfig(
        name="S1-F-self-learning",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=30 * 24 * 3600,
        climate_entities=["climate.trv_s1f"],
        sensor_entity="sensor.room_s1f",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        seed_boost_model=False,
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={"boost_bootstrap_prior_c": 1.5},
    )


# ── Long restart equivalence (14 days) ───────────────────────────────────────

def scenario_restart_equiv_continuous() -> ScenarioConfig:
    """14-day continuous run for restart equivalence check."""
    return ScenarioConfig(
        name="restart-equiv-continuous",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=14 * 24 * 3600,
        climate_entities=["climate.trv_req"],
        sensor_entity="sensor.room_req",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        seed_boost_model=True,
        boost_seed_factor_c=1.5,
    )


def scenario_restart_equiv_with_restarts() -> ScenarioConfig:
    """14-day run with restarts at days 3, 7, 10 for equivalence check."""
    return ScenarioConfig(
        name="restart-equiv-restarts",
        room_profile=_standard_profile(),
        initial_room_temp=17.0,
        start_utc=_MON_0000_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=14 * 24 * 3600,
        climate_entities=["climate.trv_rer"],
        sensor_entity="sensor.room_rer",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(150.0),
        seed_boost_model=True,
        boost_seed_factor_c=1.5,
        restart_steps={
            int(3 * 24 * 3600 / 300),
            int(7 * 24 * 3600 / 300),
            int(10 * 24 * 3600 / 300),
        },
    )


# ── Item 6: Extended Self-Learning Scenarios (90 / 180 days) ─────────────────
#
# Start: 2024-10-01 00:00 UTC — beginning of heating season (autumn → winter)
# So 90 days covers Oct–Dec (full autumn + early winter, dropping outdoor temps).
# 180 days covers Oct–Mar (full heating season including coldest period).
#
# Restart points: Day 3, 14, 30, 60, 89 (for 90-day); + 120, 179 (for 180-day).
# Model snapshots every day (288 steps at 5-min).

_HEATING_SEASON_START_UTC = datetime(2024, 10, 1, 0, 0, 0, tzinfo=timezone.utc)


def _restart_steps_days(*days: int, step_s: float = 300.0) -> set:
    """Convert a list of day indices to step indices."""
    return {int(d * 24 * 3600 / step_s) for d in days}


def _steps_per_day(step_s: float = 300.0) -> int:
    return int(86400 / step_s)


# ── Profile A — Fast, insulated room ─────────────────────────────────────────

def scenario_s1g_90d_profile_a(*, seed: int = 42) -> ScenarioConfig:
    """90-day run: Profile A (fast insulated), heating season, full restart schedule."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 90 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    return ScenarioConfig(
        name="S1G-90d-ProfileA",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_a"],
        sensor_entity="sensor.room_s1g_a",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14, 30, 60, 89),
    )


def scenario_s1g_90d_profile_a_baseline(*, seed: int = 42) -> ScenarioConfig:
    """90-day run: Profile A, deterministic baseline (no adaptive control)."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 90 * 24 * 3600
    return ScenarioConfig(
        name="S1G-90d-ProfileA-baseline",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_ab"],
        sensor_entity="sensor.room_s1g_ab",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        model_snapshot_interval_steps=0,
    )


# ── Profile B — Slow, high-inertia room ──────────────────────────────────────

def scenario_s1g_90d_profile_b(*, seed: int = 42) -> ScenarioConfig:
    """90-day run: Profile B (slow inertia), heating season, full restart schedule."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 90 * 24 * 3600
    return ScenarioConfig(
        name="S1G-90d-ProfileB",
        room_profile=PROFILE_SLOW_INERTIA,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_b"],
        sensor_entity="sensor.room_s1g_b",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(80.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14, 30, 60, 89),
    )


def scenario_s1g_90d_profile_b_baseline(*, seed: int = 42) -> ScenarioConfig:
    """90-day baseline: Profile B, deterministic only."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 90 * 24 * 3600
    return ScenarioConfig(
        name="S1G-90d-ProfileB-baseline",
        room_profile=PROFILE_SLOW_INERTIA,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_bb"],
        sensor_entity="sensor.room_s1g_bb",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(80.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        model_snapshot_interval_steps=0,
    )


# ── Profile C — Difficult room (high loss, window events, noisy sensor) ───────

def scenario_s1g_90d_profile_c(*, seed: int = 42) -> ScenarioConfig:
    """90-day run: Profile C (difficult), with window events, full restart schedule."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 90 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    # Window events: ~3 per day on average, each ~30 min open
    win_sched = _window_schedule_random(
        n_steps, seed=seed + 1,
        mean_open_duration_steps=6,
        mean_closed_duration_steps=72,
    )
    return ScenarioConfig(
        name="S1G-90d-ProfileC",
        room_profile=PROFILE_DIFFICULT,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_c"],
        sensor_entity="sensor.room_s1g_c",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-5.0,
                                          base_summer_c=14.0, day_amplitude=5.0),
        solar_gain_fn=_solar_gain(50.0),    # north-facing: less solar gain
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        window_schedule=win_sched,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14, 30, 60, 89),
    )


def scenario_s1g_90d_profile_c_baseline(*, seed: int = 42) -> ScenarioConfig:
    """90-day baseline: Profile C, deterministic only, same window events."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 90 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    win_sched = _window_schedule_random(
        n_steps, seed=seed + 1,
        mean_open_duration_steps=6,
        mean_closed_duration_steps=72,
    )
    return ScenarioConfig(
        name="S1G-90d-ProfileC-baseline",
        room_profile=PROFILE_DIFFICULT,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_cb"],
        sensor_entity="sensor.room_s1g_cb",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-5.0,
                                          base_summer_c=14.0, day_amplitude=5.0),
        solar_gain_fn=_solar_gain(50.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        window_schedule=win_sched,
        model_snapshot_interval_steps=0,
    )


# ── 180-day variants (full heating season Oct → Mar) ─────────────────────────

def scenario_s1g_180d_profile_a(*, seed: int = 42) -> ScenarioConfig:
    """180-day run: Profile A, full heating season (Oct → Mar)."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 180 * 24 * 3600
    return ScenarioConfig(
        name="S1G-180d-ProfileA",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_180a"],
        sensor_entity="sensor.room_s1g_180a",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14, 30, 60, 89, 120, 179),
    )


def scenario_s1g_180d_profile_b(*, seed: int = 42) -> ScenarioConfig:
    """180-day run: Profile B (slow inertia), full heating season."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 180 * 24 * 3600
    return ScenarioConfig(
        name="S1G-180d-ProfileB",
        room_profile=PROFILE_SLOW_INERTIA,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_180b"],
        sensor_entity="sensor.room_s1g_180b",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(80.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14, 30, 60, 89, 120, 179),
    )


def scenario_s1g_180d_profile_c(*, seed: int = 42) -> ScenarioConfig:
    """180-day run: Profile C (difficult), window events, full restart schedule."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 180 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    win_sched = _window_schedule_random(
        n_steps, seed=seed + 1,
        mean_open_duration_steps=6,
        mean_closed_duration_steps=72,
    )
    return ScenarioConfig(
        name="S1G-180d-ProfileC",
        room_profile=PROFILE_DIFFICULT,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_180c"],
        sensor_entity="sensor.room_s1g_180c",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-5.0,
                                          base_summer_c=14.0, day_amplitude=5.0),
        solar_gain_fn=_solar_gain(50.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        window_schedule=win_sched,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14, 30, 60, 89, 120, 179),
    )


# ── 180-day baseline variants ────────────────────────────────────────────────
# Deterministic counterparts to the existing 180-day adaptive scenarios above.
# Same seed, same physics, same outdoor temp function, same window events (Profile C).

def scenario_s1g_180d_profile_a_baseline(*, seed: int = 42) -> ScenarioConfig:
    """180-day baseline: Profile A, deterministic only (no adaptive control)."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 180 * 24 * 3600
    return ScenarioConfig(
        name="S1G-180d-ProfileA-baseline",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_180ab"],
        sensor_entity="sensor.room_s1g_180ab",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        model_snapshot_interval_steps=0,
    )


def scenario_s1g_180d_profile_b_baseline(*, seed: int = 42) -> ScenarioConfig:
    """180-day baseline: Profile B, deterministic only."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 180 * 24 * 3600
    return ScenarioConfig(
        name="S1G-180d-ProfileB-baseline",
        room_profile=PROFILE_SLOW_INERTIA,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_180bb"],
        sensor_entity="sensor.room_s1g_180bb",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(80.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        model_snapshot_interval_steps=0,
    )


def scenario_s1g_180d_profile_c_baseline(*, seed: int = 42) -> ScenarioConfig:
    """180-day baseline: Profile C, deterministic only, same window events."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 180 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    win_sched = _window_schedule_random(
        n_steps, seed=seed + 1,
        mean_open_duration_steps=6,
        mean_closed_duration_steps=72,
    )
    return ScenarioConfig(
        name="S1G-180d-ProfileC-baseline",
        room_profile=PROFILE_DIFFICULT,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1g_180cb"],
        sensor_entity="sensor.room_s1g_180cb",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-5.0,
                                          base_summer_c=14.0, day_amplitude=5.0),
        solar_gain_fn=_solar_gain(50.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        window_schedule=win_sched,
        model_snapshot_interval_steps=0,
    )


# ── Item 8: 30-day A/B Quick Scenarios ───────────────────────────────────────
#
# Short 30-day pairs for fast A/B validation.  Same physics, same seed,
# same window events as the 90-day variants but shorter duration.
# Start: autumn start (Oct 1) → covers the main heating-season onset.

def scenario_s1h_30d_profile_a(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / 30-day adaptive: Profile A (fast insulated), autumn onset."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    return ScenarioConfig(
        name="S1H-30d-ProfileA",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_a"],
        sensor_entity="sensor.room_s1h_a",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14),
    )


def scenario_s1h_30d_profile_a_baseline(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / 30-day baseline: Profile A, deterministic only."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    return ScenarioConfig(
        name="S1H-30d-ProfileA-baseline",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_ab"],
        sensor_entity="sensor.room_s1h_ab",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        zone_cfg_overrides={
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        model_snapshot_interval_steps=0,
    )


def scenario_s1h_30d_profile_b(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / 30-day adaptive: Profile B (slow inertia), autumn onset."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    return ScenarioConfig(
        name="S1H-30d-ProfileB",
        room_profile=PROFILE_SLOW_INERTIA,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_b"],
        sensor_entity="sensor.room_s1h_b",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(80.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14),
    )


def scenario_s1h_30d_profile_b_baseline(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / 30-day baseline: Profile B, deterministic only."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    return ScenarioConfig(
        name="S1H-30d-ProfileB-baseline",
        room_profile=PROFILE_SLOW_INERTIA,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_bb"],
        sensor_entity="sensor.room_s1h_bb",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(80.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        zone_cfg_overrides={
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        model_snapshot_interval_steps=0,
    )


# ── Item 8 Scenario D: Stress / Failure ───────────────────────────────────────
# Profile: Difficult — high heat loss, noisy sensor.
# Additional stress: very frequent window events + TRV fault injection.
# Purpose: prove adaptive degrades gracefully and never becomes worse than baseline.

def scenario_s1h_30d_stress(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / Scenario D: Stress/Failure — adaptive run with heavy faults."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    # Very frequent windows: every ~2h, 30 min each
    win_sched = _window_schedule_random(
        n_steps, seed=seed + 7,
        mean_open_duration_steps=6,
        mean_closed_duration_steps=24,   # ~2h between events (vs 6h normally)
    )
    # TRV fault at days 5, 10, 20 (unavailable for 12h = 144 steps each)
    step_s = 300.0
    _fault = []
    for fault_day in (5, 10, 20):
        fail_step = int(fault_day * 86400 / step_s)
        restore_step = fail_step + 144  # 12h unavailability
        _fault.append((fail_step, "climate.trv_s1h_d1", False))
        if restore_step < n_steps:
            _fault.append((restore_step, "climate.trv_s1h_d1", True))
    return ScenarioConfig(
        name="S1H-30d-Stress",
        room_profile=PROFILE_DIFFICULT,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=step_s,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_d1"],
        sensor_entity="sensor.room_s1h_d",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-5.0,
                                          base_summer_c=14.0, day_amplitude=5.0),
        solar_gain_fn=_solar_gain(50.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
        },
        seed=seed,
        window_schedule=win_sched,
        fault_schedule=_fault,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14),
    )


def scenario_s1h_30d_stress_baseline(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / Scenario D baseline: same stress conditions, deterministic only."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    win_sched = _window_schedule_random(
        n_steps, seed=seed + 7,  # same seed → identical window events
        mean_open_duration_steps=6,
        mean_closed_duration_steps=24,
    )
    step_s = 300.0
    _fault = []
    for fault_day in (5, 10, 20):
        fail_step = int(fault_day * 86400 / step_s)
        restore_step = fail_step + 144
        _fault.append((fail_step, "climate.trv_s1h_d1b", False))
        if restore_step < n_steps:
            _fault.append((restore_step, "climate.trv_s1h_d1b", True))
    return ScenarioConfig(
        name="S1H-30d-Stress-baseline",
        room_profile=PROFILE_DIFFICULT,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=step_s,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_d1b"],
        sensor_entity="sensor.room_s1h_db",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-5.0,
                                          base_summer_c=14.0, day_amplitude=5.0),
        solar_gain_fn=_solar_gain(50.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        window_schedule=win_sched,
        fault_schedule=_fault,
        zone_cfg_overrides={
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
        },
        model_snapshot_interval_steps=0,
    )


# ── Item 8 Scenario E: Minimal TRV-only Setup ────────────────────────────────
# Profile: Fast Insulated — reliable room, single TRV, no extra sensors.
# "Minimal" means: no boost seeding, no zone_cfg enrichments beyond basics.
# The coordinator runs with factory-default confidence thresholds.
# Purpose: prove system is fully functional with pure TRV-based minimal config.

def scenario_s1h_30d_trvonly(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / Scenario E: TRV-only minimal setup, adaptive run."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    return ScenarioConfig(
        name="S1H-30d-TRVonly",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_e"],
        sensor_entity="sensor.room_s1h_e",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        # No boost seeding, no enriched zone config — pure factory defaults
        zone_cfg_overrides={
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
        },
        seed=seed,
        seed_boost_model=False,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14),
    )


def scenario_s1h_30d_trvonly_baseline(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / Scenario E baseline: TRV-only minimal setup, deterministic."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    return ScenarioConfig(
        name="S1H-30d-TRVonly-baseline",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_eb"],
        sensor_entity="sensor.room_s1h_eb",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        zone_cfg_overrides={
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
        },
        model_snapshot_interval_steps=0,
    )


# ── Item 8 Scenario F: Multi-TRV Mixed Zone ──────────────────────────────────
# Two TRVs in the same zone.  One becomes unavailable mid-run (partial failure).
# Purpose: prove learning and control work correctly under device heterogeneity
# and partial failure, and that the zone does not generate incorrect metrics.

def scenario_s1h_30d_multitrvmixed(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / Scenario F: two TRVs, one faults on day 8, adaptive run."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    step_s = 300.0
    # TRV-2 becomes unavailable on day 8, restores on day 15, fails again on day 22
    _fault = []
    for fd, rd in [(8, 15), (22, 28)]:
        fs = int(fd * 86400 / step_s)
        rs = int(rd * 86400 / step_s)
        _fault.append((fs, "climate.trv_s1h_f2", False))
        if rs < n_steps:
            _fault.append((rs, "climate.trv_s1h_f2", True))
    return ScenarioConfig(
        name="S1H-30d-MultiTRVmixed",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=step_s,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_f1", "climate.trv_s1h_f2"],
        sensor_entity="sensor.room_s1h_f",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "climate_entities": ["climate.trv_s1h_f1", "climate.trv_s1h_f2"],
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
        },
        seed=seed,
        fault_schedule=_fault,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(3, 14),
    )


def scenario_s1h_30d_multitrvmixed_baseline(*, seed: int = 42) -> ScenarioConfig:
    """Item 8 / Scenario F baseline: two TRVs, same fault schedule, deterministic."""
    start = _HEATING_SEASON_START_UTC
    duration_s = 30 * 24 * 3600
    n_steps = int(duration_s / 300.0)
    step_s = 300.0
    _fault = []
    for fd, rd in [(8, 15), (22, 28)]:
        fs = int(fd * 86400 / step_s)
        rs = int(rd * 86400 / step_s)
        _fault.append((fs, "climate.trv_s1h_f2b", False))
        if rs < n_steps:
            _fault.append((rs, "climate.trv_s1h_f2b", True))
    return ScenarioConfig(
        name="S1H-30d-MultiTRVmixed-baseline",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=step_s,
        duration_s=duration_s,
        climate_entities=["climate.trv_s1h_f1b", "climate.trv_s1h_f2b"],
        sensor_entity="sensor.room_s1h_fb",
        outdoor_temp_fn=_outdoor_seasonal(start, base_winter_c=-3.0,
                                          base_summer_c=16.0, day_amplitude=4.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=False,
        active_control=True,
        baseline_mode=True,
        seed=seed,
        fault_schedule=_fault,
        zone_cfg_overrides={
            "climate_entities": ["climate.trv_s1h_f1b", "climate.trv_s1h_f2b"],
            "night_temp": 14.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
        },
        model_snapshot_interval_steps=0,
    )


# ── Clean model scenarios (focused, 30-day) ───────────────────────────────────
#
# These scenarios are designed so a single model gets sufficient clean episodes.
# Each scenario uses a room profile and conditions engineered for that model.

_SPRING_START_UTC = datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
_WINTER_START_UTC = datetime(2024, 1, 8, 0, 0, 0, tzinfo=timezone.utc)


def scenario_clean_heat_rate_7d(*, seed: int = 42) -> ScenarioConfig:
    """Clean HeatRate convergence: Profile A, winter, very low noise, 7 days."""
    return ScenarioConfig(
        name="clean-heat-rate-7d",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=_WINTER_START_UTC,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=7 * 24 * 3600,
        climate_entities=["climate.trv_chr"],
        sensor_entity="sensor.room_chr",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 2.0),
        solar_gain_fn=_solar_gain(50.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={"boost_bootstrap_prior_c": 1.5},
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
    )


def scenario_clean_supply_delay_30d(*, seed: int = 42) -> ScenarioConfig:
    """Clean OnsetDelay convergence: heavy mass, 20-min supply delay, clean sensor, 30 days."""
    start = _WINTER_START_UTC
    return ScenarioConfig(
        name="clean-supply-delay-30d",
        room_profile=PROFILE_CLEAN_SUPPLY_DELAY,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=30 * 24 * 3600,
        climate_entities=["climate.trv_csd"],
        sensor_entity="sensor.room_csd",
        outdoor_temp_fn=_outdoor_sinusoidal(-3.0, 3.0),
        solar_gain_fn=_solar_gain(50.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={"boost_bootstrap_prior_c": 1.5},
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(7, 14, 21),
    )


def scenario_clean_afterheat_30d(*, seed: int = 42) -> ScenarioConfig:
    """Clean Afterheat convergence: strong heater, mild spring outdoor, low noise, 30 days.

    Spring outdoor (5°C base) ensures large afterheat signal:
      Q_in(afterheat) = 2500W + 20W; Q_out = 70×17.5 = 1225W; net = +1295W → slope ≈ +0.016°C/min
    This exceeds the heating_slope_eps=0.01°C/min threshold, enabling AFTERHEAT regime.

    Schedule: night setback 22:00→14°C, comfort 06:00→21°C.  The 7°C nightly drop
    (21→14°C) guarantees a daily morning boost preheat from ~14°C → 22.5°C, giving:
      (a) full supply-delay warmup fraction → strong afterheat_w=2500W at cutoff
      (b) slope at T+1: (2500*0.88+20−70×17.5)/5e6×60 ≈ +0.013°C/min > 0.01 ✓
    Drive end detected via controller_demand True→False; the pre-cutoff boost setpoint
    (e.g. 30+°C) is captured via _prev_trv_setpoint so setpoint_before > setpoint_after
    enables _drive_end_reliability() inference without HeatingDriveEndEvent.
    """
    start = _SPRING_START_UTC
    return ScenarioConfig(
        name="clean-afterheat-30d",
        room_profile=PROFILE_CLEAN_AFTERHEAT,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=30 * 24 * 3600,
        climate_entities=["climate.trv_cah"],
        sensor_entity="sensor.room_cah",
        outdoor_temp_fn=_outdoor_sinusoidal(5.0, 3.0),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "comfort_temp": 21.0,
            "night_temp": 14.0,   # 7°C drop → daily boost preheat from 14→22.5°C
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(7, 14, 21),
    )


def scenario_clean_heat_loss_30d(*, seed: int = 42) -> ScenarioConfig:
    """Clean HeatLoss convergence: high heat loss, night setback (22:00→10°C), winter, 30 days.

    Night setback to 10°C ensures PASSIVE_COOLING regime is detectable in the 120-point
    rolling trajectory window:
      Night: target=10°C, room cools from 21°C; rate ≈ 0.041°C/min >> 0.01 threshold
      PASSIVE_COOLING detected at ~step 48 of night (240 min), when 48 negative slopes
      displace enough heating history in the rolling window (median flips negative).
      Room at detection: ≈11.2°C, still well above 10°C target → free cooling episode.
      Day:   target=21°C, TPI heats normally
    A 14°C target would not work: only 34 steps of cooling (171 min) before TPI engages,
    insufficient to flip the 120-point median (needs N > 47.9).
    No window events to avoid contamination.
    """
    start = _WINTER_START_UTC
    return ScenarioConfig(
        name="clean-heat-loss-30d",
        room_profile=PROFILE_CLEAN_HEAT_LOSS,
        initial_room_temp=17.0,
        start_utc=start,
        tz_offset_hours=1.0,
        step_s=300.0,
        duration_s=30 * 24 * 3600,
        climate_entities=["climate.trv_chl"],
        sensor_entity="sensor.room_chl",
        outdoor_temp_fn=_outdoor_sinusoidal(-5.0, 3.0),
        solar_gain_fn=_solar_gain(0.0),   # no solar — pure thermal
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={
            "boost_bootstrap_prior_c": 1.5,
            "night_temp": 10.0,
            "comfort_temp": 21.0,
            "sched_wd_morning": "06:00",
            "sched_wd_night": "22:00",
            "sched_we_morning": "07:00",
            "sched_we_night": "22:00",
        },
        seed=seed,
        model_snapshot_interval_steps=_steps_per_day(300.0),
        restart_steps=_restart_steps_days(7, 14, 21),
    )
