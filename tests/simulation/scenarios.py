"""Pre-built scenario factory functions for S1 soak tests.

All scenarios use 5-minute steps over 7 days (2016 ticks).
Start time: 2024-01-08 00:00 UTC = Monday 01:00 CET (UTC+1).
Comfort period (weekdays): 06:00–22:00 local.

Four-mode matrix scenarios (S1a Part 2):
  mode_a  — Learning OFF + Active OFF (INACTIVE): passive room, no ThermoSmart control
  mode_b  — Learning ON  + Active OFF (SHADOW_ONLY): shadow learning only, no dispatch
  mode_c  — Learning OFF + Active ON  (DETERMINISTIC): TPI with static defaults
  mode_d  — Learning ON  + Active ON  (ADAPTIVE): full adaptive (equivalent to S1-A default)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from tests.simulation.physics import RoomProfile
from tests.simulation.runner import ScenarioConfig

# ── Shared helpers ───────────────────────────────────────────────────────────

_MON_0000_UTC = datetime(2024, 1, 8, 0, 0, 0, tzinfo=timezone.utc)


def _outdoor_sinusoidal(base_c: float = -5.0, amplitude: float = 3.0) -> callable:
    """Realistic day/night outdoor temp: coldest ~02:00, warmest ~14:00."""
    def fn(sim_time: datetime) -> float:
        hour = sim_time.hour + sim_time.minute / 60
        return base_c + amplitude * math.cos(2 * math.pi * (hour - 14.0) / 24.0)
    return fn


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
