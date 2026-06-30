"""Helpers for real Home Assistant fixture tests (Docker/CI only)."""
from __future__ import annotations

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.thermosmart.const import DOMAIN
from tests.helpers import make_zone_config


# ── Single-TRV default setup ─────────────────────────────────────────────────


def seed_states(hass) -> None:
    """Provide the entity states a zone coordinator reads during setup/refresh."""
    hass.states.async_set("weather.home", "sunny",
                          {"temperature": 8.0, "humidity": 70.0, "friendly_name": "Home"})
    hass.states.async_set(
        "climate.test_trv", "heat",
        {"temperature": 20.0, "current_temperature": 19.0, "min_temp": 5.0, "max_temp": 30.0,
         "hvac_modes": ["heat", "off"], "supported_features": 1, "friendly_name": "TRV"})
    hass.states.async_set("sensor.test_temp", "19.0",
                          {"unit_of_measurement": "°C", "device_class": "temperature"})


def make_zone_entry(**overrides) -> MockConfigEntry:
    data = make_zone_config(**overrides)
    return MockConfigEntry(domain=DOMAIN, data=data, options={}, title="Test Zone")


async def setup_zone(hass, **overrides):
    """Set up a real ThermoSmart zone entry; return (ok, entry, entry_data)."""
    seed_states(hass)
    entry = make_zone_entry(**overrides)
    entry.add_to_hass(hass)
    ok = await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
    return ok, entry, entry_data


# ── Multi-TRV helpers ─────────────────────────────────────────────────────────


def seed_multi_trv_states(hass, *,
                           sp_temp: float = 20.0,
                           dv_temp: float = 20.0,
                           current_temp: float = 19.0) -> None:
    """Seed states for a two-TRV zone: one Setpoint-TRV, one Direct-Valve-TRV.

    climate.setpoint_trv  — controlled via climate.set_temperature
    climate.valve_trv     — coordinator maps to number.valve_position (direct valve)
    """
    hass.states.async_set("weather.home", "sunny",
                          {"temperature": 8.0, "humidity": 70.0, "friendly_name": "Home"})
    hass.states.async_set(
        "climate.setpoint_trv", "heat",
        {"temperature": sp_temp, "current_temperature": current_temp,
         "min_temp": 5.0, "max_temp": 30.0,
         "hvac_modes": ["heat", "off"], "supported_features": 1,
         "friendly_name": "Setpoint TRV"})
    hass.states.async_set(
        "climate.valve_trv", "heat",
        {"temperature": dv_temp, "current_temperature": current_temp,
         "min_temp": 5.0, "max_temp": 30.0,
         "hvac_modes": ["heat", "off"], "supported_features": 1,
         "friendly_name": "Valve TRV"})
    hass.states.async_set("sensor.test_temp", str(current_temp),
                          {"unit_of_measurement": "°C", "device_class": "temperature"})
    # Simulate the valve number entity
    hass.states.async_set("number.valve_position", "0",
                          {"min": 0, "max": 100, "step": 1, "unit_of_measurement": "%",
                           "friendly_name": "Valve Position"})


async def setup_multi_trv_zone(hass, **overrides):
    """Set up a zone with setpoint TRV + direct-valve TRV.

    Returns (ok, entry, entry_data, coord) where coord._auto_valve_map is pre-set
    to simulate the Valve TRV having a direct valve number entity.
    """
    seed_multi_trv_states(hass, **{k: v for k, v in overrides.items()
                                    if k in ("sp_temp", "dv_temp", "current_temp")})
    zone_overrides = {k: v for k, v in overrides.items()
                      if k not in ("sp_temp", "dv_temp", "current_temp")}
    zone_overrides.setdefault("climate_entities",
                              ["climate.setpoint_trv", "climate.valve_trv"])
    zone_overrides.setdefault("temp_sensors", ["sensor.test_temp"])

    entry = MockConfigEntry(domain=DOMAIN,
                            data=make_zone_config(**zone_overrides),
                            options={}, title="Multi-TRV Zone")
    entry.add_to_hass(hass)
    ok = await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})

    coord = entry_data.get("coordinator")
    if coord is not None:
        # Simulate auto-detection result: valve_trv maps to the valve number entity.
        # This mirrors what async_detect_device_entities() would produce for a SONOFF TRVZB.
        coord._auto_valve_map = {"climate.valve_trv": "number.valve_position"}

    return ok, entry, entry_data


# ── Service-call capture helpers ──────────────────────────────────────────────


def capture_climate_calls(hass, service: str = "set_temperature"):
    """Return a list that accumulates climate.<service> call data dicts."""
    from pytest_homeassistant_custom_component.common import async_mock_service
    return async_mock_service(hass, "climate", service)


def capture_number_calls(hass, service: str = "set_value"):
    """Return a list that accumulates number.<service> call data dicts."""
    from pytest_homeassistant_custom_component.common import async_mock_service
    return async_mock_service(hass, "number", service)


def set_trv_state(hass, entity_id: str, *,
                  setpoint: float,
                  current_temp: float = 19.0,
                  state: str = "heat") -> None:
    """Update a TRV entity state (simulates TRV reporting new state)."""
    hass.states.async_set(
        entity_id, state,
        {"temperature": setpoint, "current_temperature": current_temp,
         "min_temp": 5.0, "max_temp": 30.0,
         "hvac_modes": ["heat", "off"], "supported_features": 1,
         "friendly_name": entity_id})
