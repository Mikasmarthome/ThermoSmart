"""Helpers for real Home Assistant fixture tests (Docker/CI only)."""
from __future__ import annotations

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.thermosmart.const import DOMAIN
from tests.helpers import make_zone_config


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
    """Set up a real ThermoSmart zone entry; return (entry, entry_data)."""
    seed_states(hass)
    entry = make_zone_entry(**overrides)
    entry.add_to_hass(hass)
    ok = await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
    return ok, entry, entry_data
