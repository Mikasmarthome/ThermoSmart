"""Integration tests for ThermoSmart Options Flow (Welle 2).

Requires Linux / Docker / CI (homeassistant.runner needs fcntl).
Run in Docker:
  docker run --rm \\
    -v "<project>:/app" -w /app python:3.12-slim \\
    bash -c "pip install -r requirements_test.txt -q && \\
             python -m pytest tests/test_options_flow.py --override-ini='addopts=' -v"
"""
from __future__ import annotations

from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.thermosmart.const import (
    DOMAIN,
    CONF_WEATHER_ENTITY,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR,
    CONF_OUTDOOR_SOLAR_SENSOR,
    CONF_OUTDOOR_RAIN_SENSOR,
    CONF_LEARNING_ENABLED,
    CONF_SCHEDULE_ENABLED,
    CONF_PRESENCE_PERSONS,
    CONF_HOME_ZONE,
    CONF_VACATION_TEMP,
    CONF_WINDOW_OPEN_TEMP,
    CONF_ECO_TEMP,
    CONF_VALVE_MAINTENANCE,
    CONF_MANAGE_TEMP_SOURCE,
    CONF_CALIBRATION_INVERT,
    CONF_SCHED_WD_MORNING,
    CONF_SCHED_WD_NIGHT,
    CONF_SCHED_WE_MORNING,
    CONF_SCHED_WE_NIGHT,
)

# ── Shared zone entry data ────────────────────────────────────────────────────

_BASE_DATA = {
    "entry_type": "zone",
    "name": "Wohnzimmer",
    "climate_entities": ["climate.test_trv"],
    "temp_sensors": [],
    "humidity_sensors": [],
    "window_sensors": [],
    "window_open_delay": 5,
    "window_close_delay": 2,
    CONF_VALVE_MAINTENANCE: True,
    CONF_MANAGE_TEMP_SOURCE: False,
    CONF_CALIBRATION_INVERT: False,
    "comfort_temp": 21.0,
    "night_temp": 18.0,
    "away_temp": 17.0,
    CONF_VACATION_TEMP: 12.0,
    CONF_ECO_TEMP: 18.0,
    CONF_WINDOW_OPEN_TEMP: 12.0,
    "temp_tolerance": 0.5,
    CONF_SCHEDULE_ENABLED: True,
    CONF_SCHED_WD_MORNING: "06:00:00",
    CONF_SCHED_WD_NIGHT: "22:00:00",
    CONF_SCHED_WE_MORNING: "08:00:00",
    CONF_SCHED_WE_NIGHT: "23:00:00",
    CONF_PRESENCE_PERSONS: [],
    CONF_HOME_ZONE: "zone.home",
    CONF_LEARNING_ENABLED: True,
    CONF_WEATHER_ENTITY: None,
    CONF_OUTDOOR_TEMP_SENSOR: None,
    CONF_OUTDOOR_HUMIDITY_SENSOR: None,
    CONF_OUTDOOR_WIND_SENSOR: None,
    CONF_OUTDOOR_SOLAR_SENSOR: None,
    CONF_OUTDOOR_RAIN_SENSOR: None,
}

# Form-step inputs matching the current values (no changes)
_STEP_INIT = {k: _BASE_DATA[k] for k in [
    "name", "climate_entities", "temp_sensors", "humidity_sensors",
    "window_sensors", "window_open_delay", "window_close_delay",
    CONF_VALVE_MAINTENANCE, CONF_MANAGE_TEMP_SOURCE, CONF_CALIBRATION_INVERT,
]}
_STEP_SCHEDULE = {k: _BASE_DATA[k] for k in [
    "comfort_temp", "night_temp", "away_temp", CONF_VACATION_TEMP, CONF_ECO_TEMP,
    CONF_WINDOW_OPEN_TEMP, "temp_tolerance", CONF_SCHEDULE_ENABLED,
    CONF_SCHED_WD_MORNING, CONF_SCHED_WD_NIGHT, CONF_SCHED_WE_MORNING, CONF_SCHED_WE_NIGHT,
]}
_STEP_PRESENCE = {k: _BASE_DATA[k] for k in [
    CONF_PRESENCE_PERSONS, CONF_HOME_ZONE, CONF_LEARNING_ENABLED,
]}
_STEP_WEATHER = {k: _BASE_DATA[k] for k in [
    CONF_WEATHER_ENTITY, CONF_OUTDOOR_TEMP_SENSOR, CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR, CONF_OUTDOOR_SOLAR_SENSOR, CONF_OUTDOOR_RAIN_SENSOR,
]}


def _zone_entry(**overrides) -> MockConfigEntry:
    data = {**_BASE_DATA, **overrides}
    return MockConfigEntry(domain=DOMAIN, data=data, options={})


async def _run_options_flow(hass, entry) -> dict:
    """Run the full 4-step options flow with unchanged values, return final result."""
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _STEP_INIT
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _STEP_SCHEDULE
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _STEP_PRESENCE
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _STEP_WEATHER
    )
    return result


# ── System Options ────────────────────────────────────────────────────────────

class TestSystemOptionsFlow:
    async def test_system_options_aborts_immediately(self, hass, enable_custom_integrations):
        """Options flow for the system entry aborts with system_no_options."""
        entry = MockConfigEntry(domain=DOMAIN, data={"entry_type": "system"})
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        assert result["type"] == FlowResultType.ABORT
        assert result["reason"] == "system_no_options"


# ── Zone Options – basic ──────────────────────────────────────────────────────

class TestZoneOptionsFlowBasic:
    async def test_init_step_is_a_form(self, hass, enable_custom_integrations):
        """Options flow starts with a form step (init)."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

    async def test_happy_path_returns_create_entry(self, hass, enable_custom_integrations):
        """Full 4-step options flow ends with CREATE_ENTRY."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await _run_options_flow(hass, entry)
        assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_options_flow_has_four_steps(self, hass, enable_custom_integrations):
        """Options flow traverses init → schedule → presence → weather."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        r1 = await hass.config_entries.options.async_init(entry.entry_id)
        assert r1["step_id"] == "init"

        r2 = await hass.config_entries.options.async_configure(r1["flow_id"], _STEP_INIT)
        assert r2["step_id"] == "schedule"

        r3 = await hass.config_entries.options.async_configure(r2["flow_id"], _STEP_SCHEDULE)
        assert r3["step_id"] == "presence"

        r4 = await hass.config_entries.options.async_configure(r3["flow_id"], _STEP_PRESENCE)
        assert r4["step_id"] == "weather"

        r5 = await hass.config_entries.options.async_configure(r4["flow_id"], _STEP_WEATHER)
        assert r5["type"] == FlowResultType.CREATE_ENTRY


# ── Zone Options – pre-fill ───────────────────────────────────────────────────

class TestZoneOptionsFlowPrefill:
    async def test_entry_type_preserved_in_result(self, hass, enable_custom_integrations):
        """entry_type=zone survives the options flow (it is not editable)."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await _run_options_flow(hass, entry)
        assert result["data"]["entry_type"] == "zone"

    async def test_options_from_data_field_are_visible_in_result(
        self, hass, enable_custom_integrations
    ):
        """Fields stored in entry.data are merged into the options-flow result."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await _run_options_flow(hass, entry)
        assert result["data"]["name"] == "Wohnzimmer"
        assert result["data"]["comfort_temp"] == 21.0

    async def test_options_field_overrides_data_field(self, hass, enable_custom_integrations):
        """If the same key exists in both entry.data and entry.options, options wins."""
        entry = MockConfigEntry(
            domain=DOMAIN,
            data={**_BASE_DATA},
            options={"comfort_temp": 22.0},
        )
        entry.add_to_hass(hass)

        # The init form is pre-filled from _current() = {**data, **options}
        # We re-submit without changing comfort_temp — the schema default from options wins.
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], _STEP_INIT
        )
        # Submit schedule with the options-overridden comfort_temp
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {**_STEP_SCHEDULE, "comfort_temp": 22.0}
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], _STEP_PRESENCE
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], _STEP_WEATHER
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["comfort_temp"] == 22.0


# ── Zone Options – validation ─────────────────────────────────────────────────

class TestZoneOptionsFlowValidation:
    async def test_empty_name_shows_name_required(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {**_STEP_INIT, "name": ""}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"
        assert result["errors"].get("name") == "name_required"

    async def test_empty_climate_entities_shows_required(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {**_STEP_INIT, "climate_entities": []}
        )
        assert result["errors"].get("climate_entities") == "required"

    async def test_night_above_comfort_shows_error(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        await hass.config_entries.options.async_configure(result["flow_id"], _STEP_INIT)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {**_STEP_SCHEDULE, "comfort_temp": 21.0, "night_temp": 22.0},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["errors"].get("night_temp") == "night_temp_too_high"

    async def test_away_above_night_shows_error(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        await hass.config_entries.options.async_configure(result["flow_id"], _STEP_INIT)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {**_STEP_SCHEDULE, "night_temp": 18.0, "away_temp": 19.0},
        )
        assert result["errors"].get("away_temp") == "away_temp_too_high"

    async def test_invalid_weather_entity_shows_error(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        await hass.config_entries.options.async_configure(result["flow_id"], _STEP_INIT)
        await hass.config_entries.options.async_configure(result["flow_id"], _STEP_SCHEDULE)
        await hass.config_entries.options.async_configure(result["flow_id"], _STEP_PRESENCE)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {**_STEP_WEATHER, CONF_WEATHER_ENTITY: "weather.nonexistent"},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "weather"
        assert result["errors"].get(CONF_WEATHER_ENTITY) == "entity_not_found"


# ── Zone Options – merge behavior ─────────────────────────────────────────────

class TestZoneOptionsFlowMerge:
    async def test_result_contains_updated_name(self, hass, enable_custom_integrations):
        """Changed name appears in options-flow result."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {**_STEP_INIT, "name": "Neuer Name"}
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], _STEP_SCHEDULE
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], _STEP_PRESENCE
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], _STEP_WEATHER
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["name"] == "Neuer Name"

    async def test_unrelated_fields_survive_options_update(self, hass, enable_custom_integrations):
        """Fields not touched by any form step (entry_type) remain in result."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await _run_options_flow(hass, entry)
        # entry_type is in entry.data but never appears in any options-flow form
        assert result["data"]["entry_type"] == "zone"

    async def test_new_options_override_original_data(self, hass, enable_custom_integrations):
        """Changed comfort_temp in options flow overrides original value from entry.data."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        await hass.config_entries.options.async_configure(result["flow_id"], _STEP_INIT)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {**_STEP_SCHEDULE, "comfort_temp": 23.0}
        )
        await hass.config_entries.options.async_configure(result["flow_id"], _STEP_PRESENCE)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], _STEP_WEATHER
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["comfort_temp"] == 23.0

    async def test_result_data_is_complete_not_partial(self, hass, enable_custom_integrations):
        """Options-flow result contains all zone fields, not just the changed ones."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await _run_options_flow(hass, entry)
        data = result["data"]
        assert "name" in data
        assert "climate_entities" in data
        assert "comfort_temp" in data
        assert CONF_LEARNING_ENABLED in data
        assert CONF_WEATHER_ENTITY in data
