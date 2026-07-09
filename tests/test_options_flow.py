"""Integration tests for ThermoSmart Options Hub (S1a Item 18).

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

_STEP_DEVICES = {k: _BASE_DATA[k] for k in [
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
    CONF_PRESENCE_PERSONS, CONF_HOME_ZONE,
]}
_STEP_WEATHER = {k: _BASE_DATA[k] for k in [
    CONF_WEATHER_ENTITY, CONF_OUTDOOR_TEMP_SENSOR, CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR, CONF_OUTDOOR_SOLAR_SENSOR, CONF_OUTDOOR_RAIN_SENSOR,
]}


def _zone_entry(**overrides) -> MockConfigEntry:
    data = {**_BASE_DATA, **overrides}
    return MockConfigEntry(domain=DOMAIN, data=data, options={})


async def _open_hub(hass, entry) -> dict:
    """Open the options hub (async_step_init), return the menu result."""
    return await hass.config_entries.options.async_init(entry.entry_id)


async def _select_section(hass, hub_result, section: str) -> dict:
    """Select a hub section from the menu, return the section form result."""
    return await hass.config_entries.options.async_configure(
        hub_result["flow_id"], {"next_step_id": section}
    )


async def _save_section(hass, form_result, data: dict) -> dict:
    """Submit a section form, return the CREATE_ENTRY result."""
    return await hass.config_entries.options.async_configure(
        form_result["flow_id"], data
    )


# ── System Options ────────────────────────────────────────────────────────────


class TestSystemOptionsFlow:
    async def test_system_options_aborts_immediately(self, hass, enable_custom_integrations):
        """Options flow for the system entry aborts with system_no_options."""
        entry = MockConfigEntry(domain=DOMAIN, data={"entry_type": "system"})
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        assert result["type"] == FlowResultType.ABORT
        assert result["reason"] == "system_no_options"


# ── Hub Entry Point ───────────────────────────────────────────────────────────


class TestOptionsHub:
    async def test_init_step_shows_menu(self, hass, enable_custom_integrations):
        """Options flow now starts with a menu (hub), not a form."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await _open_hub(hass, entry)
        assert result["type"] == FlowResultType.MENU
        assert result["step_id"] == "init"

    async def test_hub_menu_has_four_sections(self, hass, enable_custom_integrations):
        """Hub menu exposes all four sections."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await _open_hub(hass, entry)
        for section in ("devices", "schedule", "presence", "weather"):
            assert section in result["menu_options"], f"section '{section}' missing"

    async def test_hub_devices_section_shows_form(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        result = await _select_section(hass, hub, "devices")
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "devices"

    async def test_hub_schedule_section_shows_form(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        result = await _select_section(hass, hub, "schedule")
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "schedule"

    async def test_hub_presence_section_shows_form(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        result = await _select_section(hass, hub, "presence")
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "presence"

    async def test_hub_weather_section_shows_form(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        result = await _select_section(hass, hub, "weather")
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "weather"

    async def test_devices_section_saves_immediately(self, hass, enable_custom_integrations):
        """Devices section returns CREATE_ENTRY after one submit — no 4-step gauntlet."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "devices")
        result = await _save_section(hass, form, _STEP_DEVICES)
        assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_schedule_section_saves_immediately(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "schedule")
        result = await _save_section(hass, form, _STEP_SCHEDULE)
        assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_presence_section_saves_immediately(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "presence")
        result = await _save_section(hass, form, _STEP_PRESENCE)
        assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_weather_section_saves_immediately(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "weather")
        result = await _save_section(hass, form, _STEP_WEATHER)
        assert result["type"] == FlowResultType.CREATE_ENTRY


# ── Partial Save — Unrelated Fields Preserved ─────────────────────────────────


class TestOptionsHubPartialSave:
    async def test_devices_save_preserves_schedule_fields(self, hass, enable_custom_integrations):
        """Saving the devices section must not erase comfort_temp."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "devices")
        result = await _save_section(hass, form, _STEP_DEVICES)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["comfort_temp"] == 21.0
        assert result["data"][CONF_LEARNING_ENABLED] is True

    async def test_schedule_save_preserves_devices_fields(self, hass, enable_custom_integrations):
        """Saving the schedule section must not erase climate_entities."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "schedule")
        result = await _save_section(hass, form, _STEP_SCHEDULE)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["climate_entities"] == ["climate.test_trv"]
        assert result["data"]["name"] == "Wohnzimmer"

    async def test_presence_save_preserves_all_other_fields(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "presence")
        result = await _save_section(hass, form, _STEP_PRESENCE)

        assert result["data"]["comfort_temp"] == 21.0
        assert result["data"]["climate_entities"] == ["climate.test_trv"]
        # Config-Flow UX Cleanup: the presence section no longer asks for
        # learning_enabled, but a value already stored on the entry (backward
        # compatibility) must survive an unrelated section save untouched.
        assert result["data"][CONF_LEARNING_ENABLED] is True

    async def test_weather_save_preserves_all_other_fields(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "weather")
        result = await _save_section(hass, form, _STEP_WEATHER)

        assert result["data"]["comfort_temp"] == 21.0
        assert result["data"]["name"] == "Wohnzimmer"

    async def test_result_data_is_complete_after_single_section(
        self, hass, enable_custom_integrations
    ):
        """Even after saving just one section, result contains all zone fields."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "devices")
        result = await _save_section(hass, form, _STEP_DEVICES)

        data = result["data"]
        assert "name" in data
        assert "climate_entities" in data
        assert "comfort_temp" in data
        assert CONF_LEARNING_ENABLED in data
        assert CONF_WEATHER_ENTITY in data

    async def test_entry_type_preserved_through_hub_save(self, hass, enable_custom_integrations):
        """entry_type=zone survives a partial hub save."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "devices")
        result = await _save_section(hass, form, _STEP_DEVICES)
        assert result["data"]["entry_type"] == "zone"

    async def test_changed_field_written_by_section_save(self, hass, enable_custom_integrations):
        """Changing comfort_temp in schedule section persists the new value."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "schedule")
        result = await _save_section(hass, form, {**_STEP_SCHEDULE, "comfort_temp": 23.0})

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["comfort_temp"] == 23.0

    async def test_options_field_overrides_data_field(self, hass, enable_custom_integrations):
        """If the same key exists in entry.data and entry.options, options wins."""
        entry = MockConfigEntry(
            domain=DOMAIN,
            data={**_BASE_DATA},
            options={"comfort_temp": 22.0},
        )
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "schedule")
        # Re-submit with the options-overridden value
        result = await _save_section(hass, form, {**_STEP_SCHEDULE, "comfort_temp": 22.0})
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["comfort_temp"] == 22.0


# ── Validation within hub sections ───────────────────────────────────────────


class TestOptionsHubValidation:
    async def test_devices_empty_name_shows_name_required(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "devices")
        result = await _save_section(hass, form, {**_STEP_DEVICES, "name": ""})
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "devices"
        assert result["errors"].get("name") == "name_required"

    async def test_devices_empty_climate_entities_shows_required(
        self, hass, enable_custom_integrations
    ):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "devices")
        result = await _save_section(hass, form, {**_STEP_DEVICES, "climate_entities": []})
        assert result["errors"].get("climate_entities") == "required"

    async def test_schedule_night_above_comfort_shows_error(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "schedule")
        result = await _save_section(
            hass, form, {**_STEP_SCHEDULE, "comfort_temp": 21.0, "night_temp": 22.0}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["errors"].get("night_temp") == "night_temp_too_high"

    async def test_schedule_away_above_night_shows_error(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "schedule")
        result = await _save_section(
            hass, form, {**_STEP_SCHEDULE, "night_temp": 18.0, "away_temp": 19.0}
        )
        assert result["errors"].get("away_temp") == "away_temp_too_high"

    async def test_weather_invalid_entity_shows_error(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "weather")
        result = await _save_section(
            hass, form, {**_STEP_WEATHER, CONF_WEATHER_ENTITY: "weather.nonexistent"}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "weather"
        assert result["errors"].get(CONF_WEATHER_ENTITY) == "entity_not_found"

    async def test_weather_valid_entity_succeeds(self, hass, enable_custom_integrations):
        hass.states.async_set("weather.home", "sunny")
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "weather")
        result = await _save_section(
            hass, form, {**_STEP_WEATHER, CONF_WEATHER_ENTITY: "weather.home"}
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_WEATHER_ENTITY] == "weather.home"

    async def test_weather_no_entity_succeeds(self, hass, enable_custom_integrations):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "weather")
        result = await _save_section(hass, form, _STEP_WEATHER)
        assert result["type"] == FlowResultType.CREATE_ENTRY


# ── Old config compatibility ──────────────────────────────────────────────────


class TestOptionsHubOldConfigCompatibility:
    async def test_old_flat_entry_opens_in_hub(self, hass, enable_custom_integrations):
        """An existing flat entry (all fields in data, empty options) opens the hub correctly."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        assert result["type"] == FlowResultType.MENU

    async def test_entry_with_only_partial_options_handled_safely(
        self, hass, enable_custom_integrations
    ):
        """Entry with options containing only a subset of keys works fine."""
        entry = MockConfigEntry(
            domain=DOMAIN,
            data={**_BASE_DATA},
            options={"comfort_temp": 22.0},
        )
        entry.add_to_hass(hass)

        hub = await hass.config_entries.options.async_init(entry.entry_id)
        assert hub["type"] == FlowResultType.MENU

    async def test_empty_options_entry_handled_safely(self, hass, enable_custom_integrations):
        """Entry with empty options dict opens hub without crash."""
        entry = MockConfigEntry(domain=DOMAIN, data={**_BASE_DATA}, options={})
        entry.add_to_hass(hass)

        hub = await hass.config_entries.options.async_init(entry.entry_id)
        assert hub["type"] == FlowResultType.MENU

    async def test_name_change_via_devices_section_from_old_entry(
        self, hass, enable_custom_integrations
    ):
        """Old flat entry can have its name changed via the hub's devices section."""
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "devices")
        result = await _save_section(hass, form, {**_STEP_DEVICES, "name": "Neuer Name"})
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["name"] == "Neuer Name"

    async def test_legacy_learning_enabled_key_survives_without_ui_field(
        self, hass, enable_custom_integrations
    ):
        """Config-Flow UX Cleanup: an entry created before the global learning
        question was removed from the UI (learning_enabled=False stored in
        data) must keep loading and saving correctly — no crash, no forced
        migration, no KeyError, even though no step asks for this field
        anymore."""
        entry = MockConfigEntry(
            domain=DOMAIN, data={**_BASE_DATA, CONF_LEARNING_ENABLED: False}, options={},
        )
        entry.add_to_hass(hass)

        hub = await hass.config_entries.options.async_init(entry.entry_id)
        assert hub["type"] == FlowResultType.MENU

        form = await _select_section(hass, hub, "weather")
        result = await _save_section(hass, form, _STEP_WEATHER)
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_LEARNING_ENABLED] is False

    async def test_presence_section_schema_has_no_learning_field(
        self, hass, enable_custom_integrations
    ):
        entry = _zone_entry()
        entry.add_to_hass(hass)

        hub = await _open_hub(hass, entry)
        form = await _select_section(hass, hub, "presence")
        schema_keys = {str(k) for k in form["data_schema"].schema}
        assert CONF_LEARNING_ENABLED not in schema_keys
