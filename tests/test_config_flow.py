"""Integration tests for ThermoSmart Config Flow (Welle 2).

Requires Linux / Docker / CI (homeassistant.runner needs fcntl).
Run in Docker:
  docker run --rm \\
    -v "<project>:/app" -w /app python:3.12-slim \\
    bash -c "pip install -r requirements_test.txt -q && \\
             python -m pytest tests/test_config_flow.py --override-ini='addopts=' -v"
"""
from __future__ import annotations

from unittest.mock import patch

from homeassistant import config_entries
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

# ── Shared form-step data ─────────────────────────────────────────────────────

_DEVICES = {
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
}

_SCHEDULE = {
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
}

_PRESENCE = {
    CONF_PRESENCE_PERSONS: [],
    CONF_HOME_ZONE: "zone.home",
    CONF_LEARNING_ENABLED: True,
}

_WEATHER_NONE = {
    CONF_WEATHER_ENTITY: None,
    CONF_OUTDOOR_TEMP_SENSOR: None,
    CONF_OUTDOOR_HUMIDITY_SENSOR: None,
    CONF_OUTDOOR_WIND_SENSOR: None,
    CONF_OUTDOOR_SOLAR_SENSOR: None,
    CONF_OUTDOOR_RAIN_SENSOR: None,
}

_PATCH = "custom_components.thermosmart.async_setup_entry"


async def _navigate_to_weather(hass) -> dict:
    """Run menu → add_zone → schedule → presence, return weather-step result."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "add_zone"}
    )
    result = await hass.config_entries.flow.async_configure(result["flow_id"], _DEVICES)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], _SCHEDULE)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], _PRESENCE)
    return result


# ── System Entry ──────────────────────────────────────────────────────────────

class TestSystemFlow:
    async def test_user_step_shows_menu(self, hass, enable_custom_integrations):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        assert result["type"] == FlowResultType.MENU
        assert "add_system" in result["menu_options"]
        assert "add_zone" in result["menu_options"]

    async def test_system_happy_path_creates_entry(self, hass, enable_custom_integrations):
        """add_system creates an entry with entry_type=system in a single step."""
        with patch(_PATCH, return_value=True):
            result = await hass.config_entries.flow.async_init(
                DOMAIN, context={"source": config_entries.SOURCE_USER}
            )
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], {"next_step_id": "add_system"}
            )
            await hass.async_block_till_done()

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "ThermoSmart System"
        assert result["data"] == {"entry_type": "system"}

    async def test_system_duplicate_aborts(self, hass, enable_custom_integrations):
        """A second system entry is rejected with system_already_configured."""
        MockConfigEntry(domain=DOMAIN, data={"entry_type": "system"}).add_to_hass(hass)

        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "add_system"}
        )
        assert result["type"] == FlowResultType.ABORT
        assert result["reason"] == "system_already_configured"

    async def test_system_options_aborts_immediately(self, hass, enable_custom_integrations):
        """Options flow on a system entry aborts with system_no_options."""
        entry = MockConfigEntry(domain=DOMAIN, data={"entry_type": "system"})
        entry.add_to_hass(hass)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        assert result["type"] == FlowResultType.ABORT
        assert result["reason"] == "system_no_options"


# ── Zone Flow – navigation ────────────────────────────────────────────────────

class TestZoneFlowNavigation:
    async def test_add_zone_shows_devices_form(self, hass, enable_custom_integrations):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "add_zone"}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "add_zone"

    async def test_zone_flow_completes_and_creates_entry(self, hass, enable_custom_integrations):
        """Full 4-step happy path creates an entry with correct title and data."""
        with patch(_PATCH, return_value=True):
            result = await _navigate_to_weather(hass)
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], _WEATHER_NONE
            )
            await hass.async_block_till_done()

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "Wohnzimmer"
        assert result["data"]["name"] == "Wohnzimmer"
        assert result["data"]["climate_entities"] == ["climate.test_trv"]

    async def test_zone_flow_entry_contains_schedule_fields(self, hass, enable_custom_integrations):
        with patch(_PATCH, return_value=True):
            result = await _navigate_to_weather(hass)
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], _WEATHER_NONE
            )
            await hass.async_block_till_done()

        data = result["data"]
        assert data["comfort_temp"] == 21.0
        assert data["night_temp"] == 18.0
        assert data["away_temp"] == 17.0
        assert data[CONF_LEARNING_ENABLED] is True

    async def test_zone_flow_entry_has_no_weather_entity_when_omitted(
        self, hass, enable_custom_integrations
    ):
        with patch(_PATCH, return_value=True):
            result = await _navigate_to_weather(hass)
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], _WEATHER_NONE
            )
            await hass.async_block_till_done()

        assert result["data"].get(CONF_WEATHER_ENTITY) is None


# ── Zone Flow – add_zone step validation ──────────────────────────────────────

class TestZoneFlowDeviceValidation:
    async def _to_add_zone(self, hass) -> dict:
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        return await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "add_zone"}
        )

    async def test_empty_name_shows_name_required(self, hass, enable_custom_integrations):
        result = await self._to_add_zone(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {**_DEVICES, "name": ""}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "add_zone"
        assert result["errors"].get("name") == "name_required"

    async def test_whitespace_name_shows_name_required(self, hass, enable_custom_integrations):
        result = await self._to_add_zone(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {**_DEVICES, "name": "   "}
        )
        assert result["errors"].get("name") == "name_required"

    async def test_empty_climate_entities_shows_required(self, hass, enable_custom_integrations):
        result = await self._to_add_zone(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {**_DEVICES, "climate_entities": []}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["errors"].get("climate_entities") == "required"

    async def test_valid_devices_advances_to_schedule(self, hass, enable_custom_integrations):
        result = await self._to_add_zone(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], _DEVICES
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "schedule"


# ── Zone Flow – schedule step validation ──────────────────────────────────────

class TestZoneFlowScheduleValidation:
    async def _to_schedule(self, hass) -> dict:
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "add_zone"}
        )
        return await hass.config_entries.flow.async_configure(result["flow_id"], _DEVICES)

    async def test_night_above_comfort_shows_error(self, hass, enable_custom_integrations):
        result = await self._to_schedule(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {**_SCHEDULE, "comfort_temp": 21.0, "night_temp": 22.0}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "schedule"
        assert result["errors"].get("night_temp") == "night_temp_too_high"

    async def test_away_above_night_shows_error(self, hass, enable_custom_integrations):
        result = await self._to_schedule(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {**_SCHEDULE, "night_temp": 18.0, "away_temp": 19.0}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["errors"].get("away_temp") == "away_temp_too_high"

    async def test_both_errors_shown_simultaneously(self, hass, enable_custom_integrations):
        """night>comfort AND away>night both appear at once."""
        result = await self._to_schedule(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {**_SCHEDULE, "comfort_temp": 20.0, "night_temp": 21.0, "away_temp": 22.0},
        )
        assert "night_temp" in result["errors"]
        assert "away_temp" in result["errors"]

    async def test_valid_schedule_advances_to_presence(self, hass, enable_custom_integrations):
        result = await self._to_schedule(hass)
        result = await hass.config_entries.flow.async_configure(result["flow_id"], _SCHEDULE)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "presence"


# ── Zone Flow – weather step validation ───────────────────────────────────────

class TestZoneFlowWeatherValidation:
    async def test_nonexistent_weather_entity_shows_error(self, hass, enable_custom_integrations):
        result = await _navigate_to_weather(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {**_WEATHER_NONE, CONF_WEATHER_ENTITY: "weather.nonexistent"},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "weather"
        assert result["errors"].get(CONF_WEATHER_ENTITY) == "entity_not_found"

    async def test_no_weather_entity_succeeds(self, hass, enable_custom_integrations):
        with patch(_PATCH, return_value=True):
            result = await _navigate_to_weather(hass)
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], _WEATHER_NONE
            )
            await hass.async_block_till_done()
        assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_existing_weather_entity_succeeds(self, hass, enable_custom_integrations):
        hass.states.async_set("weather.home", "sunny")
        with patch(_PATCH, return_value=True):
            result = await _navigate_to_weather(hass)
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"],
                {**_WEATHER_NONE, CONF_WEATHER_ENTITY: "weather.home"},
            )
            await hass.async_block_till_done()
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_WEATHER_ENTITY] == "weather.home"


# ── Zone Flow – weather defaults from existing entry ──────────────────────────

class TestZoneFlowWeatherDefaults:
    async def test_weather_defaults_copied_from_existing_entry(
        self, hass, enable_custom_integrations
    ):
        """_defaults_from_existing() propagates weather entity into new zone's form defaults."""
        MockConfigEntry(
            domain=DOMAIN,
            data={
                "entry_type": "zone",
                "name": "Schlafzimmer",
                CONF_WEATHER_ENTITY: "weather.home",
                CONF_OUTDOOR_TEMP_SENSOR: "sensor.outdoor_temp",
            },
            options={},
        ).add_to_hass(hass)
        hass.states.async_set("weather.home", "sunny")

        with patch(_PATCH, return_value=True):
            result = await _navigate_to_weather(hass)
            # Submit empty dict → schema fills in defaults from existing entry
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], {}
            )
            await hass.async_block_till_done()

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_WEATHER_ENTITY] == "weather.home"
        assert result["data"].get(CONF_OUTDOOR_TEMP_SENSOR) == "sensor.outdoor_temp"
