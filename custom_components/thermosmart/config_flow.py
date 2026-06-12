"""Config flow für ThermoSmart."""
from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_CALIBRATION_INVERT,
    CONF_MANAGE_TEMP_SOURCE,
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
    CONF_ECO_TEMP,
    CONF_VALVE_MAINTENANCE,
    CONF_SCHED_WD_MORNING, CONF_SCHED_WD_NIGHT,
    CONF_SCHED_WE_MORNING, CONF_SCHED_WE_NIGHT,
    DEFAULT_LEARNING_ENABLED,
    TEMP_ECO,
)


# ── Hilfklasse: EntitySelector der None und "" als "nicht gesetzt" akzeptiert ─

class _NullableEntitySelector(selector.EntitySelector):
    """EntitySelector that treats None and '' as unset (returns None, no error).

    HA 2026.x sends null for empty optional entity pickers and auto-validates
    data_schema before calling the step handler. The standard EntitySelector
    rejects None via cv.entity_id_or_uuid(None). This subclass short-circuits
    that path while remaining a proper EntitySelector for frontend serialization.
    """
    def __call__(self, data):
        if data is None or data == "":
            return None
        return super().__call__(data)


# ── Validierung ─────────────────────────────────────────────────────────────

def _validate_temps(data: dict) -> dict[str, str]:
    """Cross-field temperature sanity check.

    Rules:
      - night_temp  must be strictly less than comfort_temp
        (night >= comfort would invert the day/night schedule)
      - away_temp   must not exceed night_temp
        (away > night makes no operational sense)

    Error keys map to existing translations in strings.json / de.json:
      night_temp_too_high  →  shown on the night_temp field
      away_temp_too_high   →  shown on the away_temp field
    """
    errors: dict[str, str] = {}
    try:
        comfort = float(data.get("comfort_temp", 21.0))
        night   = float(data.get("night_temp",   18.0))
        away    = float(data.get("away_temp",    17.0))
    except (TypeError, ValueError):
        return errors   # Schema-Validierung ist bereits vorab passiert

    if night >= comfort:
        errors["night_temp"] = "night_temp_too_high"
    if away > night:
        errors["away_temp"] = "away_temp_too_high"
    return errors


# ── Schema-Hilfsfunktionen (eine pro Schritt) ────────────────────────────────

def _schema_devices(d: dict) -> vol.Schema:
    return vol.Schema({
        vol.Required("name", default=d.get("name", "")):
            selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.TEXT)),

        vol.Required("climate_entities", default=d.get("climate_entities", [])):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="climate", multiple=True)),

        vol.Optional("temp_sensors", default=d.get("temp_sensors", [])):
            selector.EntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class="temperature", multiple=True
            )),
        vol.Optional("humidity_sensors", default=d.get("humidity_sensors", [])):
            selector.EntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class="humidity", multiple=True
            )),
        vol.Optional("window_sensors", default=d.get("window_sensors", [])):
            selector.EntitySelector(selector.EntitySelectorConfig(
                domain="binary_sensor", device_class=["window", "opening", "door"], multiple=True
            )),

        vol.Required("window_open_delay", default=d.get("window_open_delay", 5)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=0, max=60, step=1, unit_of_measurement="min", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("window_close_delay", default=d.get("window_close_delay", 2)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=0, max=30, step=1, unit_of_measurement="min", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Optional(CONF_VALVE_MAINTENANCE, default=d.get(CONF_VALVE_MAINTENANCE, True)):
            selector.BooleanSelector(),

        vol.Optional(CONF_CALIBRATION_INVERT, default=d.get(CONF_CALIBRATION_INVERT, False)):
            selector.BooleanSelector(),

        vol.Optional(CONF_MANAGE_TEMP_SOURCE, default=d.get(CONF_MANAGE_TEMP_SOURCE, False)):
            selector.BooleanSelector(),
    })


def _schema_schedule(d: dict) -> vol.Schema:
    return vol.Schema({
        vol.Required("comfort_temp", default=d.get("comfort_temp", 21.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=30, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("night_temp", default=d.get("night_temp", 18.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=30, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("away_temp", default=d.get("away_temp", 17.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=30, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required(CONF_VACATION_TEMP, default=d.get(CONF_VACATION_TEMP, 12.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=20, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required(CONF_ECO_TEMP, default=d.get(CONF_ECO_TEMP, TEMP_ECO)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=15, max=22, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("temp_tolerance", default=d.get("temp_tolerance", 0.5)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=0.1, max=2.0, step=0.1, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required(CONF_SCHEDULE_ENABLED, default=d.get(CONF_SCHEDULE_ENABLED, True)):
            selector.BooleanSelector(),
        vol.Optional(CONF_SCHED_WD_MORNING, default=d.get(CONF_SCHED_WD_MORNING, "06:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WD_NIGHT, default=d.get(CONF_SCHED_WD_NIGHT, "22:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WE_MORNING, default=d.get(CONF_SCHED_WE_MORNING, "08:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WE_NIGHT, default=d.get(CONF_SCHED_WE_NIGHT, "23:00")):
            selector.TimeSelector(),
    })


def _schema_presence(d: dict) -> vol.Schema:
    return vol.Schema({
        vol.Optional(CONF_PRESENCE_PERSONS, default=d.get(CONF_PRESENCE_PERSONS, [])):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="person", multiple=True)),

        vol.Optional(CONF_HOME_ZONE, default=d.get(CONF_HOME_ZONE, "zone.home")):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="zone")),

        vol.Required(CONF_LEARNING_ENABLED, default=d.get(CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED)):
            selector.BooleanSelector(),
    })


def _schema_weather(d: dict) -> vol.Schema:
    return vol.Schema({
        vol.Optional(CONF_WEATHER_ENTITY, default=d.get(CONF_WEATHER_ENTITY) or None):
            _NullableEntitySelector(selector.EntitySelectorConfig(domain="weather")),

        vol.Optional(CONF_OUTDOOR_TEMP_SENSOR, default=d.get(CONF_OUTDOOR_TEMP_SENSOR) or None):
            _NullableEntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class="temperature"
            )),
        vol.Optional(CONF_OUTDOOR_HUMIDITY_SENSOR, default=d.get(CONF_OUTDOOR_HUMIDITY_SENSOR) or None):
            _NullableEntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class="humidity"
            )),
        vol.Optional(CONF_OUTDOOR_WIND_SENSOR, default=d.get(CONF_OUTDOOR_WIND_SENSOR) or None):
            _NullableEntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class="wind_speed"
            )),
        vol.Optional(CONF_OUTDOOR_SOLAR_SENSOR, default=d.get(CONF_OUTDOOR_SOLAR_SENSOR) or None):
            _NullableEntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class="irradiance"
            )),
        vol.Optional(CONF_OUTDOOR_RAIN_SENSOR, default=d.get(CONF_OUTDOOR_RAIN_SENSOR) or None):
            _NullableEntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class=["precipitation", "precipitation_intensity"]
            )),
    })


# ── Config Flow (Ersteinrichtung) ────────────────────────────────────────────

class ThermoSmartConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Einrichtung: Entweder ThermoSmart System (globale Schalter) oder eine Heizzone."""

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict = {}

    # Startmenü: System oder Zone
    async def async_step_user(self, user_input: dict | None = None):
        return self.async_show_menu(
            step_id="user",
            menu_options=["add_system", "add_zone"],
        )

    # Option A: Globale Steuerung einrichten (kein TRV nötig)
    async def async_step_add_system(self, user_input: dict | None = None):
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            if entry.data.get("entry_type") == "system":
                return self.async_abort(reason="system_already_configured")
        return self.async_create_entry(
            title="ThermoSmart System",
            data={"entry_type": "system"},
        )

    # Option B: Heizzone hinzufügen (4-Schritt-Flow)
    async def async_step_add_zone(self, user_input: dict | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            if not user_input.get("name", "").strip():
                errors["name"] = "name_required"
            elif not user_input.get("climate_entities"):
                errors["climate_entities"] = "required"
            else:
                self._data.update(user_input)
                return await self.async_step_schedule()

        return self.async_show_form(
            step_id="add_zone",
            data_schema=_schema_devices(self._data),
            errors=errors,
        )

    async def async_step_schedule(self, user_input: dict | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _validate_temps(user_input)
            if not errors:
                self._data.update(user_input)
                return await self.async_step_presence()

        return self.async_show_form(
            step_id="schedule",
            data_schema=_schema_schedule(self._data),
            errors=errors,
        )

    async def async_step_presence(self, user_input: dict | None = None):
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_weather()

        return self.async_show_form(
            step_id="presence",
            data_schema=_schema_presence(self._data),
        )

    async def async_step_weather(self, user_input: dict | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            weather = user_input.get(CONF_WEATHER_ENTITY, "")
            if weather and self.hass.states.get(weather) is None:
                errors[CONF_WEATHER_ENTITY] = "entity_not_found"
            else:
                self._data.update(user_input)
                return self.async_create_entry(
                    title=self._data["name"],
                    data=self._data,
                )

        defaults = self._defaults_from_existing()
        return self.async_show_form(
            step_id="weather",
            data_schema=_schema_weather({**defaults, **self._data}),
            errors=errors,
        )

    def _defaults_from_existing(self) -> dict:
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            data = {**entry.data, **entry.options}
            if data.get(CONF_WEATHER_ENTITY):
                return {k: data[k] for k in (
                    CONF_WEATHER_ENTITY,
                    CONF_OUTDOOR_TEMP_SENSOR,
                    CONF_OUTDOOR_HUMIDITY_SENSOR,
                    CONF_OUTDOOR_WIND_SENSOR,
                    CONF_OUTDOOR_SOLAR_SENSOR,
                    CONF_OUTDOOR_RAIN_SENSOR,
                ) if data.get(k)}
        return {}

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        if config_entry.data.get("entry_type") == "system":
            return ThermoSmartSystemOptionsFlow()
        return ThermoSmartOptionsFlow()


# ── Options Flow (System-Entry: keine konfigurierbaren Optionen) ─────────────

class ThermoSmartSystemOptionsFlow(config_entries.OptionsFlow):
    async def async_step_init(self, user_input=None):
        return self.async_abort(reason="system_no_options")


# ── Options Flow (Zone bearbeiten) ───────────────────────────────────────────

class ThermoSmartOptionsFlow(config_entries.OptionsFlow):
    """Gleiche 4 Schritte wie die Ersteinrichtung, vorausgefüllt mit aktuellen Werten."""

    def __init__(self) -> None:
        self._data: dict = {}

    def _current(self) -> dict:
        return {**self.config_entry.data, **self.config_entry.options}

    async def async_step_init(self, user_input: dict | None = None):
        errors: dict[str, str] = {}
        current = self._current()
        if user_input is not None:
            if not user_input.get("name", "").strip():
                errors["name"] = "name_required"
            elif not user_input.get("climate_entities"):
                errors["climate_entities"] = "required"
            else:
                self._data.update(user_input)
                return await self.async_step_schedule()

        return self.async_show_form(
            step_id="init",
            data_schema=_schema_devices(current),
            errors=errors,
        )

    async def async_step_schedule(self, user_input: dict | None = None):
        current = self._current()
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _validate_temps(user_input)
            if not errors:
                self._data.update(user_input)
                return await self.async_step_presence()

        return self.async_show_form(
            step_id="schedule",
            data_schema=_schema_schedule(current),
            errors=errors,
        )

    async def async_step_presence(self, user_input: dict | None = None):
        current = self._current()
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_weather()

        return self.async_show_form(
            step_id="presence",
            data_schema=_schema_presence(current),
        )

    async def async_step_weather(self, user_input: dict | None = None):
        errors: dict[str, str] = {}
        current = self._current()
        if user_input is not None:
            weather = user_input.get(CONF_WEATHER_ENTITY, "")
            if weather and self.hass.states.get(weather) is None:
                errors[CONF_WEATHER_ENTITY] = "entity_not_found"
            else:
                self._data.update(user_input)
                return self.async_create_entry(title="", data={**current, **self._data})

        return self.async_show_form(
            step_id="weather",
            data_schema=_schema_weather(current),
            errors=errors,
        )
