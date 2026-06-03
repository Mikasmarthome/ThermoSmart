"""Config flow für ThermoSmart – 4 Schritte pro Zone."""
from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_WEATHER_ENTITY,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR,
    CONF_OUTDOOR_SOLAR_SENSOR,
    CONF_OUTDOOR_RAIN_SENSOR,
    CONF_LEARNING_ENABLED,
    CONF_PRESENCE_PERSONS,
    CONF_HOME_ZONE,
    CONF_VACATION_TEMP,
    CONF_VACATION_BOOLEAN,
    CONF_VALVE_MAINTENANCE,
    CONF_SCHED_WD_MORNING, CONF_SCHED_WD_NIGHT,
    CONF_SCHED_WE_MORNING, CONF_SCHED_WE_NIGHT,
    DEFAULT_LEARNING_ENABLED,
)


# ── Schema-Hilfsfunktionen (eine pro Schritt) ────────────────────────────────

def _schema_devices(d: dict) -> vol.Schema:
    """Schritt 1: Geräte & Sensoren."""
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
            selector.EntitySelector(selector.EntitySelectorConfig(domain="binary_sensor", multiple=True)),

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
    })


def _schema_schedule(d: dict) -> vol.Schema:
    """Schritt 2: Temperaturen & Zeitplan."""
    return vol.Schema({
        # ── Zieltemperaturen ───────────────────────────────────────────
        vol.Required("comfort_temp", default=d.get("comfort_temp", 21.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=30, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("night_temp", default=d.get("night_temp", 18.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=25, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("away_temp", default=d.get("away_temp", 17.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=25, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required(CONF_VACATION_TEMP, default=d.get(CONF_VACATION_TEMP, 12.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=20, step=0.5, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("temp_tolerance", default=d.get("temp_tolerance", 0.5)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=0.1, max=2.0, step=0.1, unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),

        # ── Werktag-Zeitplan ───────────────────────────────────────────
        # ── Werktag-Zeitplan ───────────────────────────────────────────
        vol.Optional(CONF_SCHED_WD_MORNING, default=d.get(CONF_SCHED_WD_MORNING, "06:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WD_NIGHT, default=d.get(CONF_SCHED_WD_NIGHT, "22:00")):
            selector.TimeSelector(),

        # ── Wochenend-Zeitplan ─────────────────────────────────────────
        vol.Optional(CONF_SCHED_WE_MORNING, default=d.get(CONF_SCHED_WE_MORNING, "08:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WE_NIGHT, default=d.get(CONF_SCHED_WE_NIGHT, "23:00")):
            selector.TimeSelector(),
    })


def _schema_presence(d: dict) -> vol.Schema:
    """Schritt 3: Präsenz & Automatik."""
    # EntitySelector validiert leere Strings als ungültige Entity-ID.
    # Lösung: default nur setzen wenn ein Wert existiert – dann sendet HA
    # None statt "" wenn das Feld leer bleibt, und vol.Optional akzeptiert None.
    vacation_existing = d.get(CONF_VACATION_BOOLEAN) or None
    vacation_key = (
        vol.Optional(CONF_VACATION_BOOLEAN, default=vacation_existing)
        if vacation_existing
        else vol.Optional(CONF_VACATION_BOOLEAN)
    )

    return vol.Schema({
        vol.Optional(CONF_PRESENCE_PERSONS, default=d.get(CONF_PRESENCE_PERSONS, [])):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="person", multiple=True)),

        vol.Optional(CONF_HOME_ZONE, default=d.get(CONF_HOME_ZONE, "zone.home")):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="zone")),

        # Wirklich optional – funktioniert mit input_boolean, binary_sensor,
        # calendar, switch … jede Entity deren state == "on"
        vacation_key:
            selector.EntitySelector(selector.EntitySelectorConfig()),

        vol.Required(CONF_LEARNING_ENABLED, default=d.get(CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED)):
            selector.BooleanSelector(),
    })


def _schema_weather(d: dict) -> vol.Schema:
    """Schritt 4: Wetter & Außensensoren (alles optional)."""
    return vol.Schema({
        # Kein Standardwert – Nutzer wählt aktiv seine Wetter-Entity
        vol.Optional(CONF_WEATHER_ENTITY, default=d.get(CONF_WEATHER_ENTITY, "")):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="weather")),

        vol.Optional(CONF_OUTDOOR_TEMP_SENSOR, default=d.get(CONF_OUTDOOR_TEMP_SENSOR, "")):
            selector.EntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class="temperature"
            )),
        vol.Optional(CONF_OUTDOOR_HUMIDITY_SENSOR, default=d.get(CONF_OUTDOOR_HUMIDITY_SENSOR, "")):
            selector.EntitySelector(selector.EntitySelectorConfig(
                domain="sensor", device_class="humidity"
            )),
        vol.Optional(CONF_OUTDOOR_WIND_SENSOR, default=d.get(CONF_OUTDOOR_WIND_SENSOR, "")):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
        vol.Optional(CONF_OUTDOOR_SOLAR_SENSOR, default=d.get(CONF_OUTDOOR_SOLAR_SENSOR, "")):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
        vol.Optional(CONF_OUTDOOR_RAIN_SENSOR, default=d.get(CONF_OUTDOOR_RAIN_SENSOR, "")):
            selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
    })


# ── Config Flow (Ersteinrichtung) ────────────────────────────────────────────

class ThermoSmartConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """4-Schritt-Einrichtung: Ein Eintrag = Eine Heizzone."""

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict = {}

    # Schritt 1: Geräte & Sensoren
    async def async_step_user(self, user_input: dict | None = None):
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
            step_id="user",
            data_schema=_schema_devices(self._data),
            errors=errors,
        )

    # Schritt 2: Temperaturen & Zeitplan
    async def async_step_schedule(self, user_input: dict | None = None):
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_presence()

        return self.async_show_form(
            step_id="schedule",
            data_schema=_schema_schedule(self._data),
        )

    # Schritt 3: Präsenz & Automatik
    async def async_step_presence(self, user_input: dict | None = None):
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_weather()

        return self.async_show_form(
            step_id="presence",
            data_schema=_schema_presence(self._data),
        )

    # Schritt 4: Wetter & Außensensoren
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
        """Wetter-Entity und Außensensoren aus vorhandenem Eintrag vorbelegen."""
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
        return ThermoSmartOptionsFlow()


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
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_presence()

        return self.async_show_form(
            step_id="schedule",
            data_schema=_schema_schedule(current),
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
                # Alle gespeicherten Werte mit geänderten Feldern zusammenführen
                return self.async_create_entry(title="", data={**current, **self._data})

        return self.async_show_form(
            step_id="weather",
            data_schema=_schema_weather(current),
            errors=errors,
        )
