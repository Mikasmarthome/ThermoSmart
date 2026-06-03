"""Config flow für ThermoSmart – ein Eintrag pro Zone."""
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
    CONF_VACATION_BOOLEAN,
    CONF_CALIBRATION_ENTITIES,
    CONF_QUIRK_ENTITIES,
    CONF_VALVE_MAINTENANCE,
    CONF_SCHED_WD_MORNING, CONF_SCHED_WD_DAY, CONF_SCHED_WD_DAY_TEMP,
    CONF_SCHED_WD_EVENING, CONF_SCHED_WD_NIGHT,
    CONF_SCHED_WE_MORNING, CONF_SCHED_WE_NIGHT,
    DEFAULT_WEATHER_ENTITY,
    DEFAULT_LEARNING_ENABLED,
)


class ThermoSmartConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Ersteinrichtung: Ein Eintrag = Eine Heizzone."""

    VERSION = 1

    async def async_step_user(self, user_input: dict | None = None):
        errors: dict[str, str] = {}

        if user_input is not None:
            if not user_input.get("name", "").strip():
                errors["name"] = "name_required"
            elif not user_input.get("climate_entities"):
                errors["climate_entities"] = "required"
            elif self.hass.states.get(user_input.get(CONF_WEATHER_ENTITY, "")) is None:
                errors[CONF_WEATHER_ENTITY] = "entity_not_found"
            else:
                return self.async_create_entry(
                    title=user_input["name"],
                    data=user_input,
                )

        # Wetter & Außensensoren aus bestehenden Einträgen vorbelegen
        defaults = self._defaults_from_existing()

        return self.async_show_form(
            step_id="user",
            data_schema=_zone_schema(defaults),
            errors=errors,
        )

    def _defaults_from_existing(self) -> dict:
        """Wetter-Entity und Außensensoren aus vorhandenem Eintrag übernehmen."""
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


class ThermoSmartOptionsFlow(config_entries.OptionsFlow):
    """Zone bearbeiten – alle Einstellungen änderbar."""

    async def async_step_init(self, user_input: dict | None = None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(
            step_id="init",
            data_schema=_zone_schema(current),
        )


def _zone_schema(d: dict | None = None) -> vol.Schema:
    """Vollständiges Zonen-Schema für Ersteinrichtung und Bearbeitung."""
    d = d or {}
    return vol.Schema({

        # ── Zonenname ────────────────────────────────────────────────
        vol.Required("name", default=d.get("name", "")):
            selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.TEXT)
            ),

        # ── Thermostate / TRVs ────────────────────────────────────────
        vol.Required("climate_entities", default=d.get("climate_entities", [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(domain="climate", multiple=True)
            ),
        vol.Optional(CONF_CALIBRATION_ENTITIES, default=d.get(CONF_CALIBRATION_ENTITIES, [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(domain="number", multiple=True)
            ),
        vol.Optional(CONF_QUIRK_ENTITIES, default=d.get(CONF_QUIRK_ENTITIES, [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(domain="switch", multiple=True)
            ),
        vol.Optional(CONF_VALVE_MAINTENANCE, default=d.get(CONF_VALVE_MAINTENANCE, True)):
            selector.BooleanSelector(),

        # ── Innensensoren ─────────────────────────────────────────────
        vol.Optional("temp_sensors", default=d.get("temp_sensors", [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor", device_class="temperature", multiple=True
                )
            ),
        vol.Optional("humidity_sensors", default=d.get("humidity_sensors", [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor", device_class="humidity", multiple=True
                )
            ),

        # ── Fenstersensoren ───────────────────────────────────────────
        vol.Optional("window_sensors", default=d.get("window_sensors", [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(domain="binary_sensor", multiple=True)
            ),

        # ── Präsenz & Urlaub ──────────────────────────────────────────
        vol.Optional(CONF_PRESENCE_PERSONS, default=d.get(CONF_PRESENCE_PERSONS, [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(domain="person", multiple=True)
            ),
        vol.Optional(CONF_HOME_ZONE, default=d.get(CONF_HOME_ZONE, "zone.home")):
            selector.EntitySelector(
                selector.EntitySelectorConfig(domain="zone")
            ),
        vol.Optional(CONF_VACATION_BOOLEAN, default=d.get(CONF_VACATION_BOOLEAN, "")):
            selector.EntitySelector(
                selector.EntitySelectorConfig()
            ),

        # ── Wetter-Entity ─────────────────────────────────────────────
        vol.Required(
            CONF_WEATHER_ENTITY,
            default=d.get(CONF_WEATHER_ENTITY, DEFAULT_WEATHER_ENTITY),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="weather")
        ),

        # ── Außensensoren (eigene Wetterstation) ──────────────────────
        vol.Optional(
            CONF_OUTDOOR_TEMP_SENSOR,
            default=d.get(CONF_OUTDOOR_TEMP_SENSOR, ""),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
        ),
        vol.Optional(
            CONF_OUTDOOR_HUMIDITY_SENSOR,
            default=d.get(CONF_OUTDOOR_HUMIDITY_SENSOR, ""),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor", device_class="humidity")
        ),
        vol.Optional(
            CONF_OUTDOOR_WIND_SENSOR,
            default=d.get(CONF_OUTDOOR_WIND_SENSOR, ""),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),
        vol.Optional(
            CONF_OUTDOOR_SOLAR_SENSOR,
            default=d.get(CONF_OUTDOOR_SOLAR_SENSOR, ""),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),
        vol.Optional(
            CONF_OUTDOOR_RAIN_SENSOR,
            default=d.get(CONF_OUTDOOR_RAIN_SENSOR, ""),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),

        # ── Temperaturen ──────────────────────────────────────────────
        vol.Required("comfort_temp", default=d.get("comfort_temp", 21.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=30, step=0.5,
                unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("night_temp", default=d.get("night_temp", 18.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=25, step=0.5,
                unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("away_temp", default=d.get("away_temp", 17.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=25, step=0.5,
                unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),

        # ── Zeitplan ──────────────────────────────────────────────────
        vol.Optional(CONF_SCHED_WD_MORNING, default=d.get(CONF_SCHED_WD_MORNING, "06:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WD_DAY, default=d.get(CONF_SCHED_WD_DAY, "09:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WD_DAY_TEMP, default=d.get(CONF_SCHED_WD_DAY_TEMP, 19.0)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=5, max=25, step=0.5,
                unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Optional(CONF_SCHED_WD_EVENING, default=d.get(CONF_SCHED_WD_EVENING, "17:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WD_NIGHT, default=d.get(CONF_SCHED_WD_NIGHT, "22:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WE_MORNING, default=d.get(CONF_SCHED_WE_MORNING, "08:00")):
            selector.TimeSelector(),
        vol.Optional(CONF_SCHED_WE_NIGHT, default=d.get(CONF_SCHED_WE_NIGHT, "23:00")):
            selector.TimeSelector(),

        # ── TRV-Verhalten ─────────────────────────────────────────────
        vol.Required("temp_tolerance", default=d.get("temp_tolerance", 0.5)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=0.1, max=2.0, step=0.1,
                unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("window_open_delay", default=d.get("window_open_delay", 5)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=0, max=60, step=1,
                unit_of_measurement="min", mode=selector.NumberSelectorMode.BOX,
            )),
        vol.Required("window_close_delay", default=d.get("window_close_delay", 2)):
            selector.NumberSelector(selector.NumberSelectorConfig(
                min=0, max=30, step=1,
                unit_of_measurement="min", mode=selector.NumberSelectorMode.BOX,
            )),

        # ── Lernalgorithmus ───────────────────────────────────────────
        vol.Required(
            CONF_LEARNING_ENABLED,
            default=d.get(CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED),
        ): selector.BooleanSelector(),
    })
