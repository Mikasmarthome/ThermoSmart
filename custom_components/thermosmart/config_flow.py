"""Config flow for ThermoSmart integration."""
from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_WEATHER_ENTITY,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_LEARNING_ENABLED,
    DEFAULT_WEATHER_ENTITY,
    DEFAULT_LEARNING_ENABLED,
)


class ThermoSmartConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for ThermoSmart."""

    VERSION = 1

    async def async_step_user(self, user_input: dict | None = None):
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate that the weather entity exists
            weather_entity = user_input.get(CONF_WEATHER_ENTITY, DEFAULT_WEATHER_ENTITY)
            state = self.hass.states.get(weather_entity)
            if state is None:
                errors[CONF_WEATHER_ENTITY] = "entity_not_found"
            else:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="ThermoSmart",
                    data=user_input,
                )

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_WEATHER_ENTITY,
                    default=DEFAULT_WEATHER_ENTITY,
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="weather")
                ),
                vol.Optional(
                    CONF_OUTDOOR_TEMP_SENSOR,
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
                ),
                vol.Required(
                    CONF_LEARNING_ENABLED,
                    default=DEFAULT_LEARNING_ENABLED,
                ): selector.BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        return ThermoSmartOptionsFlow()


class ThermoSmartOptionsFlow(config_entries.OptionsFlow):
    """Handle ThermoSmart options."""

    async def async_step_init(self, user_input: dict | None = None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_LEARNING_ENABLED,
                    default=self.config_entry.options.get(
                        CONF_LEARNING_ENABLED,
                        self.config_entry.data.get(CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED),
                    ),
                ): selector.BooleanSelector(),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
