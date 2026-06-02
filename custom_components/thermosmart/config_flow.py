"""Config flow für ThermoSmart – Einrichtung und Zonenverwaltung."""
from __future__ import annotations

import re
import voluptuous as vol

from homeassistant import config_entries
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


def _slugify(text: str) -> str:
    """Zonenname → gültiger zone_id Schlüssel."""
    text = text.lower().strip()
    text = re.sub(r'[äöüß]', lambda m: {'ä':'ae','ö':'oe','ü':'ue','ß':'ss'}[m.group()], text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text or "zone"


class ThermoSmartConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Ersteinrichtung von ThermoSmart."""

    VERSION = 1

    async def async_step_user(self, user_input: dict | None = None):
        errors: dict[str, str] = {}

        if user_input is not None:
            weather_entity = user_input.get(CONF_WEATHER_ENTITY, DEFAULT_WEATHER_ENTITY)
            if self.hass.states.get(weather_entity) is None:
                errors[CONF_WEATHER_ENTITY] = "entity_not_found"
            else:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(title="ThermoSmart", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_WEATHER_ENTITY, default=DEFAULT_WEATHER_ENTITY):
                    selector.EntitySelector(
                        selector.EntitySelectorConfig(domain="weather")
                    ),
                vol.Optional(CONF_OUTDOOR_TEMP_SENSOR):
                    selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor", device_class="temperature"
                        )
                    ),
                vol.Required(CONF_LEARNING_ENABLED, default=DEFAULT_LEARNING_ENABLED):
                    selector.BooleanSelector(),
            }),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        return ThermoSmartOptionsFlow()


class ThermoSmartOptionsFlow(config_entries.OptionsFlow):
    """Optionen & Zonenverwaltung nach der Ersteinrichtung."""

    def __init__(self) -> None:
        self._zones: list[dict] = []
        self._editing_zone_id: str | None = None

    async def async_step_init(self, user_input: dict | None = None):
        """Hauptmenü: Einstellungen oder Zonen verwalten."""
        self._zones = list(self.config_entry.options.get("zones", []))
        menu = ["settings", "zone_add"]
        if self._zones:
            menu.append("zone_manage")
        return self.async_show_menu(step_id="init", menu_options=menu)

    # ── Globale Einstellungen ─────────────────────────────────────────────
    async def async_step_settings(self, user_input: dict | None = None):
        if user_input is not None:
            new_opts = dict(self.config_entry.options)
            new_opts.update(user_input)
            return self.async_create_entry(title="", data=new_opts)

        return self.async_show_form(
            step_id="settings",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_LEARNING_ENABLED,
                    default=self.config_entry.options.get(
                        CONF_LEARNING_ENABLED,
                        self.config_entry.data.get(CONF_LEARNING_ENABLED, True),
                    ),
                ): selector.BooleanSelector(),
            }),
        )

    # ── Zone hinzufügen ───────────────────────────────────────────────────
    async def async_step_zone_add(self, user_input: dict | None = None):
        errors: dict[str, str] = {}

        if user_input is not None:
            if not user_input.get("climate_entities"):
                errors["climate_entities"] = "required"
            else:
                zone_id = _slugify(user_input["name"])
                # Falls Zone mit gleichem ID schon existiert → umbenennen
                existing_ids = {z["zone_id"] for z in self._zones}
                base_id = zone_id
                counter = 2
                while zone_id in existing_ids:
                    zone_id = f"{base_id}_{counter}"
                    counter += 1

                new_zone = {
                    "zone_id": zone_id,
                    "name": user_input["name"],
                    "climate_entities": user_input.get("climate_entities", []),
                    "temp_sensors": user_input.get("temp_sensors", []),
                    "humidity_sensors": user_input.get("humidity_sensors", []),
                    "window_sensors": user_input.get("window_sensors", []),
                    "comfort_temp": float(user_input.get("comfort_temp", 21.0)),
                    "night_temp": float(user_input.get("night_temp", 18.0)),
                    "away_temp": float(user_input.get("away_temp", 17.0)),
                }
                self._zones.append(new_zone)
                new_opts = dict(self.config_entry.options)
                new_opts["zones"] = self._zones
                return self.async_create_entry(title="", data=new_opts)

        return self.async_show_form(
            step_id="zone_add",
            data_schema=_zone_schema(),
            errors=errors,
        )

    # ── Zone auswählen (bearbeiten / löschen) ────────────────────────────
    async def async_step_zone_manage(self, user_input: dict | None = None):
        if user_input is not None:
            selected = user_input.get("zone")
            if selected == "__done__":
                return self.async_create_entry(
                    title="", data=dict(self.config_entry.options)
                )
            self._editing_zone_id = selected
            return await self.async_step_zone_edit()

        options = [
            selector.SelectOptionDict(value=z["zone_id"], label=z["name"])
            for z in self._zones
        ]
        options.append(selector.SelectOptionDict(value="__done__", label="✓ Fertig"))

        return self.async_show_form(
            step_id="zone_manage",
            data_schema=vol.Schema({
                vol.Required("zone"): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=options)
                ),
            }),
        )

    # ── Zone bearbeiten ────────────────────────────────────────────────────
    async def async_step_zone_edit(self, user_input: dict | None = None):
        zone = next(
            (z for z in self._zones if z["zone_id"] == self._editing_zone_id), None
        )
        if zone is None:
            return await self.async_step_init()

        if user_input is not None:
            if user_input.get("delete_zone"):
                self._zones = [
                    z for z in self._zones if z["zone_id"] != self._editing_zone_id
                ]
            else:
                updated = {
                    "zone_id": self._editing_zone_id,
                    "name": user_input.get("name", zone["name"]),
                    "climate_entities": user_input.get("climate_entities", []),
                    "temp_sensors": user_input.get("temp_sensors", []),
                    "humidity_sensors": user_input.get("humidity_sensors", []),
                    "window_sensors": user_input.get("window_sensors", []),
                    "comfort_temp": float(user_input.get("comfort_temp", 21.0)),
                    "night_temp": float(user_input.get("night_temp", 18.0)),
                    "away_temp": float(user_input.get("away_temp", 17.0)),
                }
                self._zones = [
                    updated if z["zone_id"] == self._editing_zone_id else z
                    for z in self._zones
                ]
            new_opts = dict(self.config_entry.options)
            new_opts["zones"] = self._zones
            return self.async_create_entry(title="", data=new_opts)

        return self.async_show_form(
            step_id="zone_edit",
            data_schema=_zone_schema(zone),
            description_placeholders={"zone_name": zone["name"]},
        )


def _zone_schema(zone: dict | None = None) -> vol.Schema:
    """Formular-Schema für Zone hinzufügen / bearbeiten."""
    d = zone or {}
    return vol.Schema({
        vol.Required("name", default=d.get("name", "")):
            selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.TEXT)
            ),
        vol.Required("climate_entities", default=d.get("climate_entities", [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(domain="climate", multiple=True)
            ),
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
        vol.Optional("window_sensors", default=d.get("window_sensors", [])):
            selector.EntitySelector(
                selector.EntitySelectorConfig(domain="binary_sensor", multiple=True)
            ),
        vol.Required("comfort_temp", default=d.get("comfort_temp", 21.0)):
            selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=5, max=30, step=0.5,
                    unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX
                )
            ),
        vol.Required("night_temp", default=d.get("night_temp", 18.0)):
            selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=5, max=25, step=0.5,
                    unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX
                )
            ),
        vol.Required("away_temp", default=d.get("away_temp", 17.0)):
            selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=5, max=25, step=0.5,
                    unit_of_measurement="°C", mode=selector.NumberSelectorMode.BOX
                )
            ),
        vol.Optional("delete_zone", default=False):
            selector.BooleanSelector(),
    })
