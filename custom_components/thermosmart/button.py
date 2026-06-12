"""Button platform for ThermoSmart — system-level one-shot actions."""
from __future__ import annotations

import logging
import os

from homeassistant.components.button import ButtonEntity
from homeassistant.components.persistent_notification import async_create as _pn_create
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, VERSION
from .export import async_export_learning_data, build_export_notification_message

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    cfg = {**entry.data, **entry.options}
    if cfg.get("entry_type") != "system":
        return
    async_add_entities([ThermoSmartExportButton(hass, entry)])


class ThermoSmartExportButton(ButtonEntity):
    """Button that triggers a voluntary anonymized learning data export."""

    _attr_has_entity_name = True
    _attr_translation_key = "export_learning_data"
    _attr_icon = "mdi:database-export-outline"
    _attr_unique_id = f"{DOMAIN}_export_learning_data"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, "global")},
            name="ThermoSmart System",
            manufacturer="ThermoSmart",
            model="Global Control",
            sw_version=VERSION,
        )

    async def async_press(self) -> None:
        """Run the export and show a persistent notification with the file path."""
        filepath = await async_export_learning_data(self._hass)
        filename = os.path.basename(filepath)
        _pn_create(
            self._hass,
            message=build_export_notification_message(filename),
            title="ThermoSmart — Learning Data Export",
            notification_id="thermosmart_export",
        )
        _LOGGER.info("ThermoSmart: export button pressed → %s", filename)
