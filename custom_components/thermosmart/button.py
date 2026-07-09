"""Button platform for ThermoSmart — system-level one-shot actions."""
from __future__ import annotations

import logging
import os

from homeassistant.components.button import ButtonEntity
from homeassistant.components.persistent_notification import async_create as _pn_create
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, VERSION
from .export import (
    async_export_learning_data,
    async_export_support_data,
    build_export_notification_message,
    build_support_notification_message,
)

_LOGGER = logging.getLogger(__name__)

_DEVICE_INFO = DeviceInfo(
    identifiers={(DOMAIN, "global")},
    name="ThermoSmart System",
    manufacturer="ThermoSmart",
    model="Global Control",
    sw_version=VERSION,
)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    cfg = {**entry.data, **entry.options}
    if cfg.get("entry_type") != "system":
        return
    async_add_entities([
        ThermoSmartResearchExportButton(hass, entry),
        ThermoSmartSupportExportButton(hass, entry),
    ])


class ThermoSmartResearchExportButton(ButtonEntity):
    """Button that triggers an anonymized learning-data research export."""

    _attr_has_entity_name = True
    _attr_translation_key = "create_research_export"
    _attr_icon = "mdi:database-export"
    _attr_entity_category = EntityCategory.CONFIG
    # Stable unique_id — unchanged from the original export button so existing
    # HA entity registry entries are updated in-place (no orphan entity).
    _attr_unique_id = f"{DOMAIN}_export_learning_data"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._attr_device_info = _DEVICE_INFO

    async def async_press(self) -> None:
        filepath = await async_export_learning_data(self._hass)
        filename = os.path.basename(filepath)
        _pn_create(
            self._hass,
            message=build_export_notification_message(filename),
            title="ThermoSmart – Research-Export",
            notification_id="thermosmart_research_export",
        )
        _LOGGER.info("ThermoSmart: research export button pressed → %s", filename)


class ThermoSmartSupportExportButton(ButtonEntity):
    """Button that triggers a support-oriented diagnostics export."""

    _attr_has_entity_name = True
    _attr_translation_key = "create_support_export"
    _attr_icon = "mdi:lifebuoy"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_unique_id = f"{DOMAIN}_create_support_export"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._attr_device_info = _DEVICE_INFO

    async def async_press(self) -> None:
        filepath = await async_export_support_data(self._hass)
        filename = os.path.basename(filepath)
        _pn_create(
            self._hass,
            message=build_support_notification_message(filename),
            title="ThermoSmart – Support-Export",
            notification_id="thermosmart_support_export",
        )
        _LOGGER.info("ThermoSmart: support export button pressed → %s", filename)
