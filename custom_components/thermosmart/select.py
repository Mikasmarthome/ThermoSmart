"""Select platform für ThermoSmart."""
from __future__ import annotations
import logging

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN, VERSION,
    HEATING_MODE_AUTO, HEATING_MODE_COMFORT, HEATING_MODE_NIGHT,
    HEATING_MODE_AWAY, HEATING_MODE_VACATION, HEATING_MODE_ECO,
)
from . import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)

MODE_LABELS = {
    HEATING_MODE_AUTO:     "Auto",
    HEATING_MODE_COMFORT:  "Komfort",
    HEATING_MODE_ECO:      "Eco",
    HEATING_MODE_NIGHT:    "Nacht",
    HEATING_MODE_AWAY:     "Abwesend",
    HEATING_MODE_VACATION: "Urlaub",
}
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([ThermoSmartModeSelect(coordinator, entry)])


class ThermoSmartModeSelect(SelectEntity, RestoreEntity):
    _attr_has_entity_name = True
    _attr_icon = "mdi:home-thermometer-outline"

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_mode"
        self._attr_name = "Heizmodus"
        self._attr_options = list(MODE_LABELS.values())
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"ThermoSmart – {entry.data.get('name', 'Zone')}",
            manufacturer="ThermoSmart",
            model="AI Heating Controller",
            sw_version=VERSION,
            entry_type="service",  # type: ignore[arg-type]
        )
        self._mode = HEATING_MODE_AUTO

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last and last.state in self._attr_options:
            self._mode = LABEL_TO_MODE.get(last.state, HEATING_MODE_AUTO)

    @property
    def current_option(self) -> str:
        return MODE_LABELS[self._mode]

    async def async_select_option(self, option: str) -> None:
        self._mode = LABEL_TO_MODE.get(option, HEATING_MODE_AUTO)
        self.async_write_ha_state()
        self._coordinator.set_mode(self._mode)
        await self._coordinator.async_request_refresh()
