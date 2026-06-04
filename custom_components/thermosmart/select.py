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
    HEATING_MODES,
)
from .coordinator import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([ThermoSmartModeSelect(coordinator, entry)])


class ThermoSmartModeSelect(SelectEntity, RestoreEntity):
    _attr_has_entity_name = True
    _attr_translation_key = "mode"
    _attr_icon = "mdi:home-thermometer-outline"
    _attr_options = HEATING_MODES  # ["auto", "comfort", "eco", "night", "away", "vacation"]

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_mode"
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
        if last and last.state in HEATING_MODES:
            self._mode = last.state
            # Nur zum Coordinator syncen wenn der noch im Default-Modus ist –
            # Globaler Urlaubsschalter (läuft vor select) hat sonst Vorrang.
            if self._coordinator._mode == HEATING_MODE_AUTO:
                self._coordinator.set_mode(self._mode)

    @property
    def current_option(self) -> str:
        return self._mode

    async def async_select_option(self, option: str) -> None:
        if option not in HEATING_MODES:
            return
        self._mode = option
        self.async_write_ha_state()
        self._coordinator.set_mode(self._mode)
        await self._coordinator.async_request_refresh()
