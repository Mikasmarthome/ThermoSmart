"""Number platform für ThermoSmart – manueller Temperatur-Override."""
from __future__ import annotations
import logging

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN, VERSION
from . import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([ThermoSmartTemperatureOverride(coordinator, entry)])


class ThermoSmartTemperatureOverride(NumberEntity, RestoreEntity):
    _attr_has_entity_name = True
    _attr_mode = NumberMode.BOX
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_native_min_value = 5.0
    _attr_native_max_value = 30.0
    _attr_native_step = 0.5
    _attr_icon = "mdi:thermometer-lines"

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._attr_unique_id = f"{entry.entry_id}_temp_override"
        self._attr_name = "Temperatur Override"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"ThermoSmart – {entry.data.get('name', 'Zone')}",
            manufacturer="ThermoSmart",
            model="AI Heating Controller",
            sw_version=VERSION,
            entry_type="service",  # type: ignore[arg-type]
        )
        self._value: float | None = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last and last.state not in ("unknown", "unavailable", "None", "0.0", "0"):
            try:
                v = float(last.state)
                if 5.0 <= v <= 30.0:
                    self._value = v
                    self._coordinator.set_override(v)
            except ValueError:
                pass

    @property
    def native_value(self) -> float | None:
        return self._value

    async def async_set_native_value(self, value: float) -> None:
        self._value = value
        self.async_write_ha_state()
        self._coordinator.set_override(value)
        await self._coordinator.async_request_refresh()
