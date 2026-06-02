"""Switch platform für ThermoSmart – Aktive Steuerung & Lernmodus."""
from __future__ import annotations
import logging

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
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
    async_add_entities([
        ThermoSmartActiveSwitch(coordinator, entry),
        ThermoSmartLearningSwitch(coordinator, entry),
    ])


def _device_info(entry: ConfigEntry) -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=f"ThermoSmart – {entry.data.get('name', 'Zone')}",
        manufacturer="ThermoSmart",
        model="AI Heating Controller",
        sw_version=VERSION,
        entry_type="service",  # type: ignore[arg-type]
    )


class ThermoSmartActiveSwitch(SwitchEntity, RestoreEntity):
    """Aktive Steuerung – Standard: AUS (Beobachtungsmodus).

    Solange AUS: ThermoSmart berechnet + lernt, schreibt aber KEIN Thermostat.
    Erst wenn AN: ThermoSmart übernimmt die Steuerung dieser Zone.
    """
    _attr_icon = "mdi:thermostat-auto"

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        zone_name = entry.data.get("name", "Zone")
        self._attr_unique_id = f"{entry.entry_id}_active_control"
        self._attr_name = f"ThermoSmart {zone_name} Aktive Steuerung"
        self._attr_device_info = _device_info(entry)
        self._is_on = False

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        self._is_on = last is not None and last.state == "on"
        self._coordinator.set_active_control(self._is_on)

    @property
    def is_on(self) -> bool:
        return self._is_on

    @property
    def extra_state_attributes(self) -> dict:
        return {
            "modus": "Aktive Steuerung" if self._is_on else "Beobachtungsmodus",
            "hinweis": (
                "Thermostat wird von ThermoSmart gesteuert"
                if self._is_on
                else "ThermoSmart beobachtet nur – deine Automationen laufen weiter"
            ),
        }

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self.async_write_ha_state()
        self._coordinator.set_active_control(True)

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        self._coordinator.set_active_control(False)


class ThermoSmartLearningSwitch(SwitchEntity, RestoreEntity):
    """Lernmodus – Standard: AN."""
    _attr_icon = "mdi:brain"

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        zone_name = entry.data.get("name", "Zone")
        self._attr_unique_id = f"{entry.entry_id}_learning"
        self._attr_name = f"ThermoSmart {zone_name} Lernmodus"
        self._attr_device_info = _device_info(entry)
        self._is_on = True

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None:
            self._is_on = last.state == "on"

    @property
    def is_on(self) -> bool:
        return self._is_on

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self.async_write_ha_state()
        self._get_engine().set_enabled(True)

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        self._get_engine().set_enabled(False)

    def _get_engine(self):
        return self.hass.data.get(DOMAIN, {}).get(
            self._entry.entry_id, {}
        ).get("learning_engine")
