"""Select platform für ThermoSmart."""
from __future__ import annotations
import logging

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
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
            model="Self-learning Heating Controller",
            sw_version=VERSION,
            entry_type="service",  # type: ignore[arg-type]
        )

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last and last.state in HEATING_MODES:
            # Nur zum Coordinator syncen wenn der noch im Default-Modus ist –
            # Globaler Urlaubsschalter (läuft vor select) hat sonst Vorrang.
            if self._coordinator._mode == HEATING_MODE_AUTO:
                self._coordinator.set_mode(last.state)

        # Als Coordinator-Listener registrieren → Entity aktualisiert sich
        # wenn das Climate-Entity oder ein anderer Pfad den Modus ändert.
        self._coordinator.async_add_listener(self._handle_coordinator_update)

    async def async_will_remove_from_hass(self) -> None:
        self._coordinator.async_remove_listener(self._handle_coordinator_update)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Coordinator hat Daten aktualisiert → State neu schreiben."""
        self.async_write_ha_state()

    @property
    def current_option(self) -> str:
        """Liest immer direkt vom Coordinator – bleibt in sync mit Climate-Entity."""
        return self._coordinator._mode

    async def async_select_option(self, option: str) -> None:
        if option not in HEATING_MODES:
            return
        self._coordinator.set_mode(option)
        self.async_write_ha_state()
        await self._coordinator.async_request_refresh()
