"""Number platform for ThermoSmart – manual temperature overrides per zone."""
from __future__ import annotations

import logging

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN,
    VERSION,
)
from . import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)

# Kein Override aktiv
OVERRIDE_NONE = 0.0

# Temperaturbereich für Overrides
OVERRIDE_MIN = 5.0
OVERRIDE_MAX = 30.0
OVERRIDE_STEP = 0.5


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ThermoSmart number entities."""
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    entities: list[NumberEntity] = []
    for zone_id, zone_cfg in coordinator.zones.items():
        entities.append(
            ThermoSmartTemperatureOverride(entry, zone_id, zone_cfg)
        )
    async_add_entities(entities)


def _device_info(entry: ConfigEntry, zone_id: str, zone_name: str) -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, f"{entry.entry_id}_{zone_id}")},
        name=f"ThermoSmart – {zone_name}",
        manufacturer="ThermoSmart",
        model="AI Heating Controller",
        sw_version=VERSION,
        entry_type="service",  # type: ignore[arg-type]
    )


class ThermoSmartTemperatureOverride(NumberEntity, RestoreEntity):
    """
    Manueller Temperatur-Override für eine Zone.

    Wert = 0  → kein Override, ThermoSmart steuert automatisch
    Wert > 0  → dieser Wert wird als Zieltemperatur verwendet,
                Wetterkorrektur und Lernalgorithmus werden überbrückt
    """

    _attr_mode = NumberMode.BOX
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_native_min_value = OVERRIDE_MIN
    _attr_native_max_value = OVERRIDE_MAX
    _attr_native_step = OVERRIDE_STEP
    _attr_icon = "mdi:thermometer-lines"

    def __init__(
        self,
        entry: ConfigEntry,
        zone_id: str,
        zone_cfg: dict,
    ) -> None:
        self._zone_id = zone_id
        self._zone_cfg = zone_cfg
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_{zone_id}_temp_override"
        self._attr_name = f"ThermoSmart {zone_cfg['name']} Temperatur Override"
        self._attr_device_info = _device_info(entry, zone_id, zone_cfg["name"])
        self._override_value: float = OVERRIDE_NONE

    async def async_added_to_hass(self) -> None:
        """Gespeicherten Wert nach HA-Neustart wiederherstellen."""
        await super().async_added_to_hass()
        last_state = await self.async_get_last_state()
        if last_state and last_state.state not in ("unknown", "unavailable", "None"):
            try:
                restored = float(last_state.state)
                if OVERRIDE_MIN <= restored <= OVERRIDE_MAX:
                    self._override_value = restored
                    _LOGGER.debug(
                        "ThermoSmart %s: Override wiederhergestellt auf %.1f°C",
                        self._zone_id,
                        restored,
                    )
            except ValueError:
                pass

    @property
    def native_value(self) -> float:
        return self._override_value

    @property
    def extra_state_attributes(self) -> dict:
        return {
            "zone": self._zone_id,
            "override_active": self._override_value >= OVERRIDE_MIN,
            "hint": (
                "Override aktiv – automatische Steuerung pausiert"
                if self._override_value >= OVERRIDE_MIN
                else "Kein Override – ThermoSmart steuert automatisch"
            ),
        }

    async def async_set_native_value(self, value: float) -> None:
        """Override setzen und Coordinator informieren."""
        self._override_value = value
        self.async_write_ha_state()

        # Coordinator über den neuen Override informieren
        coordinator = (
            self.hass.data.get(DOMAIN, {})
            .get(self._entry.entry_id, {})
            .get("coordinator")
        )
        if coordinator:
            coordinator.set_override(self._zone_id, value)
            await coordinator.async_request_refresh()

        _LOGGER.info(
            "ThermoSmart %s: Override gesetzt auf %.1f°C",
            self._zone_id,
            value,
        )

    def clear_override(self) -> None:
        """Override zurücksetzen (wird von Services aufgerufen)."""
        self._override_value = OVERRIDE_NONE
        self.async_write_ha_state()
