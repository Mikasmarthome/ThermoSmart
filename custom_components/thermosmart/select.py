"""Select platform für ThermoSmart – Heizmodus pro Zone."""
from __future__ import annotations

import logging

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN,
    VERSION,
    ZONES,
    HEATING_MODES,
    HEATING_MODE_AUTO,
    HEATING_MODE_COMFORT,
    HEATING_MODE_NIGHT,
    HEATING_MODE_AWAY,
    HEATING_MODE_VACATION,
)

_LOGGER = logging.getLogger(__name__)

# Deutsche Bezeichnungen für die UI
MODE_LABELS: dict[str, str] = {
    HEATING_MODE_AUTO:     "Auto",
    HEATING_MODE_COMFORT:  "Komfort",
    HEATING_MODE_NIGHT:    "Nacht",
    HEATING_MODE_AWAY:     "Abwesend",
    HEATING_MODE_VACATION: "Urlaub",
}
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    entities: list[SelectEntity] = []
    for zone_id, zone_cfg in ZONES.items():
        entities.append(ThermoSmartModeSelect(entry, zone_id, zone_cfg))
    async_add_entities(entities, restore=True)


def _device_info(entry: ConfigEntry, zone_id: str, zone_name: str) -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, f"{entry.entry_id}_{zone_id}")},
        name=f"ThermoSmart – {zone_name}",
        manufacturer="ThermoSmart",
        model="AI Heating Controller",
        sw_version=VERSION,
        entry_type="service",  # type: ignore[arg-type]
    )


class ThermoSmartModeSelect(SelectEntity, RestoreEntity):
    """Heizmodus-Auswahl pro Zone.

    Auto     → ThermoSmart entscheidet (Zeitplan + Lernen + Präsenz)
    Komfort  → Komforttemperatur dauerhaft
    Nacht    → Nachttemperatur dauerhaft
    Abwesend → Abwesenheitstemperatur dauerhaft
    Urlaub   → Frostschutztemperatur (12°C)
    """

    _attr_icon = "mdi:home-thermometer-outline"

    def __init__(
        self, entry: ConfigEntry, zone_id: str, zone_cfg: dict
    ) -> None:
        self._entry = entry
        self._zone_id = zone_id
        self._attr_unique_id = f"{DOMAIN}_{zone_id}_mode"
        self._attr_name = f"ThermoSmart {zone_cfg['name']} Modus"
        self._attr_options = list(MODE_LABELS.values())
        self._attr_device_info = _device_info(entry, zone_id, zone_cfg["name"])
        self._current_mode: str = HEATING_MODE_AUTO

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last and last.state in self._attr_options:
            self._current_mode = LABEL_TO_MODE.get(last.state, HEATING_MODE_AUTO)

    @property
    def current_option(self) -> str:
        return MODE_LABELS[self._current_mode]

    @property
    def extra_state_attributes(self) -> dict:
        from .const import ZONE_TEMPS
        temps = ZONE_TEMPS.get(self._zone_id, {})
        return {
            "mode_key": self._current_mode,
            "comfort_temp": temps.get("comfort"),
            "night_temp": temps.get("night"),
            "away_temp": temps.get("away"),
            "vacation_temp": temps.get("vacation"),
        }

    async def async_select_option(self, option: str) -> None:
        """Modus wechseln und Coordinator informieren."""
        new_mode = LABEL_TO_MODE.get(option, HEATING_MODE_AUTO)
        self._current_mode = new_mode
        self.async_write_ha_state()

        coordinator = (
            self.hass.data.get(DOMAIN, {})
            .get(self._entry.entry_id, {})
            .get("coordinator")
        )
        if coordinator:
            coordinator.set_zone_mode(self._zone_id, new_mode)
            await coordinator.async_request_refresh()

        _LOGGER.info(
            "ThermoSmart %s: Modus → %s (%s)", self._zone_id, option, new_mode
        )

    def get_mode(self) -> str:
        """Aktuellen Modus-Schlüssel zurückgeben (für Coordinator)."""
        return self._current_mode
