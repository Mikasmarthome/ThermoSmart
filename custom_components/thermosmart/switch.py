"""Switch platform für ThermoSmart.

Schalter:
  1. ThermoSmart Aktive Steuerung  (global, Standard: AUS)
     → Solange AUS: ThermoSmart berechnet alles, schreibt aber KEIN
       Thermostat an. Deine bestehenden Automationen laufen weiter.
     → Erst wenn AN: ThermoSmart übernimmt die Steuerung.

  2. Lernmodus pro Zone  (Standard: AN)
     → AUS = Lernalgorithmus sammelt keine neuen Daten für diese Zone.
"""
from __future__ import annotations

import logging

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN, VERSION, ZONES

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Switch-Entities einrichten."""
    entities: list[SwitchEntity] = []

    # Globaler Master-Schalter
    entities.append(ThermoSmartMasterSwitch(entry))

    # Lernschalter pro Zone
    for zone_id, zone_cfg in ZONES.items():
        entities.append(ThermoSmartLearningSwitch(entry, zone_id, zone_cfg))

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


class ThermoSmartMasterSwitch(SwitchEntity, RestoreEntity):
    """Globaler Master-Schalter: steuert ob ThermoSmart aktiv die Thermostate schreibt.

    Standard: AUS (Beobachtungsmodus).
    Erst wenn AN: ThermoSmart übernimmt die Heizungssteuerung.
    """

    _attr_icon = "mdi:thermostat-auto"
    _attr_has_entity_name = False

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_active_control"
        self._attr_name = "ThermoSmart Aktive Steuerung"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="ThermoSmart",
            manufacturer="ThermoSmart",
            model="AI Heating Controller",
            sw_version=VERSION,
            entry_type="service",  # type: ignore[arg-type]
        )
        self._is_on = False  # ← Standard: AUS (Beobachtungsmodus)

    async def async_added_to_hass(self) -> None:
        """Zustand nach Neustart wiederherstellen."""
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None and last.state == "on":
            self._is_on = True
            _LOGGER.info("ThermoSmart: Aktive Steuerung wiederhergestellt → AN")
        else:
            self._is_on = False
            _LOGGER.info(
                "ThermoSmart: Aktive Steuerung ist AUS – Beobachtungsmodus aktiv"
            )
        # Coordinator informieren
        coordinator = (
            self.hass.data.get(DOMAIN, {})
            .get(self._entry.entry_id, {})
            .get("coordinator")
        )
        if coordinator:
            coordinator.set_active_control(self._is_on)

    @property
    def is_on(self) -> bool:
        return self._is_on

    @property
    def extra_state_attributes(self) -> dict:
        return {
            "mode": "Aktive Steuerung" if self._is_on else "Beobachtungsmodus",
            "hint": (
                "ThermoSmart steuert die Thermostate"
                if self._is_on
                else "ThermoSmart berechnet nur Empfehlungen – deine Automationen laufen weiter"
            ),
        }

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self.async_write_ha_state()
        self._notify_coordinator(True)
        _LOGGER.warning(
            "ThermoSmart: Aktive Steuerung EINGESCHALTET – "
            "ThermoSmart übernimmt jetzt die Heizungssteuerung!"
        )

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        self._notify_coordinator(False)
        _LOGGER.info("ThermoSmart: Aktive Steuerung AUS – Beobachtungsmodus")

    def _notify_coordinator(self, active: bool) -> None:
        coordinator = (
            self.hass.data.get(DOMAIN, {})
            .get(self._entry.entry_id, {})
            .get("coordinator")
        )
        if coordinator:
            coordinator.set_active_control(active)


class ThermoSmartLearningSwitch(SwitchEntity, RestoreEntity):
    """Lernmodus-Schalter pro Zone.

    AUS → Lernalgorithmus sammelt keine neuen Daten.
    AN  → Jede Messung verbessert die Vorheizzeit-Schätzung.
    """

    _attr_icon = "mdi:brain"

    def __init__(
        self, entry: ConfigEntry, zone_id: str, zone_cfg: dict
    ) -> None:
        self._entry = entry
        self._zone_id = zone_id
        self._attr_unique_id = f"{DOMAIN}_{zone_id}_learning"
        self._attr_name = f"ThermoSmart {zone_cfg['name']} Lernmodus"
        self._attr_device_info = _device_info(entry, zone_id, zone_cfg["name"])
        self._is_on = True  # Standard: AN

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
        self._update_engine(True)

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        self._update_engine(False)

    def _update_engine(self, enabled: bool) -> None:
        learning_engine = (
            self.hass.data.get(DOMAIN, {})
            .get(self._entry.entry_id, {})
            .get("learning_engine")
        )
        if learning_engine:
            learning_engine.set_enabled(enabled)
