"""Switch platform für ThermoSmart."""
from __future__ import annotations
import logging

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN, VERSION, DOMAIN_GLOBAL_SUMMER, DOMAIN_GLOBAL_VACATION
from . import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    entities = [
        ThermoSmartActiveSwitch(coordinator, entry),
        ThermoSmartLearningSwitch(coordinator, entry),
    ]
    # Globale Schalter nur einmal anlegen (beim ersten Zone-Setup)
    if "global_switches_created" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["global_switches_created"] = True
        entities.extend([
            ThermoSmartGlobalSummerSwitch(hass, entry),
            ThermoSmartGlobalVacationSwitch(hass, entry),
        ])
    async_add_entities(entities)


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
    """Aktive Steuerung – Standard: AUS (Beobachtungsmodus)."""
    _attr_has_entity_name = True
    _attr_icon = "mdi:thermostat-auto"

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_active_control"
        self._attr_name = "Aktive Steuerung"
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
    _attr_has_entity_name = True
    _attr_icon = "mdi:brain"

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_learning"
        self._attr_name = "Lernmodus"
        self._attr_device_info = _device_info(entry)
        self._is_on = True

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None:
            self._is_on = last.state == "on"
        engine = self._get_engine()
        if engine is not None:
            engine.set_zone_enabled(self._entry.entry_id, self._is_on)

    @property
    def is_on(self) -> bool:
        return self._is_on

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self.async_write_ha_state()
        engine = self._get_engine()
        if engine:
            engine.set_zone_enabled(self._entry.entry_id, True)

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        engine = self._get_engine()
        if engine:
            engine.set_zone_enabled(self._entry.entry_id, False)

    def _get_engine(self):
        return self.hass.data.get(DOMAIN, {}).get("learning_engine")


# ── Globale Schalter (domain-weit, steuern alle Zonen) ───────────────────────

def _global_device_info() -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, "global")},
        name="ThermoSmart System",
        manufacturer="ThermoSmart",
        model="Global Control",
    )

def _all_coordinators(hass: HomeAssistant):
    skip = {"learning_engine", "global_switches_created"}
    for key, data in hass.data.get(DOMAIN, {}).items():
        if key not in skip and isinstance(data, dict):
            if coord := data.get("coordinator"):
                yield coord


class ThermoSmartGlobalSummerSwitch(SwitchEntity, RestoreEntity):
    """Sommer-Modus für alle ThermoSmart-Zonen gleichzeitig."""
    _attr_has_entity_name = False
    _attr_name = "ThermoSmart – Sommer-Modus"
    _attr_icon = "mdi:weather-sunny"
    _attr_unique_id = DOMAIN_GLOBAL_SUMMER

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self._hass = hass
        self._attr_device_info = _global_device_info()
        self._is_on = False

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        self._is_on = last is not None and last.state == "on"
        if self._is_on:
            for coord in _all_coordinators(self._hass):
                coord.set_summer_override(True)

    @property
    def is_on(self) -> bool:
        return self._is_on

    @property
    def extra_state_attributes(self) -> dict:
        zones = sum(1 for _ in _all_coordinators(self._hass))
        return {"aktive_zonen": zones, "modus": "Sommer erzwungen" if self._is_on else "Automatisch"}

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self.async_write_ha_state()
        for coord in _all_coordinators(self._hass):
            coord.set_summer_override(True)
            await coord.async_request_refresh()

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        for coord in _all_coordinators(self._hass):
            coord.set_summer_override(None)   # Zurück zur automatischen Erkennung
            await coord.async_request_refresh()


class ThermoSmartGlobalVacationSwitch(SwitchEntity, RestoreEntity):
    """Urlaubs-Modus für alle ThermoSmart-Zonen gleichzeitig."""
    _attr_has_entity_name = False
    _attr_name = "ThermoSmart – Urlaubsmodus"
    _attr_icon = "mdi:airplane"
    _attr_unique_id = DOMAIN_GLOBAL_VACATION

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self._hass = hass
        self._attr_device_info = _global_device_info()
        self._is_on = False

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        self._is_on = last is not None and last.state == "on"
        if self._is_on:
            for coord in _all_coordinators(self._hass):
                coord.set_vacation_override(True)

    @property
    def is_on(self) -> bool:
        return self._is_on

    @property
    def extra_state_attributes(self) -> dict:
        zones = sum(1 for _ in _all_coordinators(self._hass))
        return {"aktive_zonen": zones}

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self.async_write_ha_state()
        for coord in _all_coordinators(self._hass):
            coord.set_vacation_override(True)
            await coord.async_request_refresh()

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        for coord in _all_coordinators(self._hass):
            coord.set_vacation_override(False)
            await coord.async_request_refresh()
