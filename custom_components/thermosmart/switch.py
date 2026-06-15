"""Switch platform für ThermoSmart."""
from __future__ import annotations
import logging

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN, VERSION, DOMAIN_GLOBAL_VACATION, DOMAIN_GLOBAL_DEBUG
from .coordinator import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    cfg = {**entry.data, **entry.options}

    # System-Entry: globale Schalter
    # (Sommer-Override: select.ThermoSmartGlobalSummerSelect)
    if cfg.get("entry_type") == "system":
        async_add_entities([
            ThermoSmartGlobalVacationSwitch(hass, entry),
            ThermoSmartGlobalDebugSwitch(hass, entry),
        ])
        return

    # Zone-Entry: zone-spezifische Schalter
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    entities = [
        ThermoSmartActiveSwitch(coordinator, entry),
        ThermoSmartLearningSwitch(coordinator, entry),
    ]

    # Backward-Compat: globaler Urlaubsschalter an die erste Zone anhängen wenn kein System-Entry
    has_system = any(
        e.data.get("entry_type") == "system"
        for e in hass.config_entries.async_entries(DOMAIN)
    )
    if not has_system and "global_switches_created" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["global_switches_created"] = True
        entities.append(ThermoSmartGlobalVacationSwitch(hass, entry))

    async_add_entities(entities)


def _device_info(entry: ConfigEntry) -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=f"ThermoSmart – {entry.data.get('name', 'Zone')}",
        manufacturer="ThermoSmart",
        model="Self-learning Heating Controller",
        sw_version=VERSION,
        entry_type="service",  # type: ignore[arg-type]
    )


class ThermoSmartActiveSwitch(SwitchEntity, RestoreEntity):
    """Active control switch – default OFF (observation mode)."""
    _attr_has_entity_name = True
    _attr_translation_key = "active_control"
    _attr_icon = "mdi:thermostat-auto"

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_active_control"
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
            "mode": "active_control" if self._is_on else "observation",
        }

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self.async_write_ha_state()
        self._coordinator.set_active_control(True)
        await self._coordinator.async_request_refresh()

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        self._coordinator.set_active_control(False)
        await self._coordinator.async_request_refresh()


class ThermoSmartLearningSwitch(SwitchEntity, RestoreEntity):
    """Learning mode switch – default ON."""
    _attr_has_entity_name = True
    _attr_translation_key = "learning"
    _attr_icon = "mdi:brain"

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_learning"
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
        await self._coordinator.async_request_refresh()

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self.async_write_ha_state()
        engine = self._get_engine()
        if engine:
            engine.set_zone_enabled(self._entry.entry_id, False)
        await self._coordinator.async_request_refresh()

    def _get_engine(self):
        return self.hass.data.get(DOMAIN, {}).get("learning_engine")


# ── Globale Schalter (domain-weit, steuern alle Zonen) ───────────────────────

def _global_device_info() -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, "global")},
        name="ThermoSmart System",
        manufacturer="ThermoSmart",
        model="Global Control",
        sw_version=VERSION,
    )

def _all_coordinators(hass: HomeAssistant):
    skip = {"learning_engine", "global_switches_created"}
    for key, data in hass.data.get(DOMAIN, {}).items():
        if key not in skip and isinstance(data, dict) and data.get("type") != "system":
            if coord := data.get("coordinator"):
                yield coord


class ThermoSmartGlobalVacationSwitch(SwitchEntity, RestoreEntity):
    """Vacation mode – applies to all ThermoSmart zones simultaneously."""
    _attr_has_entity_name = True
    _attr_translation_key = "vacation_mode"
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
        # Store for coordinators that load after this entity (race condition fix)
        self._hass.data[DOMAIN]["global_vacation_override"] = self._is_on
        if self._is_on:
            for coord in _all_coordinators(self._hass):
                coord.set_vacation_override(True)
                await coord.async_request_refresh()

    @property
    def is_on(self) -> bool:
        return self._is_on

    @property
    def extra_state_attributes(self) -> dict:
        zones = sum(1 for _ in _all_coordinators(self._hass))
        return {"active_zones": zones}

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self._hass.data[DOMAIN]["global_vacation_override"] = True
        self.async_write_ha_state()
        for coord in _all_coordinators(self._hass):
            coord.set_vacation_override(True)
            await coord.async_request_refresh()

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self._hass.data[DOMAIN]["global_vacation_override"] = False
        self.async_write_ha_state()
        for coord in _all_coordinators(self._hass):
            coord.set_vacation_override(False)
            await coord.async_request_refresh()


_THERMOSMART_LOGGER_NAME = "custom_components.thermosmart"


class ThermoSmartGlobalDebugSwitch(SwitchEntity, RestoreEntity):
    """Global debug-log switch – sets the ThermoSmart logger to DEBUG at runtime."""
    _attr_has_entity_name = True
    _attr_translation_key = "debug_log"
    _attr_icon = "mdi:bug-check"
    _attr_unique_id = DOMAIN_GLOBAL_DEBUG

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self._hass = hass
        self._attr_device_info = _global_device_info()
        self._is_on = False
        self._prev_level: int = logging.NOTSET

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None and last.state == "on":
            self._is_on = True
            self._set_debug(True)

    @property
    def is_on(self) -> bool:
        return self._is_on

    async def async_turn_on(self, **kwargs) -> None:
        self._is_on = True
        self._set_debug(True)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs) -> None:
        self._is_on = False
        self._set_debug(False)
        self.async_write_ha_state()

    def _set_debug(self, enable: bool) -> None:
        ts_logger = logging.getLogger(_THERMOSMART_LOGGER_NAME)
        if enable:
            self._prev_level = ts_logger.level
            ts_logger.setLevel(logging.DEBUG)
            _LOGGER.info("ThermoSmart System: Debug logging enabled")
        else:
            ts_logger.setLevel(self._prev_level)
            _LOGGER.info("ThermoSmart System: Debug logging disabled")
