"""Sensor platform for ThermoSmart – exposes zone diagnostics as HA sensors."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    ATTR_PREHEAT_MINUTES,
    ATTR_LEARNING_CONFIDENCE,
    ATTR_WEATHER_OFFSET,
    ICON_THERMOSMART,
    ICON_ZONE,
    ICON_LEARNING,
    ICON_PREHEAT,
    ICON_WEATHER_ADJUST,
    VERSION,
    ZONES,
)
from . import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ThermoSmart sensors."""
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]

    entities: list[SensorEntity] = []
    for zone_id, zone_cfg in ZONES.items():
        entities.extend([
            ThermoSmartTargetTempSensor(coordinator, entry, zone_id, zone_cfg),
            ThermoSmartPreheatSensor(coordinator, entry, zone_id, zone_cfg),
            ThermoSmartConfidenceSensor(coordinator, entry, zone_id, zone_cfg),
            ThermoSmartWeatherOffsetSensor(coordinator, entry, zone_id, zone_cfg),
        ])

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


class _ThermoSmartBaseSensor(CoordinatorEntity, SensorEntity):
    """Base class for all ThermoSmart sensors."""

    def __init__(
        self,
        coordinator: ThermoSmartCoordinator,
        entry: ConfigEntry,
        zone_id: str,
        zone_cfg: dict,
        sensor_key: str,
    ) -> None:
        super().__init__(coordinator)
        self._zone_id = zone_id
        self._zone_cfg = zone_cfg
        self._sensor_key = sensor_key
        self._entry = entry
        self._attr_device_info = _device_info(entry, zone_id, zone_cfg["name"])

    @property
    def _zone_data(self) -> dict:
        if self.coordinator.data is None:
            return {}
        return self.coordinator.data.get("zones", {}).get(self._zone_id, {})

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success


class ThermoSmartTargetTempSensor(_ThermoSmartBaseSensor):
    """Adjusted target temperature after weather offset."""

    def __init__(self, coordinator, entry, zone_id, zone_cfg):
        super().__init__(coordinator, entry, zone_id, zone_cfg, "adjusted_target")
        self._attr_unique_id = f"{DOMAIN}_{zone_id}_adjusted_target"
        self._attr_name = f"ThermoSmart {zone_cfg['name']} Zieltemperatur"
        self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
        self._attr_device_class = SensorDeviceClass.TEMPERATURE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_ZONE

    @property
    def native_value(self):
        return self._zone_data.get("adjusted_target")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        d = self._zone_data
        return {
            "base_target": d.get("base_target"),
            "weather_offset": d.get("weather_offset"),
            "window_open": d.get("window_open"),
            "current_temp": d.get("current_temp"),
            "outdoor_temp": d.get("outdoor_temp"),
            "weather_condition": d.get("weather_condition"),
        }


class ThermoSmartPreheatSensor(_ThermoSmartBaseSensor):
    """Recommended preheat lead time in minutes."""

    def __init__(self, coordinator, entry, zone_id, zone_cfg):
        super().__init__(coordinator, entry, zone_id, zone_cfg, "preheat_minutes")
        self._attr_unique_id = f"{DOMAIN}_{zone_id}_preheat_minutes"
        self._attr_name = f"ThermoSmart {zone_cfg['name']} Vorheizzeit"
        self._attr_native_unit_of_measurement = "min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_PREHEAT

    @property
    def native_value(self):
        return self._zone_data.get("preheat_minutes", 0)


class ThermoSmartConfidenceSensor(_ThermoSmartBaseSensor):
    """Learning confidence as a percentage."""

    def __init__(self, coordinator, entry, zone_id, zone_cfg):
        super().__init__(coordinator, entry, zone_id, zone_cfg, "learning_confidence")
        self._attr_unique_id = f"{DOMAIN}_{zone_id}_confidence"
        self._attr_name = f"ThermoSmart {zone_cfg['name']} Lernfortschritt"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_LEARNING

    @property
    def native_value(self):
        raw = self._zone_data.get("learning_confidence", 0.0)
        return round(raw * 100, 1)


class ThermoSmartWeatherOffsetSensor(_ThermoSmartBaseSensor):
    """Current weather-based temperature offset."""

    def __init__(self, coordinator, entry, zone_id, zone_cfg):
        super().__init__(coordinator, entry, zone_id, zone_cfg, "weather_offset")
        self._attr_unique_id = f"{DOMAIN}_{zone_id}_weather_offset"
        self._attr_name = f"ThermoSmart {zone_cfg['name']} Wetterkorrektur"
        self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_WEATHER_ADJUST

    @property
    def native_value(self):
        return self._zone_data.get("weather_offset", 0.0)
