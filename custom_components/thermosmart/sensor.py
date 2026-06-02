"""Sensor platform für ThermoSmart."""
from __future__ import annotations
from typing import Any
import logging

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, VERSION, ICON_ZONE, ICON_LEARNING, ICON_PREHEAT, ICON_WEATHER_ADJUST
from . import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([
        ThermoSmartTargetTempSensor(coordinator, entry),
        ThermoSmartPreheatSensor(coordinator, entry),
        ThermoSmartConfidenceSensor(coordinator, entry),
        ThermoSmartWeatherOffsetSensor(coordinator, entry),
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


class _Base(CoordinatorEntity, SensorEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry, key: str):
        super().__init__(coordinator)
        self._key = key
        self._attr_device_info = _device_info(entry)

    @property
    def _zone(self) -> dict:
        if self.coordinator.data is None:
            return {}
        return self.coordinator.data.get("zone", {})

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success


class ThermoSmartTargetTempSensor(_Base):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "adjusted_target")
        self._attr_unique_id = f"{entry.entry_id}_adjusted_target"
        self._attr_name = "Zieltemperatur"
        self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
        self._attr_device_class = SensorDeviceClass.TEMPERATURE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_ZONE

    @property
    def native_value(self):
        return self._zone.get("adjusted_target")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        z = self._zone
        return {
            "basis": z.get("base_target"),
            "wetterkorrektur": z.get("weather_offset"),
            "prognose_unterdrückung": f"{z.get('forecast_suppression', 0)}%",
            "fenster_offen": z.get("window_open"),
            "ist_temperatur": z.get("current_temp"),
            "außentemperatur": z.get("outdoor_temp"),
            "modus": z.get("mode"),
        }


class ThermoSmartPreheatSensor(_Base):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "preheat_minutes")
        self._attr_unique_id = f"{entry.entry_id}_preheat_minutes"
        self._attr_name = "Vorheizzeit"
        self._attr_native_unit_of_measurement = "min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_PREHEAT

    @property
    def native_value(self):
        return self._zone.get("preheat_minutes", 0)


class ThermoSmartConfidenceSensor(_Base):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "learning_confidence")
        self._attr_unique_id = f"{entry.entry_id}_confidence"
        self._attr_name = "Vorhersage-Konfidenz"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_LEARNING

    @property
    def native_value(self):
        return round((self._zone.get("learning_confidence", 0.0) * 100), 1)

    @property
    def extra_state_attributes(self) -> dict:
        pct = self.native_value or 0
        if pct < 25:   status = "Sammle Daten"
        elif pct < 50: status = "Erste Muster erkennbar"
        elif pct < 80: status = "Wird zuverlässiger"
        elif pct < 100: status = "Hohe Zuverlässigkeit"
        else:           status = "Maximale Zuverlässigkeit"
        return {"status": status}


class ThermoSmartWeatherOffsetSensor(_Base):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "weather_offset")
        self._attr_unique_id = f"{entry.entry_id}_weather_offset"
        self._attr_name = "Wetterkorrektur"
        self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_WEATHER_ADJUST

    @property
    def native_value(self):
        return self._zone.get("weather_offset", 0.0)
