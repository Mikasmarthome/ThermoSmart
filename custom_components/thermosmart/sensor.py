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
        ThermoSmartTRVSetpointSensor(coordinator, entry),
        ThermoSmartPreheatSensor(coordinator, entry),
        ThermoSmartConfidenceSensor(coordinator, entry),
        ThermoSmartWeatherOffsetSensor(coordinator, entry),
        ThermoSmartStatusSensor(coordinator, entry),
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
            "ist_sommer": z.get("is_summer", False),
        }


class ThermoSmartTRVSetpointSensor(_Base):
    """Zeigt den tatsächlichen Setpoint der ans TRV gesendet wird (inkl. Boost)."""

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "trv_setpoint")
        self._attr_unique_id = f"{entry.entry_id}_trv_setpoint"
        self._attr_name = "TRV Setpoint"
        self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
        self._attr_device_class = SensorDeviceClass.TEMPERATURE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:thermometer-chevron-up"

    @property
    def native_value(self):
        z = self._zone
        return z.get("trv_setpoint") or z.get("adjusted_target")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        z = self._zone
        trv = z.get("trv_setpoint")
        target = z.get("adjusted_target")
        boost = round(trv - target, 1) if trv is not None and target is not None else 0.0
        return {
            "zieltemperatur": target,
            "boost_delta": f"+{boost}°C" if boost > 0 else "0°C",
            "boost_faktor": z.get("boost_factor", 1.0),
            "boost_aktiv": boost > 0,
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
        if pct < 25:    status = "Sammle Daten"
        elif pct < 50:  status = "Erste Muster erkennbar"
        elif pct < 80:  status = "Wird zuverlässiger"
        elif pct < 100: status = "Hohe Zuverlässigkeit"
        else:           status = "Maximale Zuverlässigkeit"

        # Lernstatistiken aus LearningEngine
        le = self.coordinator.learning_engine
        stats = le.get_stats(self.coordinator.zone_id) if le else {}
        return {
            "status": status,
            "beobachtungen": stats.get("total_observations", 0),
            "mit_wind_daten": stats.get("with_wind_data", 0),
            "mit_solar_daten": stats.get("with_solar_data", 0),
            "mit_heizrate": stats.get("with_heat_rate", 0),
            "mit_abkühlrate": stats.get("with_cool_rate", 0),
            "abkühlrate_°c_min": stats.get("avg_cool_rate_per_min"),
            "älteste_beobachtung": stats.get("oldest"),
        }


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


class ThermoSmartStatusSensor(_Base):
    """Übersichts-Sensor: aktueller Betriebsstatus der Zone."""

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "status")
        self._attr_unique_id = f"{entry.entry_id}_status"
        self._attr_name = "Status"
        self._attr_icon = "mdi:home-thermometer"

    @property
    def native_value(self) -> str:
        z = self._zone
        if not self.coordinator._active_control:
            return "Beobachtungsmodus"
        if z.get("is_summer"):
            return "Sommer – Heizung aus"
        if z.get("window_open"):
            return "Fenster offen"
        mode = z.get("mode", "auto")
        if mode == "vacation":
            return "Urlaub"
        if mode == "away":
            return "Abwesend"
        curr = z.get("current_temp")
        target = z.get("adjusted_target")
        if curr is not None and target is not None:
            if curr < target - 0.5:
                return "Heizt"
            return "Temperatur gehalten"
        return "Aktiv"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        z = self._zone
        data = self.coordinator.data or {}
        presence = data.get("presence", {})
        return {
            "ist_temperatur": z.get("current_temp"),
            "zieltemperatur": z.get("adjusted_target"),
            "außentemperatur": z.get("outdoor_temp"),
            "prognose_hochtemp": z.get("forecast_high"),
            "wetter": z.get("weather_condition"),
            "personen_zuhause": len(presence.get("persons_home", [])),
            "alle_weg": presence.get("all_away", False),
            "urlaub": presence.get("vacation", False),
            "sommer_modus": z.get("is_summer", False),
        }
