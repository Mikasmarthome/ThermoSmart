"""Sensor platform für ThermoSmart."""
from __future__ import annotations
from typing import Any
import logging

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN, VERSION, ICON_ZONE, ICON_LEARNING, ICON_PREHEAT, ICON_WEATHER_ADJUST,
    SOLAR_GAIN_THRESHOLD_W, SOLAR_GAIN_MIN_OUTDOOR_C, SOLAR_GAIN_MAX_REDUCTION,
)
from .coordinator import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([
        ThermoSmartTargetTempSensor(coordinator, entry),
        ThermoSmartTRVSetpointSensor(coordinator, entry),
        ThermoSmartTpiSensor(coordinator, entry),
        ThermoSmartPreheatSensor(coordinator, entry),
        ThermoSmartConfidenceSensor(coordinator, entry),
        ThermoSmartWeatherOffsetSensor(coordinator, entry),
        ThermoSmartStatusSensor(coordinator, entry),
        ThermoSmartTempSlopeSensor(coordinator, entry),
        ThermoSmartTempEma1hSensor(coordinator, entry),
        ThermoSmartHeatingPowerSensor(coordinator, entry),
        ThermoSmartHeatLossRateSensor(coordinator, entry),
        ThermoSmartSunIntensitySensor(coordinator, entry),
        ThermoSmartOutcomeScoreSensor(coordinator, entry),
        ThermoSmartTRVObservationsSensor(coordinator, entry),
        ThermoSmartWindowCoolingRateSensor(coordinator, entry),
    ])


def _device_info(entry: ConfigEntry) -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=f"ThermoSmart – {entry.data.get('name', 'Zone')}",
        manufacturer="ThermoSmart",
        model="Self-learning Heating Controller",
        sw_version=VERSION,
        entry_type="service",  # type: ignore[arg-type]
    )


class _Base(CoordinatorEntity, SensorEntity):
    _attr_has_entity_name = True
    _attr_entity_registry_enabled_default = False

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
    _attr_entity_registry_enabled_default = True
    _attr_translation_key = "adjusted_target"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "adjusted_target")
        self._attr_unique_id = f"{entry.entry_id}_adjusted_target"
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
            "effective_target": z.get("effective_target"),   # actual TRV goal (with offset)
            "weather_correction": z.get("weather_offset"),
            "forecast_suppression": f"{z.get('forecast_suppression', 0)}%",
            "window_open": z.get("window_open"),
            "current_temperature": z.get("current_temp"),
            "outdoor_temperature": z.get("outdoor_temp"),
            "mode": z.get("mode"),
            "is_summer": z.get("is_summer", False),
        }


class ThermoSmartTRVSetpointSensor(_Base):
    """Actual setpoint sent to the TRV (including boost)."""
    _attr_entity_registry_enabled_default = True
    _attr_translation_key = "trv_setpoint"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "trv_setpoint")
        self._attr_unique_id = f"{entry.entry_id}_trv_setpoint"
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
            "target_temperature": target,
            "boost_delta": f"+{boost}°C" if boost > 0 else "0°C",
            "boost_factor":           z.get("boost_factor", 1.0),           # compat: 1.0=neutral
            "boost_offset_c":         z.get("boost_offset_c", 0.0),         # LE2 raw prediction: 0.0=neutral
            "applied_boost_offset_c": z.get("applied_boost_offset_c", 0.0), # confirmed dispatch truth
            "boost_active": boost > 0,
        }


class ThermoSmartPreheatSensor(_Base):
    _attr_entity_registry_enabled_default = True
    _attr_translation_key = "preheat_minutes"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "preheat_minutes")
        self._attr_unique_id = f"{entry.entry_id}_preheat_minutes"
        self._attr_native_unit_of_measurement = "min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_PREHEAT

    @property
    def native_value(self):
        return self._zone.get("preheat_minutes", 0)


class ThermoSmartConfidenceSensor(_Base):
    _attr_entity_registry_enabled_default = True
    _attr_translation_key = "confidence"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "confidence")
        self._attr_unique_id = f"{entry.entry_id}_confidence"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_LEARNING

    @property
    def native_value(self):
        shadow = getattr(self.coordinator, "_le2_shadow", None)
        if shadow is None:
            return 0.0
        progress_pct, _ = shadow.learning_progress_safe()
        return progress_pct

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        shadow = getattr(self.coordinator, "_le2_shadow", None)
        if shadow is None:
            return {
                "data_source": "current_learning_engine",
                "progress_status": "cold_start",
                "reason": "shadow_unavailable",
                "bootstrap_excluded": True,
                "historical_snapshot_used": False,
            }
        _, attrs = shadow.learning_progress_safe()
        return attrs


class ThermoSmartWeatherOffsetSensor(_Base):
    _attr_entity_registry_enabled_default = False
    _attr_translation_key = "weather_offset"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "weather_offset")
        self._attr_unique_id = f"{entry.entry_id}_weather_offset"
        self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_WEATHER_ADJUST

    @property
    def native_value(self):
        return self._zone.get("weather_offset", 0.0)


class ThermoSmartStatusSensor(_Base):
    """Current operating status of the zone."""
    _attr_entity_registry_enabled_default = True
    _attr_translation_key = "status"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "status")
        self._attr_unique_id = f"{entry.entry_id}_status"
        self._attr_device_class = SensorDeviceClass.ENUM
        self._attr_options = [
            "disabled",
            "observation_mode",
            "controlling_no_learning",
            "heating_failure",
            "preheating",
            "summer",
            "window_open",
            "vacation",
            "away",
            "heating",
            "idle",
            "active",
        ]
        self._attr_icon = "mdi:home-thermometer"

    @property
    def native_value(self) -> str:
        z = self._zone
        active = self.coordinator._active_control
        le = self.coordinator.learning_engine
        learning = le.is_zone_enabled(self.coordinator.zone_id) if le else True

        if not active and not learning:
            return "disabled"
        if not active:
            return "observation_mode"
        if not learning:
            return "controlling_no_learning"
        if z.get("heating_failure"):
            return "heating_failure"
        if z.get("preheat_active"):
            return "preheating"
        if z.get("is_summer"):
            return "summer"
        if z.get("window_open"):
            return "window_open"
        mode = z.get("mode", "auto")
        if mode == "vacation":
            return "vacation"
        if mode == "away":
            return "away"
        curr = z.get("current_temp")
        # Use effective_target (weather-adjusted) for accurate heating/idle status
        target = z.get("effective_target") or z.get("adjusted_target")
        if curr is not None and target is not None:
            if curr < target - 0.5:
                return "heating"
            return "idle"
        return "active"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        z = self._zone
        data = self.coordinator.data or {}
        presence = data.get("presence", {})
        return {
            "current_temperature": z.get("current_temp"),
            "target_temperature": z.get("adjusted_target"),
            "outdoor_temperature": z.get("outdoor_temp"),
            "forecast_high": z.get("forecast_high"),
            "weather_condition": z.get("weather_condition"),
            "persons_home": len(presence.get("persons_home", [])),
            "all_away": presence.get("all_away", False),
            "vacation": presence.get("vacation", False),
            "summer_mode": z.get("is_summer", False),
            "heating_failure": z.get("heating_failure", False),
            "slope_window_active": getattr(self.coordinator, "_slope_window_active", False),
        }


# ── Diagnostic sensors ────────────────────────────────────────────────────────

class ThermoSmartTempSlopeSensor(_Base):
    """Temperature change rate – positive = warming, negative = cooling."""
    _attr_entity_registry_enabled_default = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "temp_slope"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "temp_slope")
        self._attr_unique_id = f"{entry.entry_id}_temp_slope"
        self._attr_native_unit_of_measurement = "K/min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:chart-line"

    @property
    def native_value(self):
        return self._zone.get("temp_slope", 0.0)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        slope = self._zone.get("temp_slope")
        if slope is None:
            return {
                "trend": "unknown",
                "data_source": "zone_temperature_history",
                "reason": "insufficient_history",
                "zone_bound": True,
            }
        if slope > 0.01:
            trend = "heating"
        elif slope < -0.01:
            trend = "cooling"
        else:
            trend = "stable"
        return {"trend": trend, "data_source": "zone_temperature_history", "zone_bound": True}


class ThermoSmartHeatingPowerSensor(_Base):
    """Average heating rate (K/min) — LE 2.0 source, compatibility unit kept.

    Value is LE 2.0 HEAT_RATE (°C/h) divided by 60 for backwards-compatible
    K/min display.  LE v1 is no longer read for this value.
    """
    _attr_entity_registry_enabled_default = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "heating_power"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "heating_power")
        self._attr_unique_id = f"{entry.entry_id}_heating_power"
        self._attr_native_unit_of_measurement = "K/min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:thermometer-plus"

    @property
    def native_value(self):
        sh = getattr(self.coordinator, "_le2_shadow", None)
        if sh is not None:
            rate_c_per_h, status = sh.read_heat_rate_safe()
            if status == "valid" and rate_c_per_h > 0:
                return round(rate_c_per_h / 60.0, 5)
        # No shadow or no valid rate: entity shows "unavailable".
        # LE v1 is NOT read — its heat rate is historical data only.
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        sh = getattr(self.coordinator, "_le2_shadow", None)
        if sh is not None:
            rate_c_per_h, status = sh.read_heat_rate_safe()
            attrs: dict[str, Any] = {
                "source": "learning_engine",
                "rate_per_hour_K": round(rate_c_per_h, 4) if rate_c_per_h else None,
                "status": status,
            }
            if status != "valid":
                attrs["data_status"] = "learning"
            return attrs
        return {"source": "unavailable", "data_status": "learning"}


class ThermoSmartTpiSensor(_Base):
    """TPI Duty-Cycle – how far the valve should theoretically open (0–100%).

    With a recognised direct-valve entity (valve_position, pi_heating_demand,
    heating_demand, level): written directly to the valve.
    Without (e.g. SONOFF TRVZB): converted to a setpoint boost via
    climate.set_temperature.
    """
    _attr_entity_registry_enabled_default = True
    _attr_translation_key = "tpi_duty_cycle"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "tpi_duty_cycle")
        self._attr_unique_id = f"{entry.entry_id}_tpi_duty_cycle"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:valve"

    @property
    def native_value(self):
        return self._zone.get("tpi_duty_cycle", 0.0)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        z = self._zone
        coef_int = z.get("tpi_coef_int")
        coef_ext = z.get("tpi_coef_ext")
        valve_direct = z.get("tpi_valve_direct", False)
        return {
            "coef_int": coef_int,
            "coef_ext": coef_ext,
            "coef_source": "default" if coef_int is None or coef_int == 0.6 and coef_ext == 0.01 else "learned",
            "valve_direct": valve_direct,
            "mode": "direct_valve" if valve_direct else "setpoint_boost",
        }


class ThermoSmartHeatLossRateSensor(_Base):
    """Learned heat loss rate (K/min) — LE 2.0 source, compatibility unit kept.

    Value is LE 2.0 HEAT_LOSS_RATE (°C/h) divided by 60 for backwards-compatible
    K/min display.  LE v1 is no longer read for this value when shadow is attached.
    """
    _attr_entity_registry_enabled_default = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "heat_loss"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "heat_loss")
        self._attr_unique_id = f"{entry.entry_id}_heat_loss_rate"
        self._attr_native_unit_of_measurement = "K/min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:thermometer-minus"

    @property
    def native_value(self):
        sh = getattr(self.coordinator, "_le2_shadow", None)
        if sh is not None:
            rate_c_per_h, status = sh.read_heat_loss_rate_safe()
            if status == "valid" and rate_c_per_h >= 0:
                return round(rate_c_per_h / 60.0, 5)
        # No shadow or no valid rate: entity shows "unavailable".
        # LE v1 is NOT read — its heat loss rate is historical data only.
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        sh = getattr(self.coordinator, "_le2_shadow", None)
        if sh is not None:
            rate_c_per_h, status = sh.read_heat_loss_rate_safe()
            attrs: dict[str, Any] = {
                "source": "learning_engine",
                "rate_per_hour_K": round(rate_c_per_h, 4) if rate_c_per_h else None,
                "status": status,
            }
            if status != "valid":
                attrs["data_status"] = "learning"
            return attrs
        return {"source": "unavailable", "data_status": "learning"}


class ThermoSmartSunIntensitySensor(_Base):
    """Solar heat gain – how much the sun is currently reducing heating demand."""
    _attr_entity_registry_enabled_default = False
    _attr_translation_key = "sun_intensity"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "sun_intensity")
        self._attr_unique_id = f"{entry.entry_id}_sun_intensity"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:weather-sunny"

    @property
    def native_value(self):
        data = self.coordinator.data or {}
        weather = data.get("weather", {})
        solar = weather.get("solar_radiation") or 0.0
        outdoor = weather.get("temperature") or 0.0
        if solar <= SOLAR_GAIN_THRESHOLD_W or outdoor <= SOLAR_GAIN_MIN_OUTDOOR_C:
            return 0
        reduction = min((solar - SOLAR_GAIN_THRESHOLD_W) / 1000.0, SOLAR_GAIN_MAX_REDUCTION)
        return round(reduction * 100, 1)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data or {}
        weather = data.get("weather", {})
        return {
            "solar_w_m2": weather.get("solar_radiation"),
            "outdoor_temperature": weather.get("temperature"),
        }


class ThermoSmartOutcomeScoreSensor(_Base):
    """Outcome score of recent heating sessions (0–100%).

    Sourced from the active zone-separated learning engine (OutcomeModel).
    Returns None while in cold start (no episodes yet); updates as episodes
    are completed and evaluated. Uses the weighted performance score across
    physical target achievement, comfort quality and controller quality.
    """
    _attr_entity_registry_enabled_default = False
    _attr_translation_key = "outcome_score"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "outcome_score")
        self._attr_unique_id = f"{entry.entry_id}_outcome_score"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:target"

    @property
    def native_value(self):
        shadow = getattr(self.coordinator, "_le2_shadow", None)
        if shadow is None:
            return None
        score_pct, _status = shadow.read_outcome_score_safe()
        return score_pct

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        shadow = getattr(self.coordinator, "_le2_shadow", None)
        if shadow is None:
            return {"data_source": "current_learning_engine", "data_status": "not_available"}
        return shadow.read_outcome_attributes_safe()


class ThermoSmartTRVObservationsSensor(_Base):
    """Number of learned TRV setpoint observations from observation mode."""
    _attr_entity_registry_enabled_default = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "trv_observations"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "trv_observations")
        self._attr_unique_id = f"{entry.entry_id}_trv_observations"
        self._attr_native_unit_of_measurement = None
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:database-eye"

    @property
    def native_value(self):
        shadow = getattr(self.coordinator, "_le2_shadow", None)
        if shadow is not None:
            breakdown = shadow.confidence_display_attributes()
            return breakdown.get("trv_observations", 0)
        return 0

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        shadow = getattr(self.coordinator, "_le2_shadow", None)
        if shadow is not None:
            breakdown = shadow.confidence_display_attributes()
            return {
                "source": "runtime",
                "window_events": breakdown.get("window_events", 0),
                "mode": "observation" if not self.coordinator._active_control else "active",
            }
        return {
            "source": "unavailable",
            "mode": "observation" if not self.coordinator._active_control else "active",
        }


class ThermoSmartTempEma1hSensor(_Base):
    """1-hour exponential moving average of indoor temperature.

    Smoothed trend indicator – slower than temp_slope, filters short spikes.
    Useful for detecting whether the room is genuinely warming or cooling over time.
    """
    _attr_entity_registry_enabled_default = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "temp_ema_1h"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "temp_ema_1h")
        self._attr_unique_id = f"{entry.entry_id}_temp_ema_1h"
        self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
        self._attr_device_class = SensorDeviceClass.TEMPERATURE
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:chart-bell-curve"

    @property
    def native_value(self):
        return self._zone.get("temp_ema_1h")


class ThermoSmartWindowCoolingRateSensor(_Base):
    """Learned cooling rate when a window is open (K/min).

    Estimated from past window-open events weighted by similarity to current
    outdoor conditions (temperature, wind speed).
    Returns None until at least 2 window events have been recorded.
    """
    _attr_entity_registry_enabled_default = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "window_cooling_rate"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "window_cooling_rate")
        self._attr_unique_id = f"{entry.entry_id}_window_cooling_rate"
        self._attr_native_unit_of_measurement = "K/min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:window-open"

    @property
    def native_value(self):
        # LE1 is frozen — its window cooling rate is a historical snapshot, never updated.
        # LE2 models window cooling via HeatLossModel but has no dedicated public
        # read method for K/min rate yet. Return None until a live source exists.
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        shadow = getattr(self.coordinator, "_le2_shadow", None)
        if shadow is not None:
            breakdown = shadow.confidence_display_attributes()
            return {
                "data_status": "learning",
                "window_events": breakdown.get("window_events", 0),
                "source": "runtime",
            }
        return {"data_status": "learning"}
