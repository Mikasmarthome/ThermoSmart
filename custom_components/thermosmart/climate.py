"""Climate platform für ThermoSmart – virtuelle Thermostat-Entität pro Zone."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.climate import (
    PRESET_AWAY,
    PRESET_COMFORT,
    PRESET_ECO,
    PRESET_SLEEP,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature, ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN, VERSION,
    HEATING_MODE_AUTO, HEATING_MODE_COMFORT, HEATING_MODE_NIGHT,
    HEATING_MODE_AWAY, HEATING_MODE_VACATION, HEATING_MODE_ECO,
    TEMP_FROST_PROTECTION,
)
from .coordinator import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)

# Heizmodus → HA-Standard-Preset-Namen
# PRESET_COMFORT/ECO/SLEEP/AWAY sind offizielle HA-Konstanten → Cards, Sprachassistenten,
# Automationen funktionieren ohne zusätzliche Konfiguration.
# "auto" und "vacation" haben keine HA-Entsprechung → bleiben als Custom-Strings.
MODE_TO_PRESET = {
    HEATING_MODE_AUTO:     "auto",
    HEATING_MODE_COMFORT:  PRESET_COMFORT,   # "comfort"
    HEATING_MODE_ECO:      PRESET_ECO,       # "eco"
    HEATING_MODE_NIGHT:    PRESET_SLEEP,     # "sleep"
    HEATING_MODE_AWAY:     PRESET_AWAY,      # "away"
    HEATING_MODE_VACATION: "vacation",
}
PRESET_TO_MODE = {v: k for k, v in MODE_TO_PRESET.items()}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([ThermoSmartClimate(coordinator, entry)])


class ThermoSmartClimate(CoordinatorEntity, ClimateEntity):
    """Virtuelle Climate-Entität für eine ThermoSmart-Zone.

    Zeigt: Ist-Temperatur, Zieltemperatur, Modus.
    Steuert: Zieltemperatur (Override), HVAC-Modus, Preset.
    """

    _attr_has_entity_name = True
    _attr_name = None  # Primärentität des Geräts → name = Gerätename
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
    _attr_preset_modes = list(MODE_TO_PRESET.values())
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.PRESET_MODE
        | ClimateEntityFeature.TURN_ON
        | ClimateEntityFeature.TURN_OFF
    )
    _attr_target_temperature_step = 0.5
    _attr_min_temp = 5.0
    _attr_max_temp = 30.0

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        zone_name = entry.data.get("name", "Zone")
        self._attr_unique_id = f"{entry.entry_id}_climate"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"ThermoSmart – {zone_name}",
            manufacturer="ThermoSmart",
            model="AI Heating Controller",
            sw_version=VERSION,
            entry_type="service",  # type: ignore[arg-type]
        )

    # ── Eigenschaften ────────────────────────────────────────────────

    @property
    def _zone(self) -> dict:
        if self.coordinator.data is None:
            return {}
        return self.coordinator.data.get("zone", {})

    @property
    def current_temperature(self) -> float | None:
        live = self.coordinator._live_temp
        return live if live is not None else self._zone.get("current_temp")

    @property
    def current_humidity(self) -> float | None:
        live = self.coordinator._live_humidity
        return live if live is not None else self._zone.get("indoor_humidity")

    @property
    def target_temperature(self) -> float | None:
        return self._zone.get("adjusted_target")

    @property
    def hvac_mode(self) -> HVACMode:
        if not self.coordinator._active_control:
            return HVACMode.OFF
        return HVACMode.HEAT

    @property
    def hvac_action(self) -> HVACAction:
        if not self.coordinator._active_control:
            return HVACAction.OFF
        curr = self._zone.get("current_temp")
        target = self._zone.get("adjusted_target")
        if curr is None or target is None:
            return HVACAction.IDLE
        tolerance = self.coordinator.zone_cfg.get("temp_tolerance", 0.4)
        if curr < target - tolerance:
            return HVACAction.HEATING
        return HVACAction.IDLE

    @property
    def preset_mode(self) -> str:
        return MODE_TO_PRESET.get(self.coordinator._mode, "auto")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        z = self._zone
        trv = z.get("trv_setpoint")
        target = z.get("adjusted_target")
        boost_delta = round(trv - target, 1) if trv is not None and target is not None and trv > target else 0.0
        return {
            "forecast_suppression": f"{z.get('forecast_suppression', 0)}%",
            "preheat_time":         f"{z.get('preheat_minutes', 0)} min",
            "learning_confidence":  f"{round((z.get('learning_confidence', 0) * 100), 1)}%",
            "weather_correction":   z.get("weather_offset"),
            "outdoor_temperature":  z.get("outdoor_temp"),
            "trv_setpoint":         trv,
            "boost_delta":          f"+{boost_delta}°C" if boost_delta > 0 else "0°C",
            "boost_factor":         z.get("boost_factor", 1.0),
            "observation_mode":     not self.coordinator._active_control,
            "window_open":          bool(z.get("window_open", False)),
            "summer_mode":          self.coordinator._is_summer,
            "heating_failure":      self.coordinator._heating_failure_notified,
            "override_active":      bool(z.get("override_active", False)),
            "override_temp":        z.get("override_temp"),
            "preheat_active":       bool(z.get("preheat_active", False)),
        }

    # ── Steuerung ────────────────────────────────────────────────────

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Zieltemperatur manuell setzen (Override)."""
        temp = kwargs.get(ATTR_TEMPERATURE)
        if temp is None:
            return
        self.coordinator.set_override(float(temp))
        await self.coordinator.async_request_refresh()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Aktive Steuerung ein- oder ausschalten."""
        active = hvac_mode == HVACMode.HEAT
        self.coordinator.set_active_control(active)
        self.async_write_ha_state()
        await self.coordinator.async_request_refresh()

    async def async_turn_on(self) -> None:
        await self.async_set_hvac_mode(HVACMode.HEAT)

    async def async_turn_off(self) -> None:
        await self.async_set_hvac_mode(HVACMode.OFF)

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Heizmodus wechseln (Auto, Komfort, Nacht, Abwesend, Urlaub)."""
        mode = PRESET_TO_MODE.get(preset_mode, HEATING_MODE_AUTO)
        self.coordinator.set_mode(mode)
        await self.coordinator.async_request_refresh()

