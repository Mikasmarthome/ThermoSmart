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
        ThermoSmartHeatingPowerSensor(coordinator, entry),
        ThermoSmartHeatLossRateSensor(coordinator, entry),
        ThermoSmartSunIntensitySensor(coordinator, entry),
        ThermoSmartOutcomeScoreSensor(coordinator, entry),
        ThermoSmartTRVObservationsSensor(coordinator, entry),
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
    _attr_entity_registry_enabled_default = False  # Standard: ausgeblendet; aktive Sensoren überschreiben

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
    _attr_entity_registry_enabled_default = True

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
    _attr_entity_registry_enabled_default = True

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
    _attr_entity_registry_enabled_default = True

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "learning_confidence")
        self._attr_unique_id = f"{entry.entry_id}_confidence"
        self._attr_name = "Lernfortschritt"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = ICON_LEARNING

    @property
    def native_value(self):
        return round((self._zone.get("learning_confidence", 0.0) * 100), 1)

    @property
    def extra_state_attributes(self) -> dict:
        le = self.coordinator.learning_engine
        if le is None:
            return {}
        breakdown = le.get_confidence_breakdown(self.coordinator.zone_id)
        phase = le.get_cold_start_phase(self.coordinator.zone_id)
        stats = le.get_stats(self.coordinator.zone_id)
        return {
            "lernphase": phase,
            "raum_muster_%": breakdown["raum_muster_%"],
            "trv_effizienz_%": breakdown["trv_effizienz_%"],
            "fenster_abkühlung_%": breakdown["fenster_abkühlung_%"],
            "prognose_vertrauen_%": breakdown["prognose_vertrauen_%"],
            "beobachtungen_gesamt": breakdown["beobachtungen_gesamt"],
            "trv_beobachtungen": breakdown["trv_beobachtungen"],
            "fenster_ereignisse": breakdown["fenster_ereignisse"],
            "älteste_beobachtung": stats.get("oldest"),
        }


class ThermoSmartWeatherOffsetSensor(_Base):
    _attr_entity_registry_enabled_default = False

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
    _attr_entity_registry_enabled_default = True

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "status")
        self._attr_unique_id = f"{entry.entry_id}_status"
        self._attr_name = "Status"
        self._attr_icon = "mdi:home-thermometer"

    @property
    def native_value(self) -> str:
        z = self._zone
        active = self.coordinator._active_control
        le = self.coordinator.learning_engine
        learning = le.is_zone_enabled(self.coordinator.zone_id) if le else True

        if not active and not learning:
            return "Deaktiviert"
        if not active:
            return "Beobachtungsmodus"
        if not learning:
            return "Steuert (Lernmodus aus)"
        # Aktiv + Lernen an → detaillierter Status
        if z.get("heating_failure"):
            return "Heizungsausfall!"
        if z.get("preheat_active"):
            return "Vorheizen"
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
            "heizungsausfall": z.get("heating_failure", False),
            "slope_fenster_aktiv": getattr(self.coordinator, "_slope_window_active", False),
        }


# ── Diagnose-Sensoren ─────────────────────────────────────────────────────────

class ThermoSmartTempSlopeSensor(_Base):
    """Temperatur-Änderungsrate – positiv = Raum wärmer, negativ = kühler."""
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "temp_slope")
        self._attr_unique_id = f"{entry.entry_id}_temp_slope"
        self._attr_name = "Temperatur Slope"
        self._attr_native_unit_of_measurement = "K/min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:chart-line"

    @property
    def native_value(self):
        return self._zone.get("temp_slope", 0.0)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        slope = self._zone.get("temp_slope", 0.0)
        if slope > 0.01:
            trend = "Aufheizend"
        elif slope < -0.01:
            trend = "Abkühlend"
        else:
            trend = "Stabil"
        return {"trend": trend}



class ThermoSmartHeatingPowerSensor(_Base):
    """Durchschnittliche Aufheizrate (K/min) – aus gelernten Heizphasen."""
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "heating_power")
        self._attr_unique_id = f"{entry.entry_id}_heating_power"
        self._attr_name = "Heating Power"
        self._attr_native_unit_of_measurement = "K/min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:thermometer-plus"

    @property
    def native_value(self):
        le = self.coordinator.learning_engine
        if le is None:
            return None
        stats = le.get_stats(self.coordinator.zone_id)
        return stats.get("avg_heat_rate_per_min")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        le = self.coordinator.learning_engine
        if le is None:
            return {}
        stats = le.get_stats(self.coordinator.zone_id)
        return {
            "messungen": stats.get("with_heat_rate", 0),
            "boost_faktor": self.coordinator.learning_engine.get_boost_factor(self.coordinator.zone_id),
        }


class ThermoSmartTpiSensor(_Base):
    """TPI Duty-Cycle – wie weit das Ventil theoretisch geöffnet sein sollte (0–100%).

    Zeigt den berechneten Duty-Cycle und ob direkte Ventilsteuerung aktiv ist.
    Bei TRVs mit valve_opening_degree: Wert wird direkt ans Ventil geschrieben.
    Bei anderen TRVs: Wert wird in einen Boost-Setpoint umgerechnet.
    """
    _attr_entity_registry_enabled_default = True

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "tpi_duty_cycle")
        self._attr_unique_id = f"{entry.entry_id}_tpi_duty_cycle"
        self._attr_name = "TPI Duty-Cycle"
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
            "koeffizient_intern": coef_int,
            "koeffizient_extern": coef_ext,
            "quelle_koeffizienten": (
                "gelernt" if coef_int not in (None, 0.6) else "standard"
            ),
            "ventil_direkt": valve_direct,
            "modus": "Direkte Ventilsteuerung" if valve_direct else "Setpoint-Konvertierung",
        }


class ThermoSmartHeatLossRateSensor(_Base):
    """Gelernte Wärmeverlustrate (°C/min) – wie schnell der Raum auskühlt.

    Entscheidend für die Vorheizzeit-Berechnung: Wenn die Heizung startet,
    kühlt der Raum gleichzeitig noch ab. Effektive Aufheizrate =
    Heizrate minus Wärmeverlustrate.
    """
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "heat_loss_rate")
        self._attr_unique_id = f"{entry.entry_id}_heat_loss_rate"
        self._attr_name = "Wärmeverlustrate"
        self._attr_native_unit_of_measurement = "K/min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:thermometer-minus"

    @property
    def native_value(self):
        le = self.coordinator.learning_engine
        if le is None:
            return None
        return le.get_heat_loss_rate(self.coordinator.zone_id)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        le = self.coordinator.learning_engine
        if le is None:
            return {}
        zone_id = self.coordinator.zone_id
        rate = le.get_heat_loss_rate(zone_id)
        rate_per_h = round(rate * 60, 3) if rate else None
        obs_count = sum(
            1 for o in le._observations.get(zone_id, []) if o.get("cool_rate")
        )
        return {
            "rate_pro_stunde_K": rate_per_h,
            "ema_aktiv": zone_id in le._heat_loss_ema,
            "messungen": obs_count,
        }


class ThermoSmartSunIntensitySensor(_Base):
    """Solarer Wärmeeintrag – wie stark die Sonne gerade den Heizbedarf reduziert."""
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "sun_intensity")
        self._attr_unique_id = f"{entry.entry_id}_sun_intensity"
        self._attr_name = "Sun Intensity Heatup"
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
            "außentemperatur": weather.get("temperature"),
        }



class ThermoSmartOutcomeScoreSensor(_Base):
    """Outcome-Score der letzten Heizsitzungen (0–100%).

    Bewertet wie gut eine Heizentscheidung war:
    - Ziel schnell und stabil erreicht → hoher Score
    - Ziel zu spät oder gar nicht erreicht → niedriger Score
    - Starkes Überschießen → niedriger Score
    Funktioniert sowohl für ThermoSmart (aktive Steuerung) als auch
    für externe Regler im Beobachtungsmodus.
    """
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "outcome_score")
        self._attr_unique_id = f"{entry.entry_id}_outcome_score"
        self._attr_name = "Outcome Score"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:target"

    @property
    def native_value(self):
        le = self.coordinator.learning_engine
        if le is None:
            return None
        stats = le.get_outcome_stats(self.coordinator.zone_id)
        return stats.get("avg_score_%")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        le = self.coordinator.learning_engine
        if le is None:
            return {}
        return le.get_outcome_stats(self.coordinator.zone_id)


class ThermoSmartTRVObservationsSensor(_Base):
    """Anzahl gelernter TRV-Setpoint-Beobachtungen aus dem Beobachtungsmodus."""
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "trv_observations")
        self._attr_unique_id = f"{entry.entry_id}_trv_observations"
        self._attr_name = "TRV Beobachtungen"
        self._attr_native_unit_of_measurement = None
        self._attr_state_class = SensorStateClass.TOTAL_INCREASING
        self._attr_icon = "mdi:database-eye"

    @property
    def native_value(self):
        le = self.coordinator.learning_engine
        if le is None:
            return 0
        return le.get_trv_stats(self.coordinator.zone_id).get("trv_observations", 0)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        le = self.coordinator.learning_engine
        if le is None:
            return {}
        stats = le.get_trv_stats(self.coordinator.zone_id)
        return {
            "setpoint_effizienz": stats.get("avg_setpoint_efficiency"),
            "fenster_abkühlrate": stats.get("avg_window_cooling_rate"),
            "lernmodus": "Beobachtungsmodus" if not self.coordinator._active_control else "Aktiv",
        }
