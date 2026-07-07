"""TRV control, calibration and quirk management – ThermoSmart TRVControlMixin."""
from __future__ import annotations

import asyncio
import logging
import math

from homeassistant.const import UnitOfTemperature
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from .device_profiles import DeviceProfile, get_profile, VALVE_MAX_LIMIT, SETPOINT_HVAC_FIRST
from .temperature_units import from_internal_temperature_c, to_internal_temperature_c
from .const import (
    AUTO_QUIRK_PATTERNS,
    AUTO_CALIBRATION_PATTERNS,
    AUTO_EXT_TEMP_PATTERNS,
    AUTO_TEMP_SOURCE_PATTERNS,
    AUTO_VALVE_PATTERNS,
    CONF_CALIBRATION_ENTITIES,
    CONF_CALIBRATION_INVERT,
    CONF_MANAGE_TEMP_SOURCE,
    CONF_QUIRK_ENTITIES,
    CONF_WINDOW_OPEN_TEMP,
    TEMP_SOURCE_SENSOR_GRACE_SECONDS,
    WINDOW_OPEN_SETPOINT,
    TPI_VALVE_BUMP_PCT,
    TPI_VALVE_BUMP_DELAY,
)

_LOGGER = logging.getLogger(__name__)


# ── Dispatch statistics (privacy-safe, no entity IDs) ────────────────────────

class _DispatchStats:
    """Accumulates per-entity dispatch results without exposing entity IDs."""
    __slots__ = ("targets_total", "targets_succeeded", "targets_failed",
                 "failure_reasons", "effective_setpoints",
                 "effective_setpoints_by_entity")

    def __init__(self) -> None:
        self.targets_total: int = 0
        self.targets_succeeded: int = 0
        self.targets_failed: int = 0
        self.failure_reasons: list = []
        self.effective_setpoints: list = []  # actual per-device °C values written (successes only)
        # entity_id → effective °C written (successes only; keyed for counterfactual baseline).
        self.effective_setpoints_by_entity: dict = {}

    def record(self, exc=None, *, effective_c=None, entity_id: str | None = None) -> None:
        """Record one dispatch attempt; pass exc to mark it as failed.

        effective_c: actual setpoint °C that was written (for success, ignored on failure).
        entity_id: entity whose setpoint was written; populates effective_setpoints_by_entity.
        """
        self.targets_total += 1
        if exc is not None:
            self.targets_failed += 1
            self.failure_reasons.append(_normalize_dispatch_error(exc))
        else:
            self.targets_succeeded += 1
            if effective_c is not None:
                self.effective_setpoints.append(float(effective_c))
                if entity_id is not None:
                    self.effective_setpoints_by_entity[entity_id] = float(effective_c)

    def record_gather(self, results) -> None:
        """Process the list of results from asyncio.gather(return_exceptions=True).

        For per-entity effective setpoint tracking, use record() directly with effective_c.
        """
        for r in results:
            if isinstance(r, BaseException):
                self.targets_failed += 1
                self.failure_reasons.append(_normalize_dispatch_error(r))
            else:
                self.targets_succeeded += 1
        self.targets_total += len(results)

    def merge(self, other: "_DispatchStats") -> "_DispatchStats":
        merged = _DispatchStats()
        merged.targets_total = self.targets_total + other.targets_total
        merged.targets_succeeded = self.targets_succeeded + other.targets_succeeded
        merged.targets_failed = self.targets_failed + other.targets_failed
        merged.failure_reasons = self.failure_reasons + other.failure_reasons
        merged.effective_setpoints = self.effective_setpoints + other.effective_setpoints
        merged.effective_setpoints_by_entity = {
            **self.effective_setpoints_by_entity, **other.effective_setpoints_by_entity}
        return merged

    @property
    def status(self) -> str:
        if self.targets_total == 0:
            return "not_attempted"
        if self.targets_failed == 0:
            return "fully_succeeded"
        if self.targets_succeeded == 0:
            return "failed"
        return "partially_succeeded"


def _normalize_dispatch_error(exc: BaseException) -> str:
    """Return a privacy-safe dispatch error label (no entity IDs, no service payloads)."""
    try:
        from homeassistant.exceptions import (
            ServiceNotFound, ServiceValidationError, HomeAssistantError)
        if isinstance(exc, ServiceNotFound):
            return "service_not_found"
        if isinstance(exc, ServiceValidationError):
            return "service_validation_error"
        if isinstance(exc, HomeAssistantError):
            return "ha_service_error"
    except ImportError:
        pass
    return type(exc).__name__


class TRVControlMixin:
    """TRV-Steuerung, automatische Kalibrierung, Quirk-Management und Geräteerkennung."""

    # ── Geräteerkennung ──────────────────────────────────────────────

    async def async_detect_device_entities(self) -> None:
        """Erkennt Quirk-Switches und Kalibrierungs-Entities via Device Registry."""
        cfg = self.zone_cfg
        climate_entities = cfg.get("climate_entities", [])
        if not climate_entities:
            return

        ent_reg = er.async_get(self.hass)

        device_to_climate: dict[str, str] = {}
        for entity_id in climate_entities:
            entry = ent_reg.async_get(entity_id)
            if entry and entry.device_id:
                device_to_climate[entry.device_id] = entity_id

        if not device_to_climate:
            return

        # ── Device profile detection ──────────────────────────────────────────
        dev_reg = dr.async_get(self.hass)
        profile_map: dict[str, DeviceProfile] = {}
        for device_id, climate_id in device_to_climate.items():
            dev_entry = dev_reg.async_get(device_id)
            manufacturer = dev_entry.manufacturer if dev_entry else None
            model = dev_entry.model if dev_entry else None
            profile = get_profile(model, manufacturer)
            profile_map[climate_id] = profile
            _LOGGER.debug(
                "ThermoSmart '%s': device profile → %s (entity=%s, manufacturer=%r, model=%r)",
                self.zone_name, profile.identifier, climate_id, manufacturer, model,
            )

        quirks: list[str] = []
        cal_map: dict[str, str] = {}

        for device_id, climate_id in device_to_climate.items():
            for entry in er.async_entries_for_device(ent_reg, device_id):
                domain = entry.entity_id.split(".")[0]

                if domain == "switch":
                    for pattern in AUTO_QUIRK_PATTERNS:
                        if pattern in entry.entity_id:
                            quirks.append(entry.entity_id)
                            break

                elif domain == "number":
                    if any(p in entry.entity_id for p in AUTO_CALIBRATION_PATTERNS):
                        cal_map[climate_id] = entry.entity_id

        manual_quirks = set(cfg.get(CONF_QUIRK_ENTITIES, []))
        new_quirks = [e for e in quirks if e not in manual_quirks]
        if new_quirks:
            _LOGGER.info("ThermoSmart '%s': Quirk-Autodetect: %s", self.zone_name, new_quirks)
        if cal_map:
            _LOGGER.info("ThermoSmart '%s': Kalibrierungs-Autodetect: %s", self.zone_name, cal_map)

        # External Temperature Input Entities (TRVZB: Raumtemp direkt ins TRV schreiben)
        ext_temp_map: dict[str, str] = {}
        for device_id, climate_id in device_to_climate.items():
            for entry in er.async_entries_for_device(ent_reg, device_id):
                if entry.entity_id.split(".")[0] != "number":
                    continue
                eid_lower = entry.entity_id.lower()
                tk = getattr(entry, "translation_key", None) or ""
                name = (getattr(entry, "original_name", None) or "").lower()
                for pattern in AUTO_EXT_TEMP_PATTERNS:
                    if pattern in eid_lower or pattern in tk.lower() or pattern in name:
                        ext_temp_map[climate_id] = entry.entity_id
                        break

        if ext_temp_map:
            _LOGGER.info(
                "ThermoSmart '%s': External-Temp-Input-Autodetect: %s",
                self.zone_name, ext_temp_map,
            )

        # Direct valve position control (TPI duty-cycle written as 0–100%).
        valve_map: dict[str, str] = {}
        for device_id, climate_id in device_to_climate.items():
            for entry in er.async_entries_for_device(ent_reg, device_id):
                if entry.entity_id.split(".")[0] != "number":
                    continue
                eid_lower = entry.entity_id.lower()
                tk = (getattr(entry, "translation_key", None) or "").lower()
                name = (getattr(entry, "original_name", None) or "").lower()
                for pattern in AUTO_VALVE_PATTERNS:
                    if pattern in eid_lower or pattern in tk or pattern in name:
                        # Nicht mit ext_temp_map überschneiden
                        if entry.entity_id not in ext_temp_map.values():
                            valve_map[climate_id] = entry.entity_id
                        break

        if valve_map:
            _LOGGER.info(
                "ThermoSmart '%s': Ventil-Autodetect (direkte %-Steuerung): %s",
                self.zone_name, valve_map,
            )

        # temperature_sensor select entity (SONOFF TRVZB: switches TRV between
        # internal and external temperature source)
        temp_source_map: dict[str, str] = {}
        for device_id, climate_id in device_to_climate.items():
            for entry in er.async_entries_for_device(ent_reg, device_id):
                if entry.entity_id.split(".")[0] != "select":
                    continue
                eid_lower = entry.entity_id.lower()
                tk = (getattr(entry, "translation_key", None) or "").lower()
                name = (getattr(entry, "original_name", None) or "").lower()
                for pattern in AUTO_TEMP_SOURCE_PATTERNS:
                    if pattern in eid_lower or pattern in tk or pattern in name:
                        temp_source_map[climate_id] = entry.entity_id
                        break

        if temp_source_map:
            _LOGGER.info(
                "ThermoSmart '%s': Temperature-Source-Select-Autodetect: %s",
                self.zone_name, temp_source_map,
            )

        self._device_profiles = profile_map
        await self._reset_valve_opening_degree()

        self._auto_quirk_entities = quirks
        self._auto_calibration_map = cal_map
        self._auto_ext_temp_map = ext_temp_map
        self._auto_valve_map = valve_map
        self._auto_temp_source_map = temp_source_map

    async def _reset_valve_opening_degree(self) -> bool:
        """Reset valve_opening_degree to 100% on all managed TRV devices.

        valve_opening_degree is a max-opening limit on SONOFF TRVZB (and similar
        devices) — not a live valve position.  Previous ThermoSmart versions wrote
        the TPI duty-cycle to it, which capped heating capacity at the duty-cycle
        value (e.g. 0% blocked the valve entirely).  Resetting to 100% restores
        full heating capacity on upgrade.

        Called from async_detect_device_entities (fires immediately on Integration
        Reload when Zigbee2MQTT entities are already available) and from
        _async_update_data on the first coordinator refreshes as a safety-net for
        full HA restart, where Z2M entities are not yet available at setup time.

        Returns:
            True  — recovery complete or nothing to do (caller should not retry).
            False — at least one relevant entity was unavailable; caller should
                    retry on the next refresh cycle.
        """
        cfg = self.zone_cfg
        climate_entities = cfg.get("climate_entities", [])
        if not climate_entities:
            return True

        ent_reg = er.async_get(self.hass)
        device_to_climate: dict[str, str] = {}
        for entity_id in climate_entities:
            entry = ent_reg.async_get(entity_id)
            if entry and entry.device_id:
                device_to_climate[entry.device_id] = entity_id

        retry_needed = False
        for device_id in device_to_climate:
            climate_id = device_to_climate[device_id]
            _profile = getattr(self, "_device_profiles", {}).get(climate_id)
            if _profile is not None and _profile.valve_semantics != VALVE_MAX_LIMIT:
                continue
            for entry in er.async_entries_for_device(ent_reg, device_id):
                if entry.entity_id.split(".")[0] != "number":
                    continue
                eid_lower = entry.entity_id.lower()
                tk = (getattr(entry, "translation_key", None) or "").lower()
                name = (getattr(entry, "original_name", None) or "").lower()
                if not ("valve_opening_degree" in eid_lower
                        or "valve_opening_degree" in tk
                        or "valve_opening_degree" in name):
                    continue
                state = self.hass.states.get(entry.entity_id)
                if state is None or state.state in ("unavailable", "unknown"):
                    retry_needed = True
                    continue
                try:
                    if float(state.state) < 100.0:
                        _LOGGER.info(
                            "ThermoSmart '%s': Resetting %s from %s%% to 100%% "
                            "(valve_opening_degree is a max-opening limit, not a live "
                            "position — TPI duty-cycle is no longer written here)",
                            self.zone_name, entry.entity_id, state.state,
                        )
                        self.hass.async_create_task(
                            self.hass.services.async_call(
                                "number", "set_value",
                                {"entity_id": entry.entity_id, "value": 100},
                                blocking=False,
                            )
                        )
                except (TypeError, ValueError):
                    pass

        return not retry_needed

    # ── TRV-Offline-Tracking ─────────────────────────────────────────

    def _get_trv_state(self, entity_id: str):
        """Gibt TRV-State zurück; loggt Offline/Online-Übergänge."""
        state = self.hass.states.get(entity_id)
        if state is None or state.state in ("unavailable", "unknown"):
            if entity_id not in self._trv_offline:
                self._trv_offline.add(entity_id)
                _LOGGER.warning(
                    "ThermoSmart '%s': TRV %s nicht erreichbar – Steuerung ausgesetzt",
                    self.zone_name, entity_id,
                )
            return None
        if entity_id in self._trv_offline:
            self._trv_offline.discard(entity_id)
            _LOGGER.info(
                "ThermoSmart '%s': TRV %s wieder erreichbar",
                self.zone_name, entity_id,
            )
        return state

    # ── Watchdog ─────────────────────────────────────────────────────

    # HVAC-Modi die ThermoSmart niemals beim TRV haben möchte:
    # - "auto"      → TRV nutzt eigenen internen Zeitplan, ignoriert externe Setpoints
    # - "heat_cool" → Automatisches Heizen/Kühlen ohne externe Kontrolle
    # - "cool"      → Kühlmodus, nicht gewünscht für Heizzonen
    # - "off"       → Watchdog setzt zurück auf heat
    _UNWANTED_MODES = {"auto", "heat_cool", "cool", "dry", "fan_only", "off"}

    async def _watchdog_hvac(self, cfg: dict, recommendation: dict) -> None:
        """TRVs in unerwünschten Modi auf 'heat' zurücksetzen.

        Fängt alle Modi die ThermoSmart's Steuerung untergraben würden:
        - 'off':       TRV heizt gar nicht mehr
        - 'auto':      TRV nutzt eigenen internen Zeitplan (TRVZB-Hauptproblem!)
                       Im Auto-Mode ignoriert der TRVZB externe set_temperature-Befehle
                       vollständig und folgt nur noch seinem eigenen Wochenprogramm.
        - 'heat_cool': Automatischer Modus ohne externe Kontrolle
        - andere:      Kühlen, Fan etc.

        Sonderfall no-off-Mode: TRVs ohne 'heat'-Unterstützung werden übersprungen.
        """
        if recommendation.get("window_open"):
            return
        if recommendation.get("forecast_suppression", 0) >= 100:
            return

        for entity_id in cfg.get("climate_entities", []):
            state = self._get_trv_state(entity_id)
            if state is None:
                continue

            _profile = self._device_profiles.get(entity_id)
            if _profile is not None and not _profile.hvac_watchdog:
                _LOGGER.debug(
                    "ThermoSmart '%s': Watchdog skipped for %s – profile '%s' has hvac_watchdog=False",
                    self.zone_name, entity_id, _profile.identifier,
                )
                continue

            if state.state not in self._UNWANTED_MODES:
                continue  # TRV ist in 'heat' – alles gut

            hvac_modes = state.attributes.get("hvac_modes", [])
            if "heat" not in hvac_modes:
                _LOGGER.debug(
                    "ThermoSmart '%s': %s hat keinen 'heat'-Mode – Watchdog übersprungen",
                    self.zone_name, entity_id,
                )
                continue

            _LOGGER.info(
                "ThermoSmart Watchdog '%s': %s ist im Modus '%s' → erzwinge 'heat'",
                self.zone_name, entity_id, state.state,
            )
            await self.hass.services.async_call(
                "climate", "set_hvac_mode",
                {"entity_id": entity_id, "hvac_mode": "heat"},
                blocking=False,
            )

    # ── Kalibrierung ─────────────────────────────────────────────────

    async def _async_calibrate_trvs(self, cfg: dict, recommendation: dict) -> None:
        """Lokale TRV-Kalibrierung: Offset zwischen Raumsensor und TRV-Sensor.

        EMA-geglättet (α=0.25), Schreiben nur bei Änderung > 0.5°C.
        Überspringt wenn Heizkörper gerade heizt (Sensor verfälscht durch Radiatorkontakt).
        """
        room_temp = recommendation.get("current_temp")
        if room_temp is None:
            return

        climate_entities = cfg.get("climate_entities", [])
        manual_cal = cfg.get(CONF_CALIBRATION_ENTITIES, [])

        if not self._auto_calibration_map and not manual_cal:
            return

        for i, climate_id in enumerate(climate_entities):
            cal_entity = self._auto_calibration_map.get(climate_id)
            if not cal_entity:
                cal_entity = manual_cal[i] if i < len(manual_cal) else None
            if not cal_entity:
                continue

            _cal_profile = self._device_profiles.get(climate_id)
            if _cal_profile is not None and not _cal_profile.calibration_sync:
                _LOGGER.debug(
                    "ThermoSmart '%s': calibration skipped for %s – profile '%s' has calibration_sync=False",
                    self.zone_name, climate_id, _cal_profile.identifier,
                )
                continue

            # Skip calibration write when the TRV is confirmed to be using its
            # external temperature source – the offset is ignored by the firmware
            # in that mode, so writing it would be a no-op at best.
            # Only skip on state == "external"; unavailable/unknown means we
            # cannot be sure, so we fall through and calibrate as normal.
            _temp_source_map = getattr(self, "_auto_temp_source_map", {})
            _sel_entity = _temp_source_map.get(climate_id)
            if _sel_entity:
                _sel_state = self.hass.states.get(_sel_entity)
                if _sel_state is not None and _sel_state.state == "external":
                    _LOGGER.debug(
                        "ThermoSmart '%s': skipping calibration for %s – temperature_sensor is 'external'",
                        self.zone_name, climate_id,
                    )
                    continue

            trv_state = self._get_trv_state(climate_id)
            trv_temp: float | None = None
            if trv_state:
                # climate.current_temperature carries no unit attribute of its
                # own — HA already normalizes it to the system unit.
                trv_temp = to_internal_temperature_c(
                    self.hass, trv_state.attributes.get("current_temperature"))

            if trv_temp is None:
                continue

            if trv_temp - room_temp > 3.0:
                _LOGGER.debug(
                    "ThermoSmart '%s': Kalibrierung %s übersprungen – TRV %.1f°C > Raum %.1f°C",
                    self.zone_name, cal_entity, trv_temp, room_temp,
                )
                continue

            raw_offset = room_temp - trv_temp

            # Inversion priority:
            #   1. Key present in cfg → user made an explicit choice; use it as-is.
            #      (key present + False means user consciously disabled inversion)
            #   2. Key absent (old ConfigEntry pre-dating the field) → use profile default.
            #      This is safe: old entries without the key never had user-configured
            #      inversion, so applying the profile default cannot reverse a user setting.
            _invert = (
                cfg[CONF_CALIBRATION_INVERT]
                if CONF_CALIBRATION_INVERT in cfg
                else (_cal_profile.calibration_inverted if _cal_profile is not None else False)
            )
            if _invert:
                raw_offset = -raw_offset

            _plaus_limit = (
                _cal_profile.calibration_plausibility_limit
                if _cal_profile is not None else 7.0
            )
            if abs(raw_offset) > _plaus_limit:
                _LOGGER.warning(
                    "ThermoSmart '%s': Kalibrierungs-Offset %.1f°C für %s unplausibel – übersprungen",
                    self.zone_name, raw_offset, cal_entity,
                )
                continue

            prev = self._calibration_offsets.get(cal_entity, raw_offset)
            smoothed = round(0.25 * raw_offset + 0.75 * prev, 1)
            self._calibration_offsets[cal_entity] = smoothed

            cal_state = self.hass.states.get(cal_entity)
            if cal_state is None:
                continue
            try:
                current_cal = float(cal_state.state)
            except (TypeError, ValueError):
                current_cal = 0.0

            if abs(smoothed - current_cal) < 0.5:
                continue

            clamped = max(-5.0, min(5.0, smoothed))
            _LOGGER.info(
                "ThermoSmart '%s': TRV-Kalibrierung %s → %.1f°C (Raum=%.1f°C, TRV=%.1f°C)",
                self.zone_name, cal_entity, clamped, room_temp, trv_temp,
            )
            self.hass.async_create_task(self.hass.services.async_call(
                "number", "set_value",
                {"entity_id": cal_entity, "value": clamped},
                blocking=False,
            ))

    # ── Quirks ───────────────────────────────────────────────────────

    async def _async_apply_quirks(self, cfg: dict) -> None:
        """TRV-interne Logiken deaktivieren die mit ThermoSmart konkurrieren."""
        manual = cfg.get(CONF_QUIRK_ENTITIES, [])
        quirk_entities = list(dict.fromkeys(manual + self._auto_quirk_entities))
        if not quirk_entities:
            return
        tasks = []
        for entity_id in quirk_entities:
            state = self.hass.states.get(entity_id)
            if state is None or state.state in ("unavailable", "unknown"):
                continue
            if state.state == "on":
                _LOGGER.info(
                    "ThermoSmart '%s': Quirk deaktivieren %s (war 'on')",
                    self.zone_name, entity_id,
                )
                tasks.append(self.hass.services.async_call(
                    "switch", "turn_off", {"entity_id": entity_id}, blocking=False,
                ))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    _LOGGER.debug(
                        "ThermoSmart '%s': Quirk-Deaktivierung fehlgeschlagen: %s",
                        self.zone_name, result,
                    )

    # ── Hilfsroutinen ────────────────────────────────────────────────

    @staticmethod
    def _round_to_step(value: float, step: float) -> float:
        """Round value to the nearest multiple of step (round half up)."""
        return round(math.floor(value / step + 0.5) * step, 2)

    def _snap_to_device_step(self, entity_id: str, state, value: float) -> float:
        """Snap value to the TRV's target_temp_step if the attribute is present.

        Known documented limitation (Temperature-Unit audit): unlike
        min_temp/max_temp/current_temperature/temperature, HA's climate
        platform does NOT normalize target_temp_step to the system unit —
        it is exposed exactly as the entity itself reports it. A step is an
        INTERVAL, not an absolute reading, so it cannot be corrected with
        to_internal_temperature_c() (a point conversion; Fahrenheit-to-Celsius
        includes a +32 offset that must never apply to a delta). Converting
        it correctly would need HA's separate interval-conversion API. Left
        unconverted here: the practical impact is at most a slightly coarser
        or finer snap granularity on a non-Celsius-system installation with a
        device reporting a foreign-unit step — never a safety issue, since
        min/max clamping (already unit-safe above) always runs before and
        after this snap.
        """
        try:
            step = float(state.attributes.get("target_temp_step", 0))
            if step > 0:
                snapped = self._round_to_step(value, step)
                if snapped != value:
                    _LOGGER.debug(
                        "ThermoSmart '%s': %s step-snap %.4f°C → %.2f°C (step=%.2f)",
                        self.zone_name, entity_id, value, snapped, step,
                    )
                return snapped
        except (TypeError, ValueError):
            pass
        return value

    def resolve_device_effective_setpoint(
        self, entity_id: str, state, setpoint: float, profile=None
    ) -> float:
        """Compute the effective setpoint after all per-device transforms.

        Single authority for both the dispatch path and the counterfactual baseline:

            device_applied_boost = resolve_device_effective_setpoint(eid, state, baseline + boost)
                                 - resolve_device_effective_setpoint(eid, state, baseline)

        Transform order (matches _apply_temperature exactly — no drift possible):
          1. HA min_temp clamp
          2. Profile minimum_setpoint clamp
          3. Step-snap (target_temp_step attribute)
          4. Re-clamp min after snap (prevents snap from going below device/profile floor)
          5. HA max_temp clamp (device ceiling)
          6. Profile maximum_setpoint clamp (may be stricter than HA max)

        Args:
            entity_id: for step-snap debug logging.
            state: HA state object with .attributes dict.
            setpoint: input °C (e.g. tpi_baseline_setpoint or baseline + approved_boost).
            profile: DeviceProfile or None.
        Returns:
            Effective °C after all per-device clamps, snap, and ceiling.
        """
        eff = setpoint
        # min_temp/max_temp are HA climate capability_attributes — already
        # normalized to the HA system unit (never the device's native unit)
        # by HA's own climate platform, so a plain unit-less conversion here
        # (falls back to hass.config.units.temperature_unit) is correct and
        # prevents a Fahrenheit-system installation from silently disabling
        # this plausibility band (e.g. 41°F would fail a raw "<= 30" check).
        dev_min = to_internal_temperature_c(self.hass, state.attributes.get("min_temp"))
        dev_max = to_internal_temperature_c(self.hass, state.attributes.get("max_temp"))
        # 1. HA min_temp clamp
        if dev_min is not None and 0 < dev_min <= 30 and dev_min > eff:
            _LOGGER.debug(
                "ThermoSmart '%s': clamped setpoint for %s from %.1f°C to %.1f°C"
                " due to device min_temp",
                self.zone_name, entity_id, eff, dev_min,
            )
            eff = dev_min
        # 2. Profile minimum_setpoint clamp
        if profile is not None and getattr(profile, "minimum_setpoint", None) is not None:
            if profile.minimum_setpoint > eff:
                _LOGGER.debug(
                    "ThermoSmart '%s': clamped setpoint for %s from %.1f°C to %.1f°C"
                    " due to profile minimum_setpoint",
                    self.zone_name, entity_id, eff, profile.minimum_setpoint,
                )
                eff = profile.minimum_setpoint
        # 3. Step-snap (target_temp_step attribute)
        eff = self._snap_to_device_step(entity_id, state, eff)
        # 4. Re-clamp min after snap (snap may have rounded down below floor)
        if dev_min is not None and 0 < dev_min <= 30 and dev_min > eff:
            eff = dev_min
        if profile is not None and getattr(profile, "minimum_setpoint", None) is not None:
            if profile.minimum_setpoint > eff:
                eff = profile.minimum_setpoint
        # 5. HA max_temp clamp (device ceiling, applied after step-snap)
        if dev_max is not None and dev_max > 0 and eff > dev_max:
            _LOGGER.debug(
                "ThermoSmart '%s': clamped setpoint for %s from %.1f°C to %.1f°C"
                " due to device max_temp",
                self.zone_name, entity_id, eff, dev_max,
            )
            eff = dev_max
        # 6. Profile maximum_setpoint clamp (may be stricter than HA max)
        if profile is not None and getattr(profile, "maximum_setpoint", None) is not None:
            if eff > profile.maximum_setpoint:
                _LOGGER.debug(
                    "ThermoSmart '%s': clamped setpoint for %s from %.1f°C to %.1f°C"
                    " due to profile maximum_setpoint",
                    self.zone_name, entity_id, eff, profile.maximum_setpoint,
                )
                eff = profile.maximum_setpoint
        return eff

    # ── Temperatur schreiben ──────────────────────────────────────────

    async def _apply_temperature(self, cfg: dict, recommendation: dict) -> "_DispatchStats":
        """TRV-Setpoints schreiben oder 5°C bei geöffnetem Fenster setzen.

        _last_written_setpoints is updated only for entities where the service call
        actually succeeded.  This prevents false manual-override detection after a
        failed dispatch, since the manual-override listener compares the TRV's
        echoed state against _last_written_setpoints.
        """
        stats = _DispatchStats()
        target = recommendation.get("adjusted_target")

        if target is None:
            if recommendation.get("window_open"):
                effective_frost = cfg.get(CONF_WINDOW_OPEN_TEMP, WINDOW_OPEN_SETPOINT)
                # Collect (entity_id, frost_temp, coro) so results can be paired with entities.
                _frost_pending: list = []
                for entity_id in cfg.get("climate_entities", []):
                    state = self._get_trv_state(entity_id)
                    if state is None:
                        continue
                    frost_temp = cfg.get(CONF_WINDOW_OPEN_TEMP, WINDOW_OPEN_SETPOINT)
                    # min_temp is already normalized to the HA system unit by
                    # HA's climate platform — see resolve_device_effective_setpoint().
                    device_min = to_internal_temperature_c(self.hass, state.attributes.get("min_temp"))
                    if device_min is not None and 0 < device_min <= 30 and device_min > frost_temp:
                        _LOGGER.debug(
                            "ThermoSmart '%s': clamped window-open setpoint for %s"
                            " from %.1f°C to %.1f°C due to device min_temp",
                            self.zone_name, entity_id, frost_temp, device_min,
                        )
                        frost_temp = device_min
                    _wp = self._device_profiles.get(entity_id)
                    if _wp is not None and not _wp.is_active:
                        _LOGGER.debug(
                            "ThermoSmart '%s': skipping setpoint write for %s (profile is_active=False)",
                            self.zone_name, entity_id,
                        )
                        continue
                    if _wp is not None and _wp.minimum_setpoint is not None and _wp.minimum_setpoint > frost_temp:
                        _LOGGER.debug(
                            "ThermoSmart '%s': clamped window-open setpoint for %s"
                            " from %.1f°C to %.1f°C due to profile minimum_setpoint",
                            self.zone_name, entity_id, frost_temp, _wp.minimum_setpoint,
                        )
                        frost_temp = _wp.minimum_setpoint
                    frost_temp = self._snap_to_device_step(entity_id, state, frost_temp)
                    # Final clamp: snap must not fall below device or profile minimum
                    if device_min is not None and 0 < device_min <= 30 and device_min > frost_temp:
                        frost_temp = device_min
                    if _wp is not None and _wp.minimum_setpoint is not None and _wp.minimum_setpoint > frost_temp:
                        frost_temp = _wp.minimum_setpoint
                    effective_frost = frost_temp  # track clamped value for sensor display
                    current_setpoint_c = to_internal_temperature_c(
                        self.hass, state.attributes.get("temperature"))
                    if current_setpoint_c is not None and abs(current_setpoint_c - frost_temp) < 0.3:
                        continue
                    _frost_pending.append((entity_id, frost_temp, self.hass.services.async_call(
                        "climate", "set_temperature",
                        {"entity_id": entity_id,
                         "temperature": from_internal_temperature_c(self.hass, frost_temp)},
                        blocking=True,
                    )))
                if _frost_pending:
                    _results = await asyncio.gather(
                        *[coro for _, _, coro in _frost_pending], return_exceptions=True)
                    for (eid, sp, _), result in zip(_frost_pending, _results):
                        if isinstance(result, BaseException):
                            _LOGGER.debug(
                                "ThermoSmart '%s': window-open setpoint failed for %s: %s",
                                self.zone_name, eid, result,
                            )
                            stats.record(result)
                        else:
                            self._last_written_setpoints[eid] = sp  # success only
                            stats.record(None)
                # Immediately reflect the written frost temp in the sensor display.
                # Without this, the display fallback (which runs before _apply_temperature)
                # would show the stale pre-window-open setpoint for one full cycle.
                recommendation["trv_setpoint"] = effective_frost
            return stats

        if not (5.0 <= target <= 30.0):
            _LOGGER.warning(
                "ThermoSmart '%s': Ziel %.1f°C außerhalb Sicherheitsbereich",
                self.zone_name, target,
            )
            return stats

        trv_setpoint = recommendation.get("trv_setpoint", target)
        tolerance = cfg.get("temp_tolerance", 0.5)

        # Collect gather-path entities so results can be paired with entity IDs and setpoints.
        _sp_pending: list = []  # (entity_id, effective_setpoint, coro)
        for entity_id in cfg.get("climate_entities", []):
            state = self._get_trv_state(entity_id)
            if state is None:
                continue
            _profile = self._device_profiles.get(entity_id)
            if _profile is not None and not _profile.is_active:
                _LOGGER.debug(
                    "ThermoSmart '%s': skipping setpoint write for %s (profile is_active=False)",
                    self.zone_name, entity_id,
                )
                continue
            effective_setpoint = self.resolve_device_effective_setpoint(
                entity_id, state, trv_setpoint, _profile)
            current_setpoint = to_internal_temperature_c(self.hass, state.attributes.get("temperature"))
            if current_setpoint is not None:
                if abs(current_setpoint - effective_setpoint) < tolerance:
                    continue
            _LOGGER.debug(
                "ThermoSmart '%s' → %s: %.1f°C (Ziel=%.1f°C, Boost+%.1f°C)",
                self.zone_name, entity_id, effective_setpoint, target, effective_setpoint - target,
            )
            # Note: _last_written_setpoints is updated below only after confirmed success.

            if _profile is not None and _profile.setpoint_method == SETPOINT_HVAC_FIRST:
                # Sequential write: set HVAC mode first, wait, then write setpoint.
                # Required for TRVs that silently ignore climate.set_temperature unless
                # they are already in the correct HVAC mode (e.g. Eurotronic Spirit).
                mode = _profile.hvac_mode_before_write or "heat"
                if state.state != mode:
                    try:
                        await self.hass.services.async_call(
                            "climate", "set_hvac_mode",
                            {"entity_id": entity_id, "hvac_mode": mode},
                            blocking=True,
                        )
                    except Exception as err:
                        _LOGGER.warning(
                            "ThermoSmart '%s': HVAC mode pre-write failed for %s: %s",
                            self.zone_name, entity_id, err,
                        )
                    if _profile.setpoint_delay_seconds > 0.0:
                        await asyncio.sleep(_profile.setpoint_delay_seconds)
                _hvac_first_exc = None
                try:
                    await self.hass.services.async_call(
                        "climate", "set_temperature",
                        {"entity_id": entity_id,
                         "temperature": from_internal_temperature_c(self.hass, effective_setpoint)},
                        blocking=True,
                    )
                    self._last_written_setpoints[entity_id] = effective_setpoint  # success only
                except Exception as err:
                    _LOGGER.debug(
                        "ThermoSmart '%s': HVAC_FIRST setpoint failed for %s: %s",
                        self.zone_name, entity_id, err,
                    )
                    _hvac_first_exc = err
                stats.record(_hvac_first_exc,
                             effective_c=effective_setpoint if _hvac_first_exc is None else None,
                             entity_id=entity_id if _hvac_first_exc is None else None)
            else:
                _sp_pending.append((entity_id, effective_setpoint, self.hass.services.async_call(
                    "climate", "set_temperature",
                    {"entity_id": entity_id,
                     "temperature": from_internal_temperature_c(self.hass, effective_setpoint)},
                    blocking=True,
                )))

        if _sp_pending:
            _gather_results = await asyncio.gather(
                *[coro for _, _, coro in _sp_pending], return_exceptions=True)
            for (eid, sp, _), result in zip(_sp_pending, _gather_results):
                if isinstance(result, BaseException):
                    _LOGGER.debug(
                        "ThermoSmart '%s': Temperatur-Setpoint fehlgeschlagen: %s",
                        self.zone_name, result,
                    )
                    stats.record(result)
                else:
                    self._last_written_setpoints[eid] = sp  # success only
                    stats.record(None, effective_c=sp, entity_id=eid)

        if trv_setpoint > target:
            if self.zone_id not in self._boost_active:
                self._boost_active[self.zone_id] = {
                    "target": target,
                    "setpoint": trv_setpoint,
                    "started": self._now_local(),
                }
            else:
                self._boost_active[self.zone_id]["target"] = target
                self._boost_active[self.zone_id]["setpoint"] = trv_setpoint

        return stats

    # ── Direkte Ventilsteuerung (TPI) ────────────────────────────────

    async def _async_set_valve_percent(self, cfg: dict, duty_cycle: float) -> "_DispatchStats":
        """Schreibt TPI Duty-Cycle direkt als Ventilprozent (0–100%) auf unterstützte TRVs.

        Unterstützte Entities: valve_position, pi_heating_demand, heating_demand, level.
        valve_opening_degree is intentionally excluded (max-opening limit, not a live
        position — see AUTO_VALVE_PATTERNS in const.py).

        Valve-Bump-Workaround (TRVZB-Motor):
        Beim Schließen (neuer Wert < letzter Wert) kurz weiter öffnen und dann
        auf Zielwert setzen – verhindert steckende Motoren bei kleinen Schließbewegungen.

        Returns _DispatchStats with per-valve write counts.
        Individual valve writes use blocking=False (fire-and-forget at device level)
        but the service call itself is awaited so exceptions are captured in stats.
        """
        stats = _DispatchStats()
        valve_map = getattr(self, "_auto_valve_map", {})
        if not valve_map:
            return stats

        target_pct = max(0, min(100, round(duty_cycle)))

        for climate_id in cfg.get("climate_entities", []):
            valve_entity = valve_map.get(climate_id)
            if not valve_entity:
                continue

            state = self.hass.states.get(valve_entity)
            if state is None or state.state in ("unavailable", "unknown"):
                continue

            # Homematic IP (hahomematic) uses 0.0–1.0 scale instead of 0–100.
            # Detect by checking if current state value is <= 1.0 and entity contains "level".
            is_fractional = "level" in valve_entity.lower()

            try:
                raw = float(state.state)
                current_pct = round(raw * 100 if is_fractional else raw)
            except (TypeError, ValueError):
                current_pct = None

            _profile = self._device_profiles.get(climate_id)
            # Valve-Bump: beim Schließen kurz öffnen dann schließen
            # Verhindert TRVZB-Motor-Sticking bei kleinen Schließbewegungen
            if (
                (_profile is None or _profile.valve_bump_on_close)
                and current_pct is not None
                and target_pct < current_pct
                and (current_pct - target_pct) >= 5
            ):
                bump_pct = min(100, current_pct + TPI_VALVE_BUMP_PCT)
                bump_val = round(bump_pct / 100, 2) if is_fractional else bump_pct
                _LOGGER.debug(
                    "ThermoSmart '%s': Valve-Bump %s → %d%% → %d%%",
                    self.zone_name, valve_entity, bump_pct, target_pct,
                )
                try:
                    await self.hass.services.async_call(
                        "number", "set_value",
                        {"entity_id": valve_entity, "value": bump_val},
                        blocking=False,
                    )
                except Exception:
                    pass  # bump is a transient motor-unstick; not tracked in stats
                await asyncio.sleep(TPI_VALVE_BUMP_DELAY)

            if current_pct == target_pct:
                # Already at target — no write needed; counts as a "no-op success".
                # Not added to stats because no command was issued.
                continue

            write_val = round(target_pct / 100, 2) if is_fractional else target_pct
            _LOGGER.debug(
                "ThermoSmart '%s': Ventil %s → %d%%",
                self.zone_name, valve_entity, target_pct,
            )
            # Await the service call so exceptions are captured in stats.
            # blocking=False means HA does not wait for device acknowledgement (fire-and-forget
            # at device level), but the Python coroutine returns quickly after queuing the call.
            _valve_exc = None
            try:
                await self.hass.services.async_call(
                    "number", "set_value",
                    {"entity_id": valve_entity, "value": write_val},
                    blocking=False,
                )
            except Exception as err:
                _LOGGER.debug(
                    "ThermoSmart '%s': valve write failed for %s: %s",
                    self.zone_name, valve_entity, err,
                )
                _valve_exc = err
            stats.record(_valve_exc)

        return stats

    # ── External Temperature Input (TRVZB) ───────────────────────────

    async def _async_write_external_temp(self, cfg: dict, recommendation: dict) -> None:
        """Schreibt die Raumtemperatur direkt in die external_temperature_input-Entity des TRV.

        Beim Sonoff TRVZB (und ähnlichen TRVs) gibt es eine number-Entity
        'external_temperature_input'. Wird dort die echte Raumtemperatur eingetragen,
        nutzt die TRV-Firmware diesen Wert statt des internen (oft ungenauen) Sensors.
        Das macht die TRV-eigene Regelung deutlich präziser und vermeidet
        Konflikte zwischen TRV-interner Logik und ThermoSmart-Kalibrierung.

        Schreibt nur wenn Wert sich um mehr als 0.5°C geändert hat.
        Klemmt auf 0–99.9°C (TRVZB-Wertebereich).
        """
        room_temp = recommendation.get("current_temp")
        if room_temp is None:
            return

        ext_map = getattr(self, "_auto_ext_temp_map", {})
        if not ext_map:
            return

        for climate_id in cfg.get("climate_entities", []):
            _profile = self._device_profiles.get(climate_id)
            if _profile is not None and not _profile.allow_external_temp_input:
                continue
            ext_entity = ext_map.get(climate_id)
            if not ext_entity:
                continue

            state = self.hass.states.get(ext_entity)
            if state is None or state.state in ("unavailable", "unknown"):
                continue

            # This is a `number` entity — number.set_value performs no unit
            # conversion of its own (unlike climate.set_temperature), so the
            # value read/written here must use this entity's own
            # unit_of_measurement, not the HA system unit.
            ext_unit = state.attributes.get("unit_of_measurement")
            current_val = to_internal_temperature_c(self.hass, state.state, unit=ext_unit)

            if current_val is not None and abs(current_val - room_temp) < 0.5:
                continue

            clamped = round(max(0.0, min(99.9, room_temp)), 1)
            write_value = clamped
            if ext_unit in (UnitOfTemperature.FAHRENHEIT, UnitOfTemperature.KELVIN):
                write_value = round(
                    TemperatureConverter.convert(clamped, UnitOfTemperature.CELSIUS, ext_unit), 1)
            _LOGGER.debug(
                "ThermoSmart '%s': TRVZB external_temperature_input → %.1f°C (%s)",
                self.zone_name, clamped, ext_entity,
            )
            self.hass.async_create_task(
                self.hass.services.async_call(
                    "number", "set_value",
                    {"entity_id": ext_entity, "value": write_value},
                    blocking=False,
                )
            )

    # ── Temperature Source Management (external/internal select) ─────

    async def _async_manage_temp_source(self, cfg: dict, recommendation: dict) -> None:
        """Switch TRV temperature_sensor select between 'external' and 'internal'.

        Only operates when manage_temp_source is enabled for this zone.  Supported
        TRVs (e.g. SONOFF TRVZB) expose a select entity that controls whether the
        TRV firmware uses its built-in sensor ('internal') or an externally-provided
        value written to external_temperature_input ('external').

        Active control + external sensors available  →  set to 'external'
        Observation mode or summer mode              →  restore to 'internal' if owned
        All sensors unavailable > grace period       →  restore to 'internal' if owned
        Sensor recovery during active control        →  set back to 'external'

        Ownership: ThermoSmart only restores to 'internal' when it was the one that
        last set the select to 'external'.  Manual changes are detected and ownership
        is silently abandoned so the user's choice is never overwritten.
        """
        if not cfg.get(CONF_MANAGE_TEMP_SOURCE, False):
            return

        temp_source_map: dict[str, str] = getattr(self, "_auto_temp_source_map", {})
        if not temp_source_map:
            return

        temp_sensors = [s for s in cfg.get("temp_sensors", []) if s]
        if not temp_sensors:
            return

        ext_map: dict[str, str] = getattr(self, "_auto_ext_temp_map", {})
        now = self._now_local()

        # Zone-wide sensor availability check
        sensors_available = any(
            (st := self.hass.states.get(sid)) is not None
            and st.state not in ("unavailable", "unknown", "None")
            for sid in temp_sensors
        )

        # Effective active state: exclude summer mode (no heating, temp source irrelevant)
        is_summer = recommendation.get("is_summer", False)
        effectively_active = self._active_control and not is_summer

        for climate_id in cfg.get("climate_entities", []):
            select_entity = temp_source_map.get(climate_id)
            if not select_entity:
                continue

            _profile = self._device_profiles.get(climate_id)
            if _profile is not None and not _profile.allow_temp_source_select:
                _LOGGER.debug(
                    "ThermoSmart '%s': temp source select skipped for %s – profile '%s' has allow_temp_source_select=False",
                    self.zone_name, climate_id, _profile.identifier,
                )
                continue

            # ── Readiness guards ─────────────────────────────────────
            climate_state = self.hass.states.get(climate_id)
            if climate_state is None or climate_state.state in ("unavailable", "unknown"):
                _LOGGER.debug(
                    "ThermoSmart '%s': temp source guard – TRV %s unavailable",
                    self.zone_name, climate_id,
                )
                continue

            select_state = self.hass.states.get(select_entity)
            if select_state is None or select_state.state in ("unavailable", "unknown"):
                _LOGGER.debug(
                    "ThermoSmart '%s': temp source guard – select %s unavailable",
                    self.zone_name, select_entity,
                )
                continue

            options = select_state.attributes.get("options", [])
            if "external" not in options:
                _LOGGER.debug(
                    "ThermoSmart '%s': %s has no 'external' option – skipping",
                    self.zone_name, select_entity,
                )
                continue

            ext_entity = ext_map.get(climate_id)
            if ext_entity:
                ext_state = self.hass.states.get(ext_entity)
                if ext_state is None or ext_state.state in ("unavailable", "unknown"):
                    _LOGGER.debug(
                        "ThermoSmart '%s': temp source guard – ext_temp entity %s unavailable",
                        self.zone_name, ext_entity,
                    )
                    continue

            current_value = select_state.state
            owned = self._temp_source_owned.get(select_entity)

            # Detect manual override: user changed a select we thought we owned
            if owned and current_value != owned:
                _LOGGER.info(
                    "ThermoSmart '%s': %s changed to '%s' by user – abandoning ownership",
                    self.zone_name, select_entity, current_value,
                )
                self._temp_source_owned.pop(select_entity, None)
                owned = None

            # ── Observation / summer mode: restore if owned ──────────
            if not effectively_active:
                if owned == "external" and current_value == "external":
                    if "internal" in options:
                        await self._set_temp_source(select_entity, "internal", climate_id)
                    self._temp_source_owned.pop(select_entity, None)
                continue

            # ── Active control ───────────────────────────────────────
            if not sensors_available:
                if self._sensor_unavail_since is None:
                    self._sensor_unavail_since = now
                    _LOGGER.warning(
                        "ThermoSmart '%s': All temp sensors unavailable – "
                        "starting %ds grace period before switching %s to internal",
                        self.zone_name, TEMP_SOURCE_SENSOR_GRACE_SECONDS, select_entity,
                    )
                elapsed = (now - self._sensor_unavail_since).total_seconds()
                if elapsed >= TEMP_SOURCE_SENSOR_GRACE_SECONDS:
                    if owned == "external" and current_value == "external":
                        if "internal" in options:
                            _LOGGER.warning(
                                "ThermoSmart '%s': Grace period elapsed (%.0fs) – "
                                "switching %s to internal",
                                self.zone_name, elapsed, select_entity,
                            )
                            await self._set_temp_source(select_entity, "internal", climate_id)
                        self._temp_source_owned.pop(select_entity, None)
            else:
                # Sensors back: clear grace period timer, set to external if needed
                if self._sensor_unavail_since is not None:
                    _LOGGER.info(
                        "ThermoSmart '%s': Temp sensors available again",
                        self.zone_name,
                    )
                    self._sensor_unavail_since = None

                if current_value != "external":
                    await self._set_temp_source(select_entity, "external", climate_id)
                    self._temp_source_owned[select_entity] = "external"

    async def _set_temp_source(self, select_entity: str, option: str, climate_id: str) -> None:
        """Write a temperature_sensor select option (fire-and-forget)."""
        _LOGGER.info(
            "ThermoSmart '%s': temperature_sensor %s → '%s' (TRV: %s)",
            self.zone_name, select_entity, option, climate_id,
        )
        self.hass.async_create_task(
            self.hass.services.async_call(
                "select", "select_option",
                {"entity_id": select_entity, "option": option},
                blocking=False,
            )
        )

    async def _async_restore_temp_source(self) -> None:
        """Restore all owned temperature_sensor selects to 'internal' on cleanup.

        Called at entry unload / zone remove.  Best-effort: logs and skips any
        entity that is unavailable or was already changed by the user.
        """
        temp_source_map: dict[str, str] = getattr(self, "_auto_temp_source_map", {})
        if not temp_source_map:
            return

        for climate_id, select_entity in temp_source_map.items():
            if self._temp_source_owned.get(select_entity) != "external":
                continue

            select_state = self.hass.states.get(select_entity)
            if select_state is None or select_state.state in ("unavailable", "unknown"):
                _LOGGER.info(
                    "ThermoSmart '%s': Cannot restore %s – entity unavailable, skipping",
                    self.zone_name, select_entity,
                )
                continue

            if select_state.state != "external":
                _LOGGER.info(
                    "ThermoSmart '%s': %s already at '%s' – skipping restore",
                    self.zone_name, select_entity, select_state.state,
                )
                continue

            options = select_state.attributes.get("options", [])
            if "internal" not in options:
                _LOGGER.debug(
                    "ThermoSmart '%s': %s has no 'internal' option – cannot restore",
                    self.zone_name, select_entity,
                )
                continue

            _LOGGER.info(
                "ThermoSmart '%s': Restoring %s → internal (cleanup/unload)",
                self.zone_name, select_entity,
            )
            try:
                await self.hass.services.async_call(
                    "select", "select_option",
                    {"entity_id": select_entity, "option": "internal"},
                    blocking=True,
                )
            except Exception as err:
                _LOGGER.warning(
                    "ThermoSmart '%s': Failed to restore %s to internal: %s",
                    self.zone_name, select_entity, err,
                )

        self._temp_source_owned.clear()
