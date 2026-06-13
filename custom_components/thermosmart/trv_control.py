"""TRV control, calibration and quirk management – ThermoSmart TRVControlMixin."""
from __future__ import annotations

import asyncio
import logging

from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

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
    TEMP_SOURCE_SENSOR_GRACE_SECONDS,
    WINDOW_OPEN_SETPOINT,
    TPI_VALVE_BUMP_PCT,
    TPI_VALVE_BUMP_DELAY,
)

_LOGGER = logging.getLogger(__name__)


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

            trv_temp: float | None = None
            trv_state = self._get_trv_state(climate_id)
            if trv_state:
                try:
                    trv_temp = float(trv_state.attributes.get("current_temperature", 0))
                except (TypeError, ValueError):
                    pass

            if trv_temp is None:
                continue

            if trv_temp - room_temp > 3.0:
                _LOGGER.debug(
                    "ThermoSmart '%s': Kalibrierung %s übersprungen – TRV %.1f°C > Raum %.1f°C",
                    self.zone_name, cal_entity, trv_temp, room_temp,
                )
                continue

            raw_offset = room_temp - trv_temp

            # Kalibrierungs-Inversion (z.B. ME167 invertiert das Offset-Vorzeichen)
            if cfg.get(CONF_CALIBRATION_INVERT, False):
                raw_offset = -raw_offset

            if abs(raw_offset) > 7.0:
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

    # ── Temperatur schreiben ──────────────────────────────────────────

    async def _apply_temperature(self, cfg: dict, recommendation: dict) -> None:
        """TRV-Setpoints schreiben oder 5°C bei geöffnetem Fenster setzen."""
        target = recommendation.get("adjusted_target")

        if target is None:
            if recommendation.get("window_open"):
                frost_temp = WINDOW_OPEN_SETPOINT
                tasks = []
                for entity_id in cfg.get("climate_entities", []):
                    state = self._get_trv_state(entity_id)
                    if state is None:
                        continue
                    try:
                        if abs(float(state.attributes.get("temperature", 0)) - frost_temp) < 0.3:
                            continue
                    except (TypeError, ValueError):
                        pass
                    self._last_written_setpoints[entity_id] = frost_temp
                    tasks.append(self.hass.services.async_call(
                        "climate", "set_temperature",
                        {"entity_id": entity_id, "temperature": frost_temp},
                        blocking=True,
                    ))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            return

        if not (5.0 <= target <= 30.0):
            _LOGGER.warning(
                "ThermoSmart '%s': Ziel %.1f°C außerhalb Sicherheitsbereich",
                self.zone_name, target,
            )
            return

        trv_setpoint = recommendation.get("trv_setpoint", target)
        tolerance = cfg.get("temp_tolerance", 0.5)

        tasks = []
        for entity_id in cfg.get("climate_entities", []):
            state = self._get_trv_state(entity_id)
            if state is None:
                continue
            current_setpoint = state.attributes.get("temperature")
            if current_setpoint is not None:
                try:
                    if abs(float(current_setpoint) - trv_setpoint) < tolerance:
                        continue
                except (TypeError, ValueError):
                    pass
            _LOGGER.debug(
                "ThermoSmart '%s' → %s: %.1f°C (Ziel=%.1f°C, Boost+%.1f°C)",
                self.zone_name, entity_id, trv_setpoint, target, trv_setpoint - target,
            )
            self._last_written_setpoints[entity_id] = trv_setpoint
            tasks.append(self.hass.services.async_call(
                "climate", "set_temperature",
                {"entity_id": entity_id, "temperature": trv_setpoint},
                blocking=True,
            ))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    _LOGGER.debug(
                        "ThermoSmart '%s': Temperatur-Setpoint fehlgeschlagen: %s",
                        self.zone_name, result,
                    )

        if trv_setpoint > target:
            if self.zone_id not in self._boost_active:
                self._boost_active[self.zone_id] = {
                    "target": target,
                    "setpoint": trv_setpoint,
                    "started": dt_util.now(),
                }
            else:
                self._boost_active[self.zone_id]["target"] = target
                self._boost_active[self.zone_id]["setpoint"] = trv_setpoint

    # ── Direkte Ventilsteuerung (TPI) ────────────────────────────────

    async def _async_set_valve_percent(self, cfg: dict, duty_cycle: float) -> bool:
        """Schreibt TPI Duty-Cycle direkt als Ventilprozent (0–100%) auf unterstützte TRVs.

        Unterstützte Entities: valve_position, pi_heating_demand, heating_demand, level.
        valve_opening_degree is intentionally excluded (max-opening limit, not a live
        position — see AUTO_VALVE_PATTERNS in const.py).

        Valve-Bump-Workaround (TRVZB-Motor):
        Beim Schließen (neuer Wert < letzter Wert) kurz weiter öffnen und dann
        auf Zielwert setzen – verhindert steckende Motoren bei kleinen Schließbewegungen.

        Returns True wenn mindestens ein Ventil geschrieben wurde.
        """
        valve_map = getattr(self, "_auto_valve_map", {})
        if not valve_map:
            return False

        target_pct = max(0, min(100, round(duty_cycle)))
        wrote_any = False

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

            # Valve-Bump: beim Schließen kurz öffnen dann schließen
            # Verhindert TRVZB-Motor-Sticking bei kleinen Schließbewegungen
            if (
                current_pct is not None
                and target_pct < current_pct
                and (current_pct - target_pct) >= 5
            ):
                bump_pct = min(100, current_pct + TPI_VALVE_BUMP_PCT)
                bump_val = round(bump_pct / 100, 2) if is_fractional else bump_pct
                _LOGGER.debug(
                    "ThermoSmart '%s': Valve-Bump %s → %d%% → %d%%",
                    self.zone_name, valve_entity, bump_pct, target_pct,
                )
                await self.hass.services.async_call(
                    "number", "set_value",
                    {"entity_id": valve_entity, "value": bump_val},
                    blocking=False,
                )
                await asyncio.sleep(TPI_VALVE_BUMP_DELAY)

            if current_pct == target_pct:
                wrote_any = True
                continue

            write_val = round(target_pct / 100, 2) if is_fractional else target_pct
            _LOGGER.debug(
                "ThermoSmart '%s': Ventil %s → %d%%",
                self.zone_name, valve_entity, target_pct,
            )
            self.hass.async_create_task(
                self.hass.services.async_call(
                    "number", "set_value",
                    {"entity_id": valve_entity, "value": write_val},
                    blocking=False,
                )
            )
            wrote_any = True

        return wrote_any

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
            ext_entity = ext_map.get(climate_id)
            if not ext_entity:
                continue

            state = self.hass.states.get(ext_entity)
            if state is None or state.state in ("unavailable", "unknown"):
                continue

            try:
                current_val = float(state.state)
            except (TypeError, ValueError):
                current_val = None

            if current_val is not None and abs(current_val - room_temp) < 0.5:
                continue

            clamped = round(max(0.0, min(99.9, room_temp)), 1)
            _LOGGER.debug(
                "ThermoSmart '%s': TRVZB external_temperature_input → %.1f°C (%s)",
                self.zone_name, clamped, ext_entity,
            )
            self.hass.async_create_task(
                self.hass.services.async_call(
                    "number", "set_value",
                    {"entity_id": ext_entity, "value": clamped},
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
        now = dt_util.now()

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
