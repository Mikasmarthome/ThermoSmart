"""Valve maintenance – ThermoSmart MaintenanceMixin."""
from __future__ import annotations

import asyncio
import contextlib
import logging

from .const import (
    CONF_VALVE_MAINTENANCE,
    VALVE_MAINTENANCE_HOUR,
    VALVE_MAINTENANCE_WEEKDAY,
    VALVE_MAINTENANCE_BOOST_TEMP,
    VALVE_MAINTENANCE_DURATION_SEC,
    VALVE_MAINTENANCE_DURATION_SUMMER_SEC,
    HEATING_MODE_VACATION,
)
from .temperature_units import from_internal_temperature_c

_LOGGER = logging.getLogger(__name__)


class MaintenanceMixin:
    """Wöchentliche Ventil-Übung gegen Festklemmen durch Kalk oder Gummi."""

    async def _async_valve_maintenance(self, cfg: dict, recommendation: dict) -> None:
        """Ventil vollständig auf und zu fahren wenn es lange stillstand.

        Wird NUR ausgeführt wenn Ventile wahrscheinlich längere Zeit stillstanden:
          - Sommer-Modus: Heizung komplett aus, Ventile wochenlang nicht bewegt
          - Urlaubsmodus: Zieltemp ~12°C, Ventile kaum bewegt
        Beobachtungsmodus: externer Regler bewegt Ventile – keine Wartung nötig.
        Im Winter mit aktiver Steuerung bewegen sich Ventile ohnehin regelmäßig.
        """
        if not cfg.get(CONF_VALVE_MAINTENANCE, True):
            return
        if self._maintenance_running:
            return

        in_vacation = recommendation.get("mode") == HEATING_MODE_VACATION
        valve_likely_idle = self._is_summer or in_vacation
        if not valve_likely_idle:
            return

        now = self._now_local()
        if now.weekday() != VALVE_MAINTENANCE_WEEKDAY or now.hour != VALVE_MAINTENANCE_HOUR:
            return

        # Nur einmal pro Woche auslösen
        if self._last_maintenance is not None:
            if (now - self._last_maintenance).days < 6:
                return

        self._last_maintenance = now
        self._maintenance_running = True
        duration = VALVE_MAINTENANCE_DURATION_SUMMER_SEC if self._is_summer else VALVE_MAINTENANCE_DURATION_SEC

        # Rückkehrtemperatur nach Wartung: aktuelle Zieltemp bevorzugt,
        # Fallback abhängig vom Modus (nicht hardcoded comfort_temp im Urlaub).
        low_temp_mode = self._is_summer or in_vacation
        target_after = recommendation.get("adjusted_target") or cfg.get(
            "vacation_temp" if low_temp_mode else "comfort_temp",
            12.0 if low_temp_mode else 21.0,
        )
        climate_entities = cfg.get("climate_entities", [])

        _LOGGER.info(
            "ThermoSmart '%s': Ventil-Wartung gestartet – fahre auf %.0f°C",
            self.zone_name, VALVE_MAINTENANCE_BOOST_TEMP,
        )

        async def _run_maintenance() -> None:
            try:
                _boost_dispatch = from_internal_temperature_c(self.hass, VALVE_MAINTENANCE_BOOST_TEMP)
                tasks = [
                    self.hass.services.async_call(
                        "climate", "set_temperature",
                        {"entity_id": eid, "temperature": _boost_dispatch},
                        blocking=True,
                    )
                    for eid in climate_entities
                    if self.hass.states.get(eid)
                    and self.hass.states.get(eid).state not in ("unavailable", "unknown")
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(duration)

                _return_dispatch = from_internal_temperature_c(self.hass, target_after)
                tasks = [
                    self.hass.services.async_call(
                        "climate", "set_temperature",
                        {"entity_id": eid, "temperature": _return_dispatch},
                        blocking=True,
                    )
                    for eid in climate_entities
                    if self.hass.states.get(eid)
                    and self.hass.states.get(eid).state not in ("unavailable", "unknown")
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                _LOGGER.info(
                    "ThermoSmart '%s': Ventil-Wartung abgeschlossen → zurück auf %.1f°C",
                    self.zone_name, target_after,
                )
            finally:
                self._maintenance_running = False

        self._maintenance_task = self.hass.async_create_task(_run_maintenance())

    async def async_cancel_maintenance(self) -> None:
        """Cancel any in-flight valve-maintenance task and await its teardown.

        Must be called on unload/reload so a maintenance cycle (which can run
        for minutes, holding the boost setpoint) never outlives its config
        entry and keeps calling services against a torn-down zone. Safe to
        call when no maintenance task exists or it already finished.
        """
        task = self._maintenance_task
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._maintenance_running = False
