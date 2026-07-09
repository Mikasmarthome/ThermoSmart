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

        Active Control ist eine harte Grenze für jeden echten Dispatch — auch
        für diese Wartungsroutine. Ist Active Control aus, wird kein
        climate.set_temperature Service Call ausgelöst, unabhängig von
        Sommer-/Urlaubsmodus.
        """
        if not cfg.get(CONF_VALVE_MAINTENANCE, True):
            return
        if self._maintenance_running:
            return
        if not self._active_control:
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

        def _dispatch_payloads(setpoint_c: float) -> list[dict]:
            """Resolve one setpoint through the SAME per-device clamp/step/
            profile-active logic as regular TRV dispatch (Device Compatibility
            P1) — min_temp/max_temp/target_temp_step and a de-activated device
            profile are respected exactly like ``_apply_temperature()``'s own
            dispatch, instead of writing the raw constant straight to
            ``climate.set_temperature``. Unit conversion happens exactly once,
            on the already-clamped/snapped value, matching the regular
            dispatch path — no double conversion.
            """
            payloads = []
            for eid in climate_entities:
                state = self.hass.states.get(eid)
                if state is None or state.state in ("unavailable", "unknown"):
                    continue
                profile = self._device_profiles.get(eid)
                if profile is not None and not profile.is_active:
                    continue
                effective = self.resolve_device_effective_setpoint(
                    eid, state, setpoint_c, profile)
                payloads.append({
                    "entity_id": eid,
                    "temperature": from_internal_temperature_c(self.hass, effective),
                })
            return payloads

        async def _run_maintenance() -> None:
            try:
                tasks = [
                    self.hass.services.async_call(
                        "climate", "set_temperature", payload, blocking=True,
                    )
                    for payload in _dispatch_payloads(VALVE_MAINTENANCE_BOOST_TEMP)
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(duration)

                tasks = [
                    self.hass.services.async_call(
                        "climate", "set_temperature", payload, blocking=True,
                    )
                    for payload in _dispatch_payloads(target_after)
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
