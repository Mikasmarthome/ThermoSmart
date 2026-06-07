"""Summer mode detection and frost protection – ThermoSmart SeasonMixin."""
from __future__ import annotations

import asyncio
import logging

from .const import SUMMER_THRESHOLD, WINTER_THRESHOLD, TEMP_FROST_PROTECTION

_LOGGER = logging.getLogger(__name__)


class SeasonMixin:
    """Sommer-Erkennung (rollierender Puffer bis zu 72h) und Frostschutz im Sommer."""

    def _update_summer_mode(self, weather_data: dict) -> None:
        """Sommer-Erkennung: rollierender Temperatur-Puffer oder globaler Sommer-Schalter."""
        # Globaler Override aktiv → bereits in set_summer_override() gesetzt
        if self._summer_override is not None:
            return

        prev_summer = self._is_summer

        outdoor = weather_data.get("temperature")
        if outdoor is None:
            return
        self._outdoor_temp_history.append(outdoor)

        if len(self._outdoor_temp_history) < 3:
            return

        avg = sum(self._outdoor_temp_history) / len(self._outdoor_temp_history)

        if avg >= SUMMER_THRESHOLD:
            self._is_summer = True
        elif avg <= WINTER_THRESHOLD:
            self._is_summer = False

        if self._is_summer != prev_summer:
            _LOGGER.warning(
                "ThermoSmart '%s': Saisonwechsel – %s (72h-Ø %.1f°C)",
                self.zone_name,
                "SOMMER – Heizung deaktiviert" if self._is_summer else "WINTER – Heizung aktiv",
                avg,
            )

    async def _apply_frost_protection(self, cfg: dict) -> None:
        """Im Sommer: TRVs auf Frostschutztemperatur (12°C) setzen.

        Fehler eines einzelnen TRVs (z. B. ZigBee-Dropout nach dem
        Verfügbarkeits-Check) brechen den Frostschutz der übrigen TRVs
        nicht ab. Jede Exception wird mit der zugehörigen Entity-ID
        geloggt; der Coordinator bleibt weiterhin funktionsfähig.
        """
        tasks: list = []
        task_ids: list[str] = []

        for entity_id in cfg.get("climate_entities", []):
            state = self.hass.states.get(entity_id)
            if not state or state.state in ("unavailable", "unknown"):
                continue
            try:
                if abs(float(state.attributes.get("temperature", 0)) - TEMP_FROST_PROTECTION) < 0.5:
                    continue
            except (TypeError, ValueError):
                pass
            tasks.append(self.hass.services.async_call(
                "climate", "set_temperature",
                {"entity_id": entity_id, "temperature": TEMP_FROST_PROTECTION},
                blocking=True,
            ))
            task_ids.append(entity_id)

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for entity_id, result in zip(task_ids, results):
            if isinstance(result, BaseException):
                _LOGGER.warning(
                    "ThermoSmart '%s': Frostschutz für %s fehlgeschlagen: %s",
                    self.zone_name, entity_id, result,
                )
