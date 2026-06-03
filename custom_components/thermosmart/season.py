"""Summer mode detection and frost protection – ThermoSmart SeasonMixin."""
from __future__ import annotations

import asyncio
import logging

from .const import SUMMER_THRESHOLD, WINTER_THRESHOLD, TEMP_FROST_PROTECTION

_LOGGER = logging.getLogger(__name__)


class SeasonMixin:
    """Sommer-Erkennung (72h-Rollmittel) und Frostschutz im Sommer."""

    def _update_summer_mode(self, weather_data: dict) -> None:
        """Sommer-Erkennung: automatisch (72h-Ø) oder via globalem Sommer-Schalter."""
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
        """Im Sommer: TRVs auf Frostschutztemperatur (12°C) setzen."""
        tasks = []
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
        if tasks:
            await asyncio.gather(*tasks)
