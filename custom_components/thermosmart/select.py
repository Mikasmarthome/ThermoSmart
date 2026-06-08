"""Select platform für ThermoSmart."""
from __future__ import annotations
import logging

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN, VERSION,
    HEATING_MODE_AUTO, HEATING_MODE_COMFORT, HEATING_MODE_NIGHT,
    HEATING_MODE_AWAY, HEATING_MODE_VACATION, HEATING_MODE_ECO,
    HEATING_MODES,
    DOMAIN_GLOBAL_SUMMER,
)
from .coordinator import ThermoSmartCoordinator

_LOGGER = logging.getLogger(__name__)

# Optionen für den globalen Sommer-Select (machine-readable lowercase keys)
SUMMER_OPT_AUTOMATIC = "automatic"   # 72h-Durchschnitt entscheidet
SUMMER_OPT_ON        = "on"          # Sommer erzwungen
SUMMER_OPT_OFF       = "off"         # Winter erzwungen

_SUMMER_OPTS = [SUMMER_OPT_AUTOMATIC, SUMMER_OPT_ON, SUMMER_OPT_OFF]

# Override-Wert je Option
_SUMMER_OVERRIDE_MAP: dict[str, bool | None] = {
    SUMMER_OPT_AUTOMATIC: None,
    SUMMER_OPT_ON:        True,
    SUMMER_OPT_OFF:       False,
}

# Backward-Compat: alte Session-Werte (capitalized, vor beta.24) auf neue Keys mappen
_LEGACY_SUMMER_MAP: dict[str, str] = {
    "Automatic": SUMMER_OPT_AUTOMATIC,
    "On":        SUMMER_OPT_ON,
    "Off":       SUMMER_OPT_OFF,
}


def _all_coordinators(hass: HomeAssistant):
    """Alle Zone-Coordinatoren liefern (globale Keys und System-Entries überspringen)."""
    skip = {"learning_engine", "global_switches_created", "global_summer_select_created"}
    for key, data in hass.data.get(DOMAIN, {}).items():
        if key not in skip and isinstance(data, dict) and data.get("type") != "system":
            if coord := data.get("coordinator"):
                yield coord


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    cfg = {**entry.data, **entry.options}

    # System-Entry: globaler Sommer-Select
    if cfg.get("entry_type") == "system":
        async_add_entities([ThermoSmartGlobalSummerSelect(hass, entry)])
        return

    # Zone-Entry: Modus-Select
    coordinator: ThermoSmartCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    entities: list = [ThermoSmartModeSelect(coordinator, entry)]

    # Backward-Compat: globaler Sommer-Select an erste Zone anhängen wenn kein System-Entry
    has_system = any(
        e.data.get("entry_type") == "system"
        for e in hass.config_entries.async_entries(DOMAIN)
    )
    if not has_system and "global_summer_select_created" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["global_summer_select_created"] = True
        entities.append(ThermoSmartGlobalSummerSelect(hass, entry))

    async_add_entities(entities)


class ThermoSmartModeSelect(SelectEntity, RestoreEntity):
    _attr_has_entity_name = True
    _attr_translation_key = "mode"
    _attr_icon = "mdi:home-thermometer-outline"
    _attr_options = HEATING_MODES  # ["auto", "comfort", "eco", "night", "away", "vacation"]

    def __init__(self, coordinator: ThermoSmartCoordinator, entry: ConfigEntry):
        self._coordinator = coordinator
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_mode"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"ThermoSmart – {entry.data.get('name', 'Zone')}",
            manufacturer="ThermoSmart",
            model="Self-learning Heating Controller",
            sw_version=VERSION,
            entry_type="service",  # type: ignore[arg-type]
        )

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last and last.state in HEATING_MODES:
            # Nur zum Coordinator syncen wenn der noch im Default-Modus ist –
            # Globaler Urlaubsschalter (läuft vor select) hat sonst Vorrang.
            if self._coordinator._mode == HEATING_MODE_AUTO:
                self._coordinator.set_mode(last.state)

        # Als Coordinator-Listener registrieren → Entity aktualisiert sich
        # wenn das Climate-Entity oder ein anderer Pfad den Modus ändert.
        self._coordinator.async_add_listener(self._handle_coordinator_update)

    async def async_will_remove_from_hass(self) -> None:
        self._coordinator.async_remove_listener(self._handle_coordinator_update)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Coordinator hat Daten aktualisiert → State neu schreiben."""
        self.async_write_ha_state()

    @property
    def current_option(self) -> str:
        """Liest immer direkt vom Coordinator – bleibt in sync mit Climate-Entity."""
        return self._coordinator._mode

    async def async_select_option(self, option: str) -> None:
        if option not in HEATING_MODES:
            return
        self._coordinator.set_mode(option)
        self.async_write_ha_state()
        await self._coordinator.async_request_refresh()


# ── Globaler Sommer-Select (domain-weit, steuert alle Zonen) ─────────────────

class ThermoSmartGlobalSummerSelect(SelectEntity, RestoreEntity):
    """Dreistufiger Sommer-Override: automatic / on / off.

    automatic → 72h-Außentemperaturdurchschnitt entscheidet (≥18°C → Sommer)
    on        → Sommermodus dauerhaft erzwungen (Heizung deaktiviert, Frostschutz aktiv)
    off       → Wintermodus dauerhaft erzwungen (normale Heizregelung)

    current_option zeigt den Override-Modus (nicht den aktuell erkannten Wert).
    Das Attribut effective_summer spiegelt den echten _is_summer-Zustand jeder Zone.
    """
    _attr_has_entity_name = True
    _attr_translation_key = "summer_mode"
    _attr_icon = "mdi:weather-sunny"
    _attr_unique_id = DOMAIN_GLOBAL_SUMMER
    _attr_options = _SUMMER_OPTS

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, "global")},
            name="ThermoSmart System",
            manufacturer="ThermoSmart",
            model="Global Control",
            sw_version=VERSION,
        )
        self._current_option: str = SUMMER_OPT_AUTOMATIC
        self._unsub_listeners: list = []

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        # Zustand aus letzter HA-Session wiederherstellen
        last = await self.async_get_last_state()
        if last and last.state:
            # Migrate legacy capitalized values (pre-beta.24: "Automatic"/"On"/"Off")
            restored = _LEGACY_SUMMER_MAP.get(last.state, last.state)
            self._current_option = restored if restored in _SUMMER_OPTS else SUMMER_OPT_AUTOMATIC
        else:
            self._current_option = SUMMER_OPT_AUTOMATIC

        # Store for coordinators that load after this entity (race condition fix)
        self._hass.data[DOMAIN]["global_summer_override"] = self._current_option

        # Override sofort auf alle bereits geladenen Coordinatoren anwenden
        override_val = _SUMMER_OVERRIDE_MAP[self._current_option]
        for coord in _all_coordinators(self._hass):
            coord.set_summer_override(override_val)
            await coord.async_request_refresh()

        # Als Listener auf alle Coordinatoren registrieren
        # → async_write_ha_state() aktualisiert extra_state_attributes (effective_summer)
        for coord in _all_coordinators(self._hass):
            unsub = coord.async_add_listener(self._handle_coordinator_update)
            self._unsub_listeners.append(unsub)

        _LOGGER.debug(
            "ThermoSmart GlobalSummerSelect: wiederhergestellt als '%s'",
            self._current_option,
        )

    async def async_will_remove_from_hass(self) -> None:
        for unsub in self._unsub_listeners:
            unsub()
        self._unsub_listeners.clear()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Coordinator-Update → extra_state_attributes (effective_summer) aktualisieren."""
        self.async_write_ha_state()

    @property
    def current_option(self) -> str:
        """Zeigt den aktiven Override-Modus, nicht den auto-erkannten Zustand."""
        return self._current_option

    @property
    def extra_state_attributes(self) -> dict:
        coords = list(_all_coordinators(self._hass))
        is_summer = any(c._is_summer for c in coords)
        return {
            "effective_summer": is_summer,
            "override": self._current_option,
            "active_zones": len(coords),
        }

    async def async_select_option(self, option: str) -> None:
        if option not in _SUMMER_OPTS:
            return
        self._current_option = option
        self._hass.data[DOMAIN]["global_summer_override"] = option
        override_val = _SUMMER_OVERRIDE_MAP[option]
        for coord in _all_coordinators(self._hass):
            coord.set_summer_override(override_val)
            await coord.async_request_refresh()
        self.async_write_ha_state()
        _LOGGER.info(
            "ThermoSmart GlobalSummerSelect: '%s' gesetzt (override=%s)",
            option, override_val,
        )
