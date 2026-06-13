"""ThermoSmart device profile definitions.

All known device profiles are defined here as frozen DeviceProfile instances.
Adding support for a new TRV model means adding an entry here and a
corresponding match rule in registry.py — no other files need to change.
"""
from __future__ import annotations

from .capabilities import (
    DeviceProfile,
    SETPOINT_CLIMATE,
    SETPOINT_HVAC_FIRST,
    VALVE_READ_ONLY,
    VALVE_MAX_LIMIT,
)

# ── Generic fallback ─────────────────────────────────────────────────────────

GENERIC = DeviceProfile(
    identifier="generic",
    display_name="Generic TRV",
)

# ── Named profiles ───────────────────────────────────────────────────────────

SONOFF_TRVZB = DeviceProfile(
    identifier="sonoff_trvzb",
    display_name="SONOFF TRVZB",
    valve_semantics=VALVE_MAX_LIMIT,
    has_temp_source_select=True,
    quirk_entities=("external_temperature_input",),
)

TUYA_TS0601 = DeviceProfile(
    identifier="tuya_ts0601",
    display_name="Tuya TS0601",
)

EVE_THERMO = DeviceProfile(
    identifier="eve_thermo",
    display_name="Eve Thermo (SEA80x)",
    valve_semantics=VALVE_READ_ONLY,
    # Valve position is exposed via HomeKit as a read-only sensor.
    # ThermoSmart reads it for learning; direct writes are not possible.
)

AQARA_TRV = DeviceProfile(
    identifier="aqara_trv",
    display_name="Aqara TRV",
)

EUROTRONIC_SPIRIT = DeviceProfile(
    identifier="eurotronic_spirit",
    display_name="Eurotronic Spirit",
    setpoint_method=SETPOINT_HVAC_FIRST,
    hvac_mode_before_write="heat",
)

DANFOSS_ALLY = DeviceProfile(
    identifier="danfoss_ally",
    display_name="Danfoss Ally",
)

BOSCH_BTH = DeviceProfile(
    identifier="bosch_bth",
    display_name="Bosch BTH-RA",
)

TADO = DeviceProfile(
    identifier="tado",
    display_name="tado°",
    is_active=False,
    warning=(
        "tado° devices rely on the tado° cloud API and cannot be controlled "
        "directly via a local climate entity. ThermoSmart can observe tado° "
        "entities in read-only mode but will not write setpoints. Use the "
        "official tado° integration for scheduling and control."
    ),
)

# ── Registry of all profiles ─────────────────────────────────────────────────

ALL_PROFILES: tuple[DeviceProfile, ...] = (
    GENERIC,
    SONOFF_TRVZB,
    TUYA_TS0601,
    EVE_THERMO,
    AQARA_TRV,
    EUROTRONIC_SPIRIT,
    DANFOSS_ALLY,
    BOSCH_BTH,
    TADO,
)
