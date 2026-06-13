"""ThermoSmart device capability definitions.

Defines the DeviceProfile dataclass and the string constants used for its
fields.  Profiles are immutable value objects — frozen dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Setpoint write method ────────────────────────────────────────────────────

SETPOINT_CLIMATE = "climate_set_temperature"
# Standard: climate.set_temperature.  Works for the vast majority of TRVs.

SETPOINT_HVAC_FIRST = "hvac_then_setpoint"
# Some TRVs require an explicit HVAC mode write (e.g. "heat") before
# accepting a new setpoint.  Use when direct setpoint writes are silently
# ignored unless the device is already in heat mode.

# ── Valve position semantics ─────────────────────────────────────────────────

VALVE_READ_ONLY = "read_only"
# valve_opening_degree is a sensor — ThermoSmart reads it for learning but
# never writes to it.

VALVE_MAX_LIMIT = "max_limit"
# valve_opening_degree configures a maximum-opening limit, not live position.
# Writing to it would restrict the TRV's range — ThermoSmart must not write it.

VALVE_DIRECT = "direct_control"
# valve_opening_degree is directly steerable (position target).
# Reserved for future use; no current profile uses this.


@dataclass(frozen=True)
class DeviceProfile:
    """Describes the capabilities and quirks of a specific TRV model family.

    All fields have safe defaults that match a well-behaved generic TRV.
    Profile instances are immutable; create new ones rather than mutating.
    """

    identifier: str
    display_name: str

    setpoint_method: str = SETPOINT_CLIMATE
    valve_semantics: str = VALVE_READ_ONLY
    calibration_inverted: bool = False
    hvac_mode_before_write: str | None = None
    burst_guard_seconds: float = 5.0
    setpoint_delay_seconds: float = 0.0
    valve_control_uncertain: bool = False
    quirk_entities: tuple[str, ...] = ()
    warning: str | None = None
    is_active: bool = True
    hvac_watchdog: bool = True
    valve_bump_on_close: bool = True
    allow_external_temp_input: bool = True
    allow_temp_source_select: bool = True
    calibration_sync: bool = True
    calibration_plausibility_limit: float = 7.0
