"""ThermoSmart device profile system.

Public API
----------
get_profile(model, manufacturer) -> DeviceProfile
    Look up the profile for a TRV entity.  Always returns a DeviceProfile;
    falls back to the generic profile when no specific match is found.

DeviceProfile
    Frozen dataclass describing a TRV family's capabilities and quirks.
    Re-exported here so callers only need one import.

device_profile_status(profile) -> dict
    Pure presentation helper — a small, public-safe user/support-facing
    summary (display name, active/observe-only, warning) of one profile.
    Never matches/enforces/mutates anything.

Capability constants (SETPOINT_*, VALVE_*)
    Re-exported for use in type checks or custom profile construction.
"""
from .capabilities import (
    DeviceProfile,
    SETPOINT_CLIMATE,
    SETPOINT_HVAC_FIRST,
    VALVE_READ_ONLY,
    VALVE_MAX_LIMIT,
    VALVE_DIRECT,
    device_profile_status,
)
from .registry import get_profile

__all__ = [
    "DeviceProfile",
    "get_profile",
    "device_profile_status",
    "SETPOINT_CLIMATE",
    "SETPOINT_HVAC_FIRST",
    "VALVE_READ_ONLY",
    "VALVE_MAX_LIMIT",
    "VALVE_DIRECT",
]
