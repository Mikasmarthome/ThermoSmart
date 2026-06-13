"""ThermoSmart device profile registry.

Maps HA device registry strings (model, manufacturer) to a DeviceProfile.
Matching is case-insensitive substring search — tolerant of the varied
strings that different Zigbee coordinators and cloud bridges report.

Always returns a DeviceProfile (falls back to GENERIC); callers never need
a None check.
"""
from __future__ import annotations

import logging

from .capabilities import DeviceProfile
from .profiles import ALL_PROFILES, GENERIC

_LOGGER = logging.getLogger(__name__)

# Each entry: (profile_identifier, model_substrings, manufacturer_substrings)
# A match on ANY model substring OR ANY manufacturer substring is sufficient.
# More specific entries should appear before more general ones.
_MATCH_TABLE: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("sonoff_trvzb",      ("TRVZB",),                    ("SONOFF",)),
    ("tuya_ts0601",       ("TS0601",),                   ("_TZE", "Tuya")),
    ("eve_thermo",        ("Eve Thermo", "SEA80"),        ("Eve Systems", "Eve")),
    ("aqara_trv",         ("SRTS-A01",),                 ("Aqara", "LUMI")),
    ("eurotronic_spirit", ("Spirit",),                   ("Eurotronic",)),
    ("danfoss_ally",      ("Ally",),                     ("Danfoss",)),
    ("bosch_bth",         ("BTH-RA",),                   ("Bosch",)),
    ("tado",              ("tado",),                     ("tado",)),
)

_BY_ID: dict[str, DeviceProfile] = {p.identifier: p for p in ALL_PROFILES}


def get_profile(model: str | None, manufacturer: str | None) -> DeviceProfile:
    """Return the best-matching DeviceProfile for the given device strings.

    Args:
        model:        Value of device_entry.model from the HA device registry.
        manufacturer: Value of device_entry.manufacturer.

    Returns:
        A DeviceProfile — guaranteed non-None.  Falls back to GENERIC when no
        specific match is found.
    """
    model_lc = (model or "").lower()
    mfr_lc = (manufacturer or "").lower()

    for identifier, model_patterns, mfr_patterns in _MATCH_TABLE:
        if any(p.lower() in model_lc for p in model_patterns) or \
           any(p.lower() in mfr_lc for p in mfr_patterns):
            profile = _BY_ID.get(identifier, GENERIC)
            _LOGGER.debug(
                "ThermoSmart: device profile '%s' matched (model=%r, manufacturer=%r)",
                profile.identifier, model, manufacturer,
            )
            return profile

    _LOGGER.debug(
        "ThermoSmart: no device profile match for model=%r manufacturer=%r — using generic",
        model, manufacturer,
    )
    return GENERIC
