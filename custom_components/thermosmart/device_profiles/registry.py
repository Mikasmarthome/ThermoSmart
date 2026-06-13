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
# A device matches if ANY model substring OR ANY manufacturer substring is
# found (case-insensitive) in the corresponding device registry string.
# More specific entries must appear before broader catch-alls.
#
# Ordering rules applied here:
#   1. Entries matched exclusively by model come before entries with broad
#      manufacturer patterns (e.g. eurotronic_spzb0001 before eurotronic_spirit).
#   2. Narrow manufacturer entries come before wide ones
#      (e.g. moes_tv02 / beca_bht_002 before tuya_ts0601 which matches "_TZE").
#   3. Warning / inactive profiles are last so active profiles win on overlap.
_MATCH_TABLE: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # ── Precise model matches first ──────────────────────────────────────────
    ("sonoff_trvzb",        ("TRVZB",),                       ("SONOFF",)),
    ("innr_cozb0001",       ("COZB001",),                     ("innr", "Innr")),
    ("avatto_me167",        ("ME167",),                       ("Avatto", "AVATTO")),
    # Beca before Tuya: BHT-002-GCLZB also carries a _TZE manufacturer string
    ("beca_bht_002",        ("BHT-002", "BHT002"),            ("BECA", "Beca", "BecaSmart")),
    # Moes before Tuya: Moes TV02 uses _TZE manufacturer fingerprints too
    ("moes_tv02",           ("TV02",),                        ("MOES", "Moes")),
    # SPZB0001 matched on model only — no manufacturer tuple prevents false
    # positives when another Eurotronic device lacks the SPZB0001 model string.
    ("eurotronic_spzb0001", ("SPZB0001",),                    ()),
    ("eurotronic_spirit",   ("Spirit",),                      ("Eurotronic",)),
    ("aqara_trv",           ("SRTS-A01",),                    ("Aqara", "LUMI")),
    ("danfoss_ally",        ("Ally",),                        ("Danfoss",)),
    ("bosch_bth",           ("BTH-RA",),                      ("Bosch",)),
    # ── Broader Tuya catch-all after all specific _TZE variants ─────────────
    ("tuya_ts0601",         ("TS0601",),                      ("_TZE", "Tuya")),
    # ── HomeKit bridge — short "Eve" string kept last among active profiles ──
    ("eve_thermo",          ("Eve Thermo", "SEA80"),          ("Eve Systems",)),
    # ── Warning / inactive profiles ──────────────────────────────────────────
    ("daikin_climate",      (),                               ("Daikin",)),
    ("netatmo",             ("NRV", "Smart Radiator Valve"),  ("Netatmo",)),
    ("tado",                ("tado",),                        ("tado",)),
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
