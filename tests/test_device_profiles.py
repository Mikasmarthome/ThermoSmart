"""Tests for the ThermoSmart device profile registry and capability definitions.

All tests are pure Python — no HA fixtures required.
"""
from __future__ import annotations

import dataclasses

import pytest

from custom_components.thermosmart.device_profiles import get_profile
from custom_components.thermosmart.device_profiles.capabilities import (
    DeviceProfile,
    SETPOINT_CLIMATE,
    SETPOINT_HVAC_FIRST,
    VALVE_READ_ONLY,
    VALVE_MAX_LIMIT,
)
from custom_components.thermosmart.device_profiles.profiles import (
    GENERIC,
    SONOFF_TRVZB,
    TUYA_TS0601,
    MOES_TV02,
    DANFOSS_ALLY,
    EUROTRONIC_SPIRIT,
    EUROTRONIC_SPZB0001,
    EVE_THERMO,
    AQARA_TRV,
    AVATTO_ME167,
    BECA_BHT_002,
    TADO,
    DAIKIN_CLIMATE,
    NETATMO,
    ALL_PROFILES,
)


# ── Registry matching ────────────────────────────────────────────────────────

class TestGetProfile:
    def test_unknown_device_returns_generic(self):
        assert get_profile("Unknown TRV Model", "Unknown Brand") is GENERIC

    def test_none_inputs_return_generic(self):
        assert get_profile(None, None) is GENERIC

    def test_empty_strings_return_generic(self):
        assert get_profile("", "") is GENERIC

    def test_one_none_one_empty_returns_generic(self):
        assert get_profile(None, "") is GENERIC
        assert get_profile("", None) is GENERIC

    # ── SONOFF TRVZB ────────────────────────────────────────────────────────
    def test_sonoff_matched_by_model(self):
        assert get_profile("TRVZB", None).identifier == "sonoff_trvzb"

    def test_sonoff_matched_by_manufacturer(self):
        assert get_profile("SomeModel", "SONOFF").identifier == "sonoff_trvzb"

    def test_sonoff_model_case_insensitive(self):
        assert get_profile("trvzb", None).identifier == "sonoff_trvzb"

    def test_sonoff_manufacturer_case_insensitive(self):
        assert get_profile(None, "sonoff").identifier == "sonoff_trvzb"

    def test_sonoff_partial_model_match(self):
        """Matching works as substring — real devices report longer strings."""
        assert get_profile("SONOFF TRVZB zigbee thermostat v2", None).identifier == "sonoff_trvzb"

    # ── Tuya TS0601 ─────────────────────────────────────────────────────────
    def test_tuya_matched_by_model(self):
        assert get_profile("TS0601", "Some Brand").identifier == "tuya_ts0601"

    def test_tuya_matched_by_tze_manufacturer(self):
        """_TZE manufacturer prefix matches Tuya catch-all."""
        assert get_profile("SomeModel", "_TZE200_something").identifier == "tuya_ts0601"

    def test_tuya_matched_by_manufacturer_name(self):
        assert get_profile("SomeModel", "Tuya").identifier == "tuya_ts0601"

    # ── Priority: specific models win over broad Tuya catch-all ─────────────
    def test_moes_takes_priority_over_tuya(self):
        """Moes TV02 is matched before the broad _TZE Tuya catch-all."""
        assert get_profile("TV02", "MOES").identifier == "moes_tv02"

    def test_beca_takes_priority_over_tuya(self):
        """Beca BHT-002 is matched before the broad _TZE Tuya catch-all."""
        assert get_profile("BHT-002-GCLZB", "BecaSmart").identifier == "beca_bht_002"

    # ── Danfoss ─────────────────────────────────────────────────────────────
    def test_danfoss_ally_matched_by_manufacturer(self):
        assert get_profile("Ally", "Danfoss").identifier == "danfoss_ally"

    def test_danfoss_matched_by_model_ally(self):
        assert get_profile("Ally TRV", None).identifier == "danfoss_ally"

    # ── Eurotronic ──────────────────────────────────────────────────────────
    def test_eurotronic_spzb0001_matched_by_model(self):
        assert get_profile("SPZB0001", None).identifier == "eurotronic_spzb0001"

    def test_eurotronic_spirit_matched_by_model(self):
        assert get_profile("Spirit Zigbee", None).identifier == "eurotronic_spirit"

    def test_eurotronic_spirit_matched_by_manufacturer(self):
        assert get_profile("UnknownModel", "Eurotronic").identifier == "eurotronic_spirit"

    # ── Avatto / Beca / Bosch ────────────────────────────────────────────────
    def test_avatto_me167_matched(self):
        assert get_profile("ME167", "AVATTO").identifier == "avatto_me167"

    def test_beca_bht002_matched(self):
        assert get_profile("BHT002", "Beca").identifier == "beca_bht_002"

    # ── Inactive profiles ────────────────────────────────────────────────────
    def test_tado_matched_and_inactive(self):
        profile = get_profile("tado smart radiator", "tado")
        assert profile.identifier == "tado"
        assert profile.is_active is False

    def test_daikin_matched_and_inactive(self):
        profile = get_profile("SomeModel", "Daikin")
        assert profile.identifier == "daikin_climate"
        assert profile.is_active is False

    def test_netatmo_matched_and_inactive(self):
        profile = get_profile("NRV", "Netatmo")
        assert profile.identifier == "netatmo"
        assert profile.is_active is False

    # ── Always returns a valid profile ───────────────────────────────────────
    def test_always_returns_device_profile_instance(self):
        """get_profile never returns None — always a DeviceProfile."""
        result = get_profile("totally-unknown", "mystery-brand")
        assert isinstance(result, DeviceProfile)

    def test_all_profile_ids_resolvable(self):
        """Every identifier referenced in _MATCH_TABLE resolves to a real profile."""
        from custom_components.thermosmart.device_profiles.registry import _MATCH_TABLE, _BY_ID
        for identifier, _, _ in _MATCH_TABLE:
            assert identifier in _BY_ID, f"Profile '{identifier}' in match table but not in profiles"


# ── Profile capability verification ─────────────────────────────────────────

class TestProfileCapabilities:
    def test_all_profiles_have_non_empty_identifier(self):
        for p in ALL_PROFILES:
            assert isinstance(p.identifier, str) and p.identifier.strip()

    def test_all_profiles_have_non_empty_display_name(self):
        for p in ALL_PROFILES:
            assert isinstance(p.display_name, str) and p.display_name.strip()

    def test_all_profile_identifiers_are_unique(self):
        identifiers = [p.identifier for p in ALL_PROFILES]
        assert len(identifiers) == len(set(identifiers))

    def test_all_active_profiles_have_positive_burst_guard(self):
        for p in ALL_PROFILES:
            if p.is_active:
                assert p.burst_guard_seconds > 0, f"{p.identifier} has non-positive burst_guard"

    def test_sonoff_has_valve_max_limit_semantics(self):
        assert SONOFF_TRVZB.valve_semantics == VALVE_MAX_LIMIT

    def test_eve_thermo_is_read_only_no_calibration(self):
        assert EVE_THERMO.valve_semantics == VALVE_READ_ONLY
        assert EVE_THERMO.allow_external_temp_input is False
        assert EVE_THERMO.allow_temp_source_select is False
        assert EVE_THERMO.calibration_sync is False

    def test_avatto_has_inverted_calibration(self):
        assert AVATTO_ME167.calibration_inverted is True

    def test_eurotronic_devices_use_hvac_first(self):
        assert EUROTRONIC_SPIRIT.setpoint_method == SETPOINT_HVAC_FIRST
        assert EUROTRONIC_SPZB0001.setpoint_method == SETPOINT_HVAC_FIRST

    def test_eurotronic_hvac_mode_is_heat(self):
        assert EUROTRONIC_SPIRIT.hvac_mode_before_write == "heat"
        assert EUROTRONIC_SPZB0001.hvac_mode_before_write == "heat"

    def test_inactive_profiles_have_warning_text(self):
        for p in [TADO, DAIKIN_CLIMATE, NETATMO]:
            assert p.is_active is False
            assert p.warning is not None and len(p.warning) > 20

    def test_generic_profile_safe_defaults(self):
        """GENERIC profile should be maximally permissive for unknown devices."""
        assert GENERIC.is_active is True
        assert GENERIC.setpoint_method == SETPOINT_CLIMATE
        assert GENERIC.hvac_watchdog is True
        assert GENERIC.calibration_sync is True

    def test_profiles_are_immutable(self):
        """Profiles are frozen dataclasses — mutation must raise an error."""
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            GENERIC.identifier = "hacked"  # type: ignore[misc]

    def test_beca_has_valve_control_uncertain(self):
        """Beca BHT-002 has uncertain valve control flagged."""
        assert BECA_BHT_002.valve_control_uncertain is True

    def test_all_profiles_have_calibration_plausibility_limit(self):
        """calibration_plausibility_limit must be a positive float on all profiles."""
        for p in ALL_PROFILES:
            assert p.calibration_plausibility_limit > 0

    def test_minimum_setpoint_within_bounds_if_set(self):
        """When minimum_setpoint is set, it must be a positive number."""
        for p in ALL_PROFILES:
            if p.minimum_setpoint is not None:
                assert p.minimum_setpoint > 0, f"{p.identifier} minimum_setpoint not positive"
