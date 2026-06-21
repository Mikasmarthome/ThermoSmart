"""Integration tests for ThermoSmart _async_update_data (Welle 3).

Tests the indoor safety bypass (v1.1.1 fix) and the complete update loop
using a testable coordinator subclass that stubs out all TRV/maintenance
side-effects while keeping the coordinator's own logic intact.

Pure Python + MagicMock – runs on Windows without Docker.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from custom_components.thermosmart.coordinator import ThermoSmartCoordinator
from custom_components.thermosmart.const import HEATING_MODE_AUTO

from tests.helpers import make_coordinator, make_state, set_hass_states


class _TestableCoordinator(ThermoSmartCoordinator):
    """Subclass with TRV/maintenance side-effects removed for integration testing.

    Only overrides external I/O methods; all coordinator logic (mode evaluation,
    recommendation pipeline, indoor safety bypass, etc.) runs unchanged.
    """

    async def _reset_valve_opening_degree(self) -> bool:
        return True

    async def _async_apply_quirks(self, cfg: dict) -> None:
        pass

    async def _watchdog_hvac(self, cfg: dict, recommendation: dict) -> None:
        pass

    async def _async_calibrate_trvs(self, cfg: dict, recommendation: dict) -> None:
        pass

    async def _apply_temperature(self, cfg: dict, recommendation: dict) -> None:
        pass

    async def _async_set_valve_percent(self, cfg: dict, duty: float) -> None:
        pass

    async def _apply_frost_protection(self, cfg: dict) -> None:
        pass

    async def _async_valve_maintenance(self, cfg: dict, recommendation: dict) -> None:
        pass

    async def _async_write_external_temp(self, cfg: dict, recommendation: dict) -> None:
        pass

    async def _async_manage_temp_source(self, cfg: dict, recommendation: dict) -> None:
        pass

    async def _async_observe_trv_setpoints(
        self, cfg: dict, recommendation: dict, weather: dict
    ) -> None:
        pass

    def _check_boost_outcome(self, cfg: dict) -> None:
        pass

    def _check_heating_failure(self, recommendation: dict) -> None:
        pass


def _make_testable(zone_cfg_overrides: dict | None = None) -> _TestableCoordinator:
    """Create a _TestableCoordinator with mocked dependencies."""
    from unittest.mock import patch as _patch

    base = make_coordinator(zone_cfg_overrides)
    with _patch("homeassistant.helpers.frame.report_usage"):
        coord = _TestableCoordinator(
            base.hass, base.entry, base.weather_engine, base.learning_engine
        )
    coord._valve_reset_done = True  # skip valve reset loop
    return coord


# ── TestIndoorSafetyBypass ────────────────────────────────────────────────────


class TestIndoorSafetyBypass:
    async def test_bypass_activates_when_indoor_below_night_temp(self):
        """v1.1.1: effective_summer becomes False when indoor < night_temp."""
        coord = _make_testable()
        coord._is_summer = True
        coord._summer_override = None

        # Indoor below night_temp (18.0)
        set_hass_states(coord, {"sensor.test_temp": make_state("16.0")})

        result = await coord._async_update_data()
        assert result["zone"]["is_summer"] is False

    async def test_bypass_sets_indoor_safety_active_flag(self):
        coord = _make_testable()
        coord._is_summer = True
        coord._summer_override = None
        coord._indoor_safety_active = False

        set_hass_states(coord, {"sensor.test_temp": make_state("16.0")})

        await coord._async_update_data()
        assert coord._indoor_safety_active is True

    async def test_no_bypass_when_indoor_above_night_temp(self):
        """Indoor >= night_temp → effective_summer = True, no bypass."""
        coord = _make_testable()
        coord._is_summer = True
        coord._summer_override = None

        # Indoor above night_temp (18.0)
        set_hass_states(coord, {"sensor.test_temp": make_state("20.0")})

        result = await coord._async_update_data()
        assert result["zone"]["is_summer"] is True

    async def test_bypass_blocked_by_summer_override(self):
        """With _summer_override set (not None), indoor safety check is skipped."""
        coord = _make_testable()
        coord._is_summer = True
        coord._summer_override = True  # forced summer, bypass check skipped

        set_hass_states(coord, {"sensor.test_temp": make_state("15.0")})

        result = await coord._async_update_data()
        assert result["zone"]["is_summer"] is True

    async def test_bypass_not_active_when_not_summer(self):
        coord = _make_testable()
        coord._is_summer = False
        coord._summer_override = None

        set_hass_states(coord, {"sensor.test_temp": make_state("15.0")})

        result = await coord._async_update_data()
        assert result["zone"]["is_summer"] is False
        assert coord._indoor_safety_active is False


# ── TestIndoorSafetyHysteresis ────────────────────────────────────────────────


class TestIndoorSafetyHysteresis:
    async def test_safety_cleared_when_indoor_reaches_threshold_plus_hysteresis(self):
        """indoor >= night_temp + 0.5 → _indoor_safety_active cleared."""
        coord = _make_testable()
        coord._is_summer = True
        coord._summer_override = None
        coord._indoor_safety_active = True  # bypass was active

        night_temp = coord.zone_cfg.get("night_temp", 18.0)
        # Indoor at night_temp + 0.6 (above hysteresis band of +0.5)
        set_hass_states(coord, {"sensor.test_temp": make_state(str(night_temp + 0.6))})

        await coord._async_update_data()
        assert coord._indoor_safety_active is False

    async def test_safety_not_cleared_within_hysteresis_band(self):
        """indoor at night_temp + 0.4 (inside hysteresis) → flag stays True."""
        coord = _make_testable()
        coord._is_summer = True
        coord._summer_override = None
        coord._indoor_safety_active = True

        night_temp = coord.zone_cfg.get("night_temp", 18.0)
        # Just above night_temp but inside hysteresis band (not yet >= +0.5)
        set_hass_states(coord, {"sensor.test_temp": make_state(str(night_temp + 0.4))})

        await coord._async_update_data()
        # indoor (18.4) < night_temp (18.0) + 0.5 → flag not cleared
        # BUT indoor (18.4) >= night_temp (18.0) → no bypass activation either
        # The elif branch: _indoor_safety_active and indoor >= safety+0.5 → NOT met
        # So flag remains True from pre-condition
        assert coord._indoor_safety_active is True

    async def test_safety_active_cleared_when_override_set(self):
        """When _summer_override is not None, _indoor_safety_active is cleared."""
        coord = _make_testable()
        coord._is_summer = True
        coord._summer_override = True
        coord._indoor_safety_active = True

        set_hass_states(coord, {"sensor.test_temp": make_state("15.0")})

        await coord._async_update_data()
        # The else branch (override is not None): clears _indoor_safety_active
        assert coord._indoor_safety_active is False


# ── TestUpdateDataStructure ───────────────────────────────────────────────────


class TestUpdateDataStructure:
    async def test_update_data_returns_required_keys(self):
        coord = _make_testable()
        set_hass_states(coord, {"sensor.test_temp": make_state("20.0")})
        result = await coord._async_update_data()
        assert "weather" in result
        assert "zone" in result
        assert "active_control" in result
        assert "presence" in result

    async def test_presence_structure_in_result(self):
        coord = _make_testable()
        result = await coord._async_update_data()
        presence = result["presence"]
        assert "persons_home" in presence
        assert "all_away" in presence
        assert "vacation" in presence

    async def test_is_summer_field_present_in_zone(self):
        coord = _make_testable()
        result = await coord._async_update_data()
        assert "is_summer" in result["zone"]
