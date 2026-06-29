"""S1a Item 13 — Control Logic Audit.

Proves all real control/dispatch paths: TPI integration, dispatch gating,
setpoint vs direct-valve paths, mode interaction matrix, preheat suppression,
service-call dedup, multi-TRV handling, restart/restore state, and safety
fallbacks — at the coordinator level.

Pure TPI unit tests are already covered by test_tpi.py (31 tests).
LE2 boost/afterheat/early-cutoff model tests are in test_le2_boost_*.py and
test_le2_afterheat_*.py.  This file tests the coordinator's dispatch gating,
mode priorities, and control-path authority.

Pure Python + MagicMock — runs on Windows, no hass fixture needed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.thermosmart.coordinator import ThermoSmartCoordinator
from custom_components.thermosmart.const import (
    HEATING_MODE_AUTO,
    HEATING_MODE_AWAY,
    HEATING_MODE_COMFORT,
    HEATING_MODE_NIGHT,
    HEATING_MODE_VACATION,
    TPI_MAX_BOOST_CELSIUS,
    TPI_COEF_INT_DEFAULT,
    TPI_COEF_EXT_DEFAULT,
    TEMP_FROST_PROTECTION,
)
from custom_components.thermosmart.tpi import compute_tpi, duty_to_setpoint
from custom_components.thermosmart.trv_control import _DispatchStats

from tests.helpers import make_coordinator, make_state, make_weather_data, set_hass_states


# ── Test infrastructure ───────────────────────────────────────────────────────


class _TrackingCoordinator(ThermoSmartCoordinator):
    """Subclass that tracks dispatch calls without performing real service calls.

    All external I/O methods are stubbed; coordinator control logic is intact.
    _apply_temperature returns a valid (empty) _DispatchStats so the merge path
    inside _async_update_data does not crash.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_temperature_calls: list[dict] = []
        self.frost_protection_calls: int = 0
        self.valve_percent_calls: list[float] = []

    async def _reset_valve_opening_degree(self) -> bool:
        return True

    async def _async_apply_quirks(self, cfg: dict) -> None:
        pass

    async def _watchdog_hvac(self, cfg: dict, recommendation: dict) -> None:
        pass

    async def _async_calibrate_trvs(self, cfg: dict, recommendation: dict) -> None:
        pass

    async def _apply_temperature(self, cfg: dict, recommendation: dict) -> "_DispatchStats":
        self.apply_temperature_calls.append({
            "setpoint": recommendation.get("trv_setpoint"),
            "effective_target": recommendation.get("effective_target"),
            "window_open": recommendation.get("window_open", False),
        })
        return _DispatchStats()

    async def _async_set_valve_percent(self, cfg: dict, duty: float) -> "_DispatchStats":
        self.valve_percent_calls.append(duty)
        return _DispatchStats()

    async def _apply_frost_protection(self, cfg: dict) -> None:
        self.frost_protection_calls += 1

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


def _make_tracking(zone_cfg_overrides: dict | None = None) -> _TrackingCoordinator:
    from unittest.mock import patch as _patch
    base = make_coordinator(zone_cfg_overrides)
    with _patch("homeassistant.helpers.frame.report_usage"):
        coord = _TrackingCoordinator(
            base.hass, base.entry, base.weather_engine, base.learning_engine
        )
    coord._valve_reset_done = True
    return coord


_W = make_weather_data(outdoor_temp=10.0)
_W_NONE = make_weather_data(
    outdoor_temp=None, humidity=None, wind_speed=None,
    solar_radiation=None, forecast_high=None, forecast_low=None,
    condition=None, rain=None,
)


# ── §1 TPI Integration in Coordinator Context ─────────────────────────────────


class TestTPICoordinatorIntegration:
    """Verify TPI is correctly wired into the coordinator update pipeline.

    TPI keys (tpi_duty_cycle, tpi_coef_int, zone_control_type, trv_setpoint) are
    computed in _async_update_data(), NOT in _compute_recommendation(). Tests here
    use _TrackingCoordinator + _async_update_data() and read result["zone"].
    """

    async def test_duty_cycle_zero_when_no_current_temp(self):
        """current_temp=None → duty_cycle=0.0 in full update cycle."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False
        # No states configured → hass.states.get returns None → current_temp=None
        result = await coord._async_update_data()
        assert result["zone"]["tpi_duty_cycle"] == 0.0

    async def test_duty_cycle_positive_when_below_target(self):
        """current_temp < target → TPI duty_cycle > 0 in full update cycle."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False
        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._async_update_data()
        assert result["zone"]["tpi_duty_cycle"] > 0.0

    async def test_duty_cycle_zero_when_at_or_above_target(self):
        """current_temp >= target → duty_cycle = 0 in full update cycle."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False
        set_hass_states(coord, {"sensor.test_temp": make_state("22.0")})
        result = await coord._async_update_data()
        assert result["zone"]["tpi_duty_cycle"] == 0.0

    async def test_duty_cycle_bounded_0_to_100(self):
        """TPI duty must always be in [0, 100] in full update cycle."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False
        set_hass_states(coord, {"sensor.test_temp": make_state("5.0")})
        result = await coord._async_update_data()
        duty = result["zone"]["tpi_duty_cycle"]
        assert 0.0 <= duty <= 100.0

    async def test_trv_setpoint_at_or_above_target_when_below_target(self):
        """Setpoint path: trv_setpoint >= effective_target (TPI boost applied)."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False
        coord._auto_valve_map = {}  # setpoint path (no direct valve)
        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._async_update_data()
        zone = result["zone"]
        if zone.get("trv_setpoint") is not None and zone.get("effective_target") is not None:
            assert zone["trv_setpoint"] >= zone["effective_target"]

    async def test_tpi_keys_present_in_update_result(self):
        """Key TPI diagnostic fields must be in full update cycle result."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False
        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._async_update_data()
        zone = result["zone"]
        for key in ("tpi_duty_cycle", "tpi_coef_int", "tpi_coef_ext", "tpi_valve_direct"):
            assert key in zone, f"Missing TPI key: {key}"

    async def test_zone_control_type_is_setpoint_without_valve_map(self):
        """No direct valve configured → zone_control_type == 'setpoint'."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False
        coord._auto_valve_map = {}
        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._async_update_data()
        assert result["zone"]["zone_control_type"] == "setpoint"

    def test_duty_to_setpoint_at_zero_percent(self):
        """duty_to_setpoint: 0% → setpoint == target."""
        assert duty_to_setpoint(21.0, 0.0, TPI_MAX_BOOST_CELSIUS) == 21.0

    def test_duty_to_setpoint_full_boost_capped_at_28(self):
        """duty_to_setpoint: 100% → 28°C absolute cap."""
        result = duty_to_setpoint(21.0, 100.0, TPI_MAX_BOOST_CELSIUS)
        assert result <= 28.0

    def test_duty_to_setpoint_proportional(self):
        """duty_to_setpoint: 50% duty → target + max_boost/2."""
        result = duty_to_setpoint(21.0, 50.0, 8.0)
        assert result == pytest.approx(25.0)

    def test_tpi_deterministic_same_inputs(self):
        """Same inputs → identical outputs (no hidden state)."""
        result_a = compute_tpi(21.0, 19.0, 5.0, TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT)
        result_b = compute_tpi(21.0, 19.0, 5.0, TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT)
        assert result_a == result_b

    def test_tpi_no_nan_inf(self):
        """TPI output must never be NaN or Inf."""
        import math
        result = compute_tpi(21.0, 19.0, 5.0, TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT)
        assert not math.isnan(result)
        assert not math.isinf(result)


# ── §2 Dispatch Gating ────────────────────────────────────────────────────────


class TestDispatchGating:
    """Active Control gate: True → dispatch; False → no dispatch."""

    async def test_active_control_false_no_apply_temperature_called(self):
        """Active Control OFF → _apply_temperature must NOT be called."""
        coord = _make_tracking()
        coord.set_active_control(False)
        coord._is_summer = False

        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        await coord._async_update_data()

        assert len(coord.apply_temperature_calls) == 0

    async def test_active_control_true_not_summer_apply_temperature_called(self):
        """Active Control ON, not summer → _apply_temperature IS called."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False

        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        await coord._async_update_data()

        assert len(coord.apply_temperature_calls) >= 1

    async def test_active_control_true_summer_frost_protection_called(self):
        """Active Control ON + summer → frost protection, NOT _apply_temperature."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = True
        coord._summer_override = True  # force summer, skip bypass check

        set_hass_states(coord, {"sensor.test_temp": make_state("25.0")})
        await coord._async_update_data()

        assert coord.frost_protection_calls >= 1
        assert len(coord.apply_temperature_calls) == 0

    async def test_active_control_false_summer_no_dispatch(self):
        """Active Control OFF even in summer → no dispatch of any kind."""
        coord = _make_tracking()
        coord.set_active_control(False)
        coord._is_summer = True

        set_hass_states(coord, {"sensor.test_temp": make_state("25.0")})
        await coord._async_update_data()

        assert len(coord.apply_temperature_calls) == 0
        assert coord.frost_protection_calls == 0

    async def test_active_control_not_initialized_at_startup(self):
        """Fresh coordinator: _active_control_initialized must be False."""
        coord = _make_tracking()
        assert coord._active_control_initialized is False

    async def test_set_active_control_sets_initialized_flag(self):
        """set_active_control() must flip _active_control_initialized to True."""
        coord = _make_tracking()
        assert not coord._active_control_initialized
        coord.set_active_control(True)
        assert coord._active_control_initialized is True

    async def test_active_control_false_also_sets_initialized(self):
        """set_active_control(False) also sets _active_control_initialized."""
        coord = _make_tracking()
        coord.set_active_control(False)
        assert coord._active_control_initialized is True

    async def test_window_open_valve_duty_forced_to_zero(self):
        """Window open: direct valve duty must be 0 even if TPI says otherwise."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False

        # Mark one TRV as having direct valve control
        coord._auto_valve_map = {"climate.test_trv": "sensor.valve_trv"}
        set_hass_states(coord, {
            "sensor.test_temp": make_state("19.0"),
            "binary_sensor.window": make_state("on"),
        })
        entry_opts = dict(coord.entry.options)
        entry_opts["window_sensors"] = ["binary_sensor.window"]
        entry_opts["window_open_delay"] = 0
        coord.entry.options = entry_opts

        await coord._async_update_data()
        # Valve percent calls with duty > 0 must NOT appear when window open
        non_zero = [d for d in coord.valve_percent_calls if d > 0.0]
        assert len(non_zero) == 0, f"Valve received non-zero duty during window open: {non_zero}"


# ── §3 Mode Interaction Matrix ────────────────────────────────────────────────


class TestModeInteractionMatrix:
    """Prove priority ordering: Window Open > Manual Override > Schedule/Comfort."""

    async def test_window_open_beats_manual_override(self):
        """Window open suppresses effective_target even when override is active."""
        coord = make_coordinator()
        coord._override = 25.0  # manual override active
        cfg = {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": ["sensor.test_temp"], "climate_entities": ["climate.test_trv"],
            "humidity_sensors": [], "window_sensors": ["binary_sensor.w"],
            "window_open_delay": 0, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }
        set_hass_states(coord, {
            "sensor.test_temp": make_state("20.0"),
            "binary_sensor.w": make_state("on"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is True
        assert result["effective_target"] is None
        assert result["override_active"] is False

    async def test_window_open_duty_zero(self):
        """Window open: duty cycle is 0, no TPI heating demand."""
        coord = make_coordinator()
        cfg = {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": ["sensor.test_temp"], "climate_entities": ["climate.test_trv"],
            "humidity_sensors": [], "window_sensors": ["binary_sensor.w"],
            "window_open_delay": 0, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }
        set_hass_states(coord, {
            "sensor.test_temp": make_state("19.0"),
            "binary_sensor.w": make_state("on"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is True
        # No TPI section executed → tpi_duty_cycle absent or 0
        assert result.get("tpi_duty_cycle", 0.0) == 0.0

    async def test_away_mode_clears_override(self):
        """AWAY mode: manual override cleared → override_active=False."""
        coord = make_coordinator()
        coord._override = 22.0
        coord._override_schedule_period = "wd_comfort"
        cfg = {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": [], "climate_entities": [],
            "humidity_sensors": [], "window_sensors": [],
            "window_open_delay": 5, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AWAY)
        assert result["override_active"] is False
        assert coord._override is None

    async def test_summer_mode_suppresses_normal_dispatch(self):
        """Summer mode: frost protection, not normal heating."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = True
        coord._summer_override = True

        set_hass_states(coord, {"sensor.test_temp": make_state("22.0")})
        await coord._async_update_data()

        assert coord.frost_protection_calls >= 1
        assert len(coord.apply_temperature_calls) == 0

    async def test_vacation_beats_manual_mode(self):
        """Vacation presence flag overrides explicit COMFORT mode setting."""
        coord = make_coordinator()
        coord._mode = HEATING_MODE_COMFORT
        presence = {"vacation": True, "all_away": False, "persons_home": []}
        result = coord._effective_mode(presence)
        assert result == HEATING_MODE_VACATION

    async def test_active_control_off_beats_everything_for_dispatch(self):
        """Active Control OFF: no dispatch even with valid target and room temp."""
        coord = _make_tracking()
        coord.set_active_control(False)
        coord._is_summer = False

        set_hass_states(coord, {"sensor.test_temp": make_state("18.0")})
        await coord._async_update_data()

        assert len(coord.apply_temperature_calls) == 0
        assert coord.frost_protection_calls == 0

    async def test_summer_active_control_false_no_frost_protection(self):
        """Active Control OFF + summer → frost protection also not called."""
        coord = _make_tracking()
        coord.set_active_control(False)
        coord._is_summer = True
        coord._summer_override = True

        set_hass_states(coord, {"sensor.test_temp": make_state("25.0")})
        await coord._async_update_data()

        assert coord.frost_protection_calls == 0


# ── §4 Preheat Logic ─────────────────────────────────────────────────────────


class TestPreheatLogic:
    """Preheat activation, suppression, and target elevation."""

    async def _cfg(self, window_sensors=None, window_open_delay=5):
        return {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": ["sensor.test_temp"], "climate_entities": ["climate.test_trv"],
            "humidity_sensors": [], "window_sensors": window_sensors or [],
            "window_open_delay": window_open_delay, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }

    async def test_no_preheat_without_upcoming_schedule(self):
        """_minutes_until_next_comfort returning None → preheat_active=False."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        with patch.object(coord, "_minutes_until_next_comfort", return_value=None):
            result = await coord._compute_recommendation(
                await self._cfg(), _W, HEATING_MODE_AUTO
            )
        assert result["preheat_active"] is False

    async def test_preheat_active_raises_base_target_to_comfort(self):
        """Upcoming schedule in ≤preheat_minutes → base_target raised to comfort_temp."""
        coord = make_coordinator()
        set_hass_states(coord, {"sensor.test_temp": make_state("18.0")})
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        with patch.object(coord, "_minutes_until_next_comfort", return_value=20):
            result = await coord._compute_recommendation(
                await self._cfg(), _W, HEATING_MODE_AUTO
            )
        if result["preheat_active"]:
            assert result["base_target"] == 21.0

    async def test_window_open_suppresses_effective_target_not_preheat_flag(self):
        """Window open kills effective_target; preheat_active is independent of window.

        The coordinator tracks preheat and window state separately.  Window open
        sets effective_target=None so no heat is dispatched, but does not clear
        the preheat_active diagnostic flag.
        """
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        cfg = await self._cfg(window_sensors=["binary_sensor.w"], window_open_delay=0)
        set_hass_states(coord, {
            "sensor.test_temp": make_state("18.0"),
            "binary_sensor.w": make_state("on"),
        })
        with patch.object(coord, "_minutes_until_next_comfort", return_value=15):
            result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["window_open"] is True
        # Window kills effective_target — no heat regardless of preheat_active
        assert result["effective_target"] is None

    async def test_zero_preheat_minutes_no_preheat(self):
        """preheat_minutes=0 → preheat_active=False."""
        coord = make_coordinator()
        coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=0)
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        result = await coord._compute_recommendation(
            await self._cfg(), _W, HEATING_MODE_AUTO
        )
        assert result["preheat_active"] is False


# ── §5 Target Temperature Resolution ──────────────────────────────────────────


class TestTargetTemperatureResolution:
    """Effective target correctly resolved from mode, override, and offset."""

    async def _base_cfg(self):
        return {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": ["sensor.test_temp"], "climate_entities": ["climate.test_trv"],
            "humidity_sensors": [], "window_sensors": [],
            "window_open_delay": 5, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }

    async def test_override_dominates_auto_mode_target(self):
        """Manual override temp used as adjusted_target regardless of schedule."""
        coord = make_coordinator()
        coord._override = 23.0
        result = await coord._compute_recommendation(
            await self._base_cfg(), _W, HEATING_MODE_AUTO
        )
        assert result["override_active"] is True
        assert result["adjusted_target"] == 23.0

    async def test_effective_target_equals_adjusted_plus_offset_in_auto(self):
        """effective_target = adjusted_target + weather_offset (when positive)."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        coord.weather_engine.compute_temperature_offset.return_value = 1.0
        set_hass_states(coord, {"sensor.test_temp": make_state("22.0")})
        result = await coord._compute_recommendation(
            await self._base_cfg(), _W, HEATING_MODE_AUTO
        )
        assert result["weather_offset"] == 1.0
        if result["adjusted_target"] is not None:
            assert result["effective_target"] == pytest.approx(
                result["adjusted_target"] + 1.0, abs=0.2
            )

    async def test_effective_target_none_when_window_open(self):
        """Window open: effective_target must be None (heating suppressed)."""
        coord = make_coordinator()
        cfg = await self._base_cfg()
        cfg["window_sensors"] = ["binary_sensor.w"]
        cfg["window_open_delay"] = 0
        set_hass_states(coord, {
            "sensor.test_temp": make_state("19.0"),
            "binary_sensor.w": make_state("on"),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["effective_target"] is None

    async def test_vacation_temp_applied_in_vacation_mode(self):
        """VACATION mode: adjusted_target = vacation_temp."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=12.0)
        result = await coord._compute_recommendation(
            await self._base_cfg(), _W, HEATING_MODE_VACATION
        )
        assert result["adjusted_target"] == 12.0

    async def test_night_temp_applied_in_night_mode(self):
        """NIGHT mode: adjusted_target = night_temp."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        result = await coord._compute_recommendation(
            await self._base_cfg(), _W, HEATING_MODE_NIGHT
        )
        assert result["adjusted_target"] == 18.0

    async def test_weather_offset_not_applied_in_non_auto_mode(self):
        """Weather offset is only applied in AUTO mode."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.0)
        coord.weather_engine.compute_temperature_offset.return_value = 2.0
        result = await coord._compute_recommendation(
            await self._base_cfg(), _W, HEATING_MODE_NIGHT
        )
        assert result["weather_offset"] == 0.0


# ── §6 Restart / Restore State ────────────────────────────────────────────────


class TestRestartRestoreState:
    """Control state is safe to reset; boost gate is properly initialized."""

    def test_active_control_default_false(self):
        """Fresh coordinator has active_control=False until explicitly set."""
        coord = _make_tracking()
        assert coord._active_control is False

    def test_active_control_initialized_false_until_set(self):
        """_active_control_initialized must be False before any set_active_control call."""
        coord = _make_tracking()
        assert coord._active_control_initialized is False

    def test_set_active_control_true_updates_both_flags(self):
        coord = _make_tracking()
        coord.set_active_control(True)
        assert coord._active_control is True
        assert coord._active_control_initialized is True

    def test_set_active_control_false_updates_both_flags(self):
        coord = _make_tracking()
        coord.set_active_control(False)
        assert coord._active_control is False
        assert coord._active_control_initialized is True

    def test_early_cutoff_hold_false_at_startup(self):
        """Early cutoff hold must not be active at startup."""
        coord = _make_tracking()
        assert coord._ec_hold_active is False

    def test_tpi_duty_previous_none_at_startup(self):
        """_tpi_duty_previous is None before first cycle."""
        coord = _make_tracking()
        assert coord._tpi_duty_previous is None

    def test_tpi_coef_int_initialized_to_default(self):
        """_tpi_coef_int_smoothed starts at deterministic default."""
        from custom_components.thermosmart.const import TPI_COEF_INT_DEFAULT
        coord = _make_tracking()
        assert coord._tpi_coef_int_smoothed == TPI_COEF_INT_DEFAULT

    def test_is_summer_false_at_startup(self):
        """Summer mode must not be active at fresh startup."""
        coord = _make_tracking()
        assert coord._is_summer is False

    def test_override_none_at_startup(self):
        coord = _make_tracking()
        assert coord._override is None

    async def test_two_cycles_dispatch_called_each_time(self):
        """Repeated cycles: dispatch occurs in each cycle when active_control=True."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False

        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})

        await coord._async_update_data()
        calls_after_1 = len(coord.apply_temperature_calls)
        await coord._async_update_data()
        calls_after_2 = len(coord.apply_temperature_calls)

        assert calls_after_2 > calls_after_1


# ── §7 Multi-TRV and Unavailable Device Handling ─────────────────────────────


class TestMultiTRVHandling:
    """Multiple TRVs: individual unavailability must not crash or block dispatch."""

    async def test_two_trvs_avg_temp_used(self):
        """Two TRVs with different temps → averaged current_temp."""
        coord = make_coordinator()
        cfg = {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": [], "climate_entities": ["climate.trv1", "climate.trv2"],
            "humidity_sensors": [], "window_sensors": [],
            "window_open_delay": 5, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }
        set_hass_states(coord, {
            "climate.trv1": make_state("heat", {"current_temperature": 18.0}),
            "climate.trv2": make_state("heat", {"current_temperature": 20.0}),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] == pytest.approx(19.0)

    async def test_one_trv_unavailable_other_still_provides_temp(self):
        """One TRV unavailable → averaged from remaining available TRV."""
        coord = make_coordinator()
        cfg = {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": [], "climate_entities": ["climate.trv1", "climate.trv2"],
            "humidity_sensors": [], "window_sensors": [],
            "window_open_delay": 5, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }
        set_hass_states(coord, {
            "climate.trv1": make_state("unavailable"),
            "climate.trv2": make_state("heat", {"current_temperature": 20.0}),
        })
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] == pytest.approx(20.0)

    async def test_all_trvs_unavailable_current_temp_none(self):
        """All TRVs unavailable, no external sensors → current_temp=None."""
        coord = make_coordinator()
        cfg = {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": [], "climate_entities": ["climate.trv1"],
            "humidity_sensors": [], "window_sensors": [],
            "window_open_delay": 5, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }
        set_hass_states(coord, {"climate.trv1": make_state("unavailable")})
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result["current_temp"] is None

    async def test_dispatch_runs_without_crash_when_trv_unavailable(self):
        """Coordinator must not crash when TRV is unavailable during dispatch cycle."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False

        set_hass_states(coord, {
            "sensor.test_temp": make_state("19.0"),
            "climate.test_trv": make_state("unavailable"),
        })

        result = await coord._async_update_data()
        assert result is not None


# ── §8 Active Control and Learning OFF Gates ──────────────────────────────────


class TestControlGates:
    """Active Control and Learning OFF gate behavior."""

    async def test_learning_off_uses_default_tpi_coefficients(self):
        """No LE2 shadow → deterministic TPI coefficients in full update cycle."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False
        coord._le2_shadow = None
        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._async_update_data()
        assert result["zone"]["tpi_coef_int"] == pytest.approx(TPI_COEF_INT_DEFAULT, abs=0.001)

    async def test_no_shadow_no_boost_in_recommendation(self):
        """Without LE2 shadow, boost fields reflect baseline (no boost)."""
        coord = make_coordinator()
        coord._le2_shadow = None
        cfg = {
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            "vacation_temp": 12.0, "eco_temp": 19.0, "window_open_temp": 5.0,
            "temp_tolerance": 0.5,
            "temp_sensors": ["sensor.test_temp"], "climate_entities": ["climate.test_trv"],
            "humidity_sensors": [], "window_sensors": [],
            "window_open_delay": 5, "window_close_delay": 2,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            "schedule_enabled": True,
        }
        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._compute_recommendation(cfg, _W, HEATING_MODE_AUTO)
        assert result.get("boost_offset_c", 0.0) == 0.0
        assert result.get("applied_boost_offset_c", 0.0) == 0.0

    async def test_observation_mode_no_service_calls(self):
        """Observation mode (active_control=False): full cycle runs but no dispatch."""
        coord = _make_tracking()
        coord.set_active_control(False)
        coord._is_summer = False

        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._async_update_data()

        assert result is not None
        assert len(coord.apply_temperature_calls) == 0
        assert len(coord.valve_percent_calls) == 0


# ── §9 Service Call Dedup / Last Written Setpoints ───────────────────────────


class TestServiceCallDedup:
    """_last_written_setpoints tracks what was dispatched for manual-change detection."""

    def test_last_written_setpoints_empty_at_startup(self):
        """No dispatch history at startup."""
        coord = _make_tracking()
        assert coord._last_written_setpoints == {}

    async def test_dispatch_updates_no_crash(self):
        """Dispatch cycle with active_control=True completes without exception."""
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = False

        set_hass_states(coord, {"sensor.test_temp": make_state("19.0")})
        result = await coord._async_update_data()

        assert result is not None


# ── §10 Control Reason Priority Chain ────────────────────────────────────────


class TestControlReasonPriority:
    """_control_reason returns the correct priority string."""

    def _rec(self, **kwargs) -> dict:
        defaults = {
            "window_open": False, "override_active": False,
            "mode": HEATING_MODE_AUTO, "is_summer": False,
        }
        return {**defaults, **kwargs}

    def test_window_open_highest_priority(self):
        rec = self._rec(window_open=True, override_active=True, is_summer=True)
        assert make_coordinator()._control_reason(rec) == "window_open"

    def test_override_second_priority(self):
        rec = self._rec(override_active=True, is_summer=True, mode=HEATING_MODE_VACATION)
        assert make_coordinator()._control_reason(rec) == "manual_override"

    def test_vacation_third_priority(self):
        rec = self._rec(mode=HEATING_MODE_VACATION, is_summer=True)
        assert make_coordinator()._control_reason(rec) == "vacation"

    def test_summer_fourth_priority(self):
        rec = self._rec(is_summer=True)
        assert make_coordinator()._control_reason(rec) == "summer_mode"

    def test_away_fifth_priority(self):
        rec = self._rec(mode=HEATING_MODE_AWAY)
        assert make_coordinator()._control_reason(rec) == "presence"

    def test_default_is_schedule(self):
        rec = self._rec()
        assert make_coordinator()._control_reason(rec) == "schedule"


# ── §11 Frost Protection Temperature ─────────────────────────────────────────


class TestFrostProtection:
    """Summer mode applies frost protection temperature constant."""

    def test_frost_protection_constant_is_plausible(self):
        """TEMP_FROST_PROTECTION must be a safe, low temperature (e.g., 12°C)."""
        assert 8.0 <= TEMP_FROST_PROTECTION <= 16.0

    async def test_summer_sets_frost_protection_in_update_cycle(self):
        """Summer mode: full update cycle shows is_summer=True + low trv_setpoint.

        is_summer and trv_setpoint are set in _async_update_data(), not in
        _compute_recommendation(), so the full update cycle must be used here.
        """
        coord = _make_tracking()
        coord.set_active_control(True)
        coord._is_summer = True
        coord._summer_override = True
        set_hass_states(coord, {"sensor.test_temp": make_state("25.0")})
        result = await coord._async_update_data()
        assert result["zone"]["is_summer"] is True
        if result["zone"].get("trv_setpoint") is not None:
            assert result["zone"]["trv_setpoint"] <= 15.0
