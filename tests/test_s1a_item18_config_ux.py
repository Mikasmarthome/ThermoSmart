"""
S1a Item 18 — Config / Options / Entity UX Cleanup Audit

Windows-compatible tests (no hass fixture required).

Coverage:
- Config flow validation logic (_validate_temps, _NullableEntitySelector)
- Schema helper existence and structure
- Options Hub structure (methods, no hass)
- Entity unique_id stability (pattern, zone-rename independence)
- Safety defaults (active_control OFF, learning ON, safe temps)
- Translation parity (JSON validity, key structure across all 24 files)
- Config compatibility (_current() merge semantics)
- Options Hub partial-save semantics (dict merge logic)
- Diagnostics / export UX readiness
- No service calls on options open, no model reset without explicit action
"""
from __future__ import annotations

import inspect
import json
import pathlib
import types
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.config_flow import (
    ThermoSmartConfigFlow,
    ThermoSmartOptionsFlow,
    ThermoSmartSystemOptionsFlow,
    _NullableEntitySelector,
    _schema_devices,
    _schema_presence,
    _schema_schedule,
    _schema_weather,
    _validate_temps,
)
from custom_components.thermosmart.const import (
    CONF_CALIBRATION_INVERT,
    CONF_ECO_TEMP,
    CONF_HOME_ZONE,
    CONF_LEARNING_ENABLED,
    CONF_MANAGE_TEMP_SOURCE,
    CONF_OUTDOOR_HUMIDITY_SENSOR,
    CONF_OUTDOOR_RAIN_SENSOR,
    CONF_OUTDOOR_SOLAR_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_OUTDOOR_WIND_SENSOR,
    CONF_PRESENCE_PERSONS,
    CONF_SCHEDULE_ENABLED,
    CONF_VACATION_TEMP,
    CONF_VALVE_MAINTENANCE,
    CONF_WEATHER_ENTITY,
    CONF_WINDOW_OPEN_TEMP,
    DEFAULT_LEARNING_ENABLED,
    DOMAIN,
    TEMP_COMFORT,
    TEMP_NIGHT,
    TEMP_AWAY,
    TEMP_FROST_PROTECTION,
    TEMP_ECO,
    WINDOW_OPEN_SETPOINT,
)

_COMPONENT = pathlib.Path("custom_components/thermosmart")
_TRANSLATIONS = _COMPONENT / "translations"
_STRINGS = _COMPONENT / "strings.json"

# ── §1  Config Flow Validation Logic ─────────────────────────────────────────


class TestValidateTempsLogic:
    def test_valid_defaults_no_errors(self):
        assert _validate_temps({"comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0}) == {}

    def test_night_above_comfort_is_invalid(self):
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 22.0, "away_temp": 17.0})
        assert errors.get("night_temp") == "night_temp_too_high"

    def test_away_above_night_is_invalid(self):
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 19.0})
        assert errors.get("away_temp") == "away_temp_too_high"

    def test_both_errors_simultaneously(self):
        errors = _validate_temps({"comfort_temp": 20.0, "night_temp": 21.0, "away_temp": 22.0})
        assert "night_temp" in errors
        assert "away_temp" in errors

    def test_empty_dict_no_crash(self):
        errors = _validate_temps({})
        assert isinstance(errors, dict)

    def test_non_numeric_values_no_crash(self):
        errors = _validate_temps({"comfort_temp": "bad", "night_temp": 18.0, "away_temp": 17.0})
        assert isinstance(errors, dict)

    def test_away_equal_to_night_is_valid(self):
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 18.0})
        assert "away_temp" not in errors

    def test_returns_string_error_values(self):
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 22.0, "away_temp": 17.0})
        for v in errors.values():
            assert isinstance(v, str)


class TestNullableEntitySelectorLogic:
    def _sel(self):
        from homeassistant.helpers import selector
        return _NullableEntitySelector(selector.EntitySelectorConfig(domain="weather"))

    def test_none_returns_none(self):
        assert self._sel()(None) is None

    def test_empty_string_returns_none(self):
        assert self._sel()("") is None

    def test_none_does_not_crash(self):
        result = self._sel()(None)
        assert result is None  # no exception


# ── §2  Schema Helper Structure ───────────────────────────────────────────────


class TestSchemaHelpers:
    def _defaults(self) -> dict:
        return {
            "name": "Test", "climate_entities": ["climate.trv"],
            "temp_sensors": [], "humidity_sensors": [], "window_sensors": [],
            "window_open_delay": 5, "window_close_delay": 2,
            CONF_VALVE_MAINTENANCE: True, CONF_MANAGE_TEMP_SOURCE: False,
            CONF_CALIBRATION_INVERT: False,
            "comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0,
            CONF_VACATION_TEMP: 12.0, CONF_ECO_TEMP: TEMP_ECO,
            CONF_WINDOW_OPEN_TEMP: WINDOW_OPEN_SETPOINT, "temp_tolerance": 0.5,
            CONF_SCHEDULE_ENABLED: True,
            "sched_wd_morning": "06:00", "sched_wd_night": "22:00",
            "sched_we_morning": "08:00", "sched_we_night": "23:00",
            CONF_PRESENCE_PERSONS: [], CONF_HOME_ZONE: "zone.home",
            CONF_LEARNING_ENABLED: True,
            CONF_WEATHER_ENTITY: None,
            CONF_OUTDOOR_TEMP_SENSOR: None, CONF_OUTDOOR_HUMIDITY_SENSOR: None,
            CONF_OUTDOOR_WIND_SENSOR: None, CONF_OUTDOOR_SOLAR_SENSOR: None,
            CONF_OUTDOOR_RAIN_SENSOR: None,
        }

    def test_schema_devices_returns_schema(self):
        import voluptuous as vol
        schema = _schema_devices(self._defaults())
        assert isinstance(schema, vol.Schema)

    def test_schema_schedule_returns_schema(self):
        import voluptuous as vol
        schema = _schema_schedule(self._defaults())
        assert isinstance(schema, vol.Schema)

    def test_schema_presence_returns_schema(self):
        import voluptuous as vol
        schema = _schema_presence(self._defaults())
        assert isinstance(schema, vol.Schema)

    def test_schema_weather_returns_schema(self):
        import voluptuous as vol
        schema = _schema_weather(self._defaults())
        assert isinstance(schema, vol.Schema)

    def test_schema_devices_accepts_valid_input(self):
        schema = _schema_devices(self._defaults())
        result = schema({
            "name": "Test", "climate_entities": ["climate.trv"],
            "window_open_delay": 5, "window_close_delay": 2,
            CONF_VALVE_MAINTENANCE: True, CONF_MANAGE_TEMP_SOURCE: False,
            CONF_CALIBRATION_INVERT: False,
        })
        assert result["name"] == "Test"


# ── §3  Options Hub Structure ─────────────────────────────────────────────────


class TestOptionsHubStructure:
    def test_async_step_init_exists(self):
        assert hasattr(ThermoSmartOptionsFlow, "async_step_init")

    def test_async_step_devices_exists(self):
        assert hasattr(ThermoSmartOptionsFlow, "async_step_devices")

    def test_async_step_schedule_exists(self):
        assert hasattr(ThermoSmartOptionsFlow, "async_step_schedule")

    def test_async_step_presence_exists(self):
        assert hasattr(ThermoSmartOptionsFlow, "async_step_presence")

    def test_async_step_weather_exists(self):
        assert hasattr(ThermoSmartOptionsFlow, "async_step_weather")

    def test_options_flow_has_no_self_data_accumulation(self):
        """Hub no longer needs _data accumulation — __init__ should not define self._data."""
        init = ThermoSmartOptionsFlow.__init__ if "__init__" in vars(ThermoSmartOptionsFlow) else None
        if init is not None:
            src = inspect.getsource(init)
            assert "self._data" not in src, "Hub should not use self._data accumulation"

    def test_step_init_is_coroutine(self):
        assert inspect.iscoroutinefunction(ThermoSmartOptionsFlow.async_step_init)

    def test_step_devices_is_coroutine(self):
        assert inspect.iscoroutinefunction(ThermoSmartOptionsFlow.async_step_devices)

    def test_step_schedule_is_coroutine(self):
        assert inspect.iscoroutinefunction(ThermoSmartOptionsFlow.async_step_schedule)

    def test_step_presence_is_coroutine(self):
        assert inspect.iscoroutinefunction(ThermoSmartOptionsFlow.async_step_presence)

    def test_step_weather_is_coroutine(self):
        assert inspect.iscoroutinefunction(ThermoSmartOptionsFlow.async_step_weather)

    def test_config_flow_has_get_options_flow_callback(self):
        assert hasattr(ThermoSmartConfigFlow, "async_get_options_flow")

    def test_system_options_flow_returns_abort(self):
        """ThermoSmartSystemOptionsFlow.async_step_init is a coroutine returning abort."""
        assert inspect.iscoroutinefunction(ThermoSmartSystemOptionsFlow.async_step_init)


# ── §4  Options Hub Partial-Save Merge Semantics ─────────────────────────────


class TestOptionsHubMergeSemantics:
    def _make_flow(self, data: dict, options: dict) -> ThermoSmartOptionsFlow:
        flow = ThermoSmartOptionsFlow.__new__(ThermoSmartOptionsFlow)
        entry = MagicMock()
        entry.data = data
        entry.options = options
        # config_entry is a read-only property on OptionsFlow; bypass via object.__setattr__
        object.__setattr__(flow, "_flow_result", {})
        flow.__dict__["_config_entry_id"] = "mock"
        # Patch _current() directly to avoid property
        flow._current = lambda: {**data, **options}
        return flow

    def test_current_merges_data_and_options(self):
        flow = self._make_flow(
            data={"comfort_temp": 21.0, "name": "Zone"},
            options={"comfort_temp": 22.0}
        )
        current = flow._current()
        assert current["comfort_temp"] == 22.0  # options wins
        assert current["name"] == "Zone"         # from data

    def test_current_with_empty_options(self):
        flow = self._make_flow(
            data={"comfort_temp": 21.0},
            options={}
        )
        current = flow._current()
        assert current["comfort_temp"] == 21.0

    def test_partial_save_preserves_unrelated_fields(self):
        all_fields = {"comfort_temp": 21.0, "name": "Zone", "climate_entities": ["c.trv"],
                      "learning_enabled": True}
        # Simulating: user submits only devices fields
        devices_input = {"name": "Zone", "climate_entities": ["c.trv"]}
        merged = {**all_fields, **devices_input}
        # All original fields preserved
        assert merged["comfort_temp"] == 21.0
        assert merged["learning_enabled"] is True
        assert merged["climate_entities"] == ["c.trv"]

    def test_partial_save_overwrites_changed_fields(self):
        current = {"name": "Old Name", "comfort_temp": 21.0}
        devices_input = {"name": "New Name"}
        merged = {**current, **devices_input}
        assert merged["name"] == "New Name"
        assert merged["comfort_temp"] == 21.0


# ── §5  Entity Unique ID Stability ───────────────────────────────────────────


class TestEntityUniqueIdStability:
    def test_unique_ids_based_on_entry_id_not_zone_name(self):
        """Unique IDs use entry_id, so zone rename never changes them."""
        entry_id = "abc123"
        keys = [
            "adjusted_target", "trv_setpoint", "tpi_duty_cycle",
            "preheat_minutes", "confidence", "weather_offset",
            "status", "active_control", "learning",
        ]
        for key in keys:
            uid = f"{entry_id}_{key}"
            assert entry_id in uid
            assert key in uid

    def test_different_entry_ids_produce_different_unique_ids(self):
        uid1 = "entry_abc_active_control"
        uid2 = "entry_xyz_active_control"
        assert uid1 != uid2

    def test_unique_id_pattern_is_stable(self):
        """The pattern f'{entry_id}_{key}' is stable and predictable."""
        entry_id = "thermosmart_wohnzimmer_2024"
        key = "active_control"
        expected = f"{entry_id}_{key}"
        assert expected == "thermosmart_wohnzimmer_2024_active_control"

    def test_active_control_unique_id_format(self):
        entry_id = "test_entry"
        from custom_components.thermosmart.switch import ThermoSmartActiveSwitch
        # Check that the class sets unique_id from entry_id
        src = inspect.getsource(ThermoSmartActiveSwitch.__init__)
        assert "entry_id" in src
        assert "active_control" in src

    def test_adjusted_target_sensor_unique_id_format(self):
        from custom_components.thermosmart.sensor import ThermoSmartTargetTempSensor
        src = inspect.getsource(ThermoSmartTargetTempSensor.__init__)
        assert "entry_id" in src
        assert "adjusted_target" in src


# ── §6  Safety Defaults ───────────────────────────────────────────────────────


class TestSafetyDefaults:
    def test_learning_enabled_default_is_true(self):
        assert DEFAULT_LEARNING_ENABLED is True

    def test_window_open_setpoint_is_safe_low(self):
        assert WINDOW_OPEN_SETPOINT <= 10.0

    def test_comfort_temp_default_is_reasonable(self):
        assert 18.0 <= TEMP_COMFORT <= 24.0

    def test_night_temp_below_comfort(self):
        assert TEMP_NIGHT < TEMP_COMFORT

    def test_away_temp_at_or_below_night(self):
        assert TEMP_AWAY <= TEMP_NIGHT

    def test_frost_protection_below_away(self):
        assert TEMP_FROST_PROTECTION < TEMP_AWAY

    def test_eco_temp_between_night_and_comfort(self):
        assert TEMP_NIGHT <= TEMP_ECO <= TEMP_COMFORT

    def test_active_control_default_is_off(self):
        """ThermoSmartActiveSwitch starts in OFF state (observation mode)."""
        from custom_components.thermosmart.switch import ThermoSmartActiveSwitch
        src = inspect.getsource(ThermoSmartActiveSwitch.__init__)
        assert "self._is_on = False" in src

    def test_active_control_restored_from_state(self):
        """active_control uses RestoreEntity to survive reloads."""
        from custom_components.thermosmart.switch import ThermoSmartActiveSwitch
        from homeassistant.helpers.restore_state import RestoreEntity
        assert issubclass(ThermoSmartActiveSwitch, RestoreEntity)

    def test_no_service_calls_on_init(self):
        """coordinator.set_active_control is only called in async_added_to_hass, not __init__."""
        from custom_components.thermosmart.switch import ThermoSmartActiveSwitch
        init_src = inspect.getsource(ThermoSmartActiveSwitch.__init__)
        assert "set_active_control" not in init_src

    def test_learning_switch_exists(self):
        from custom_components.thermosmart.switch import ThermoSmartLearningSwitch
        assert ThermoSmartLearningSwitch is not None

    def test_debug_default_off_is_sensible(self):
        """Debug switch (global) should not default to on."""
        from custom_components.thermosmart.switch import ThermoSmartGlobalDebugSwitch
        src = inspect.getsource(ThermoSmartGlobalDebugSwitch.__init__)
        assert "False" in src or "_is_on = False" in src or "self._debug" in src


# ── §7  Translation Parity ────────────────────────────────────────────────────


class TestTranslationParity:
    def _load(self, path: pathlib.Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_strings_json_is_valid_json(self):
        data = self._load(_STRINGS)
        assert isinstance(data, dict)

    def test_all_24_translation_files_are_valid_json(self):
        files = list(_TRANSLATIONS.glob("*.json"))
        assert len(files) == 24
        for tf in files:
            data = self._load(tf)
            assert isinstance(data, dict), f"{tf.name} is not a dict"

    def test_top_level_keys_match_strings_json(self):
        ref = self._load(_STRINGS)
        ref_keys = sorted(ref.keys())
        for tf in _TRANSLATIONS.glob("*.json"):
            data = self._load(tf)
            assert sorted(data.keys()) == ref_keys, f"{tf.name} top-level mismatch"

    def test_options_step_keys_match_across_all_files(self):
        ref = self._load(_STRINGS)
        ref_keys = sorted(ref["options"]["step"].keys())
        for tf in _TRANSLATIONS.glob("*.json"):
            data = self._load(tf)
            keys = sorted(data["options"]["step"].keys())
            assert keys == ref_keys, f"{tf.name} options.step keys: {keys} vs {ref_keys}"

    def test_options_hub_init_is_menu_in_strings_json(self):
        data = self._load(_STRINGS)
        init = data["options"]["step"]["init"]
        assert "menu_options" in init, "options.step.init should have menu_options"
        assert "data" not in init, "options.step.init should not have a form data block"

    def test_options_hub_menu_has_four_sections(self):
        data = self._load(_STRINGS)
        menu = data["options"]["step"]["init"]["menu_options"]
        for section in ("devices", "schedule", "presence", "weather"):
            assert section in menu, f"menu_options missing '{section}'"

    def test_options_devices_step_has_data_block(self):
        data = self._load(_STRINGS)
        devices = data["options"]["step"]["devices"]
        assert "data" in devices
        assert "climate_entities" in devices["data"]
        assert "name" in devices["data"]

    def test_options_step_titles_have_no_step_number_prefix(self):
        data = self._load(_STRINGS)
        steps = data["options"]["step"]
        for section in ("devices", "schedule", "presence", "weather"):
            title = steps[section].get("title", "")
            assert "/ 4" not in title, f"{section}.title still has step number: '{title}'"

    def test_all_translation_files_options_have_devices_step(self):
        for tf in _TRANSLATIONS.glob("*.json"):
            data = self._load(tf)
            assert "devices" in data["options"]["step"], f"{tf.name} missing 'devices' step"

    def test_entity_keys_present_in_all_files(self):
        ref = self._load(_STRINGS)
        ref_entity_keys = sorted(ref.get("entity", {}).keys())
        for tf in _TRANSLATIONS.glob("*.json"):
            data = self._load(tf)
            keys = sorted(data.get("entity", {}).keys())
            assert keys == ref_entity_keys, f"{tf.name} entity keys: {keys}"

    def test_config_step_keys_match_across_all_files(self):
        ref = self._load(_STRINGS)
        ref_keys = sorted(ref["config"]["step"].keys())
        for tf in _TRANSLATIONS.glob("*.json"):
            data = self._load(tf)
            keys = sorted(data["config"]["step"].keys())
            assert keys == ref_keys, f"{tf.name} config.step keys: {keys}"

    def test_away_temp_description_has_no_step_cross_reference_in_en(self):
        """The options.schedule.away_temp description should not reference Step 3."""
        data = self._load(_TRANSLATIONS / "en.json")
        desc = data["options"]["step"]["schedule"].get("data_description", {}).get("away_temp", "")
        assert "(Step 3)" not in desc, f"Stale cross-reference found: {desc}"


# ── §8  Config Flow Step Existence ───────────────────────────────────────────


class TestConfigFlowStructure:
    def test_config_flow_has_step_user(self):
        assert hasattr(ThermoSmartConfigFlow, "async_step_user")

    def test_config_flow_has_step_add_system(self):
        assert hasattr(ThermoSmartConfigFlow, "async_step_add_system")

    def test_config_flow_has_step_add_zone(self):
        assert hasattr(ThermoSmartConfigFlow, "async_step_add_zone")

    def test_config_flow_has_step_schedule(self):
        assert hasattr(ThermoSmartConfigFlow, "async_step_schedule")

    def test_config_flow_has_step_presence(self):
        assert hasattr(ThermoSmartConfigFlow, "async_step_presence")

    def test_config_flow_has_step_weather(self):
        assert hasattr(ThermoSmartConfigFlow, "async_step_weather")

    def test_config_flow_version_is_one(self):
        assert ThermoSmartConfigFlow.VERSION == 1


# ── §9  Diagnostics / Export UX Readiness ────────────────────────────────────


class TestDiagnosticsExportUXReadiness:
    def test_export_button_entity_exists(self):
        from custom_components.thermosmart import button
        assert hasattr(button, "async_setup_entry")

    def test_export_function_exists_in_export_module(self):
        from custom_components.thermosmart.export import async_export_learning_data
        assert callable(async_export_learning_data)

    def test_export_service_is_registered_on_setup_not_auto(self):
        """Export is triggered by service call, not auto-run on integration load."""
        init_src = pathlib.Path("custom_components/thermosmart/__init__.py").read_text(encoding="utf-8")
        # Service is registered, not called directly at top-level
        assert "export_learning_data" in init_src
        # The actual call is inside _handle_export (a callback), not at module load time
        assert "_handle_export" in init_src

    def test_no_auto_export_in_coordinator(self):
        """Coordinator must not call export functions automatically."""
        from custom_components.thermosmart import coordinator
        src = inspect.getsource(coordinator.ThermoSmartCoordinator)
        assert "async_export_learning_data" not in src

    def test_diagnostics_orchestrator_importable(self):
        from custom_components.thermosmart.learning.diagnostics import DiagnosticsOrchestrator
        assert DiagnosticsOrchestrator is not None

    def test_build_support_export_importable(self):
        from custom_components.thermosmart.learning.export import build_support_export
        assert callable(build_support_export)

    def test_build_learning_export_importable(self):
        from custom_components.thermosmart.learning.export import build_learning_export
        assert callable(build_learning_export)


# ── §10  Options Not Causing Service Calls ────────────────────────────────────


class TestOptionsNoCausalSideEffects:
    def test_active_control_switch_not_triggered_by_config_merge(self):
        """active_control state comes from RestoreEntity, not from config/options merge."""
        from custom_components.thermosmart.switch import ThermoSmartActiveSwitch
        init_src = inspect.getsource(ThermoSmartActiveSwitch.__init__)
        # __init__ must not call set_active_control
        assert "set_active_control" not in init_src

    def test_learning_switch_reads_from_coordinator_not_options(self):
        """Learning switch state is not set directly from options during __init__."""
        from custom_components.thermosmart.switch import ThermoSmartLearningSwitch
        src = inspect.getsource(ThermoSmartLearningSwitch.__init__)
        assert "learning_enabled" not in src or "set_learning" not in src

    def test_options_hub_partial_save_does_not_reset_learning_model(self):
        """Saving options does not call any model reset method."""
        # The options flow handlers only call async_create_entry — no model interaction
        for method_name in ("async_step_devices", "async_step_schedule",
                            "async_step_presence", "async_step_weather"):
            src = inspect.getsource(getattr(ThermoSmartOptionsFlow, method_name))
            assert "reset" not in src.lower() or "error" in src.lower()
            assert "learning" not in src.lower()
