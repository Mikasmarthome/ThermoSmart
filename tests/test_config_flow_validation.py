"""Tests for config flow validation helpers.

_validate_temps() is a pure Python function — no HA fixtures required.
_NullableEntitySelector short-circuit paths (None / "") are also pure.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.config_flow import (
    _validate_temps,
    _NullableEntitySelector,
)
from homeassistant.helpers import selector


# ── _validate_temps ──────────────────────────────────────────────────────────

class TestValidateTemps:
    def test_valid_default_config_no_errors(self):
        """comfort=21, night=18, away=17 → typical config, no errors."""
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0})
        assert errors == {}

    def test_night_strictly_below_comfort_is_valid(self):
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 20.9, "away_temp": 17.0})
        assert "night_temp" not in errors

    def test_night_equal_to_comfort_is_valid(self):
        """night == comfort: the code uses strict '>' so equality is accepted.

        While operationally meaningless, the validator only rejects night > comfort.
        This test documents the actual behavior — adjust if the rule is tightened.
        """
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 21.0, "away_temp": 17.0})
        assert "night_temp" not in errors

    def test_night_above_comfort_is_invalid(self):
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 22.0, "away_temp": 17.0})
        assert "night_temp" in errors
        assert errors["night_temp"] == "night_temp_too_high"

    def test_away_equal_to_night_is_valid(self):
        """away == night is allowed — both are reduced presence modes."""
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 18.0})
        assert "away_temp" not in errors

    def test_away_below_night_is_valid(self):
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 15.0})
        assert errors == {}

    def test_away_above_night_is_invalid(self):
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 19.0})
        assert "away_temp" in errors
        assert errors["away_temp"] == "away_temp_too_high"

    def test_both_errors_simultaneously(self):
        """Multiple violations each produce their own error key."""
        errors = _validate_temps({"comfort_temp": 20.0, "night_temp": 21.0, "away_temp": 22.0})
        assert "night_temp" in errors
        assert "away_temp" in errors

    def test_empty_dict_uses_code_defaults_no_crash(self):
        """Missing keys fall back to code defaults — must not raise."""
        errors = _validate_temps({})
        assert isinstance(errors, dict)

    def test_non_numeric_values_handled_gracefully(self):
        """Non-numeric values are a schema-level violation; validate_temps must not crash."""
        errors = _validate_temps({
            "comfort_temp": "not_a_number",
            "night_temp": 18.0,
            "away_temp": 17.0,
        })
        assert isinstance(errors, dict)

    def test_float_boundary_night_just_below_comfort(self):
        """Floating point boundary: 20.9 < 21.0 → no error."""
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 20.9, "away_temp": 17.0})
        assert "night_temp" not in errors

    def test_returns_dict(self):
        result = _validate_temps({"comfort_temp": 21.0, "night_temp": 18.0, "away_temp": 17.0})
        assert isinstance(result, dict)

    def test_error_keys_are_strings(self):
        """All returned error keys and values must be strings."""
        errors = _validate_temps({"comfort_temp": 21.0, "night_temp": 22.0, "away_temp": 17.0})
        for k, v in errors.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_extreme_valid_temps(self):
        """Extreme but valid temperatures should produce no errors."""
        errors = _validate_temps({"comfort_temp": 30.0, "night_temp": 5.0, "away_temp": 5.0})
        assert errors == {}


# ── _NullableEntitySelector ──────────────────────────────────────────────────

class TestNullableEntitySelector:
    """Test the None/empty short-circuit in _NullableEntitySelector.

    Only the None and empty-string paths are tested here since they bypass
    the HA entity_id validator — the valid entity path requires HA runtime state.
    """

    def _make_selector(self) -> _NullableEntitySelector:
        return _NullableEntitySelector(
            selector.EntitySelectorConfig(domain="weather")
        )

    def test_none_input_returns_none(self):
        sel = self._make_selector()
        assert sel(None) is None

    def test_empty_string_returns_none(self):
        sel = self._make_selector()
        assert sel("") is None

    def test_none_does_not_call_parent_validator(self):
        """Ensure None never reaches the HA entity_id validator (which would raise)."""
        sel = self._make_selector()
        # If super().__call__(None) were invoked it would raise TypeError/ValueError
        result = sel(None)
        assert result is None  # reached cleanly

    def test_empty_string_does_not_call_parent_validator(self):
        sel = self._make_selector()
        result = sel("")
        assert result is None
