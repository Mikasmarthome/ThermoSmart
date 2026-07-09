"""Unit tests for ThermoSmart Coordinator mode logic (Welle 3).

Tests _effective_mode, _control_reason, set_override, set_mode,
set_vacation_override, set_summer_override, and _is_person_home.

Pure Python + MagicMock – runs on Windows without Docker.
No HA fixtures needed (uses MagicMock hass).
"""
from __future__ import annotations

from custom_components.thermosmart.const import (
    HEATING_MODE_AUTO,
    HEATING_MODE_AWAY,
    HEATING_MODE_COMFORT,
    HEATING_MODE_ECO,
    HEATING_MODE_NIGHT,
    HEATING_MODE_VACATION,
)

from tests.helpers import make_coordinator, make_state


# ── TestEffectiveMode ─────────────────────────────────────────────────────────


class TestEffectiveMode:
    def _presence(self, vacation=False, all_away=False):
        return {"vacation": vacation, "all_away": all_away, "persons_home": []}

    def test_vacation_presence_returns_vacation_mode(self):
        coord = make_coordinator()
        result = coord._effective_mode(self._presence(vacation=True))
        assert result == HEATING_MODE_VACATION

    def test_vacation_overrides_manual_mode(self):
        """Vacation presence flag beats any manually set mode."""
        coord = make_coordinator()
        coord._mode = HEATING_MODE_COMFORT
        result = coord._effective_mode(self._presence(vacation=True))
        assert result == HEATING_MODE_VACATION

    def test_manual_comfort_mode_returned(self):
        coord = make_coordinator()
        coord._mode = HEATING_MODE_COMFORT
        result = coord._effective_mode(self._presence())
        assert result == HEATING_MODE_COMFORT

    def test_manual_night_mode_returned(self):
        coord = make_coordinator()
        coord._mode = HEATING_MODE_NIGHT
        result = coord._effective_mode(self._presence())
        assert result == HEATING_MODE_NIGHT

    def test_manual_mode_beats_all_away(self):
        """Explicit mode (non-AUTO) takes priority over presence-away."""
        coord = make_coordinator()
        coord._mode = HEATING_MODE_COMFORT
        result = coord._effective_mode(self._presence(all_away=True))
        assert result == HEATING_MODE_COMFORT

    def test_auto_all_away_returns_away(self):
        coord = make_coordinator()
        coord._mode = HEATING_MODE_AUTO
        result = coord._effective_mode(self._presence(all_away=True))
        assert result == HEATING_MODE_AWAY

    def test_auto_no_presence_returns_auto(self):
        coord = make_coordinator()
        coord._mode = HEATING_MODE_AUTO
        result = coord._effective_mode(self._presence())
        assert result == HEATING_MODE_AUTO


# ── TestControlReason ─────────────────────────────────────────────────────────


class TestControlReason:
    def _rec(self, **kwargs) -> dict:
        defaults = {
            "window_open": False,
            "override_active": False,
            "mode": HEATING_MODE_AUTO,
            "is_summer": False,
        }
        return {**defaults, **kwargs}

    def test_window_open_highest_priority(self):
        rec = self._rec(window_open=True, override_active=True)
        assert make_coordinator()._control_reason(rec) == "window_open"

    def test_override_active_second_priority(self):
        rec = self._rec(override_active=True, mode=HEATING_MODE_VACATION)
        assert make_coordinator()._control_reason(rec) == "manual_override"

    def test_vacation_mode_third_priority(self):
        rec = self._rec(mode=HEATING_MODE_VACATION)
        assert make_coordinator()._control_reason(rec) == "vacation"

    def test_summer_mode_fourth_priority(self):
        rec = self._rec(is_summer=True)
        assert make_coordinator()._control_reason(rec) == "summer_mode"

    def test_away_mode_fifth_priority(self):
        rec = self._rec(mode=HEATING_MODE_AWAY)
        assert make_coordinator()._control_reason(rec) == "presence"

    def test_default_is_schedule(self):
        rec = self._rec()
        assert make_coordinator()._control_reason(rec) == "schedule"


# ── TestSetOverride ───────────────────────────────────────────────────────────


class TestSetOverride:
    def test_value_above_threshold_sets_override(self):
        coord = make_coordinator()
        coord.set_override(22.0)
        assert coord._override == 22.0

    def test_value_at_threshold_boundary_sets_override(self):
        coord = make_coordinator()
        coord.set_override(5.0)
        assert coord._override == 5.0

    def test_value_below_threshold_clears_override(self):
        coord = make_coordinator()
        coord._override = 20.0
        coord.set_override(4.9)
        assert coord._override is None

    def test_zero_value_clears_override(self):
        coord = make_coordinator()
        coord._override = 20.0
        coord.set_override(0.0)
        assert coord._override is None


# ── TestSetMode ───────────────────────────────────────────────────────────────


class TestSetMode:
    def test_mode_change_clears_manual_override(self):
        coord = make_coordinator()
        coord._mode = HEATING_MODE_AUTO
        coord._override = 22.0
        coord._override_schedule_period = "wd_comfort"
        coord.set_mode(HEATING_MODE_COMFORT)
        assert coord._override is None
        assert coord._override_schedule_period is None

    def test_same_mode_preserves_manual_override(self):
        coord = make_coordinator()
        coord._mode = HEATING_MODE_COMFORT
        coord._override = 22.0
        coord.set_mode(HEATING_MODE_COMFORT)
        assert coord._override == 22.0

    def test_set_mode_updates_internal_mode(self):
        coord = make_coordinator()
        coord.set_mode(HEATING_MODE_ECO)
        assert coord._mode == HEATING_MODE_ECO


# ── TestSetVacationOverride ───────────────────────────────────────────────────


class TestSetVacationOverride:
    def test_activate_sets_vacation_mode(self):
        coord = make_coordinator()
        coord.set_vacation_override(True)
        assert coord._mode == HEATING_MODE_VACATION

    def test_activate_saves_previous_mode_as_pre_vacation(self):
        coord = make_coordinator()
        coord._mode = HEATING_MODE_COMFORT
        coord.set_vacation_override(True)
        assert coord._pre_vacation_mode == HEATING_MODE_COMFORT

    def test_deactivate_restores_pre_vacation_mode(self):
        coord = make_coordinator()
        coord._mode = HEATING_MODE_NIGHT
        coord.set_vacation_override(True)
        coord.set_vacation_override(False)
        assert coord._mode == HEATING_MODE_NIGHT

    def test_deactivate_without_pre_vacation_mode_restores_auto(self):
        coord = make_coordinator()
        coord._pre_vacation_mode = None
        coord._mode = HEATING_MODE_VACATION
        coord.set_vacation_override(False)
        assert coord._mode == HEATING_MODE_AUTO

    def test_activate_clears_manual_override(self):
        coord = make_coordinator()
        coord._override = 22.0
        coord._override_schedule_period = "wd_comfort"
        coord.set_vacation_override(True)
        assert coord._override is None

    def test_double_activate_does_not_overwrite_pre_vacation_mode(self):
        """Second activate keeps the original pre_vacation_mode (not VACATION)."""
        coord = make_coordinator()
        coord._mode = HEATING_MODE_NIGHT
        coord.set_vacation_override(True)   # saves NIGHT
        coord.set_vacation_override(True)   # mode is already VACATION → not saved again
        assert coord._pre_vacation_mode == HEATING_MODE_NIGHT


# ── TestSetSummerOverride ─────────────────────────────────────────────────────


class TestSetSummerOverride:
    def test_true_forces_is_summer(self):
        coord = make_coordinator()
        coord._is_summer = False
        coord.set_summer_override(True)
        assert coord._is_summer is True

    def test_false_forces_is_summer_off(self):
        coord = make_coordinator()
        coord._is_summer = True
        coord.set_summer_override(False)
        assert coord._is_summer is False

    def test_none_leaves_is_summer_unchanged(self):
        coord = make_coordinator()
        coord._is_summer = True
        coord.set_summer_override(None)
        assert coord._is_summer is True

    def test_true_clears_manual_override(self):
        coord = make_coordinator()
        coord._override = 22.0
        coord.set_summer_override(True)
        assert coord._override is None

    def test_sets_summer_override_attribute(self):
        coord = make_coordinator()
        coord.set_summer_override(True)
        assert coord._summer_override is True
        coord.set_summer_override(None)
        assert coord._summer_override is None


# ── TestIsPersonHome ──────────────────────────────────────────────────────────


class TestIsPersonHome:
    def test_home_state_with_default_zone_returns_true(self):
        coord = make_coordinator()
        assert coord._is_person_home("home", "zone.home") is True

    def test_away_state_with_default_zone_returns_false(self):
        coord = make_coordinator()
        assert coord._is_person_home("work", "zone.home") is False

    def test_person_state_matches_zone_slug(self):
        """person_state == slug of zone_entity (without 'zone.' prefix) → home."""
        coord = make_coordinator()
        assert coord._is_person_home("office", "zone.office") is True

    def test_empty_zone_entity_treats_home_as_home(self):
        coord = make_coordinator()
        assert coord._is_person_home("home", "") is True

    def test_person_away_non_matching_zone(self):
        coord = make_coordinator()
        assert coord._is_person_home("work", "zone.office") is False

    def test_friendly_name_match(self):
        """person_state matching zone friendly_name → home."""
        coord = make_coordinator()
        coord.hass.states.get.side_effect = lambda eid: (
            make_state("20", attributes={"friendly_name": "Büro"})
            if eid == "zone.office" else None
        )
        assert coord._is_person_home("büro", "zone.office") is True
