"""Unit tests for ThermoSmart summer mode detection (Welle 3).

Tests _update_summer_mode (SeasonMixin) and set_summer_override behavior.
The indoor safety bypass (v1.1.1 fix) in _async_update_data is tested in
test_coordinator_integration.py because it requires the full update loop.

Pure Python + MagicMock – runs on Windows without Docker.
"""
from __future__ import annotations

from custom_components.thermosmart.const import SUMMER_THRESHOLD, WINTER_THRESHOLD

from tests.helpers import make_coordinator, make_weather_data


# ── TestUpdateSummerMode ──────────────────────────────────────────────────────


class TestUpdateSummerMode:
    def test_no_outdoor_temp_leaves_is_summer_unchanged(self):
        coord = make_coordinator()
        coord._is_summer = False
        coord._update_summer_mode({})  # no "temperature" key
        assert coord._is_summer is False

    def test_none_outdoor_temp_leaves_is_summer_unchanged(self):
        coord = make_coordinator()
        coord._is_summer = False
        coord._update_summer_mode({"temperature": None})
        assert coord._is_summer is False

    def test_single_sample_does_not_trigger_threshold(self):
        """< 3 samples → no state change even if temp is above SUMMER_THRESHOLD."""
        coord = make_coordinator()
        coord._is_summer = False
        coord._update_summer_mode({"temperature": 25.0})
        assert coord._is_summer is False

    def test_two_samples_do_not_trigger_threshold(self):
        coord = make_coordinator()
        coord._is_summer = False
        coord._update_summer_mode({"temperature": 25.0})
        coord._update_summer_mode({"temperature": 25.0})
        assert coord._is_summer is False

    def test_three_samples_above_summer_threshold_sets_summer(self):
        coord = make_coordinator()
        for _ in range(3):
            coord._update_summer_mode({"temperature": SUMMER_THRESHOLD + 1.0})
        assert coord._is_summer is True

    def test_three_samples_at_summer_threshold_sets_summer(self):
        """avg == SUMMER_THRESHOLD (not just >) still triggers summer."""
        coord = make_coordinator()
        for _ in range(3):
            coord._update_summer_mode({"temperature": SUMMER_THRESHOLD})
        assert coord._is_summer is True

    def test_three_samples_below_winter_threshold_clears_summer(self):
        coord = make_coordinator()
        coord._is_summer = True
        for _ in range(3):
            coord._update_summer_mode({"temperature": WINTER_THRESHOLD - 1.0})
        assert coord._is_summer is False

    def test_three_samples_at_winter_threshold_clears_summer(self):
        """avg == WINTER_THRESHOLD still triggers winter (≤ check)."""
        coord = make_coordinator()
        coord._is_summer = True
        for _ in range(3):
            coord._update_summer_mode({"temperature": WINTER_THRESHOLD})
        assert coord._is_summer is False

    def test_avg_between_thresholds_no_state_change(self):
        """avg between WINTER_THRESHOLD and SUMMER_THRESHOLD → no change."""
        coord = make_coordinator()
        between = (SUMMER_THRESHOLD + WINTER_THRESHOLD) / 2
        for _ in range(3):
            coord._update_summer_mode({"temperature": between})
        assert coord._is_summer is False  # started False, stays False

    def test_already_summer_between_thresholds_stays_summer(self):
        """Hysteresis: once summer, stays summer until avg drops to WINTER_THRESHOLD."""
        coord = make_coordinator()
        coord._is_summer = True
        between = (SUMMER_THRESHOLD + WINTER_THRESHOLD) / 2
        for _ in range(5):
            coord._update_summer_mode({"temperature": between})
        assert coord._is_summer is True  # no change — between thresholds

    def test_summer_override_blocks_detection(self):
        """When _summer_override is not None, _update_summer_mode returns early."""
        coord = make_coordinator()
        coord._summer_override = True  # override active
        coord._is_summer = False
        for _ in range(5):
            coord._update_summer_mode({"temperature": SUMMER_THRESHOLD + 5.0})
        assert coord._is_summer is False  # override prevented automatic detection

    def test_winter_override_blocks_detection(self):
        coord = make_coordinator()
        coord._summer_override = False
        coord._is_summer = True
        for _ in range(5):
            coord._update_summer_mode({"temperature": SUMMER_THRESHOLD + 5.0})
        assert coord._is_summer is True  # override prevented automatic clearing

    def test_summer_transition_clears_manual_override(self):
        """When season transitions to summer, any manual override is cleared."""
        coord = make_coordinator()
        coord._is_summer = False
        coord._override = 22.0
        coord._override_schedule_period = "wd_comfort"
        for _ in range(3):
            coord._update_summer_mode({"temperature": SUMMER_THRESHOLD + 2.0})
        assert coord._is_summer is True
        assert coord._override is None

    def test_winter_transition_does_not_clear_override(self):
        """Winter transition does NOT clear manual override (only summer does)."""
        coord = make_coordinator()
        coord._is_summer = True
        coord._override = 22.0
        for _ in range(3):
            coord._update_summer_mode({"temperature": WINTER_THRESHOLD - 2.0})
        assert coord._is_summer is False
        assert coord._override == 22.0  # override preserved on winter transition

    def test_samples_accumulate_across_calls(self):
        """History deque accumulates samples: avg is calculated over all retained values."""
        coord = make_coordinator()
        # Fill with cold samples
        for _ in range(100):
            coord._update_summer_mode({"temperature": 5.0})
        assert coord._is_summer is False
        # Add a few hot samples — average still low
        for _ in range(3):
            coord._update_summer_mode({"temperature": 25.0})
        # Deque has 864 slots max; the few hot samples don't move the average enough
        assert coord._is_summer is False


# ── TestSetSummerOverride – integration with _update_summer_mode ──────────────


class TestSummerOverrideInteraction:
    def test_set_override_none_re_enables_auto_detection(self):
        """After set_summer_override(None), _update_summer_mode runs again."""
        coord = make_coordinator()
        coord.set_summer_override(True)       # override active
        coord.set_summer_override(None)       # back to auto
        # Now auto detection can proceed
        assert coord._summer_override is None
        for _ in range(3):
            coord._update_summer_mode({"temperature": WINTER_THRESHOLD - 2.0})
        assert coord._is_summer is False  # override was True before, now auto cleared it

    def test_outdoor_history_not_appended_while_override_active(self):
        """While override is set, outdoor_temp_history stays empty."""
        coord = make_coordinator()
        coord.set_summer_override(True)
        for _ in range(5):
            coord._update_summer_mode({"temperature": 25.0})
        assert len(coord._outdoor_temp_history) == 0
