"""The learning ON/OFF semantics, with the coordinator as the single owner (S7.2).

Effective learning = config ``learning_enabled`` AND the user's Learning
switch. The coordinator owns the switch state (``set_learning_switch`` /
``_learning_switch_on``); the switch entity, integration setup and the status
sensor all read it through the coordinator — no legacy engine flag.

The status sensor now reflects the *effective* state: with config OFF and the
switch ON it reports learning as not controlling, matching
``_learning_mode_enabled`` (this corrects the pre-S7.2 mismatch).

Restore ordering (switch restored before the first refresh, active control
after) is covered by test_learning_switch_restore_ordering.py.
"""
from __future__ import annotations

import pytest

from tests.helpers import make_coordinator


# ── coordinator._learning_mode_enabled ──────────────────────────────────────

class TestLearningModeEnabled:
    def test_config_off_is_off_regardless_of_switch(self):
        c = make_coordinator({"learning_enabled": False})
        c.set_learning_switch(True)
        assert c._learning_mode_enabled(c.zone_cfg) is False

    def test_config_on_switch_on_is_on(self):
        c = make_coordinator({"learning_enabled": True})
        c.set_learning_switch(True)
        assert c._learning_mode_enabled(c.zone_cfg) is True

    def test_config_on_switch_off_is_off(self):
        c = make_coordinator({"learning_enabled": True})
        c.set_learning_switch(False)
        assert c._learning_mode_enabled(c.zone_cfg) is False

    def test_defaults_on_without_an_explicit_switch_set(self):
        c = make_coordinator({"learning_enabled": True})
        assert c._learning_switch_on is True
        assert c._learning_mode_enabled(c.zone_cfg) is True


# ── status sensor native_value (learning gate) ──────────────────────────────

def _status_sensor(coord):
    from custom_components.thermosmart.sensor import ThermoSmartStatusSensor
    return ThermoSmartStatusSensor(coord, coord.entry)


def _prime_zone(coord, **zone):
    base = {"current_temp": 21.0, "effective_target": 21.0, "adjusted_target": 21.0,
            "mode": "auto"}
    base.update(zone)
    coord.data = {"zone": base}


class TestStatusSensorLearningGate:
    def test_controlling_no_learning_when_switch_off(self):
        c = make_coordinator({"learning_enabled": True})
        c.set_learning_switch(False)
        c._active_control = True
        _prime_zone(c)
        assert _status_sensor(c).native_value == "controlling_no_learning"

    def test_observation_mode_when_not_active(self):
        c = make_coordinator({"learning_enabled": True})
        c.set_learning_switch(True)
        c._active_control = False
        _prime_zone(c)
        assert _status_sensor(c).native_value == "observation_mode"

    def test_disabled_when_not_active_and_switch_off(self):
        c = make_coordinator({"learning_enabled": True})
        c.set_learning_switch(False)
        c._active_control = False
        _prime_zone(c)
        assert _status_sensor(c).native_value == "disabled"

    def test_config_off_switch_on_reports_controlling_no_learning(self):
        # S7.2 fix: the sensor reads the effective state, so config OFF +
        # switch ON controlling → 'controlling_no_learning', matching
        # _learning_mode_enabled (which is False).
        c = make_coordinator({"learning_enabled": False})
        c.set_learning_switch(True)
        c._active_control = True
        _prime_zone(c)
        assert c._learning_mode_enabled(c.zone_cfg) is False
        assert _status_sensor(c).native_value == "controlling_no_learning"
