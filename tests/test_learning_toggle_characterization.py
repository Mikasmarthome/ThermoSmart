"""Characterization of the current learning ON/OFF semantics (S7.0).

Pins the effective-learning logic and the status-sensor output exactly as they
are today, before the toggle owner moves from LE v1 to the coordinator (S7.2).

Today the effective state is: config ``learning_enabled`` AND the restored/user
switch state, where the switch state lives in the LearningEngine
(``is_zone_enabled``). ``coordinator._learning_mode_enabled`` combines both, but
the status sensor reads ``is_zone_enabled`` *directly* — so with config OFF and
the switch ON it still reports learning as active. That mismatch is the
BUG(S7.2) pinned here and corrected, with an explicit test, in the ownership
commit.

Restore ordering (switch restored before the first refresh, active control
after) is already covered by test_learning_switch_restore_ordering.py.
"""
from __future__ import annotations

import pytest

from tests.helpers import make_coordinator


# ── coordinator._learning_mode_enabled ──────────────────────────────────────

class TestLearningModeEnabled:
    def test_config_off_is_off_regardless_of_switch(self):
        c = make_coordinator({"learning_enabled": False})
        c.learning_engine.is_zone_enabled.return_value = True
        assert c._learning_mode_enabled(c.zone_cfg) is False

    def test_config_on_switch_on_is_on(self):
        c = make_coordinator({"learning_enabled": True})
        c.learning_engine.is_zone_enabled.return_value = True
        assert c._learning_mode_enabled(c.zone_cfg) is True

    def test_config_on_switch_off_is_off(self):
        c = make_coordinator({"learning_enabled": True})
        c.learning_engine.is_zone_enabled.return_value = False
        assert c._learning_mode_enabled(c.zone_cfg) is False

    def test_no_engine_defaults_on(self):
        c = make_coordinator({"learning_enabled": True})
        c.learning_engine = None
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
        c.learning_engine.is_zone_enabled.return_value = False
        c._active_control = True
        _prime_zone(c)
        assert _status_sensor(c).native_value == "controlling_no_learning"

    def test_observation_mode_when_not_active(self):
        c = make_coordinator({"learning_enabled": True})
        c.learning_engine.is_zone_enabled.return_value = True
        c._active_control = False
        _prime_zone(c)
        assert _status_sensor(c).native_value == "observation_mode"

    def test_disabled_when_not_active_and_switch_off(self):
        c = make_coordinator({"learning_enabled": True})
        c.learning_engine.is_zone_enabled.return_value = False
        c._active_control = False
        _prime_zone(c)
        assert _status_sensor(c).native_value == "disabled"

    def test_config_off_switch_on_still_reports_learning_active_bug(self):
        # BUG(S7.2): the sensor reads is_zone_enabled directly, ignoring the
        # config flag, so effective learning is OFF yet the status is NOT
        # 'controlling_no_learning'. _learning_mode_enabled already says False.
        c = make_coordinator({"learning_enabled": False})
        c.learning_engine.is_zone_enabled.return_value = True
        c._active_control = True
        _prime_zone(c, current_temp=21.0, effective_target=21.0)
        assert c._learning_mode_enabled(c.zone_cfg) is False
        assert _status_sensor(c).native_value != "controlling_no_learning"
