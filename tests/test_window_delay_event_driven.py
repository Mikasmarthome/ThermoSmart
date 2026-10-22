"""Event-driven window open/close delay coverage.

Every window-related test elsewhere in the suite either sets
window_open_delay/window_close_delay to 0 (bypassing the delay mechanism
entirely) or calls _check_window_open()/_compute_recommendation() directly
without ever triggering the coordinator's real setup_event_listeners()/
_handle_state_change() callback — the actual code path a live Home Assistant
instance uses when a window sensor fires a state-change event. This file
closes that gap.

Root cause: _window_open_at/_window_close_at are pure in-memory dicts with no
persistence. Any coordinator restart/reload occurring while a window
transition is inside its configured delay window silently discards that
grace period, so the window was reported as closed immediately if a reload
happened to land inside the close_delay. The fix reconstructs the missing
dict entry from the window sensor's own last_changed timestamp (which HA
preserves independently of the integration) so the delay resumes exactly
where it left off instead of resetting to zero.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_coordinator, make_state, set_hass_states

_T0 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
_SENSOR = "binary_sensor.window"


def _state_with_last_changed(value, last_changed):
    s = make_state(value)
    s.last_changed = last_changed
    return s


def _wire_window_listener(coord):
    """Register the real setup_event_listeners() callback and return it.

    async_track_state_change_event is called once per tracked-entity group
    (window sensors, climate entities, temp sensors, ...) — capture by
    matching on the entity list rather than assuming call order/count.
    """
    captured: dict[tuple, callable] = {}

    def _fake_track(hass, entity_ids, callback_fn):
        captured[tuple(entity_ids)] = callback_fn
        return lambda: None

    with patch(
        "custom_components.thermosmart.coordinator.async_track_state_change_event",
        side_effect=_fake_track,
    ):
        coord.setup_event_listeners()

    for entities, cb in captured.items():
        if _SENSOR in entities:
            return cb
    raise AssertionError(f"no listener registered for {_SENSOR}; got {list(captured)}")


def _event(old_state, new_state):
    e = MagicMock()
    e.data = {
        "entity_id": _SENSOR,
        "old_state": MagicMock(state=old_state),
        "new_state": MagicMock(state=new_state),
    }
    return e


def _make(clock, **cfg_overrides):
    overrides = {
        "window_sensors": [_SENSOR],
        "window_open_delay": 5,
        "window_close_delay": 2,
        "temp_sensors": ["sensor.test_temp"],
    }
    overrides.update(cfg_overrides)
    coord = make_coordinator(overrides, clock=clock)
    set_hass_states(coord, {
        _SENSOR: make_state("off"),
        "sensor.test_temp": make_state("19.0"),
    })
    return coord


class TestEventDrivenOpenDelay:
    def test_open_delay_prevents_immediate_reaction(self):
        clock = FakeClock(_T0)
        coord = _make(clock)
        cb = _wire_window_listener(coord)
        cfg = coord.zone_cfg

        set_hass_states(coord, {_SENSOR: make_state("on"), "sensor.test_temp": make_state("19.0")})
        cb(_event("off", "on"))

        assert coord._check_window_open(cfg, current_temp=19.0) is False

    def test_open_delay_activates_after_configured_duration(self):
        clock = FakeClock(_T0)
        coord = _make(clock)
        cb = _wire_window_listener(coord)
        cfg = coord.zone_cfg

        set_hass_states(coord, {_SENSOR: make_state("on"), "sensor.test_temp": make_state("19.0")})
        cb(_event("off", "on"))
        clock.set_utc(clock.now_utc() + timedelta(minutes=5))

        assert coord._check_window_open(cfg, current_temp=19.0) is True


class TestEventDrivenCloseDelay:
    def test_close_delay_prevents_immediate_return_to_normal(self):
        clock = FakeClock(_T0)
        coord = _make(clock)
        cb = _wire_window_listener(coord)
        cfg = coord.zone_cfg

        set_hass_states(coord, {_SENSOR: make_state("on"), "sensor.test_temp": make_state("19.0")})
        cb(_event("off", "on"))
        clock.set_utc(clock.now_utc() + timedelta(minutes=5))
        assert coord._check_window_open(cfg, current_temp=19.0) is True

        set_hass_states(coord, {_SENSOR: make_state("off"), "sensor.test_temp": make_state("19.0")})
        cb(_event("on", "off"))

        assert coord._check_window_open(cfg, current_temp=19.0) is True, (
            "close_delay must hold the effective window state open immediately "
            "after the raw sensor reports closed"
        )

    def test_close_delay_releases_only_after_configured_duration(self):
        clock = FakeClock(_T0)
        coord = _make(clock)
        cb = _wire_window_listener(coord)
        cfg = coord.zone_cfg

        set_hass_states(coord, {_SENSOR: make_state("on"), "sensor.test_temp": make_state("19.0")})
        cb(_event("off", "on"))
        clock.set_utc(clock.now_utc() + timedelta(minutes=5))
        coord._check_window_open(cfg, current_temp=19.0)

        set_hass_states(coord, {_SENSOR: make_state("off"), "sensor.test_temp": make_state("19.0")})
        cb(_event("on", "off"))

        clock.set_utc(clock.now_utc() + timedelta(minutes=1))
        assert coord._check_window_open(cfg, current_temp=19.0) is True, "still within close_delay"

        clock.set_utc(clock.now_utc() + timedelta(minutes=2))
        assert coord._check_window_open(cfg, current_temp=19.0) is False, "close_delay elapsed"


class TestQuickOpenClose:
    def test_quick_open_close_below_open_delay_causes_no_heating_pause(self):
        clock = FakeClock(_T0)
        coord = _make(clock)
        cb = _wire_window_listener(coord)
        cfg = coord.zone_cfg

        set_hass_states(coord, {_SENSOR: make_state("on"), "sensor.test_temp": make_state("19.0")})
        cb(_event("off", "on"))
        clock.set_utc(clock.now_utc() + timedelta(minutes=1))  # well below open_delay=5min

        set_hass_states(coord, {_SENSOR: make_state("off"), "sensor.test_temp": make_state("19.0")})
        cb(_event("on", "off"))

        assert coord._check_window_open(cfg, current_temp=19.0) is False, (
            "a window that closes before open_delay elapsed must never have "
            "triggered a pause, and must not start a close_delay either"
        )


class TestMultipleWindowSensors:
    def test_one_sensor_still_in_close_delay_keeps_window_effectively_open(self):
        sensor_a, sensor_b = "binary_sensor.window_a", "binary_sensor.window_b"
        clock = FakeClock(_T0)
        coord = make_coordinator({
            "window_sensors": [sensor_a, sensor_b],
            "window_open_delay": 5,
            "window_close_delay": 2,
            "temp_sensors": ["sensor.test_temp"],
        }, clock=clock)
        set_hass_states(coord, {
            sensor_a: make_state("off"),
            sensor_b: make_state("off"),
            "sensor.test_temp": make_state("19.0"),
        })

        captured: dict[tuple, callable] = {}

        def _fake_track(hass, entity_ids, callback_fn):
            captured[tuple(entity_ids)] = callback_fn
            return lambda: None

        with patch(
            "custom_components.thermosmart.coordinator.async_track_state_change_event",
            side_effect=_fake_track,
        ):
            coord.setup_event_listeners()
        cb = next(v for k, v in captured.items() if sensor_a in k and sensor_b in k)

        def _ev(entity_id, old_state, new_state):
            e = MagicMock()
            e.data = {
                "entity_id": entity_id,
                "old_state": MagicMock(state=old_state),
                "new_state": MagicMock(state=new_state),
            }
            return e

        cfg = coord.zone_cfg

        # Both open, wait past open_delay.
        set_hass_states(coord, {sensor_a: make_state("on"), sensor_b: make_state("on"),
                                 "sensor.test_temp": make_state("19.0")})
        cb(_ev(sensor_a, "off", "on"))
        cb(_ev(sensor_b, "off", "on"))
        clock.set_utc(clock.now_utc() + timedelta(minutes=5))
        assert coord._check_window_open(cfg, current_temp=19.0) is True

        # sensor_a closes; sensor_b stays open.
        set_hass_states(coord, {sensor_a: make_state("off"), sensor_b: make_state("on"),
                                 "sensor.test_temp": make_state("19.0")})
        cb(_ev(sensor_a, "on", "off"))
        assert coord._check_window_open(cfg, current_temp=19.0) is True, "sensor_b still open"

        # sensor_b closes too, sensor_a still inside its close_delay.
        set_hass_states(coord, {sensor_a: make_state("off"), sensor_b: make_state("off"),
                                 "sensor.test_temp": make_state("19.0")})
        cb(_ev(sensor_b, "on", "off"))
        assert coord._check_window_open(cfg, current_temp=19.0) is True, (
            "sensor_a's close_delay must still hold the window effectively open"
        )

        clock.set_utc(clock.now_utc() + timedelta(minutes=3))
        assert coord._check_window_open(cfg, current_temp=19.0) is False


class TestUnknownUnavailableWindowSensor:
    @pytest.mark.parametrize("bad_state", ["unknown", "unavailable"])
    def test_unknown_or_unavailable_sensor_treated_as_closed_not_crashing(self, bad_state):
        clock = FakeClock(_T0)
        coord = _make(clock)
        set_hass_states(coord, {_SENSOR: make_state(bad_state), "sensor.test_temp": make_state("19.0")})
        cfg = coord.zone_cfg

        assert coord._check_window_open(cfg, current_temp=19.0) is False


class TestRestartDuringDelay:
    """Simulates a coordinator restart/reload (fresh in-memory dicts) while a
    real window sensor transition happened moments before, using the window
    sensor's own last_changed timestamp — the only state that survives an
    integration reload — to reconstruct the interrupted delay window."""

    def test_restart_during_open_delay_does_not_grant_free_pause(self):
        """Window opened 2min ago (open_delay=5min still pending) — a restart
        must not retroactively treat the open_delay as already elapsed."""
        clock = FakeClock(_T0)
        coord = _make(clock)
        opened_at = _T0 - timedelta(minutes=2)
        set_hass_states(coord, {
            _SENSOR: _state_with_last_changed("on", opened_at),
            "sensor.test_temp": make_state("19.0"),
        })
        cfg = coord.zone_cfg

        assert coord._check_window_open(cfg, current_temp=19.0) is False

    def test_restart_after_open_delay_already_elapsed_reports_open(self):
        clock = FakeClock(_T0)
        coord = _make(clock)
        opened_at = _T0 - timedelta(minutes=6)
        set_hass_states(coord, {
            _SENSOR: _state_with_last_changed("on", opened_at),
            "sensor.test_temp": make_state("19.0"),
        })
        cfg = coord.zone_cfg

        assert coord._check_window_open(cfg, current_temp=19.0) is True

    def test_restart_during_close_delay_preserves_remaining_grace_period(self):
        """This is the confirmed root cause: closed 1min ago (close_delay=2min
        still pending) — before the fix, a fresh coordinator's empty
        _window_close_at dict caused this to report closed immediately."""
        clock = FakeClock(_T0)
        coord = _make(clock)
        closed_at = _T0 - timedelta(minutes=1)
        set_hass_states(coord, {
            _SENSOR: _state_with_last_changed("off", closed_at),
            "sensor.test_temp": make_state("19.0"),
        })
        cfg = coord.zone_cfg

        assert coord._check_window_open(cfg, current_temp=19.0) is True

        clock.set_utc(clock.now_utc() + timedelta(minutes=2))
        assert coord._check_window_open(cfg, current_temp=19.0) is False

    def test_restart_after_close_delay_already_elapsed_reports_closed(self):
        clock = FakeClock(_T0)
        coord = _make(clock)
        closed_at = _T0 - timedelta(minutes=5)
        set_hass_states(coord, {
            _SENSOR: _state_with_last_changed("off", closed_at),
            "sensor.test_temp": make_state("19.0"),
        })
        cfg = coord.zone_cfg

        assert coord._check_window_open(cfg, current_temp=19.0) is False

    def test_restart_with_no_last_changed_info_falls_back_to_legacy_behavior(self):
        """Plain mock state (no last_changed) must behave exactly as before
        the fix — this guards against a regression for any caller that can't
        supply a real HA State object."""
        clock = FakeClock(_T0)
        coord = _make(clock)
        set_hass_states(coord, {_SENSOR: make_state("off"), "sensor.test_temp": make_state("19.0")})
        cfg = coord.zone_cfg

        assert coord._check_window_open(cfg, current_temp=19.0) is False


# ── Realistic HA states: short airing must never activate the window override ──
#
# The tests above hand the coordinator plain MagicMock states, which carry no
# ``last_changed`` datetime — so the restart-reconstruction branch of
# _check_window_open() was never exercised together with the live event
# listener. Real Home Assistant states always carry last_changed. With them, a
# window closed *before* the open delay elapsed used to be reconstructed as
# "closed less than close_delay ago" and switched the window override on for
# the close delay — although the window had never been confirmed open.

_SENSOR_B = "binary_sensor.window_b"


class _LiveWindowRig:
    """Real listener + real _check_window_open() driven by HA-like states."""

    def __init__(self, *, open_delay=5, close_delay=2, sensors=(_SENSOR,)):
        self.clock = FakeClock(_T0)
        self.coord = make_coordinator(
            {
                "window_sensors": list(sensors),
                "window_open_delay": open_delay,
                "window_close_delay": close_delay,
                "temp_sensors": ["sensor.test_temp"],
            },
            clock=self.clock,
        )
        # Sensors have been closed for a long time before the scenario starts —
        # a state that changed moments ago would legitimately look like a close
        # delay still running after a restart (see the reconstruction tests).
        self.sensors = {sid: ("off", _T0 - timedelta(days=1)) for sid in sensors}
        self.coord.hass.states.get.side_effect = self._get
        self.coord.hass.async_create_task = lambda c: getattr(c, "close", lambda: None)()

        captured = {}

        def _fake_track(hass, entity_ids, callback_fn):
            captured[tuple(entity_ids)] = callback_fn
            return lambda: None

        with patch(
            "custom_components.thermosmart.coordinator.async_track_state_change_event",
            side_effect=_fake_track,
        ), patch(
            "custom_components.thermosmart.coordinator.async_call_later",
            side_effect=lambda hass, delay, cb: (lambda: None),
        ):
            self.coord.setup_event_listeners()
        self.listener = next(cb for ents, cb in captured.items() if sensors[0] in ents)

    def _get(self, entity_id):
        if entity_id in self.sensors:
            value, changed = self.sensors[entity_id]
            return _state_with_last_changed(value, changed)
        if entity_id == "sensor.test_temp":
            return make_state("19.0")
        return None

    def set(self, sensor, value):
        old_value, _ = self.sensors[sensor]
        self.sensors[sensor] = (value, self.clock.now_utc())
        event = MagicMock()
        event.data = {
            "entity_id": sensor,
            "old_state": MagicMock(state=old_value),
            "new_state": MagicMock(state=value),
        }
        self.listener(event)

    def wait(self, *, minutes=0, seconds=0):
        self.clock.advance(minutes * 60 + seconds)

    def window_open(self):
        # several refreshes in a row must all agree (a refresh never bypasses the delay)
        results = {self.coord._check_window_open(self.coord.zone_cfg, current_temp=19.0)
                   for _ in range(3)}
        assert len(results) == 1
        return results.pop()


class TestOpenDelayWithRealisticStates:
    def test_override_only_after_the_full_open_delay(self):
        rig = _LiveWindowRig(open_delay=5)
        rig.set(_SENSOR, "on")
        assert rig.window_open() is False
        rig.wait(minutes=4, seconds=59)
        assert rig.window_open() is False
        rig.wait(seconds=1)
        assert rig.window_open() is True

    def test_zero_delay_reacts_immediately(self):
        rig = _LiveWindowRig(open_delay=0)
        rig.set(_SENSOR, "on")
        assert rig.window_open() is True

    def test_confirmed_window_is_released_only_after_the_close_delay(self):
        rig = _LiveWindowRig(open_delay=5, close_delay=2)
        rig.set(_SENSOR, "on")
        rig.wait(minutes=6)
        assert rig.window_open() is True
        rig.set(_SENSOR, "off")
        assert rig.window_open() is True            # close delay running
        rig.wait(minutes=1, seconds=59)
        assert rig.window_open() is True
        rig.wait(seconds=1)
        assert rig.window_open() is False


class TestShortAiringNeverActivatesTheOverride:
    def test_closing_before_the_open_delay_never_activates_it(self):
        rig = _LiveWindowRig(open_delay=5, close_delay=2)
        rig.set(_SENSOR, "on")
        rig.wait(minutes=3)
        assert rig.window_open() is False
        rig.set(_SENSOR, "off")
        for _ in range(10):                         # right after, and well past close_delay
            assert rig.window_open() is False
            rig.wait(seconds=30)

    def test_reopening_after_a_short_airing_starts_a_complete_new_delay(self):
        rig = _LiveWindowRig(open_delay=5, close_delay=2)
        rig.set(_SENSOR, "on")
        rig.wait(minutes=2)
        rig.set(_SENSOR, "off")
        rig.wait(seconds=30)
        rig.set(_SENSOR, "on")
        rig.wait(minutes=4, seconds=59)
        assert rig.window_open() is False
        rig.wait(seconds=1)
        assert rig.window_open() is True

    def test_sensor_flapping_inside_the_delay_never_activates_it(self):
        rig = _LiveWindowRig(open_delay=5, close_delay=2)
        for _ in range(6):
            rig.set(_SENSOR, "on")
            rig.wait(seconds=40)
            assert rig.window_open() is False
            rig.set(_SENSOR, "off")
            assert rig.window_open() is False
            rig.wait(seconds=20)

    def test_closing_exactly_at_the_open_delay_counts_as_confirmed(self):
        rig = _LiveWindowRig(open_delay=5, close_delay=2)
        rig.set(_SENSOR, "on")
        rig.wait(minutes=5)
        rig.set(_SENSOR, "off")
        assert rig.window_open() is True            # normal close delay applies
        rig.wait(minutes=2)
        assert rig.window_open() is False


class TestMultipleWindowsWithRealisticStates:
    def test_a_short_airing_of_one_window_does_not_disturb_another_open_one(self):
        rig = _LiveWindowRig(open_delay=5, close_delay=2, sensors=(_SENSOR, _SENSOR_B))
        rig.set(_SENSOR, "on")
        rig.wait(minutes=6)
        assert rig.window_open() is True            # window A confirmed open
        rig.set(_SENSOR_B, "on")
        rig.wait(minutes=1)
        rig.set(_SENSOR_B, "off")                   # B: short airing, never confirmed
        assert rig.window_open() is True            # still A
        rig.set(_SENSOR, "off")
        rig.wait(minutes=2)
        assert rig.window_open() is False           # B left no phantom close delay behind

    def test_a_short_airing_alone_never_activates_the_override(self):
        rig = _LiveWindowRig(open_delay=5, close_delay=2, sensors=(_SENSOR, _SENSOR_B))
        rig.set(_SENSOR_B, "on")
        rig.wait(minutes=1)
        rig.set(_SENSOR_B, "off")
        assert rig.window_open() is False


class TestRestartReconstructionIsUnchangedForUnobservedSensors:
    def test_close_delay_still_resumes_for_a_sensor_the_listener_has_not_seen(self):
        clock = FakeClock(_T0)
        coord = _make(clock)
        closed_30s_ago = _T0 - timedelta(seconds=30)
        set_hass_states(coord, {
            _SENSOR: _state_with_last_changed("off", closed_30s_ago),
            "sensor.test_temp": make_state("19.0"),
        })
        assert coord._check_window_open(coord.zone_cfg, current_temp=19.0) is True
