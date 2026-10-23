"""Runtime behaviour of the presence arrival/departure delay: wake-up timer
ownership, flapping, reload/unload and restart bootstrap.

The pure transition table is covered in test_presence_delay.py; this file
drives the real _get_presence_state() with person states and a recording
replacement for async_call_later so timer scheduling and cancellation can be
asserted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from custom_components.thermosmart.const import (
    CONF_HOME_ZONE,
    CONF_PRESENCE_ARRIVAL_DELAY,
    CONF_PRESENCE_DEPARTURE_DELAY,
    CONF_PRESENCE_PERSONS,
)
from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_coordinator, make_state, set_hass_states

_START = datetime(2026, 1, 12, 12, 0, tzinfo=timezone.utc)
_A = "person.a"
_B = "person.b"


class _Timers:
    """Recording stand-in for async_call_later."""

    def __init__(self):
        self.scheduled: list[dict] = []

    def __call__(self, hass, delay, callback_fn):
        entry = {"delay": delay, "callback": callback_fn, "cancelled": False}
        self.scheduled.append(entry)

        def _cancel():
            entry["cancelled"] = True

        return _cancel

    @property
    def active(self):
        return [t for t in self.scheduled if not t["cancelled"]]


def _rig(*, arrival=0, departure=0, persons=(_A,), states=None):
    clock = FakeClock(_START)
    coord = make_coordinator(
        {
            CONF_PRESENCE_PERSONS: list(persons),
            CONF_HOME_ZONE: "zone.home",
            CONF_PRESENCE_ARRIVAL_DELAY: arrival,
            CONF_PRESENCE_DEPARTURE_DELAY: departure,
        },
        clock=clock,
    )
    timers = _Timers()
    coord.hass.async_create_task = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )
    patcher = patch(
        "custom_components.thermosmart.coordinator.async_call_later", timers
    )
    patcher.start()
    set_hass_states(coord, {p: make_state(v) for p, v in (states or {}).items()})
    return coord, clock, timers, patcher


def _set(coord, **people):
    set_hass_states(coord, {p: make_state(v) for p, v in people.items()})


def _state(coord, *, a=None, b=None):
    values = {}
    if a is not None:
        values[_A] = a
    if b is not None:
        values[_B] = b
    _set(coord, **values)


class TestTimerOwnership:
    def test_departure_schedules_exactly_one_wakeup_for_the_delay(self):
        coord, clock, timers, patcher = _rig(departure=10, states={_A: "home"})
        try:
            assert coord._get_presence_state()["all_away"] is False
            _state(coord, a="not_home")
            coord._get_presence_state()
            coord._get_presence_state()  # repeated refresh must not add a timer
            assert [t["delay"] for t in timers.active] == [600]
        finally:
            patcher.stop()

    def test_zero_delay_never_schedules_a_timer(self):
        coord, clock, timers, patcher = _rig(states={_A: "home"})
        try:
            coord._get_presence_state()
            _state(coord, a="not_home")
            assert coord._get_presence_state()["all_away"] is True
            _state(coord, a="home")
            assert coord._get_presence_state()["all_away"] is False
            assert timers.scheduled == []
        finally:
            patcher.stop()

    def test_wakeup_requests_a_refresh_and_releases_the_slot(self):
        coord, clock, timers, patcher = _rig(departure=5, states={_A: "home"})
        try:
            coord._get_presence_state()
            _state(coord, a="not_home")
            coord._get_presence_state()
            clock.set_utc(_START + timedelta(minutes=5))
            timers.active[0]["callback"](clock.now_utc())
            assert coord._presence_delay_cancel is None
            assert coord.hass.async_create_task.called
            assert coord._get_presence_state()["all_away"] is True
            assert len(timers.scheduled) == 1  # confirmation scheduled nothing new
        finally:
            patcher.stop()

    def test_wakeup_firing_a_hair_early_still_confirms(self):
        coord, clock, timers, patcher = _rig(departure=5, states={_A: "home"})
        try:
            coord._get_presence_state()
            _state(coord, a="not_home")
            coord._get_presence_state()
            clock.set_utc(_START + timedelta(minutes=5) - timedelta(milliseconds=200))
            assert coord._get_presence_state()["all_away"] is True
        finally:
            patcher.stop()


class TestFlapping:
    def test_departure_flapping_never_goes_away_and_leaves_no_live_timer(self):
        coord, clock, timers, patcher = _rig(departure=5, states={_A: "home"})
        try:
            coord._get_presence_state()
            for minute in range(1, 9):
                _state(coord, a="not_home" if minute % 2 else "home")
                clock.set_utc(_START + timedelta(minutes=minute))
                assert coord._get_presence_state()["all_away"] is False
            # ends on "home": nothing pending, every superseded timer cancelled
            assert coord._presence_pending_target is None
            assert timers.active == []
        finally:
            patcher.stop()

    def test_reversal_restarts_the_clock_from_the_new_transition(self):
        coord, clock, timers, patcher = _rig(departure=5, states={_A: "home"})
        try:
            coord._get_presence_state()
            _state(coord, a="not_home")
            coord._get_presence_state()                       # departure pending since 0:00
            clock.set_utc(_START + timedelta(minutes=3))
            _state(coord, a="home")
            coord._get_presence_state()                       # cancelled
            clock.set_utc(_START + timedelta(minutes=4))
            _state(coord, a="not_home")
            coord._get_presence_state()                       # new departure since 0:04
            clock.set_utc(_START + timedelta(minutes=6))      # 2 min after the new start
            assert coord._get_presence_state()["all_away"] is False
            clock.set_utc(_START + timedelta(minutes=9))
            assert coord._get_presence_state()["all_away"] is True
        finally:
            patcher.stop()

    def test_arrival_flapping_stays_away(self):
        coord, clock, timers, patcher = _rig(arrival=5, states={_A: "not_home"})
        try:
            assert coord._get_presence_state()["all_away"] is True
            for minute in range(1, 9):
                _state(coord, a="home" if minute % 2 else "not_home")
                clock.set_utc(_START + timedelta(minutes=minute))
                assert coord._get_presence_state()["all_away"] is True
            assert timers.active == []
        finally:
            patcher.stop()


class TestMultiplePeople:
    def test_second_person_still_home_never_starts_a_departure_timer(self):
        coord, clock, timers, patcher = _rig(
            departure=5, persons=(_A, _B), states={_A: "home", _B: "home"}
        )
        try:
            coord._get_presence_state()
            _state(coord, a="not_home", b="home")
            assert coord._get_presence_state()["all_away"] is False
            assert timers.scheduled == []
            _state(coord, a="not_home", b="not_home")
            coord._get_presence_state()
            assert [t["delay"] for t in timers.active] == [300]
        finally:
            patcher.stop()

    def test_second_person_arriving_during_arrival_delay_keeps_the_original_deadline(self):
        coord, clock, timers, patcher = _rig(
            arrival=5, persons=(_A, _B), states={_A: "not_home", _B: "not_home"}
        )
        try:
            assert coord._get_presence_state()["all_away"] is True
            _state(coord, a="home", b="not_home")
            assert coord._get_presence_state()["all_away"] is True   # pending
            clock.set_utc(_START + timedelta(minutes=2))
            _state(coord, a="home", b="home")                        # same target, no restart
            assert coord._get_presence_state()["all_away"] is True
            clock.set_utc(_START + timedelta(minutes=5))
            assert coord._get_presence_state()["all_away"] is False
        finally:
            patcher.stop()


class TestReloadUnloadAndRestart:
    def test_unload_cancels_the_running_timer(self):
        coord, clock, timers, patcher = _rig(departure=5, states={_A: "home"})
        try:
            coord._get_presence_state()
            _state(coord, a="not_home")
            coord._get_presence_state()
            assert len(timers.active) == 1
            coord.cleanup_event_listeners()
            assert timers.active == []
            assert coord._presence_delay_cancel is None
        finally:
            patcher.stop()

    def test_reload_discards_pending_state_and_adopts_current_presence(self):
        """A reload creates a new coordinator: the old pending departure is gone
        (timer cancelled by the unload) and the new one adopts the real state."""
        old, clock, timers, patcher = _rig(departure=10, states={_A: "home"})
        try:
            old._get_presence_state()
            _state(old, a="not_home")
            old._get_presence_state()
            old.cleanup_event_listeners()
            assert timers.active == []

            new, _, new_timers, new_patcher = _rig(departure=10, states={_A: "not_home"})
            try:
                assert new._get_presence_state()["all_away"] is True   # no artificial delay
                assert new._presence_pending_target is None
                assert new_timers.scheduled == []
            finally:
                new_patcher.stop()
        finally:
            patcher.stop()

    def test_restart_with_everyone_home_and_delays_is_home_immediately(self):
        coord, clock, timers, patcher = _rig(arrival=5, departure=5, states={_A: "home"})
        try:
            assert coord._get_presence_state()["all_away"] is False
            assert timers.scheduled == []
        finally:
            patcher.stop()

    def test_person_states_not_yet_available_do_not_latch_away(self):
        """Right after a Home Assistant start the person entities can be missing
        or unavailable. That must not be adopted as everyone away, otherwise the
        arrival delay would hold the zone in away mode after every restart."""
        coord, clock, timers, patcher = _rig(arrival=5, states={})
        try:
            set_hass_states(coord, {})                                # entity not there yet
            assert coord._get_presence_state()["all_away"] is True    # raw value, nothing latched
            assert coord._presence_all_away_confirmed is None
            _state(coord, a="unavailable")
            assert coord._get_presence_state()["all_away"] is True
            assert coord._presence_all_away_confirmed is None

            _state(coord, a="home")                                   # real state arrives
            assert coord._get_presence_state()["all_away"] is False   # adopted, no 5 min wait
            assert timers.scheduled == []
        finally:
            patcher.stop()

    def test_unavailable_after_bootstrap_is_debounced_like_away(self):
        coord, clock, timers, patcher = _rig(departure=5, states={_A: "home"})
        try:
            coord._get_presence_state()
            _state(coord, a="unavailable")
            assert coord._get_presence_state()["all_away"] is False   # short dropout ignored
            _state(coord, a="home")
            assert coord._get_presence_state()["all_away"] is False
            assert timers.active == []
        finally:
            patcher.stop()


class TestOptionsOverrideData:
    def test_options_value_wins_over_original_entry_data(self):
        coord, clock, timers, patcher = _rig(departure=0, states={_A: "home"})
        try:
            coord.entry.options = {CONF_PRESENCE_DEPARTURE_DELAY: 7.0}  # saved later via options flow
            coord._get_presence_state()
            _state(coord, a="not_home")
            assert coord._get_presence_state()["all_away"] is False
            assert [t["delay"] for t in timers.active] == [420.0]
        finally:
            patcher.stop()

    def test_options_zero_wins_over_original_nonzero_data(self):
        coord, clock, timers, patcher = _rig(departure=10, states={_A: "home"})
        try:
            coord.entry.options = {CONF_PRESENCE_DEPARTURE_DELAY: 0}
            coord._get_presence_state()
            _state(coord, a="not_home")
            assert coord._get_presence_state()["all_away"] is True
        finally:
            patcher.stop()
