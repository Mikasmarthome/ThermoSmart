"""Tests for presence arrival/departure delay debouncing (Settings UX Polish block).

Mirrors window.py's _check_window_open() pattern: _resolve_presence_all_away()
is a pure, timestamp-based evaluator re-run every coordinator cycle. With both
delays at 0 minutes (the default), behavior must be byte-identical to the
pre-existing instant switch — no pending state is ever introduced.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.thermosmart.const import (
    CONF_PRESENCE_PERSONS,
    CONF_HOME_ZONE,
    CONF_PRESENCE_ARRIVAL_DELAY,
    CONF_PRESENCE_DEPARTURE_DELAY,
)
from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_coordinator, make_state, set_hass_states

_START = datetime(2026, 1, 12, 12, 0, tzinfo=timezone.utc)


def _coord(*, arrival=0, departure=0, persons=("person.mika",), clock=None):
    return make_coordinator(
        zone_cfg_overrides={
            CONF_PRESENCE_PERSONS: list(persons),
            CONF_HOME_ZONE: "zone.home",
            CONF_PRESENCE_ARRIVAL_DELAY: arrival,
            CONF_PRESENCE_DEPARTURE_DELAY: departure,
        },
        clock=clock or FakeClock(_START),
    )


class TestPresenceDelayBootstrap:
    """First evaluation after startup/reload adopts the raw state immediately —
    no artificial delay is ever applied to the very first reading."""

    def test_bootstrap_adopts_raw_home_state_immediately(self):
        coord = _coord(arrival=10, departure=10)
        assert coord._resolve_presence_all_away(False) is False
        assert coord._presence_pending_target is None

    def test_bootstrap_adopts_raw_away_state_immediately(self):
        coord = _coord(arrival=10, departure=10)
        assert coord._resolve_presence_all_away(True) is True
        assert coord._presence_pending_target is None


class TestPresenceDelayZeroIsInstant:
    """0/0 minutes (the default) must reproduce the exact pre-existing behavior."""

    def test_zero_delay_departure_is_instant(self):
        coord = _coord(arrival=0, departure=0)
        assert coord._resolve_presence_all_away(False) is False  # bootstrap: home
        assert coord._resolve_presence_all_away(True) is True    # instant switch to away
        assert coord._presence_pending_target is None

    def test_zero_delay_arrival_is_instant(self):
        coord = _coord(arrival=0, departure=0)
        assert coord._resolve_presence_all_away(True) is True    # bootstrap: away
        assert coord._resolve_presence_all_away(False) is False  # instant switch to home
        assert coord._presence_pending_target is None


class TestPresenceDelayDeparture:
    def test_departure_delay_pending_then_confirmed(self):
        clock = FakeClock(_START)
        coord = _coord(arrival=0, departure=5, clock=clock)
        assert coord._resolve_presence_all_away(False) is False  # bootstrap: home
        assert coord._resolve_presence_all_away(True) is False   # pending departure, still home
        assert coord._presence_pending_target is True

        clock.set_utc(_START + timedelta(minutes=4))
        assert coord._resolve_presence_all_away(True) is False   # not yet elapsed

        clock.set_utc(_START + timedelta(minutes=5))
        assert coord._resolve_presence_all_away(True) is True    # confirmed
        assert coord._presence_pending_target is None

    def test_departure_pending_cancelled_by_return(self):
        clock = FakeClock(_START)
        coord = _coord(arrival=0, departure=5, clock=clock)
        coord._resolve_presence_all_away(False)  # bootstrap
        coord._resolve_presence_all_away(True)   # start pending departure
        assert coord._presence_pending_target is True

        clock.set_utc(_START + timedelta(minutes=2))
        result = coord._resolve_presence_all_away(False)  # returned before delay elapsed
        assert result is False
        assert coord._presence_pending_target is None

        clock.set_utc(_START + timedelta(minutes=10))
        assert coord._resolve_presence_all_away(False) is False  # stays home, no stale flip


class TestPresenceDelayArrival:
    def test_arrival_delay_pending_then_confirmed(self):
        clock = FakeClock(_START)
        coord = _coord(arrival=5, departure=0, clock=clock)
        assert coord._resolve_presence_all_away(True) is True   # bootstrap: away
        assert coord._resolve_presence_all_away(False) is True  # pending arrival, still away
        assert coord._presence_pending_target is False

        clock.set_utc(_START + timedelta(minutes=5))
        assert coord._resolve_presence_all_away(False) is False  # confirmed
        assert coord._presence_pending_target is None

    def test_arrival_pending_cancelled_by_leaving_again(self):
        clock = FakeClock(_START)
        coord = _coord(arrival=5, departure=0, clock=clock)
        coord._resolve_presence_all_away(True)   # bootstrap: away
        coord._resolve_presence_all_away(False)  # pending arrival

        clock.set_utc(_START + timedelta(minutes=2))
        result = coord._resolve_presence_all_away(True)  # left again before delay elapsed
        assert result is True
        assert coord._presence_pending_target is None


class TestPresenceDelayRepeatedEvents:
    def test_repeated_identical_raw_value_does_not_restart_timer(self):
        clock = FakeClock(_START)
        coord = _coord(arrival=0, departure=5, clock=clock)
        coord._resolve_presence_all_away(False)  # bootstrap
        coord._resolve_presence_all_away(True)   # start pending
        first_since = coord._presence_pending_since

        clock.set_utc(_START + timedelta(minutes=3))
        coord._resolve_presence_all_away(True)   # repeated identical raw value
        assert coord._presence_pending_since == first_since  # timer not restarted

        clock.set_utc(_START + timedelta(minutes=5))
        assert coord._resolve_presence_all_away(True) is True  # confirms at the ORIGINAL deadline


class TestPresenceDelayCleanup:
    def test_cleanup_event_listeners_cancels_presence_timer(self):
        clock = FakeClock(_START)
        coord = _coord(arrival=0, departure=5, clock=clock)
        coord._resolve_presence_all_away(False)  # bootstrap
        coord._resolve_presence_all_away(True)   # start pending, schedules async_call_later
        assert coord._presence_delay_cancel is not None

        coord.cleanup_event_listeners()
        assert coord._presence_delay_cancel is None
        assert coord._event_unsub == []


class TestPresenceDelayIntegration:
    """End-to-end through _get_presence_state() with multiple aggregated persons."""

    async def test_aggregation_at_least_one_home_means_present(self):
        coord = _coord(arrival=0, departure=5, persons=("person.mika", "person.luisa"))
        set_hass_states(coord, {
            "person.mika": make_state("not_home"),
            "person.luisa": make_state("home"),
        })
        presence = coord._get_presence_state()
        assert presence["all_away"] is False  # bootstrap: Luisa is home

    async def test_departure_delay_holds_home_until_both_leave_and_delay_elapses(self):
        clock = FakeClock(_START)
        coord = _coord(arrival=0, departure=5, persons=("person.mika", "person.luisa"), clock=clock)
        set_hass_states(coord, {
            "person.mika": make_state("home"),
            "person.luisa": make_state("home"),
        })
        assert coord._get_presence_state()["all_away"] is False  # bootstrap

        # Mika leaves, Luisa stays — must NOT start a departure delay at all.
        set_hass_states(coord, {
            "person.mika": make_state("not_home"),
            "person.luisa": make_state("home"),
        })
        assert coord._get_presence_state()["all_away"] is False
        assert coord._presence_pending_target is None

        # Luisa leaves too — now both are away, departure delay starts.
        set_hass_states(coord, {
            "person.mika": make_state("not_home"),
            "person.luisa": make_state("not_home"),
        })
        assert coord._get_presence_state()["all_away"] is False  # still pending
        assert coord._presence_pending_target is True

        clock.set_utc(_START + timedelta(minutes=5))
        assert coord._get_presence_state()["all_away"] is True  # confirmed

    async def test_return_during_departure_delay_cancels_pending(self):
        clock = FakeClock(_START)
        coord = _coord(arrival=0, departure=5, persons=("person.mika",), clock=clock)
        set_hass_states(coord, {"person.mika": make_state("home")})
        assert coord._get_presence_state()["all_away"] is False  # bootstrap

        set_hass_states(coord, {"person.mika": make_state("not_home")})
        coord._get_presence_state()  # start pending departure
        assert coord._presence_pending_target is True

        clock.set_utc(_START + timedelta(minutes=2))
        set_hass_states(coord, {"person.mika": make_state("home")})
        assert coord._get_presence_state()["all_away"] is False
        assert coord._presence_pending_target is None
