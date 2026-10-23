"""Characterisation of how ThermoSmart tells its own TRV writes from changes made
elsewhere, and of the (so far unused) timestamped command ledger.

Active logic: the coordinator remembers the setpoint it last wrote per TRV
(``_last_written_setpoints``). A state change of the TRV that differs from it by
more than 0.4 degrees asks for an immediate refresh, which re-asserts the
setpoint; a matching state (the TRV echoing our write) does not.

The ledger in the learning capture layer records timestamped sent commands with
an echo window. Nothing in the integration feeds it, so it stays empty. It is
pinned here so the choice between reusing it for a future actuator layer and
dropping it is made on facts.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.thermosmart.learning.runtime.capture import (
    CommandLedger,
    CommandType,
    SentCommand,
)
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from tests.helpers import make_coordinator, make_state
from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_dispatching_coordinator

_TRV = "climate.trv"


def _rig(*, active=True, window_open=False, last_written=21.0):
    coord = make_coordinator({"climate_entities": [_TRV], "temp_sensors": ["sensor.test_temp"]})
    coord._active_control = active
    coord.data = {"zone": {"window_open": window_open}}
    if last_written is not None:
        coord._last_written_setpoints[_TRV] = last_written
    coord.async_request_refresh = AsyncMock()
    coord.hass.async_create_task = lambda c: getattr(c, "close", lambda: None)()
    captured = {}

    def _track(hass, entity_ids, callback_fn):
        captured[tuple(entity_ids)] = callback_fn
        return lambda: None

    with patch("custom_components.thermosmart.coordinator.async_track_state_change_event",
               side_effect=_track):
        coord.setup_event_listeners()
    listener = next(cb for ids, cb in captured.items() if ids == (_TRV,))
    return coord, listener


def _change(listener, setpoint):
    event = MagicMock()
    event.data = {"entity_id": _TRV,
                  "new_state": make_state("heat", {"temperature": setpoint})}
    listener(event)


class TestTrvEchoAndDeviationHandling:
    def test_echo_of_our_own_write_does_not_trigger_a_refresh(self):
        coord, listener = _rig(last_written=21.0)
        _change(listener, 21.0)
        _change(listener, 21.3)       # within the 0.4 tolerance
        coord.async_request_refresh.assert_not_called()

    def test_a_different_setpoint_triggers_an_immediate_refresh(self):
        coord, listener = _rig(last_written=21.0)
        _change(listener, 22.0)
        coord.async_request_refresh.assert_called_once()

    def test_observation_mode_never_reacts(self):
        coord, listener = _rig(active=False, last_written=21.0)
        _change(listener, 25.0)
        coord.async_request_refresh.assert_not_called()

    def test_window_open_suppresses_the_reaction(self):
        coord, listener = _rig(window_open=True, last_written=21.0)
        _change(listener, 25.0)
        coord.async_request_refresh.assert_not_called()

    def test_nothing_written_yet_means_nothing_to_compare(self):
        coord, listener = _rig(last_written=None)
        _change(listener, 25.0)
        coord.async_request_refresh.assert_not_called()

    def test_a_state_without_a_setpoint_is_ignored(self):
        coord, listener = _rig(last_written=21.0)
        _change(listener, None)
        coord.async_request_refresh.assert_not_called()


class TestCommandLedgerToday:
    def _command(self, n, sent_ts="2026-01-12T06:00:00+00:00", value=21.0):
        return SentCommand(command_id=f"c{n}", command_type=CommandType.SETPOINT,
                           trv_binding_id="trv_a", value=value, sent_ts=sent_ts)

    def test_records_prunes_and_round_trips(self):
        ledger = CommandLedger()
        ledger.ingest([self._command(1), self._command(2, "2026-01-12T07:30:00+00:00")])
        assert ledger.size == 2

        ledger.prune("2026-01-12T07:31:00+00:00")             # default max age one hour
        assert ledger.size == 1

        restored = CommandLedger()
        restored.restore(ledger.serialize())
        assert restored.serialize() == ledger.serialize()

    async def test_the_integration_never_feeds_the_ledger(self):
        coord = make_dispatching_coordinator({"schedule_enabled": False}, indoor="19.5")
        shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
        await shadow.async_setup()
        for _ in range(3):
            await coord._async_update_data()

        zone_runtime = shadow._runtime._zone(shadow._zone)
        assert zone_runtime.capture.ledger.size == 0
