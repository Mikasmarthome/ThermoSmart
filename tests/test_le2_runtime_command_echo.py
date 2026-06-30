"""Phase 17 command/echo correlation (multi-TRV, delayed echo, expiry)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    CommandLedger,
    CommandType,
    SentCommand,
)


def led():
    return CommandLedger()


class TestCommandEcho:
    def test_own_setpoint_is_echo(self):
        l = led()
        l.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                             "2026-06-22T05:00:00+00:00"))
        assert l.is_echo("trv1", 22.0, "2026-06-22T05:00:10+00:00")

    def test_delayed_echo_within_window(self):
        l = led()
        l.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                             "2026-06-22T05:00:00+00:00", echo_window_s=120))
        assert l.is_echo("trv1", 22.0, "2026-06-22T05:01:50+00:00")

    def test_echo_expired(self):
        l = led()
        l.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                             "2026-06-22T05:00:00+00:00", echo_window_s=120))
        assert not l.is_echo("trv1", 22.0, "2026-06-22T05:10:00+00:00")

    def test_multiple_trvs_separate(self):
        l = led()
        l.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                             "2026-06-22T05:00:00+00:00"))
        l.record(SentCommand("c2", CommandType.SETPOINT, "trv2", 20.0,
                             "2026-06-22T05:00:00+00:00"))
        assert l.is_echo("trv1", 22.0, "2026-06-22T05:00:10+00:00")
        assert l.is_echo("trv2", 20.0, "2026-06-22T05:00:10+00:00")
        assert not l.is_echo("trv1", 20.0, "2026-06-22T05:00:10+00:00")

    def test_different_value_not_echo(self):
        l = led()
        l.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                             "2026-06-22T05:00:00+00:00"))
        assert not l.is_echo("trv1", 23.5, "2026-06-22T05:00:10+00:00")

    def test_mode_command_not_setpoint_echo(self):
        l = led()
        l.record(SentCommand("c1", CommandType.MODE, "trv1", 1.0,
                             "2026-06-22T05:00:00+00:00"))
        assert not l.is_echo("trv1", 1.0, "2026-06-22T05:00:10+00:00")

    def test_echo_before_command_not_matched(self):
        l = led()
        l.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                             "2026-06-22T05:00:00+00:00"))
        # a change observed BEFORE we sent the command is a real user action
        assert not l.is_echo("trv1", 22.0, "2026-06-22T04:59:50+00:00")
