"""Phase 19A: Single Dispatch path."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DecisionMode, DeviceAdapter
from custom_components.thermosmart.learning.decision.dispatch import SingleDispatcher
from tests.helpers_decision import run


class TestSingleDispatch:
    def test_shadow_command_not_sent(self):
        sent = []
        t, d = run(DecisionMode.SHADOW, sink=sent.append)
        assert sent == [] and not d.dispatched and d.shadow_only

    def test_control_command_sent_once(self):
        sent = []
        t, d = run(DecisionMode.CONTROL, sink=sent.append)
        assert len(sent) == 1 and d.dispatched and not d.shadow_only

    def test_no_sink_does_not_send(self):
        t, d = run(DecisionMode.CONTROL, sink=None)
        assert not d.dispatched and d.reason == "no_sink"

    def test_only_device_control_command_accepted(self):
        disp = SingleDispatcher(sink=lambda c: None)
        cmd = DeviceAdapter().build_command("z", setpoint_c=21.0, shadow_only=False)
        assert disp.dispatch(cmd).dispatched and disp.dispatch_count == 1

    def test_records_last_command(self):
        disp = SingleDispatcher()
        cmd = DeviceAdapter().build_command("z", setpoint_c=21.0)
        disp.dispatch(cmd)
        assert disp.last_command is cmd

    def test_control_no_change_not_dispatched(self):
        # baseline already optimal -> no boost reduction -> nothing dispatched
        from tests.helpers_decision import preds, rec
        sent = []
        t, d = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=21.0),
                   predictions=preds(boost=0.0), sink=sent.append)
        assert sent == [] and not d.dispatched
