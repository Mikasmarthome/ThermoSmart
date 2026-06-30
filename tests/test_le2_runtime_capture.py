"""Phase 17 capture + decision identity + command echo tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    CaptureCoordinator,
    CommandLedger,
    CommandType,
    ControllerDecisionInput,
    DecisionType,
    ManualCorrectionObservation,
    RuntimeCycleInput,
    SentCommand,
    make_decision_id,
)
from tests.helpers_runtime import cycle_input


class TestCapture:
    def test_basic_snapshot(self):
        c = CaptureCoordinator("lr")
        snap = c.capture(cycle_input("2026-06-22T05:00:00+00:00"))
        assert snap.zone_id == "lr" and snap.target_c == 21.0
        assert snap.decision_id.startswith("dec_")

    def test_trv_only(self):
        c = CaptureCoordinator("lr")
        snap = c.capture(cycle_input("2026-06-22T05:00:00+00:00"))
        assert snap.indoor_temp is not None and snap.outdoor_temp is None

    def test_missing_indoor_temp(self):
        c = CaptureCoordinator("lr")
        snap = c.capture(cycle_input("2026-06-22T05:00:00+00:00", indoor=None))
        assert snap.indoor_temp is None
        assert snap.indoor_temp_quality is DataQuality.UNAVAILABLE

    def test_exact_zero(self):
        c = CaptureCoordinator("lr")
        snap = c.capture(cycle_input("2026-06-22T05:00:00+00:00", indoor=0.0, target=0.0))
        assert snap.indoor_temp.value == 0.0 and snap.target_c == 0.0

    def test_zone_mismatch_rejected(self):
        c = CaptureCoordinator("lr")
        with pytest.raises(ValueError):
            c.capture(cycle_input("2026-06-22T05:00:00+00:00", zone="other"))


class TestDecisionIdentity:
    def test_deterministic_id(self):
        a = make_decision_id("lr", DecisionType.NORMAL, "2026-06-22T05:00:00+00:00", "tok")
        b = make_decision_id("lr", DecisionType.NORMAL, "2026-06-22T05:00:00+00:00", "tok")
        assert a == b and a.startswith("dec_")

    def test_no_entity_name_in_id(self):
        did = make_decision_id("living_room_climate", DecisionType.NORMAL, "t")
        assert "living_room" not in did

    def test_continuation_same_decision(self):
        c = CaptureCoordinator("lr")
        s1 = c.capture(cycle_input("2026-06-22T05:00:00+00:00"))
        s2 = c.capture(cycle_input("2026-06-22T05:05:00+00:00"))
        assert s1.decision_id == s2.decision_id

    def test_target_jump_new_decision(self):
        c = CaptureCoordinator("lr")
        s1 = c.capture(cycle_input("2026-06-22T05:00:00+00:00", decision_target=21.0))
        s2 = c.capture(cycle_input("2026-06-22T05:05:00+00:00", decision_target=22.0))
        assert s1.decision_id != s2.decision_id

    def test_supersede_new_decision(self):
        c = CaptureCoordinator("lr")
        s1 = c.capture(cycle_input("2026-06-22T05:00:00+00:00"))
        s2 = c.capture(cycle_input("2026-06-22T05:05:00+00:00", superseded=True))
        assert s1.decision_id != s2.decision_id

    def test_zone_scoped_ids_differ(self):
        a = make_decision_id("a", DecisionType.NORMAL, "t", "tok")
        b = make_decision_id("b", DecisionType.NORMAL, "t", "tok")
        assert a != b


class TestCommandEcho:
    def test_echo_detected(self):
        led = CommandLedger()
        led.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                               "2026-06-22T05:00:00+00:00"))
        assert led.is_echo("trv1", 22.0, "2026-06-22T05:00:30+00:00")

    def test_echo_outside_window(self):
        led = CommandLedger()
        led.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                               "2026-06-22T05:00:00+00:00", echo_window_s=60))
        assert not led.is_echo("trv1", 22.0, "2026-06-22T05:05:00+00:00")

    def test_different_trv_not_echo(self):
        led = CommandLedger()
        led.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                               "2026-06-22T05:00:00+00:00"))
        assert not led.is_echo("trv2", 22.0, "2026-06-22T05:00:30+00:00")

    def test_capture_filters_echo_correction(self):
        c = CaptureCoordinator("lr")
        inp = cycle_input(
            "2026-06-22T05:00:30+00:00",
            sent_commands=(SentCommand("c1", CommandType.SETPOINT, "_zone", 22.0,
                                       "2026-06-22T05:00:00+00:00"),),
            manual_corrections=(ManualCorrectionObservation(
                "m1", 21.0, 22.0, "ui", "setpoint_change", "2026-06-22T05:00:10+00:00"),))
        snap = c.capture(inp)
        assert "m1" in snap.rejected_echo_corrections
        assert snap.accepted_manual_corrections == ()

    def test_capture_keeps_real_correction(self):
        c = CaptureCoordinator("lr")
        inp = cycle_input(
            "2026-06-22T05:00:30+00:00",
            manual_corrections=(ManualCorrectionObservation(
                "m1", 21.0, 22.5, "ui", "setpoint_change", "2026-06-22T05:00:10+00:00"),))
        snap = c.capture(inp)
        assert len(snap.accepted_manual_corrections) == 1

    def test_ledger_bounded(self):
        led = CommandLedger(max_entries=3)
        for i in range(6):
            led.record(SentCommand(f"c{i}", CommandType.SETPOINT, "trv1", float(i),
                                   f"2026-06-22T05:0{i}:00+00:00"))
        assert led.size == 3

    def test_ledger_prune(self):
        led = CommandLedger()
        led.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                               "2026-06-22T04:00:00+00:00"))
        removed = led.prune("2026-06-22T06:00:00+00:00", max_age_s=3600)
        assert removed == 1 and led.size == 0

    def test_ledger_roundtrip(self):
        led = CommandLedger()
        led.record(SentCommand("c1", CommandType.SETPOINT, "trv1", 22.0,
                               "2026-06-22T05:00:00+00:00"))
        other = CommandLedger()
        other.restore(led.serialize())
        assert other.is_echo("trv1", 22.0, "2026-06-22T05:00:30+00:00")


class TestCaptureRoundtrip:
    def test_serialize_restore(self):
        c = CaptureCoordinator("lr")
        c.capture(cycle_input("2026-06-22T05:00:00+00:00"))
        other = CaptureCoordinator("lr")
        other.restore(c.serialize())
        assert other.open_decision is not None
        assert other.open_decision.decision_id == c.open_decision.decision_id
