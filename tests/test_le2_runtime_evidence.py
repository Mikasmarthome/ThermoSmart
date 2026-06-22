"""Phase 17 evidence materialisation tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import (
    CaptureCoordinator,
    DecisionType,
    EvidenceMaterializer,
    ManualCorrectionObservation,
)
from tests.helpers_runtime import cycle_input


def materialize(inp):
    cap = CaptureCoordinator(inp.zone_id)
    snap = cap.capture(inp)
    return EvidenceMaterializer().materialize(snap), snap


class TestEvidence:
    def test_manual_correction_context_bound_to_decision(self):
        inp = cycle_input(
            "2026-06-22T05:00:30+00:00",
            manual_corrections=(ManualCorrectionObservation(
                "m1", 21.0, 22.5, "ui", "setpoint_change", "2026-06-22T05:00:10+00:00"),))
        ev, snap = materialize(inp)
        assert len(ev.manual_correction_contexts) == 1
        ctx = ev.manual_correction_contexts[0]
        assert ctx.correction_event_id == "m1" and ctx.decision_id == snap.decision_id

    def test_manual_correction_context_zero_valid(self):
        inp = cycle_input(
            "2026-06-22T05:00:30+00:00", target=0.0, indoor=0.0,
            manual_corrections=(ManualCorrectionObservation(
                "m1", 0.0, 0.5, "ui", "setpoint_change", "2026-06-22T05:00:10+00:00"),))
        ev, _ = materialize(inp)
        ctx = ev.manual_correction_contexts[0]
        assert ctx.previous_thermosmart_target_c == 0.0

    def test_echo_correction_not_materialized(self):
        from custom_components.thermosmart.learning.runtime import SentCommand, CommandType
        inp = cycle_input(
            "2026-06-22T05:00:30+00:00",
            sent_commands=(SentCommand("c1", CommandType.SETPOINT, "_zone", 22.0,
                                       "2026-06-22T05:00:00+00:00"),),
            manual_corrections=(ManualCorrectionObservation(
                "m1", 21.0, 22.0, "ui", "setpoint_change", "2026-06-22T05:00:10+00:00"),))
        ev, _ = materialize(inp)
        assert ev.manual_correction_contexts == ()
        assert any("echo_filtered" in n for n in ev.notes)

    def test_boost_context_materialized(self):
        inp = cycle_input("2026-06-22T05:00:00+00:00", decision_type=DecisionType.BOOST,
                          target=21.0, setpoint=23.0)
        ev, snap = materialize(inp)
        assert ev.boost_context is not None
        assert ev.boost_context.requested_offset_c == pytest.approx(2.0)
        assert ev.boost_context.decision_id == snap.decision_id

    def test_no_boost_context_when_not_boost(self):
        ev, _ = materialize(cycle_input("2026-06-22T05:00:00+00:00"))
        assert ev.boost_context is None

    def test_context_json_serializable(self):
        import json
        inp = cycle_input(
            "2026-06-22T05:00:30+00:00",
            manual_corrections=(ManualCorrectionObservation(
                "m1", 21.0, 22.5, "ui", "setpoint_change", "2026-06-22T05:00:10+00:00"),))
        ev, _ = materialize(inp)
        json.dumps(ev.manual_correction_contexts[0].to_dict())

    def test_zone_in_evidence(self):
        ev, _ = materialize(cycle_input("2026-06-22T05:00:00+00:00"))
        assert ev.zone_id == "lr"
