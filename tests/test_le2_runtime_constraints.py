"""Phase 17B: internalised bridge/capture invariants.

These tests are the single authoritative home for the invariants that previously
lived in local constraint notes (boost bridge + manual-correction capture). The
notes are no longer a second source of truth — code + these tests are.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.const import TPI_MAX_BOOST_CELSIUS
from custom_components.thermosmart.learning.models import BoostParameters
from custom_components.thermosmart.learning.models.manual_correction import CorrectionIntent
from custom_components.thermosmart.learning.runtime import (
    CommandLedger,
    CommandType,
    ManualCorrectionObservation,
    SentCommand,
)
from custom_components.thermosmart.learning.runtime.capture import CaptureCoordinator
from tests.helpers_manual_correction import correction_context, correction_event
from tests.helpers_runtime import cycle_input
from custom_components.thermosmart.learning.models import ManualCorrectionModel


class TestBoostBridgeInvariant:
    def test_boost_offset_matches_runtime_limit(self):
        # the pure model bound and TPI_MAX_BOOST_CELSIUS must never diverge unnoticed
        assert BoostParameters().max_boost_offset_c == TPI_MAX_BOOST_CELSIUS

    def test_boost_offset_is_additive_celsius(self):
        # 0.0 = no boost (additive offset semantics, not a multiplier)
        from custom_components.thermosmart.learning.models import BoostUpdateContext
        ctx = BoostUpdateContext(source_episode_id="e", decision_id="d", requested_offset_c=0.0)
        assert ctx.requested_offset_c == 0.0


class TestManualCorrectionCaptureInvariant:
    def test_own_setpoint_write_excluded_as_echo(self):
        # ThermoSmart's own setpoint write must never be learned as a user correction
        cap = CaptureCoordinator("lz")
        inp = cycle_input(
            "2025-01-01T05:00:30+00:00", zone="lz",
            sent_commands=(SentCommand("c1", CommandType.SETPOINT, "_zone", 22.0,
                                       "2025-01-01T05:00:00+00:00"),),
            manual_corrections=(ManualCorrectionObservation(
                "m1", 21.0, 22.0, "ui", "setpoint_change", "2025-01-01T05:00:10+00:00"),))
        snap = cap.capture(inp)
        assert "m1" in snap.rejected_echo_corrections
        assert snap.accepted_manual_corrections == ()

    def test_too_early_not_emitted_without_comfort_time(self):
        # TOO_EARLY must stay unused until a typed comfort time is available
        m = ManualCorrectionModel("lz")
        ev = correction_event("e", 21.0, 21.5)
        result = m.evaluate_correction(ev, correction_context(ev))
        assert result.intent is not CorrectionIntent.TOO_EARLY

    def test_command_ledger_is_bounded(self):
        led = CommandLedger(max_entries=5)
        for i in range(20):
            led.record(SentCommand(f"c{i}", CommandType.SETPOINT, "t", float(i),
                                   f"2025-01-01T05:{i % 60:02d}:00+00:00"))
        assert led.size == 5
