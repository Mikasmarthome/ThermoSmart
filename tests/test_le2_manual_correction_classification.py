"""Phase 15 ManualCorrectionModel classification + attribution tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.raw_schemas import (
    CorrectionActionType,
    CorrectionSource,
)
from custom_components.thermosmart.learning.models import (
    CorrectionDirection,
    CorrectionIntent,
    CorrectionSourceClass,
    ManualCorrectionModel,
)
from tests.helpers_manual_correction import correction_context, correction_event


def m():
    return ManualCorrectionModel("lz")


def ev_eval(prior, new, **ctxkw):
    ev = correction_event("e", prior, new, source=ctxkw.pop("source", CorrectionSource.UI),
                          action=ctxkw.pop("action", CorrectionActionType.SETPOINT_CHANGE))
    ctx = correction_context(ev, **ctxkw) if ctxkw else correction_context(ev)
    return m().evaluate_correction(ev, ctx)


class TestClassification:
    def test_warmer_direct(self):
        e = ev_eval(21.0, 21.5)
        assert e.direction is CorrectionDirection.WARMER
        assert e.intent in (CorrectionIntent.WARMER_PREFERENCE, CorrectionIntent.TOO_LATE)

    def test_cooler_direct(self):
        e = ev_eval(21.0, 20.5)
        assert e.direction is CorrectionDirection.COOLER
        assert e.intent in (CorrectionIntent.COOLER_PREFERENCE,
                            CorrectionIntent.OVERSHOOT_CORRECTION)

    def test_too_late_preheat(self):
        e = ev_eval(21.0, 21.5, in_preheat=True, time_since_decision_s=300.0)
        assert e.intent is CorrectionIntent.TOO_LATE

    def test_overshoot_correction(self):
        e = ev_eval(21.0, 20.5, outcome_overshoot_c=1.5)
        assert e.intent is CorrectionIntent.OVERSHOOT_CORRECTION

    def test_undershoot_correction(self):
        e = ev_eval(21.0, 21.5, outcome_undershoot_c=1.0)
        assert e.intent is CorrectionIntent.UNDERSHOOT_CORRECTION

    def test_boost_too_strong(self):
        e = ev_eval(21.0, 20.5, in_boost=True)
        assert e.intent is CorrectionIntent.BOOST_TOO_STRONG

    def test_boost_too_weak(self):
        e = ev_eval(21.0, 21.5, in_boost=True)
        assert e.intent is CorrectionIntent.BOOST_TOO_WEAK

    def test_temporary_override(self):
        e = ev_eval(21.0, 21.5, reverted_within_s=120.0)
        assert e.intent is CorrectionIntent.TEMPORARY_OVERRIDE

    def test_mode_change(self):
        e = ev_eval(21.0, 21.5, action=CorrectionActionType.MODE_CHANGE)
        assert e.intent is CorrectionIntent.MODE_CHANGE

    def test_schedule_exception(self):
        e = ev_eval(21.0, 21.5, action=CorrectionActionType.OVERRIDE, schedule_period="evening")
        assert e.intent is CorrectionIntent.SCHEDULE_EXCEPTION

    def test_unknown_intent_no_context(self):
        ev = correction_event("e", 21.0, 21.5)
        e = m().evaluate_correction(ev, None)
        assert e.intent in (CorrectionIntent.WARMER_PREFERENCE,)


class TestAttribution:
    def test_direct_ha_user(self):
        e = ev_eval(21.0, 21.5)
        assert e.attribution.source_class is CorrectionSourceClass.DIRECT_USER
        assert e.attribution.reliability >= 0.8

    def test_physical_trv(self):
        e = ev_eval(21.0, 21.5, physical_trv=True)
        assert e.attribution.source_class is CorrectionSourceClass.PHYSICAL_TRV

    def test_external_automation(self):
        ev = correction_event("e", 21.0, 21.5, source=CorrectionSource.AUTOMATION)
        e = m().evaluate_correction(ev, correction_context(ev))
        assert e.attribution.source_class is CorrectionSourceClass.EXTERNAL_AUTOMATION
        assert e.attribution.reliability < 0.6

    def test_internal_echo_class(self):
        ev = correction_event("e", 21.0, 21.5)
        e = m().evaluate_correction(ev, correction_context(ev, last_sent_setpoint_c=21.5))
        assert e.attribution.source_class is CorrectionSourceClass.INTERNAL_ECHO

    def test_device_sync_class(self):
        ev = correction_event("e", 21.0, 21.5)
        e = m().evaluate_correction(ev, correction_context(ev, is_device_sync=True))
        assert e.attribution.source_class is CorrectionSourceClass.DEVICE_SYNC

    def test_unknown_source(self):
        ev = correction_event("e", 21.0, 21.5, source=CorrectionSource.UNKNOWN)
        e = m().evaluate_correction(ev, correction_context(ev))
        assert e.attribution.source_class is CorrectionSourceClass.UNKNOWN
        assert e.attribution.reliability < 0.6

    def test_decision_bound_raises_attribution(self):
        ev = correction_event("e", 21.0, 21.5)
        bound = m().evaluate_correction(
            ev, correction_context(ev, decision_id="d1", time_since_decision_s=300.0))
        assert bound.attribution.decision_bound

    def test_far_correction_weaker_attribution_intent(self):
        # several hours after the decision -> preference, less decision-attributable
        e = ev_eval(21.0, 21.5, time_since_decision_s=4 * 3600.0)
        assert e.intent is CorrectionIntent.WARMER_PREFERENCE


class TestEvidenceSeparation:
    def test_single_correction_low_preference_evidence(self):
        e = ev_eval(21.0, 21.5)
        # comfort preference evidence present but persistence stays weak for a single event
        assert e.pattern.persistence_likelihood <= 0.5

    def test_controller_error_evidence_for_overshoot(self):
        e = ev_eval(21.0, 20.5, outcome_overshoot_c=1.5)
        assert e.controller_error_evidence > 0.0
