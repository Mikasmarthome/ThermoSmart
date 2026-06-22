"""Phase 15 ManualCorrectionModel TRV-only, no-temp, numeric guards,
confidence-integration, diagnostics/export tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceComponent,
    ConfidenceInput,
    ConfidencePurpose,
    EvidenceDomain,
)
from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.raw_schemas import CorrectionSource
from custom_components.thermosmart.learning.models import (
    ManualCorrectionModel,
    ManualCorrectionPredictionContext,
)
from tests.helpers_manual_correction import (
    correction_context,
    correction_event,
    recurring_direct,
)


def m():
    return ManualCorrectionModel("lz")


def trained(n=6):
    model = m()
    recurring_direct(model, n)
    return model


class TestTrvOnly:
    def test_works_without_optional_data(self):
        ev = correction_event("e", 21.0, 21.5)
        assert m().update(ev, correction_context(ev)).accepted

    def test_pattern_without_temperature(self):
        model = m()
        recurring_direct(model, 6)  # no current_temperature in context
        assert model.predict_correction(ManualCorrectionPredictionContext()).preference_bias_c > 0


class TestNoTemperature:
    def test_direction_size_source_evaluable(self):
        ev = correction_event("e", 21.0, 21.5)
        e = m().evaluate_correction(ev, correction_context(ev))  # no temp
        assert e.direction.value == "warmer" and e.magnitude_c == pytest.approx(0.5)
        assert not e.has_temperature

    def test_no_invented_overshoot(self):
        ev = correction_event("e", 21.0, 20.5)
        e = m().evaluate_correction(ev, correction_context(ev))  # no temp, no overshoot ctx
        assert e.intent.value != "overshoot_correction"

    def test_preference_pattern_possible_without_temp(self):
        model = m()
        recurring_direct(model, 6)
        assert not model.predict_correction(ManualCorrectionPredictionContext()).fallback_used


class TestNumericGuards:
    def test_zero_correction_no_preference(self):
        assert not m().update(correction_event("e", 21.0, 21.0)).accepted

    def test_same_old_new(self):
        e = m().evaluate_correction(correction_event("e", 21.0, 21.0), None)
        assert e.magnitude_c == 0.0 and e.direction.value == "neutral"

    def test_negative_targets(self):
        e = m().evaluate_correction(correction_event("e", -2.0, -1.0), None)
        assert e.signed_magnitude_c == pytest.approx(1.0)

    def test_finite_outputs(self):
        rec = trained().predict_correction(ManualCorrectionPredictionContext())
        assert math.isfinite(rec.preference_bias_c) and math.isfinite(rec.confidence)


class TestConfidenceIntegration:
    def _agg(self):
        return ConfidenceAggregator()

    def test_manual_correction_contribution(self):
        r = self._agg().aggregate(ConfidenceInput(
            ConfidencePurpose.MANUAL_CORRECTION,
            components=(ConfidenceComponent(kind=ComponentKind.MANUAL_CORRECTION,
                                            contribution=trained(8).confidence()),)))
        assert 0.0 <= r.value <= 1.0

    def test_user_correction_independent_domain(self):
        c = trained(8).confidence()
        assert EvidenceDomain.USER_CORRECTION.value in c.evidence_domains

    def test_correlation_with_outcome_visible(self):
        r = self._agg().aggregate(ConfidenceInput(
            ConfidencePurpose.MANUAL_CORRECTION,
            components=(
                ConfidenceComponent(kind=ComponentKind.MANUAL_CORRECTION,
                                    contribution=trained(8).confidence(),
                                    evidence_domains=(EvidenceDomain.USER_CORRECTION,
                                                      EvidenceDomain.CONTROLLER_EVENTS)),
                ConfidenceComponent(kind=ComponentKind.OUTCOME,
                                    contribution=trained(8).confidence(),
                                    evidence_domains=(EvidenceDomain.CONTROLLER_EVENTS,)))))
        assert "correlated_evidence_discounted" in r.reason_codes

    def test_low_attribution_limits_confidence(self):
        c = m().confidence()
        assert c.value <= 0.3


class TestDiagnostics:
    def test_fields(self):
        d = trained().diagnostics()
        assert math.isfinite(d.preference_bias_c) and d.model_version == 1
        assert d.warmer_count >= 1 and d.evaluation_version == 1

    def test_intent_distribution(self):
        assert trained().diagnostics().intent_distribution

    def test_rejection_counts(self):
        model = m()
        model.update(correction_event("e", 21.0, 21.0))  # zero correction
        assert sum(model.diagnostics().rejection_counts.values()) >= 1


class TestExport:
    def test_support_no_ids(self):
        exp = trained().export(ExportScope.SUPPORT)
        assert "preference_bias_c" in exp and "intent_distribution" in exp
        assert "samples" not in exp
        import json
        assert "correction_event_id" not in json.dumps(exp)

    def test_research_pseudonymized(self):
        exp = trained().export(ExportScope.RESEARCH)
        for s in exp["samples"]:
            assert "correction_event_id" not in s and "decision_id" not in s
            assert "direction" in s and "intent" in s and "source_class" in s

    def test_raw_not_provided(self):
        assert "unsupported" in trained().export(ExportScope.RAW)


class TestPurityRuntime:
    def test_input_not_mutated(self):
        ev = correction_event("e", 21.0, 21.5)
        m().update(ev, correction_context(ev))
        assert ev.prior_target == 21.0 and ev.new_target == 21.5

    def test_deterministic(self):
        assert trained()._state.preference_bias_c == trained()._state.preference_bias_c
