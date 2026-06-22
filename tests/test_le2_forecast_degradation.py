"""Phase 13 ForecastModel numeric guards, no-forecast, confidence-integration,
diagnostics/export tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceComponent,
    ConfidenceInput,
    ConfidencePurpose,
    ConfidenceStatus,
    EvidenceDomain,
)
from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.models import (
    ForecastModel,
    ForecastPredictionContext,
    ForecastUpdateContext,
)
from tests.helpers_forecast import decision, evaluation, pair


def m():
    return ForecastModel("lz")


def trained(n=6, forecast=10.0, observed=10.5):
    model = m()
    for i in range(n):
        pair(model, i, forecast=forecast, observed=observed)
    return model


class TestNumericGuards:
    def test_exact_forecast(self):
        model = m()
        pair(model, 0, forecast=10.0, observed=10.0)
        assert model._state.general.abs_error.value == pytest.approx(0.0, abs=1e-9)

    def test_negative_temperatures(self):
        model = m()
        pair(model, 0, forecast=-5.0, observed=-3.0)
        assert model._state.general.bias.value == pytest.approx(2.0, abs=0.05)

    def test_zero_horizon(self):
        model = m()
        model.register_decision(decision("d", 10.0, created_offset_h=0.0))
        r = model.update(evaluation("e", "d", 11.0, created_offset_h=0.0, horizon_h=0.0))
        assert r.accepted  # zero horizon allowed (min_horizon_s=0)

    def test_trust_finite_empty(self):
        tr = m().predict_trust(ForecastPredictionContext())
        assert math.isfinite(tr.trust)

    def test_small_error(self):
        model = m()
        pair(model, 0, forecast=10.0, observed=10.01)
        assert model._state.general.abs_error.value < 0.1


class TestNoForecastSafety:
    def test_empty_state(self):
        assert m()._state.general.sample_count == 0

    def test_prediction_neutral_unavailable(self):
        tr = m().predict_trust(ForecastPredictionContext(forecast_available=False))
        assert not tr.available

    def test_no_crash_unavailable_update(self):
        # an unavailable forecast must not be learned as a negative provider sample
        model = m()
        model.register_decision(decision("d", 10.0))
        # no matching forecast value via context, decision present -> learns from decision;
        # but with reality quality below min it is simply rejected, not a crash
        r = model.update(evaluation("e", "d", 10.5),
                         ForecastUpdateContext(source_decision_id="d", reality_quality=0.0))
        assert not r.accepted


class TestConfidenceIntegration:
    def _agg(self):
        return ConfidenceAggregator()

    def _forecast_component(self, model, status=ConfidenceStatus.HEALTHY):
        return ConfidenceComponent(kind=ComponentKind.FORECAST,
                                   contribution=model.confidence(), status=status)

    def test_forecast_contribution_for_forecast_purpose(self):
        c = trained().confidence()
        r = self._agg().aggregate(ConfidenceInput(
            ConfidencePurpose.FORECAST, components=(self._forecast_component(trained()),)))
        assert 0.0 <= r.value <= 1.0

    def test_forecast_optional_for_preheat_local_path(self):
        # local preheat path: missing forecast -> no forecast cap
        r = self._agg().aggregate(ConfidenceInput(ConfidencePurpose.PREHEAT, components=(
            ConfidenceComponent(kind=ComponentKind.HEAT_RATE,
                                contribution=_strong_local(),
                                evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),)))
        assert "forecast_unavailable" not in [c.cap.value for c in r.applied_caps]

    def test_forecast_dependent_path_capped_without_forecast(self):
        r = self._agg().aggregate(ConfidenceInput(
            ConfidencePurpose.PREHEAT, forecast_required=True, components=(
                ConfidenceComponent(kind=ComponentKind.HEAT_RATE, contribution=_strong_local(),
                                    evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),)))
        assert "forecast_unavailable" in [c.cap.value for c in r.applied_caps]

    def test_no_double_counting_weather_and_forecast(self):
        # forecast component declares both forecast+weather; correlation discount applies
        r = self._agg().aggregate(ConfidenceInput(ConfidencePurpose.FORECAST, components=(
            ConfidenceComponent(kind=ComponentKind.FORECAST, contribution=trained().confidence(),
                                evidence_domains=(EvidenceDomain.FORECAST, EvidenceDomain.WEATHER)),
            ConfidenceComponent(kind=ComponentKind.OUTDOOR, contribution=trained().confidence(),
                                evidence_domains=(EvidenceDomain.WEATHER,)))))
        assert "correlated_evidence_discounted" in r.reason_codes


class TestDiagnostics:
    def test_fields(self):
        d = trained().diagnostics()
        assert math.isfinite(d.general_trust) and d.model_version == 1
        assert d.general_bias_c > 0

    def test_open_expired_counts(self):
        model = trained()
        model.register_decision(decision("open1", 10.0, created_offset_h=99))
        assert model.diagnostics().open_decisions >= 1

    def test_rejection_counts(self):
        model = m()
        model.update(evaluation("e", "ghost", 11.0))  # missing decision
        assert sum(model.diagnostics().rejection_counts.values()) >= 1


class TestExport:
    def test_support_no_ids_or_weather_series(self):
        exp = trained().export(ExportScope.SUPPORT)
        assert "general_trust" in exp and "bias_c" in exp
        assert "samples" not in exp and "open_decisions" not in exp
        import json
        assert "decision_id" not in json.dumps(exp)

    def test_research_pseudonymized(self):
        exp = trained().export(ExportScope.RESEARCH)
        for s in exp["samples"]:
            assert "source_decision_id" not in s and "evaluation_id" not in s
            assert "absolute_error_c" in s and "horizon_bucket" in s

    def test_raw_not_provided(self):
        assert "unsupported" in trained().export(ExportScope.RAW)


class TestPurityRuntime:
    def test_input_not_mutated(self):
        d = decision("d", 10.0)
        ev = evaluation("e", "d", 11.0)
        model = m()
        model.register_decision(d)
        model.update(ev)
        assert d.forecast_high == 10.0 and ev.observed_temp_at_eval == 11.0

    def test_deterministic(self):
        assert trained()._state.general.bias.value == trained()._state.general.bias.value


def _strong_local():
    from custom_components.thermosmart.learning.contracts import ConfidenceContribution
    return ConfidenceContribution(value=0.9, evidence_count=50, reliability=0.9,
                                  evidence_domains=("thermal_trajectory",))
