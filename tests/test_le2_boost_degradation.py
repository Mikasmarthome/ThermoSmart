"""Phase 14 BoostModel TRV-only, no-temp, numeric guards, confidence-integration,
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
    EvidenceDomain,
)
from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
from custom_components.thermosmart.learning.models import (
    BoostModel,
    BoostPredictionContext,
)
from custom_components.thermosmart.learning.models.boost import DimensionStatus
from tests.helpers_boost import boost_context, boost_episode, good_boost


def m():
    return BoostModel("lz")


def trained(n=6):
    model = m()
    for i in range(n):
        good_boost(model, i)
    return model


class TestTrvOnly:
    def test_absolute_effect_without_optional_data(self):
        ep = boost_episode("e", [18.0, 19.5, 20.5, 21.0, 21.0])
        ev = m().evaluate_boost(ep, boost_context(ep))  # no outdoor/valve/forecast
        assert ev.heating_gain.computable and ev.heating_gain.score > 0.0

    def test_no_crash_minimal(self):
        assert good_boost(m(), 0).accepted

    def test_relative_conservative_without_baseline(self):
        ep = boost_episode("e", [18.0, 19.5, 20.5, 21.0, 21.0])
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.time_gain.status is DimensionStatus.NOT_COMPUTABLE


class TestNoTemperature:
    def test_no_invented_thermal_effect(self):
        ep = boost_episode("e", [None, None, None, None], reason=EpisodeReason.TIMEOUT)
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.heating_gain.score is None
        assert ev.effect.temperature_gain_c is None

    def test_partial_controller_evaluation(self):
        ep = boost_episode("e", [None, None, None, None], reason=EpisodeReason.TIMEOUT)
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.controller_quality.computable

    def test_reduced_or_rejected(self):
        ep = boost_episode("e", [None, None, None, None], reason=EpisodeReason.TIMEOUT)
        r = m().update(ep, boost_context(ep))
        # accepted only as a strongly reduced partial sample (or rejected) — never full
        if r.accepted:
            assert r.weight < 0.5


class TestNumericGuards:
    def test_zero_offset_rejected(self):
        ep = boost_episode("e", [18.0, 19.5, 21.0])
        r = m().update(ep, boost_context(ep, requested_offset_c=0.0))
        assert not r.accepted

    def test_negative_temperatures(self):
        ep = boost_episode("e", [-5.0, -3.0, -1.0, 0.0], start_temp=-5.0, target=0.0)
        ev = m().evaluate_boost(ep, boost_context(ep, start_deficit_c=5.0))
        assert math.isfinite(ev.heating_gain.score)

    def test_constant_temperature(self):
        ep = boost_episode("e", [20.0, 20.0, 20.0, 20.0], start_temp=20.0, target=21.0)
        ev = m().evaluate_boost(ep, boost_context(ep, start_deficit_c=1.0))
        assert ev.heating_gain.score == pytest.approx(0.0, abs=1e-6)

    def test_target_already_reached(self):
        ep = boost_episode("e", [21.0, 21.0, 21.0], start_temp=21.0, target=21.0)
        ev = m().evaluate_boost(ep, boost_context(ep, start_deficit_c=0.0))
        assert ev.target_achievement.score == 1.0

    def test_all_dimensions_in_range(self):
        for vals in ([18.0, 19.5, 21.0, 21.0], [19.0, 21.0, 23.0, 22.5]):
            ev = m().evaluate_boost(boost_episode("e", vals), boost_context(boost_episode("e", vals)))
            for n in ("heating_gain", "comfort_impact", "overshoot_penalty",
                      "controller_quality", "data_quality"):
                s = ev.get(n).score
                if s is not None:
                    assert 0.0 <= s <= 1.0


class TestConfidenceIntegration:
    def _agg(self):
        return ConfidenceAggregator()

    def test_boost_contribution(self):
        r = self._agg().aggregate(ConfidenceInput(ConfidencePurpose.BOOST, components=(
            ConfidenceComponent(kind=ComponentKind.BOOST, contribution=trained().confidence(),
                                evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            ConfidenceComponent(kind=ComponentKind.HEAT_RATE, contribution=_strong(),
                                evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)))))
        assert 0.0 <= r.value <= 1.0

    def test_correlated_outcome_trajectory_discounted(self):
        r = self._agg().aggregate(ConfidenceInput(ConfidencePurpose.BOOST, components=(
            ConfidenceComponent(kind=ComponentKind.HEAT_RATE, contribution=_strong(),
                                evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            ConfidenceComponent(kind=ComponentKind.OUTCOME, contribution=_strong(),
                                evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)))))
        assert "correlated_evidence_discounted" in r.reason_codes

    def test_independent_controller_evidence(self):
        r = self._agg().aggregate(ConfidenceInput(ConfidencePurpose.BOOST, components=(
            ConfidenceComponent(kind=ComponentKind.HEAT_RATE, contribution=_strong(),
                                evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),
            ConfidenceComponent(kind=ComponentKind.OUTCOME, contribution=_strong(),
                                evidence_domains=(EvidenceDomain.OUTCOME_HISTORY,)))))
        assert 0.0 <= r.value <= 1.0

    def test_low_attribution_limits_confidence(self):
        c = m().confidence()
        assert c.value <= 0.35  # cold start / no evidence


class TestDiagnostics:
    def test_fields(self):
        d = trained().diagnostics()
        assert math.isfinite(d.general_recommended_offset_c) and d.model_version == 1
        assert d.evaluation_version == 1

    def test_full_partial(self):
        model = trained()
        assert model.diagnostics().full_partial[0] >= 1

    def test_rejection_counts(self):
        model = m()
        ep = boost_episode("e", [18.0, 19.5, 21.0], confounder_flags=("window_open",))
        model.update(ep, boost_context(ep))
        assert sum(model.diagnostics().rejection_counts.values()) >= 1


class TestExport:
    def test_support_no_ids_or_traj(self):
        exp = trained().export(ExportScope.SUPPORT)
        assert "recommended_offset_c" in exp and "overshoot_c" in exp
        import json
        blob = json.dumps(exp)
        assert "samples" not in exp and "decision" not in blob

    def test_research_pseudonymized(self):
        exp = trained().export(ExportScope.RESEARCH)
        for s in exp["samples"]:
            assert "source_episode_id" not in s and "decision_id" not in s
            assert "overshoot_c" in s and "comfort" in s

    def test_raw_not_provided(self):
        assert "unsupported" in trained().export(ExportScope.RAW)


class TestPurityRuntime:
    def test_input_not_mutated(self):
        ep = boost_episode("e", [19.0, 21.0, 23.0, 22.5])
        before = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        m().update(ep, boost_context(ep))
        after = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        assert before == after

    def test_deterministic(self):
        assert trained()._state.general.effective_factor.value == \
            trained()._state.general.effective_factor.value


def _strong():
    from custom_components.thermosmart.learning.contracts import ConfidenceContribution
    return ConfidenceContribution(value=0.85, evidence_count=40, reliability=0.9,
                                  evidence_domains=("thermal_trajectory",))
