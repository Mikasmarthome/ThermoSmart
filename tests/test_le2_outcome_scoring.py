"""Phase 11 OutcomeModel deterministic evaluation / dimension separation tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
from custom_components.thermosmart.learning.models import (
    DimensionStatus,
    OutcomeModel,
    OutcomeUpdateContext,
)
from tests.helpers_outcome import (
    no_temp,
    not_reached,
    outcome_episode,
    reached_clean,
    reached_overshoot,
)


def m():
    return OutcomeModel("lz")


class TestEvaluation:
    def test_perfect_reach_no_overshoot(self):
        e = m().evaluate_outcome(reached_clean())
        assert e.dimensions.physical_target.score >= 0.9
        assert e.dimensions.comfort_quality.score >= 0.9

    def test_reach_with_overshoot_comfort_lower(self):
        e = m().evaluate_outcome(reached_overshoot())
        assert e.dimensions.physical_target.score >= 0.7   # physically reached
        assert e.dimensions.comfort_quality.score < 0.6     # but uncomfortable

    def test_reach_unstable(self):
        e = m().evaluate_outcome(outcome_episode("u", [19.0, 21.0, 20.0, 21.5, 20.0, 21.5]))
        assert e.dimensions.comfort_quality.score < 0.9

    def test_not_reached(self):
        e = m().evaluate_outcome(not_reached())
        assert e.dimensions.physical_target.score < 0.6

    def test_timeout_good_data(self):
        e = m().evaluate_outcome(not_reached())
        assert e.dimensions.data_quality.score > 0.4
        assert e.reason is EpisodeReason.TIMEOUT

    def test_timeout_without_temp_partial(self):
        e = m().evaluate_outcome(no_temp(reason=EpisodeReason.TIMEOUT))
        assert e.dimensions.physical_target.status is DimensionStatus.NOT_COMPUTABLE
        assert e.dimensions.comfort_quality.status is DimensionStatus.NOT_COMPUTABLE
        assert e.dimensions.controller_quality.computable

    def test_manual_correction(self):
        e = m().evaluate_outcome(reached_clean(reason=EpisodeReason.MANUAL_CORRECTION))
        assert e.dimensions.controller_quality.score < 0.7

    def test_superseded_not_attributable(self):
        e = m().evaluate_outcome(reached_clean(reason=EpisodeReason.SUPERSEDED))
        assert e.dimensions.controller_quality.status is DimensionStatus.NOT_COMPUTABLE

    def test_mode_change_neutral(self):
        e = m().evaluate_outcome(reached_clean(reason=EpisodeReason.MODE_CHANGE))
        assert e.dimensions.controller_quality.status is DimensionStatus.NOT_COMPUTABLE

    def test_reproducible(self):
        a = m().evaluate_outcome(reached_overshoot())
        b = m().evaluate_outcome(reached_overshoot())
        assert a.performance_score == b.performance_score
        assert a.dimensions.comfort_quality.score == b.dimensions.comfort_quality.score

    def test_scoring_version_present(self):
        from custom_components.thermosmart.learning.models import SCORING_VERSION
        assert m().evaluate_outcome(reached_clean()).scoring_version == SCORING_VERSION

    def test_evaluation_does_not_mutate_state(self):
        model = m()
        before = model.serialize_state()
        model.evaluate_outcome(reached_clean())
        assert model.serialize_state() == before


class TestDimensionSeparation:
    def test_physical_separate_from_comfort(self):
        e = m().evaluate_outcome(reached_overshoot())
        assert e.dimensions.physical_target.score != e.dimensions.comfort_quality.score

    def test_comfort_separate_from_controller(self):
        e = m().evaluate_outcome(reached_clean())
        assert e.dimensions.comfort_quality.name == "comfort_quality"
        assert e.dimensions.controller_quality.name == "controller_quality"

    def test_data_quality_not_in_performance(self):
        # performance is weighted over physical+comfort+controller only
        e = m().evaluate_outcome(reached_clean())
        dq = e.dimensions.data_quality.score
        perf = e.performance_score
        # lowering reliability lowers dq but performance unchanged for same trajectory
        e2 = m().evaluate_outcome(reached_clean(reliability=0.5))
        assert e2.dimensions.data_quality.score < dq
        assert e2.performance_score == perf

    def test_partial_dimensions(self):
        e = m().evaluate_outcome(no_temp())
        computable = [n for n in ("physical_target", "comfort_quality", "controller_quality")
                      if e.dimensions.get(n).computable]
        assert computable == ["controller_quality"]

    def test_not_computable_has_no_score(self):
        e = m().evaluate_outcome(no_temp())
        assert e.dimensions.physical_target.score is None

    def test_no_hidden_averaging_over_data_quality(self):
        # overall_reliability depends on data quality, performance_score does not
        e = m().evaluate_outcome(reached_clean(reliability=0.5))
        assert e.performance_score is not None
        assert e.overall_reliability < e.performance_score


class TestOverallScore:
    def test_performance_weighted(self):
        e = m().evaluate_outcome(reached_clean())
        assert 0.0 <= e.performance_score <= 1.0

    def test_overall_reliability_bounded_by_data_quality(self):
        e = m().evaluate_outcome(reached_clean(reliability=0.6))
        assert e.overall_reliability <= e.dimensions.data_quality.score + 1e-9

    def test_controller_only_performance_when_no_temp(self):
        e = m().evaluate_outcome(no_temp())
        # only controller contributes -> performance equals controller score
        assert e.performance_score == pytest.approx(
            e.dimensions.controller_quality.score, abs=1e-6)

    def test_avoidable_vs_physics_overshoot(self):
        # same overshoot, but expected afterheat explains it -> controller not penalised
        ep = reached_overshoot("ov")
        penalised = m().evaluate_outcome(ep)
        excused = m().evaluate_outcome(
            ep, OutcomeUpdateContext(source_episode_id="ov",
                                     afterheat_expected_overshoot_c=2.0))
        assert excused.dimensions.controller_quality.score > \
            penalised.dimensions.controller_quality.score
