"""Phase 11 OutcomeModel TRV-only / numeric guards / diagnostics / export tests."""
from __future__ import annotations

import math

import pytest

from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
from custom_components.thermosmart.learning.models import (
    DimensionStatus,
    OutcomeModel,
    OutcomeUpdateContext,
)
from tests.helpers_outcome import (
    no_temp,
    outcome_episode,
    reached_clean,
    reached_overshoot,
)


def m():
    return OutcomeModel("lz")


def trained(n=4):
    model = m()
    for i in range(n):
        model.update(reached_clean(f"e{i}"))
    return model


class TestTrvOnly:
    def test_full_evaluation_trv_only(self):
        # only target, temperature, setpoint/controller and time -> sensible eval
        e = m().evaluate_outcome(reached_clean())
        assert e.performance_score is not None
        assert e.dimensions.data_quality.missing_evidence  # optional context missing noted

    def test_no_crash_without_optional_data(self):
        assert m().update(reached_clean()).accepted


class TestNoActualTemperature:
    def test_physical_not_computable(self):
        e = m().evaluate_outcome(no_temp())
        assert e.dimensions.physical_target.status is DimensionStatus.NOT_COMPUTABLE

    def test_comfort_not_computable(self):
        e = m().evaluate_outcome(no_temp())
        assert e.dimensions.comfort_quality.status is DimensionStatus.NOT_COMPUTABLE

    def test_controller_partial(self):
        e = m().evaluate_outcome(no_temp(reason=EpisodeReason.TIMEOUT))
        assert e.dimensions.controller_quality.computable

    def test_low_data_quality(self):
        e = m().evaluate_outcome(no_temp())
        assert e.dimensions.data_quality.score < 0.6

    def test_no_invented_success(self):
        e = m().evaluate_outcome(no_temp(reason=EpisodeReason.TIMEOUT))
        assert e.dimensions.physical_target.score is None


class TestNumericGuards:
    def test_exact_target(self):
        e = m().evaluate_outcome(outcome_episode("x", [21.0, 21.0, 21.0, 21.0]))
        assert math.isfinite(e.performance_score)

    def test_negative_temperatures(self):
        e = m().evaluate_outcome(outcome_episode("n", [-5.0, -3.0, -1.0, 0.0], target=0.0,
                                                 start_temp=-5.0))
        assert math.isfinite(e.dimensions.physical_target.score)

    def test_oscillation(self):
        e = m().evaluate_outcome(outcome_episode("o", [21.0, 19.0, 23.0, 19.0, 23.0, 19.0]))
        assert e.dimensions.comfort_quality.score < 0.9

    def test_late_drop(self):
        e = m().evaluate_outcome(outcome_episode("l", [21.0, 21.0, 21.0, 18.0]))
        assert math.isfinite(e.dimensions.comfort_quality.score)

    def test_all_dimensions_in_range(self):
        for ep in (reached_clean(), reached_overshoot(), no_temp()):
            e = m().evaluate_outcome(ep)
            for n in ("physical_target", "comfort_quality", "controller_quality", "data_quality"):
                d = e.dimensions.get(n)
                if d.score is not None:
                    assert 0.0 <= d.score <= 1.0


class TestDiagnostics:
    def test_fields(self):
        d = trained().diagnostics()
        assert d.general_physical > 0 and d.model_version == 1 and d.scoring_version == 1
        assert d.reached_rate is not None

    def test_full_partial(self):
        model = m()
        model.update(reached_clean("a"))
        model.update(no_temp("b", reason=EpisodeReason.TIMEOUT))
        assert model.diagnostics().full_partial == (1, 1)

    def test_rejection_counts(self):
        model = m()
        model.update(reached_clean("x", reliability=0.05))  # low data quality
        assert sum(model.diagnostics().rejection_counts.values()) >= 1


class TestExport:
    def test_support_no_raw_or_ids(self):
        exp = trained().export(ExportScope.SUPPORT)
        assert "physical" in exp and "reached_rate" in exp and "scoring_version" in exp
        assert "samples" not in exp and "learning_zone_id" not in exp

    def test_research_dimensions_visible(self):
        exp = trained().export(ExportScope.RESEARCH)
        assert exp["scoring_version"] == 1
        for s in exp["samples"]:
            assert "source_episode_id" not in s and "learning_zone_id" not in s
            assert "physical" in s and "comfort" in s and "controller" in s

    def test_raw_not_provided(self):
        assert "unsupported" in trained().export(ExportScope.RAW)


class TestPurityRuntime:
    def test_input_not_mutated(self):
        ep = reached_overshoot()
        before = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        m().update(ep)
        after = tuple((p.offset_ms, p.value) for p in ep.trajectory.points)
        assert before == after

    def test_deterministic(self):
        assert trained()._state.general.physical.value == \
            trained()._state.general.physical.value
