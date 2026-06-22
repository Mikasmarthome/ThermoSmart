"""Phase 14 BoostModel deterministic evaluation + attribution tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
from custom_components.thermosmart.learning.models import (
    BoostModel,
    BoostUpdateContext,
)
from custom_components.thermosmart.learning.models.boost import DimensionStatus
from tests.helpers_boost import boost_context, boost_episode


def m():
    return BoostModel("lz")


class TestEvaluation:
    def test_sensible_boost_comfort_gain(self):
        ep = boost_episode("e", [18.0, 19.5, 20.5, 21.0, 21.0])
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.heating_gain.score > 0.5 and ev.comfort_impact.score > 0.8
        assert ev.target_achievement.score == 1.0

    def test_fast_reach_with_overshoot(self):
        ep = boost_episode("e", [19.0, 21.0, 23.0, 22.5])
        ev = m().evaluate_boost(ep, boost_context(ep, requested_offset_c=3.0))
        assert ev.target_achievement.score >= 0.8
        assert ev.comfort_impact.score < 0.6
        assert ev.overshoot_penalty.score < 0.6

    def test_boost_without_measurable_benefit(self):
        ep = boost_episode("e", [20.5, 20.6, 20.7, 20.8], start_temp=20.5, target=21.0)
        ev = m().evaluate_boost(ep, boost_context(ep, start_deficit_c=0.5))
        assert ev.heating_gain.score < 0.8

    def test_target_not_reached(self):
        ep = boost_episode("e", [18.0, 18.3, 18.6, 18.8], reason=EpisodeReason.TIMEOUT)
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.target_achievement.score < 0.5

    def test_timeout(self):
        ep = boost_episode("e", [18.0, 18.5, 19.0], reason=EpisodeReason.TIMEOUT)
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.controller_quality.score < 0.7

    def test_mode_change_controller_not_computable(self):
        ep = boost_episode("e", [18.0, 19.0, 20.0], reason=EpisodeReason.MODE_CHANGE)
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.controller_quality.status is DimensionStatus.NOT_COMPUTABLE

    def test_superseded_controller_not_computable(self):
        ep = boost_episode("e", [18.0, 19.0, 20.0], reason=EpisodeReason.SUPERSEDED)
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.controller_quality.status is DimensionStatus.NOT_COMPUTABLE

    def test_missing_baseline_time_gain_not_computable(self):
        ep = boost_episode("e", [18.0, 19.5, 21.0, 21.0])
        ev = m().evaluate_boost(ep, boost_context(ep))  # no expected_non_boost_duration
        assert ev.time_gain.status is DimensionStatus.NOT_COMPUTABLE
        assert "no_baseline" in ev.time_gain.reason_codes

    def test_baseline_enables_time_gain(self):
        ep = boost_episode("e", [18.0, 19.5, 20.5, 21.0, 21.0])
        ev = m().evaluate_boost(ep, boost_context(ep, expected_non_boost_duration_s=2400.0))
        assert ev.time_gain.computable

    def test_partial_evaluation_no_temp(self):
        ep = boost_episode("e", [None, None, None, None], reason=EpisodeReason.TIMEOUT)
        ev = m().evaluate_boost(ep, boost_context(ep))
        assert ev.heating_gain.status is DimensionStatus.NOT_COMPUTABLE
        assert ev.comfort_impact.status is DimensionStatus.NOT_COMPUTABLE
        assert ev.controller_quality.computable
        assert ev.partial

    def test_reproducible(self):
        ep = boost_episode("e", [19.0, 21.0, 23.0, 22.5])
        a = m().evaluate_boost(ep, boost_context(ep))
        b = m().evaluate_boost(ep, boost_context(ep))
        assert a.comfort_impact.score == b.comfort_impact.score

    def test_evaluation_does_not_mutate_state(self):
        model = m()
        ep = boost_episode("e", [18.0, 19.5, 21.0])
        before = model.serialize_state()
        model.evaluate_boost(ep, boost_context(ep))
        assert model.serialize_state() == before

    def test_version_present(self):
        from custom_components.thermosmart.learning.models import EVALUATION_VERSION
        ep = boost_episode("e", [18.0, 19.5, 21.0])
        assert m().evaluate_boost(ep, boost_context(ep)).evaluation_version == EVALUATION_VERSION


class TestAttribution:
    def test_confounder_reduces_attribution(self):
        clean = boost_episode("e", [18.0, 19.5, 21.0])
        dirty = boost_episode("e", [18.0, 19.5, 21.0], confounder_flags=("solar_gain",))
        a_clean = m().evaluate_boost(clean, boost_context(clean)).effect.attribution_reliability
        a_dirty = m().evaluate_boost(dirty, boost_context(dirty)).effect.attribution_reliability
        assert a_dirty < a_clean

    def test_relative_benefit_needs_baseline_and_attribution(self):
        ep = boost_episode("e", [18.0, 19.5, 20.5, 21.0, 21.0])
        no_base = m().evaluate_boost(ep, boost_context(ep))
        assert no_base.effect.relative_benefit is None

    def test_manual_correction_ref_lowers_attribution(self):
        ep = boost_episode("e", [18.0, 19.5, 21.0])
        base = m().evaluate_boost(ep, boost_context(ep)).effect.attribution_reliability
        mc = m().evaluate_boost(
            ep, boost_context(ep, manual_correction_ref="mc1")).effect.attribution_reliability
        assert mc < base
