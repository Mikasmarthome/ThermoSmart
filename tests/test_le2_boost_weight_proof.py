"""B2b-4b Item 1: single-weight proof through the REAL BoostModel.update() path.

Each independent cause influences the final adaptive update weight EXACTLY once
(Architecture B: confounder_reliability applied once to the weight; overall_reliability
is gate/diagnostic only and is NOT multiplied into the weight)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.models import BoostModel
from tests.helpers_boost import boost_episode, boost_context_with_comparison


def _weight(flags=()):
    """Drive the real update path and return the stored effective weight of the sample."""
    m = BoostModel("lz")
    ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1",
                       confounder_flags=flags)
    res = m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
    samples = m._state.recent_samples + m._state.diagnostic_samples
    return (samples[-1].effective_weight if samples else None), res


class TestSingleWeightProof:
    def test_clean_baseline_weight_positive(self):
        w, res = _weight(())
        assert res.accepted and w > 0

    def test_one_degrading_exactly_0_7(self):
        clean, _ = _weight(())
        one, _ = _weight(("occupancy_heat_gain",))
        assert one == pytest.approx(clean * 0.7, rel=1e-6)

    def test_two_independent_causes(self):
        clean, _ = _weight(())
        two, _ = _weight(("occupancy_heat_gain", "forecast_or_outdoor_shift"))
        assert two == pytest.approx(clean * 0.49, rel=1e-6)   # 0.7 * 0.7

    def test_blocking_confounder_no_adaptive_weight(self):
        # additional_radiator is BLOCKING -> CONFOUNDED -> non-adaptive (no factor move)
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1",
                           confounder_flags=("additional_radiator",))
        before = m._state.general.effective_factor.effective_n
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        assert m._state.general.effective_factor.effective_n == before  # nothing learned

    def test_interrupting_confounder_no_adaptive(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1",
                           confounder_flags=("sensor_gap",))
        before = m._state.general.effective_factor.effective_n
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        assert m._state.general.effective_factor.effective_n == before

    def test_attribution_is_confounder_free(self):
        # the single authority is confounder_reliability; attribution carries no confounder
        m = BoostModel("lz")
        clean = boost_episode("c", [18.0, 19.5, 21.0], decision_id="c")
        dirty = boost_episode("d", [18.0, 19.5, 21.0], decision_id="d",
                              confounder_flags=("occupancy_heat_gain",))
        a_clean = m.evaluate_boost(clean, boost_context_with_comparison(clean)).effect\
            .attribution_reliability
        a_dirty = m.evaluate_boost(dirty, boost_context_with_comparison(dirty)).effect\
            .attribution_reliability
        assert a_clean == pytest.approx(a_dirty)
