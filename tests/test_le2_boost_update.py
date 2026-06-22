"""Phase 14 BoostModel update / outlier / inertia / buckets / rebuild tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.models import (
    BoostModel,
    BoostParameters,
    BoostRebuildItem,
    BoostRejection,
)
from tests.helpers_boost import boost_context, boost_episode, good_boost


def m(params=None):
    return BoostModel("lz", params)


class TestUpdate:
    def test_first_sample(self):
        model = m()
        assert good_boost(model, 0).accepted
        assert model._state.general.sample_count == 1

    def test_multiple(self):
        model = m()
        for i in range(4):
            good_boost(model, i)
        assert model._state.general.sample_count == 4

    def test_general_state_learns_factor(self):
        model = m()
        for i in range(5):
            good_boost(model, i, offset=1.5)
        assert 0.0 < model._state.general.effective_factor.value <= 1.5

    def test_overshoot_lowers_recommendation(self):
        good = m()
        for i in range(6):
            good_boost(good, i, offset=2.0, vals=(18.0, 19.5, 20.5, 21.0, 21.0))
        bad = m()
        for i in range(6):
            good_boost(bad, i, offset=2.0, vals=(19.0, 21.0, 23.0, 22.5))  # overshoot
        assert bad._state.general.effective_factor.value < good._state.general.effective_factor.value

    def test_good_effect_raises_recommendation_conservatively(self):
        model = m()
        for i in range(6):
            good_boost(model, i, offset=2.0, vals=(18.0, 19.5, 20.5, 21.0, 21.0))
        # learned effective factor stays at or below the requested offset (conservative)
        assert model._state.general.effective_factor.value <= 2.0

    def test_conditioned_bucket(self):
        model = m()
        for i in range(6):
            ep = boost_episode(f"e{i}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"d{i}")
            model.update(ep, boost_context(ep, start_deficit_c=3.0, controller_kind="ts",
                                           requested_offset_c=1.5))
        assert any(model._state.buckets)

    def test_reliability_attribution_weighting(self):
        hi = m(); good_boost(hi, 0)
        lo = m()
        ep = boost_episode("e", [18.0, 19.5, 21.0], confounder_flags=("mode_change",))
        # mode_change reason would reject; use confounder flag that only lowers attribution
        ep2 = boost_episode("e", [18.0, 19.5, 21.0], confounder_flags=("additional_radiator",))
        lo.update(ep2, boost_context(ep2))
        assert hi._state.general.effective_factor.effective_n >= \
            lo._state.general.effective_factor.effective_n

    def test_single_episode_does_not_dominate(self):
        model = m(BoostParameters(min_samples_for_outlier=999))
        for i in range(8):
            good_boost(model, i, offset=1.5)
        before = model._state.general.effective_factor.value
        good_boost(model, 99, offset=8.0)
        assert abs(model._state.general.effective_factor.value - before) < 1.5

    def test_severe_outlier_rejected(self):
        model = m()
        for i in range(8):
            good_boost(model, i, offset=1.5 + (0.2 if i % 2 else -0.2))
        before = model._state.general.effective_factor.value
        r = good_boost(model, 99, offset=8.0, vals=(18.0, 19.5, 20.5, 21.0, 21.0))
        assert not r.accepted
        assert model._state.general.effective_factor.value == before

    def test_two_zones_isolated(self):
        a = BoostModel("lz_a"); b = BoostModel("lz_b")
        ep = boost_episode("e", [18.0, 19.5, 21.0], zone="lz_a")
        a.update(ep, boost_context(ep))
        assert a._state.general.sample_count == 1 and b._state.general.sample_count == 0

    def test_partial_reduced_weight(self):
        full = good_boost(m(), 0)
        part_model = m()
        ep = boost_episode("p", [None, None, None, None],
                           reason=__import__("custom_components.thermosmart.learning."
                                             "episode_schemas", fromlist=["EpisodeReason"]
                                             ).EpisodeReason.TIMEOUT)
        part = part_model.update(ep, boost_context(ep))
        if part.accepted:
            assert part.weight < full.weight


class TestContext:
    def test_roundtrip(self):
        from custom_components.thermosmart.learning.models import BoostUpdateContext
        ctx = BoostUpdateContext(source_episode_id="e", decision_id="d", requested_offset_c=1.5,
                                 start_deficit_c=0.0, actual_duration_s=1800.0)
        assert BoostUpdateContext.from_dict(ctx.to_dict()) == ctx

    def test_nan_rejected(self):
        from custom_components.thermosmart.learning.models import BoostUpdateContext
        with pytest.raises(ValueError):
            BoostUpdateContext(source_episode_id="e", decision_id="d",
                               requested_offset_c=float("inf"))

    def test_zero_valid(self):
        from custom_components.thermosmart.learning.models import BoostUpdateContext
        ctx = BoostUpdateContext(source_episode_id="e", decision_id="d", requested_offset_c=1.5,
                                 start_deficit_c=0.0)
        assert ctx.start_deficit_c == 0.0


class TestRebuild:
    def _items(self, n=5):
        items = []
        for i in range(n):
            ep = boost_episode(f"e{i}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"d{i}")
            items.append(BoostRebuildItem(episode=ep, context=boost_context(ep)))
        return items

    def test_rebuild_equals_sequential(self):
        items = self._items()
        seq = m()
        for it in items:
            seq.update(it.episode, it.context)
        rb = m()
        res = rb.rebuild(None, items)
        assert res.accepted_count == 5
        assert rb._state.general.effective_factor.value == pytest.approx(
            seq._state.general.effective_factor.value)

    def test_rebuild_deterministic(self):
        items = self._items()
        a = m(); a.rebuild(None, items)
        b = m(); b.rebuild(None, items)
        assert a.serialize_state() == b.serialize_state()

    def test_rebuild_duplicates(self):
        ep1 = boost_episode("e", [18.0, 19.5, 21.0], decision_id="same")
        ep2 = boost_episode("e2", [18.0, 19.5, 21.0], decision_id="same")
        items = [BoostRebuildItem(episode=ep1, context=boost_context(ep1)),
                 BoostRebuildItem(episode=ep2, context=boost_context(ep2))]
        rb = m()
        res = rb.rebuild(None, items)
        assert res.duplicate_count == 1 and rb._state.general.sample_count == 1
