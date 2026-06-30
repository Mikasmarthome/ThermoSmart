"""Phase 9 HeatLossModel update / outlier / inertia / context / multi-zone tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.models import (
    HeatLossModel,
    HeatLossParameters,
    HeatLossRebuildItem,
    HeatLossRejection,
    HeatLossUpdateContext,
)
from tests.helpers_heat_loss import cooling_episode


def m(params=None):
    return HeatLossModel("lz", params)


class TestUpdate:
    def test_first_sample(self):
        model = m()
        r = model.update(cooling_episode("e", 1.0))
        assert r.accepted
        assert model._state.general.rate_c_per_h == pytest.approx(1.0, abs=0.05)

    def test_multiple_samples(self):
        model = m()
        for i, loss in enumerate([0.8, 1.0, 1.2]):
            model.update(cooling_episode(f"e{i}", loss, start_offset_min=i * 60))
        assert 0.8 < model._state.general.rate_c_per_h < 1.2

    def test_general_is_positive(self):
        model = m()
        model.update(cooling_episode("e", 1.5))
        assert model._state.general.rate_c_per_h > 0  # positive cooling magnitude

    def test_reliability_weighting(self):
        hi = m(); hi.update(cooling_episode("a", 0.8)); hi.update(cooling_episode("b", 2.0, reliability=0.9, start_offset_min=60))
        lo = m(); lo.update(cooling_episode("a", 0.8)); lo.update(cooling_episode("b", 2.0, reliability=0.45, start_offset_min=60))
        assert hi._state.general.rate_c_per_h > lo._state.general.rate_c_per_h

    def test_severe_outlier_rejected(self):
        model = m()
        for i, loss in enumerate([0.9, 1.0, 1.1, 1.0, 0.9]):
            model.update(cooling_episode(f"e{i}", loss, start_offset_min=i * 60))
        before = model._state.general.rate_c_per_h
        r = model.update(cooling_episode("out", 5.0, start_offset_min=600))
        assert not r.accepted
        assert HeatLossRejection.SEVERE_OUTLIER.value in r.confounder_flags
        assert model._state.general.rate_c_per_h == before

    def test_single_episode_does_not_dominate(self):
        model = m(HeatLossParameters(min_samples_for_outlier=999))
        for i in range(8):
            model.update(cooling_episode(f"e{i}", 1.0, start_offset_min=i * 60))
        before = model._state.general.rate_c_per_h
        model.update(cooling_episode("spike", 3.0, start_offset_min=800))
        assert model._state.general.rate_c_per_h - before < 0.6

    def test_without_outdoor_only_general(self):
        model = m()
        model.update(cooling_episode("e", 1.0))  # no context
        assert model._state.buckets == {}

    def test_conditioned_bucket_via_context(self):
        model = m()
        for i in range(6):
            ep = cooling_episode(f"e{i}", 1.0, start_offset_min=i * 60)
            ctx = HeatLossUpdateContext(source_episode_id=ep.episode_id, outdoor_temperature_c=2.0)
            model.update(ep, ctx)
        assert any(b.sample_count >= 5 for b in model._state.buckets.values())

    def test_normalized_coefficient_with_delta(self):
        model = m()
        ep = cooling_episode("e", 1.0)
        ctx = HeatLossUpdateContext(source_episode_id="e", outdoor_temperature_c=5.0,
                                    indoor_outdoor_delta_c=15.0)
        model.update(ep, ctx)
        assert model._state.normalized.has_evidence

    def test_small_delta_no_normalized_update(self):
        model = m()
        ep = cooling_episode("e", 1.0)
        ctx = HeatLossUpdateContext(source_episode_id="e", outdoor_temperature_c=20.0,
                                    indoor_outdoor_delta_c=1.0)  # below min_io_delta
        model.update(ep, ctx)
        assert not model._state.normalized.has_evidence

    def test_context_episode_mismatch(self):
        model = m()
        r = model.update(cooling_episode("e1", 1.0),
                         HeatLossUpdateContext(source_episode_id="OTHER", outdoor_temperature_c=2.0))
        assert HeatLossRejection.CONTEXT_EPISODE_MISMATCH.value in r.confounder_flags

    def test_zero_outdoor_valid(self):
        model = m()
        ep = cooling_episode("e0", 1.0)
        model.update(ep, HeatLossUpdateContext(source_episode_id="e0", outdoor_temperature_c=0.0))
        assert model._state.buckets

    def test_nan_outdoor_rejected_at_context(self):
        with pytest.raises(ValueError):
            HeatLossUpdateContext(source_episode_id="e", outdoor_temperature_c=float("inf"))

    def test_context_roundtrip(self):
        ctx = HeatLossUpdateContext(source_episode_id="e", outdoor_temperature_c=0.0,
                                    indoor_outdoor_delta_c=10.0, wind_bucket="1")
        assert HeatLossUpdateContext.from_dict(ctx.to_dict()) == ctx


class TestRebuild:
    def test_rebuild_equals_sequential(self):
        items = []
        for i in range(6):
            ep = cooling_episode(f"e{i}", 1.0, start_offset_min=i * 60)
            ctx = HeatLossUpdateContext(source_episode_id=ep.episode_id, outdoor_temperature_c=2.0)
            items.append(HeatLossRebuildItem(episode=ep, context=ctx))
        seq = m()
        for it in items:
            seq.update(it.episode, it.context)
        rb = m()
        res = rb.rebuild(None, items)
        assert res.accepted_count == 6 and res.processed_count == 6
        assert set(rb._state.buckets) == set(seq._state.buckets)

    def test_rebuild_without_context_no_buckets(self):
        eps = [cooling_episode(f"e{i}", 1.0, start_offset_min=i * 60) for i in range(6)]
        rb = m()
        rb.rebuild(None, eps)
        assert rb._state.buckets == {}

    def test_rebuild_duplicates(self):
        eps = [cooling_episode("same", 1.0), cooling_episode("same", 1.0, start_offset_min=60)]
        rb = m()
        res = rb.rebuild(None, eps)
        assert res.duplicate_count == 1
        assert rb._state.general.sample_count == 1


class TestMultiZone:
    def test_two_zones_isolated(self):
        a = HeatLossModel("lz_a"); b = HeatLossModel("lz_b")
        a.update(cooling_episode("e", 1.0, zone="lz_a"))
        assert a._state.general.sample_count == 1 and b._state.general.sample_count == 0
