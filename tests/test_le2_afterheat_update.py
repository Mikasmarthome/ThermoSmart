"""Phase 10 AfterheatModel update / near-zero / outlier / inertia / rebuild tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.models import (
    AfterheatModel,
    AfterheatParameters,
    AfterheatRebuildItem,
    AfterheatRejection,
    AfterheatUpdateContext,
)
from tests.helpers_afterheat import afterheat_episode


def m(params=None):
    return AfterheatModel("lz", params)


class TestUpdate:
    def test_first_sample(self):
        model = m()
        assert model.update(afterheat_episode("e", 0.5)).accepted
        assert model._state.general.rise.value == pytest.approx(0.5, abs=0.05)

    def test_multiple_samples(self):
        model = m()
        for i, rise in enumerate([0.4, 0.5, 0.6]):
            model.update(afterheat_episode(f"e{i}", rise, start_offset_min=i * 30))
        assert 0.4 < model._state.general.rise.value < 0.6

    def test_reliability_and_drive_end_weighting(self):
        hi = m()
        hi.update(afterheat_episode("a", 0.4))
        hi.update(afterheat_episode("b", 1.0, reliability=0.9, start_offset_min=30))
        lo = m()
        lo.update(afterheat_episode("a", 0.4))
        lo.update(afterheat_episode("b", 1.0, reliability=0.45, start_offset_min=30))
        assert hi._state.general.rise.value > lo._state.general.rise.value

    def test_severe_outlier_rejected(self):
        model = m()
        for i, rise in enumerate([0.45, 0.5, 0.55, 0.5, 0.45]):
            model.update(afterheat_episode(f"e{i}", rise, start_offset_min=i * 30))
        before = model._state.general.rise.value
        r = model.update(afterheat_episode("out", 3.5, start_offset_min=300))
        assert not r.accepted and AfterheatRejection.SEVERE_OUTLIER.value in r.confounder_flags
        assert model._state.general.rise.value == before

    def test_single_episode_does_not_dominate(self):
        model = m(AfterheatParameters(min_samples_for_outlier=999))
        for i in range(8):
            model.update(afterheat_episode(f"e{i}", 0.5, start_offset_min=i * 30))
        before = model._state.general.rise.value
        model.update(afterheat_episode("spike", 2.0, start_offset_min=300))
        assert model._state.general.rise.value - before < 0.4

    def test_conditioned_bucket_via_context(self):
        model = m()
        for i in range(6):
            ep = afterheat_episode(f"e{i}", 0.5, start_offset_min=i * 30)
            ctx = AfterheatUpdateContext(source_episode_id=ep.episode_id, heating_duration_s=2400.0)
            model.update(ep, ctx)
        assert any(b.sample_count >= 5 for b in model._state.buckets.values())

    def test_without_context_only_general(self):
        model = m()
        model.update(afterheat_episode("e", 0.5))
        assert model._state.buckets == {}

    def test_context_mismatch(self):
        r = m().update(afterheat_episode("e1", 0.5),
                       AfterheatUpdateContext(source_episode_id="OTHER"))
        assert AfterheatRejection.CONTEXT_MISMATCH.value in r.confounder_flags

    def test_context_wrong_zone(self):
        r = m().update(afterheat_episode("e1", 0.5),
                       AfterheatUpdateContext(source_episode_id="e1", learning_zone_id="other"))
        assert AfterheatRejection.CONTEXT_MISMATCH.value in r.confounder_flags

    def test_context_roundtrip(self):
        ctx = AfterheatUpdateContext(source_episode_id="e", heating_duration_s=0.0,
                                     drive_end_source="valve_closing",
                                     drive_end_source_reliability=0.9)
        assert AfterheatUpdateContext.from_dict(ctx.to_dict()) == ctx

    def test_nan_context_rejected(self):
        with pytest.raises(ValueError):
            AfterheatUpdateContext(source_episode_id="e", heating_duration_s=float("inf"))


class TestNearZero:
    def test_near_zero_learned(self):
        model = m()
        r = model.update(afterheat_episode("z", 0.0))
        assert r.accepted and model._state.near_zero_count == 1
        assert model._state.general.rise.value == pytest.approx(0.0, abs=1e-6)

    def test_many_near_zero_keep_prediction_low(self):
        model = m()
        for i in range(10):
            model.update(afterheat_episode(f"z{i}", 0.0, start_offset_min=i * 30))
        from custom_components.thermosmart.learning.models import AfterheatPredictionContext
        pred = model.predict_afterheat(AfterheatPredictionContext())
        assert pred.residual_rise_c == pytest.approx(0.0, abs=0.05)

    def test_bad_data_not_learned_as_zero(self):
        # flat but too short -> rejected, not a near-zero sample
        r = m().update(afterheat_episode("z", 0.0, duration_min=1.0))
        assert not r.accepted
        assert m()._state.near_zero_count == 0

    def test_zero_preserved_not_truthiness(self):
        model = m()
        model.update(afterheat_episode("z", 0.0))
        assert model._state.general.rise.has_evidence  # 0.0 counts as evidence


class TestMultiZone:
    def test_two_zones_isolated(self):
        a = AfterheatModel("lz_a")
        b = AfterheatModel("lz_b")
        a.update(afterheat_episode("e", 0.5, zone="lz_a"))
        assert a._state.general.sample_count == 1 and b._state.general.sample_count == 0


class TestRebuild:
    def test_rebuild_equals_sequential(self):
        items = []
        for i in range(5):
            ep = afterheat_episode(f"e{i}", 0.5, start_offset_min=i * 30)
            ctx = AfterheatUpdateContext(source_episode_id=ep.episode_id, heating_duration_s=2400.0)
            items.append(AfterheatRebuildItem(episode=ep, context=ctx))
        seq = m()
        for it in items:
            seq.update(it.episode, it.context)
        rb = m()
        res = rb.rebuild(None, items)
        assert res.accepted_count == 5 and res.processed_count == 5
        assert rb._state.general.rise.value == pytest.approx(seq._state.general.rise.value)
        assert set(rb._state.buckets) == set(seq._state.buckets)

    def test_rebuild_without_context_no_buckets(self):
        eps = [afterheat_episode(f"e{i}", 0.5, start_offset_min=i * 30) for i in range(5)]
        rb = m()
        rb.rebuild(None, eps)
        assert rb._state.buckets == {}

    def test_rebuild_reproduces_near_zero(self):
        eps = [afterheat_episode(f"z{i}", 0.0, start_offset_min=i * 30) for i in range(4)]
        rb = m()
        rb.rebuild(None, eps)
        assert rb._state.near_zero_count == 4

    def test_rebuild_duplicates(self):
        eps = [afterheat_episode("same", 0.5), afterheat_episode("same", 0.5, start_offset_min=30)]
        rb = m()
        res = rb.rebuild(None, eps)
        assert res.duplicate_count == 1 and rb._state.general.sample_count == 1
