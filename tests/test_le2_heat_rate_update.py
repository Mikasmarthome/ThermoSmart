"""Phase 8 HeatRateModel update / outlier / inertia / multi-zone tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.models import (
    HeatRateModel,
    HeatRateParameters,
    HeatRatePredictionContext,
    HeatRateRebuildItem,
    HeatRateRejection,
    HeatRateUpdateContext,
)
from tests.helpers_heat_rate import heating_episode


def m(params=None):
    return HeatRateModel("lz", params)


class TestUpdate:
    def test_first_sample_sets_general(self):
        model = m()
        r = model.update(heating_episode("e1", 2.4))
        assert r.accepted
        assert model._state.general.rate_c_per_h == pytest.approx(2.4, abs=0.05)
        assert model._state.general.sample_count == 1

    def test_multiple_samples_average(self):
        model = m()
        for i, rate in enumerate([2.0, 2.4, 2.8]):
            model.update(heating_episode(f"e{i}", rate, start_offset_min=i * 60))
        assert 2.0 < model._state.general.rate_c_per_h < 2.8

    def test_reliability_weighting(self):
        hi = m()
        hi.update(heating_episode("a", 2.0))
        hi.update(heating_episode("b", 4.0, reliability=0.9, start_offset_min=60))
        lo = m()
        lo.update(heating_episode("a", 2.0))
        lo.update(heating_episode("b", 4.0, reliability=0.45, start_offset_min=60))
        # higher-reliability second sample pulls the rate further toward 4.0
        assert hi._state.general.rate_c_per_h > lo._state.general.rate_c_per_h

    def test_severe_outlier_rejected(self):
        model = m()
        for i, rate in enumerate([2.3, 2.4, 2.5, 2.4, 2.3]):  # build dispersion
            model.update(heating_episode(f"e{i}", rate, start_offset_min=i * 60))
        before = model._state.general.rate_c_per_h
        r = model.update(heating_episode("out", 9.0, start_offset_min=600))
        assert not r.accepted
        assert HeatRateRejection.SEVERE_OUTLIER.value in r.confounder_flags
        assert model._state.general.rate_c_per_h == before  # untouched
        assert model._state.outlier_counts.get("severe") == 1

    def test_single_episode_does_not_dominate(self):
        # disable outlier rejection to isolate inertia
        params = HeatRateParameters(min_samples_for_outlier=999)
        model = m(params)
        for i in range(8):
            model.update(heating_episode(f"e{i}", 2.4, start_offset_min=i * 60))
        before = model._state.general.rate_c_per_h
        model.update(heating_episode("spike", 6.0, start_offset_min=800))
        after = model._state.general.rate_c_per_h
        # one divergent sample moves the rate only modestly
        assert after - before < 1.0

    def test_conditioned_bucket_created_and_used(self):
        model = m()
        for i in range(6):
            ep = heating_episode(f"e{i}", 3.0, start_offset_min=i * 60)
            ctx = HeatRateUpdateContext(source_episode_id=ep.episode_id,
                                        outdoor_temperature_c=2.0)
            model.update(ep, ctx)
        assert any(b.sample_count >= 5 for b in model._state.buckets.values())
        pred = model.predict_heat_rate(HeatRatePredictionContext(
            current_temp=18.0, target=21.0, outdoor_temp=Measurement(2.0, DataQuality.OK)))
        assert pred.bucket.startswith("outdoor:")
        assert "conditioned_bucket" in pred.reason_codes

    def test_without_context_only_general(self):
        model = m()
        model.update(heating_episode("e", 2.4))  # no context
        assert model._state.buckets == {}
        assert model._state.general.sample_count == 1

    def test_zero_outdoor_is_valid_context(self):
        model = m()
        ep = heating_episode("e0", 3.0)
        model.update(ep, HeatRateUpdateContext(source_episode_id="e0", outdoor_temperature_c=0.0))
        assert model._state.buckets  # bucket created for outdoor 0.0

    def test_nan_outdoor_context_rejected(self):
        with pytest.raises(ValueError):
            HeatRateUpdateContext(source_episode_id="e", outdoor_temperature_c=float("nan"))

    def test_context_episode_mismatch_rejected(self):
        model = m()
        ep = heating_episode("e1", 2.4)
        ctx = HeatRateUpdateContext(source_episode_id="OTHER", outdoor_temperature_c=2.0)
        r = model.update(ep, ctx)
        assert not r.accepted
        assert HeatRateRejection.CONTEXT_EPISODE_MISMATCH.value in r.confounder_flags

    def test_wrong_zone_episode_rejected(self):
        model = HeatRateModel("lz")
        ep = heating_episode("e", 2.4, zone="other_zone")
        r = model.update(ep)
        assert not r.accepted
        assert HeatRateRejection.WRONG_ZONE.value in r.confounder_flags

    def test_context_roundtrip(self):
        ctx = HeatRateUpdateContext(source_episode_id="e1", outdoor_temperature_c=0.0,
                                    device_profile_key="cast_iron", wind_bucket="2")
        back = HeatRateUpdateContext.from_dict(ctx.to_dict())
        assert back == ctx

    def test_conditioned_rebuild_matches_sequential(self):
        items = []
        for i in range(6):
            ep = heating_episode(f"e{i}", 3.0, start_offset_min=i * 60)
            ctx = HeatRateUpdateContext(source_episode_id=ep.episode_id, outdoor_temperature_c=2.0)
            items.append(HeatRateRebuildItem(episode=ep, context=ctx))
        seq = m()
        for it in items:
            seq.update(it.episode, it.context)
        rb = m()
        rb.rebuild(None, items)
        assert set(rb._state.buckets.keys()) == set(seq._state.buckets.keys())
        for k in seq._state.buckets:
            assert rb._state.buckets[k].rate_c_per_h == pytest.approx(
                seq._state.buckets[k].rate_c_per_h)

    def test_rebuild_without_context_no_conditioned_buckets(self):
        eps = [heating_episode(f"e{i}", 3.0, start_offset_min=i * 60) for i in range(6)]
        rb = m()
        rb.rebuild(None, eps)
        assert rb._state.buckets == {}


class TestMultiZone:
    def test_two_zones_isolated(self):
        a = HeatRateModel("lz_a")
        b = HeatRateModel("lz_b")
        a.update(heating_episode("e", 2.4, zone="lz_a"))
        assert a._state.general.sample_count == 1
        assert b._state.general.sample_count == 0

    def test_multiple_trvs_one_zone_model(self):
        # several TRV bindings feed one zone episode -> a single zone model
        model = m()
        model.update(heating_episode("z1", 2.4))
        model.update(heating_episode("z2", 2.6, start_offset_min=60))
        assert model._state.general.sample_count == 2  # not split per TRV
