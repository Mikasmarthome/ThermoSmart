"""Phase 13 ForecastModel update / bias / outlier / buckets / rebuild tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.models import (
    ForecastModel,
    ForecastParameters,
    ForecastRebuildItem,
    ForecastUpdateContext,
)
from tests.helpers_forecast import decision, evaluation, pair


def m(params=None):
    return ForecastModel("lz", params)


class TestUpdate:
    def test_first_sample(self):
        model = m()
        pair(model, 0, forecast=10.0, observed=11.0)
        assert model._state.general.sample_count == 1
        assert model._state.general.bias.value == pytest.approx(1.0, abs=0.01)

    def test_positive_bias(self):
        model = m()
        for i in range(5):
            pair(model, i, forecast=10.0, observed=11.0)
        assert model._state.general.bias.value > 0.5  # reality warmer than forecast

    def test_negative_bias(self):
        model = m()
        for i in range(5):
            pair(model, i, forecast=10.0, observed=9.0)
        assert model._state.general.bias.value < -0.5

    def test_perfect_forecast_low_error(self):
        model = m()
        for i in range(5):
            pair(model, i, forecast=10.0, observed=10.0)
        assert abs(model._state.general.bias.value) < 0.05
        assert model._state.general.abs_error.value < 0.05

    def test_absolute_error_tracked(self):
        model = m()
        pair(model, 0, forecast=10.0, observed=8.0)
        assert model._state.general.abs_error.value == pytest.approx(2.0, abs=0.05)

    def test_horizon_bucket_created(self):
        model = m()
        for i in range(6):
            pair(model, i, forecast=10.0, observed=10.5, horizon_h=6.0)
        assert model._state.horizon_buckets

    def test_source_bucket_via_context(self):
        model = m()
        for i in range(6):
            off = i * 24.0
            model.register_decision(decision(f"d{i}", 10.0, created_offset_h=off))
            ctx = ForecastUpdateContext(source_decision_id=f"d{i}", forecast_source_key="metno",
                                        horizon_s=6 * 3600)
            model.update(evaluation(f"e{i}", f"d{i}", 10.5, created_offset_h=off), ctx)
        assert any(k.startswith("src:") for k in model._state.source_buckets)

    def test_reality_quality_weighting(self):
        hi = m(); lo = m()
        hi.register_decision(decision("d", 10.0))
        hi.update(evaluation("e", "d", 12.0),
                  ForecastUpdateContext(source_decision_id="d", reality_quality=1.0))
        lo.register_decision(decision("d", 10.0))
        lo.update(evaluation("e", "d", 12.0),
                  ForecastUpdateContext(source_decision_id="d", reality_quality=0.4))
        assert hi._state.general.abs_error.effective_n > lo._state.general.abs_error.effective_n

    def test_severe_outlier_rejected(self):
        model = m()
        # varied small errors so the model has a non-zero dispersion of its own
        for i in range(8):
            pair(model, i, forecast=10.0, observed=10.1 + (0.1 if i % 2 else -0.1))
        before = model._state.general.abs_error.value
        r = pair(model, 99, forecast=10.0, observed=18.0)  # huge sudden error
        assert not r.accepted
        assert model._state.general.abs_error.value == before

    def test_single_evaluation_does_not_dominate(self):
        model = m(ForecastParameters(min_samples_for_outlier=999))
        for i in range(8):
            pair(model, i, forecast=10.0, observed=10.0)
        before = model._state.general.bias.value
        pair(model, 99, forecast=10.0, observed=13.0)
        assert abs(model._state.general.bias.value - before) < 1.0

    def test_two_zones_isolated(self):
        a = ForecastModel("lz_a"); b = ForecastModel("lz_b")
        a.register_decision(decision("d", 10.0, zone="lz_a"))
        a.update(evaluation("e", "d", 11.0, zone="lz_a"))
        assert a._state.general.sample_count == 1 and b._state.general.sample_count == 0


class TestContext:
    def test_roundtrip(self):
        ctx = ForecastUpdateContext(source_decision_id="d", forecast_temperature_c=10.0,
                                    actual_temperature_c=11.0, horizon_s=0.0,
                                    forecast_source_key="metno", reality_quality=1.0)
        assert ForecastUpdateContext.from_dict(ctx.to_dict()) == ctx

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            ForecastUpdateContext(source_decision_id="d", forecast_temperature_c=float("inf"))

    def test_zero_valid(self):
        ctx = ForecastUpdateContext(source_decision_id="d", horizon_s=0.0,
                                    forecast_temperature_c=0.0)
        assert ctx.forecast_temperature_c == 0.0


class TestRebuild:
    def _items(self, n=5):
        items = []
        for i in range(n):
            off = i * 24.0
            items.append(ForecastRebuildItem(
                decision=decision(f"d{i}", 10.0, created_offset_h=off),
                evaluation=evaluation(f"e{i}", f"d{i}", 10.5, created_offset_h=off)))
        return items

    def test_rebuild_equals_sequential(self):
        items = self._items()
        seq = m()
        for it in items:
            seq.register_decision(it.decision)
            seq.update(it.evaluation)
        rb = m()
        res = rb.rebuild(None, items)
        assert res.accepted_count == 5 and res.processed_count == 5
        assert rb._state.general.bias.value == pytest.approx(seq._state.general.bias.value)

    def test_rebuild_deterministic(self):
        items = self._items()
        a = m(); a.rebuild(None, items)
        b = m(); b.rebuild(None, items)
        assert a.serialize_state() == b.serialize_state()

    def test_rebuild_rejects_unlinked_evaluation(self):
        items = [ForecastRebuildItem(evaluation=evaluation("e", "ghost", 11.0))]
        rb = m()
        res = rb.rebuild(None, items)
        assert res.accepted_count == 0 and res.rejected_count == 1

    def test_rebuild_duplicates(self):
        d = decision("d", 10.0)
        items = [
            ForecastRebuildItem(decision=d, evaluation=evaluation("e", "d", 11.0)),
            ForecastRebuildItem(evaluation=evaluation("e", "d", 11.0, horizon_h=7.0)),
        ]
        rb = m()
        res = rb.rebuild(None, items)
        assert res.duplicate_count == 1 and rb._state.general.sample_count == 1
