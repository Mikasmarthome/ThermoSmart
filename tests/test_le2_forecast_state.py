"""Phase 13 ForecastModel state / serialize / migrate tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models import (
    ForecastModel,
    ForecastParameters,
)
from tests.helpers_forecast import decision, evaluation, pair


def m(params=None):
    return ForecastModel("lz", params)


def trained(n=5):
    model = m()
    for i in range(n):
        pair(model, i, forecast=10.0, observed=10.5)
    return model


class TestSerialize:
    def test_roundtrip(self):
        a = trained()
        s = a.serialize_state()
        json.dumps(s)
        b = m()
        b.deserialize_state(s)
        assert b._state.general.bias.value == a._state.general.bias.value
        assert b._state.general.abs_error.value == a._state.general.abs_error.value

    def test_versions(self):
        s = trained().serialize_state()
        assert s["model_version"] == 1 and s["parameter_version"] == 1

    def test_zero_preserved(self):
        model = m()
        pair(model, 0, forecast=10.0, observed=10.0)  # zero error
        s = model.serialize_state()
        assert s["general"]["bias"]["value"] == pytest.approx(0.0, abs=1e-9)

    def test_open_decisions_persisted(self):
        model = m()
        model.register_decision(decision("d1", 10.0))
        s = model.serialize_state()
        assert "d1" in s["open_decisions"]
        b = m(); b.deserialize_state(s)
        assert "d1" in b._state.open_decisions


class TestValidate:
    def test_wrong_zone(self):
        with pytest.raises(ValueError):
            ForecastModel("other").deserialize_state(trained().serialize_state())

    def test_wrong_version(self):
        s = trained().serialize_state()
        s["model_version"] = 9
        assert any("model_version" in e for e in m().validate_state(s))

    def test_corrupted_nan(self):
        s = trained().serialize_state()
        s["general"]["bias"]["value"] = float("nan")
        assert any("non-finite" in e for e in m().validate_state(s))


class TestDedupResetMigrate:
    def test_dedup_bounded(self):
        model = m(ForecastParameters(dedup_max_ids=3))
        for i in range(8):
            pair(model, i, forecast=10.0, observed=10.2)
        assert len(model._state.processed_eval_ids) == 3

    def test_reset(self):
        model = trained()
        model.reset()
        assert model._state.general.sample_count == 0 and model._state.horizon_buckets == {}

    def test_migrate_identity(self):
        s = trained().serialize_state()
        assert m().migrate_state(1, 1, s) == s

    def test_migrate_unknown_raises(self):
        with pytest.raises(ValueError):
            m().migrate_state(0, 1, {})

    def test_research_history_bounded(self):
        model = m(ForecastParameters(research_sample_cap=4))
        for i in range(10):
            pair(model, i, forecast=10.0, observed=10.2)
        assert len(model._state.recent_samples) == 4
