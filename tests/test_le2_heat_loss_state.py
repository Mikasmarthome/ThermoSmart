"""Phase 9 HeatLossModel state / rebuild / migrate tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models import HeatLossModel, HeatLossParameters
from tests.helpers_heat_loss import cooling_episode


def m(params=None):
    return HeatLossModel("lz", params)


def trained(n=3):
    model = m()
    for i in range(n):
        model.update(cooling_episode(f"e{i}", 1.0, start_offset_min=i * 60))
    return model


class TestSerialize:
    def test_roundtrip(self):
        a = trained()
        state = a.serialize_state()
        json.dumps(state)  # JSON-compatible
        b = m()
        b.deserialize_state(state)
        assert b._state.general.rate_c_per_h == a._state.general.rate_c_per_h
        assert b._state.normalized.key == a._state.normalized.key

    def test_zero_preserved(self):
        assert m().serialize_state()["general"]["rate_c_per_h"] == 0.0

    def test_versions(self):
        s = trained().serialize_state()
        assert s["model_version"] == 1 and s["parameter_version"] == 1


class TestValidate:
    def test_wrong_zone(self):
        s = trained().serialize_state()
        with pytest.raises(ValueError):
            HeatLossModel("other").deserialize_state(s)

    def test_wrong_version(self):
        s = trained().serialize_state()
        s["model_version"] = 7
        assert any("model_version" in e for e in m().validate_state(s))

    def test_corrupted_nan(self):
        s = trained().serialize_state()
        s["general"]["rate_c_per_h"] = float("nan")
        assert any("non-finite" in e for e in m().validate_state(s))


class TestDedup:
    def test_bounded(self):
        model = m(HeatLossParameters(dedup_max_ids=3, min_samples_for_outlier=999))
        for i in range(8):
            model.update(cooling_episode(f"e{i}", 1.0, start_offset_min=i * 60))
        assert len(model._state.processed_ids) == 3


class TestResetMigrate:
    def test_reset(self):
        model = trained()
        model.reset()
        assert model._state.general.sample_count == 0
        assert model._state.normalized.sample_count == 0

    def test_migrate_identity(self):
        s = trained().serialize_state()
        assert m().migrate_state(1, 1, s) == s

    def test_migrate_unknown_raises(self):
        with pytest.raises(ValueError):
            m().migrate_state(0, 1, {})
