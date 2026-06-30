"""Phase 10 AfterheatModel state / migrate tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models import AfterheatModel, AfterheatParameters
from tests.helpers_afterheat import afterheat_episode


def m(params=None):
    return AfterheatModel("lz", params)


def trained(n=3):
    model = m()
    for i in range(n):
        model.update(afterheat_episode(f"e{i}", 0.5, start_offset_min=i * 30))
    return model


class TestSerialize:
    def test_roundtrip(self):
        a = trained()
        state = a.serialize_state()
        json.dumps(state)
        b = m()
        b.deserialize_state(state)
        assert b._state.general.rise.value == a._state.general.rise.value
        assert b._state.general.duration.value == a._state.general.duration.value

    def test_zero_preserved(self):
        model = m()
        model.update(afterheat_episode("z", 0.0))
        s = model.serialize_state()
        assert s["general"]["rise"]["value"] == pytest.approx(0.0, abs=1e-9)
        b = m(); b.deserialize_state(s)
        assert b._state.general.rise.has_evidence

    def test_versions_and_counts(self):
        s = trained().serialize_state()
        assert s["model_version"] == 1
        assert "inferred_drive_end_count" in s and "near_zero_count" in s


class TestValidate:
    def test_wrong_zone(self):
        with pytest.raises(ValueError):
            AfterheatModel("other").deserialize_state(trained().serialize_state())

    def test_wrong_version(self):
        s = trained().serialize_state()
        s["model_version"] = 9
        assert any("model_version" in e for e in m().validate_state(s))

    def test_corrupted_nan(self):
        s = trained().serialize_state()
        s["general"]["rise"]["value"] = float("nan")
        assert any("non-finite" in e for e in m().validate_state(s))


class TestDedupResetMigrate:
    def test_dedup_bounded(self):
        model = m(AfterheatParameters(dedup_max_ids=3, min_samples_for_outlier=999))
        for i in range(8):
            model.update(afterheat_episode(f"e{i}", 0.5, start_offset_min=i * 30))
        assert len(model._state.processed_ids) == 3

    def test_reset(self):
        model = trained()
        model.reset()
        assert model._state.general.sample_count == 0 and model._state.near_zero_count == 0

    def test_migrate_identity(self):
        s = trained().serialize_state()
        assert m().migrate_state(1, 1, s) == s

    def test_migrate_unknown_raises(self):
        with pytest.raises(ValueError):
            m().migrate_state(0, 1, {})
