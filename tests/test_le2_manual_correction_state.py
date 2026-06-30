"""Phase 15 ManualCorrectionModel state / serialize / migrate tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models import (
    ManualCorrectionModel,
    ManualCorrectionParameters,
)
from tests.helpers_manual_correction import correction_context, correction_event, recurring_direct


def m(params=None):
    return ManualCorrectionModel("lz", params)


def trained(n=6):
    model = m()
    recurring_direct(model, n)
    return model


class TestSerialize:
    def test_roundtrip(self):
        a = trained()
        s = a.serialize_state()
        json.dumps(s)
        b = m()
        b.deserialize_state(s)
        assert b._state.preference_bias_c == a._state.preference_bias_c
        assert b._state.general.signed_magnitude.value == a._state.general.signed_magnitude.value
        assert b._state.distinct_days == a._state.distinct_days

    def test_versions(self):
        s = trained().serialize_state()
        assert s["model_version"] == 1 and s["evaluation_version"] == 1

    def test_zero_bias_preserved(self):
        s = m().serialize_state()
        assert s["preference_bias_c"] == pytest.approx(0.0, abs=1e-9)


class TestValidate:
    def test_wrong_zone(self):
        with pytest.raises(ValueError):
            ManualCorrectionModel("other").deserialize_state(trained().serialize_state())

    def test_wrong_version(self):
        s = trained().serialize_state()
        s["model_version"] = 9
        assert any("model_version" in e for e in m().validate_state(s))

    def test_corrupted_nan_bias(self):
        s = trained().serialize_state()
        s["preference_bias_c"] = float("nan")
        assert any("non-finite" in e for e in m().validate_state(s))


class TestDedupResetMigrate:
    def test_dedup_bounded(self):
        model = m(ManualCorrectionParameters(dedup_max_ids=3))
        recurring_direct(model, 8)
        assert len(model._state.processed_ids) == 3

    def test_reset(self):
        model = trained()
        model.reset()
        assert model._state.general.sample_count == 0 and model._state.preference_bias_c == 0.0

    def test_migrate_identity(self):
        s = trained().serialize_state()
        assert m().migrate_state(1, 1, s) == s

    def test_migrate_unknown_raises(self):
        with pytest.raises(ValueError):
            m().migrate_state(0, 1, {})

    def test_research_history_bounded(self):
        model = m(ManualCorrectionParameters(research_sample_cap=4))
        recurring_direct(model, 10)
        assert len(model._state.recent_samples) == 4
