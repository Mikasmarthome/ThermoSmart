"""Phase 14 BoostModel state / serialize / migrate tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models import BoostModel, BoostParameters
from tests.helpers_boost import boost_context, boost_episode, good_boost


def m(params=None):
    return BoostModel("lz", params)


def trained(n=5):
    model = m()
    for i in range(n):
        good_boost(model, i)
    return model


class TestSerialize:
    def test_roundtrip(self):
        a = trained()
        s = a.serialize_state()
        json.dumps(s)
        b = m()
        b.deserialize_state(s)
        assert b._state.general.effective_factor.value == a._state.general.effective_factor.value
        assert b._state.general.overshoot.value == a._state.general.overshoot.value

    def test_versions(self):
        s = trained().serialize_state()
        assert s["model_version"] == 1 and s["evaluation_version"] == 1

    def test_zero_preserved(self):
        model = m()
        # offset scaled by comfort can be 0 if comfort 0; use a clean zero-gain case
        ep = boost_episode("z", [21.0, 21.0, 21.0, 21.0], start_temp=21.0, target=21.0)
        model.update(ep, boost_context(ep, requested_offset_c=1.5, start_deficit_c=0.0))
        s = model.serialize_state()
        assert s["general"]["gain"]["value"] == pytest.approx(0.0, abs=1e-9)


class TestValidate:
    def test_wrong_zone(self):
        with pytest.raises(ValueError):
            BoostModel("other").deserialize_state(trained().serialize_state())

    def test_wrong_version(self):
        s = trained().serialize_state()
        s["model_version"] = 9
        assert any("model_version" in e for e in m().validate_state(s))

    def test_corrupted_nan(self):
        s = trained().serialize_state()
        s["general"]["effective_factor"]["value"] = float("nan")
        assert any("non-finite" in e for e in m().validate_state(s))


class TestDedupResetMigrate:
    def test_dedup_bounded(self):
        model = m(BoostParameters(dedup_max_ids=3))
        for i in range(8):
            good_boost(model, i)
        assert len(model._state.processed_ids) == 3

    def test_reset(self):
        model = trained()
        model.reset()
        assert model._state.general.sample_count == 0 and model._state.buckets == {}

    def test_migrate_identity(self):
        s = trained().serialize_state()
        assert m().migrate_state(1, 1, s) == s

    def test_migrate_unknown_raises(self):
        with pytest.raises(ValueError):
            m().migrate_state(0, 1, {})

    def test_research_history_bounded(self):
        model = m(BoostParameters(research_sample_cap=4))
        for i in range(10):
            good_boost(model, i)
        assert len(model._state.recent_samples) == 4
