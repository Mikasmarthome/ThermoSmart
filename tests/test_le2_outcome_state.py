"""Phase 11 OutcomeModel state / serialize / migrate tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models import (
    OutcomeModel,
    OutcomeParameters,
    OutcomeUpdateContext,
)
from tests.helpers_outcome import reached_clean


def m(params=None):
    return OutcomeModel("lz", params)


def trained(n=4, controller_kind="ts"):
    model = m()
    for i in range(n):
        ep = reached_clean(f"e{i}")
        model.update(ep, OutcomeUpdateContext(source_episode_id=ep.episode_id,
                                              controller_kind=controller_kind))
    return model


class TestSerialize:
    def test_roundtrip(self):
        a = trained()
        s = a.serialize_state()
        json.dumps(s)
        b = m()
        b.deserialize_state(s)
        assert b._state.general.physical.value == a._state.general.physical.value
        assert set(b._state.buckets) == set(a._state.buckets)

    def test_versions(self):
        s = trained().serialize_state()
        assert s["model_version"] == 1 and s["scoring_version"] == 1

    def test_zero_preserved(self):
        model = m()
        # controller-only partial keeps data_quality but physical sample 0 initially
        s = model.serialize_state()
        assert s["general"]["physical"]["value"] == pytest.approx(0.0, abs=1e-9)


class TestValidate:
    def test_wrong_zone(self):
        with pytest.raises(ValueError):
            OutcomeModel("other").deserialize_state(trained().serialize_state())

    def test_wrong_version(self):
        s = trained().serialize_state()
        s["model_version"] = 9
        assert any("model_version" in e for e in m().validate_state(s))

    def test_corrupted_nan(self):
        s = trained().serialize_state()
        s["general"]["physical"]["value"] = float("nan")
        assert any("non-finite" in e for e in m().validate_state(s))


class TestDedupResetMigrate:
    def test_dedup_bounded(self):
        model = m(OutcomeParameters(dedup_max_ids=3))
        for i in range(8):
            model.update(reached_clean(f"e{i}"))
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
        model = m(OutcomeParameters(research_sample_cap=5))
        for i in range(12):
            model.update(reached_clean(f"e{i}"))
        assert len(model._state.recent_samples) == 5
