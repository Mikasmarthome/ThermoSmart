"""Phase 8 HeatRateModel state / rebuild / migrate tests."""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.learning.models import HeatRateModel, HeatRateParameters
from tests.helpers_heat_rate import heating_episode


def m(params=None):
    return HeatRateModel("lz", params)


def trained(n=3):
    model = m()
    for i in range(n):
        model.update(heating_episode(f"e{i}", 2.4, start_offset_min=i * 60))
    return model


class TestSerialize:
    def test_roundtrip(self):
        a = trained()
        state = a.serialize_state()
        # must be JSON-compatible
        json.dumps(state)
        b = m()
        b.deserialize_state(state)
        assert b._state.general.rate_c_per_h == a._state.general.rate_c_per_h
        assert b._state.processed_ids == a._state.processed_ids

    def test_versions_present(self):
        state = trained().serialize_state()
        assert state["model_version"] == 1 and state["parameter_version"] == 1

    def test_zero_preserved(self):
        state = m().serialize_state()
        assert state["general"]["rate_c_per_h"] == 0.0


class TestValidate:
    def test_wrong_zone_rejected(self):
        state = trained().serialize_state()
        other = HeatRateModel("other_zone")
        assert any("learning_zone_id" in e for e in other.validate_state(state))
        with pytest.raises(ValueError):
            other.deserialize_state(state)

    def test_wrong_version_rejected(self):
        state = trained().serialize_state()
        state["model_version"] = 999
        assert any("model_version" in e for e in m().validate_state(state))

    def test_corrupted_nan_rejected(self):
        state = trained().serialize_state()
        state["general"]["rate_c_per_h"] = float("nan")
        assert any("non-finite" in e for e in m().validate_state(state))


class TestDedup:
    def test_processed_ids_bounded(self):
        params = HeatRateParameters(dedup_max_ids=3, min_samples_for_outlier=999)
        model = m(params)
        for i in range(10):
            model.update(heating_episode(f"e{i}", 2.4, start_offset_min=i * 60))
        assert len(model._state.processed_ids) == 3


class TestResetRebuild:
    def test_reset(self):
        model = trained()
        model.reset()
        assert model._state.general.sample_count == 0

    def test_rebuild_equals_sequential(self):
        episodes = [heating_episode(f"e{i}", 2.0 + 0.1 * i, start_offset_min=i * 60)
                    for i in range(5)]
        seq = m()
        for ep in episodes:
            seq.update(ep)
        rb = m()
        result = rb.rebuild(None, episodes)
        assert result.accepted_count == 5 and result.processed_count == 5
        assert result.started_from_prior is True
        assert rb._state.general.rate_c_per_h == pytest.approx(
            seq._state.general.rate_c_per_h, abs=1e-9)

    def test_rebuild_skips_duplicates(self):
        episodes = [heating_episode("same", 2.4), heating_episode("same", 2.4, start_offset_min=60)]
        rb = m()
        result = rb.rebuild(None, episodes)
        assert result.duplicate_count == 1
        assert result.accepted_count + result.rejected_count == result.processed_count
        assert rb._state.general.sample_count == 1

    def test_rebuild_deterministic_order(self):
        eps = [heating_episode(f"e{i}", 2.0 + 0.1 * i, start_offset_min=i * 60) for i in range(4)]
        a = m(); a.rebuild(None, eps)
        b = m(); b.rebuild(None, list(reversed(eps)))  # same set, different input order
        assert a._state.general.rate_c_per_h == pytest.approx(b._state.general.rate_c_per_h)


class TestMigrate:
    def test_v1_identity(self):
        state = trained().serialize_state()
        assert m().migrate_state(1, 1, state) == state

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError):
            m().migrate_state(0, 1, {})
