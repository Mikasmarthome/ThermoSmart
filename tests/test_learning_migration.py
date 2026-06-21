"""Characterization tests for ThermoSmartStore._async_migrate_func (Welle 4c / Tier A).

The current storage version is 1 and the migration function is a pure NO-OP:
it returns the supplied data unchanged. These tests FREEZE that v1 baseline so
that any future LE 2.0 migration (v1 → v2) is introduced deliberately and
visibly — a failing test here will flag the moment migration behaviour changes.

No real Store I/O: ThermoSmartStore is constructed with a mocked hass.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning_engine import ThermoSmartStore
from custom_components.thermosmart.const import STORAGE_VERSION, STORAGE_KEY
from tests.helpers import make_mock_hass


def make_store() -> ThermoSmartStore:
    return ThermoSmartStore(make_mock_hass(), STORAGE_VERSION, STORAGE_KEY)


class TestMigrationBaseline:
    async def test_returns_same_object_identity(self):
        store = make_store()
        old = {"observations": {"z": [{"ts": "2025-01-01T00:00:00"}]}}
        result = await store._async_migrate_func(1, 0, old)
        # current behaviour: identical object handed straight back (no copy)
        assert result is old

    async def test_payload_unchanged(self):
        store = make_store()
        old = {
            "observations": {"z": [{"ts": "2025-01-01T00:00:00", "target": 21.0}]},
            "boost_factors": {"z": 0.9},
            "forecast_bias": {"z": 0.6},
        }
        before = {
            "observations": {"z": [{"ts": "2025-01-01T00:00:00", "target": 21.0}]},
            "boost_factors": {"z": 0.9},
            "forecast_bias": {"z": 0.6},
        }
        result = await store._async_migrate_func(1, 0, old)
        assert result == before

    async def test_empty_payload_passthrough(self):
        store = make_store()
        result = await store._async_migrate_func(1, 0, {})
        assert result == {}

    @pytest.mark.parametrize(
        "major,minor",
        [(1, 0), (1, 1), (0, 9), (2, 0)],
    )
    async def test_noop_regardless_of_version_numbers(self, major, minor):
        store = make_store()
        old = {"k": "v"}
        result = await store._async_migrate_func(major, minor, old)
        # no version-conditional branch exists yet → always unchanged
        assert result is old

    async def test_nested_structures_not_mutated(self):
        store = make_store()
        old = {"outcome_log": {"z": [{"outcome_score": 0.8, "controller": "ts"}]}}
        result = await store._async_migrate_func(1, 0, old)
        assert result["outcome_log"]["z"][0]["outcome_score"] == 0.8
        assert result["outcome_log"]["z"][0]["controller"] == "ts"

    def test_storage_version_is_one(self):
        # The migration baseline is anchored to STORAGE_VERSION == 1.
        assert STORAGE_VERSION == 1
