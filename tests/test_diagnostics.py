"""ThermoSmart diagnostics — Home Assistant's standard config-entry
diagnostics hook (custom_components/thermosmart/diagnostics.py).

Written for the export -> HA Diagnostics migration: the previous custom
export system (private file storage, an authenticated HTTP download view,
notifications, buttons, and a service) was removed entirely in favor of
HA's built-in "Download diagnostics" action. These tests exercise real
user-/support-relevant behavior of the replacement, not implementation
details of the removed delivery layer:
  - the "system" entry gets a small overview, each "zone" entry gets its
    own full diagnostics payload
  - the expected core diagnostic areas are present for a zone
  - the payload is privacy-safe (no entity ids/device ids/person data)
  - no file is written and no custom HTTP view is required — the function
    just returns a plain dict, which is HA's own contract for this hook
  - a zone with no coordinator attached yet (right after setup) is handled
    robustly, not by raising
  - missing optional data (no Learning shadow, no storage metadata) never
    raises — every block degrades to an explicit "unavailable" marker
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    async_get_config_entry_diagnostics,
)

_FORBIDDEN_SUBSTRINGS = (
    "entity_id", "device_id", "unique_id", "person.", "presence_persons",
    "latitude", "longitude", "climate.", "sensor.", "zone.home",
)


def _blob_has_no_forbidden_substrings(payload: dict) -> list[str]:
    blob = json.dumps(payload, default=str).lower()
    return [s for s in _FORBIDDEN_SUBSTRINGS if s in blob]


def _entry(entry_id: str, entry_type: str = "zone") -> MagicMock:
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.data = {"entry_type": entry_type}
    entry.options = {}
    return entry


def _base_hass(entries: list) -> MagicMock:
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = entries
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    hass.data = {DOMAIN: {}}
    return hass


class _FakeMetaStore:
    def __init__(self, data=None):
        self.data = data

    async def async_load(self):
        return self.data

    async def async_save(self, data):
        self.data = data

    async def async_remove(self):
        self.data = None


def _patch_metadata_factory(monkeypatch, stores_by_key: dict):
    """Make every StorageMetadataStore constructed anywhere read from
    ``stores_by_key`` instead of touching a real HA Store — hermetic, no
    file I/O, independent of whatever the test's MagicMock hass supports."""
    from custom_components.thermosmart.learning.storage.stores import HomeAssistantStoreFactory

    def _create(self, key, version):
        return stores_by_key.setdefault(key, _FakeMetaStore())

    monkeypatch.setattr(HomeAssistantStoreFactory, "create", _create)


class TestSystemEntryDiagnostics:
    async def test_returns_overview_with_zone_count_and_hashes(self, monkeypatch):
        system_entry = _entry("sys_1", "system")
        zone1 = _entry("zone_1")
        zone2 = _entry("zone_2")
        hass = _base_hass([system_entry, zone1, zone2])

        result = await async_get_config_entry_diagnostics(hass, system_entry)

        assert result["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
        assert "thermosmart_version" in result
        assert "ha_version" in result
        assert result["zone_count"] == 2
        assert len(result["zones"]) == 2
        assert all("zone_hash" in z for z in result["zones"])
        # zone hashes are deterministic per entry_id and distinct across zones
        assert result["zones"][0]["zone_hash"] != result["zones"][1]["zone_hash"]

    async def test_system_overview_does_not_include_zone_learning_data(self, monkeypatch):
        system_entry = _entry("sys_1", "system")
        zone1 = _entry("zone_1")
        hass = _base_hass([system_entry, zone1])

        result = await async_get_config_entry_diagnostics(hass, system_entry)
        assert "learning_progress" not in result
        assert "historical_learning_snapshot" not in result


class TestZoneEntryDiagnostics:
    async def test_no_coordinator_returns_full_shape_without_crashing(self, monkeypatch):
        zone = _entry("zone_1")
        hass = _base_hass([zone])
        _patch_metadata_factory(monkeypatch, {})

        result = await async_get_config_entry_diagnostics(hass, zone)

        for key in (
            "schema_version", "thermosmart_version", "zone_hash", "config_flags",
            "analytics", "historical_learning_snapshot", "runtime_state",
            "runtime_health", "runtime_pending", "learning_progress",
            "episode_history", "research_daily", "critical_events",
            "adaptation", "adaptation_history", "device_profile", "storage_context",
        ):
            assert key in result, f"missing diagnostics key: {key}"

        assert result["runtime_state"] is None
        assert result["runtime_health"] is None
        assert result["adaptation"] is None
        assert result["device_profile"] is None
        # Coordinator-less blocks degrade to an explicit unavailable marker,
        # never a raise and never a silently-omitted key.
        assert result["learning_progress"]["available"] is False
        assert result["episode_history"]["available"] is False
        assert result["research_daily"]["available"] is False
        assert result["critical_events"]["available"] is False
        assert result["adaptation_history"]["application_layer_status"] == "reserved"

    async def test_with_coordinator_populates_runtime_state(self, monkeypatch):
        zone = _entry("zone_1")
        hass = _base_hass([zone])
        _patch_metadata_factory(monkeypatch, {})

        coord = MagicMock()
        coord._active_control = True
        coord._learning_enabled = True
        coord._current_mode = "auto"
        coord.data = {"learning_confidence": 0.73}
        coord._learning_shadow = None  # simplest valid state — every helper falls back safely
        hass.data = {DOMAIN: {"zone_1": {"coordinator": coord}}}

        result = await async_get_config_entry_diagnostics(hass, zone)

        assert result["runtime_state"] == {
            "active_control": True,
            "learning_enabled": True,
            "current_mode": "auto",
            "confidence": 0.73,
        }
        # a coordinator with no learning shadow still yields explicit
        # unavailable markers, not a crash
        assert result["learning_progress"]["available"] is False
        assert result["device_profile"] is None

    async def test_reserved_application_layer_not_surfaced(self, monkeypatch):
        """The always-empty 'reserved' application/orchestration placeholder
        blocks (adaptation_application/orchestration_preview from the old
        export system) are deliberately dropped from diagnostics — see
        diagnostics.py's module docstring."""
        zone = _entry("zone_1")
        hass = _base_hass([zone])
        _patch_metadata_factory(monkeypatch, {})

        result = await async_get_config_entry_diagnostics(hass, zone)
        assert "reserved_diagnostics" not in result
        assert "adaptation_application" not in result
        assert "orchestration_preview" not in result

    async def test_runtime_models_raw_coefficients_not_surfaced(self, monkeypatch):
        """The raw Learning model-coefficient dump (formerly "runtime_models",
        built by the now-removed _learning_research_data()) is intentionally
        not part of diagnostics — internal research detail, not actionable
        for support. learning_progress is the human-readable equivalent."""
        zone = _entry("zone_1")
        hass = _base_hass([zone])
        _patch_metadata_factory(monkeypatch, {})

        result = await async_get_config_entry_diagnostics(hass, zone)
        assert "runtime_models" not in result

    async def test_full_zone_diagnostics_has_no_forbidden_identifiers(self, monkeypatch):
        zone = _entry("zone_1")
        hass = _base_hass([zone])
        _patch_metadata_factory(monkeypatch, {})

        result = await async_get_config_entry_diagnostics(hass, zone)
        assert _blob_has_no_forbidden_substrings(result) == []

    async def test_missing_storage_metadata_does_not_break_diagnostics(self, monkeypatch):
        from custom_components.thermosmart.learning.storage.stores import HomeAssistantStoreFactory

        def _boom(self, key, version):
            raise RuntimeError("simulated storage-metadata factory failure")

        monkeypatch.setattr(HomeAssistantStoreFactory, "create", _boom)

        zone = _entry("zone_1")
        hass = _base_hass([zone])

        result = await async_get_config_entry_diagnostics(hass, zone)
        assert result["storage_context"]["available"] is False
        assert result["storage_context"]["granularity"] == "store_level"
        # the rest of the diagnostics still completed
        assert result["historical_learning_snapshot"]["available"] is True

    async def test_result_is_a_plain_json_serializable_dict(self, monkeypatch):
        """HA's diagnostics contract is "return a dict" — nothing more. No
        file should be written and no custom view/endpoint is needed for
        this to work; json.dumps succeeding is a reasonable proxy for
        "safe to hand to HA's diagnostics download machinery"."""
        zone = _entry("zone_1")
        hass = _base_hass([zone])
        _patch_metadata_factory(monkeypatch, {})

        result = await async_get_config_entry_diagnostics(hass, zone)
        assert isinstance(result, dict)
        json.dumps(result, default=str)  # must not raise

    async def test_clock_from_coordinator_is_used_when_available(self, monkeypatch):
        zone = _entry("zone_1")
        hass = _base_hass([zone])
        _patch_metadata_factory(monkeypatch, {})

        coord = MagicMock()
        coord._clock.now_utc.return_value = datetime(2026, 1, 13, 8, 0, tzinfo=timezone.utc)
        coord._learning_shadow = None
        coord.data = {}
        hass.data = {DOMAIN: {"zone_1": {"coordinator": coord}}}

        result = await async_get_config_entry_diagnostics(hass, zone)
        assert result["generated_at_utc"] == "2026-01-13T08:00:00+00:00"
