"""Session-wide state in hass.data[DOMAIN] must not count as a loaded entry.

async_unload_entry() decides whether the shared LearningEngine can be released
by checking whether any config entry is still loaded. Keys that hold
session-wide flags (e.g. the bundled-card registration state) are not config
entries; counting them kept the LearningEngine in memory after the last zone
was unloaded.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components import thermosmart as ts_init
from custom_components.thermosmart.const import DOMAIN
from tests.helpers import make_mock_hass


def _zone_entry(entry_id: str) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.data = {"entry_type": "zone"}
    entry.options = {}
    return entry


def _hass(*, loaded_entry_ids, extra=None):
    hass = make_mock_hass()
    le = MagicMock()
    le.async_save = AsyncMock()
    hass.data = {
        DOMAIN: {
            "learning_engine": le,
            "card_registered": True,
            "card_static_registered": True,
            "card_frontend_registered": True,
            **{eid: {"type": "zone"} for eid in loaded_entry_ids},
            **(extra or {}),
        }
    }
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    return hass, le


class TestUnloadSessionState:
    async def test_last_zone_unload_releases_the_learning_engine(self):
        hass, le = _hass(loaded_entry_ids=["zone_1"])

        assert await ts_init.async_unload_entry(hass, _zone_entry("zone_1")) is True

        le.async_save.assert_awaited_once()
        assert "learning_engine" not in hass.data[DOMAIN]

    async def test_card_registration_state_survives_the_unload(self):
        hass, _ = _hass(loaded_entry_ids=["zone_1"])

        await ts_init.async_unload_entry(hass, _zone_entry("zone_1"))

        # static route and frontend URL stay registered for the HA session
        assert hass.data[DOMAIN]["card_registered"] is True
        assert hass.data[DOMAIN]["card_static_registered"] is True
        assert hass.data[DOMAIN]["card_frontend_registered"] is True

    async def test_learning_engine_is_kept_while_another_zone_is_loaded(self):
        hass, le = _hass(loaded_entry_ids=["zone_1", "zone_2"])

        await ts_init.async_unload_entry(hass, _zone_entry("zone_1"))

        assert hass.data[DOMAIN]["learning_engine"] is le
        le.async_save.assert_awaited_once()          # saved, but not released

    async def test_system_entry_alone_does_not_keep_the_learning_engine(self):
        hass, le = _hass(
            loaded_entry_ids=["zone_1"],
            extra={"sys_1": {"type": "system"}},
        )

        await ts_init.async_unload_entry(hass, _zone_entry("zone_1"))

        assert "learning_engine" not in hass.data[DOMAIN]


class TestConfigSchema:
    def test_integration_declares_a_config_entry_only_schema(self, caplog):
        """hassfest: integrations defining async_setup need a CONFIG_SCHEMA."""
        assert hasattr(ts_init, "CONFIG_SCHEMA")
        # no YAML configuration: passes through silently
        assert ts_init.CONFIG_SCHEMA({}) == {}
        assert "does not support YAML setup" not in caplog.text
        # YAML configuration for the domain is reported as unsupported
        ts_init.CONFIG_SCHEMA({DOMAIN: {"anything": 1}})
        assert "does not support YAML setup" in caplog.text
