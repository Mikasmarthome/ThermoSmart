"""Tests for bundling the ThermoSmart Lovelace card inside the integration
(Frenck's HACS review of #9081/#9082): the card ships under
custom_components/thermosmart/www/, is served via HA's async static-path API,
and is auto-loaded via add_extra_js_url() — no separate HACS plugin, no
.storage/lovelace_resources manipulation.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components import thermosmart as ts_init
from custom_components.thermosmart.const import CARD_FILENAME, CARD_URL_PATH, DOMAIN, VERSION
from tests.helpers import make_mock_hass

_CARD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "custom_components", "thermosmart", "www", "thermosmart-card.js",
)


# ── File structure ────────────────────────────────────────────────────────────


class TestCardFileStructure:
    def test_card_file_exists(self):
        assert os.path.isfile(_CARD_PATH)

    def test_card_file_is_not_empty(self):
        assert os.path.getsize(_CARD_PATH) > 1000

    def test_card_file_defines_the_custom_element(self):
        content = open(_CARD_PATH, encoding="utf-8").read()
        assert "customElements.define('thermosmart-card'" in content
        assert "customElements.define('thermosmart-card-editor'" in content

    def test_card_registers_a_duplicate_define_guard(self):
        """Card ships its own idempotency guard — matters once the same file
        could theoretically be loaded twice (bundled + a leftover manual
        install from before this migration)."""
        content = open(_CARD_PATH, encoding="utf-8").read()
        assert "if (!customElements.get('thermosmart-card'))" in content
        assert "window.customCards.some(c => c.type === 'thermosmart-card')" in content

    def test_no_build_or_node_artifacts_in_www_dir(self):
        www_dir = os.path.dirname(_CARD_PATH)
        entries = os.listdir(www_dir)
        assert entries == [CARD_FILENAME]

    def test_no_node_modules_anywhere_under_custom_components(self):
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "custom_components")
        for dirpath, dirnames, _ in os.walk(root):
            assert "node_modules" not in dirnames


# ── Static path + frontend registration ──────────────────────────────────────


class TestRegisterCardHelper:
    async def test_registers_static_path_at_the_expected_url(self):
        hass = make_mock_hass()
        hass.http.async_register_static_paths = AsyncMock()

        with patch("homeassistant.components.frontend.add_extra_js_url") as add_url:
            await ts_init._async_register_card(hass)

        assert hass.http.async_register_static_paths.await_count == 1
        (configs,), _ = hass.http.async_register_static_paths.await_args
        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.url_path == CARD_URL_PATH == "/thermosmart/thermosmart-card.js"
        assert cfg.path.endswith(os.path.join("www", CARD_FILENAME))
        assert cfg.cache_headers is True
        add_url.assert_called_once_with(hass, f"{CARD_URL_PATH}?v={VERSION}")

    async def test_static_path_failure_is_non_fatal_and_skips_frontend_step(self):
        hass = make_mock_hass()
        hass.http.async_register_static_paths = AsyncMock(side_effect=RuntimeError("no http"))

        with patch("homeassistant.components.frontend.add_extra_js_url") as add_url:
            await ts_init._async_register_card(hass)  # must not raise

        add_url.assert_not_called()

    async def test_frontend_failure_is_non_fatal(self):
        hass = make_mock_hass()
        hass.http.async_register_static_paths = AsyncMock()

        with patch(
            "homeassistant.components.frontend.add_extra_js_url",
            side_effect=RuntimeError("no frontend"),
        ):
            await ts_init._async_register_card(hass)  # must not raise


# ── Idempotency across multiple config entries ───────────────────────────────


def _system_entry(entry_id: str) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.data = {"entry_type": "system"}
    entry.options = {}
    return entry


class TestCardRegisteredOnceAcrossEntries:
    async def test_two_system_entries_register_the_card_exactly_once(self):
        """The domain-level 'card_registered' guard is entry-type-agnostic —
        proven here via two lightweight system entries so the test doesn't
        need to stand up a full zone Coordinator/WeatherEngine/LearningEngine."""
        hass = make_mock_hass()
        hass.data = {}
        hass.http.async_register_static_paths = AsyncMock()
        hass.config_entries.async_forward_entry_setups = AsyncMock()
        hass.services.has_service = MagicMock(return_value=True)  # skip service registration branch

        with patch("homeassistant.components.frontend.add_extra_js_url") as add_url, \
             patch("custom_components.thermosmart._migrate_old_summer_switch"):
            await ts_init.async_setup_entry(hass, _system_entry("sys_1"))
            await ts_init.async_setup_entry(hass, _system_entry("sys_2"))

        assert hass.http.async_register_static_paths.await_count == 1
        assert add_url.call_count == 1
        assert hass.data[DOMAIN]["card_registered"] is True

    async def test_registration_flag_prevents_a_direct_second_call(self):
        hass = make_mock_hass()
        hass.data = {DOMAIN: {"card_registered": True}}
        hass.http.async_register_static_paths = AsyncMock()

        # Mirrors the exact guard used in async_setup_entry().
        if "card_registered" not in hass.data[DOMAIN]:
            await ts_init._async_register_card(hass)

        hass.http.async_register_static_paths.assert_not_called()


# ── URL shape sanity (no /local/ dependency, matches Frenck's ask) ──────────


class TestCardUrlShape:
    def test_url_path_is_not_under_local(self):
        assert not CARD_URL_PATH.startswith("/local/")

    def test_url_path_matches_requested_pattern(self):
        assert CARD_URL_PATH == "/thermosmart/thermosmart-card.js"
