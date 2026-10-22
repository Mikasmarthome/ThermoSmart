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
from custom_components.thermosmart.const import (
    CARD_ASSET_VERSION,
    CARD_FILENAME,
    CARD_URL_PATH,
    DOMAIN,
)
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
    async def test_integration_setup_registers_card_before_config_entries(self):
        hass = make_mock_hass()
        hass.data = {}
        hass.http.async_register_static_paths = AsyncMock()

        with patch("homeassistant.components.frontend.add_extra_js_url") as add_url:
            assert await ts_init.async_setup(hass, {}) is True

        assert hass.data[DOMAIN]["card_registered"] is True
        add_url.assert_called_once_with(hass, f"{CARD_URL_PATH}?v={CARD_ASSET_VERSION}")

    async def test_registers_static_path_at_the_expected_url(self):
        hass = make_mock_hass()
        hass.data = {}
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
        add_url.assert_called_once_with(hass, f"{CARD_URL_PATH}?v={CARD_ASSET_VERSION}")

    async def test_static_path_failure_is_non_fatal_and_skips_frontend_step(self):
        hass = make_mock_hass()
        hass.data = {}
        hass.http.async_register_static_paths = AsyncMock(side_effect=RuntimeError("no http"))

        with patch("homeassistant.components.frontend.add_extra_js_url") as add_url:
            await ts_init._async_register_card(hass)  # must not raise

        add_url.assert_not_called()

    async def test_frontend_failure_is_non_fatal(self):
        hass = make_mock_hass()
        hass.data = {}
        hass.http.async_register_static_paths = AsyncMock()

        with patch(
            "homeassistant.components.frontend.add_extra_js_url",
            side_effect=RuntimeError("no frontend"),
        ):
            await ts_init._async_register_card(hass)  # must not raise

        assert hass.data[DOMAIN].get("card_registered") is not True

    async def test_frontend_failure_can_be_retried_without_duplicate_static_route(self):
        hass = make_mock_hass()
        hass.data = {}
        hass.http.async_register_static_paths = AsyncMock()
        add_url = MagicMock(side_effect=[RuntimeError("temporary frontend failure"), None])

        with patch("homeassistant.components.frontend.add_extra_js_url", add_url):
            await ts_init._async_register_card(hass)
            await ts_init._async_register_card(hass)

        assert hass.http.async_register_static_paths.await_count == 1
        assert add_url.call_count == 2
        assert hass.data[DOMAIN]["card_registered"] is True


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


# ── Card version / encoding consistency ───────────────────────────────────────


class TestCardAssetVersionConsistency:
    """CARD_ASSET_VERSION is the cache-busting query string of the card URL; it
    must be bumped together with the card's own CARD_VERSION, otherwise
    browsers keep serving a stale copy of a changed card."""

    def test_asset_version_matches_the_version_inside_the_card(self):
        import re
        content = open(_CARD_PATH, encoding="utf-8").read()
        match = re.search(r"const CARD_VERSION = '([^']+)'", content)
        assert match, "CARD_VERSION constant not found in thermosmart-card.js"
        assert match.group(1) == CARD_ASSET_VERSION

    def test_registered_url_carries_the_asset_version(self):
        from custom_components.thermosmart.const import CARD_URL_PATH
        assert f"{CARD_URL_PATH}?v={CARD_ASSET_VERSION}".endswith(f"?v={CARD_ASSET_VERSION}")


class TestCardFileEncoding:
    """Guards the shipped card against encoding damage. UTF-8 double encoding
    (text decoded as Latin-1/CP1252 and encoded again) leaves byte sequences
    that start with the UTF-8 bytes of the two letters A-tilde / A-circumflex
    and breaks the script in the browser."""

    # UTF-8 encodings of U+00C3, U+00C2, and of the 3-byte mojibake lead-in
    # that a double-encoded en/em dash or quote produces.
    _MOJIBAKE_MARKERS = (
        bytes([0xC3, 0x83]),
        bytes([0xC3, 0x82]),
        bytes([0xC3, 0xA2, 0xE2, 0x82, 0xAC]),
    )

    def test_card_is_valid_utf8(self):
        with open(_CARD_PATH, "rb") as fh:
            fh.read().decode("utf-8")  # raises on invalid UTF-8

    def test_card_contains_no_double_encoding_markers(self):
        with open(_CARD_PATH, "rb") as fh:
            data = fh.read()
        for marker in self._MOJIBAKE_MARKERS:
            assert marker not in data, f"double-encoding marker {marker!r} found"

    def test_card_has_no_byte_order_mark(self):
        with open(_CARD_PATH, "rb") as fh:
            assert fh.read(3) != bytes([0xEF, 0xBB, 0xBF])
