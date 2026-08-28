"""HACS default#9081 maintainer review — item 3: export/support notification
localization.

Covers:
  - no hardcoded German (or any single-language) user-facing text remains in
    export.py's notification-building code
  - notification text lives in custom_components/thermosmart/notifications/
    <lang>.json — a ThermoSmart-owned resource, not strings.json/
    translations/*.json (hassfest validates those two against a fixed
    top-level-key allow-list that has no slot for freeform, placeholder-
    bearing notification text; a "notification" key there fails CI)
  - strings.json and every translations/*.json file no longer carry a
    "notification" key
  - the English default renders correctly
  - the German translation renders correctly and differs from English
  - _async_notification_text() falls back to English, then to the in-code
    constant, if the per-language catalog file can't be read
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.export import (
    _NOTIFICATION_FALLBACK_EN,
    _async_notification_text,
    _load_notification_catalog,
    async_build_export_notification,
    async_build_support_notification,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_DIR = REPO_ROOT / "custom_components" / "thermosmart" / "translations"
NOTIFICATIONS_DIR = REPO_ROOT / "custom_components" / "thermosmart" / "notifications"
STRINGS_PATH = REPO_ROOT / "custom_components" / "thermosmart" / "strings.json"

_GERMAN_MARKERS = (
    "wurde erstellt", "enthält", "gelöscht", "Prüfe", "übertragen", "Öffnen",
)


def _all_translation_files() -> list[Path]:
    return [Path(p) for p in glob.glob(str(TRANSLATIONS_DIR / "*.json"))]


def _all_notification_files() -> list[Path]:
    return [Path(p) for p in glob.glob(str(NOTIFICATIONS_DIR / "*.json"))]


def _hass_with_real_executor(language: str) -> MagicMock:
    """A hass mock whose async_add_executor_job actually runs the given sync
    function synchronously — matches the real hass contract closely enough
    for _load_notification_catalog(), a small local JSON read."""
    hass = MagicMock()
    hass.config.language = language
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    return hass


class TestNoHardcodedGermanText:
    def test_export_py_source_has_no_hardcoded_german_notification_text(self):
        export_py = (
            REPO_ROOT / "custom_components" / "thermosmart" / "export.py"
        ).read_text(encoding="utf-8")
        for marker in _GERMAN_MARKERS:
            assert marker not in export_py, (
                f"Found hardcoded German text marker {marker!r} in export.py — "
                "notification text must come from notifications/<lang>.json, "
                "not a literal string in code."
            )

    def test_old_hardcoded_builder_functions_removed(self):
        export_py = (
            REPO_ROOT / "custom_components" / "thermosmart" / "export.py"
        ).read_text(encoding="utf-8")
        assert "def build_export_notification_message" not in export_py
        assert "def build_support_notification_message" not in export_py


class TestHassfestSchemaCompliance:
    """The actual bug this file guards against: hassfest's strings.json
    schema has a fixed top-level-key allow-list (config, options, entity,
    services, issues, ...) that does not include "notification" — putting
    freeform notification text there fails real hassfest CI even though it
    is syntactically valid JSON and even though a bespoke loader could read
    it at runtime."""

    def test_strings_json_has_no_notification_key(self):
        strings = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
        assert "notification" not in strings

    def test_no_translation_file_has_a_notification_key(self):
        for path in _all_translation_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "notification" not in data, (
                f"{path.name} still has a top-level 'notification' key — "
                "hassfest rejects this; move it to notifications/<lang>.json"
            )

    def test_config_step_user_has_a_description(self):
        """Hassfest requires a description on the config flow's 'user' step
        for helper integrations (integration_type: helper) — see
        manifest.json / test_hacs_review_9081_metadata.py."""
        strings = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
        assert strings["config"]["step"]["user"].get("description")

    def test_every_translation_file_config_step_user_has_a_description(self):
        for path in _all_translation_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["config"]["step"]["user"].get("description"), (
                f"{path.name}: config.step.user.description missing/empty"
            )


class TestNotificationCatalogCompleteness:
    def test_english_catalog_has_both_keys(self):
        catalog = json.loads((NOTIFICATIONS_DIR / "en.json").read_text(encoding="utf-8"))
        for key in ("export_created", "support_created"):
            assert key in catalog
            assert catalog[key].get("title")
            assert catalog[key].get("message")

    def test_every_notification_file_has_title_and_message_for_both_keys(self):
        files = _all_notification_files()
        assert len(files) >= 20, "expected the full set of ThermoSmart notification catalogs"
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("export_created", "support_created"):
                assert data[key].get("title"), f"{path.name}: {key}.title missing/empty"
                assert data[key].get("message"), f"{path.name}: {key}.message missing/empty"

    def test_all_notification_files_are_valid_json(self):
        for path in [*_all_notification_files(), STRINGS_PATH, *_all_translation_files()]:
            json.loads(path.read_text(encoding="utf-8"))  # raises on invalid JSON

    def test_message_placeholders_are_consistent_across_languages(self):
        """Every language must use the same {placeholder} names as English so
        .format(**placeholders) never raises a KeyError for any language."""
        import re
        placeholder_re = re.compile(r"\{(\w+)\}")
        en = json.loads((NOTIFICATIONS_DIR / "en.json").read_text(encoding="utf-8"))
        en_placeholders = {
            key: set(placeholder_re.findall(en[key]["message"]))
            for key in ("export_created", "support_created")
        }
        for path in _all_notification_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("export_created", "support_created"):
                found = set(placeholder_re.findall(data[key]["message"]))
                assert found == en_placeholders[key], (
                    f"{path.name}: {key} placeholders {found} do not match "
                    f"English {en_placeholders[key]}"
                )


class TestLoadNotificationCatalog:
    def test_loads_english_catalog_from_disk(self):
        catalog = _load_notification_catalog("en")
        assert "export_created" in catalog
        assert "support_created" in catalog

    def test_unknown_language_returns_empty_dict(self):
        assert _load_notification_catalog("xx") == {}


class TestNotificationRendering:
    async def test_english_default_renders(self):
        hass = _hass_with_real_executor("en")
        title, message = await async_build_export_notification(hass, "thermosmart_research_20260101T000000_abcdef.json")
        assert "Research export created" in title
        assert "thermosmart_research_20260101T000000_abcdef.json" in message
        assert "/api/thermosmart/export/thermosmart_research_20260101T000000_abcdef.json" in message
        assert "24" in message  # retention_hours placeholder rendered

    async def test_german_translation_renders_and_differs_from_english(self):
        hass = _hass_with_real_executor("de")
        title_de, message_de = await async_build_support_notification(hass, "thermosmart_support_20260101T000000_abcdef.json")
        assert "Support-Export erstellt" in title_de
        assert "wurde erstellt" in message_de

        hass.config.language = "en"
        title_en, message_en = await async_build_support_notification(hass, "thermosmart_support_20260101T000000_abcdef.json")
        assert title_de != title_en
        assert message_de != message_en

    async def test_missing_translation_falls_back_to_english_then_constant(self):
        hass = _hass_with_real_executor("xx")  # not a real language — nothing resolves
        title, message = await async_build_export_notification(hass, "thermosmart_research_20260101T000000_abcdef.json")
        # xx.json doesn't exist -> falls back to English catalog, which does exist,
        # so the in-code constant is never actually needed here — this still proves
        # the fallback chain reaches a non-empty, correct result.
        assert title == _NOTIFICATION_FALLBACK_EN["export_created"]["title"] or "Research export created" in title
        assert "thermosmart_research_20260101T000000_abcdef.json" in message

    async def test_catalog_load_exception_falls_back_to_constant(self):
        hass = MagicMock()
        hass.config.language = "de"

        async def boom(*args, **kwargs):
            raise RuntimeError("disk unavailable")

        hass.async_add_executor_job = boom
        title, message = await _async_notification_text(
            hass, "export_created",
            filename="x.json", download_url="/api/thermosmart/export/x.json",
            retention_hours=24,
        )
        assert title  # never empty/raises
        assert message
