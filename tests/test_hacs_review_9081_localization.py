"""HACS default#9081 maintainer review — item 3: export/support notification
localization.

Covers:
  - no hardcoded German (or any single-language) user-facing text remains in
    export.py's notification-building code
  - the "notification" translation key exists, with matching structure, in
    strings.json and every translations/*.json file (no language can break
    due to a missing key)
  - the English default renders correctly
  - the German translation renders correctly and differs from English
  - _async_notification_text() falls back to English, then to the in-code
    constant, if HA's translation loader can't resolve a key
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.export import (
    _NOTIFICATION_FALLBACK_EN,
    _async_notification_text,
    async_build_export_notification,
    async_build_support_notification,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_DIR = REPO_ROOT / "custom_components" / "thermosmart" / "translations"
STRINGS_PATH = REPO_ROOT / "custom_components" / "thermosmart" / "strings.json"

_GERMAN_MARKERS = (
    "wurde erstellt", "enthält", "gelöscht", "Prüfe", "übertragen", "Öffnen",
)


def _all_translation_files() -> list[Path]:
    return [Path(p) for p in glob.glob(str(TRANSLATIONS_DIR / "*.json"))]


class TestNoHardcodedGermanText:
    def test_export_py_source_has_no_hardcoded_german_notification_text(self):
        export_py = (
            REPO_ROOT / "custom_components" / "thermosmart" / "export.py"
        ).read_text(encoding="utf-8")
        for marker in _GERMAN_MARKERS:
            assert marker not in export_py, (
                f"Found hardcoded German text marker {marker!r} in export.py — "
                "notification text must come from strings.json/translations "
                "via async_get_translations(), not a literal string in code."
            )

    def test_old_hardcoded_builder_functions_removed(self):
        export_py = (
            REPO_ROOT / "custom_components" / "thermosmart" / "export.py"
        ).read_text(encoding="utf-8")
        assert "def build_export_notification_message" not in export_py
        assert "def build_support_notification_message" not in export_py


class TestTranslationKeyCompleteness:
    def test_strings_json_has_notification_key(self):
        strings = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
        assert "notification" in strings
        for key in ("export_created", "support_created"):
            assert key in strings["notification"]
            assert "title" in strings["notification"][key]
            assert "message" in strings["notification"][key]

    def test_every_translation_file_has_notification_key(self):
        files = _all_translation_files()
        assert len(files) >= 20, "expected the full set of ThermoSmart translation files"
        missing = []
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            notif = data.get("notification")
            if not notif or "export_created" not in notif or "support_created" not in notif:
                missing.append(path.name)
        assert missing == [], f"translation files missing the notification key: {missing}"

    def test_every_translation_file_has_title_and_message_for_both_keys(self):
        for path in _all_translation_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            notif = data["notification"]
            for key in ("export_created", "support_created"):
                assert notif[key].get("title"), f"{path.name}: {key}.title missing/empty"
                assert notif[key].get("message"), f"{path.name}: {key}.message missing/empty"

    def test_all_translation_files_are_valid_json(self):
        for path in [STRINGS_PATH, *_all_translation_files()]:
            json.loads(path.read_text(encoding="utf-8"))  # raises on invalid JSON

    def test_message_placeholders_are_consistent_across_languages(self):
        """Every language must use the same {placeholder} names as English so
        .format(**placeholders) never raises a KeyError for any language."""
        placeholder_re = re.compile(r"\{(\w+)\}")
        en = json.loads((TRANSLATIONS_DIR / "en.json").read_text(encoding="utf-8"))
        en_placeholders = {
            key: set(placeholder_re.findall(en["notification"][key]["message"]))
            for key in ("export_created", "support_created")
        }
        for path in _all_translation_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("export_created", "support_created"):
                found = set(placeholder_re.findall(data["notification"][key]["message"]))
                assert found == en_placeholders[key], (
                    f"{path.name}: {key} placeholders {found} do not match "
                    f"English {en_placeholders[key]}"
                )


def _hass_with_translations(catalog: dict[str, dict]) -> MagicMock:
    """A minimal hass mock whose async_get_translations-equivalent behavior
    is patched at the module level in each test (see monkeypatch usage
    below) — this just carries hass.config.language."""
    hass = MagicMock()
    hass.config.language = catalog.get("_language", "en")
    return hass


class TestNotificationRendering:
    async def test_english_default_renders(self, monkeypatch):
        hass = MagicMock()
        hass.config.language = "en"

        async def fake_get_translations(hass_, language, category, integrations=None, config_flow=None):
            strings = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
            prefix = f"component.{DOMAIN}.notification"
            flat = {}
            for key, val in strings["notification"].items():
                flat[f"{prefix}.{key}.title"] = val["title"]
                flat[f"{prefix}.{key}.message"] = val["message"]
            return flat if language == "en" else {}

        monkeypatch.setattr(
            "custom_components.thermosmart.export.async_get_translations",
            fake_get_translations,
        )
        title, message = await async_build_export_notification(hass, "thermosmart_research_20260101T000000_abcdef.json")
        assert "Research export created" in title
        assert "thermosmart_research_20260101T000000_abcdef.json" in message
        assert "/api/thermosmart/export/thermosmart_research_20260101T000000_abcdef.json" in message
        assert "24" in message  # retention_hours placeholder rendered

    async def test_german_translation_renders_and_differs_from_english(self, monkeypatch):
        hass = MagicMock()
        hass.config.language = "de"
        de = json.loads((TRANSLATIONS_DIR / "de.json").read_text(encoding="utf-8"))
        en = json.loads((TRANSLATIONS_DIR / "en.json").read_text(encoding="utf-8"))

        async def fake_get_translations(hass_, language, category, integrations=None, config_flow=None):
            source = de if language == "de" else en
            prefix = f"component.{DOMAIN}.notification"
            flat = {}
            for key, val in source["notification"].items():
                flat[f"{prefix}.{key}.title"] = val["title"]
                flat[f"{prefix}.{key}.message"] = val["message"]
            return flat

        monkeypatch.setattr(
            "custom_components.thermosmart.export.async_get_translations",
            fake_get_translations,
        )
        title_de, message_de = await async_build_support_notification(hass, "thermosmart_support_20260101T000000_abcdef.json")
        assert "Support-Export erstellt" in title_de
        assert "wurde erstellt" in message_de
        hass.config.language = "en"
        title_en, message_en = await async_build_support_notification(hass, "thermosmart_support_20260101T000000_abcdef.json")
        assert title_de != title_en
        assert message_de != message_en

    async def test_missing_translation_falls_back_to_english_then_constant(self, monkeypatch):
        hass = MagicMock()
        hass.config.language = "xx"  # not a real language — nothing resolves

        async def empty_translations(hass_, language, category, integrations=None, config_flow=None):
            return {}

        monkeypatch.setattr(
            "custom_components.thermosmart.export.async_get_translations",
            empty_translations,
        )
        title, message = await async_build_export_notification(hass, "thermosmart_research_20260101T000000_abcdef.json")
        # Falls all the way back to the in-code English constant — never blank.
        assert title == _NOTIFICATION_FALLBACK_EN["export_created"]["title"]
        assert "thermosmart_research_20260101T000000_abcdef.json" in message

    async def test_translation_loader_exception_does_not_break_notification(self, monkeypatch):
        hass = MagicMock()
        hass.config.language = "de"

        async def boom(*args, **kwargs):
            raise RuntimeError("translation cache unavailable")

        monkeypatch.setattr(
            "custom_components.thermosmart.export.async_get_translations", boom
        )
        title, message = await _async_notification_text(
            hass, "export_created",
            filename="x.json", download_url="/api/thermosmart/export/x.json",
            retention_hours=24,
        )
        assert title  # never empty/raises
        assert message
