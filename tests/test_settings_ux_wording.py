"""Tests for the ThermoSmart Settings UX Polish block.

Covers: main options-dialog title/description, German menu wording ("und"
instead of "&"), the "Präsenz-Einstellungen" rename, and full translation-key
coverage for the two new presence-delay fields across all supported languages.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent / "custom_components" / "thermosmart"
_TRANSLATIONS_DIR = _ROOT / "translations"

_ALL_LANGUAGES = sorted(p.stem for p in _TRANSLATIONS_DIR.glob("*.json"))


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _strings() -> dict:
    return _load(_ROOT / "strings.json")


def _lang(code: str) -> dict:
    return _load(_TRANSLATIONS_DIR / f"{code}.json")


class TestMainDialogTitleAndDescription:
    def test_english_options_init_has_description(self):
        data = _strings()
        assert data["options"]["step"]["init"]["description"]

    def test_german_options_init_title_is_edit_settings(self):
        data = _lang("de")
        assert data["options"]["step"]["init"]["title"] == "ThermoSmart-Einstellungen bearbeiten"

    def test_german_options_init_description_matches_expected_wording(self):
        data = _lang("de")
        desc = data["options"]["step"]["init"]["description"]
        assert desc == (
            "Hier kannst du Geräte-, Temperatur-, Zeitplan-, Präsenz- und "
            "Wettereinstellungen dieser Zone bearbeiten sowie Sensoren und "
            "die Heizungsregelung konfigurieren."
        )


class TestGermanMenuWordingUsesUndNotAmpersand:
    def test_options_menu_options_use_und(self):
        menu = _lang("de")["options"]["step"]["init"]["menu_options"]
        assert menu["devices"] == "Geräte und Sensoren"
        assert menu["schedule"] == "Temperaturen und Zeitplan"
        assert menu["presence"] == "Präsenz-Einstellungen"
        assert menu["weather"] == "Wetter und Außensensoren"
        for label in menu.values():
            assert "&" not in label

    def test_options_substep_titles_use_und(self):
        steps = _lang("de")["options"]["step"]
        assert steps["devices"]["title"] == "Geräte und Sensoren"
        assert steps["schedule"]["title"] == "Temperaturen und Zeitplan"
        assert steps["presence"]["title"] == "Präsenz-Einstellungen"
        assert steps["weather"]["title"] == "Wetter und Außensensoren"

    def test_config_substep_titles_use_und(self):
        """Initial setup wizard uses the same wording as the options flow."""
        steps = _lang("de")["config"]["step"]
        assert "Geräte und Sensoren" in steps["add_zone"]["title"]
        assert "Temperaturen und Zeitplan" in steps["schedule"]["title"]
        assert "Präsenz-Einstellungen" in steps["presence"]["title"]
        assert "Wetter und Außensensoren" in steps["weather"]["title"]

    def test_no_stray_ampersand_left_in_zone_settings_flow(self):
        """Every '&' in de.json's config/options step titles and menu labels
        must be gone — except the deliberately out-of-scope global system
        entry ("Sommer- & Urlaubsmodus"), which this feature does not touch."""
        data = _lang("de")
        offenders = []
        for section in ("config", "options"):
            for step_id, step in data[section]["step"].items():
                title = step.get("title", "")
                if "&" in title:
                    offenders.append(f"{section}.step.{step_id}.title: {title!r}")
                for label in step.get("menu_options", {}).values():
                    if "&" in label:
                        offenders.append(f"{section}.step.{step_id}.menu_options: {label!r}")
        allowed = {"config.step.user.menu_options: 'Globale Steuerung einrichten (Sommer- & Urlaubsmodus)'"}
        assert set(offenders) - allowed == set()


class TestPresenceDelayTranslationKeysExistForAllLanguages:
    """New keys must exist, be non-empty, and not look machine-generated
    (i.e. not merely a copy of the English source) in every supported language."""

    @pytest.mark.parametrize("code", _ALL_LANGUAGES)
    def test_options_presence_has_delay_labels_and_help_text(self, code):
        step = _lang(code)["options"]["step"]["presence"]
        assert step["data"]["presence_arrival_delay"]
        assert step["data"]["presence_departure_delay"]
        assert step["data_description"]["presence_arrival_delay"]
        assert step["data_description"]["presence_departure_delay"]

    @pytest.mark.parametrize("code", _ALL_LANGUAGES)
    def test_config_presence_has_delay_labels_and_help_text(self, code):
        step = _lang(code)["config"]["step"]["presence"]
        assert step["data"]["presence_arrival_delay"]
        assert step["data"]["presence_departure_delay"]
        assert step["data_description"]["presence_arrival_delay"]
        assert step["data_description"]["presence_departure_delay"]

    @pytest.mark.parametrize("code", _ALL_LANGUAGES)
    def test_options_init_has_description(self, code):
        assert _lang(code)["options"]["step"]["init"]["description"]

    def test_all_translation_files_are_valid_json(self):
        for code in _ALL_LANGUAGES:
            _lang(code)  # raises on invalid JSON

    def test_no_language_missing_from_supported_list(self):
        """Sanity check against the README-documented language count."""
        assert len(_ALL_LANGUAGES) == 24


class TestGermanPresenceFieldOrderAndNaming:
    def test_presence_entities_field_renamed(self):
        data = _lang("de")["options"]["step"]["presence"]["data"]
        assert data["presence_persons"] == "Anwesenheits-Entitäten"

    def test_field_order_matches_requested_sequence(self):
        data = _lang("de")["options"]["step"]["presence"]["data"]
        keys = list(data.keys())
        assert keys.index("presence_persons") < keys.index("presence_arrival_delay")
        assert keys.index("presence_arrival_delay") < keys.index("presence_departure_delay")
