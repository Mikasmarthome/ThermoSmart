"""HACS default#9081 maintainer review — branding, manifest, and hacs.json checks.

Covers item 1 (brand asset duplicates removed, README points at the in-tree
logo), item 2 (HA minimum version 2024.12.0 in hacs.json and the README
badge), item 4/5 (manifest.json integration_type/iot_class), and the
config-flow "user" step description (previously mandatory under hassfest's
"helper" rule; kept as good practice now that ThermoSmart is classified
as integration_type: device — see TestManifest/TestConfigFlowUserStepDescription).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = REPO_ROOT / "custom_components" / "thermosmart" / "brand"
TRANSLATIONS_DIR = REPO_ROOT / "custom_components" / "thermosmart" / "translations"
STRINGS_PATH = REPO_ROOT / "custom_components" / "thermosmart" / "strings.json"


class TestBrandAssets:
    def test_top_level_duplicates_removed(self):
        assert not (REPO_ROOT / "brand" / "icon.png").exists()
        assert not (REPO_ROOT / "brand" / "logo.png").exists()
        assert not (REPO_ROOT / "custom_components" / "thermosmart" / "icon.png").exists()

    def test_screenshots_directory_kept(self):
        assert (REPO_ROOT / "brand" / "screenshots").is_dir()

    def test_in_tree_brand_assets_kept(self):
        assert (BRAND_DIR / "icon.png").is_file()
        assert (BRAND_DIR / "logo.png").is_file()

    def test_readme_logo_points_at_in_tree_path(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "custom_components/thermosmart/brand/logo.png" in readme
        assert "raw.githubusercontent.com/Mikasmarthome/ThermoSmart/main/brand/logo.png" not in readme


class TestHomeAssistantMinimumVersion:
    def test_hacs_json_minimum_version(self):
        hacs = json.loads((REPO_ROOT / "hacs.json").read_text(encoding="utf-8"))
        assert hacs["homeassistant"] == "2024.12.0"

    def test_readme_badge_matches_hacs_json(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "HA-2024.12%2B" in readme
        assert "HA-2024.1%2B" not in readme


class TestManifest:
    def _manifest(self) -> dict:
        path = REPO_ROOT / "custom_components" / "thermosmart" / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_integration_type_is_device(self):
        """ThermoSmart is a device-/climate-control integration, not a pure
        helper — reclassified from "helper" to "device"."""
        assert self._manifest()["integration_type"] == "device"

    def test_iot_class_is_calculated(self):
        assert self._manifest()["iot_class"] == "calculated"

    def test_manifest_still_valid_json_with_required_keys(self):
        manifest = self._manifest()
        for key in ("domain", "name", "codeowners", "config_flow", "documentation", "version"):
            assert key in manifest
        assert manifest["domain"] == "thermosmart"

    def test_http_declared_as_a_dependency(self):
        """__init__.py's _async_register_card() imports
        homeassistant.components.http.StaticPathConfig to serve the bundled
        Lovelace card, so hassfest's dependency validator requires 'http' to
        be declared. This is unrelated to the export system (removed in the
        export -> HA Diagnostics migration): 'http' stays required purely
        for the card's static path registration. frontend is *not* declared
        here on purpose — hassfest's ALLOWED_USED_COMPONENTS allow-list
        exempts it (frontend is treated as always available), and the
        frontend-specific registration code in __init__.py already treats it
        as best-effort/non-fatal (try/except) rather than a hard requirement.
        """
        assert "http" in self._manifest()["dependencies"]

    def test_http_not_duplicated_in_after_dependencies(self):
        manifest = self._manifest()
        assert "http" not in manifest.get("after_dependencies", [])


class TestConfigFlowUserStepDescription:
    """Hassfest required a description on the config flow's 'user' step while
    ThermoSmart was integration_type: helper. Now that it's reclassified to
    integration_type: device, hassfest no longer mandates this — but the
    description remains good UX, so it's kept and still tested here."""

    def _translation_files(self) -> list[Path]:
        return [Path(p) for p in glob.glob(str(TRANSLATIONS_DIR / "*.json"))]

    def test_strings_json_user_step_has_a_description(self):
        strings = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
        assert strings["config"]["step"]["user"].get("description")

    def test_every_translation_file_user_step_has_a_description(self):
        for path in self._translation_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["config"]["step"]["user"].get("description"), (
                f"{path.name}: config.step.user.description missing/empty"
            )
