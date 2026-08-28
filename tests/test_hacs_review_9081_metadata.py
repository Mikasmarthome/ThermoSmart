"""HACS default#9081 maintainer review — branding, manifest, and hacs.json checks.

Covers item 1 (brand asset duplicates removed, README points at the in-tree
logo), item 2 (HA minimum version 2024.12.0 in hacs.json and the README
badge), and item 4/5 (manifest.json integration_type/iot_class).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = REPO_ROOT / "custom_components" / "thermosmart" / "brand"


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

    def test_integration_type_is_helper(self):
        assert self._manifest()["integration_type"] == "helper"

    def test_iot_class_is_calculated(self):
        assert self._manifest()["iot_class"] == "calculated"

    def test_manifest_still_valid_json_with_required_keys(self):
        manifest = self._manifest()
        for key in ("domain", "name", "codeowners", "config_flow", "documentation", "version"):
            assert key in manifest
        assert manifest["domain"] == "thermosmart"

    def test_http_declared_as_a_dependency(self):
        """export.py's ThermoSmartExportDownloadView subclasses
        homeassistant.components.http.HomeAssistantView at module level (a
        hard, non-optional import), so hassfest's dependency validator
        requires 'http' to be declared. frontend is *not* declared here on
        purpose — hassfest's ALLOWED_USED_COMPONENTS allow-list exempts it
        (frontend is treated as always available), and the frontend-specific
        registration code in __init__.py already treats it as best-effort/
        non-fatal (try/except) rather than a hard requirement."""
        assert "http" in self._manifest()["dependencies"]

    def test_http_not_duplicated_in_after_dependencies(self):
        manifest = self._manifest()
        assert "http" not in manifest.get("after_dependencies", [])
