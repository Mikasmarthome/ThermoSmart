"""HACS default#9081 maintainer review — item 6: exports must not be publicly
reachable under /local/, and item 7's blocking-I/O half (os.makedirs no
longer runs on the event loop).

Covers:
  - async_export_learning_data()/async_export_support_data() write into the
    private "<config>/thermosmart_exports/" directory, never "www"
  - the resulting notification message contains an authenticated
    /api/thermosmart/export/... link, never a /local/... or /config/www/...
    path
  - ThermoSmartExportDownloadView requires authentication by default and
    rejects filenames that don't match ThermoSmart's own generated pattern
    (no path traversal, no serving arbitrary files from the directory)
  - the directory-creation + file-write happens inside a single executor
    job, never a bare os.makedirs()/open() call on the event loop
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.export import (
    _EXPORT_DIR_NAME,
    _EXPORT_FILENAME_RE,
    ThermoSmartExportDownloadView,
    _export_dir,
    async_build_export_notification,
    async_export_learning_data,
    async_export_support_data,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _mock_hass(tmp_path):
    hass = MagicMock()
    hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    hass.config_entries.async_entries.return_value = []
    hass.data = {}
    return hass


class TestPrivateStorageLocation:
    def test_export_dir_name_is_not_www(self):
        assert _EXPORT_DIR_NAME != "www"
        assert "www" not in _EXPORT_DIR_NAME

    async def test_research_export_writes_outside_www(self, tmp_path):
        hass = _mock_hass(tmp_path)
        hass.data = {"thermosmart": {}}
        filepath = await async_export_learning_data(hass, ts=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc))
        assert "www" not in Path(filepath).parts
        assert _EXPORT_DIR_NAME in Path(filepath).parts
        assert Path(filepath).is_file()

    async def test_support_export_writes_outside_www(self, tmp_path):
        hass = _mock_hass(tmp_path)
        hass.data = {"thermosmart": {}}
        filepath = await async_export_support_data(hass, ts=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc))
        assert "www" not in Path(filepath).parts
        assert _EXPORT_DIR_NAME in Path(filepath).parts

    def test_export_py_source_never_touches_www(self):
        export_py = (
            REPO_ROOT / "custom_components" / "thermosmart" / "export.py"
        ).read_text(encoding="utf-8")
        assert 'hass.config.path("www")' not in export_py
        # No code path builds a /local/... or /config/www/... URL for a user
        # (the docstring explains *why* we avoid /local/, which is fine).
        assert "/local/{filename}" not in export_py
        assert "/config/www/{filename}" not in export_py


class TestNotificationLinksAreAuthenticated:
    async def test_notification_message_uses_authenticated_api_path(self):
        hass = MagicMock()
        hass.config.language = "en"
        hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))

        _, message = await async_build_export_notification(hass, "thermosmart_research_20260101T000000_abcdef.json")
        assert "/api/thermosmart/export/thermosmart_research_20260101T000000_abcdef.json" in message
        assert "/local/" not in message
        assert "/config/www/" not in message


class TestDownloadView:
    def test_requires_auth_defaults_true(self):
        view = ThermoSmartExportDownloadView(MagicMock())
        assert view.requires_auth is True

    def test_url_pattern_is_scoped_to_thermosmart(self):
        assert ThermoSmartExportDownloadView.url == "/api/thermosmart/export/{filename}"

    async def test_rejects_filename_not_matching_generated_pattern(self):
        hass = MagicMock()
        view = ThermoSmartExportDownloadView(hass)
        request = MagicMock()
        with pytest.raises(Exception) as exc_info:
            await view.get(request, "../../etc/passwd")
        assert "404" in str(exc_info.value) or "NotFound" in type(exc_info.value).__name__

    async def test_rejects_filename_with_wrong_extension(self):
        hass = MagicMock()
        view = ThermoSmartExportDownloadView(hass)
        request = MagicMock()
        with pytest.raises(Exception):
            await view.get(request, "thermosmart_research_20260101T000000_abcdef.txt")

    async def test_rejects_nonexistent_but_well_formed_filename(self, tmp_path):
        hass = MagicMock()
        hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
        hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
        view = ThermoSmartExportDownloadView(hass)
        request = MagicMock()
        with pytest.raises(Exception):
            await view.get(request, "thermosmart_research_20260101T000000_abcdef.json")

    async def test_accepts_well_formed_existing_filename(self, tmp_path):
        hass = MagicMock()
        hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
        hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
        export_dir = _export_dir(hass)
        os.makedirs(export_dir, exist_ok=True)
        filename = "thermosmart_support_20260101T000000_abcdef.json"
        Path(export_dir, filename).write_text("{}", encoding="utf-8")
        view = ThermoSmartExportDownloadView(hass)
        request = MagicMock()
        response = await view.get(request, filename)
        assert str(response._path) == str(Path(export_dir, filename))

    def test_filename_regex_rejects_path_traversal_components(self):
        for bad in ("../secret.json", "..\\secret.json", "a/b.json", "a\\b.json", ""):
            assert not _EXPORT_FILENAME_RE.match(bad)

    def test_filename_regex_accepts_generated_pattern(self):
        assert _EXPORT_FILENAME_RE.match("thermosmart_research_20260101T120000_abc123.json")
        assert _EXPORT_FILENAME_RE.match("thermosmart_support_20261231T235959_0f1e2d.json")


class TestNoBlockingIoOnEventLoop:
    def test_no_bare_os_makedirs_outside_executor_helper(self):
        """os.makedirs must only ever appear inside a function that itself
        runs inside hass.async_add_executor_job (i.e. inside
        _async_write_export_file's nested _write()), never directly in an
        async def body."""
        export_py = (
            REPO_ROOT / "custom_components" / "thermosmart" / "export.py"
        ).read_text(encoding="utf-8")
        makedirs_lines = [
            i for i, line in enumerate(export_py.splitlines(), start=1)
            if "os.makedirs" in line
        ]
        assert len(makedirs_lines) == 1, (
            "expected exactly one os.makedirs call, inside the shared "
            "_async_write_export_file executor helper"
        )

    def test_no_bare_os_remove_outside_executor_helper(self):
        export_py = (
            REPO_ROOT / "custom_components" / "thermosmart" / "export.py"
        ).read_text(encoding="utf-8")
        remove_lines = [
            i for i, line in enumerate(export_py.splitlines(), start=1)
            if "os.remove" in line
        ]
        # One in _async_remove_export_file, one in the startup scan.
        assert len(remove_lines) == 2

    async def test_write_export_file_uses_single_executor_job(self, tmp_path):
        from custom_components.thermosmart.export import _async_write_export_file

        hass = MagicMock()
        hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
        calls = []

        async def tracking_executor_job(fn, *args):
            calls.append(fn)
            return fn(*args)

        hass.async_add_executor_job = tracking_executor_job
        filepath = await _async_write_export_file(hass, "thermosmart_research_20260101T000000_abcdef.json", {"a": 1})
        assert len(calls) == 1
        assert json.loads(Path(filepath).read_text(encoding="utf-8")) == {"a": 1}
